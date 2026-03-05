"""Unit tests for SHOT adaptation utilities."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.train.shot import (
    SHOTAdapter,
    SHOTConfig,
    binary_probs_from_logits,
    entropy_loss_from_probs,
    prior_guided_alignment_loss,
)


class _TargetDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self._x = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.5, 0.3, 0.2],
                [0.9, 0.1, 0.4],
            ],
            dtype=torch.float32,
        )
        self._label = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)

    def __len__(self) -> int:
        return int(self._x.size(0))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": self._x[index],
            "label": self._label[index],
        }


class _ToySHOTModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.output_head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> dict[str, torch.Tensor]:
        del label
        features = torch.tanh(self.encoder(x))
        logits = self.output_head(features)
        return {"logits": logits}


def test_shot_losses_are_finite() -> None:
    logits = torch.tensor([0.0, 2.0, -2.0], dtype=torch.float32)
    probs = binary_probs_from_logits(logits=logits, epsilon=1.0e-6)
    entropy = entropy_loss_from_probs(probs=probs, epsilon=1.0e-6)
    align = prior_guided_alignment_loss(
        probs=probs,
        target_prior=torch.tensor([0.99, 0.01]),
        epsilon=1.0e-6,
    )

    assert probs.shape == (3, 2)
    assert torch.isfinite(entropy)
    assert torch.isfinite(align)


def test_shot_adapter_freezes_classifier_and_updates_encoder(tmp_path: Path) -> None:
    model = _ToySHOTModel()
    loader = DataLoader(_TargetDataset(), batch_size=2, shuffle=False)

    initial_encoder_weight = model.encoder.weight.detach().clone()
    initial_head_weight = model.output_head.weight.detach().clone()

    adapter = SHOTAdapter(
        model=model,
        device=torch.device("cpu"),
        config=SHOTConfig(epochs=2, lr=5e-2, beta=0.3, use_amp=False),
    )
    csv_path = tmp_path / "shot_adapt.csv"
    adapter.adapt(target_loader=loader, csv_path=csv_path)

    assert csv_path.exists()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert "target_pos_prob_mean" in reader.fieldnames
        assert "target_pred_pos_rate" in reader.fieldnames
        assert "selected_pred_pos_rate" in reader.fieldnames
        assert "test_auroc" in reader.fieldnames
        assert "test_auprc" in reader.fieldnames
        assert "test_sensitivity" in reader.fieldnames
        assert "test_specificity" in reader.fieldnames
    assert not torch.allclose(model.encoder.weight.detach(), initial_encoder_weight)
    assert torch.allclose(model.output_head.weight.detach(), initial_head_weight)


def test_shot_pseudo_labels_handle_class_collapse() -> None:
    model = _ToySHOTModel()
    adapter = SHOTAdapter(
        model=model,
        device=torch.device("cpu"),
        config=SHOTConfig(class_count_threshold=99),
    )
    features = torch.randn(5, 4)
    logits = torch.full((5,), 6.0, dtype=torch.float32)
    probs = binary_probs_from_logits(logits=logits, epsilon=1.0e-6)
    pseudo = adapter._build_pseudo_labels(features=features, probs=probs)

    assert pseudo.shape == (5,)
    assert pseudo.dtype == torch.long
    assert torch.all((pseudo == 0) | (pseudo == 1))


def test_shot_asymmetric_threshold_mask() -> None:
    model = _ToySHOTModel()
    adapter = SHOTAdapter(
        model=model,
        device=torch.device("cpu"),
        config=SHOTConfig(tau_pos=0.88, tau_neg=0.98),
    )
    probs = torch.tensor(
        [
            [0.985, 0.015],  # keep as negative
            [0.970, 0.030],  # drop as negative
            [0.090, 0.910],  # keep as positive
            [0.130, 0.870],  # drop as positive
        ],
        dtype=torch.float32,
    )
    mask = adapter._build_selection_mask(probs=probs)
    expected = torch.tensor([True, False, True, False])
    assert torch.equal(mask, expected)


def test_shot_weighted_pseudo_label_loss_respects_mask() -> None:
    model = _ToySHOTModel()
    adapter = SHOTAdapter(
        model=model,
        device=torch.device("cpu"),
        config=SHOTConfig(pos_weight=5.0, neg_weight=1.0, normalize_class_weights=True),
    )
    logits = torch.tensor([0.2, -0.1, 1.5], dtype=torch.float32)
    pseudo = torch.tensor([1, 0, 1], dtype=torch.long)
    mask = torch.tensor([True, False, True], dtype=torch.bool)
    loss = adapter._masked_weighted_pseudo_label_loss(
        logits=logits,
        pseudo_labels=pseudo,
        mask=mask,
    )
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0
