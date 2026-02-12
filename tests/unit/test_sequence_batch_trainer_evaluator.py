"""Unit tests for trainer/evaluator handling mixed sequence+tensor batches."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.evaluate import Evaluator
from src.train.base import OptimizerConfig, SchedulerConfig, Trainer
from src.utils.losses import LossConfig


class _SequenceBatchDataset(Dataset[dict[str, object]]):
    """Dataset emitting raw sequences plus tensor labels."""

    def __init__(self) -> None:
        self._rows = [
            {"seq_a": "AAAA", "seq_b": "CC", "label": torch.tensor(1.0)},
            {"seq_a": "BBB", "seq_b": "DDDD", "label": torch.tensor(0.0)},
        ]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self._rows[index]


def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
    labels: list[torch.Tensor] = []
    seq_a: list[str] = []
    seq_b: list[str] = []
    for sample in batch:
        label_value = sample.get("label")
        seq_a_value = sample.get("seq_a")
        seq_b_value = sample.get("seq_b")
        assert isinstance(label_value, torch.Tensor)
        assert isinstance(seq_a_value, str)
        assert isinstance(seq_b_value, str)
        labels.append(label_value)
        seq_a.append(seq_a_value)
        seq_b.append(seq_b_value)
    return {
        "seq_a": seq_a,
        "seq_b": seq_b,
        "label": torch.stack(labels, dim=0),
    }


class _SequenceModel(nn.Module):
    """Toy model using sequence lengths to produce logits."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(1, 1)

    def forward(
        self,
        batch: dict[str, object] | None = None,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        merged_batch: dict[str, object] = {}
        if batch is not None:
            merged_batch.update(batch)
        merged_batch.update(kwargs)
        seq_a = merged_batch.get("seq_a")
        assert isinstance(seq_a, list)
        lengths = torch.tensor(
            [float(len(seq)) for seq in seq_a],
            dtype=torch.float32,
            device=self.proj.weight.device,
        ).unsqueeze(1)
        logits = self.proj(lengths)
        return {"logits": logits}


def test_trainer_supports_sequence_batches() -> None:
    loader = DataLoader(_SequenceBatchDataset(), batch_size=2, collate_fn=_collate)
    model = _SequenceModel()
    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        optimizer_config=OptimizerConfig(optimizer_type="adamw", lr=1.0e-3),
        scheduler_config=SchedulerConfig(scheduler_type="none"),
        loss_config=LossConfig(loss_type="bce_with_logits", pos_weight=1.0),
        use_amp=False,
        total_epochs=1,
        steps_per_epoch=len(loader),
    )
    stats = trainer.train_one_epoch(loader, epoch_index=0)
    assert "loss" in stats
    assert "lr" in stats


def test_evaluator_supports_sequence_batches() -> None:
    loader = DataLoader(_SequenceBatchDataset(), batch_size=2, collate_fn=_collate)
    model = _SequenceModel()
    evaluator = Evaluator(
        metrics=["accuracy"],
        loss_config=LossConfig(loss_type="bce_with_logits", pos_weight=1.0),
    )
    metrics = evaluator.evaluate(
        model=model,
        data_loader=loader,
        device=torch.device("cpu"),
        prefix="val",
    )
    assert "val_accuracy" in metrics
    assert "val_loss" in metrics
