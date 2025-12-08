"""
Unit tests for DistributionAligner.
"""

import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.finetune.distribution_alignment import DistributionAligner


class SyntheticLogitDataset(Dataset):
    """Dataset that stores precomputed logits and labels."""

    def __init__(self, logits, labels):
        self.logits_tensor = torch.tensor(logits, dtype=torch.float32)
        self.labels_tensor = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels_tensor)

    def __getitem__(self, idx):
        return {
            "calibration_feature": self.logits_tensor[idx],
            "label": self.labels_tensor[idx],
        }


class PassthroughLogitModel(nn.Module):
    """Model that simply returns the provided logits."""

    def forward(self, batch):
        logits = batch["calibration_feature"].unsqueeze(-1)
        result = {"logits": logits}
        labels = batch.get("label")
        if labels is not None:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(-1), labels.float()
            )
            result["loss"] = loss
        return result


def build_loader(logits, labels, batch_size=8):
    """Helper to build dataloader for tests."""
    dataset = SyntheticLogitDataset(logits=logits, labels=labels)
    return DataLoader(dataset, batch_size=batch_size)


def test_distribution_aligner_aligns_prior():
    """Calibrate logits so mean probability matches target."""
    logits = torch.linspace(-3, 2, steps=40).tolist()
    labels = [0] * 28 + [1] * 12
    loader = build_loader(logits, labels, batch_size=10)
    model = PassthroughLogitModel()
    aligner = DistributionAligner(target_prior=0.5, search_steps=32)

    model.eval()
    with torch.no_grad():
        result = aligner.calibrate_and_search(model, loader, torch.device("cpu"))

    raw_prior = torch.sigmoid(torch.tensor(logits)).mean().item()
    assert result["predicted_prior"] == pytest.approx(raw_prior, rel=1e-4)
    assert result["calibrated_prior"] == pytest.approx(0.5, rel=1e-3)
    assert (
        result["bias"] > 0
    )  # target_prior > predicted_prior so bias should increase logits
    assert 0.0 <= result["threshold"] <= 1.0
    assert "f1" in result["metrics"]


def test_distribution_aligner_supports_balanced_accuracy_metric():
    """search_metric switch should work without errors."""
    logits = torch.linspace(-2, 3, steps=30).tolist()
    labels = [0] * 10 + [1] * 20
    loader = build_loader(logits, labels, batch_size=6)
    model = PassthroughLogitModel()
    aligner = DistributionAligner(
        target_prior=0.4, search_metric="balanced_accuracy", search_steps=16
    )

    model.eval()
    with torch.no_grad():
        result = aligner.calibrate_and_search(model, loader, torch.device("cpu"))

    assert "balanced_accuracy" in result["metrics"]
    assert 0.0 <= result["metrics"]["balanced_accuracy"] <= 1.0


def test_distribution_aligner_handles_single_class_metrics():
    """AUROC/AUPRC should return NaN when only one class is present."""
    logits = [0.0] * 12
    labels = [1] * 12
    loader = build_loader(logits, labels, batch_size=4)
    model = PassthroughLogitModel()
    aligner = DistributionAligner(target_prior=0.8, search_steps=8)

    model.eval()
    with torch.no_grad():
        result = aligner.calibrate_and_search(model, loader, torch.device("cpu"))

    assert math.isnan(result["metrics"]["auroc"])
    assert math.isnan(result["metrics"]["auprc"])


def test_distribution_aligner_validates_metric_name():
    """Invalid metric should raise ValueError."""
    with pytest.raises(ValueError, match="search_metric must be one of"):
        DistributionAligner(target_prior=0.5, search_metric="invalid")


def test_distribution_aligner_supports_rank_based_metrics():
    """AUROC/AUPRC should be accepted as search metrics without crashing."""
    logits = torch.linspace(-1, 1, steps=10).tolist()
    labels = [0] * 5 + [1] * 5
    loader = build_loader(logits, labels, batch_size=5)
    model = PassthroughLogitModel()
    aligner = DistributionAligner(
        target_prior=0.5, search_metric="auprc", search_steps=16
    )

    model.eval()
    with torch.no_grad():
        result = aligner.calibrate_and_search(model, loader, torch.device("cpu"))

    assert result["threshold"] == pytest.approx(0.5)
    assert "auprc" in result["metrics"]
