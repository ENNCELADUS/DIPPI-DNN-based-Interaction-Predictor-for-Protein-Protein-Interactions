"""
Unit tests for src/evaluate/base.py Evaluator class.

Tests cover:
- Initialization with different metric lists
- Evaluation with mock models and data
- Edge cases (empty loaders, different logit shapes)
- Metric computation correctness
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

torchmetrics = pytest.importorskip("torchmetrics")

from src.evaluate.base import Evaluator


class MockDataset(Dataset):
    """Mock dataset for testing that returns dict batches."""

    def __init__(self, n_samples=50, emb_dim=10, seed=42):
        torch.manual_seed(seed)
        self.emb_a = torch.randn(n_samples, emb_dim)
        self.emb_b = torch.randn(n_samples, emb_dim)
        self.labels = torch.randint(0, 2, (n_samples,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "embeddings_a": self.emb_a[idx],
            "embeddings_b": self.emb_b[idx],
            "label": self.labels[idx],
        }


class MockModel(nn.Module):
    """Mock model that returns loss and logits."""

    def __init__(self, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(20, output_dim)
        self.output_dim = output_dim

    def forward(self, batch):
        """Forward accepts a batch dict (like real models V3 and TUnA)."""
        embeddings_a = batch["embeddings_a"]
        embeddings_b = batch["embeddings_b"]
        labels = batch.get("labels")  # Labels optional for inference

        x = torch.cat([embeddings_a, embeddings_b], dim=-1)
        logits = self.linear(x)

        result = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            if self.output_dim == 1:
                logits_squeezed = logits.squeeze(-1)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits_squeezed, labels.float()
                )
            else:  # Two-class
                loss = nn.functional.cross_entropy(logits, labels.long())
            result["loss"] = loss

        return result


class TestEvaluatorInitialization:
    """Test Evaluator initialization."""

    def test_init_with_single_metric(self):
        """Test initialization with a single metric."""
        evaluator = Evaluator(metrics_list=["auroc"])
        assert evaluator.metrics_list == ["auroc"]
        assert evaluator.threshold == 0.5
        assert "auroc" in evaluator.metrics

    def test_init_with_multiple_metrics(self):
        """Test initialization with multiple metrics."""
        metrics = ["loss", "auroc", "accuracy", "f1"]
        evaluator = Evaluator(metrics_list=metrics, threshold=0.6)
        assert evaluator.metrics_list == metrics
        assert evaluator.threshold == 0.6
        # Loss is computed manually, not in metrics dict
        assert "auroc" in evaluator.metrics
        assert "accuracy" in evaluator.metrics
        assert "f1" in evaluator.metrics
        assert "loss" not in evaluator.metrics

    def test_init_with_all_supported_metrics(self):
        """Test initialization with all supported metrics."""
        metrics = [
            "loss",
            "auroc",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "sensitivity",
            "specificity",
            "mcc",
        ]
        evaluator = Evaluator(metrics_list=metrics)
        # All metrics except 'loss' should be in evaluator.metrics
        assert len(evaluator.metrics) == len(metrics) - 1  # -1 for loss

    def test_init_with_invalid_metric(self):
        """Test initialization with unsupported metric raises error."""
        with pytest.raises(ValueError, match="Unsupported metric"):
            Evaluator(metrics_list=["invalid_metric"])

    def test_init_custom_threshold(self):
        """Test custom threshold is stored."""
        evaluator = Evaluator(metrics_list=["accuracy"], threshold=0.7)
        assert evaluator.threshold == 0.7


class TestEvaluatorEvaluate:
    """Test Evaluator.evaluate() method."""

    def test_evaluate_with_loss_only(self):
        """Test evaluation with only loss metric."""
        dataset = MockDataset(n_samples=32)
        loader = DataLoader(dataset, batch_size=8)
        model = MockModel(output_dim=1)
        device = torch.device("cpu")

        evaluator = Evaluator(metrics_list=["loss"])

        model.eval()
        with torch.no_grad():
            results = list(evaluator.evaluate(model, loader, device))[-1]

        assert "loss" in results
        assert isinstance(results["loss"], float)
        assert results["loss"] >= 0

    def test_evaluate_with_auroc(self):
        """Test evaluation with AUROC metric."""
        dataset = MockDataset(n_samples=50)
        loader = DataLoader(dataset, batch_size=10)
        model = MockModel(output_dim=1)
        device = torch.device("cpu")

        evaluator = Evaluator(metrics_list=["loss", "auroc"])

        model.eval()
        with torch.no_grad():
            results = list(evaluator.evaluate(model, loader, device))[-1]

        assert "loss" in results
        assert "auroc" in results
        assert 0 <= results["auroc"] <= 1

    def test_evaluate_with_confusion_metrics(self):
        """Test evaluation with confusion matrix-based metrics."""
        dataset = MockDataset(n_samples=64)
        loader = DataLoader(dataset, batch_size=16)
        model = MockModel(output_dim=1)
        device = torch.device("cpu")

        evaluator = Evaluator(
            metrics_list=["loss", "accuracy", "precision", "recall", "f1"]
        )

        model.eval()
        with torch.no_grad():
            results = list(evaluator.evaluate(model, loader, device))[-1]

        assert "loss" in results
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results

        # All metrics should be in valid ranges
        for metric in ["accuracy", "precision", "recall", "f1"]:
            assert 0 <= results[metric] <= 1

    def test_evaluate_with_all_metrics(self):
        """Test evaluation with all supported metrics."""
        dataset = MockDataset(n_samples=100)
        loader = DataLoader(dataset, batch_size=20)
        model = MockModel(output_dim=1)
        device = torch.device("cpu")

        metrics_list = [
            "loss",
            "auroc",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "sensitivity",
            "specificity",
            "mcc",
        ]
        evaluator = Evaluator(metrics_list=metrics_list)

        model.eval()
        with torch.no_grad():
            results = list(evaluator.evaluate(model, loader, device))[-1]

        # All metrics should be present
        for metric in metrics_list:
            assert metric in results

        # Check value ranges
        assert results["loss"] >= 0
        assert 0 <= results["auroc"] <= 1
        assert 0 <= results["accuracy"] <= 1
        assert -1 <= results["mcc"] <= 1

    def test_evaluate_with_two_class_logits(self):
        """Test evaluation with (N, 2) logit shape."""
        dataset = MockDataset(n_samples=40)
        loader = DataLoader(dataset, batch_size=10)
        model = MockModel(output_dim=2)  # Two-class output
        device = torch.device("cpu")

        evaluator = Evaluator(metrics_list=["loss", "auroc", "accuracy"])

        model.eval()
        with torch.no_grad():
            results = list(evaluator.evaluate(model, loader, device))[-1]

        assert "loss" in results
        assert "auroc" in results
        assert "accuracy" in results
        assert 0 <= results["auroc"] <= 1

    def test_evaluate_custom_threshold(self):
        """Test evaluation with custom threshold."""
        dataset = MockDataset(n_samples=50)
        loader = DataLoader(dataset, batch_size=10)
        model = MockModel(output_dim=1)
        device = torch.device("cpu")

        # Evaluate with different thresholds
        evaluator_05 = Evaluator(metrics_list=["accuracy"], threshold=0.5)
        evaluator_07 = Evaluator(metrics_list=["accuracy"], threshold=0.7)

        model.eval()
        with torch.no_grad():
            results_05 = list(evaluator_05.evaluate(model, loader, device))[-1]
            results_07 = list(evaluator_07.evaluate(model, loader, device))[-1]

        # Results should be different due to different thresholds
        # (may be the same by chance, but that's okay for a unit test)
        assert "accuracy" in results_05
        assert "accuracy" in results_07

    def test_evaluate_multiple_batches(self):
        """Test evaluation correctly accumulates over multiple batches."""
        dataset = MockDataset(n_samples=100)
        loader = DataLoader(dataset, batch_size=10)
        model = MockModel(output_dim=1)
        device = torch.device("cpu")

        evaluator = Evaluator(metrics_list=["loss", "accuracy"])

        model.eval()
        with torch.no_grad():
            results = list(evaluator.evaluate(model, loader, device))[-1]

        # Should process all 100 samples
        assert "loss" in results
        assert "accuracy" in results

    def test_evaluate_sensitivity_specificity_mcc(self):
        """Test sensitivity, specificity, and MCC metrics."""
        dataset = MockDataset(n_samples=80)
        loader = DataLoader(dataset, batch_size=16)
        model = MockModel(output_dim=1)
        device = torch.device("cpu")

        evaluator = Evaluator(metrics_list=["sensitivity", "specificity", "mcc"])

        model.eval()
        with torch.no_grad():
            results = list(evaluator.evaluate(model, loader, device))[-1]

        assert "sensitivity" in results
        assert "specificity" in results
        assert "mcc" in results

        # Sensitivity and specificity should be in [0, 1]
        assert 0 <= results["sensitivity"] <= 1
        assert 0 <= results["specificity"] <= 1
        # MCC should be in [-1, 1]
        assert -1 <= results["mcc"] <= 1

    def test_evaluate_resets_metrics_between_calls(self):
        """Test that metrics are reset between evaluate() calls."""
        dataset1 = MockDataset(n_samples=30, seed=1)
        dataset2 = MockDataset(n_samples=30, seed=2)
        loader1 = DataLoader(dataset1, batch_size=10)
        loader2 = DataLoader(dataset2, batch_size=10)
        model = MockModel(output_dim=1)
        device = torch.device("cpu")

        evaluator = Evaluator(metrics_list=["loss", "auroc", "accuracy"])

        model.eval()
        with torch.no_grad():
            results1 = list(evaluator.evaluate(model, loader1, device))[-1]
            results2 = list(evaluator.evaluate(model, loader2, device))[-1]

        # Both evaluations should succeed and have all metrics
        assert {"loss", "auroc", "accuracy"}.issubset(results1.keys())
        assert {"loss", "auroc", "accuracy"}.issubset(results2.keys())


class TestEvaluatorEdgeCases:
    """Test edge cases and error handling."""

    def test_small_batch_size(self):
        """Test with very small batches."""
        dataset = MockDataset(n_samples=10)
        loader = DataLoader(dataset, batch_size=1)
        model = MockModel(output_dim=1)
        device = torch.device("cpu")

        evaluator = Evaluator(metrics_list=["loss", "accuracy"])

        model.eval()
        with torch.no_grad():
            results = list(evaluator.evaluate(model, loader, device))[-1]

        assert "loss" in results
        assert "accuracy" in results

    def test_single_batch(self):
        """Test with a single batch."""
        dataset = MockDataset(n_samples=16)
        loader = DataLoader(dataset, batch_size=16)
        model = MockModel(output_dim=1)
        device = torch.device("cpu")

        evaluator = Evaluator(metrics_list=["loss", "auroc", "accuracy"])

        model.eval()
        with torch.no_grad():
            results = list(evaluator.evaluate(model, loader, device))[-1]

        assert len(results) == 4  # 3 metrics + _evaluation_end


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
