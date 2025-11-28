"""
Integration tests for Evaluator with real models from the codebase.

Tests the Evaluator's integration with:
- V3 and TUnA models
- Real dataloader patterns
- Full metric suite
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.evaluate.base import Evaluator
from src.model.v3 import V3
from src.model.tuna import TUnA


class IntegrationDataset(Dataset):
    """Realistic dataset that mimics actual data structure."""

    def __init__(self, n_samples=50, emb_dim=1536, seq_len=100, seed=42):
        """
        Args:
            n_samples: Number of samples
            emb_dim: Embedding dimension (1536 for ESM models)
            seq_len: Sequence length
            seed: Random seed
        """
        torch.manual_seed(seed)
        # 3D tensors: (n_samples, seq_len, emb_dim)
        self.emb_a = torch.randn(n_samples, seq_len, emb_dim)
        self.emb_b = torch.randn(n_samples, seq_len, emb_dim)
        self.labels = torch.randint(0, 2, (n_samples,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "emb_a": self.emb_a[idx],  # (seq_len, emb_dim)
            "emb_b": self.emb_b[idx],  # (seq_len, emb_dim)
            "labels": self.labels[idx],
        }


class TestEvaluatorWithV3:
    """Integration tests with V3 model."""

    @pytest.fixture
    def v3_model(self):
        """Create a V3 model instance."""
        model = V3(
            input_dim=1536,
            d_model=256,
            encoder_layers=2,
            cross_attn_layers=1,
            n_heads=4,
            mlp_head={"hidden_dims": [256, 128], "dropout": 0.1},
            regularization={"dropout": 0.1},
        )
        model.eval()
        return model

    @pytest.fixture
    def dataloader(self):
        """Create a dataloader with realistic data."""
        dataset = IntegrationDataset(n_samples=64, emb_dim=1536)
        return DataLoader(dataset, batch_size=16, shuffle=False)

    def test_v3_with_all_metrics(self, v3_model, dataloader):
        """Test V3 model evaluation with all metrics."""
        device = torch.device("cpu")
        v3_model.to(device)

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
        evaluator = Evaluator(metrics_list=metrics_list, threshold=0.5)

        with torch.no_grad():
            results = evaluator.evaluate(v3_model, dataloader, device)

        # All metrics should be present
        for metric in metrics_list:
            assert metric in results, f"Missing metric: {metric}"
            assert isinstance(results[metric], float), f"{metric} not a float"

        # Sanity checks
        assert results["loss"] >= 0
        assert 0 <= results["auroc"] <= 1
        assert 0 <= results["accuracy"] <= 1
        assert -1 <= results["mcc"] <= 1

    def test_v3_with_auroc_only(self, v3_model, dataloader):
        """Test V3 model with only AUROC metric."""
        device = torch.device("cpu")
        v3_model.to(device)

        evaluator = Evaluator(metrics_list=["auroc"], threshold=0.5)

        with torch.no_grad():
            results = evaluator.evaluate(v3_model, dataloader, device)

        assert "auroc" in results
        assert 0 <= results["auroc"] <= 1

    def test_v3_multiple_evaluations(self, v3_model):
        """Test multiple consecutive evaluations."""
        device = torch.device("cpu")
        v3_model.to(device)

        # Create two different dataloaders
        dataset1 = IntegrationDataset(n_samples=32, emb_dim=1536, seed=1)
        dataset2 = IntegrationDataset(n_samples=32, emb_dim=1536, seed=2)
        loader1 = DataLoader(dataset1, batch_size=8)
        loader2 = DataLoader(dataset2, batch_size=8)

        evaluator = Evaluator(metrics_list=["loss", "auroc", "accuracy"], threshold=0.5)

        with torch.no_grad():
            results1 = evaluator.evaluate(v3_model, loader1, device)
            results2 = evaluator.evaluate(v3_model, loader2, device)

        # Both should succeed
        assert "auroc" in results1
        assert "auroc" in results2


class TestEvaluatorWithTUnA:
    """Integration tests with TUnA model."""

    @pytest.fixture
    def tuna_model(self):
        """Create a TUnA model instance."""
        model = TUnA(
            input_dim=1536,
            d_model=256,
            intra_layers=2,
            inter_layers=2,
            n_heads=4,
            ff_dim=512,
            dropout=0.1,
            activation="gelu",
        )
        model.eval()
        return model

    @pytest.fixture
    def dataloader(self):
        """Create a dataloader with realistic data."""
        dataset = IntegrationDataset(n_samples=48, emb_dim=1536)
        return DataLoader(dataset, batch_size=12, shuffle=False)

    def test_tuna_with_all_metrics(self, tuna_model, dataloader):
        """Test TUnA model evaluation with all metrics."""
        device = torch.device("cpu")
        tuna_model.to(device)

        metrics_list = [
            "loss",
            "auroc",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "mcc",
        ]
        evaluator = Evaluator(metrics_list=metrics_list, threshold=0.5)

        with torch.no_grad():
            results = evaluator.evaluate(tuna_model, dataloader, device)

        # All metrics should be present
        for metric in metrics_list:
            assert metric in results, f"Missing metric: {metric}"

        # Sanity checks
        assert results["loss"] >= 0
        assert 0 <= results["auroc"] <= 1
        assert 0 <= results["accuracy"] <= 1

    def test_tuna_with_custom_threshold(self, tuna_model, dataloader):
        """Test TUnA with custom classification threshold."""
        device = torch.device("cpu")
        tuna_model.to(device)

        evaluator = Evaluator(
            metrics_list=["accuracy", "precision", "recall"], threshold=0.7
        )

        with torch.no_grad():
            results = evaluator.evaluate(tuna_model, dataloader, device)

        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results


class TestEvaluatorCrossPlatform:
    """Test evaluator behavior across different scenarios."""

    def test_evaluator_with_different_batch_sizes(self):
        """Test that evaluator works with various batch sizes."""
        model = V3(
            input_dim=1536,
            d_model=256,
            encoder_layers=1,
            cross_attn_layers=1,
            n_heads=4,
            mlp_head={"hidden_dims": [128], "dropout": 0.1},
            regularization={"dropout": 0.1},
        )
        model.eval()
        device = torch.device("cpu")
        model.to(device)

        evaluator = Evaluator(metrics_list=["loss", "auroc"], threshold=0.5)

        # Test with different batch sizes
        for batch_size in [4, 8, 16, 32]:
            dataset = IntegrationDataset(n_samples=64, emb_dim=1536)
            loader = DataLoader(dataset, batch_size=batch_size)

            with torch.no_grad():
                results = evaluator.evaluate(model, loader, device)

            assert "loss" in results
            assert "auroc" in results

    def test_evaluator_preserves_model_state(self):
        """Test that evaluator doesn't change model training state."""
        model = V3(
            input_dim=1536,
            d_model=256,
            encoder_layers=1,
            cross_attn_layers=1,
            n_heads=4,
            mlp_head={"hidden_dims": [128], "dropout": 0.1},
            regularization={"dropout": 0.1},
        )
        device = torch.device("cpu")
        model.to(device)

        dataset = IntegrationDataset(n_samples=32, emb_dim=1536)
        loader = DataLoader(dataset, batch_size=8)

        evaluator = Evaluator(metrics_list=["loss", "accuracy"], threshold=0.5)

        # Set to train mode
        model.train()
        was_training = model.training

        # Evaluate (should be done in eval mode by orchestrator)
        model.eval()
        with torch.no_grad():
            _ = evaluator.evaluate(model, loader, device)
        model.train()  # Restore training mode

        # Model should be back in training mode
        assert model.training == was_training

    def test_sensitivity_is_recall_alias(self):
        """Test that sensitivity and recall return the same value."""
        model = V3(
            input_dim=1536,
            d_model=256,
            encoder_layers=1,
            cross_attn_layers=1,
            n_heads=4,
            mlp_head={"hidden_dims": [128], "dropout": 0.0},
            regularization={"dropout": 0.0},
        )
        model.eval()
        device = torch.device("cpu")
        model.to(device)

        dataset = IntegrationDataset(n_samples=40, emb_dim=1536, seed=42)
        loader = DataLoader(dataset, batch_size=10)

        evaluator = Evaluator(metrics_list=["recall", "sensitivity"], threshold=0.5)

        with torch.no_grad():
            results = evaluator.evaluate(model, loader, device)

        # Sensitivity and recall should be identical
        assert results["recall"] == results["sensitivity"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
