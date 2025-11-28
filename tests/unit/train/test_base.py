"""
Unit tests for src/train/base.py

Tests the Trainer class implementation including:
- Optimizer building with various configs
- Scheduler building with various configs
- Training loop basic functionality
- Rebuild optimizer/scheduler
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1),
    )
    return model


@pytest.fixture
def device():
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader that returns dict batches."""
    batch1 = {
        "input": torch.randn(4, 10),
        "label": torch.randint(0, 2, (4,)).float(),
    }
    batch2 = {
        "input": torch.randn(4, 10),
        "label": torch.randint(0, 2, (4,)).float(),
    }
    return [batch1, batch2]


class TestTrainerOptimizer:
    """Test optimizer building."""

    def test_build_optimizer_dict_adamw(self, simple_model, device):
        """Test building AdamW optimizer from dict config."""
        from src.train.base import Trainer

        optimizer_cfg = {
            "type": "adamw",
            "lr": 1e-3,
            "weight_decay": 0.01,
        }

        trainer = Trainer(
            model=simple_model,
            device=device,
            optimizer_cfg=optimizer_cfg,
        )

        assert trainer.optimizer is not None
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        assert trainer.optimizer.param_groups[0]["lr"] == 1e-3
        assert trainer.optimizer.param_groups[0]["weight_decay"] == 0.01

    def test_build_optimizer_dict_adam(self, simple_model, device):
        """Test building Adam optimizer from dict config."""
        from src.train.base import Trainer

        optimizer_cfg = {
            "type": "adam",
            "lr": 5e-4,
        }

        trainer = Trainer(
            model=simple_model,
            device=device,
            optimizer_cfg=optimizer_cfg,
        )

        assert isinstance(trainer.optimizer, torch.optim.Adam)
        assert trainer.optimizer.param_groups[0]["lr"] == 5e-4

    def test_build_optimizer_with_betas(self, simple_model, device):
        """Test optimizer building with beta1/beta2 params."""
        from src.train.base import Trainer

        optimizer_cfg = {
            "type": "adamw",
            "lr": 1e-3,
            "beta1": 0.9,
            "beta2": 0.999,
        }

        trainer = Trainer(
            model=simple_model,
            device=device,
            optimizer_cfg=optimizer_cfg,
        )

        assert trainer.optimizer.param_groups[0]["betas"] == (0.9, 0.999)

    def test_build_optimizer_with_param_groups(self, device):
        """Test LLRD parameter groups with different learning rates."""
        from src.train.base import Trainer

        # Model with distinct components for LLRD
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )
        # Name the layers so we can pattern match
        model[0]._get_name = lambda: "encoder.layer1"
        model[2]._get_name = lambda: "head.output"

        optimizer_cfg = {
            "type": "adamw",
            "lr": 1e-5,  # Base LR for unmatched
            "weight_decay": 0.01,
            "param_groups": [
                {"name": "encoder", "pattern": "0", "lr": 1e-4, "weight_decay": 0.015},
                {"name": "head", "pattern": "2", "lr": 3e-4, "weight_decay": 0.0},
            ],
        }

        trainer = Trainer(
            model=model,
            device=device,
            optimizer_cfg=optimizer_cfg,
        )

        # Should have 2 configured groups (encoder and head)
        assert len(trainer.optimizer.param_groups) >= 2

        # Check learning rates are different (LLRD)
        lrs = [pg["lr"] for pg in trainer.optimizer.param_groups]
        assert 1e-4 in lrs  # encoder LR
        assert 3e-4 in lrs  # head LR

    def test_build_optimizer_default_adamw(self, simple_model, device):
        """Test that default optimizer is AdamW."""
        from src.train.base import Trainer

        optimizer_cfg = {
            "lr": 1e-3,
            # No "type" specified
        }

        trainer = Trainer(
            model=simple_model,
            device=device,
            optimizer_cfg=optimizer_cfg,
        )

        assert isinstance(trainer.optimizer, torch.optim.AdamW)

    def test_build_optimizer_callable(self, simple_model, device):
        """Test building optimizer from callable."""
        from src.train.base import Trainer

        optimizer_cfg = lambda: torch.optim.SGD(simple_model.parameters(), lr=0.1)

        trainer = Trainer(
            model=simple_model,
            device=device,
            optimizer_cfg=optimizer_cfg,
        )

        assert isinstance(trainer.optimizer, torch.optim.SGD)


class TestTrainerScheduler:
    """Test scheduler building."""

    def test_build_scheduler_none(self, simple_model, device):
        """Test that scheduler is None when not configured."""
        from src.train.base import Trainer

        trainer = Trainer(
            model=simple_model,
            device=device,
            optimizer_cfg={"type": "adamw", "lr": 1e-3},
            scheduler_cfg=None,
        )

        assert trainer.scheduler is None

    def test_build_scheduler_cosine(self, simple_model, device):
        """Test building CosineAnnealingLR scheduler."""
        from src.train.base import Trainer

        scheduler_cfg = {
            "type": "cosineannealinglr",
            "T_max": 10,
        }

        trainer = Trainer(
            model=simple_model,
            device=device,
            optimizer_cfg={"type": "adamw", "lr": 1e-3},
            scheduler_cfg=scheduler_cfg,
        )

        assert trainer.scheduler is not None
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_build_scheduler_onecycle(self, simple_model, device):
        """Test building OneCycleLR scheduler."""
        from src.train.base import Trainer

        scheduler_cfg = {
            "type": "onecycle",
            "max_lr": 1e-3,
            "total_steps": 100,
        }

        trainer = Trainer(
            model=simple_model,
            device=device,
            optimizer_cfg={"type": "adamw", "lr": 1e-4},
            scheduler_cfg=scheduler_cfg,
        )

        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.OneCycleLR)

    def test_build_scheduler_callable(self, simple_model, device):
        """Test building scheduler from callable."""
        from src.train.base import Trainer

        def scheduler_fn(optimizer):
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

        trainer = Trainer(
            model=simple_model,
            device=device,
            optimizer_cfg={"type": "adamw", "lr": 1e-3},
            scheduler_cfg=scheduler_fn,
        )

        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.StepLR)


class TestTrainerTraining:
    """Test training loop."""

    def test_train_one_epoch_returns_dict(self, simple_model, device, mock_dataloader):
        """Test that train_one_epoch returns dict with loss and lr."""
        from src.train.base import Trainer

        # Wrap model to return dict with loss (expects batch dict)
        class ModelWrapper(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.model = base_model

            def forward(self, batch):
                output = self.model(batch["input"])
                loss = nn.functional.binary_cross_entropy_with_logits(
                    output.squeeze(), batch["label"]
                )
                return {"loss": loss}

        wrapped_model = ModelWrapper(simple_model)

        trainer = Trainer(
            model=wrapped_model,
            device=device,
            optimizer_cfg={"type": "adamw", "lr": 1e-3},
        )

        result = trainer.train_one_epoch(mock_dataloader)

        assert isinstance(result, dict)
        assert "loss" in result
        assert "lr" in result
        assert isinstance(result["loss"], float)
        assert isinstance(result["lr"], float)

    def test_train_one_epoch_updates_model(self, simple_model, device, mock_dataloader):
        """Test that training actually updates model parameters."""
        from src.train.base import Trainer

        class ModelWrapper(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.model = base_model

            def forward(self, batch):
                output = self.model(batch["input"])
                loss = nn.functional.binary_cross_entropy_with_logits(
                    output.squeeze(), batch["label"]
                )
                return {"loss": loss}

        wrapped_model = ModelWrapper(simple_model)

        # Store initial parameters
        initial_params = [p.clone() for p in wrapped_model.parameters()]

        trainer = Trainer(
            model=wrapped_model,
            device=device,
            optimizer_cfg={"type": "sgd", "lr": 0.1},
        )

        trainer.train_one_epoch(mock_dataloader)

        # Check that at least one parameter changed
        params_changed = any(
            not torch.allclose(p1, p2)
            for p1, p2 in zip(initial_params, wrapped_model.parameters())
        )
        assert params_changed, "Model parameters should be updated after training"

    def test_rebuild_optimizer_and_scheduler(self, simple_model, device):
        """Test rebuilding optimizer and scheduler."""
        from src.train.base import Trainer

        trainer = Trainer(
            model=simple_model,
            device=device,
            optimizer_cfg={"type": "adamw", "lr": 1e-3},
            scheduler_cfg={"type": "steplr", "step_size": 5},
        )

        old_optimizer = trainer.optimizer
        old_scheduler = trainer.scheduler

        trainer.rebuild_optimizer_and_scheduler()

        # Should be new instances
        assert trainer.optimizer is not old_optimizer
        assert trainer.scheduler is not old_scheduler

    def test_amp_disabled_by_default(self, simple_model, device):
        """Test that AMP is disabled by default."""
        from src.train.base import Trainer

        trainer = Trainer(
            model=simple_model,
            device=device,
            optimizer_cfg={"type": "adamw", "lr": 1e-3},
        )

        assert trainer.use_amp is False
        assert trainer.scaler is None

    def test_amp_enabled(self, simple_model, device):
        """Test that AMP can be enabled."""
        from src.train.base import Trainer

        trainer = Trainer(
            model=simple_model,
            device=device,
            optimizer_cfg={"type": "adamw", "lr": 1e-3},
            amp_cfg={"enabled": True, "dtype": "fp16"},
        )

        assert trainer.use_amp is True
        assert trainer.scaler is not None


class TestTrainerEdgeCases:
    """Test edge cases and error handling."""

    def test_no_trainable_parameters_raises_error(self, device):
        """Test that error is raised when model has no trainable parameters."""
        from src.train.base import Trainer

        model = nn.Sequential(nn.Linear(10, 1))
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        with pytest.raises(ValueError, match="No trainable parameters"):
            Trainer(
                model=model,
                device=device,
                optimizer_cfg={"type": "adamw", "lr": 1e-3},
            )

    def test_unknown_optimizer_type_raises_error(self, simple_model, device):
        """Test that unknown optimizer type raises error."""
        from src.train.base import Trainer

        with pytest.raises(ValueError, match="Unknown optimizer type"):
            Trainer(
                model=simple_model,
                device=device,
                optimizer_cfg={"type": "unknown_optimizer", "lr": 1e-3},
            )

    def test_unknown_scheduler_type_raises_error(self, simple_model, device):
        """Test that unknown scheduler type raises error."""
        from src.train.base import Trainer

        with pytest.raises(ValueError, match="Unknown scheduler type"):
            Trainer(
                model=simple_model,
                device=device,
                optimizer_cfg={"type": "adamw", "lr": 1e-3},
                scheduler_cfg={"type": "unknown_scheduler"},
            )
