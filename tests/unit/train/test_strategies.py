"""Unit tests for training strategies."""

import pytest
import torch
import torch.nn as nn

from src.train.base import Trainer
from src.train.strategies import BaseStrategy, StagedUnfreeze


class SimpleModel(nn.Module):
    """Simple model for testing strategies."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 20)
        self.cross_attention = nn.Linear(20, 20)
        self.mlp_head = nn.Linear(20, 1)

    def forward(self, batch):
        x = batch["x"]
        x = self.encoder(x)
        x = self.cross_attention(x)
        x = self.mlp_head(x)
        return {"loss": x.mean()}


@pytest.fixture
def simple_model():
    """Fixture providing a simple model."""
    return SimpleModel()


@pytest.fixture
def device():
    """Fixture providing CPU device."""
    return torch.device("cpu")


class TestBaseStrategy:
    """Test BaseStrategy class."""

    def test_base_strategy_no_op(self, simple_model, device):
        """Test that BaseStrategy hooks are no-ops."""
        optimizer_cfg = {"type": "adamw", "lr": 1e-3, "weight_decay": 0.01}

        strategy = BaseStrategy()
        trainer = Trainer(
            model=simple_model,
            device=device,
            optimizer_cfg=optimizer_cfg,
            strategy=strategy,
        )

        # All hooks should run without error and do nothing
        strategy.on_train_begin(trainer)
        strategy.on_epoch_begin(trainer, 0)
        strategy.on_epoch_end(trainer, 0)

        # Model should still be trainable
        assert sum(p.numel() for p in simple_model.parameters() if p.requires_grad) > 0


class TestStagedUnfreeze:
    """Test StagedUnfreeze strategy."""

    def test_staged_unfreeze_initialization(self):
        """Test StagedUnfreeze initialization with valid schedule."""
        schedule = [
            {"at_epoch": 0, "freeze": ["encoder"]},
            {"at_epoch": 3, "unfreeze": ["encoder"]},
        ]

        strategy = StagedUnfreeze(schedule=schedule)
        assert strategy.schedule == schedule

    def test_staged_unfreeze_empty_schedule_error(self):
        """Test that empty schedule raises ValueError."""
        with pytest.raises(ValueError, match="non-empty list"):
            StagedUnfreeze(schedule=[])

    def test_staged_unfreeze_invalid_schedule_type(self):
        """Test that non-list schedule raises ValueError."""
        with pytest.raises(ValueError, match="non-empty list"):
            StagedUnfreeze(schedule="not a list")

    def test_staged_unfreeze_missing_at_epoch(self):
        """Test that schedule entry without at_epoch raises ValueError."""
        schedule = [{"freeze": ["encoder"]}]  # Missing at_epoch

        with pytest.raises(ValueError, match="missing required 'at_epoch'"):
            StagedUnfreeze(schedule=schedule)

    def test_staged_unfreeze_freeze_at_epoch_0(self, simple_model, device):
        """Test freezing encoder at epoch 0."""
        schedule = [{"at_epoch": 0, "freeze": ["encoder."]}]

        optimizer_cfg = {"type": "adamw", "lr": 1e-3, "weight_decay": 0.01}

        strategy = StagedUnfreeze(schedule=schedule)
        trainer = Trainer(
            model=simple_model,
            device=device,
            optimizer_cfg=optimizer_cfg,
            strategy=strategy,
        )

        # Initially all params should be trainable
        initial_trainable = sum(
            p.numel() for p in simple_model.parameters() if p.requires_grad
        )
        assert initial_trainable > 0

        # Apply training-begin hook (epoch 0)
        strategy.on_train_begin(trainer)

        # Encoder should be frozen
        assert not simple_model.encoder.weight.requires_grad
        assert not simple_model.encoder.bias.requires_grad

        # Other layers should still be trainable
        assert simple_model.mlp_head.weight.requires_grad
        assert simple_model.cross_attention.weight.requires_grad

    def test_staged_unfreeze_unfreeze_at_epoch_3(self, simple_model, device):
        """Test unfreezing encoder at epoch 3."""
        schedule = [
            {"at_epoch": 0, "freeze": ["encoder."]},
            {"at_epoch": 3, "unfreeze": ["encoder."]},
        ]

        optimizer_cfg = {"type": "adamw", "lr": 1e-3, "weight_decay": 0.01}

        strategy = StagedUnfreeze(schedule=schedule)
        trainer = Trainer(
            model=simple_model,
            device=device,
            optimizer_cfg=optimizer_cfg,
            strategy=strategy,
        )

        # Freeze at epoch 0 (train begin)
        strategy.on_train_begin(trainer)
        assert not simple_model.encoder.weight.requires_grad

        # Unfreeze at epoch 3
        strategy.on_epoch_begin(trainer, 3)
        assert simple_model.encoder.weight.requires_grad

    def test_staged_unfreeze_with_optimizer_rebuild(self, simple_model, device):
        """Test that optimizer is rebuilt when params change."""
        schedule = [
            {"at_epoch": 0, "freeze": ["encoder."]},
            {
                "at_epoch": 3,
                "unfreeze": ["encoder."],
                "optimizer_cfg": {
                    "type": "adamw",
                    "lr": 5e-4,  # Different LR
                    "weight_decay": 0.01,
                },
            },
        ]

        optimizer_cfg = {"type": "adamw", "lr": 1e-3, "weight_decay": 0.01}

        strategy = StagedUnfreeze(schedule=schedule)
        trainer = Trainer(
            model=simple_model,
            device=device,
            optimizer_cfg=optimizer_cfg,
            strategy=strategy,
        )

        # Apply initial freeze (epoch 0)
        strategy.on_train_begin(trainer)

        # Initial LR
        initial_lr = trainer.optimizer.param_groups[0]["lr"]
        assert initial_lr == 1e-3

        # Trigger unfreeze with new optimizer at epoch 3
        strategy.on_epoch_begin(trainer, 3)

        # LR should have changed
        new_lr = trainer.optimizer.param_groups[0]["lr"]
        assert new_lr == 5e-4

        # Encoder should be unfrozen
        assert simple_model.encoder.weight.requires_grad

    def test_staged_unfreeze_no_match_at_epoch(self, simple_model, device):
        """Test that non-matching epoch does nothing."""
        schedule = [{"at_epoch": 5, "freeze": ["encoder."]}]

        optimizer_cfg = {"type": "adamw", "lr": 1e-3, "weight_decay": 0.01}

        strategy = StagedUnfreeze(schedule=schedule)
        trainer = Trainer(
            model=simple_model,
            device=device,
            optimizer_cfg=optimizer_cfg,
            strategy=strategy,
        )

        # Trigger at epoch 0 (no match)
        strategy.on_epoch_begin(trainer, 0)

        # All params should still be trainable
        assert simple_model.encoder.weight.requires_grad
        assert simple_model.mlp_head.weight.requires_grad

    def test_staged_unfreeze_multiple_patterns(self, simple_model, device):
        """Test freezing multiple patterns at once."""
        schedule = [{"at_epoch": 0, "freeze": ["encoder.", "cross_attention."]}]

        optimizer_cfg = {"type": "adamw", "lr": 1e-3, "weight_decay": 0.01}

        strategy = StagedUnfreeze(schedule=schedule)
        trainer = Trainer(
            model=simple_model,
            device=device,
            optimizer_cfg=optimizer_cfg,
            strategy=strategy,
        )

        # Trigger freeze via train begin
        strategy.on_train_begin(trainer)

        # Both encoder and cross_attention should be frozen
        assert not simple_model.encoder.weight.requires_grad
        assert not simple_model.cross_attention.weight.requires_grad

        # Head should still be trainable
        assert simple_model.mlp_head.weight.requires_grad

    def test_freeze_then_unfreeze_override(self, simple_model, device):
        """Test that unfreeze can override freeze in same epoch."""
        schedule = [
            {
                "at_epoch": 0,
                "freeze": ["encoder.", "mlp_head."],
                "unfreeze": ["mlp_head."],  # Override freeze for head
            }
        ]

        optimizer_cfg = {"type": "adamw", "lr": 1e-3, "weight_decay": 0.01}

        strategy = StagedUnfreeze(schedule=schedule)
        trainer = Trainer(
            model=simple_model,
            device=device,
            optimizer_cfg=optimizer_cfg,
            strategy=strategy,
        )

        # Trigger at epoch 0
        strategy.on_train_begin(trainer)

        # Encoder should be frozen
        assert not simple_model.encoder.weight.requires_grad

        # Head should be unfrozen (unfreeze overrides freeze)
        assert simple_model.mlp_head.weight.requires_grad

    def test_staged_unfreeze_applies_once(self, simple_model, device):
        """Ensure a schedule entry is applied only once at epoch boundaries."""
        schedule = [{"at_epoch": 0, "freeze": ["encoder."]}]

        optimizer_cfg = {"type": "adamw", "lr": 1e-3, "weight_decay": 0.01}

        strategy = StagedUnfreeze(schedule=schedule)
        trainer = Trainer(
            model=simple_model,
            device=device,
            optimizer_cfg=optimizer_cfg,
            strategy=strategy,
        )

        strategy.on_train_begin(trainer)
        optimizer_id_after_first_apply = id(trainer.optimizer)

        # Calling on_epoch_begin again for epoch 0 should be a no-op.
        strategy.on_epoch_begin(trainer, 0)
        assert id(trainer.optimizer) == optimizer_id_after_first_apply
