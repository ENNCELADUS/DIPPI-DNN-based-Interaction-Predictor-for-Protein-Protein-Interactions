"""
Integration test for Trainer instantiation with real configs.

Tests that run.py can successfully instantiate a Trainer with:
- v3.yaml config
- tuna.yaml config
- Both pretrain and finetune configurations

This ensures end-to-end integration between config parsing, model building,
and trainer instantiation.
"""

import pytest
import torch
from pathlib import Path

from src.utils.config import load_config, extract_keys
from src.train.base import Trainer


@pytest.fixture
def v3_config():
    """Load v3.yaml config."""
    config_path = "configs/v3.yaml"
    if not Path(config_path).exists():
        pytest.skip(f"Config file not found: {config_path}")
    return load_config(config_path)


@pytest.fixture
def tuna_config():
    """Load tuna.yaml config."""
    config_path = "configs/tuna.yaml"
    if not Path(config_path).exists():
        pytest.skip(f"Config file not found: {config_path}")
    return load_config(config_path)


@pytest.fixture
def device():
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def mock_train_loader():
    """Create a minimal mock dataloader for testing."""
    # Return a simple list with one batch
    # Keys match expected model forward signature (emb_a, emb_b, len_a, len_b, label)
    batch = {
        "emb_a": torch.randn(2, 10, 1536),
        "len_a": torch.tensor([10, 10]),
        "emb_b": torch.randn(2, 10, 1536),
        "len_b": torch.tensor([10, 10]),
        "label": torch.randint(0, 2, (2,)).float(),
    }
    return [batch]


def build_model_from_config(cfg, device):
    """Helper to build model from config (mimics run.py logic)."""
    from src.model.v2 import V2
    from src.model.v3 import V3
    from src.model.tuna import TUnA

    model_name = cfg.get("model_config.model")
    model_cfg = extract_keys(cfg, "model_config")
    model_cfg = {k: v for k, v in model_cfg.items() if k != "model"}

    if model_name == "v2":
        model = V2(**model_cfg)
    elif model_name == "v3":
        model = V3(**model_cfg)
    elif model_name == "tuna":
        model = TUnA(**model_cfg)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


class TestTrainerInstantiationV3:
    """Test Trainer instantiation with v3.yaml config."""

    def test_pretrain_trainer_instantiation(self, v3_config, device):
        """Test that pretrain Trainer can be instantiated with v3.yaml."""
        # Build model
        model = build_model_from_config(v3_config, device)

        # Extract pretrain config
        pretrain_cfg = v3_config["pretrain_config"]

        # Prepare scheduler config (mimic run.py logic)
        scheduler_cfg = pretrain_cfg.get("scheduler")
        if scheduler_cfg and scheduler_cfg.get("type", "").lower() in [
            "onecycle",
            "onecyclelr",
        ]:
            scheduler_cfg = scheduler_cfg.copy()
            if (
                "total_steps" not in scheduler_cfg
                and "steps_per_epoch" not in scheduler_cfg
            ):
                # Set a dummy value for testing
                scheduler_cfg["total_steps"] = 100

        # Instantiate Trainer
        trainer = Trainer(
            model=model,
            device=device,
            optimizer_cfg=pretrain_cfg["optimizer"],
            scheduler_cfg=scheduler_cfg,
            amp_cfg={"enabled": pretrain_cfg.get("use_mixed_precision", False)},
            max_norm=pretrain_cfg.get("max_grad_norm"),
        )

        # Verify trainer was created successfully
        assert trainer is not None
        assert trainer.model is model
        assert trainer.device == device
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_finetune_trainer_instantiation(self, v3_config, device):
        """Test that finetune Trainer can be instantiated with v3.yaml."""
        # Build model
        model = build_model_from_config(v3_config, device)

        # Extract finetune config
        finetune_cfg = v3_config["finetune_config"]

        # Prepare scheduler config (use helper from run.py)
        from src.run import prepare_scheduler_config

        scheduler_cfg = prepare_scheduler_config(
            finetune_cfg.get("scheduler"), num_epochs=10, steps_per_epoch=10
        )

        # Instantiate Trainer
        trainer = Trainer(
            model=model,
            device=device,
            optimizer_cfg=finetune_cfg["optimizer"],
            scheduler_cfg=scheduler_cfg,
            amp_cfg={"enabled": finetune_cfg.get("use_mixed_precision", False)},
            strategy=None,
            max_norm=finetune_cfg.get("max_grad_norm"),
        )

        # Verify trainer was created successfully
        assert trainer is not None
        assert trainer.model is model
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None  # warmup_cosine now supported

    def test_v3_trainer_can_train_one_epoch(self, v3_config, device, mock_train_loader):
        """Test that v3 Trainer can actually run train_one_epoch."""
        # Build model
        base_model = build_model_from_config(v3_config, device)

        # Wrap model to compute loss from logits (models don't compute loss themselves)
        class ModelWithLoss(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                self.loss_fn = torch.nn.BCEWithLogitsLoss()

            def forward(self, batch):
                output = self.base_model(batch)
                logits = output["logits"].squeeze()
                labels = batch["label"]
                loss = self.loss_fn(logits, labels)
                return {"loss": loss, "logits": logits}

            def parameters(self):
                return self.base_model.parameters()

        model = ModelWithLoss(base_model)

        # Extract pretrain config
        pretrain_cfg = v3_config["pretrain_config"]

        # Prepare scheduler config
        scheduler_cfg = pretrain_cfg.get("scheduler")
        if scheduler_cfg and scheduler_cfg.get("type", "").lower() in [
            "onecycle",
            "onecyclelr",
        ]:
            scheduler_cfg = scheduler_cfg.copy()
            scheduler_cfg["total_steps"] = 100

        # Instantiate Trainer
        trainer = Trainer(
            model=model,
            device=device,
            optimizer_cfg=pretrain_cfg["optimizer"],
            scheduler_cfg=scheduler_cfg,
            amp_cfg={"enabled": False},  # Disable AMP for CPU testing
            max_norm=pretrain_cfg.get("max_grad_norm"),
        )

        # Run train_one_epoch
        result = trainer.train_one_epoch(mock_train_loader)

        # Verify result format
        assert isinstance(result, dict)
        assert "loss" in result
        assert "lr" in result
        assert isinstance(result["loss"], float)
        assert isinstance(result["lr"], float)
        assert result["loss"] >= 0  # Loss should be non-negative


class TestTrainerInstantiationTUnA:
    """Test Trainer instantiation with tuna.yaml config."""

    def test_pretrain_trainer_instantiation(self, tuna_config, device):
        """Test that pretrain Trainer can be instantiated with tuna.yaml."""
        # Build model
        model = build_model_from_config(tuna_config, device)

        # Extract pretrain config
        pretrain_cfg = tuna_config["pretrain_config"]

        # Prepare scheduler config
        scheduler_cfg = pretrain_cfg.get("scheduler")
        if scheduler_cfg and scheduler_cfg.get("type", "").lower() in [
            "onecycle",
            "onecyclelr",
        ]:
            scheduler_cfg = scheduler_cfg.copy()
            if (
                "total_steps" not in scheduler_cfg
                and "steps_per_epoch" not in scheduler_cfg
            ):
                scheduler_cfg["total_steps"] = 100

        # Handle None/null values in config
        optimizer_cfg = pretrain_cfg["optimizer"]
        amp_enabled = pretrain_cfg.get("use_mixed_precision")
        if amp_enabled is None:
            amp_enabled = False

        # Instantiate Trainer
        trainer = Trainer(
            model=model,
            device=device,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
            amp_cfg={"enabled": amp_enabled},
            max_norm=pretrain_cfg.get("max_grad_norm"),
        )

        # Verify trainer was created successfully
        assert trainer is not None
        assert trainer.model is model
        assert trainer.device == device
        assert trainer.optimizer is not None
        # StepLR scheduler should be created
        assert trainer.scheduler is not None
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.StepLR)

    def test_finetune_trainer_instantiation(self, tuna_config, device):
        """Test that finetune Trainer can be instantiated with tuna.yaml."""
        # Build model
        model = build_model_from_config(tuna_config, device)

        # Extract finetune config
        finetune_cfg = tuna_config["finetune_config"]

        # Prepare scheduler config
        scheduler_cfg = finetune_cfg.get("scheduler")
        if scheduler_cfg and scheduler_cfg.get("type", "").lower() in [
            "onecycle",
            "onecyclelr",
        ]:
            scheduler_cfg = scheduler_cfg.copy()
            if (
                "total_steps" not in scheduler_cfg
                and "steps_per_epoch" not in scheduler_cfg
            ):
                scheduler_cfg["total_steps"] = 100

        # Handle None/null values
        amp_enabled = finetune_cfg.get("use_mixed_precision")
        if amp_enabled is None:
            amp_enabled = False

        # Instantiate Trainer
        trainer = Trainer(
            model=model,
            device=device,
            optimizer_cfg=finetune_cfg["optimizer"],
            scheduler_cfg=scheduler_cfg,
            amp_cfg={"enabled": amp_enabled},
            strategy=None,
            max_norm=finetune_cfg.get("max_grad_norm"),
        )

        # Verify trainer was created successfully
        assert trainer is not None
        assert trainer.model is model
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.StepLR)

    def test_tuna_trainer_can_train_one_epoch(
        self, tuna_config, device, mock_train_loader
    ):
        """Test that TUnA Trainer can actually run train_one_epoch."""
        # Build model
        base_model = build_model_from_config(tuna_config, device)

        # Wrap model to compute loss from logits (models don't compute loss themselves)
        class ModelWithLoss(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                self.loss_fn = torch.nn.BCEWithLogitsLoss()

            def forward(self, batch):
                output = self.base_model(batch)
                logits = output["logits"].squeeze()
                labels = batch["label"]
                loss = self.loss_fn(logits, labels)
                return {"loss": loss, "logits": logits}

            def parameters(self):
                return self.base_model.parameters()

        model = ModelWithLoss(base_model)

        # Extract pretrain config
        pretrain_cfg = tuna_config["pretrain_config"]

        # Prepare scheduler config
        scheduler_cfg = pretrain_cfg.get("scheduler")

        # Instantiate Trainer
        trainer = Trainer(
            model=model,
            device=device,
            optimizer_cfg=pretrain_cfg["optimizer"],
            scheduler_cfg=scheduler_cfg,
            amp_cfg={"enabled": False},  # Disable AMP for CPU testing
            max_norm=pretrain_cfg.get("max_grad_norm"),
        )

        # Run train_one_epoch
        result = trainer.train_one_epoch(mock_train_loader)

        # Verify result format
        assert isinstance(result, dict)
        assert "loss" in result
        assert "lr" in result
        assert isinstance(result["loss"], float)
        assert isinstance(result["lr"], float)
        assert result["loss"] >= 0


class TestTrainerConfigEdgeCases:
    """Test edge cases in Trainer config handling."""

    def test_missing_scheduler_handled(self, v3_config, device):
        """Test that Trainer handles missing scheduler gracefully."""
        model = build_model_from_config(v3_config, device)
        pretrain_cfg = v3_config["pretrain_config"]

        # Instantiate without scheduler
        trainer = Trainer(
            model=model,
            device=device,
            optimizer_cfg=pretrain_cfg["optimizer"],
            scheduler_cfg=None,  # Explicitly None
            amp_cfg={"enabled": False},
        )

        assert trainer.scheduler is None

    def test_amp_disabled_by_default(self, v3_config, device):
        """Test that AMP is disabled when not specified."""
        model = build_model_from_config(v3_config, device)
        pretrain_cfg = v3_config["pretrain_config"]

        # Prepare scheduler
        scheduler_cfg = pretrain_cfg.get("scheduler")
        if scheduler_cfg and scheduler_cfg.get("type", "").lower() in [
            "onecycle",
            "onecyclelr",
        ]:
            scheduler_cfg = scheduler_cfg.copy()
            scheduler_cfg["total_steps"] = 100

        # Instantiate without amp_cfg
        trainer = Trainer(
            model=model,
            device=device,
            optimizer_cfg=pretrain_cfg["optimizer"],
            scheduler_cfg=scheduler_cfg,
        )

        assert trainer.use_amp is False
        assert trainer.scaler is None

    def test_optimizer_lr_parameter_handling(self, tuna_config, device):
        """Test that optimizer 'lr' parameter is handled correctly."""
        model = build_model_from_config(tuna_config, device)
        pretrain_cfg = tuna_config["pretrain_config"]

        # TUnA config has 'lr' directly in optimizer config
        optimizer_cfg = pretrain_cfg["optimizer"]
        assert "lr" in optimizer_cfg

        trainer = Trainer(
            model=model,
            device=device,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=None,
            amp_cfg={"enabled": False},
        )

        # Verify LR was set correctly
        assert trainer.optimizer.param_groups[0]["lr"] == optimizer_cfg["lr"]


class TestLLRDIntegration:
    """Integration tests for LLRD (Layer-wise Learning Rate Decay) with real models."""

    def test_llrd_with_v3_model(self, v3_config, device):
        """Test LLRD param_groups with V3 model."""
        model = build_model_from_config(v3_config, device)

        # LLRD config with different LRs for different components
        optimizer_cfg = {
            "type": "adamw",
            "lr": 1e-5,  # Base LR
            "weight_decay": 0.01,
            "param_groups": [
                {
                    "name": "encoder",
                    "pattern": "encoder.",
                    "lr": 1e-4,
                    "weight_decay": 0.015,
                },
                {
                    "name": "cross_attention",
                    "pattern": "cross_attention.",
                    "lr": 2e-4,
                    "weight_decay": 0.01,
                },
                {
                    "name": "head",
                    "pattern": "mlp_head.",
                    "lr": 3e-4,
                    "weight_decay": 0.0,
                },
            ],
        }

        trainer = Trainer(
            model=model,
            device=device,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=None,
            amp_cfg={"enabled": False},
        )

        # Should have multiple param groups
        assert len(trainer.optimizer.param_groups) >= 3

        # Verify different learning rates
        lrs = {pg["lr"] for pg in trainer.optimizer.param_groups}
        assert 1e-4 in lrs  # encoder
        assert 2e-4 in lrs  # cross_attention
        assert 3e-4 in lrs  # head

        # Verify all parameters are assigned to some group
        total_params_in_model = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total_params_in_groups = sum(
            sum(p.numel() for p in pg["params"])
            for pg in trainer.optimizer.param_groups
        )
        assert total_params_in_model == total_params_in_groups

    def test_llrd_with_tuna_model(self, tuna_config, device):
        """Test LLRD param_groups with TUnA model."""
        model = build_model_from_config(tuna_config, device)

        # LLRD config for TUnA
        optimizer_cfg = {
            "type": "adamw",
            "lr": 1e-5,
            "weight_decay": 0.01,
            "param_groups": [
                {
                    "name": "intra",
                    "pattern": "intra_encoder.",
                    "lr": 1e-4,
                    "weight_decay": 0.01,
                },
                {
                    "name": "inter",
                    "pattern": "inter_encoder.",
                    "lr": 2e-4,
                    "weight_decay": 0.01,
                },
                {
                    "name": "head",
                    "pattern": "head.",
                    "lr": 3e-4,
                    "weight_decay": 0.0,
                },
            ],
        }

        trainer = Trainer(
            model=model,
            device=device,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=None,
            amp_cfg={"enabled": False},
        )

        # Should have multiple param groups
        assert len(trainer.optimizer.param_groups) >= 3

        # Verify different learning rates for LLRD
        lrs = [pg["lr"] for pg in trainer.optimizer.param_groups]
        assert 1e-4 in lrs  # intra encoder
        assert 2e-4 in lrs  # inter encoder
        assert 3e-4 in lrs  # head

    def test_llrd_parameter_assignment_no_overlap(self, v3_config, device):
        """Test that parameters are assigned to exactly one group."""
        model = build_model_from_config(v3_config, device)

        optimizer_cfg = {
            "type": "adamw",
            "lr": 1e-5,
            "weight_decay": 0.01,
            "param_groups": [
                {
                    "name": "encoder",
                    "pattern": "encoder.",
                    "lr": 1e-4,
                    "weight_decay": 0.015,
                },
                {
                    "name": "head",
                    "pattern": "mlp_head.",
                    "lr": 3e-4,
                    "weight_decay": 0.0,
                },
            ],
        }

        trainer = Trainer(
            model=model,
            device=device,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=None,
            amp_cfg={"enabled": False},
        )

        # Collect all parameter ids from optimizer groups
        param_ids_in_groups = set()
        for pg in trainer.optimizer.param_groups:
            for param in pg["params"]:
                param_id = id(param)
                # Check no overlap
                assert param_id not in param_ids_in_groups
                param_ids_in_groups.add(param_id)

        # Verify all model params are in optimizer
        model_param_ids = {id(p) for p in model.parameters() if p.requires_grad}
        assert param_ids_in_groups == model_param_ids


class TestFinetuneStrategyIntegration:
    """Integration tests for finetune stage with strategies."""

    def test_staged_unfreeze_with_v3_model(self, v3_config, device):
        """Test StagedUnfreeze strategy with V3 model."""
        from src.train.strategies import StagedUnfreeze

        model = build_model_from_config(v3_config, device)

        # Strategy with schedule
        schedule = [
            {"at_epoch": 0, "freeze": ["encoder."]},
            {"at_epoch": 3, "unfreeze": ["encoder."]},
        ]
        strategy = StagedUnfreeze(schedule=schedule)

        optimizer_cfg = {"type": "adamw", "lr": 1e-3, "weight_decay": 0.01}

        trainer = Trainer(
            model=model,
            device=device,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=None,
            amp_cfg={"enabled": False},
            strategy=strategy,
        )

        # Initial state: all params trainable
        initial_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        assert initial_trainable > 0

        # Simulate epoch 0 end: freeze encoder
        strategy.on_epoch_end(trainer, 0)

        frozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert frozen_trainable < initial_trainable  # Some params frozen

        # Simulate epoch 3 end: unfreeze encoder
        strategy.on_epoch_end(trainer, 3)

        unfrozen_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        assert unfrozen_trainable == initial_trainable  # Back to initial

    def test_staged_unfreeze_with_optimizer_rebuild(self, v3_config, device):
        """Test that optimizer rebuilds correctly during staged unfreeze."""
        from src.train.strategies import StagedUnfreeze

        model = build_model_from_config(v3_config, device)

        # Strategy that changes optimizer config at unfreeze
        schedule = [
            {"at_epoch": 0, "freeze": ["encoder."]},
            {
                "at_epoch": 3,
                "unfreeze": ["encoder."],
                "optimizer_cfg": {
                    "type": "adamw",
                    "lr": 5e-4,  # Lower LR after unfreeze
                    "weight_decay": 0.01,
                },
            },
        ]
        strategy = StagedUnfreeze(schedule=schedule)

        optimizer_cfg = {"type": "adamw", "lr": 1e-3, "weight_decay": 0.01}

        trainer = Trainer(
            model=model,
            device=device,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=None,
            amp_cfg={"enabled": False},
            strategy=strategy,
        )

        # Initial LR
        initial_lr = trainer.optimizer.param_groups[0]["lr"]
        assert initial_lr == 1e-3

        # Freeze at epoch 0
        strategy.on_epoch_end(trainer, 0)

        # Unfreeze with new optimizer at epoch 3
        strategy.on_epoch_end(trainer, 3)

        # LR should have changed
        new_lr = trainer.optimizer.param_groups[0]["lr"]
        assert new_lr == 5e-4

    def test_finetune_with_llrd_and_strategy(self, v3_config, device):
        """Test finetune with both LLRD and staged unfreeze strategy."""
        from src.train.strategies import StagedUnfreeze

        model = build_model_from_config(v3_config, device)

        # Combined: LLRD + staged unfreeze
        schedule = [
            {"at_epoch": 0, "freeze": ["encoder."]},
            {
                "at_epoch": 3,
                "unfreeze": ["encoder."],
                "optimizer_cfg": {
                    "type": "adamw",
                    "lr": 1e-5,  # Base LR
                    "weight_decay": 0.01,
                    "param_groups": [
                        {
                            "name": "encoder",
                            "pattern": "encoder.",
                            "lr": 1e-4,  # Lower for encoder
                            "weight_decay": 0.015,
                        },
                        {
                            "name": "head",
                            "pattern": "mlp_head.",
                            "lr": 3e-4,  # Higher for head
                            "weight_decay": 0.0,
                        },
                    ],
                },
            },
        ]
        strategy = StagedUnfreeze(schedule=schedule)

        # Initial optimizer (head-only)
        optimizer_cfg = {"type": "adamw", "lr": 1e-3, "weight_decay": 0.0}

        trainer = Trainer(
            model=model,
            device=device,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=None,
            amp_cfg={"enabled": False},
            strategy=strategy,
        )

        # Freeze encoder at epoch 0
        strategy.on_epoch_end(trainer, 0)

        # Verify encoder is frozen
        encoder_params = [p for n, p in model.named_parameters() if "encoder." in n]
        assert all(not p.requires_grad for p in encoder_params)

        # Unfreeze and apply LLRD at epoch 3
        strategy.on_epoch_end(trainer, 3)

        # Verify encoder is unfrozen
        assert all(p.requires_grad for p in encoder_params)

        # Verify LLRD is applied (multiple param groups with different LRs)
        lrs = {pg["lr"] for pg in trainer.optimizer.param_groups}
        assert len(lrs) > 1  # Multiple learning rates
        assert 1e-4 in lrs  # Encoder LR
        assert 3e-4 in lrs  # Head LR
