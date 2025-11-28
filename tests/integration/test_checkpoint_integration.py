"""
Integration tests for checkpoint module.

Tests full checkpoint save/load workflow with real models, optimizers,
and integration with run.py orchestration.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from src.utils.checkpoint import save_checkpoint, maybe_save_best, load_checkpoint
from src.utils.config import load_config, extract_keys


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def v3_config():
    """Load v3.yaml config."""
    config_path = "configs/v3.yaml"
    if not Path(config_path).exists():
        pytest.skip(f"Config file not found: {config_path}")
    return load_config(config_path)


@pytest.fixture
def device():
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for checkpoint tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


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


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckpointWithRealModels:
    """Test checkpoint save/load with real V3 and TUnA models."""

    @patch("src.utils.checkpoint._is_rank_zero", return_value=True)
    def test_save_and_load_v3_model(self, mock_rank_zero, v3_config, device, temp_dir):
        """Test full save/load cycle with V3 model."""
        # Build model
        model = build_model_from_config(v3_config, device)

        # Save initial weights for comparison
        initial_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Save checkpoint
        ckpt_path = save_checkpoint(
            model=model,
            epoch=5,
            path=str(temp_dir),
            extra={"test_metric": 0.85},
        )

        assert ckpt_path is not None
        assert Path(ckpt_path).exists()

        # Modify model weights
        for param in model.parameters():
            param.data.fill_(999.0)

        # Verify weights changed
        changed_state = model.state_dict()
        for key in initial_state:
            assert not torch.allclose(initial_state[key], changed_state[key])

        # Load checkpoint
        metadata = load_checkpoint(model, ckpt_path, map_location=device)

        # Verify weights restored
        restored_state = model.state_dict()
        for key in initial_state:
            assert torch.allclose(initial_state[key], restored_state[key])

        # Verify metadata
        assert metadata["epoch"] == 5
        assert metadata["extra"]["test_metric"] == 0.85

    @patch("src.utils.checkpoint._is_rank_zero", return_value=True)
    def test_save_and_load_with_optimizer(
        self, mock_rank_zero, v3_config, device, temp_dir
    ):
        """Test checkpoint save/load with optimizer state."""
        model = build_model_from_config(v3_config, device)

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

        # Do a dummy training step to populate optimizer state
        dummy_input = {
            "emb_a": torch.randn(2, 10, 1536, device=device),
            "len_a": torch.tensor([10, 10], device=device),
            "emb_b": torch.randn(2, 10, 1536, device=device),
            "len_b": torch.tensor([10, 10], device=device),
            "label": torch.randint(0, 2, (2,), device=device).float(),
        }
        output = model(dummy_input)
        loss = output["logits"].sum()
        loss.backward()
        optimizer.step()

        # Save optimizer state
        initial_optim_state = optimizer.state_dict()

        # Save checkpoint with optimizer
        ckpt_path = save_checkpoint(
            model=model,
            epoch=3,
            path=str(temp_dir),
            include_optim=True,
            optimizer=optimizer,
        )

        # Create new optimizer and model
        new_model = build_model_from_config(v3_config, device)
        new_optimizer = torch.optim.AdamW(
            new_model.parameters(), lr=0.999, weight_decay=0.999
        )

        # Load checkpoint with optimizer
        load_checkpoint(
            new_model,
            ckpt_path,
            map_location=device,
            optimizer=new_optimizer,
            load_optim=True,
        )

        # Verify optimizer state loaded (check LR was restored)
        loaded_lr = new_optimizer.param_groups[0]["lr"]
        original_lr = initial_optim_state["param_groups"][0]["lr"]
        assert loaded_lr == original_lr  # Should match original, not 0.999

    @patch("src.utils.checkpoint._is_rank_zero", return_value=True)
    def test_maybe_save_best_workflow(
        self, mock_rank_zero, v3_config, device, temp_dir
    ):
        """Test best checkpoint tracking across multiple epochs."""
        model = build_model_from_config(v3_config, device)
        best_path = temp_dir / "best_model.pth"

        # Epoch 0: val_loss=0.5 (initial best)
        improved, best_metric = maybe_save_best(
            model=model,
            epoch=0,
            current_metric=0.5,
            best_so_far=float("inf"),
            mode="min",
            best_path=str(best_path),
        )
        assert improved is True
        assert best_metric == 0.5
        assert best_path.exists()

        ckpt = torch.load(best_path, weights_only=False)
        assert ckpt["epoch"] == 0

        # Epoch 1: val_loss=0.6 (worse, don't save)
        improved, best_metric = maybe_save_best(
            model=model,
            epoch=1,
            current_metric=0.6,
            best_so_far=best_metric,
            mode="min",
            best_path=str(best_path),
        )
        assert improved is False
        assert best_metric == 0.5  # Unchanged

        ckpt = torch.load(best_path, weights_only=False)
        assert ckpt["epoch"] == 0  # Still epoch 0

        # Epoch 2: val_loss=0.3 (better, save)
        improved, best_metric = maybe_save_best(
            model=model,
            epoch=2,
            current_metric=0.3,
            best_so_far=best_metric,
            mode="min",
            best_path=str(best_path),
        )
        assert improved is True
        assert best_metric == 0.3

        ckpt = torch.load(best_path, weights_only=False)
        assert ckpt["epoch"] == 2  # Updated to epoch 2

        # Epoch 3: val_loss=0.3 (equal, don't save)
        improved, best_metric = maybe_save_best(
            model=model,
            epoch=3,
            current_metric=0.3,
            best_so_far=best_metric,
            mode="min",
            best_path=str(best_path),
        )
        assert improved is False
        assert best_metric == 0.3

        ckpt = torch.load(best_path, weights_only=False)
        assert ckpt["epoch"] == 2  # Still epoch 2

    @patch("src.utils.checkpoint._is_rank_zero", return_value=True)
    def test_mode_max_with_auroc_tracking(
        self, mock_rank_zero, v3_config, device, temp_dir
    ):
        """Test best checkpoint tracking with mode='max' (e.g., AUROC)."""
        model = build_model_from_config(v3_config, device)
        best_path = temp_dir / "best_auroc.pth"

        # Simulate AUROC improvements (higher is better)
        epochs_metrics = [
            (0, 0.65),  # Initial
            (1, 0.70),  # Improved
            (2, 0.68),  # Worse
            (3, 0.75),  # Improved
            (4, 0.74),  # Worse
        ]

        best_so_far = float("-inf")
        saved_epochs = []

        for epoch, metric in epochs_metrics:
            improved, best_so_far = maybe_save_best(
                model=model,
                epoch=epoch,
                current_metric=metric,
                best_so_far=best_so_far,
                mode="max",
                best_path=str(best_path),
            )
            if improved:
                saved_epochs.append(epoch)

        # Should have saved at epochs 0, 1, 3 (improvements)
        assert saved_epochs == [0, 1, 3]

        # Final best should be epoch 3 with AUROC 0.75
        ckpt = torch.load(best_path, weights_only=False)
        assert ckpt["epoch"] == 3
        assert best_so_far == 0.75


class TestCheckpointWithTrainer:
    """Test checkpoint integration with Trainer class."""

    @patch("src.utils.checkpoint._is_rank_zero", return_value=True)
    def test_checkpoint_with_trainer_optimizer(
        self, mock_rank_zero, v3_config, device, temp_dir
    ):
        """Test saving checkpoint with Trainer's optimizer."""
        from src.train.base import Trainer

        model = build_model_from_config(v3_config, device)
        pretrain_cfg = v3_config["pretrain_config"]

        # Prepare scheduler config
        scheduler_cfg = pretrain_cfg.get("scheduler")
        if scheduler_cfg and scheduler_cfg.get("type", "").lower() in [
            "onecycle",
            "onecyclelr",
        ]:
            scheduler_cfg = scheduler_cfg.copy()
            scheduler_cfg["total_steps"] = 100

        # Create trainer
        trainer = Trainer(
            model=model,
            device=device,
            optimizer_cfg=pretrain_cfg["optimizer"],
            scheduler_cfg=scheduler_cfg,
            amp_cfg={"enabled": False},
        )

        # Save checkpoint with trainer's optimizer
        ckpt_path = save_checkpoint(
            model=model,
            epoch=7,
            path=str(temp_dir),
            include_optim=True,
            optimizer=trainer.optimizer,
        )

        assert ckpt_path is not None

        # Load into new model+trainer
        new_model = build_model_from_config(v3_config, device)
        new_trainer = Trainer(
            model=new_model,
            device=device,
            optimizer_cfg=pretrain_cfg["optimizer"],
            scheduler_cfg=scheduler_cfg,
            amp_cfg={"enabled": False},
        )

        # Load checkpoint
        metadata = load_checkpoint(
            new_model,
            ckpt_path,
            map_location=device,
            optimizer=new_trainer.optimizer,
            load_optim=True,
        )

        assert metadata["epoch"] == 7

        # Optimizer states should match
        orig_lr = trainer.optimizer.param_groups[0]["lr"]
        loaded_lr = new_trainer.optimizer.param_groups[0]["lr"]
        assert orig_lr == loaded_lr

    @patch("src.utils.checkpoint._is_rank_zero", return_value=True)
    def test_finetune_from_pretrain_checkpoint(
        self, mock_rank_zero, v3_config, device, temp_dir
    ):
        """Test loading pretrain checkpoint for finetune (no optimizer state)."""
        from src.train.base import Trainer

        model = build_model_from_config(v3_config, device)
        pretrain_cfg = v3_config["pretrain_config"]

        # Create pretrain trainer
        pretrain_trainer = Trainer(
            model=model,
            device=device,
            optimizer_cfg=pretrain_cfg["optimizer"],
            scheduler_cfg=None,
            amp_cfg={"enabled": False},
        )

        # Save pretrain checkpoint
        pretrain_ckpt = save_checkpoint(
            model=model,
            epoch=10,
            path=str(temp_dir / "pretrain"),
            include_optim=True,
            optimizer=pretrain_trainer.optimizer,
            extra={"stage": "pretrain", "val_loss": 0.25},
        )

        # Create new model for finetune
        finetune_model = build_model_from_config(v3_config, device)
        finetune_cfg = v3_config["finetune_config"]

        # Load pretrain checkpoint (weights only, not optimizer)
        metadata = load_checkpoint(
            finetune_model,
            pretrain_ckpt,
            map_location=device,
            load_optim=False,
        )

        assert metadata["epoch"] == 10
        assert metadata["extra"]["stage"] == "pretrain"

        # Create finetune trainer with fresh optimizer
        finetune_trainer = Trainer(
            model=finetune_model,
            device=device,
            optimizer_cfg=finetune_cfg["optimizer"],
            scheduler_cfg=None,
            amp_cfg={"enabled": False},
        )

        # Finetune optimizer should have its own LR (not pretrain's)
        finetune_lr = finetune_trainer.optimizer.param_groups[0]["lr"]
        pretrain_lr = pretrain_trainer.optimizer.param_groups[0]["lr"]
        # They may differ if configs differ
        assert finetune_lr is not None
        assert pretrain_lr is not None


class TestCheckpointRunPyIntegration:
    """Test checkpoint integration patterns used in run.py."""

    @patch("src.utils.checkpoint._is_rank_zero", return_value=True)
    def test_pretrain_checkpoint_pattern(
        self, mock_rank_zero, v3_config, device, temp_dir
    ):
        """Test the checkpoint pattern used in run_pretrain()."""
        from src.train.base import Trainer

        model = build_model_from_config(v3_config, device)
        pretrain_cfg = v3_config["pretrain_config"]

        trainer = Trainer(
            model=model,
            device=device,
            optimizer_cfg=pretrain_cfg["optimizer"],
            scheduler_cfg=None,
            amp_cfg={"enabled": False},
        )

        checkpoint_dir = temp_dir / "checkpoints"
        monitor_metric = "loss"  # Use "loss" not "val_loss" (matches dict key)
        save_best_only = False

        # Simulate 3 epochs
        best_metric = float("inf")

        for epoch in range(3):
            val_metrics = {"loss": 0.5 - epoch * 0.1, "auroc": 0.7 + epoch * 0.05}
            current_metric = val_metrics[monitor_metric]
            mode = "min"

            # Save best checkpoint (run.py pattern)
            improved, best_metric = maybe_save_best(
                model=model,
                epoch=epoch,
                current_metric=current_metric,
                best_so_far=best_metric,
                mode=mode,
                best_path=str(checkpoint_dir / "best_model.pth"),
                include_optim=True,
                optimizer=trainer.optimizer,
                extra={"monitor_metric": monitor_metric, "val_metrics": val_metrics},
            )

            # Save per-epoch checkpoint if not save_best_only
            if not save_best_only:
                saved_path = save_checkpoint(
                    model=model,
                    epoch=epoch,
                    path=str(checkpoint_dir),
                    include_optim=True,
                    optimizer=trainer.optimizer,
                    extra={"val_metrics": val_metrics},
                )
                assert saved_path is not None

        # Verify all checkpoints exist
        assert (checkpoint_dir / "best_model.pth").exists()
        assert (checkpoint_dir / "epoch_0000.pth").exists()
        assert (checkpoint_dir / "epoch_0001.pth").exists()
        assert (checkpoint_dir / "epoch_0002.pth").exists()

        # Best checkpoint should be epoch 2 (lowest loss)
        best_ckpt = torch.load(checkpoint_dir / "best_model.pth", weights_only=False)
        assert best_ckpt["epoch"] == 2
        assert best_ckpt["extra"]["val_metrics"]["loss"] == 0.3

    @patch("src.utils.checkpoint._is_rank_zero", return_value=True)
    def test_finetune_checkpoint_pattern(
        self, mock_rank_zero, v3_config, device, temp_dir
    ):
        """Test the checkpoint pattern used in run_finetune()."""
        from src.train.base import Trainer

        model = build_model_from_config(v3_config, device)
        finetune_cfg = v3_config["finetune_config"]

        # Simulate loading pretrain checkpoint first
        pretrain_dir = temp_dir / "pretrain"
        pretrain_ckpt_path = pretrain_dir / "best_model.pth"
        pretrain_dir.mkdir(parents=True, exist_ok=True)

        # Save a mock pretrain checkpoint
        torch.save(
            {"epoch": 20, "state_dict": model.state_dict()},
            pretrain_ckpt_path,
        )

        # Load checkpoint (run.py pattern for finetune)
        ckpt_metadata = load_checkpoint(
            model=model,
            ckpt_path=str(pretrain_ckpt_path),
            map_location=device,
            strict=True,
            load_optim=False,
        )

        assert ckpt_metadata["epoch"] == 20

        # Create finetune trainer
        trainer = Trainer(
            model=model,
            device=device,
            optimizer_cfg=finetune_cfg["optimizer"],
            scheduler_cfg=None,
            amp_cfg={"enabled": False},
        )

        # Finetune checkpoint directory
        finetune_dir = temp_dir / "finetune"
        monitor_metric = "auroc"  # Use "auroc" not "val_auroc" (matches dict key)

        # Simulate finetune epochs (mode='max' for AUROC)
        best_metric = float("-inf")

        for epoch in range(2):
            val_metrics = {"loss": 0.2, "auroc": 0.8 + epoch * 0.05}
            current_metric = val_metrics.get(monitor_metric)
            mode = "max"

            improved, best_metric = maybe_save_best(
                model=model,
                epoch=epoch,
                current_metric=current_metric,
                best_so_far=best_metric,
                mode=mode,
                best_path=str(finetune_dir / "best_model.pth"),
                include_optim=True,
                optimizer=trainer.optimizer,
                extra={"monitor_metric": monitor_metric, "val_metrics": val_metrics},
            )

        # Verify finetune best checkpoint
        assert (finetune_dir / "best_model.pth").exists()
        best_ckpt = torch.load(finetune_dir / "best_model.pth", weights_only=False)
        assert best_ckpt["epoch"] == 1  # Epoch 1 had highest AUROC

    @patch("src.utils.checkpoint._is_rank_zero", return_value=True)
    def test_eval_only_checkpoint_loading(
        self, mock_rank_zero, v3_config, device, temp_dir
    ):
        """Test checkpoint loading pattern for eval_only mode."""
        model = build_model_from_config(v3_config, device)

        # Save a mock checkpoint
        ckpt_path = temp_dir / "model_to_eval.pth"
        torch.save(
            {
                "epoch": 15,
                "state_dict": model.state_dict(),
                "extra": {"stage": "finetune", "val_auroc": 0.92},
            },
            ckpt_path,
        )

        # Create new model for evaluation
        eval_model = build_model_from_config(v3_config, device)

        # Load checkpoint (run.py eval_only pattern)
        ckpt_metadata = load_checkpoint(
            model=eval_model,
            ckpt_path=str(ckpt_path),
            map_location=device,
            strict=True,
            load_optim=False,
        )

        assert ckpt_metadata["epoch"] == 15
        assert ckpt_metadata["extra"]["val_auroc"] == 0.92

        # Verify model is in eval mode for inference
        eval_model.eval()
        assert not eval_model.training


class TestCheckpointDDPScenarios:
    """Test checkpoint behavior in DDP scenarios."""

    @patch("src.utils.checkpoint._is_rank_zero", return_value=True)
    def test_save_checkpoint_rank_zero(
        self, mock_rank_zero, v3_config, device, temp_dir
    ):
        """Test that rank 0 saves checkpoints."""
        model = build_model_from_config(v3_config, device)

        ckpt_path = save_checkpoint(model, epoch=1, path=str(temp_dir))

        assert ckpt_path is not None
        assert Path(ckpt_path).exists()

    @patch("src.utils.checkpoint._is_rank_zero", return_value=False)
    def test_save_checkpoint_non_rank_zero(
        self, mock_rank_zero, v3_config, device, temp_dir
    ):
        """Test that non-rank-0 processes don't save checkpoints."""
        model = build_model_from_config(v3_config, device)

        ckpt_path = save_checkpoint(model, epoch=1, path=str(temp_dir))

        assert ckpt_path is None
        # No files should be created
        assert len(list(temp_dir.glob("*.pth"))) == 0

    @patch("src.utils.checkpoint._is_rank_zero", return_value=False)
    def test_maybe_save_best_non_rank_zero(
        self, mock_rank_zero, v3_config, device, temp_dir
    ):
        """Test that non-rank-0 processes don't save best checkpoints."""
        model = build_model_from_config(v3_config, device)

        improved, new_best = maybe_save_best(
            model=model,
            epoch=1,
            current_metric=0.9,
            best_so_far=0.8,
            mode="max",
            best_path=str(temp_dir / "best.pth"),
        )

        assert improved is False
        assert new_best == 0.8
        assert not (temp_dir / "best.pth").exists()

    def test_load_checkpoint_all_ranks(self, v3_config, device, temp_dir):
        """Test that all ranks can load checkpoints."""
        model = build_model_from_config(v3_config, device)

        # Create a checkpoint (simulating rank 0 save)
        ckpt_path = temp_dir / "model.pth"
        torch.save({"epoch": 5, "state_dict": model.state_dict()}, ckpt_path)

        # Simulate loading on any rank (no rank check in load)
        new_model = build_model_from_config(v3_config, device)
        metadata = load_checkpoint(new_model, str(ckpt_path), map_location=device)

        assert metadata["epoch"] == 5
        # All ranks should successfully load


class TestCheckpointEdgeCases:
    """Test edge cases and error handling."""

    @patch("src.utils.checkpoint._is_rank_zero", return_value=True)
    def test_checkpoint_with_empty_extra(
        self, mock_rank_zero, v3_config, device, temp_dir
    ):
        """Test checkpoint with empty extra dict."""
        model = build_model_from_config(v3_config, device)

        ckpt_path = save_checkpoint(model, epoch=1, path=str(temp_dir), extra={})

        ckpt = torch.load(ckpt_path, weights_only=False)
        assert "extra" in ckpt
        assert ckpt["extra"] == {}

    @patch("src.utils.checkpoint._is_rank_zero", return_value=True)
    def test_checkpoint_with_none_extra(
        self, mock_rank_zero, v3_config, device, temp_dir
    ):
        """Test checkpoint with None extra."""
        model = build_model_from_config(v3_config, device)

        ckpt_path = save_checkpoint(model, epoch=1, path=str(temp_dir), extra=None)

        ckpt = torch.load(ckpt_path, weights_only=False)
        assert "extra" not in ckpt

    def test_load_checkpoint_missing_epoch(self, v3_config, device, temp_dir):
        """Test loading checkpoint without epoch field."""
        model = build_model_from_config(v3_config, device)

        # Save checkpoint without epoch
        ckpt_path = temp_dir / "no_epoch.pth"
        torch.save({"state_dict": model.state_dict()}, ckpt_path)

        # Load should handle gracefully (default to 0)
        metadata = load_checkpoint(model, str(ckpt_path), map_location=device)
        assert metadata["epoch"] == 0

    @patch("src.utils.checkpoint._is_rank_zero", return_value=True)
    def test_save_checkpoint_deeply_nested_path(
        self, mock_rank_zero, v3_config, device, temp_dir
    ):
        """Test saving checkpoint to deeply nested directory."""
        model = build_model_from_config(v3_config, device)

        deep_path = temp_dir / "a" / "b" / "c" / "d" / "e"
        ckpt_path = save_checkpoint(model, epoch=1, path=str(deep_path))

        assert ckpt_path is not None
        assert Path(ckpt_path).exists()
        assert deep_path.exists()
