"""
Integration tests for logging functionality in run.py

Tests that run.py correctly logs all required metrics to CSV files
as specified in logging_overview.md.
"""

import csv
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import torch
import torch.nn as nn

from src.stages import run_pretrain, run_finetune
from src.utils.logging import append_row


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_model():
    """Create a simple mock model for testing."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1),
    )
    return model


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader."""
    loader = Mock()
    loader.__len__ = Mock(return_value=10)
    loader.__iter__ = Mock(return_value=iter([]))
    return loader


@pytest.fixture
def pretrain_config():
    """Create a minimal pretrain config."""
    return {
        "pretrain_config": {
            "epochs": 2,
            "batch_size": 16,
            "monitor_metric": "val_auroc",
            "early_stopping_patience": 5,
            "logging_metrics": {
                "primary": "auroc",
                "secondary": "recall",
            },
            "optimizer": {
                "type": "adamw",
                "lr": 0.001,
                "weight_decay": 0.01,
            },
            "scheduler": None,
            "use_mixed_precision": False,
            "max_grad_norm": 1.0,
            "classification_threshold": 0.5,
        },
        "run_config": {
            "save_best_only": False,
        },
        "data_config": {
            "embedding_dtype": "fp32",
        },
    }


@pytest.fixture
def finetune_config():
    """Create a minimal finetune config."""
    return {
        "finetune_config": {
            "epochs": 2,
            "batch_size": 16,
            "monitor_metric": "val_auroc",
            "early_stopping_patience": 5,
            "logging_metrics": {
                "primary": "auroc",
                "secondary": "recall",
            },
            "optimizer": {
                "type": "adamw",
                "lr": 0.0001,
                "weight_decay": 0.01,
            },
            "scheduler": None,
            "use_mixed_precision": False,
            "max_grad_norm": 1.0,
            "classification_threshold": 0.5,
            "strategy": {
                "type": "staged_unfreeze",
                "schedule": [],
            },
        },
        "run_config": {
            "save_best_only": False,
        },
        "data_config": {
            "embedding_dtype": "fp32",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRunPretrainLogging:
    """Test that run_pretrain correctly logs all required metrics."""

    def test_pretrain_csv_has_correct_schema(
        self, mock_model, mock_dataloader, pretrain_config, tmp_path
    ):
        """Test that pretrain creates CSV with correct columns from logging_overview.md."""
        log_dir = tmp_path / "logs"
        checkpoint_dir = tmp_path / "checkpoints"
        device = torch.device("cpu")

        # Mock trainer to return proper metrics
        mock_trainer = Mock()
        mock_trainer.train_one_epoch_iter = Mock(
            return_value=[
                {"batch_idx": 0, "loss": 0.5, "lr": 0.001},
                {
                    "_epoch_end": True,
                    "loss": 0.5,
                    "lr": 0.001,
                    "auroc": 0.75,
                    "recall": 0.70,
                },
            ]
        )
        mock_trainer.optimizer = Mock()
        mock_trainer.optimizer.state_dict = Mock(return_value={})

        # Mock evaluator to return proper metrics
        mock_evaluator = Mock()
        mock_evaluator.evaluate = Mock(
            return_value=[
                {"loss": 0.45, "auroc": 0.78, "recall": 0.73, "_evaluation_end": True}
            ]
        )

        with (
            patch("src.stages.pretrain.Trainer", return_value=mock_trainer),
            patch("src.stages.pretrain.Evaluator", return_value=mock_evaluator),
            patch("src.stages.pretrain.maybe_save_best", return_value=(True, 0.78)),
            patch("src.stages.pretrain.save_checkpoint", return_value="checkpoint.pth"),
            patch("src.stages.pretrain.check_early_stop", return_value=True),
        ):  # Stop after 1 epoch
            run_pretrain(
                cfg=pretrain_config,
                model=mock_model,
                train_loader=mock_dataloader,
                val_loader=mock_dataloader,
                device=device,
                pretrain_run_id="test_run",
                log_dir=log_dir,
                checkpoint_dir=checkpoint_dir,
            )

        # Check that CSV was created
        csv_path = log_dir / "training_step.csv"
        assert csv_path.exists(), "training_step.csv should be created"

        # Read CSV and verify schema
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            data_rows = list(reader)

        # Expected columns from logging_overview.md line 23
        expected_columns = [
            "Epoch",
            "Epoch Time",
            "Train Loss",
            "Val Loss",
            "Val auroc",
            "Val recall",
            "Learning Rate",
        ]

        assert header == expected_columns, (
            f"CSV header mismatch. Expected {expected_columns}, got {header}"
        )
        assert len(data_rows) >= 1, "Should have at least one data row"

        # Verify first data row has correct number of columns
        assert len(data_rows[0]) == len(expected_columns), (
            "Data row should match header length"
        )

    def test_pretrain_logs_all_metrics(
        self, mock_model, mock_dataloader, pretrain_config, tmp_path
    ):
        """Test that all metrics are actually logged to CSV."""
        log_dir = tmp_path / "logs"
        checkpoint_dir = tmp_path / "checkpoints"
        device = torch.device("cpu")

        # Mock components
        mock_trainer = Mock()
        mock_trainer.train_one_epoch_iter = Mock(
            return_value=[
                {"batch_idx": 0, "loss": 0.6543, "lr": 0.0003},
                {
                    "_epoch_end": True,
                    "loss": 0.6543,
                    "lr": 0.0003,
                    "auroc": 0.7234,
                    "recall": 0.6891,
                },
            ]
        )
        mock_trainer.optimizer = Mock()
        mock_trainer.optimizer.state_dict = Mock(return_value={})

        mock_evaluator = Mock()
        mock_evaluator.evaluate = Mock(
            return_value=[
                {
                    "loss": 0.6201,
                    "auroc": 0.7456,
                    "recall": 0.7012,
                    "_evaluation_end": True,
                }
            ]
        )

        with (
            patch("src.stages.pretrain.Trainer", return_value=mock_trainer),
            patch("src.stages.pretrain.Evaluator", return_value=mock_evaluator),
            patch("src.stages.pretrain.maybe_save_best", return_value=(True, 0.7456)),
            patch("src.stages.pretrain.save_checkpoint", return_value="checkpoint.pth"),
            patch("src.stages.pretrain.check_early_stop", return_value=True),
        ):  # Stop after 1 epoch
            run_pretrain(
                cfg=pretrain_config,
                model=mock_model,
                train_loader=mock_dataloader,
                val_loader=mock_dataloader,
                device=device,
                pretrain_run_id="test_run",
                log_dir=log_dir,
                checkpoint_dir=checkpoint_dir,
            )

        # Read CSV and verify data
        csv_path = log_dir / "training_step.csv"
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) >= 1, "Should have logged at least one epoch"

        first_row = rows[0]

        # Check all required fields are present and non-empty
        assert first_row["Epoch"] == "0", "Epoch should be 0 for first epoch"
        assert float(first_row["Epoch Time"]) > 0, "Epoch time should be positive"
        assert first_row["Train Loss"] == "0.6543", "Train loss mismatch"
        assert first_row["Val Loss"] == "0.6201", "Val loss mismatch"

        # Check that trainer metrics were logged (may be 0.0 if not in trainer return)
        # assert "Train auroc" in first_row, "Train auroc column should exist"
        # assert "Train recall" in first_row, "Train recall column should exist"

        # Check validation metrics
        assert first_row["Val auroc"] == "0.7456", "Val auroc mismatch"
        assert first_row["Val recall"] == "0.7012", "Val recall mismatch"
        assert first_row["Learning Rate"] == "0.0003", "Learning rate mismatch"


class TestRunFinetuneLogging:
    """Test that run_finetune correctly logs all required metrics."""

    def test_finetune_csv_has_correct_schema(
        self, mock_model, mock_dataloader, finetune_config, tmp_path
    ):
        """Test that finetune creates CSV with correct columns."""
        log_dir = tmp_path / "logs"
        checkpoint_dir = tmp_path / "checkpoints"
        device = torch.device("cpu")

        # Create a dummy checkpoint to load (using checkpoint.py format)
        checkpoint_path = tmp_path / "pretrain_checkpoint.pth"
        torch.save(
            {"state_dict": mock_model.state_dict(), "epoch": 10}, checkpoint_path
        )

        # Mock components
        mock_trainer = Mock()
        mock_trainer.train_one_epoch_iter = Mock(
            return_value=[
                {"batch_idx": 0, "loss": 0.3, "lr": 0.0001},
                {
                    "_epoch_end": True,
                    "loss": 0.3,
                    "lr": 0.0001,
                    "auroc": 0.85,
                    "recall": 0.82,
                },
            ]
        )
        mock_trainer.optimizer = Mock()
        mock_trainer.optimizer.state_dict = Mock(return_value={})

        mock_evaluator = Mock()
        mock_evaluator.evaluate = Mock(
            return_value=[
                {"loss": 0.28, "auroc": 0.87, "recall": 0.84, "_evaluation_end": True}
            ]
        )

        with (
            patch("src.stages.finetune.Trainer", return_value=mock_trainer),
            patch("src.stages.finetune.Evaluator", return_value=mock_evaluator),
            patch("src.stages.finetune.maybe_save_best", return_value=(True, 0.87)),
            patch("src.stages.finetune.save_checkpoint", return_value="checkpoint.pth"),
            patch("src.stages.finetune.check_early_stop", return_value=True),
        ):  # Stop after 1 epoch
            run_finetune(
                cfg=finetune_config,
                model=mock_model,
                train_loader=mock_dataloader,
                val_loader=mock_dataloader,
                device=device,
                finetune_run_id="test_run",
                log_dir=log_dir,
                checkpoint_dir=checkpoint_dir,
                load_checkpoint_path=str(checkpoint_path),
            )

        # Check that CSV was created
        csv_path = log_dir / "training_step.csv"
        assert csv_path.exists(), "training_step.csv should be created"

        # Read CSV and verify schema
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            data_rows = list(reader)

        # Expected columns (same as pretrain)
        expected_columns = [
            "Epoch",
            "Epoch Time",
            "Train Loss",
            "Val Loss",
            "Val auroc",
            "Val recall",
            "Learning Rate",
        ]

        assert header == expected_columns, (
            f"CSV header mismatch. Expected {expected_columns}, got {header}"
        )
        assert len(data_rows) >= 1, "Should have at least one data row"


class TestAppendRowIntegration:
    """Test append_row in a realistic scenario."""

    def test_append_row_matches_run_py_usage(self, tmp_path):
        """Test that append_row works exactly as used in run.py."""
        log_dir = tmp_path / "logs" / "v3" / "pretrain" / "test_run"
        csv_path = log_dir / "training_step.csv"

        # Simulate run.py usage pattern
        primary = "auroc"
        secondary = "recall"
        columns = [
            "Epoch",
            "Epoch Time",
            "Train Loss",
            "Val Loss",
            f"Train {primary}",
            f"Train {secondary}",
            f"Val {primary}",
            f"Val {secondary}",
            "Learning Rate",
        ]

        # Simulate logging multiple epochs
        for epoch in range(3):
            train_metrics = {
                "loss": 0.5 - epoch * 0.1,
                "lr": 0.001,
                primary: 0.7 + epoch * 0.05,
                secondary: 0.65 + epoch * 0.05,
            }
            val_metrics = {
                "loss": 0.48 - epoch * 0.09,
                primary: 0.72 + epoch * 0.05,
                secondary: 0.68 + epoch * 0.05,
            }
            epoch_time = 120.0 + epoch * 2

            row = {
                "Epoch": epoch,
                "Epoch Time": epoch_time,
                "Train Loss": train_metrics["loss"],
                "Val Loss": val_metrics["loss"],
                f"Train {primary}": train_metrics.get(primary, 0.0),
                f"Train {secondary}": train_metrics.get(secondary, 0.0),
                f"Val {primary}": val_metrics.get(primary, 0.0),
                f"Val {secondary}": val_metrics.get(secondary, 0.0),
                "Learning Rate": train_metrics["lr"],
            }
            append_row(csv_path, row, columns)

        # Verify the results
        assert csv_path.exists()

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3, "Should have 3 epochs"

        # Check first epoch
        assert rows[0]["Epoch"] == "0"
        assert "Train auroc" in rows[0]
        assert "Val auroc" in rows[0]
        assert float(rows[0]["Train auroc"]) == pytest.approx(0.7)
        assert float(rows[0]["Val auroc"]) == pytest.approx(0.72)

        # Check last epoch
        assert rows[2]["Epoch"] == "2"
        assert float(rows[2]["Train auroc"]) == pytest.approx(0.8)
        assert float(rows[2]["Val auroc"]) == pytest.approx(0.82)
