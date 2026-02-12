"""Unit tests for generic trainer and evaluator."""

from __future__ import annotations

import logging

import pytest
import src.run as run_module
import torch
import torch.nn.functional as functional
from src.evaluate import Evaluator
from src.train.base import OptimizerConfig, SchedulerConfig, Trainer
from src.utils.losses import LossConfig
from src.utils.ohem_sample_strategy import OHEMSampleStrategy
from torch import nn
from torch.utils.data import DataLoader, Dataset


class TinyDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic toy dataset for binary classification."""

    def __init__(self) -> None:
        self._features = torch.tensor(
            [
                [0.0, 1.0, 0.5, 0.1],
                [1.0, 0.0, 0.2, 0.3],
                [0.9, 0.8, 0.1, 0.2],
                [0.1, 0.2, 0.9, 0.7],
            ],
            dtype=torch.float32,
        )
        self._labels = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)

    def __len__(self) -> int:
        """Return dataset size."""
        return int(self._features.size(0))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one dataset example."""
        feature = self._features[index]
        label = self._labels[index]
        return {"x": feature, "label": label}


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "x": torch.stack([item["x"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
    }


class TinyModel(nn.Module):
    """Minimal model implementing required forward output contract."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.linear(x).squeeze(-1)
        per_sample_loss = functional.binary_cross_entropy_with_logits(
            logits,
            label,
            reduction="none",
        )
        loss = per_sample_loss.mean()
        return {
            "logits": logits.unsqueeze(-1),
            "loss": loss,
            "per_sample_loss": per_sample_loss,
        }


def test_trainer_runs_single_epoch() -> None:
    model = TinyModel()
    loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False, collate_fn=_collate)
    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        optimizer_config=OptimizerConfig(optimizer_type="adamw", lr=1e-2),
        scheduler_config=SchedulerConfig(scheduler_type="none"),
        loss_config=LossConfig(loss_type="bce_with_logits", pos_weight=1.0, label_smoothing=0.0),
        use_amp=False,
        total_epochs=1,
        steps_per_epoch=len(loader),
        ohem_strategy=OHEMSampleStrategy(target_batch_size=1, cap_protein=4),
    )
    metrics = trainer.train_one_epoch(loader)
    assert "loss" in metrics
    assert "lr" in metrics
    assert metrics["loss"] >= 0.0


def test_trainer_heartbeat_logging(caplog: pytest.LogCaptureFixture) -> None:
    logger_name = "tests.trainer.heartbeat"
    trainer_logger = logging.getLogger(logger_name)
    trainer_logger.handlers.clear()
    trainer_logger.propagate = True
    trainer_logger.setLevel(logging.INFO)

    model = TinyModel()
    loader = DataLoader(TinyDataset(), batch_size=1, shuffle=False, collate_fn=_collate)
    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        optimizer_config=OptimizerConfig(optimizer_type="adamw", lr=1e-2),
        scheduler_config=SchedulerConfig(scheduler_type="none"),
        loss_config=LossConfig(loss_type="bce_with_logits", pos_weight=1.0, label_smoothing=0.0),
        use_amp=False,
        total_epochs=1,
        steps_per_epoch=len(loader),
        logger=trainer_logger,
        heartbeat_every_n_steps=2,
    )

    with caplog.at_level(logging.INFO, logger=logger_name):
        trainer.train_one_epoch(loader, epoch_index=0)

    messages = [record.getMessage() for record in caplog.records if record.name == logger_name]
    assert any("Epoch 1 | Step 1/4" in message for message in messages)
    assert any("Epoch 1 | Step 2/4" in message for message in messages)
    assert any("Epoch 1 | Step 4/4" in message for message in messages)
    assert not any("Epoch 1 | Step 3/4" in message for message in messages)


def test_ohem_disables_pos_weight_for_selected_batch_loss() -> None:
    model = TinyModel()
    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        optimizer_config=OptimizerConfig(optimizer_type="adamw", lr=1e-2),
        scheduler_config=SchedulerConfig(scheduler_type="none"),
        loss_config=LossConfig(loss_type="bce_with_logits", pos_weight=5.0, label_smoothing=0.0),
        use_amp=False,
        total_epochs=1,
        steps_per_epoch=1,
        ohem_strategy=OHEMSampleStrategy(target_batch_size=2, cap_protein=4, warmup_epochs=0),
    )
    logits = torch.tensor([[0.0], [1.0], [-0.5]], dtype=torch.float32)
    labels = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    batch = {
        "label": labels,
        "protein_a_id": torch.tensor([0, 1, 2], dtype=torch.long),
        "protein_b_id": torch.tensor([3, 4, 5], dtype=torch.long),
    }
    loss = trainer._select_loss(output={"logits": logits}, batch=batch, epoch_index=0)
    per_sample_unweighted = functional.binary_cross_entropy_with_logits(
        logits.squeeze(-1),
        labels,
        reduction="none",
    )
    expected = per_sample_unweighted[torch.tensor([1, 2])].mean()
    assert torch.isclose(loss, expected, atol=1e-6)


def test_training_csv_schema_header_order_regression() -> None:
    training_cfg = {
        "logging": {
            "validation_metrics": ["auprc", "auroc"],
        }
    }
    validation_metrics = run_module._training_validation_metrics(training_cfg)
    header = [
        "Epoch",
        "Epoch Time",
        "Train Loss",
        "Val Loss",
        *[f"Val {metric}" for metric in validation_metrics],
        "Learning Rate",
    ]
    assert header == [
        "Epoch",
        "Epoch Time",
        "Train Loss",
        "Val Loss",
        "Val auprc",
        "Val auroc",
        "Learning Rate",
    ]


def test_evaluate_csv_schema_header_order_and_required_columns() -> None:
    assert run_module.EVAL_CSV_COLUMNS == [
        "split",
        "auroc",
        "auprc",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "recall",
        "f1",
        "mcc",
    ]
    required_columns = {
        "split",
        "auroc",
        "auprc",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "recall",
        "f1",
        "mcc",
    }
    assert required_columns.issubset(set(run_module.EVAL_CSV_COLUMNS))
    assert len(run_module.EVAL_CSV_COLUMNS) == len(required_columns)


def test_evaluator_returns_metric_dictionary() -> None:
    model = TinyModel()
    loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False, collate_fn=_collate)
    evaluator = Evaluator(
        metrics=["accuracy", "f1", "auroc"],
        loss_config=LossConfig(loss_type="bce_with_logits", pos_weight=1.0, label_smoothing=0.0),
    )
    model.eval()
    with torch.no_grad():
        metrics = evaluator.evaluate(
            model=model,
            data_loader=loader,
            device=torch.device("cpu"),
            prefix="val",
        )
    assert "val_loss" in metrics
    assert "val_accuracy" in metrics
    assert "val_f1" in metrics
    assert "val_auroc" in metrics


def test_evaluator_without_prefix_returns_raw_metric_names() -> None:
    model = TinyModel()
    loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False, collate_fn=_collate)
    evaluator = Evaluator(
        metrics=["accuracy", "f1", "auroc"],
        loss_config=LossConfig(loss_type="bce_with_logits", pos_weight=1.0, label_smoothing=0.0),
    )
    model.eval()
    with torch.no_grad():
        metrics = evaluator.evaluate(
            model=model,
            data_loader=loader,
            device=torch.device("cpu"),
            prefix=None,
        )
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "auroc" in metrics
    assert "loss" in metrics
