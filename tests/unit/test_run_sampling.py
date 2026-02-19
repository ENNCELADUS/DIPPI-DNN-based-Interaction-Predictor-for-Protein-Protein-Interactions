"""Unit tests for sampling-related run wiring."""

from __future__ import annotations

import src.run as run_module
import torch
from src.utils.config import ConfigDict
from torch import nn


def test_build_trainer_wires_ohem_warmup_epochs() -> None:
    config: ConfigDict = {
        "training_config": {
            "epochs": 2,
            "optimizer": {"type": "adamw", "lr": 1e-3},
            "scheduler": {"type": "none"},
            "loss": {
                "type": "bce_with_logits",
                "pos_weight": 1.0,
                "label_smoothing": 0.0,
            },
            "logging": {"heartbeat_every_n_steps": 0, "validation_metrics": ["auprc"]},
        },
        "data_config": {
            "dataloader": {
                "sampling": {
                    "strategy": "ohem",
                    "cap_protein": 4,
                    "pool_multiplier": 32,
                    "warmup_epochs": 3,
                }
            }
        },
        "device_config": {
            "use_mixed_precision": False,
        },
    }
    model = nn.Linear(4, 1)
    trainer, _ = run_module.build_trainer(
        config=config,
        model=model,
        device=torch.device("cpu"),
        steps_per_epoch=2,
        stage="pretrain",
    )
    assert trainer.ohem_strategy is not None
    assert trainer.ohem_strategy.warmup_epochs == 3
    assert trainer.ohem_strategy.target_batch_size == 8
    assert trainer.ohem_strategy.cap_protein == 4


def test_build_trainer_disables_ohem_for_pretrain_override() -> None:
    config: ConfigDict = {
        "training_config": {
            "epochs": 2,
            "optimizer": {"type": "adamw", "lr": 1e-3},
            "scheduler": {"type": "none"},
            "loss": {
                "type": "bce_with_logits",
                "pos_weight": 1.0,
                "label_smoothing": 0.0,
            },
            "logging": {"heartbeat_every_n_steps": 0, "validation_metrics": ["auprc"]},
        },
        "data_config": {
            "dataloader": {
                "sampling": {"strategy": "ohem", "cap_protein": 4, "warmup_epochs": 3},
                "pretrain_sampling": {"strategy": "none"},
            }
        },
        "device_config": {"use_mixed_precision": False},
    }
    model = nn.Linear(4, 1)
    trainer, _ = run_module.build_trainer(
        config=config,
        model=model,
        device=torch.device("cpu"),
        steps_per_epoch=2,
        stage="pretrain",
    )
    assert trainer.ohem_strategy is None


def test_build_trainer_enables_ohem_for_finetune_override() -> None:
    config: ConfigDict = {
        "training_config": {
            "batch_size": 32,
            "epochs": 2,
            "optimizer": {"type": "adamw", "lr": 1e-3},
            "scheduler": {"type": "none"},
            "loss": {
                "type": "bce_with_logits",
                "pos_weight": 1.0,
                "label_smoothing": 0.0,
            },
            "logging": {"heartbeat_every_n_steps": 0, "validation_metrics": ["auprc"]},
            "strategy": {"type": "none"},
            "finetune": {
                "batch_size": 32,
                "strategy": {"type": "none", "batch_size": 16},
            },
        },
        "data_config": {
            "dataloader": {
                "sampling": {"strategy": "none"},
                "finetune_sampling": {
                    "strategy": "ohem",
                    "cap_protein": 2,
                    "warmup_epochs": 1,
                },
            }
        },
        "device_config": {"use_mixed_precision": False},
    }
    model = nn.Linear(4, 1)
    stage_config = run_module._config_for_training_stage(config=config, stage="finetune")
    trainer, _ = run_module.build_trainer(
        config=stage_config,
        model=model,
        device=torch.device("cpu"),
        steps_per_epoch=2,
        stage="finetune",
    )
    assert trainer.ohem_strategy is not None
    assert trainer.ohem_strategy.warmup_epochs == 1
    assert trainer.ohem_strategy.target_batch_size == 16
    assert trainer.ohem_strategy.cap_protein == 2
