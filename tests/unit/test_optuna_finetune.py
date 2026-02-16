"""Unit tests for Optuna finetune search configuration and parameter wiring."""

from __future__ import annotations

import copy

import optuna
import pytest
from src.tune import finetune_optuna
from src.utils.config import ConfigDict


def _base_config() -> ConfigDict:
    return {
        "run_config": {"seed": 47, "load_checkpoint_path": "artifacts/pretrain.pth"},
        "model_config": {"model": "v3"},
        "training_config": {
            "monitor_metric": "auprc",
            "finetune": {
                "batch_size": 16,
                "monitor_metric": "auprc",
                "optimizer": {
                    "lr": 1.0e-4,
                    "weight_decay": 0.01,
                },
                "scheduler": {"max_lr": 1.0e-4, "pct_start": 0.2},
                "loss": {"pos_weight": 7.0, "label_smoothing": 0.01},
                "strategy": {"unfreeze_epoch": 3},
                "optuna": {
                    "enabled": True,
                    "study_name": "unit_v3_finetune",
                    "n_trials": 20,
                    "sampler": {"type": "tpe", "seed": 47},
                    "pruner": {
                        "type": "median",
                        "n_startup_trials": 0,
                        "n_warmup_steps": 0,
                        "interval_steps": 1,
                    },
                    "parameters": {
                        "lr": {
                            "distribution": {
                                "type": "float",
                                "low": 1.0e-5,
                                "high": 1.0e-3,
                                "log": True,
                            },
                            "apply_to": ["optimizer.lr", "scheduler.max_lr"],
                        }
                    },
                },
            },
        },
        "device_config": {"device": "cpu", "ddp_enabled": False},
        "data_config": {"dataloader": {"sampling": {"strategy": "none"}}},
        "evaluate": {"metrics": ["auprc"]},
    }


def test_parse_optuna_config_valid() -> None:
    config = _base_config()

    parsed = finetune_optuna.parse_optuna_config(config)

    assert parsed.enabled is True
    assert parsed.study_name == "unit_v3_finetune"
    assert parsed.n_trials == 20
    assert parsed.direction == "maximize"
    assert parsed.metric == "auprc"
    assert parsed.sampler_type == "tpe"
    assert parsed.pruner_type == "median"
    assert len(parsed.parameters) == 1
    assert parsed.parameters[0].apply_to == ("optimizer.lr", "scheduler.max_lr")


def test_parse_optuna_config_invalid_distribution_raises() -> None:
    config = _base_config()
    training_cfg = config["training_config"]
    assert isinstance(training_cfg, dict)
    finetune_cfg = training_cfg["finetune"]
    assert isinstance(finetune_cfg, dict)
    optuna_cfg = finetune_cfg["optuna"]
    assert isinstance(optuna_cfg, dict)
    params_cfg = optuna_cfg["parameters"]
    assert isinstance(params_cfg, dict)
    lr_cfg = params_cfg["lr"]
    assert isinstance(lr_cfg, dict)
    distribution_cfg = lr_cfg["distribution"]
    assert isinstance(distribution_cfg, dict)
    distribution_cfg["type"] = "unknown"

    with pytest.raises(ValueError, match="type must be one of"):
        finetune_optuna.parse_optuna_config(config)


def test_parse_optuna_config_invalid_apply_path_raises() -> None:
    config = _base_config()
    training_cfg = config["training_config"]
    assert isinstance(training_cfg, dict)
    finetune_cfg = training_cfg["finetune"]
    assert isinstance(finetune_cfg, dict)
    optuna_cfg = finetune_cfg["optuna"]
    assert isinstance(optuna_cfg, dict)
    params_cfg = optuna_cfg["parameters"]
    assert isinstance(params_cfg, dict)
    lr_cfg = params_cfg["lr"]
    assert isinstance(lr_cfg, dict)
    lr_cfg["apply_to"] = ["optimizer.missing_key"]

    with pytest.raises(ValueError, match="Unknown finetune path"):
        finetune_optuna.parse_optuna_config(config)


def test_apply_trial_parameters_updates_multiple_targets() -> None:
    config = _base_config()
    parsed = finetune_optuna.parse_optuna_config(config)
    trial = optuna.trial.FixedTrial({"lr": 2.5e-4})
    trial_config = copy.deepcopy(config)

    sampled = finetune_optuna._apply_trial_parameters(
        trial=trial,
        config=trial_config,
        parameter_specs=parsed.parameters,
    )

    assert sampled["lr"] == 2.5e-4
    training_cfg = trial_config["training_config"]
    assert isinstance(training_cfg, dict)
    finetune_cfg = training_cfg["finetune"]
    assert isinstance(finetune_cfg, dict)
    optimizer_cfg = finetune_cfg["optimizer"]
    assert isinstance(optimizer_cfg, dict)
    scheduler_cfg = finetune_cfg["scheduler"]
    assert isinstance(scheduler_cfg, dict)
    assert optimizer_cfg["lr"] == 2.5e-4
    assert scheduler_cfg["max_lr"] == 2.5e-4
