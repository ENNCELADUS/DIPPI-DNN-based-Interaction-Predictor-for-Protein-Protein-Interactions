"""Unit tests for domain-adaptation config parsing in run.py."""

from __future__ import annotations

import pytest

import src.run as run_module
from src.utils.config import ConfigDict


def _base_config() -> ConfigDict:
    return {
        "run_config": {
            "stages": ["evaluate"],
            "seed": 7,
            "eval_run_id": "eval_run",
            "load_checkpoint_path": "artifacts/model.pth",
            "save_best_only": True,
        },
        "device_config": {
            "device": "cpu",
            "ddp_enabled": False,
            "use_mixed_precision": False,
        },
        "data_config": {},
        "model_config": {"model": "v3"},
        "training_config": {},
        "evaluate": {"metrics": ["accuracy"]},
    }


def test_domain_adaptation_defaults_to_none() -> None:
    config = _base_config()
    parsed = run_module._domain_adaptation_config(config)

    assert parsed.method == "none"
    assert parsed.target_split == "test"
    assert parsed.compare_pre_post is True
    assert parsed.shot.epochs == 15
    assert parsed.shot.tau_pos == pytest.approx(0.88)
    assert parsed.shot.tau_neg == pytest.approx(0.98)
    assert parsed.shot.pos_weight == pytest.approx(5.0)
    assert parsed.shot.neg_weight == pytest.approx(1.0)
    assert parsed.shot.prior_ema_momentum == pytest.approx(0.02)
    assert parsed.shot.lr == pytest.approx(5.0e-4)
    assert parsed.shot.weight_decay == pytest.approx(3.0e-4)
    assert parsed.shot.class_count_threshold == 2


def test_domain_adaptation_shot_config_is_parsed() -> None:
    config = _base_config()
    config["domain_adaptation"] = {
        "method": "shot",
        "target_split": "test",
        "compare_pre_post": False,
        "shot": {
            "epochs": 3,
            "beta": 0.2,
            "lr": 0.005,
            "use_amp": False,
        },
    }

    parsed = run_module._domain_adaptation_config(config)
    assert parsed.method == "shot"
    assert parsed.compare_pre_post is False
    assert parsed.shot.epochs == 3
    assert parsed.shot.beta == pytest.approx(0.2)
    assert parsed.shot.lr == pytest.approx(0.005)


def test_domain_adaptation_rejects_non_test_split() -> None:
    config = _base_config()
    config["domain_adaptation"] = {
        "method": "shot",
        "target_split": "valid",
    }

    with pytest.raises(ValueError, match="target_split"):
        run_module._domain_adaptation_config(config)


def test_domain_adaptation_rejects_invalid_threshold_order() -> None:
    config = _base_config()
    config["domain_adaptation"] = {
        "method": "shot",
        "target_split": "test",
        "shot": {
            "tau_pos": 0.95,
            "tau_neg": 0.90,
        },
    }

    with pytest.raises(ValueError, match="tau_neg"):
        run_module._domain_adaptation_config(config)
