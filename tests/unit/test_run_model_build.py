"""Unit tests for model construction entrypoints."""

from pathlib import Path

import pytest
from src.run import build_model
from src.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def _base_config(model_name: str) -> dict[str, object]:
    return {
        "model_config": {
            "model": model_name,
            "input_dim": 8,
            "d_model": 8,
            "encoder_layers": 1,
            "cross_attn_layers": 1,
            "n_heads": 2,
            "mlp_head": {
                "hidden_dims": [8, 4],
                "dropout": 0.1,
                "activation": "gelu",
                "norm": "layernorm",
            },
            "regularization": {
                "dropout": 0.1,
                "token_dropout": 0.0,
                "cross_attention_dropout": 0.1,
                "stochastic_depth": 0.0,
            },
        }
    }


def test_build_model_v3() -> None:
    model = build_model(_base_config("v3"))
    assert model.__class__.__name__ == "V3"


def test_build_model_v4() -> None:
    model = build_model(_base_config("v4"))
    assert model.__class__.__name__ == "V4"


def test_build_model_v5() -> None:
    config = _base_config("v5")
    model_config = config["model_config"]
    assert isinstance(model_config, dict)
    model_config["pair_dim"] = 4
    model_config["cnn_dim"] = 4
    model = build_model(config)
    assert model.__class__.__name__ == "V5"


@pytest.mark.parametrize(
    ("config_name", "expected_model_class"),
    [("v4.yaml", "V4"), ("v5.yaml", "V5")],
)
def test_build_model_from_pipeline_configs(
    config_name: str, expected_model_class: str
) -> None:
    config = load_config(REPO_ROOT / "configs" / config_name)
    model = build_model(config)
    assert model.__class__.__name__ == expected_model_class


def test_build_model_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model(_base_config("unknown"))
