"""Unit tests for model construction entrypoints."""

import pytest
from src.run import build_model


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


def test_build_model_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model(_base_config("unknown"))
