"""Unit tests for forward-call compatibility across model call styles."""

from __future__ import annotations

import torch

from src.model import V4, V5


def _sample_batch() -> dict[str, torch.Tensor]:
    return {
        "emb_a": torch.randn(2, 5, 8),
        "emb_b": torch.randn(2, 4, 8),
        "len_a": torch.tensor([5, 4], dtype=torch.long),
        "len_b": torch.tensor([4, 3], dtype=torch.long),
        "label": torch.tensor([1.0, 0.0], dtype=torch.float32),
    }


def test_v4_forward_supports_kwargs_and_batch_dict() -> None:
    model = V4(
        input_dim=8,
        d_model=8,
        encoder_layers=1,
        cross_attn_layers=1,
        n_heads=2,
        mlp_head={
            "hidden_dims": [16, 8],
            "dropout": 0.1,
            "activation": "gelu",
            "norm": "layernorm",
        },
        regularization={
            "dropout": 0.1,
            "token_dropout": 0.0,
            "cross_attention_dropout": 0.1,
            "stochastic_depth": 0.0,
        },
    )
    model.eval()
    batch = _sample_batch()
    with torch.no_grad():
        output_from_kwargs = model(**batch)
        output_from_batch = model(batch=batch)

    assert "logits" in output_from_kwargs
    assert "logits" in output_from_batch
    assert output_from_kwargs["logits"].shape == torch.Size([2, 1])
    assert output_from_batch["logits"].shape == torch.Size([2, 1])


def test_v5_forward_supports_kwargs_and_batch_dict() -> None:
    model = V5(
        input_dim=8,
        d_model=8,
        encoder_layers=1,
        cross_attn_layers=1,
        n_heads=2,
        pair_dim=4,
        cnn_dim=4,
        cnn_blocks=1,
        interaction_map={
            "include_pair_features": True,
            "similarity": "cosine",
            "eps": 1.0e-8,
        },
        pooling={"mode": "max_mean"},
        mlp_head={
            "hidden_dims": [8, 4],
            "dropout": 0.1,
            "activation": "gelu",
            "norm": "layernorm",
        },
        regularization={
            "dropout": 0.1,
            "token_dropout": 0.0,
            "cross_attention_dropout": 0.1,
            "stochastic_depth": 0.0,
            "cnn_dropout": 0.0,
        },
    )
    model.eval()
    batch = _sample_batch()
    with torch.no_grad():
        output_from_kwargs = model(**batch)
        output_from_batch = model(batch=batch)

    assert "logits" in output_from_kwargs
    assert "logits" in output_from_batch
    assert output_from_kwargs["logits"].shape == torch.Size([2, 1])
    assert output_from_batch["logits"].shape == torch.Size([2, 1])
