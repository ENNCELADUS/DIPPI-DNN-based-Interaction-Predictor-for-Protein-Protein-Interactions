import torch
from src.model.v3 import V3


def _build_model():
    return V3(
        input_dim=16,
        d_model=8,
        encoder_layers=2,
        cross_attn_layers=1,
        n_heads=2,
        mlp_head={
            "hidden_dims": [4],
            "dropout": 0.1,
            "activation": "gelu",
            "norm": "layernorm",
        },
        regularization={
            "dropout": 0.1,
            "token_dropout": 0.05,
            "cross_attention_dropout": 0.1,
        },
    )


def test_v3_forward_with_lengths():
    model = _build_model()
    batch_size = 3
    seq_a = 5
    seq_b = 7
    emb_a = torch.randn(batch_size, seq_a, 16)
    emb_b = torch.randn(batch_size, seq_b, 16)
    len_a = torch.tensor([5, 3, 4], dtype=torch.long)
    len_b = torch.tensor([7, 6, 2], dtype=torch.long)

    outputs = model(
        {
            "emb_a": emb_a,
            "emb_b": emb_b,
            "len_a": len_a,
            "len_b": len_b,
        }
    )

    assert "logits" in outputs
    logits = outputs["logits"]
    assert logits.shape == (batch_size, 1)
    assert torch.isfinite(logits).all()


def test_v3_forward_defaults_and_distances():
    model = _build_model()
    batch_size = 2
    seq_a = 6
    seq_b = 4
    emb_a = torch.randn(batch_size, seq_a, 16)
    emb_b = torch.randn(batch_size, seq_b, 16)
    dist_a = torch.randn(batch_size, seq_a, seq_a)
    dist_b = torch.randn(batch_size, seq_b, seq_b)

    outputs = model(
        {
            "emb_a": emb_a,
            "emb_b": emb_b,
            "dist_a": dist_a,
            "dist_b": dist_b,
        }
    )

    logits = outputs["logits"]
    assert logits.shape == (batch_size, 1)
    assert torch.isfinite(logits).all()
