import torch

from src.model.v4 import V4, ResidualMLPHead


def _build_model():
    return V4(
        input_dim=16,
        d_model=8,
        encoder_layers=1,
        cross_attn_layers=1,
        n_heads=2,
        mlp_head={
            "hidden_dims": [8],
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


def test_v4_forward_with_lengths_and_loss():
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
            "label": torch.tensor([1.0, 0.0, 1.0]),
        }
    )

    assert "logits" in outputs
    logits = outputs["logits"]
    assert logits.shape == (batch_size, 1)
    assert torch.isfinite(logits).all()
    assert "loss" in outputs
    assert torch.isfinite(outputs["loss"]).all()


def test_v4_forward_defaults():
    model = _build_model()
    batch_size = 2
    seq_a = 6
    seq_b = 4
    emb_a = torch.randn(batch_size, seq_a, 16)
    emb_b = torch.randn(batch_size, seq_b, 16)

    outputs = model({"emb_a": emb_a, "emb_b": emb_b})

    logits = outputs["logits"]
    assert logits.shape == (batch_size, 1)
    assert torch.isfinite(logits).all()


def test_v4_cross_attention_shared_weights():
    model = _build_model()
    layer = model.cross_attention.layers[0]
    assert hasattr(layer, "attn")
    assert hasattr(layer, "ffn")
    assert not hasattr(layer, "attn_a_to_b")
    assert not hasattr(layer, "attn_b_to_a")


def test_v4_pair_feature_dim_and_head():
    model = _build_model()
    expected_dim = 4 * model.d_model
    assert model.pair_norm.normalized_shape == (expected_dim,)
    assert isinstance(model.output_head, ResidualMLPHead)
