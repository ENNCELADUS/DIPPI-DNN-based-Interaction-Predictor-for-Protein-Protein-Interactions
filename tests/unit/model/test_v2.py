import torch
from src.model.v2 import V2


def _build_model():
    return V2(
        input_dim=16,
        d_model=8,
        encoder_layers=2,  # Config compatibility: still accepted but not used
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


def test_v2_forward_with_lengths():
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


def test_v2_forward_without_lengths():
    """Test V2 with default lengths (full sequences)."""
    model = _build_model()
    batch_size = 2
    seq_a = 6
    seq_b = 4
    emb_a = torch.randn(batch_size, seq_a, 16)
    emb_b = torch.randn(batch_size, seq_b, 16)

    outputs = model(
        {
            "emb_a": emb_a,
            "emb_b": emb_b,
        }
    )

    logits = outputs["logits"]
    assert logits.shape == (batch_size, 1)
    assert torch.isfinite(logits).all()


def test_v2_with_labels():
    """Test V2 with labels (training mode)."""
    model = _build_model()
    batch_size = 2
    seq_a = 6
    seq_b = 4
    emb_a = torch.randn(batch_size, seq_a, 16)
    emb_b = torch.randn(batch_size, seq_b, 16)
    labels = torch.tensor([1.0, 0.0])

    outputs = model(
        {
            "emb_a": emb_a,
            "emb_b": emb_b,
            "label": labels,
        }
    )

    assert "logits" in outputs
    assert "loss" in outputs
    assert outputs["loss"].item() >= 0.0
    assert torch.isfinite(outputs["loss"])


def test_v2_ablation_has_fewer_parameters():
    """Verify V2 has significantly fewer parameters than V3 (no transformer encoder)."""
    from src.model.v3 import V3

    v2_model = _build_model()
    v3_model = V3(
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

    v2_params = sum(p.numel() for p in v2_model.parameters())
    v3_params = sum(p.numel() for p in v3_model.parameters())

    # V2 should have significantly fewer parameters (no transformer encoder layers)
    assert v2_params < v3_params
    # Sanity check: V2 should have at least 30% fewer parameters
    assert v2_params < 0.7 * v3_params
