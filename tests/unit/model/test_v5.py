import torch

from src.model.v5 import V5


def _build_model(**overrides):
    config = {
        "input_dim": 16,
        "d_model": 8,
        "encoder_layers": 1,
        "cross_attn_layers": 1,
        "n_heads": 2,
        "pair_dim": 4,
        "cnn_dim": 6,
        "cnn_blocks": 2,
        "interaction_map": {
            "include_pair_features": True,
            "similarity": "cosine",
            "eps": 1.0e-8,
        },
        "pooling": {"mode": "max_mean"},
        "mlp_head": {
            "hidden_dims": [4],
            "dropout": 0.1,
            "activation": "gelu",
            "norm": "layernorm",
        },
        "regularization": {
            "dropout": 0.1,
            "token_dropout": 0.0,
            "cross_attention_dropout": 0.1,
            "stochastic_depth": 0.0,
            "cnn_dropout": 0.0,
        },
    }
    config.update(overrides)
    return V5(**config)


def test_v5_forward_with_lengths_and_loss():
    model = _build_model()
    batch_size = 2
    seq_a = 5
    seq_b = 7
    emb_a = torch.randn(batch_size, seq_a, 16)
    emb_b = torch.randn(batch_size, seq_b, 16)
    len_a = torch.tensor([5, 3], dtype=torch.long)
    len_b = torch.tensor([7, 6], dtype=torch.long)

    outputs = model(
        {
            "emb_a": emb_a,
            "emb_b": emb_b,
            "len_a": len_a,
            "len_b": len_b,
            "label": torch.tensor([1.0, 0.0]),
        }
    )

    assert "logits" in outputs
    logits = outputs["logits"]
    assert logits.shape == (batch_size, 1)
    assert torch.isfinite(logits).all()
    assert "loss" in outputs
    assert torch.isfinite(outputs["loss"]).all()


def test_v5_interaction_map_channels():
    model = _build_model()
    h_a = torch.randn(2, 3, model.d_model)
    h_b = torch.randn(2, 4, model.d_model)
    interaction_map = model.map_builder(h_a, h_b)

    expected_channels = 2 * model.pair_dim
    if model.map_include_pair_features:
        expected_channels += 2 * model.pair_dim
    if model.map_similarity != "none":
        expected_channels += 1

    assert interaction_map.shape == (2, expected_channels, 3, 4)


def test_v5_pooling_head_input_dim():
    model = _build_model()
    first_linear = model.output_head.layers[0]
    assert first_linear.in_features == model.cnn_dim * 2
