import torch

from src.model.v6 import V6, LoRALinear, apply_lora


class DummyProtein:
    def __init__(self, sequence: str) -> None:
        self.sequence = sequence


class DummyESM(torch.nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def encode(self, protein: DummyProtein) -> DummyProtein:
        return protein

    def logits(self, protein_tensor, logits_config: object):
        if isinstance(protein_tensor, list):
            lengths = [len(protein.sequence) for protein in protein_tensor]
            max_len = max(lengths) if lengths else 0
            embeddings = torch.randn(len(lengths), max_len + 2, self.embed_dim)
        else:
            length = len(protein_tensor.sequence)
            embeddings = torch.randn(1, length + 2, self.embed_dim)

        class Output:
            def __init__(self, embeddings: torch.Tensor) -> None:
                self.embeddings = embeddings

        return Output(embeddings)


def _build_model(monkeypatch):
    monkeypatch.setattr(V6, "_load_esm3", lambda self: DummyESM(embed_dim=16))
    monkeypatch.setattr(V6, "_load_esm_sdk", lambda self: (DummyProtein, object()))
    return V6(
        input_dim=16,
        d_model=8,
        cross_attn_layers=1,
        n_heads=2,
        mlp_head={"dropout": 0.1},
        regularization={"dropout": 0.1, "cross_attention_dropout": 0.1},
        lora={"r": 0},
    )


def test_v6_forward_shapes(monkeypatch):
    model = _build_model(monkeypatch)
    batch = {
        "seq_a": ["ACDEFG", "AAAA"],
        "seq_b": ["QQQ", "RRRRR"],
        "label": torch.tensor([1.0, 0.0]),
    }

    outputs = model(batch)
    assert "logits" in outputs
    assert outputs["logits"].shape == (2, 1)
    assert torch.isfinite(outputs["logits"]).all()
    assert "loss" in outputs
    assert torch.isfinite(outputs["loss"]).all()


def test_v6_pair_features_dim(monkeypatch):
    model = _build_model(monkeypatch)
    a = torch.randn(3, model.d_model)
    b = torch.randn(3, model.d_model)
    z = model._pair_features(a, b)
    assert z.shape == (3, model.d_model * 4)


def test_v6_embed_batch_strips_cls(monkeypatch):
    model = _build_model(monkeypatch)
    seqs = ["ACDE", "AAAAAA"]
    padded, lengths = model._embed_batch(seqs)
    assert padded.shape[0] == 2
    assert lengths.tolist() == [4, 6]


def test_v6_requires_seq_keys(monkeypatch):
    model = _build_model(monkeypatch)
    try:
        model({"seq_a": ["AAA"]})
    except KeyError:
        pass
    else:
        raise AssertionError("Expected KeyError for missing seq_b")


def test_v6_seq_type_validation(monkeypatch):
    model = _build_model(monkeypatch)
    try:
        model({"seq_a": "AAA", "seq_b": ["BBB"]})
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError for seq_a string")

    try:
        model({"seq_a": ["AAA", 123], "seq_b": ["BBB", "CCC"]})
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError for non-string sequence items")


def test_v6_batch_size_mismatch(monkeypatch):
    model = _build_model(monkeypatch)
    try:
        model({"seq_a": ["AAA"], "seq_b": ["BBB", "CCC"]})
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for mismatched batch sizes")


def test_v6_embedding_dim_mismatch(monkeypatch):
    monkeypatch.setattr(V6, "_load_esm3", lambda self: DummyESM(embed_dim=8))
    monkeypatch.setattr(V6, "_load_esm_sdk", lambda self: (DummyProtein, object()))
    model = V6(
        input_dim=16,
        d_model=8,
        cross_attn_layers=1,
        n_heads=2,
        mlp_head={"dropout": 0.1},
        regularization={"dropout": 0.1, "cross_attention_dropout": 0.1},
        lora={"r": 0},
    )
    try:
        model({"seq_a": ["AAA"], "seq_b": ["BBB"]})
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for embedding dim mismatch")


class DummyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.layernorm_qkv = torch.nn.Linear(4, 12)
        self.attn.out_proj = torch.nn.Linear(4, 4)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4),
        )


class DummyTransformer(torch.nn.Module):
    def __init__(self, num_blocks: int) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([DummyBlock() for _ in range(num_blocks)])


class DummyBackbone(torch.nn.Module):
    def __init__(self, num_blocks: int) -> None:
        super().__init__()
        self.transformer = DummyTransformer(num_blocks)


def test_apply_lora_last_n_layers():
    model = DummyBackbone(num_blocks=2)
    matched = apply_lora(
        model,
        last_n_layers=1,
        target_modules=["layernorm_qkv", "out_proj", "ffn"],
        r=4,
        alpha=8,
        dropout=0.0,
    )
    assert matched
    assert isinstance(model.transformer.blocks[1].attn.layernorm_qkv, LoRALinear)
    assert isinstance(model.transformer.blocks[1].attn.out_proj, LoRALinear)
    assert isinstance(model.transformer.blocks[1].ffn[0], LoRALinear)
    assert not isinstance(model.transformer.blocks[0].attn.layernorm_qkv, LoRALinear)
