"""Tests for the sequence embedding pipeline."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.embed.core.types import EmbeddingConfig
from src.embed.pipelines.sequence import (
    LocalModelBackendError,
    LocalSequenceBackend,
    SequenceEmbedder,
    _default_logits_config,
    _default_protein,
    _load_local_model,
)


class DummyBackend:
    """Minimal backend used to sidestep heavy ESM dependencies during tests."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, sequence: str) -> Any:
        self.calls.append(sequence)
        length = len(sequence)
        return np.arange(length * 2, dtype=np.float32).reshape(1, length, 2)


def test_sequence_embedder_rejects_remote_configuration() -> None:
    cfg = EmbeddingConfig(use_local_model=False)
    with pytest.raises(ValueError):
        SequenceEmbedder(config=cfg)


def test_embed_one_returns_result_with_injected_backend() -> None:
    embedder = SequenceEmbedder()
    backend = DummyBackend()
    embedder.attach_backend(backend)

    result = embedder.embed_one("P12345", "ac-d")

    assert result.ok is True
    assert isinstance(result.embedding, np.ndarray)
    assert result.embedding.shape == (1, 3, 2)
    assert result.metadata["sequence_length"] == 3
    assert result.metadata["original_sequence"] == "ac-d"
    assert result.metadata["cleaned_sequence"] == "ACD"
    assert backend.calls == ["ACD"]


def test_embed_one_returns_error_for_invalid_sequence() -> None:
    embedder = SequenceEmbedder()
    embedder.attach_backend(DummyBackend())

    result = embedder.embed_one("P00000", "AC1")

    assert result.ok is False
    assert "invalid protein sequence" in (result.error or "")


def test_embed_one_handles_empty_after_cleaning() -> None:
    embedder = SequenceEmbedder()
    embedder.attach_backend(DummyBackend())

    result = embedder.embed_one("P0EMPTY", "*-.")

    assert result.ok is False
    assert result.error == "empty sequence after cleaning"


def test_embed_many_mixes_success_and_failure() -> None:
    embedder = SequenceEmbedder()
    backend = DummyBackend()
    embedder.attach_backend(backend)

    items = [("GOOD", "ACD"), ("BAD", "AC1")]
    results = embedder.embed_many(items)

    assert results[0].ok is True
    assert results[1].ok is False
    assert backend.calls == ["ACD"]


def test_embedder_truncates_when_enabled() -> None:
    cfg = EmbeddingConfig(max_sequence_length=3, truncate_long_sequences=True)
    embedder = SequenceEmbedder(config=cfg)
    backend = DummyBackend()
    embedder.attach_backend(backend)

    result = embedder.embed_one("PTRUNC", "ACDEFG")

    assert result.ok is True
    assert backend.calls == ["ACD"]
    assert result.metadata["cleaned_sequence"] == "ACD"
    assert result.metadata["sequence_length"] == 3
    assert result.metadata["truncated"] is True
    assert result.metadata["truncated_from_length"] == 6
    assert result.metadata["truncate_length"] == 3


def test_embedder_errors_on_long_sequence_without_truncation() -> None:
    cfg = EmbeddingConfig(max_sequence_length=2, truncate_long_sequences=False)
    embedder = SequenceEmbedder(config=cfg)
    embedder.attach_backend(DummyBackend())

    result = embedder.embed_one("PLONG", "ACDE")

    assert result.ok is False
    assert "invalid protein sequence" in (result.error or "")


def test_embedder_falls_back_to_error_when_backend_initialisation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedder = SequenceEmbedder()

    def fail_load(model_name: str, device: str) -> Any:
        raise LocalModelBackendError("model cache missing")

    monkeypatch.setattr("src.embed.pipelines.sequence._load_local_model", fail_load)

    result = embedder.embed_one("PFAIL", "ACD")

    assert result.ok is False
    assert result.error == "model cache missing"


def test_embedder_uses_default_backend_with_stubbed_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedder = SequenceEmbedder()

    class StubModel:
        def encode(self, protein: SimpleNamespace) -> SimpleNamespace:
            return protein

        def logits(
            self, protein_tensor: SimpleNamespace, logits_config: SimpleNamespace
        ) -> SimpleNamespace:
            length = len(protein_tensor.sequence)
            embeddings = np.arange(length, dtype=np.float32).reshape(1, length, 1)
            return SimpleNamespace(embeddings=embeddings)

    monkeypatch.setattr(
        "src.embed.pipelines.sequence._load_local_model",
        lambda name, device: StubModel(),
    )
    monkeypatch.setattr(
        "src.embed.pipelines.sequence._default_protein",
        lambda sequence: SimpleNamespace(sequence=sequence),
    )
    monkeypatch.setattr(
        "src.embed.pipelines.sequence._default_logits_config",
        lambda: SimpleNamespace(flag="logits"),
    )

    result = embedder.embed_one("PSTUB", "ACD")

    assert result.ok is True
    assert isinstance(result.embedding, np.ndarray)
    assert result.embedding.shape == (1, 3, 1)


def test_embed_many_handles_empty_after_cleaning_without_backend() -> None:
    embedder = SequenceEmbedder()
    embedder.attach_backend(DummyBackend())

    results = embedder.embed_many([("EMPTY", "*-.")])

    assert results[0].ok is False
    assert results[0].error == "empty sequence after cleaning"


def test_embed_many_reports_backend_initialisation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedder = SequenceEmbedder()
    embedder.attach_backend(None)  # type: ignore[arg-type]

    def fail_load(model_name: str, device: str) -> Any:
        raise LocalModelBackendError("model cache missing")

    monkeypatch.setattr("src.embed.pipelines.sequence._load_local_model", fail_load)

    results = embedder.embed_many([("A", "ACD")])

    assert results[0].ok is False
    assert results[0].error == "model cache missing"


def test_local_sequence_backend_uses_supplied_factories() -> None:
    class StubModel:
        def encode(self, protein: SimpleNamespace) -> SimpleNamespace:
            assert protein.kind == "protein"
            return protein

        def logits(
            self, protein_tensor: SimpleNamespace, logits_config: SimpleNamespace
        ) -> SimpleNamespace:
            assert logits_config.kind == "logits"
            return SimpleNamespace(embeddings=protein_tensor.sequence)

    backend = LocalSequenceBackend(
        model=StubModel(),
        protein_factory=lambda sequence: SimpleNamespace(
            sequence=sequence, kind="protein"
        ),
        logits_factory=lambda: SimpleNamespace(kind="logits"),
    )

    assert backend.embed("ACD") == "ACD"


def test_local_sequence_backend_raises_when_embeddings_missing() -> None:
    class StubModel:
        def encode(self, protein: str) -> str:
            return protein

        def logits(self, protein_tensor: str, logits_config: object) -> SimpleNamespace:
            return SimpleNamespace()  # missing embeddings attribute

    backend = LocalSequenceBackend(
        model=StubModel(),
        protein_factory=lambda sequence: sequence,
        logits_factory=lambda: object(),
    )

    with pytest.raises(
        LocalModelBackendError, match="backend did not produce embeddings"
    ):
        backend.embed("ACD")


def test_load_local_model_with_stubbed_module(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyESM3:
        def __init__(self) -> None:
            self.loaded_name: str | None = None
            self.device: str | None = None

        @classmethod
        def from_pretrained(cls, model_name: str) -> "DummyESM3":
            inst = cls()
            inst.loaded_name = model_name
            return inst

        def to(self, device: str) -> "DummyESM3":
            self.device = device
            return self

    esm_pkg = ModuleType("esm")
    models_pkg = ModuleType("esm.models")
    esm3_pkg = ModuleType("esm.models.esm3")
    esm3_pkg.ESM3 = DummyESM3

    monkeypatch.setitem(sys.modules, "esm", esm_pkg)
    monkeypatch.setitem(sys.modules, "esm.models", models_pkg)
    monkeypatch.setitem(sys.modules, "esm.models.esm3", esm3_pkg)

    model = _load_local_model("esm3", device="cuda")

    assert isinstance(model, DummyESM3)
    assert model.loaded_name == "esm3"
    assert model.device == "cuda"


def test_default_factories_import_from_stubbed_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyProtein:
        def __init__(self, sequence: str) -> None:
            self.sequence = sequence

    class DummyLogits:
        def __init__(self) -> None:
            self.kwargs = {"sequence": True}

    api_pkg = ModuleType("esm.sdk.api")
    api_pkg.ESMProtein = DummyProtein
    api_pkg.LogitsConfig = lambda **kwargs: SimpleNamespace(**kwargs)

    monkeypatch.setitem(sys.modules, "esm", ModuleType("esm"))
    monkeypatch.setitem(sys.modules, "esm.sdk", ModuleType("esm.sdk"))
    monkeypatch.setitem(sys.modules, "esm.sdk.api", api_pkg)

    protein = _default_protein("ACD")
    logits = _default_logits_config()

    assert isinstance(protein, DummyProtein)
    assert protein.sequence == "ACD"
    assert logits.sequence is True
