"""Sequence-only embedding pipeline built around local ESM3 models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import numpy as np

from ..core.base import BaseEmbedder
from ..core.types import EmbeddingConfig, EmbeddingResult
from ..core.validation import clean_protein_sequence, ensure_sequence


class LocalModelBackendError(RuntimeError):
    """Raised when the local embedding backend cannot be initialised."""


@dataclass(slots=True)
class LocalSequenceBackend:
    """Adapter that isolates local ESM3-style inference from the embedder."""

    model: Any
    protein_factory: Callable[[str], Any] | None = None
    logits_factory: Callable[[], Any] | None = None

    def embed(self, sequence: str) -> Any:
        """Return token-level embeddings for ``sequence``."""

        logits_config = self._logits_config()
        protein = self._protein(sequence)
        protein_tensor = self.model.encode(protein)
        sequence_output = self.model.logits(protein_tensor, logits_config)
        embeddings = getattr(sequence_output, "embeddings", None)
        if embeddings is None:
            raise LocalModelBackendError("backend did not produce embeddings")
        return embeddings

    def _logits_config(self) -> Any:
        if self.logits_factory is not None:
            return self.logits_factory()
        return _default_logits_config()

    def _protein(self, sequence: str) -> Any:
        if self.protein_factory is not None:
            return self.protein_factory(sequence)
        return _default_protein(sequence)


def _load_local_model(model_name: str, *, device: str) -> Any:
    try:
        from esm.models.esm3 import ESM3  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional dep
        raise LocalModelBackendError("esm3 package is not installed") from exc

    return ESM3.from_pretrained(model_name).to(device)


def _default_protein(sequence: str) -> Any:
    from esm.sdk.api import ESMProtein  # type: ignore[import-not-found]

    return ESMProtein(sequence=sequence)


def _default_logits_config() -> Any:
    from esm.sdk.api import LogitsConfig  # type: ignore[import-not-found]

    return LogitsConfig(
        sequence=True,
        return_embeddings=True,
        return_hidden_states=False,
    )


class SequenceEmbedder(BaseEmbedder):
    """Produce embeddings by running sequences through a local ESM3 model."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        super().__init__(config=config)
        if not self.config.use_local_model:
            raise ValueError("SequenceEmbedder only supports local ESM3 models")
        self._backend: Optional[LocalSequenceBackend] = None

    def embed_one(self, uniprot_id: str, sequence: str) -> EmbeddingResult:
        """Return an embedding for ``sequence`` using the configured backend."""

        try:
            prepared, extra_metadata = self._prepare_sequence(sequence)
        except ValueError as exc:
            return self._result_with_error(uniprot_id, sequence, str(exc))

        try:
            backend = self._require_backend()
        except LocalModelBackendError as exc:
            return self._result_with_error(uniprot_id, sequence, str(exc))
        try:
            embeddings = _to_numpy(backend.embed(prepared))
        except Exception as exc:  # pragma: no cover - escalated to metadata
            return self._result_with_error(uniprot_id, sequence, str(exc))

        metadata = self._build_success_metadata(sequence, prepared, extra_metadata)
        return EmbeddingResult(
            uniprot_id=uniprot_id, embedding=embeddings, metadata=metadata
        )

    def embed_many(self, items: Iterable[tuple[str, str]]) -> list[EmbeddingResult]:
        backend: Optional[LocalSequenceBackend] = self._backend
        results = []
        for uniprot_id, sequence in items:
            try:
                prepared, extra_metadata = self._prepare_sequence(sequence)
            except ValueError as exc:
                results.append(self._result_with_error(uniprot_id, sequence, str(exc)))
                continue
            if backend is None:
                try:
                    backend = self._require_backend()
                except LocalModelBackendError as exc:
                    results.append(
                        self._result_with_error(uniprot_id, sequence, str(exc))
                    )
                    continue
            try:
                embeddings = _to_numpy(backend.embed(prepared))
            except Exception as exc:  # pragma: no cover - escalated to metadata
                results.append(self._result_with_error(uniprot_id, sequence, str(exc)))
                continue
            metadata = self._build_success_metadata(sequence, prepared, extra_metadata)
            results.append(
                EmbeddingResult(
                    uniprot_id=uniprot_id,
                    embedding=embeddings,
                    metadata=metadata,
                )
            )
        return results

    def attach_backend(self, backend: LocalSequenceBackend) -> None:
        """Inject a backend, primarily for tests."""

        self._backend = backend

    def _require_backend(self) -> LocalSequenceBackend:
        if self._backend is None:
            model = _load_local_model(
                self.config.model_name, device=self.config.resolved_device()
            )
            self._backend = LocalSequenceBackend(model)
        return self._backend

    def _result_with_error(
        self, uniprot_id: str, sequence: str, error: str
    ) -> EmbeddingResult:
        metadata = {
            "pipeline": "sequence",
            "model_name": self.config.model_name,
            "device": self.config.resolved_device(),
            "sequence_length": len(sequence),
            "original_sequence": sequence,
        }
        return EmbeddingResult(
            uniprot_id=uniprot_id, embedding=None, metadata=metadata, error=error
        )

    def _prepare_sequence(self, sequence: str) -> tuple[str, dict[str, Any]]:
        """Clean ``sequence`` and optionally truncate it."""

        max_length = self.config.max_sequence_length
        effective_max = max_length if max_length > 0 else None
        truncate_enabled = (
            self.config.truncate_long_sequences and effective_max is not None
        )

        ensure_sequence(
            sequence,
            max_length=None if truncate_enabled else effective_max,
        )
        cleaned = clean_protein_sequence(sequence)
        if not cleaned:
            raise ValueError("empty sequence after cleaning")

        extra_metadata: dict[str, Any] = {}
        if effective_max is not None and len(cleaned) > effective_max:
            if not truncate_enabled:
                raise ValueError("invalid protein sequence")
            extra_metadata = {
                "truncated": True,
                "truncated_from_length": len(cleaned),
                "truncate_length": effective_max,
            }
            cleaned = cleaned[:effective_max]

        return cleaned, extra_metadata

    def _build_success_metadata(
        self,
        original_sequence: str,
        processed_sequence: str,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        metadata = {
            "pipeline": "sequence",
            "model_name": self.config.model_name,
            "device": self.config.resolved_device(),
            "sequence_length": len(processed_sequence),
            "original_sequence": original_sequence,
            "cleaned_sequence": processed_sequence,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return metadata


def _to_numpy(value: Any) -> np.ndarray:
    """Convert backend output to a numpy array."""

    if isinstance(value, np.ndarray):
        return value

    try:
        import torch  # type: ignore[import-not-found]

        if isinstance(value, torch.Tensor):  # pragma: no cover - optional dependency
            return value.detach().cpu().numpy()
    except ModuleNotFoundError:  # pragma: no cover - torch not installed
        pass

    if hasattr(value, "numpy"):
        array = value.numpy()
        if isinstance(array, np.ndarray):
            return array

    return np.asarray(value)
