"""Structure-aware embedding pipeline built on local ESM3 models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ..core.base import BaseEmbedder
from ..core.types import EmbeddingConfig, EmbeddingResult, MultimodalStructurePayload
from ..core.validation import ensure_sequence
from ..io.structure import StructureDataNotFoundError, load_multimodal_structure
from .sequence import LocalModelBackendError, SequenceEmbedder, _load_local_model


logger = logging.getLogger(__name__)


try:  # pragma: no cover - optional dependency
    import torch
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    torch = None  # type: ignore


try:  # pragma: no cover - optional dependency
    from esm.sdk.api import ESMProtein, LogitsConfig
    from esm.utils import residue_constants
except ModuleNotFoundError:  # pragma: no cover - handled in backend
    ESMProtein = None  # type: ignore
    LogitsConfig = None  # type: ignore
    residue_constants = None  # type: ignore


MultimodalLoader = Callable[[Path, str], MultimodalStructurePayload]


class MultimodalBackendError(RuntimeError):
    """Raised when the multimodal backend fails to return embeddings."""


@dataclass(slots=True)
class LocalMultimodalBackend:
    """Adapter that prepares sequence+structure inputs for ESM3."""

    model: Any

    def embed(
        self, payload: MultimodalStructurePayload
    ) -> dict[str, dict[str, Optional[np.ndarray]]]:
        if (
            torch is None
            or ESMProtein is None
            or LogitsConfig is None
            or residue_constants is None
        ):
            raise MultimodalBackendError(
                "esm3 package with torch support is required for multimodal embeddings"
            )

        protein = self._build_protein(payload)
        protein_tensor = self.model.encode(protein)

        outputs: dict[str, dict[str, Optional[np.ndarray]]] = {}
        outputs["sequence_structure"] = self._run_logits(
            protein_tensor,
            LogitsConfig(
                sequence=True,
                structure=True,
                return_embeddings=True,
                return_hidden_states=False,
            ),
        )

        # Structure-only embeddings requested by downstream PPI tasks.
        outputs["structure_only"] = self._run_logits(
            protein_tensor,
            LogitsConfig(
                sequence=False,
                structure=True,
                return_embeddings=True,
                return_hidden_states=False,
            ),
        )

        return outputs

    def _run_logits(
        self, protein_tensor: Any, config: Any
    ) -> dict[str, Optional[np.ndarray]]:
        logits_output = self.model.logits(protein_tensor, config)
        embeddings = getattr(logits_output, "embeddings", None)
        if embeddings is None:
            raise MultimodalBackendError("backend did not produce embeddings")

        per_residue = embeddings.detach().cpu().numpy()
        mean_embedding = getattr(logits_output, "mean_embedding", None)
        mean_numpy: Optional[np.ndarray]
        if mean_embedding is not None:
            mean_numpy = mean_embedding.detach().cpu().numpy()
        else:
            mean_numpy = None
        return {"per_residue": per_residue, "mean": mean_numpy}

    def _build_protein(self, payload: MultimodalStructurePayload) -> Any:
        assert (
            torch is not None
            and residue_constants is not None
            and ESMProtein is not None
        )

        length = payload.coordinates.residue_count
        atom37_count = len(residue_constants.atom_order)
        coords = torch.full(
            (length, atom37_count, 3), float("nan"), dtype=torch.float32
        )

        atom_index = residue_constants.atom_order
        values = payload.coordinates.values
        mask_values = payload.mask.values
        for residue_idx in range(length):
            if not mask_values[residue_idx]:
                continue
            for atom_name, xyz in zip(
                payload.coordinates.atom_order, values[residue_idx]
            ):
                atom_id = atom_index.get(atom_name)
                if atom_id is None:
                    continue
                if np.isnan(xyz).any():
                    continue
                coords[residue_idx, atom_id] = torch.tensor(xyz, dtype=torch.float32)

        return ESMProtein(sequence=payload.sequence, coordinates=coords)


class MultimodalEmbedder(BaseEmbedder):
    """Produce embeddings using both sequence and structure when available."""

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        *,
        structure_loader: Callable[..., MultimodalStructurePayload] | None = None,
        backend_factory: Callable[[Any], LocalMultimodalBackend] | None = None,
        sequence_fallback: SequenceEmbedder | None = None,
    ) -> None:
        super().__init__(config=config)
        self._structure_loader = structure_loader or load_multimodal_structure
        self._backend_factory = backend_factory or (
            lambda model: LocalMultimodalBackend(model)
        )
        self._backend: Optional[LocalMultimodalBackend] = None
        self._sequence_fallback = sequence_fallback

    def embed_one(self, uniprot_id: str, sequence: str) -> EmbeddingResult:
        logger.debug(f"Processing {uniprot_id} (length={len(sequence)})")

        try:
            ensure_sequence(sequence, max_length=self.config.max_sequence_length)
        except ValueError as exc:
            logger.warning(f"{uniprot_id}: Sequence validation failed - {exc}")
            return self._sequence_error(uniprot_id, sequence, str(exc))

        try:
            logger.debug(f"{uniprot_id}: Loading structure data")
            payload = self._structure_loader(
                self.config.data_root,
                uniprot_id,
                sequence=sequence,
                allow_mask_infer=True,
            )
            logger.info(
                f"{uniprot_id}: Structure loaded - "
                f"residues={payload.coordinates.residue_count}, "
                f"coverage={payload.mask.values.mean():.2%}, "
                f"source={payload.metadata.get('pdb_id', 'unknown')}"
            )
        except StructureDataNotFoundError as exc:
            logger.info(
                f"{uniprot_id}: Structure not found, falling back to sequence-only - {exc}"
            )
            return self._fallback_to_sequence(uniprot_id, sequence, error=str(exc))
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning(
                f"{uniprot_id}: Structure loading failed, falling back - {exc}"
            )
            return self._fallback_to_sequence(uniprot_id, sequence, error=str(exc))

        try:
            logger.debug(f"{uniprot_id}: Initializing multimodal backend")
            backend = self._require_backend()
        except (LocalModelBackendError, MultimodalBackendError) as exc:
            logger.warning(
                f"{uniprot_id}: Backend initialization failed, falling back - {exc}"
            )
            return self._fallback_to_sequence(uniprot_id, sequence, error=str(exc))

        try:
            logger.debug(f"{uniprot_id}: Running multimodal embedding")
            outputs = backend.embed(payload)
            logger.info(
                f"{uniprot_id}: Multimodal embedding complete - "
                f"shape={outputs['sequence_structure']['per_residue'].shape}"
            )
        except (MultimodalBackendError, RuntimeError) as exc:
            logger.warning(f"{uniprot_id}: Embedding failed, falling back - {exc}")
            return self._fallback_to_sequence(uniprot_id, sequence, error=str(exc))

        metadata = self._build_metadata(payload)
        metadata["multimodal_tracks"] = {
            name: {
                "per_residue_shape": value["per_residue"].shape,
                "has_mean": value["mean"] is not None,
            }
            for name, value in outputs.items()
        }

        primary = outputs["sequence_structure"]
        mean_embedding = primary["mean"]
        if mean_embedding is not None:
            metadata["mean_embedding"] = mean_embedding.squeeze().tolist()

        return EmbeddingResult(
            uniprot_id=uniprot_id,
            embedding=primary["per_residue"].squeeze(),
            metadata=metadata,
        )

    def _require_backend(self) -> LocalMultimodalBackend:
        if self._backend is None:
            logger.info(
                f"Loading multimodal model: {self.config.model_name} "
                f"on device {self.config.resolved_device()}"
            )
            model = _load_local_model(
                self.config.model_name, device=self.config.resolved_device()
            )
            self._backend = self._backend_factory(model)
            logger.info("Multimodal backend ready")
        return self._backend

    def _fallback_to_sequence(
        self, uniprot_id: str, sequence: str, *, error: str | None = None
    ) -> EmbeddingResult:
        logger.debug(f"{uniprot_id}: Using sequence-only fallback")
        fallback = self._sequence_fallback
        if fallback is None:
            fallback = SequenceEmbedder(self.config)
            self._sequence_fallback = fallback

        result = fallback.embed_one(uniprot_id, sequence)
        if error is not None:
            result.metadata["fallback_reason"] = error
        result.metadata.setdefault("pipeline", "sequence")
        logger.debug(f"{uniprot_id}: Sequence-only embedding complete")
        return result

    def _sequence_error(
        self, uniprot_id: str, sequence: str, error: str
    ) -> EmbeddingResult:
        return EmbeddingResult(
            uniprot_id=uniprot_id,
            embedding=None,
            metadata={
                "pipeline": "multimodal",
                "model_name": self.config.model_name,
                "device": self.config.device,
                "sequence_length": len(sequence),
            },
            error=error,
        )

    def _build_metadata(self, payload: MultimodalStructurePayload) -> dict:
        metadata = {
            "pipeline": "multimodal",
            "model_name": self.config.model_name,
            "device": self.config.device,
            "sequence_length": len(payload.sequence),
            "residue_count": payload.coordinates.residue_count,
            "mask_complete": payload.mask.all_valid(),
            "coordinate_source": str(payload.coordinates.source)
            if payload.coordinates.source
            else None,
            "structure_coverage": float(payload.mask.values.mean()),
        }
        metadata.update(payload.metadata)
        return metadata
