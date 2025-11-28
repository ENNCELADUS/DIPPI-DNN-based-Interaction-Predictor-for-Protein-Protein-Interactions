"""Shared dataclasses used across embed pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


FORGE_API_URL_DEFAULT = "https://forge.evolutionaryscale.ai"


def _default_workspace() -> Path:
    return Path.cwd()


def _default_data_root() -> Path:
    return Path.cwd() / "data"


def _default_cache_root() -> Path:
    return Path.cwd() / ".cache"


def _default_model_cache() -> Path:
    return Path.cwd() / ".cache" / "models"


@dataclass(slots=True)
class EmbeddingConfig:
    """Runtime options shared by embed pipelines."""

    model_name: str = "esm3_sm_open_v1"
    model_revision: Optional[str] = None
    use_local_model: bool = True
    forge_api_token: Optional[str] = None
    forge_api_url: str = FORGE_API_URL_DEFAULT
    device: str = "auto"
    batch_size: int = 1
    max_sequence_length: int = 4096
    truncate_long_sequences: bool = False
    timeout_seconds: int = 600
    retry_attempts: int = 3
    workspace: Path = field(default_factory=_default_workspace)
    data_root: Path = field(default_factory=_default_data_root)
    cache_root: Path = field(default_factory=_default_cache_root)
    model_cache_dir: Path = field(default_factory=_default_model_cache)
    extras: Dict[str, Any] = field(default_factory=dict)

    def with_updates(self, **overrides: Any) -> "EmbeddingConfig":
        """Return a copy with updated fields."""

        return replace(self, **overrides)

    @property
    def is_remote(self) -> bool:
        """Whether the pipeline should contact a remote service."""

        return not self.use_local_model

    def requires_remote_token(self) -> bool:
        """Return True if remote execution is configured without credentials."""

        return self.is_remote and not self.forge_api_token

    def resolved_device(self) -> str:
        """Resolve the device string, expanding "auto" when possible."""

        return self.normalize_device(self.device)

    @staticmethod
    def normalize_device(device: str) -> str:
        """Normalize a device string, resolving "auto" to an available backend."""

        if device.lower() != "auto":
            return device

        try:
            import torch  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            return "cpu"

        if torch.cuda.is_available():  # pragma: no cover - hardware dependent
            return "cuda"
        if (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):  # pragma: no cover
            return "mps"
        return "cpu"


@dataclass(slots=True)
class EmbeddingResult:
    """Normalized result returned by pipelines."""

    uniprot_id: str
    embedding: Optional[Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        """Whether the embedding was produced successfully."""
        return self.error is None and self.embedding is not None


DEFAULT_ATOM_ORDER: Tuple[str, ...] = ("N", "CA", "C", "O")


@dataclass(slots=True)
class BackboneCoordinateSet:
    """Canonical representation of backbone coordinates.

    The coordinates are stored as a float32 array with shape ``(L, A, 3)`` where
    ``L`` is the residue count and ``A`` the number of atoms in ``atom_order``.
    Missing atoms should already be imputed with ``np.nan``.
    """

    values: np.ndarray
    atom_order: Tuple[str, ...] = DEFAULT_ATOM_ORDER
    source: Optional[Path] = None

    def __post_init__(self) -> None:
        if not isinstance(self.values, np.ndarray):
            raise TypeError("values must be a numpy.ndarray")
        if self.values.ndim != 3:
            raise ValueError("coordinate array must have shape (L, A, 3)")
        if self.values.shape[1] != len(self.atom_order):
            raise ValueError("atom axis does not match atom_order length")
        if self.values.shape[2] != 3:
            raise ValueError("last axis must contain xyz coordinates")
        if self.values.dtype.kind != "f":
            self.values = self.values.astype(np.float32, copy=False)

    @property
    def residue_count(self) -> int:
        return int(self.values.shape[0])

    def to_numpy(self) -> np.ndarray:
        """Return the underlying array (alias)."""

        return self.values

    @classmethod
    def from_legacy_records(
        cls,
        residues: Sequence[Mapping[str, Sequence[Optional[float]]]],
        *,
        atom_order: Sequence[str] = DEFAULT_ATOM_ORDER,
        source: Optional[Path] = None,
    ) -> "BackboneCoordinateSet":
        """Construct a coordinate set from legacy NPZ payloads."""

        if not residues:
            array = np.zeros((0, len(atom_order), 3), dtype=np.float32)
            return cls(array, tuple(atom_order), source=source)

        array = np.full((len(residues), len(atom_order), 3), np.nan, dtype=np.float32)
        for residue_idx, residue in enumerate(residues):
            for atom_idx, atom in enumerate(atom_order):
                coords = residue.get(atom)
                if coords is None:
                    continue
                try:
                    x, y, z = coords
                except (TypeError, ValueError):
                    continue
                array[residue_idx, atom_idx] = [x, y, z]
        return cls(array, tuple(atom_order), source=source)


@dataclass(slots=True)
class ResidueMask:
    """Boolean mask aligned with a protein sequence."""

    values: np.ndarray
    source: str = "inferred"

    def __post_init__(self) -> None:
        if not isinstance(self.values, np.ndarray):
            raise TypeError("values must be a numpy.ndarray")
        if self.values.ndim != 1:
            raise ValueError("mask must be a 1D array")
        if self.values.dtype != np.bool_:
            self.values = self.values.astype(np.bool_, copy=False)

    @property
    def length(self) -> int:
        return int(self.values.shape[0])

    def all_valid(self) -> bool:
        """Return True when every residue has coordinates."""

        return bool(self.values.all())

    @classmethod
    def from_coordinates(
        cls,
        coordinates: BackboneCoordinateSet,
        *,
        source: str = "inferred",
    ) -> "ResidueMask":
        """Derive a mask from coordinate availability."""

        finite = np.isfinite(coordinates.values)
        mask = finite.all(axis=(1, 2))
        return cls(mask, source=source)


@dataclass(slots=True)
class MultimodalStructurePayload:
    """Bundle of data required for multimodal ESM inputs."""

    uniprot_id: str
    sequence: str
    coordinates: BackboneCoordinateSet
    mask: ResidueMask
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate_alignment()

    def _validate_alignment(self) -> None:
        length = len(self.sequence)
        if self.coordinates.residue_count != length:
            raise ValueError("coordinate count does not match sequence length")
        if self.mask.length != length:
            raise ValueError("mask length does not match sequence length")

    @property
    def has_complete_structure(self) -> bool:
        """Return True if all residues have resolved coordinates."""

        return self.mask.all_valid()
