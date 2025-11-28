"""IO helpers for multimodal structure assets."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from ..core.types import (
    DEFAULT_ATOM_ORDER,
    BackboneCoordinateSet,
    MultimodalStructurePayload,
    ResidueMask,
)


class StructureDataNotFoundError(FileNotFoundError):
    """Raised when expected structure artefacts are missing."""


def _latest_file(directory: Path, pattern: str) -> Path:
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        raise StructureDataNotFoundError(
            f"No files matching pattern {pattern!r} under {directory}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _maybe_latest(directory: Path, pattern: str) -> Optional[Path]:
    try:
        return _latest_file(directory, pattern)
    except StructureDataNotFoundError:
        return None


def _parse_int(value: Any) -> Optional[int]:
    """Best-effort conversion of ``value`` to ``int``.

    Returns ``None`` when conversion fails.
    """

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalise_sequence(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _trim_sequence_candidate(
    sequence: str,
    target_length: int,
    metadata: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Attempt to trim ``sequence`` to ``target_length`` using metadata hints.

    Returns alignment metadata when trimming succeeds, ``None`` otherwise.
    """

    if len(sequence) < target_length:
        return None

    if len(sequence) == target_length:
        return {
            "aligned_sequence": sequence,
            "aligned_sequence_trimmed": False,
            "aligned_sequence_start_trim": 0,
            "aligned_sequence_end_trim": 0,
        }

    # Attempt to honour explicit residue ranges when available.
    range_hints = [
        ("structure_sequence_start", "structure_sequence_end"),
        ("sequence_start", "sequence_end"),
        ("aligned_start", "aligned_end"),
        ("residue_start", "residue_end"),
        ("start_residue", "end_residue"),
    ]

    for start_key, end_key in range_hints:
        start_val = _parse_int(metadata.get(start_key))
        end_val = _parse_int(metadata.get(end_key))
        if start_val is None or end_val is None:
            continue

        # Interpret hints as 1-based inclusive positions first.
        start_idx = max(start_val - 1, 0)
        end_idx_inclusive = max(end_val - 1, -1)
        if end_idx_inclusive >= start_idx:
            slice_len = end_idx_inclusive - start_idx + 1
            if slice_len == target_length and end_idx_inclusive + 1 <= len(sequence):
                trimmed = sequence[start_idx : end_idx_inclusive + 1]
                return {
                    "aligned_sequence": trimmed,
                    "aligned_sequence_trimmed": True,
                    "aligned_sequence_start_trim": start_idx,
                    "aligned_sequence_end_trim": len(sequence)
                    - (end_idx_inclusive + 1),
                }

        # Fall back to 0-based exclusive interpretation.
        start_idx = max(start_val, 0)
        end_idx_exclusive = max(end_val, 0)
        slice_len = end_idx_exclusive - start_idx
        if slice_len == target_length and end_idx_exclusive <= len(sequence):
            trimmed = sequence[start_idx:end_idx_exclusive]
            return {
                "aligned_sequence": trimmed,
                "aligned_sequence_trimmed": True,
                "aligned_sequence_start_trim": start_idx,
                "aligned_sequence_end_trim": len(sequence) - end_idx_exclusive,
            }

    # Distribute trimming using inferred leading offset if present.
    leading_keys = (
        "structure_sequence_start",
        "sequence_start",
        "aligned_start",
        "residue_start",
        "start_residue",
    )
    leading_offset = None
    for key in leading_keys:
        parsed = _parse_int(metadata.get(key))
        if parsed is not None:
            leading_offset = max(parsed - 1, 0)
            break

    sequence_length = len(sequence)
    diff = sequence_length - target_length
    if leading_offset is None:
        trimmed = sequence[:target_length]
        if len(trimmed) == target_length:
            return {
                "aligned_sequence": trimmed,
                "aligned_sequence_trimmed": True,
                "aligned_sequence_start_trim": 0,
                "aligned_sequence_end_trim": diff,
                "aligned_sequence_warning": "trimmed_tail_without_offset_metadata",
            }
    else:
        start_trim = min(leading_offset, diff)
        end_trim = diff - start_trim
        trimmed = sequence[start_trim : start_trim + target_length]
        if len(trimmed) == target_length:
            return {
                "aligned_sequence": trimmed,
                "aligned_sequence_trimmed": True,
                "aligned_sequence_start_trim": start_trim,
                "aligned_sequence_end_trim": end_trim,
            }

    # Final fallback: trim from the tail and record warning.
    trimmed = sequence[:target_length]
    if len(trimmed) == target_length:
        return {
            "aligned_sequence": trimmed,
            "aligned_sequence_trimmed": True,
            "aligned_sequence_start_trim": 0,
            "aligned_sequence_end_trim": sequence_length - target_length,
            "aligned_sequence_warning": "trimmed_tail_without_offset_metadata",
        }

    return None


def _resolve_aligned_sequence(
    coordinates: BackboneCoordinateSet,
    provided_sequence: Optional[str],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    residue_count = coordinates.residue_count

    candidate_sources = [
        ("provided_sequence", provided_sequence),
        ("metadata.structure_sequence", metadata.get("structure_sequence")),
        ("metadata.aligned_sequence", metadata.get("aligned_sequence")),
        ("metadata.sequence", metadata.get("sequence")),
        ("metadata.original_sequence", metadata.get("original_sequence")),
    ]

    candidates: list[tuple[str, str]] = []
    seen_sequences: set[str] = set()
    for source, candidate in candidate_sources:
        normalised = _normalise_sequence(candidate)
        if normalised is None:
            continue
        if normalised in seen_sequences:
            continue
        seen_sequences.add(normalised)
        candidates.append((source, normalised))

    if not candidates:
        raise ValueError(
            "Sequence is required; provide one or ensure metadata exposes a usable field"
        )

    # Prefer exact matches first.
    for source, candidate in candidates:
        if len(candidate) == residue_count:
            alignment_meta = {
                "aligned_sequence": candidate,
                "aligned_sequence_source": source,
                "aligned_sequence_length": residue_count,
                "aligned_sequence_source_length": len(candidate),
                "aligned_sequence_trimmed": False,
                "aligned_sequence_start_trim": 0,
                "aligned_sequence_end_trim": 0,
            }
            return alignment_meta

    # Attempt to trim candidates down to the required length.
    for source, candidate in candidates:
        trimmed_meta = _trim_sequence_candidate(candidate, residue_count, metadata)
        if trimmed_meta is None:
            continue
        return {
            "aligned_sequence": trimmed_meta["aligned_sequence"],
            "aligned_sequence_source": source,
            "aligned_sequence_length": residue_count,
            "aligned_sequence_source_length": len(candidate),
            "aligned_sequence_trimmed": trimmed_meta.get(
                "aligned_sequence_trimmed", False
            ),
            "aligned_sequence_start_trim": trimmed_meta.get(
                "aligned_sequence_start_trim", 0
            ),
            "aligned_sequence_end_trim": trimmed_meta.get(
                "aligned_sequence_end_trim", 0
            ),
            **{
                key: value
                for key, value in trimmed_meta.items()
                if key
                not in {
                    "aligned_sequence",
                    "aligned_sequence_trimmed",
                    "aligned_sequence_start_trim",
                    "aligned_sequence_end_trim",
                }
            },
        }

    candidate_lengths = ", ".join(str(len(seq)) for _, seq in candidates)
    raise ValueError(
        "Could not reconcile sequence length with coordinates: "
        f"residue count {residue_count}, candidate lengths {candidate_lengths}"
    )


def load_backbone_coordinate_set(
    data_root: Path,
    uniprot_id: str,
    *,
    atom_order: Sequence[str] = DEFAULT_ATOM_ORDER,
) -> BackboneCoordinateSet:
    """Load backbone coordinates for ``uniprot_id`` from consolidated NPZ."""

    consolidated_dir = data_root / "consolidated"
    npz_path = _latest_file(consolidated_dir, "backbone_coordinates_*.npz")
    with np.load(npz_path, allow_pickle=True) as payload:  # type: ignore[arg-type]
        if uniprot_id not in payload:
            raise KeyError(f"UniProt ID {uniprot_id!r} not present in {npz_path.name}")
        residues = payload[uniprot_id]

    # ``residues`` is a 1D object array of dicts; ``tolist`` normalises to python
    if isinstance(residues, np.ndarray):
        residue_records = residues.tolist()
    else:
        residue_records = residues

    if not isinstance(residue_records, Iterable):
        raise TypeError("Unexpected coordinate payload format")

    return BackboneCoordinateSet.from_legacy_records(
        residue_records,
        atom_order=atom_order,
        source=npz_path,
    )


def materialize_residue_masks(
    data_root: Path,
    *,
    overwrite: bool = False,
    uniprot_ids: Optional[Sequence[str]] = None,
    atom_order: Sequence[str] = DEFAULT_ATOM_ORDER,
) -> Dict[str, Path]:
    """Persist residue masks for the requested UniProt IDs.

    Returns a mapping of UniProt ID to the mask path on disk. Existing masks are
    skipped unless ``overwrite`` is ``True``.
    """

    consolidated_dir = data_root / "consolidated"
    npz_path = _latest_file(consolidated_dir, "backbone_coordinates_*.npz")
    masks_dir = data_root / "structures" / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Path] = {}
    with np.load(npz_path, allow_pickle=True) as payload:  # type: ignore[arg-type]
        ids_to_process: Iterable[str]
        if uniprot_ids is None:
            ids_to_process = payload.files
        else:
            ids_to_process = uniprot_ids

        for uniprot_id in ids_to_process:
            if uniprot_id not in payload:
                raise KeyError(
                    f"UniProt ID {uniprot_id!r} not present in {npz_path.name}"
                )
            mask_path = masks_dir / f"{uniprot_id}.npy"
            if mask_path.exists() and not overwrite:
                results[uniprot_id] = mask_path
                continue

            residues = payload[uniprot_id]
            residue_records = (
                residues.tolist() if isinstance(residues, np.ndarray) else residues
            )
            coordinate_set = BackboneCoordinateSet.from_legacy_records(
                residue_records,
                atom_order=atom_order,
                source=npz_path,
            )
            mask = ResidueMask.from_coordinates(coordinate_set, source=str(mask_path))
            np.save(mask_path, mask.values)
            results[uniprot_id] = mask_path

    return results


def load_structure_metadata(data_root: Path, uniprot_id: str) -> Dict[str, str]:
    """Return metadata for ``uniprot_id`` from the consolidated tables."""

    consolidated_dir = data_root / "consolidated"
    csv_path = _maybe_latest(consolidated_dir, "esm3_ready_proteins_*.csv")
    if csv_path is not None:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("uniprot_id") == uniprot_id:
                    return row
        raise KeyError(f"UniProt ID {uniprot_id!r} not present in {csv_path.name}")

    pkl_path = _latest_file(consolidated_dir, "esm3_ready_proteins_*.pkl")
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive
        raise RuntimeError("pandas is required to read metadata pickle") from exc

    df = pd.read_pickle(pkl_path)
    if "uniprot_id" not in df.columns:
        raise KeyError("metadata table missing 'uniprot_id' column")
    subset = df[df["uniprot_id"] == uniprot_id]
    if subset.empty:
        raise KeyError(f"UniProt ID {uniprot_id!r} not present in {pkl_path.name}")
    return subset.iloc[0].to_dict()


def load_residue_mask(
    data_root: Path,
    uniprot_id: str,
    coordinates: BackboneCoordinateSet,
    *,
    allow_infer: bool = True,
) -> ResidueMask:
    """Load or derive a residue mask aligned with ``coordinates``."""

    mask_path = data_root / "structures" / "masks" / f"{uniprot_id}.npy"
    if mask_path.exists():
        mask_values = np.load(mask_path)
        mask = ResidueMask(mask_values, source=str(mask_path))
        if mask.length != coordinates.residue_count:
            raise ValueError("mask length does not match coordinates residue count")
        return mask

    if not allow_infer:
        raise StructureDataNotFoundError(
            f"Residue mask for {uniprot_id!r} not found at {mask_path}"
        )

    inferred = ResidueMask.from_coordinates(
        coordinates, source="inferred_from_coordinates"
    )
    return inferred


def load_multimodal_structure(
    data_root: Path,
    uniprot_id: str,
    *,
    sequence: Optional[str] = None,
    allow_mask_infer: bool = True,
) -> MultimodalStructurePayload:
    """Load the multimodal payload for ``uniprot_id`` from disk."""

    coordinates = load_backbone_coordinate_set(data_root, uniprot_id)
    metadata = load_structure_metadata(data_root, uniprot_id)

    alignment = _resolve_aligned_sequence(coordinates, sequence, metadata)
    seq = alignment["aligned_sequence"]
    metadata.update(alignment)
    original_sequence = metadata.get("original_sequence")
    if original_sequence:
        metadata.setdefault("original_sequence_length", len(original_sequence))
        metadata["aligned_sequence_delta_vs_original"] = len(seq) - len(
            original_sequence
        )

    mask = load_residue_mask(
        data_root, uniprot_id, coordinates, allow_infer=allow_mask_infer
    )

    return MultimodalStructurePayload(
        uniprot_id=uniprot_id,
        sequence=seq,
        coordinates=coordinates,
        mask=mask,
        metadata=metadata,
    )


__all__ = [
    "StructureDataNotFoundError",
    "load_backbone_coordinate_set",
    "load_multimodal_structure",
    "load_residue_mask",
    "load_structure_metadata",
    "materialize_residue_masks",
]
