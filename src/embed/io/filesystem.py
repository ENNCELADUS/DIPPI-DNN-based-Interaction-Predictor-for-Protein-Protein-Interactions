"""Small helpers for reading and writing embed data."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np

from ..core.types import EmbeddingResult


def load_sequences(
    input_path: Path,
    *,
    input_format: str | None = None,
    csv_id_column: str = "uniprotID",
    csv_sequence_column: str = "sequence",
) -> Dict[str, str]:
    """Load sequences from ``input_path``.

    Supported formats:
    - ``fasta``: simple two-line-per-entry FASTA variant
    - ``json``: mapping ``{"id": "SEQUENCE"}`` or list of objects with
      ``{"id": ..., "sequence": ...}``
    """

    fmt = (input_format or "auto").lower()
    if fmt == "auto":
        fmt = _detect_input_format(input_path)

    if fmt == "fasta":
        return _load_fasta(input_path)
    if fmt == "json":
        return _load_json(input_path)
    if fmt == "csv":
        return _load_csv(
            input_path, id_column=csv_id_column, sequence_column=csv_sequence_column
        )
    raise ValueError(f"Unsupported input format: {fmt}")


def save_results(results: Iterable[EmbeddingResult], output_path: Path) -> None:
    """Persist embedding results as compressed NPZ plus error sidecar."""

    successes: list[EmbeddingResult] = []
    failures: list[EmbeddingResult] = []

    for result in results:
        (successes if result.ok else failures).append(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ids: list[str] = []
    embeddings: list[np.ndarray] = []
    metadata_payloads: list[dict] = []
    original_sequences: list[str] = []
    cleaned_sequences: list[str] = []

    for result in successes:
        ids.append(result.uniprot_id)
        embeddings.append(_ensure_numpy(result.embedding))

        meta = dict(result.metadata)
        original_sequences.append(meta.pop("original_sequence", ""))
        cleaned_sequences.append(meta.pop("cleaned_sequence", ""))
        metadata_payloads.append(meta)

    np.savez_compressed(
        output_path,
        ids=_to_object_array(ids),
        embeddings=_to_object_array(embeddings),
        metadata=_to_object_array(metadata_payloads),
        original_sequences=_to_object_array(original_sequences),
        cleaned_sequences=_to_object_array(cleaned_sequences),
    )

    if failures:
        errors = [
            {
                "uniprot_id": failure.uniprot_id,
                "error": failure.error,
                "metadata": failure.metadata,
            }
            for failure in failures
        ]
        error_path = output_path.with_suffix(output_path.suffix + ".errors.json")
        error_path.write_text(json.dumps(errors, indent=2))


def _ensure_numpy(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return np.asarray(value)
    if value is None:
        raise ValueError("Cannot serialize None embedding")
    return np.asarray(value)


def _to_object_array(items: Sequence[object]) -> np.ndarray:
    array = np.empty(len(items), dtype=object)
    for idx, item in enumerate(items):
        array[idx] = item
    return array


def _detect_input_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".fa", ".fasta", ".faa"}:
        return "fasta"
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    raise ValueError(f"Cannot infer input format from suffix {suffix!r}")


def _load_fasta(fasta_path: Path) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    current_id: str | None = None

    for line in fasta_path.read_text().splitlines():
        if not line:
            continue
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            current_id = line[1:].strip()
            sequences[current_id] = ""
            continue
        if current_id is None:
            raise ValueError("Sequence line encountered before any header")
        sequences[current_id] += line.strip()

    return sequences


def _load_json(json_path: Path) -> Dict[str, str]:
    data = json.loads(json_path.read_text())

    if isinstance(data, Mapping):
        return {str(key): str(value) for key, value in data.items()}

    if isinstance(data, list):
        sequences: Dict[str, str] = {}
        for entry in data:
            if not isinstance(entry, Mapping):
                raise ValueError("JSON sequence entries must be objects")
            try:
                identifier = entry["id"]
                sequence = entry["sequence"]
            except KeyError as exc:
                raise ValueError("JSON entries require 'id' and 'sequence'") from exc
            sequences[str(identifier)] = str(sequence)
        return sequences

    raise ValueError("Unsupported JSON structure for sequences")


def _load_csv(
    csv_path: Path, *, id_column: str, sequence_column: str
) -> Dict[str, str]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV file is missing a header row")
        missing_columns = [
            column
            for column in (id_column, sequence_column)
            if column not in reader.fieldnames
        ]
        if missing_columns:
            raise ValueError(
                "CSV file missing required column(s): "
                + ", ".join(sorted(missing_columns))
            )

        sequences: Dict[str, str] = {}
        for row_number, row in enumerate(reader, start=2):  # account for header
            identifier = row.get(id_column)
            sequence = row.get(sequence_column)
            if identifier is None or sequence is None:
                raise ValueError(
                    f"CSV row {row_number} missing identifier or sequence value"
                )
            identifier = str(identifier).strip()
            if not identifier:
                raise ValueError(f"CSV row {row_number} has empty identifier")
            sequences[identifier] = str(sequence).strip()
    return sequences
