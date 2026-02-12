"""Sequence-loading utilities for sequence-native model pipelines."""

from __future__ import annotations

import csv
from pathlib import Path

from src.utils.config import ConfigDict, as_str, get_section


def _clean_sequence(sequence: str) -> str:
    """Normalize one protein sequence string."""
    return "".join(sequence.strip().upper().split())


def _resolve_fasta_id(header: str) -> str:
    """Extract a protein ID from a FASTA header line."""
    token = header.split(maxsplit=1)[0].strip()
    parts = token.split("|")
    if len(parts) >= 2 and parts[0] in {"sp", "tr"}:
        return parts[1].strip()
    return token


def _load_from_csv(
    file_path: Path,
    required_ids: set[str],
    id_column: str,
    sequence_column: str,
) -> dict[str, str]:
    delimiter = "\t" if file_path.suffix.lower() == ".tsv" else ","
    sequences: dict[str, str] = {}
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Sequence file has no header: {file_path}")
        normalized = {name.lower(): name for name in reader.fieldnames}
        resolved_id_col = normalized.get(id_column.lower())
        resolved_seq_col = normalized.get(sequence_column.lower())
        if resolved_id_col is None:
            raise ValueError(f"Missing sequence ID column '{id_column}' in {file_path}")
        if resolved_seq_col is None:
            raise ValueError(
                f"Missing sequence column '{sequence_column}' in {file_path}"
            )

        for row in reader:
            protein_id = row.get(resolved_id_col, "").strip()
            if protein_id not in required_ids or protein_id in sequences:
                continue
            sequence = _clean_sequence(row.get(resolved_seq_col, ""))
            if sequence:
                sequences[protein_id] = sequence
            if len(sequences) == len(required_ids):
                break
    return sequences


def _load_from_fasta(file_path: Path, required_ids: set[str]) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current_id: str | None = None
    current_chunks: list[str] = []

    def _flush() -> None:
        nonlocal current_id
        nonlocal current_chunks
        if current_id is None:
            return
        if current_id in required_ids and current_id not in sequences:
            normalized = _clean_sequence("".join(current_chunks))
            if normalized:
                sequences[current_id] = normalized
        current_id = None
        current_chunks = []

    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                _flush()
                current_id = _resolve_fasta_id(line[1:])
                continue
            current_chunks.append(line)
        _flush()
    return sequences


def load_required_sequences(
    config: ConfigDict, required_ids: set[str]
) -> dict[str, str]:
    """Load required protein sequences from configured source file."""
    if not required_ids:
        return {}

    data_cfg = get_section(config, "data_config")
    sequences_cfg = get_section(data_cfg, "sequences")
    source_file = Path(
        as_str(
            sequences_cfg.get("source_file", ""), "data_config.sequences.source_file"
        )
    )
    if not source_file.exists():
        raise FileNotFoundError(f"Sequence file not found: {source_file}")

    id_column = as_str(
        sequences_cfg.get("id_column", "uniprotID"),
        "data_config.sequences.id_column",
    )
    sequence_column = as_str(
        sequences_cfg.get("sequence_column", "sequence"),
        "data_config.sequences.sequence_column",
    )

    suffix = source_file.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sequences = _load_from_csv(
            file_path=source_file,
            required_ids=required_ids,
            id_column=id_column,
            sequence_column=sequence_column,
        )
    elif suffix in {".fasta", ".fa", ".faa"}:
        sequences = _load_from_fasta(file_path=source_file, required_ids=required_ids)
    else:
        raise ValueError(
            "Unsupported sequence file format. Expected csv/tsv/fasta/fa/faa."
        )

    missing_ids = sorted(
        protein_id for protein_id in required_ids if protein_id not in sequences
    )
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        raise FileNotFoundError(
            f"Sequence source is missing required proteins: {preview} "
            f"(missing={len(missing_ids)})"
        )
    return sequences
