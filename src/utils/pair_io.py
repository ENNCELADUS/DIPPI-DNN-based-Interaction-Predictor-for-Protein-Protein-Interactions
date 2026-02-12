"""Pair-file parsing helpers for PPI datasets."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

_PAIR_A_COLUMNS = frozenset(
    {
        "a",
        "ida",
        "proteina",
        "protein1",
        "protein_a",
        "uniprot_a",
        "uniprotid_a",
    }
)
_PAIR_B_COLUMNS = frozenset(
    {
        "b",
        "idb",
        "proteinb",
        "protein2",
        "protein_b",
        "uniprot_b",
        "uniprotid_b",
    }
)
_LABEL_COLUMNS = frozenset({"label", "isinteraction", "interaction", "target", "y"})


@dataclass(frozen=True)
class PairRecord:
    """One protein-pair interaction record.

    Attributes:
        protein_a: First protein identifier.
        protein_b: Second protein identifier.
        label: Binary interaction label.
    """

    protein_a: str
    protein_b: str
    label: int


def _normalize_column(name: str) -> str:
    """Normalize a column name for resilient matching."""
    cleaned = name.strip().lower()
    return "".join(character for character in cleaned if character.isalnum())


def _infer_delimiter(file_path: Path) -> str:
    """Infer pair-file delimiter from content and suffix."""
    default_delimiter = "," if file_path.suffix.lower() == ".csv" else "\t"
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            comma_count = stripped.count(",")
            tab_count = stripped.count("\t")
            if comma_count and not tab_count:
                return ","
            if tab_count and not comma_count:
                return "\t"
            if comma_count and tab_count:
                return "," if comma_count >= tab_count else "\t"
            break
    return default_delimiter


def _find_index(
    normalized: Sequence[str], candidates: set[str] | frozenset[str]
) -> int | None:
    """Find first matching column index from candidate names."""
    for index, value in enumerate(normalized):
        if value in candidates:
            return index
    return None


def _header_indices(row: Sequence[str]) -> tuple[int, int, int | None] | None:
    """Resolve pair columns from a header row."""
    normalized = [_normalize_column(value) for value in row]
    protein_a_index = _find_index(normalized, _PAIR_A_COLUMNS)
    protein_b_index = _find_index(normalized, _PAIR_B_COLUMNS)
    if protein_a_index is None or protein_b_index is None:
        return None
    label_index = _find_index(normalized, _LABEL_COLUMNS)
    return protein_a_index, protein_b_index, label_index


def _parse_label(raw_value: str | None, default_label: int) -> int | None:
    """Parse binary label value."""
    if raw_value is None:
        return default_label
    stripped = raw_value.strip()
    if not stripped:
        return default_label
    try:
        return int(float(stripped))
    except ValueError:
        return None


def _row_value(row: Sequence[str], index: int | None) -> str | None:
    """Read one cell value by optional index."""
    if index is None:
        return None
    if index >= len(row):
        return None
    return row[index]


def _iter_rows(file_path: Path, delimiter: str) -> Iterator[list[str]]:
    """Yield normalized CSV rows from one pair file."""
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            stripped_row = [value.strip() for value in row]
            if not stripped_row:
                continue
            if all(not value for value in stripped_row):
                continue
            yield stripped_row


def iter_pair_records(file_path: Path, default_label: int = 1) -> Iterator[PairRecord]:
    """Iterate parsed pair records from CSV/TSV files.

    Args:
        file_path: Input pair file path.
        default_label: Label used when label column is absent or empty.

    Yields:
        Parsed ``PairRecord`` entries.
    """
    delimiter = _infer_delimiter(file_path)
    row_iterator = _iter_rows(file_path=file_path, delimiter=delimiter)

    header_lookup: tuple[int, int, int | None] | None = None
    for row_index, row in enumerate(row_iterator):
        if row_index == 0:
            header_lookup = _header_indices(row)
            if header_lookup is not None:
                continue

        if header_lookup is None:
            protein_a = _row_value(row, 0) or ""
            protein_b = _row_value(row, 1) or ""
            raw_label = _row_value(row, 2)
        else:
            protein_a = _row_value(row, header_lookup[0]) or ""
            protein_b = _row_value(row, header_lookup[1]) or ""
            raw_label = _row_value(row, header_lookup[2])

        if not protein_a or not protein_b:
            continue
        parsed_label = _parse_label(raw_label, default_label=default_label)
        if parsed_label is None:
            continue
        yield PairRecord(
            protein_a=protein_a,
            protein_b=protein_b,
            label=parsed_label,
        )


def read_pair_records(file_path: Path, default_label: int = 1) -> list[PairRecord]:
    """Read all pair records from one file.

    Args:
        file_path: Input pair file path.
        default_label: Label used when label column is absent or empty.

    Returns:
        Parsed pair records.

    Raises:
        ValueError: If no valid pair records are found.
    """
    records = list(iter_pair_records(file_path=file_path, default_label=default_label))
    if records:
        return records
    raise ValueError(f"No valid PPI records found in {file_path}")


def collect_required_protein_ids(split_paths: Iterable[Path]) -> set[str]:
    """Collect unique protein IDs required by split files."""
    required_ids: set[str] = set()
    for split_path in split_paths:
        if not split_path.exists():
            raise FileNotFoundError(f"Split dataset path not found: {split_path}")
        for record in iter_pair_records(file_path=split_path, default_label=1):
            required_ids.add(record.protein_a)
            required_ids.add(record.protein_b)
    if required_ids:
        return required_ids
    raise ValueError("No protein IDs found in configured split files")
