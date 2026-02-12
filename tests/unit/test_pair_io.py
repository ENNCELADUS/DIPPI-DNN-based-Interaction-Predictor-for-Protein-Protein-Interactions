"""Unit tests for pair-file parsing helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.utils.pair_io import (
    collect_required_protein_ids,
    iter_pair_records,
    read_pair_records,
)


def test_read_pair_records_parses_csv_with_header(tmp_path: Path) -> None:
    split_path = tmp_path / "pairs.csv"
    split_path.write_text(
        "uniprotID_A,uniprotID_B,isInteraction\nP1,P2,1\nP3,P4,0\n",
        encoding="utf-8",
    )

    records = read_pair_records(file_path=split_path)
    assert [
        (record.protein_a, record.protein_b, record.label) for record in records
    ] == [
        ("P1", "P2", 1),
        ("P3", "P4", 0),
    ]


def test_iter_pair_records_parses_tsv_without_header(tmp_path: Path) -> None:
    split_path = tmp_path / "pairs.txt"
    split_path.write_text("Q1\tQ2\t1\nQ3\tQ4\t0\n", encoding="utf-8")

    records = list(iter_pair_records(file_path=split_path))
    assert [
        (record.protein_a, record.protein_b, record.label) for record in records
    ] == [
        ("Q1", "Q2", 1),
        ("Q3", "Q4", 0),
    ]


def test_collect_required_protein_ids_supports_csv_splits(tmp_path: Path) -> None:
    first_split = tmp_path / "first.csv"
    second_split = tmp_path / "second.csv"
    first_split.write_text(
        "uniprotID_A,uniprotID_B,isInteraction\nA1,A2,1\n",
        encoding="utf-8",
    )
    second_split.write_text(
        "uniprotID_A,uniprotID_B,isInteraction\nA2,A3,0\n",
        encoding="utf-8",
    )

    required_ids = collect_required_protein_ids(split_paths=[first_split, second_split])
    assert required_ids == {"A1", "A2", "A3"}


def test_read_pair_records_raises_on_empty_or_invalid_split(tmp_path: Path) -> None:
    split_path = tmp_path / "invalid.csv"
    split_path.write_text(
        "uniprotID_A,uniprotID_B,isInteraction\n,\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="No valid PPI records found"):
        read_pair_records(file_path=split_path)
