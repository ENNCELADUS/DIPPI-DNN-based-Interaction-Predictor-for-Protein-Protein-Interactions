"""Tests for src.embed.core.validation."""

from __future__ import annotations

import math

import pytest

from src.embed.core.validation import (
    EXTENDED_AMINO_ACIDS,
    calculate_sequence_stats,
    clean_protein_sequence,
    count_invalid_characters,
    ensure_sequence,
    validate_protein_sequence,
)


def test_validate_protein_sequence_accepts_standard() -> None:
    assert validate_protein_sequence("ACDEFGHIKLMNPQRSTVWY")


def test_validate_protein_sequence_rejects_unknown_characters() -> None:
    assert not validate_protein_sequence("ABCD1")
    assert not validate_protein_sequence("ACD", allow_extended=False, max_length=2)


def test_validate_protein_sequence_respects_max_length() -> None:
    assert not validate_protein_sequence("AAAA", max_length=3)
    assert not validate_protein_sequence("A", min_length=2)


def test_ensure_sequence_raises_on_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        ensure_sequence("")

    with pytest.raises(ValueError):
        ensure_sequence("ACDZ", allow_extended=False)


def test_clean_protein_sequence_removes_gaps_and_stop() -> None:
    cleaned = clean_protein_sequence("ac-d*.")
    assert cleaned == "ACD"


def test_clean_protein_sequence_respects_flags() -> None:
    cleaned = clean_protein_sequence(
        "ac-d*.", remove_gaps=False, remove_stop=False, to_upper=False
    )
    assert cleaned == "ac-d*."


def test_clean_protein_sequence_handles_empty_input() -> None:
    assert clean_protein_sequence("") == ""


def test_calculate_sequence_stats_returns_expected_values() -> None:
    stats = calculate_sequence_stats("ACDA")

    assert stats["length"] == 4
    assert stats["composition"]["A"]["count"] == 2
    assert math.isclose(stats["composition"]["A"]["frequency"], 0.5)
    expected_weight = (2 * 89.09) + 121.15 + 133.10
    assert math.isclose(stats["molecular_weight"], expected_weight, rel_tol=1e-6)


def test_calculate_sequence_stats_handles_empty_string() -> None:
    stats = calculate_sequence_stats("")

    assert stats == {"length": 0, "composition": {}, "molecular_weight": 0.0}


def test_count_invalid_characters_uses_supplied_allowlist() -> None:
    allowed = {"A", "C"}
    assert count_invalid_characters("ACX", allowed) == 1
    assert count_invalid_characters("ACX", EXTENDED_AMINO_ACIDS) == 0


def test_count_invalid_characters_defaults_to_extended_set() -> None:
    assert count_invalid_characters("ACX") == 0
    assert count_invalid_characters("") == 0
