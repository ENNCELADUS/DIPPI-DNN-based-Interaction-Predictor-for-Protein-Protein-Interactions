"""Tests for src.embed.io.filesystem."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.embed.core.types import EmbeddingResult
from src.embed.io.filesystem import load_sequences, save_results


def test_load_sequences_from_json_mapping(tmp_path: Path) -> None:
    payload = {"A": "ACD", "B": "GH"}
    path = tmp_path / "seqs.json"
    path.write_text(json.dumps(payload))

    sequences = load_sequences(path)

    assert sequences == payload


def test_load_sequences_rejects_unknown_suffix(tmp_path: Path) -> None:
    path = tmp_path / "seqs.dat"
    path.write_text("dummy")

    with pytest.raises(ValueError):
        load_sequences(path)


def test_load_sequences_from_csv_defaults(tmp_path: Path) -> None:
    path = tmp_path / "seqs.csv"
    path.write_text("uniprotID,sequence,notes\nA,ACD,\nB,GH,sample\n")

    sequences = load_sequences(path)

    assert sequences == {"A": "ACD", "B": "GH"}


def test_load_sequences_from_csv_custom_columns(tmp_path: Path) -> None:
    path = tmp_path / "seqs.csv"
    path.write_text("custom_id,seq\nX1,AC\n")

    sequences = load_sequences(
        path,
        input_format="csv",
        csv_id_column="custom_id",
        csv_sequence_column="seq",
    )

    assert sequences == {"X1": "AC"}


def test_load_sequences_accepts_json_list(tmp_path: Path) -> None:
    payload = [{"id": "A", "sequence": "AC"}]
    path = tmp_path / "seqs.json"
    path.write_text(json.dumps(payload))

    sequences = load_sequences(path, input_format="json")

    assert sequences == {"A": "AC"}


def test_load_sequences_raises_on_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "seqs.json"
    path.write_text(json.dumps(["not-a-mapping"]))

    with pytest.raises(ValueError):
        load_sequences(path, input_format="json")


def test_load_sequences_requires_id_and_sequence(tmp_path: Path) -> None:
    path = tmp_path / "seqs.json"
    path.write_text(json.dumps([{"id": "A"}]))

    with pytest.raises(ValueError):
        load_sequences(path, input_format="json")


def test_load_sequences_rejects_unsupported_json_type(tmp_path: Path) -> None:
    path = tmp_path / "seqs.json"
    path.write_text(json.dumps("AC"))

    with pytest.raises(ValueError):
        load_sequences(path, input_format="json")


def test_load_sequences_allows_blank_lines_and_appends_sequence(tmp_path: Path) -> None:
    path = tmp_path / "seqs.fasta"
    path.write_text(">A\nAC\n\nTG\n")

    sequences = load_sequences(path)

    assert sequences == {"A": "ACTG"}


def test_load_sequences_raises_when_sequence_before_header(tmp_path: Path) -> None:
    path = tmp_path / "seqs.fasta"
    path.write_text("AC\n>A\nTG\n")

    with pytest.raises(ValueError):
        load_sequences(path)


def test_load_sequences_rejects_unknown_format_keyword(tmp_path: Path) -> None:
    path = tmp_path / "seqs.fasta"
    path.write_text(">A\nAC\n")

    with pytest.raises(ValueError):
        load_sequences(path, input_format="yaml")


def test_save_results_writes_npz_and_errors(tmp_path: Path) -> None:
    ok = EmbeddingResult(
        uniprot_id="A",
        embedding=np.ones((1, 2, 3), dtype=np.float32),
        metadata={
            "original_sequence": "ACD",
            "cleaned_sequence": "ACD",
            "source": "stub",
        },
    )
    fail = EmbeddingResult(
        uniprot_id="B",
        embedding=None,
        metadata={"original_sequence": "BAD"},
        error="invalid",
    )

    output = tmp_path / "results.npz"
    save_results([ok, fail], output)

    with np.load(output, allow_pickle=True) as data:
        assert data["ids"].tolist() == ["A"]
        np.testing.assert_allclose(
            np.asarray(data["embeddings"][0]), np.ones((1, 2, 3))
        )
        assert data["original_sequences"].tolist() == ["ACD"]
        assert data["cleaned_sequences"].tolist() == ["ACD"]
        assert data["metadata"].tolist() == [{"source": "stub"}]

    error_path = output.with_suffix(".npz.errors.json")
    errors = json.loads(error_path.read_text())
    assert errors == [
        {
            "uniprot_id": "B",
            "error": "invalid",
            "metadata": {"original_sequence": "BAD"},
        }
    ]
