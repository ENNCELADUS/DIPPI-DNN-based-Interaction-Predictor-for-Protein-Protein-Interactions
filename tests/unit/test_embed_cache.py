"""Unit tests for embedding cache preparation and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import src.embed.embed as embed_module
import torch
from src.embed import ensure_embeddings_ready
from src.utils.config import ConfigDict


def _write_split(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for protein_a, protein_b, label in rows:
            handle.write(f"{protein_a}\t{protein_b}\t{label}\n")


def _base_config(cache_dir: Path, processed_dir: Path) -> ConfigDict:
    return {
        "data_config": {
            "benchmark": {
                "root_dir": str(processed_dir),
                "processed_dir": str(processed_dir),
            },
            "embeddings": {
                "source": "esm3",
                "cache_dir": str(cache_dir),
                "model_name": "esm3_sm_open_v1",
                "device": "cpu",
            },
        }
    }


def _patch_fake_generator(
    monkeypatch: pytest.MonkeyPatch,
    captured_sequences: dict[str, str],
) -> dict[str, int]:
    call_counter = {"count": 0}

    def _fake_generate_missing_embeddings(
        missing_ids: set[str],
        sequences: dict[str, str],
        settings: object,
        cache_dir: Path,
        input_dim: int,
        max_sequence_length: int,
    ) -> dict[str, str]:
        del settings
        call_counter["count"] += 1
        generated_index_updates: dict[str, str] = {}
        for protein_id in sorted(missing_ids):
            captured_sequences[protein_id] = sequences[protein_id]
            seq_len = min(len(sequences[protein_id]), max_sequence_length)
            embedding = torch.full(
                (seq_len, input_dim),
                fill_value=float(seq_len),
                dtype=torch.float32,
            )
            relative_path = embed_module._embedding_relative_path(protein_id)
            output_path = cache_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(embedding, output_path)
            generated_index_updates[protein_id] = relative_path
        return generated_index_updates

    monkeypatch.setattr(
        embed_module,
        "_generate_missing_embeddings",
        _fake_generate_missing_embeddings,
    )
    return call_counter


def test_ensure_embeddings_ready_prefers_csv_over_fasta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_path = tmp_path / "train.txt"
    _write_split(split_path, [("P1", "P2", 1)])

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    csv_path = processed_dir / "proteins.csv"
    csv_path.write_text("uniprot_id,sequence\nP1,AAAA\nP2,CCCC\n", encoding="utf-8")
    fasta_path = processed_dir / "proteins.fasta"
    fasta_path.write_text(
        ">sp|P1|PROT1\nGGGG\n>sp|P2|PROT2\nTTTT\n",
        encoding="utf-8",
    )

    cache_dir = tmp_path / "cache"
    config = _base_config(cache_dir=cache_dir, processed_dir=processed_dir)
    captured_sequences: dict[str, str] = {}
    _patch_fake_generator(monkeypatch=monkeypatch, captured_sequences=captured_sequences)

    ensure_embeddings_ready(
        config=config,
        split_paths=[split_path],
        input_dim=4,
        max_sequence_length=8,
        allow_generation=True,
    )

    assert captured_sequences == {"P1": "AAAA", "P2": "CCCC"}


def test_ensure_embeddings_ready_falls_back_to_fasta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_path = tmp_path / "train.txt"
    _write_split(split_path, [("Q1", "Q2", 1)])

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = processed_dir / "proteins.fasta"
    fasta_path.write_text(
        ">sp|Q1|PROT1\nACDE\n>sp|Q2|PROT2\nFGHI\n",
        encoding="utf-8",
    )

    cache_dir = tmp_path / "cache"
    config = _base_config(cache_dir=cache_dir, processed_dir=processed_dir)
    captured_sequences: dict[str, str] = {}
    _patch_fake_generator(monkeypatch=monkeypatch, captured_sequences=captured_sequences)

    ensure_embeddings_ready(
        config=config,
        split_paths=[split_path],
        input_dim=4,
        max_sequence_length=8,
        allow_generation=True,
    )

    assert captured_sequences == {"Q1": "ACDE", "Q2": "FGHI"}


def test_ensure_embeddings_ready_incremental_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_path = tmp_path / "train.txt"
    _write_split(split_path, [("R1", "R2", 1)])

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    csv_path = processed_dir / "proteins.csv"
    csv_path.write_text("uniprot_id,sequence\nR1,AAAA\nR2,CCCC\n", encoding="utf-8")

    cache_dir = tmp_path / "cache"
    config = _base_config(cache_dir=cache_dir, processed_dir=processed_dir)
    captured_sequences: dict[str, str] = {}
    call_counter = _patch_fake_generator(
        monkeypatch=monkeypatch,
        captured_sequences=captured_sequences,
    )

    ensure_embeddings_ready(
        config=config,
        split_paths=[split_path],
        input_dim=4,
        max_sequence_length=8,
        allow_generation=True,
    )
    ensure_embeddings_ready(
        config=config,
        split_paths=[split_path],
        input_dim=4,
        max_sequence_length=8,
        allow_generation=True,
    )

    assert call_counter["count"] == 1


def test_ensure_embeddings_ready_missing_cache_when_generation_disabled(tmp_path: Path) -> None:
    split_path = tmp_path / "train.txt"
    _write_split(split_path, [("S1", "S2", 1)])

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = tmp_path / "cache"
    config = _base_config(cache_dir=cache_dir, processed_dir=processed_dir)

    with pytest.raises(FileNotFoundError, match="Missing embeddings"):
        ensure_embeddings_ready(
            config=config,
            split_paths=[split_path],
            input_dim=4,
            max_sequence_length=8,
            allow_generation=False,
        )


def test_ensure_embeddings_ready_invalid_dim_when_generation_disabled(tmp_path: Path) -> None:
    split_path = tmp_path / "train.txt"
    _write_split(split_path, [("T1", "T1", 1)])

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = tmp_path / "cache"
    embeddings_dir = cache_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    relative_path = embed_module._embedding_relative_path("T1")
    embedding_path = cache_dir / relative_path
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.ones((3, 3), dtype=torch.float32), embedding_path)

    index_payload = {"T1": relative_path}
    metadata_payload = {
        "schema_version": 1,
        "source": "esm3",
        "model_name": "esm3_sm_open_v1",
        "input_dim": 4,
        "max_sequence_length": 8,
        "format": "torch_pt_per_protein",
    }
    (cache_dir / "index.json").write_text(json.dumps(index_payload), encoding="utf-8")
    (cache_dir / "metadata.json").write_text(json.dumps(metadata_payload), encoding="utf-8")

    config = _base_config(cache_dir=cache_dir, processed_dir=processed_dir)
    with pytest.raises(ValueError, match="Invalid embeddings"):
        ensure_embeddings_ready(
            config=config,
            split_paths=[split_path],
            input_dim=4,
            max_sequence_length=8,
            allow_generation=False,
        )
