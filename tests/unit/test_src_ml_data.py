"""Unit tests for classical ML feature loading from DL embedding caches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import src.embed.embed as embed_module
import torch

from src_ml.data import load_ml_features


def _write_pairs_csv(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("uniprotID_A,uniprotID_B,isInteraction\n")
        for protein_a, protein_b, label in rows:
            handle.write(f"{protein_a},{protein_b},{label}\n")


def _write_embedding_cache(
    cache_dir: Path,
    embeddings: dict[str, torch.Tensor],
    input_dim: int,
    max_sequence_length: int,
) -> None:
    index: dict[str, str] = {}
    for protein_id, tensor in embeddings.items():
        relative_path = embed_module._embedding_relative_path(protein_id)
        output_path = cache_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, output_path)
        index[protein_id] = relative_path

    metadata = {
        "schema_version": 1,
        "source": "esm3",
        "model_name": "esm3_sm_open_v1",
        "input_dim": input_dim,
        "max_sequence_length": max_sequence_length,
        "format": "torch_pt_per_protein",
    }
    (cache_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")
    (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_load_ml_features_supports_dl_embedding_cache(tmp_path: Path) -> None:
    pairs_path = tmp_path / "finetune_train_tppni.csv"
    _write_pairs_csv(
        pairs_path,
        [
            ("P1", "P2", 1),
            ("P3", "P1", 0),
        ],
    )

    cache_dir = tmp_path / "embeddings"
    _write_embedding_cache(
        cache_dir=cache_dir,
        embeddings={
            "P1": torch.tensor([[1.0, 3.0], [3.0, 5.0]], dtype=torch.float32),
            "P2": torch.tensor([[2.0, 4.0], [6.0, 8.0]], dtype=torch.float32),
            "P3": torch.tensor([[10.0, 20.0]], dtype=torch.float32),
        },
        input_dim=2,
        max_sequence_length=8,
    )

    features, labels = load_ml_features(
        csv_path=str(pairs_path),
        embeddings_path=str(cache_dir),
        pooling="mean",
    )

    expected = np.array(
        [
            [2.0, 4.0, 4.0, 6.0],
            [10.0, 20.0, 2.0, 4.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(features, expected)
    np.testing.assert_array_equal(labels, np.array([1, 0], dtype=np.int64))
