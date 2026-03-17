"""Feature loading utilities for the classical ML PPI pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from src.embed import load_cached_embedding

# Module-level cache for ML embeddings.
_ML_EMBEDDINGS_CACHE: Optional[object] = None
_ML_EMBEDDINGS_PATH_CACHE: Optional[str] = None

INDEX_FILENAME = "index.json"
METADATA_FILENAME = "metadata.json"


def _is_main_process() -> bool:
    """Return whether this process should emit shared progress logs."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


class _ShardedEmbeddingStore:
    """
    Memory-mapped embedding store backed by sharded ``.npy`` files.

    Expects directory layout:
      - ``manifest.json``
      - ``index.npz`` (ids, shard_idx, row_idx, lengths)
      - ``shard_*.npy`` (shape ``[shard_size, max_len, embedding_dim]``)
    """

    def __init__(self, root: Path) -> None:
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest.json in {root}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "dippi_sharded_embeddings_v1":
            raise ValueError(f"Unsupported embeddings format: {manifest.get('format')}")

        index_file = manifest.get("index_file", "index.npz")
        index_path = root / str(index_file)
        if not index_path.exists():
            raise FileNotFoundError(f"Missing index file: {index_path}")

        index = np.load(index_path, allow_pickle=True, mmap_mode="r")
        ids = index["ids"]
        self._ids = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in ids
        ]
        self._id_to_pos = {protein_id: idx for idx, protein_id in enumerate(self._ids)}
        self._shard_idx = index["shard_idx"].astype(np.int64, copy=False)
        self._row_idx = index["row_idx"].astype(np.int64, copy=False)
        self._lengths = index["lengths"].astype(np.int64, copy=False)

        self.embedding_dim = int(manifest["embedding_dim"])
        self.max_len = int(manifest["max_len"])

        shard_entries = manifest.get("shards") or []
        if not shard_entries:
            raise ValueError("manifest.json has no shard entries")

        shard_files: list[str] = []
        for entry in shard_entries:
            if isinstance(entry, str):
                shard_files.append(entry)
            elif isinstance(entry, dict) and "file" in entry:
                shard_files.append(str(entry["file"]))
            else:
                raise ValueError(f"Unsupported shard entry: {entry}")

        self._shards = [np.load(root / name, mmap_mode="r") for name in shard_files]

    def __contains__(self, protein_id: str) -> bool:
        return protein_id in self._id_to_pos

    def __getitem__(self, protein_id: str) -> np.ndarray:
        if protein_id not in self._id_to_pos:
            raise KeyError(f"Protein ID '{protein_id}' not found in embeddings")

        pos = self._id_to_pos[protein_id]
        shard_id = int(self._shard_idx[pos])
        row_id = int(self._row_idx[pos])
        length = int(self._lengths[pos])

        shard = self._shards[shard_id]
        seq = shard[row_id]
        if 0 < length <= seq.shape[0]:
            vec = seq[:length].mean(axis=0)
        else:
            vec = seq.mean(axis=0)
        return np.asarray(vec, dtype=np.float32)


class _TorchCacheEmbeddingStore:
    """Mean-pooled view over the DL per-protein embedding cache."""

    def __init__(self, root: Path) -> None:
        index_path = root / INDEX_FILENAME
        metadata_path = root / METADATA_FILENAME
        if not index_path.exists():
            raise FileNotFoundError(f"Missing {INDEX_FILENAME} in {root}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing {METADATA_FILENAME} in {root}")

        raw_index = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(raw_index, dict):
            raise ValueError(f"{index_path} must contain a JSON object")
        self._index = {
            str(protein_id): str(relative_path)
            for protein_id, relative_path in raw_index.items()
        }

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(f"{metadata_path} must contain a JSON object")
        self.embedding_dim = int(metadata["input_dim"])
        self.max_sequence_length = int(metadata["max_sequence_length"])
        self._cache_dir = root
        self._mean_pool_cache: dict[str, np.ndarray] = {}

    def __contains__(self, protein_id: str) -> bool:
        return protein_id in self._index

    def __getitem__(self, protein_id: str) -> np.ndarray:
        cached = self._mean_pool_cache.get(protein_id)
        if cached is not None:
            return cached

        tensor = load_cached_embedding(
            cache_dir=self._cache_dir,
            index=self._index,
            protein_id=protein_id,
            expected_input_dim=self.embedding_dim,
            max_sequence_length=self.max_sequence_length,
        )
        vector = tensor.mean(dim=0).cpu().numpy().astype(np.float32)
        self._mean_pool_cache[protein_id] = vector
        return vector


def _load_legacy_embeddings(embeddings_path: Path) -> dict[str, np.ndarray]:
    """Load legacy serialized embedding artifacts into memory."""
    suffix = embeddings_path.suffix.lower()

    if suffix == ".pkl":
        import pickle

        with embeddings_path.open("rb") as handle:
            raw_dict = pickle.load(handle)
        embeddings_dict = {}
        for protein_id, value in raw_dict.items():
            if isinstance(value, dict) and "embeddings" in value:
                embeddings_dict[str(protein_id)] = value["embeddings"]
            else:
                embeddings_dict[str(protein_id)] = value
    elif suffix == ".npz":
        npz_data = np.load(embeddings_path, allow_pickle=True)
        if "ids" in npz_data.files and "embeddings" in npz_data.files:
            ids_array = npz_data["ids"]
            embeddings_array = npz_data["embeddings"]
            embeddings_dict = {}
            for idx, protein_id in enumerate(ids_array):
                normalized_id = (
                    protein_id.decode("utf-8")
                    if isinstance(protein_id, bytes)
                    else str(protein_id)
                )
                embeddings_dict[normalized_id] = embeddings_array[idx]
        else:
            embeddings_dict = {}
            for key in npz_data.files:
                value = npz_data[key]
                if isinstance(value, dict) and "embeddings" in value:
                    embeddings_dict[key] = value["embeddings"]
                else:
                    embeddings_dict[key] = value
    elif suffix == ".pt":
        embeddings_dict = torch.load(
            embeddings_path,
            map_location="cpu",
            weights_only=False,
        )
    else:
        raise ValueError(
            f"Unsupported embeddings format: '{suffix}'. "
            "Supported formats: .npz, .pkl, .pt, or a DL cache directory"
        )

    processed: dict[str, np.ndarray] = {}
    for protein_id, embedding in embeddings_dict.items():
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.numpy()
        if embedding.ndim == 3:
            embedding = embedding.squeeze(0).mean(axis=0)
        elif embedding.ndim == 2:
            embedding = embedding.mean(axis=0)
        processed[str(protein_id)] = np.asarray(embedding, dtype=np.float32)
    return processed


def _load_ml_embeddings(embeddings_path: str) -> object:
    """
    Load mean-pooled embeddings from disk with caching.

    Supported formats:
      - DL cache directory with ``index.json`` + ``metadata.json`` + ``.pt`` files
      - Sharded embeddings directory with ``manifest.json`` + ``index.npz``
      - ``.pt`` / ``.pkl`` / ``.npz`` legacy artifacts
    """
    global _ML_EMBEDDINGS_CACHE, _ML_EMBEDDINGS_PATH_CACHE

    if (
        _ML_EMBEDDINGS_CACHE is not None
        and _ML_EMBEDDINGS_PATH_CACHE == embeddings_path
    ):
        return _ML_EMBEDDINGS_CACHE

    path = Path(embeddings_path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    if _is_main_process():
        logging.info("Loading ML embeddings from %s", embeddings_path)

    if path.is_dir():
        if (path / INDEX_FILENAME).exists() and (path / METADATA_FILENAME).exists():
            store: object = _TorchCacheEmbeddingStore(path)
        else:
            store = _ShardedEmbeddingStore(path)
        _ML_EMBEDDINGS_CACHE = store
        _ML_EMBEDDINGS_PATH_CACHE = embeddings_path
        return store

    processed = _load_legacy_embeddings(path)
    if _is_main_process():
        logging.info("Loaded %d protein embeddings", len(processed))

    _ML_EMBEDDINGS_CACHE = processed
    _ML_EMBEDDINGS_PATH_CACHE = embeddings_path
    return processed


def _embedding_dim_from_store(embeddings: object) -> int:
    """Infer the embedding dimension from a loaded embedding source."""
    if hasattr(embeddings, "embedding_dim"):
        return int(getattr(embeddings, "embedding_dim"))
    if isinstance(embeddings, dict) and embeddings:
        first_value = next(iter(embeddings.values()))
        if isinstance(first_value, np.ndarray):
            return int(first_value.shape[0])
    raise ValueError("Unable to infer embedding_dim from embeddings source")


def load_ml_features(
    csv_path: str,
    embeddings_path: str,
    pooling: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load features and labels for classical ML models.

    For each protein pair ``(A, B)`` this creates one feature vector:
    ``[mean(embedding_A), mean(embedding_B)]``.
    """
    if pooling != "mean":
        raise ValueError(
            f"Unsupported pooling strategy: {pooling}. Only 'mean' is supported."
        )

    embeddings = _load_ml_embeddings(embeddings_path)
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    frame = pd.read_csv(path)
    required_cols = ["uniprotID_A", "uniprotID_B", "isInteraction"]
    missing = [column for column in required_cols if column not in frame.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    embedding_dim = _embedding_dim_from_store(embeddings)
    feature_dim = 2 * embedding_dim

    if _is_main_process():
        logging.info(
            "Loading ML features: %d pairs, feature_dim=%d",
            len(frame),
            feature_dim,
        )

    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    missing_proteins: set[str] = set()

    for row in frame.itertuples(index=False):
        protein_a = str(row.uniprotID_A)
        protein_b = str(row.uniprotID_B)
        label = int(row.isInteraction)

        if protein_a not in embeddings:
            missing_proteins.add(protein_a)
            continue
        if protein_b not in embeddings:
            missing_proteins.add(protein_b)
            continue

        emb_a = embeddings[protein_a]
        emb_b = embeddings[protein_b]
        feature_rows.append(np.concatenate([emb_a, emb_b]).astype(np.float32))
        labels.append(label)

    if missing_proteins:
        logging.warning(
            "Skipped %d proteins not found in embeddings. First 5: %s",
            len(missing_proteins),
            list(sorted(missing_proteins))[:5],
        )

    if feature_rows:
        features = np.stack(feature_rows, axis=0)
    else:
        features = np.zeros((0, feature_dim), dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.int64)

    n_pos = int(label_array.sum())
    n_neg = int(len(label_array) - n_pos)
    if _is_main_process():
        logging.info("Class distribution: %d positive, %d negative", n_pos, n_neg)

    return features, label_array
