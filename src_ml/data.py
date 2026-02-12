import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Mapping

import numpy as np
import pandas as pd
import torch

from src.utils.distributed import is_main_process


# =============================================================================
# ML Feature Loading (for classical ML models like RandomForest, XGBoost)
# =============================================================================

# Module-level cache for ML embeddings (separate from DL embeddings)
_ML_EMBEDDINGS_CACHE: Optional[Mapping[str, Any]] = None
_ML_EMBEDDINGS_PATH_CACHE: Optional[str] = None


class _ShardedEmbeddingStore:
    """
    Memory-mapped embedding store backed by sharded .npy files.

    Expects directory layout:
      - manifest.json
      - index.npz (ids, shard_idx, row_idx, lengths)
      - shard_*.npy (shape [shard_size, max_len, embedding_dim])
    """

    def __init__(self, root: Path) -> None:
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest.json in {root}")

        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != "dippi_sharded_embeddings_v1":
            raise ValueError(f"Unsupported embeddings format: {manifest.get('format')}")

        index_file = manifest.get("index_file", "index.npz")
        index_path = root / index_file
        if not index_path.exists():
            raise FileNotFoundError(f"Missing index file: {index_path}")

        index = np.load(index_path, allow_pickle=True, mmap_mode="r")
        ids = index["ids"]
        self._ids = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in ids
        ]
        self._id_to_pos = {pid: i for i, pid in enumerate(self._ids)}
        self._shard_idx = index["shard_idx"].astype(np.int64, copy=False)
        self._row_idx = index["row_idx"].astype(np.int64, copy=False)
        self._lengths = index["lengths"].astype(np.int64, copy=False)

        self.embedding_dim = int(manifest["embedding_dim"])
        self.max_len = int(manifest["max_len"])

        shard_entries = manifest.get("shards") or []
        if not shard_entries:
            raise ValueError("manifest.json has no shard entries")

        shard_files = []
        for entry in shard_entries:
            if isinstance(entry, str):
                shard_files.append(entry)
            elif isinstance(entry, dict) and "file" in entry:
                shard_files.append(entry["file"])
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


def _load_ml_embeddings(embeddings_path: str) -> Mapping[str, np.ndarray]:
    """
    Load mean-pooled embeddings from disk with caching.

    Expected format:
        - Directory with sharded embeddings (manifest.json + index.npz + shard_*.npy)
        - .pt: Dict[protein_id, tensor of shape (1, seq_len, embed_dim) or (seq_len, embed_dim)]
        - .pkl: Dict[protein_id, dict with 'embeddings' key containing ndarray]

    Args:
        embeddings_path: Path to embeddings file or sharded embeddings directory.

    Returns:
        Mapping from protein IDs to 1D embedding vectors.
    """
    global _ML_EMBEDDINGS_CACHE, _ML_EMBEDDINGS_PATH_CACHE

    if (
        _ML_EMBEDDINGS_CACHE is not None
        and _ML_EMBEDDINGS_PATH_CACHE == embeddings_path
    ):
        return _ML_EMBEDDINGS_CACHE

    embeddings_path_obj = Path(embeddings_path)
    if not embeddings_path_obj.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    if is_main_process():
        logging.info(f"Loading ML embeddings from {embeddings_path}...")

    if embeddings_path_obj.is_dir():
        store = _ShardedEmbeddingStore(embeddings_path_obj)
        _ML_EMBEDDINGS_CACHE = store
        _ML_EMBEDDINGS_PATH_CACHE = embeddings_path
        return store

    suffix = embeddings_path_obj.suffix.lower()

    # Load based on file extension
    if suffix == ".pkl":
        import pickle

        with open(embeddings_path, "rb") as f:
            raw_dict = pickle.load(f)
        # Handle nested dict format: {protein_id: {'embeddings': ndarray, ...}}
        embeddings_dict = {}
        for protein_id, val in raw_dict.items():
            if isinstance(val, dict) and "embeddings" in val:
                embeddings_dict[protein_id] = val["embeddings"]
            else:
                embeddings_dict[protein_id] = val
    elif suffix == ".npz":
        # NumPy compressed archive format
        npz_data = np.load(embeddings_path, allow_pickle=True)

        if "ids" in npz_data.files and "embeddings" in npz_data.files:
            # Structured format: ids array + embeddings array
            ids_array = npz_data["ids"]
            embeddings_array = npz_data["embeddings"]

            embeddings_dict = {}
            for idx, protein_id in enumerate(ids_array):
                if isinstance(protein_id, bytes):
                    protein_id = protein_id.decode("utf-8")
                elif isinstance(protein_id, np.str_):
                    protein_id = str(protein_id)
                embeddings_dict[protein_id] = embeddings_array[idx]
        else:
            # Legacy format: each key is a protein ID
            embeddings_dict = {}
            for key in npz_data.files:
                val = npz_data[key]
                if isinstance(val, dict) and "embeddings" in val:
                    embeddings_dict[key] = val["embeddings"]
                else:
                    embeddings_dict[key] = val
    elif suffix == ".pt":
        # PyTorch .pt file
        embeddings_dict = torch.load(
            embeddings_path, map_location="cpu", weights_only=False
        )
    else:
        raise ValueError(
            f"Unsupported embeddings format: '{suffix}'. "
            "Supported formats: .npz, .pkl, .pt"
        )

    # Convert tensors to numpy and mean-pool if needed
    processed: Dict[str, np.ndarray] = {}
    for protein_id, emb in embeddings_dict.items():
        if isinstance(emb, torch.Tensor):
            emb = emb.numpy()
        # Ensure 1D (mean-pooled)
        if emb.ndim == 3:
            # Shape (1, seq_len, embed_dim) -> squeeze batch -> mean over seq
            emb = emb.squeeze(0).mean(axis=0)
        elif emb.ndim == 2:
            # Shape (seq_len, embed_dim) -> mean over seq
            emb = emb.mean(axis=0)
        processed[protein_id] = emb.astype(np.float32)

    if is_main_process():
        logging.info(f"Loaded {len(processed)} protein embeddings")

    _ML_EMBEDDINGS_CACHE = processed
    _ML_EMBEDDINGS_PATH_CACHE = embeddings_path

    return processed


def load_ml_features(
    csv_path: str,
    embeddings_path: str,
    pooling: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load features and labels for classical ML models.

    For each protein pair (A, B), creates feature vector:
        [embedding_A, embedding_B] → (2 * embedding_dim,)

    Args:
        csv_path: Path to CSV with columns [uniprotID_A, uniprotID_B, isInteraction].
        embeddings_path: Path to .pt file with mean-pooled embeddings.
        pooling: Pooling strategy (currently only "mean" is supported).

    Returns:
        X: Feature matrix of shape (n_samples, 2 * embedding_dim).
        y: Labels of shape (n_samples,).

    Raises:
        FileNotFoundError: If CSV or embeddings file not found.
        KeyError: If a protein ID is missing from embeddings.
    """
    if pooling != "mean":
        raise ValueError(
            f"Unsupported pooling strategy: {pooling}. Only 'mean' is supported."
        )

    # Load embeddings
    embeddings = _load_ml_embeddings(embeddings_path)

    # Load CSV
    csv_path_obj = Path(csv_path)
    if not csv_path_obj.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate columns
    required_cols = ["uniprotID_A", "uniprotID_B", "isInteraction"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    n_samples = len(df)
    if hasattr(embeddings, "embedding_dim"):
        embedding_dim = int(getattr(embeddings, "embedding_dim"))
    elif isinstance(embeddings, dict) and embeddings:
        embedding_dim = int(next(iter(embeddings.values())).shape[0])
    else:
        raise ValueError("Unable to infer embedding_dim from embeddings source")
    feature_dim = 2 * embedding_dim

    if is_main_process():
        logging.info(
            f"Loading ML features: {n_samples} pairs, feature_dim={feature_dim}"
        )

    # Preallocate arrays
    X = np.zeros((n_samples, feature_dim), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)

    missing_proteins: set = set()

    for i, row in df.iterrows():
        protein_a = row["uniprotID_A"]
        protein_b = row["uniprotID_B"]
        label = int(row["isInteraction"])

        # Get embeddings
        if protein_a not in embeddings:
            missing_proteins.add(protein_a)
            continue
        if protein_b not in embeddings:
            missing_proteins.add(protein_b)
            continue

        emb_a = embeddings[protein_a]
        emb_b = embeddings[protein_b]

        # Concatenate: [emb_a, emb_b]
        X[i] = np.concatenate([emb_a, emb_b])
        y[i] = label

    if missing_proteins:
        logging.warning(
            f"Skipped {len(missing_proteins)} proteins not found in embeddings. "
            f"First 5: {list(missing_proteins)[:5]}"
        )
        # Filter out rows with missing proteins
        valid_mask = ~np.all(X == 0, axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        logging.info(f"After filtering: {len(y)} valid samples")

    # Log class distribution
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if is_main_process():
        logging.info(f"Class distribution: {n_pos} positive, {n_neg} negative")

    return X, y
