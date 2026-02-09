"""Sharded embedding store and cached loading helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.utils.distributed import is_main_process

_EMBEDDINGS_CACHE: Optional[Dict[str, Any]] = None
_EMBEDDINGS_PATH_CACHE: Optional[str] = None


class ShardedEmbeddingStore:
    """
    Memory-mapped fixed-length embedding store backed by shard .npy files.

    Expected directory layout:
      - manifest.json
      - index.npz (ids, shard_idx, row_idx, lengths)
      - shard_00000.npy, shard_00001.npy, ...
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {self.root}")

        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        fmt = manifest.get("format")
        if fmt != "dippi_sharded_embeddings_v1":
            raise ValueError(f"Unsupported embeddings format: {fmt}")

        self.max_len = int(manifest["max_len"])
        self.shards = manifest.get("shards", [])

        index_path = self.root / manifest.get("index_file", "index.npz")
        if not index_path.exists():
            raise FileNotFoundError(f"index file not found: {index_path}")

        index = np.load(index_path, allow_pickle=True)
        ids = index["ids"]
        shard_idx = index["shard_idx"].astype(np.int32)
        row_idx = index["row_idx"].astype(np.int32)
        lengths = index["lengths"].astype(np.int32) if "lengths" in index else None

        self._id_to_pos: Dict[str, int] = {}
        self._ids: list[str] = []
        for pos, protein_id in enumerate(ids):
            if isinstance(protein_id, bytes):
                protein_id = protein_id.decode("utf-8")
            elif isinstance(protein_id, np.str_):
                protein_id = str(protein_id)
            self._id_to_pos[protein_id] = pos
            self._ids.append(protein_id)

        self._shard_idx = shard_idx
        self._row_idx = row_idx
        self._lengths = lengths
        self._shard_cache: Dict[int, np.ndarray] = {}

    def __contains__(self, protein_id: str) -> bool:
        return protein_id in self._id_to_pos

    def __len__(self) -> int:
        return len(self._ids)

    def keys(self):
        return self._ids

    def _get_shard(self, shard_id: int) -> np.ndarray:
        cached = self._shard_cache.get(shard_id)
        if cached is not None:
            return cached
        if shard_id < 0 or shard_id >= len(self.shards):
            raise IndexError(f"Shard index out of range: {shard_id}")
        shard_file = self.root / self.shards[shard_id]["file"]
        shard_data = np.load(shard_file, mmap_mode="r")
        self._shard_cache[shard_id] = shard_data
        return shard_data

    def __getitem__(self, protein_id: str) -> Dict[str, Any]:
        if protein_id not in self._id_to_pos:
            raise KeyError(f"Protein ID '{protein_id}' not found in embeddings")
        pos = self._id_to_pos[protein_id]
        shard_id = int(self._shard_idx[pos])
        row_id = int(self._row_idx[pos])
        length = int(self._lengths[pos]) if self._lengths is not None else self.max_len
        embeddings = self._get_shard(shard_id)[row_id]
        return {"embeddings": embeddings, "length": length, "_fixed_len": True}


def load_sharded_embeddings(embeddings_path: str) -> Dict[str, Any]:
    """Load sharded embeddings and cache by path for reuse across loaders."""
    global _EMBEDDINGS_CACHE, _EMBEDDINGS_PATH_CACHE

    if _EMBEDDINGS_CACHE is not None and _EMBEDDINGS_PATH_CACHE == embeddings_path:
        return _EMBEDDINGS_CACHE

    embeddings_path_obj = Path(embeddings_path)
    if not embeddings_path_obj.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not embeddings_path_obj.is_dir():
        raise ValueError(
            "Embeddings path must be a directory containing sharded embeddings "
            "(manifest.json, index.npz, shard_*.npy)."
        )

    manifest_path = embeddings_path_obj / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {embeddings_path_obj}")

    embeddings_dict = ShardedEmbeddingStore(embeddings_path_obj)
    _EMBEDDINGS_CACHE = embeddings_dict
    _EMBEDDINGS_PATH_CACHE = embeddings_path
    if is_main_process():
        logging.info(
            "Loaded sharded embeddings from %s (%d proteins, max_len=%d)",
            embeddings_path_obj,
            len(embeddings_dict),
            embeddings_dict.max_len,
        )
    return embeddings_dict
