"""
Data loading utilities for DIPPI pipeline.

This module provides:
- ProteinPairDataset: Custom PyTorch Dataset for protein pair embeddings
- build_loader: Public API to create DataLoader instances for train/val/test
- load_ml_features: Load mean-pooled features for classical ML models

Responsibilities:
- Load protein pairs from CSV (uniprotID_A, uniprotID_B, isInteraction)
- Fetch pre-computed embeddings from sharded .npy stores
- Return batched tensors compatible with model forward pass

Does NOT:
- Perform data augmentation or transforms
- Apply truncation/padding or CLS/EOS cleaning (assumed precomputed)
- Manage distributed training internals (delegates to DistributedSampler)
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from src.utils.samplers import ImbalancedBatchSampler, OnlineHardNegativeBatchSampler
from src.utils.distributed import get_rank, get_world_size, is_main_process


# Module-level cache to avoid reloading large embeddings file
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

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        fmt = manifest.get("format")
        if fmt != "dippi_sharded_embeddings_v1":
            raise ValueError(f"Unsupported embeddings format: {fmt}")

        self.max_len = int(manifest["max_len"])
        self.embedding_dim = int(manifest["embedding_dim"])
        self.storage_dtype = str(manifest.get("storage_dtype", "fp16"))
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


def _load_embeddings(embeddings_path: str) -> Dict[str, Any]:
    """
    Load embeddings from a sharded embedding directory with module-level caching.

    Args:
        embeddings_path: Path to a sharded fixed-length embedding store directory.

    Returns:
        Dictionary-like object mapping protein IDs to embedding data

    Note:
        Uses module-level cache to avoid reloading the same store.
    """
    global _EMBEDDINGS_CACHE, _EMBEDDINGS_PATH_CACHE

    # Return cached version if already loaded
    if _EMBEDDINGS_CACHE is not None and _EMBEDDINGS_PATH_CACHE == embeddings_path:
        return _EMBEDDINGS_CACHE

    # Load embeddings
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


def _stack_protein_pairs(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function to batch protein pair samples.

    Args:
        batch: List of dicts from ProteinPairDataset.__getitem__

    Returns:
        Batched dict with keys: emb_a, emb_b, len_a, len_b, label
    """
    # Stack embeddings: (B, max_len, D)
    emb_a = torch.stack([item["emb_a"] for item in batch], dim=0)
    emb_b = torch.stack([item["emb_b"] for item in batch], dim=0)

    # Stack lengths: (B,)
    len_a = torch.tensor([item["len_a"] for item in batch], dtype=torch.long)
    len_b = torch.tensor([item["len_b"] for item in batch], dtype=torch.long)

    # Stack labels: (B, 1) for BCE loss compatibility
    labels = torch.tensor([[item["label"]] for item in batch], dtype=torch.float32)

    return {
        "emb_a": emb_a,
        "emb_b": emb_b,
        "len_a": len_a,
        "len_b": len_b,
        "label": labels,
    }


def _collate_protein_pairs(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function to batch protein pair samples.

    Supports standard batches and OHEM batches containing "role" metadata.
    """
    if not batch:
        return {}

    has_roles = any("role" in item for item in batch)
    if not has_roles:
        return _stack_protein_pairs(batch)

    role_buckets: Dict[str, List[Dict[str, Any]]] = {
        "pos": [],
        "neg_candidate": [],
        "neg_default": [],
    }
    hard_counts = set()

    for item in batch:
        role = item.get("role")
        if role not in role_buckets:
            raise ValueError(f"Unknown sampling role: {role}")
        role_buckets[role].append(item)
        hard_count = item.get("hard_count")
        if hard_count is not None:
            hard_counts.add(int(hard_count))

    if len(hard_counts) > 1:
        raise ValueError(
            f"Inconsistent hard_count values within OHEM batch: {sorted(hard_counts)}"
        )

    hard_count_value = int(next(iter(hard_counts))) if hard_counts else 0

    def _maybe_stack(items: List[Dict[str, Any]]) -> Optional[Dict[str, torch.Tensor]]:
        return _stack_protein_pairs(items) if items else None

    return {
        "_ohem": True,
        "hard_count": hard_count_value,
        "pos": _maybe_stack(role_buckets["pos"]),
        "neg_candidates": _maybe_stack(role_buckets["neg_candidate"]),
        "neg_default": _maybe_stack(role_buckets["neg_default"]),
    }


class DistributedBatchSampler:
    """
    Evenly shard a batch sampler across DDP ranks while keeping step counts aligned.

    Each process receives approximately ceil(len(batch_sampler) / world_size) batches.
    When batches are not divisible, the final batch is duplicated to pad so all ranks
    perform the same number of steps (avoids DDP hang on uneven steps).
    """

    def __init__(
        self,
        batch_sampler: Any,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        pad: bool = True,
    ) -> None:
        self.batch_sampler = batch_sampler
        self.num_replicas = int(
            num_replicas if num_replicas is not None else get_world_size()
        )
        self.rank = int(rank if rank is not None else get_rank())
        self.pad = pad

        if self.num_replicas <= 0:
            raise ValueError(
                "num_replicas must be positive for DistributedBatchSampler"
            )
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError("rank must be in the range [0, num_replicas)")

        self._total_batches = len(self.batch_sampler)
        self._max_per_rank = (
            math.ceil(self._total_batches / self.num_replicas)
            if self.num_replicas > 0
            else 0
        )

    def __iter__(self):
        local_batches: list[list[int]] = []
        last_batch: Optional[List[int]] = None

        for idx, batch in enumerate(self.batch_sampler):
            last_batch = batch
            if idx % self.num_replicas == self.rank:
                local_batches.append(batch)

        if (
            self.pad
            and self._total_batches > 0
            and len(local_batches) < self._max_per_rank
        ):
            filler = local_batches[-1] if local_batches else last_batch
            while len(local_batches) < self._max_per_rank and filler is not None:
                local_batches.append(filler)

        return iter(local_batches)

    def __len__(self) -> int:
        return self._max_per_rank

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch to the wrapped sampler if supported."""
        if hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)


class ProteinPairDataset(Dataset):
    """
    PyTorch Dataset for protein-protein interaction pairs.

    Loads protein pair metadata from CSV and fetches pre-computed embeddings
    from sharded numpy stores. Embeddings are assumed pre-cleaned and fixed-length.

    CSV format:
        uniprotID_A,uniprotID_B,isInteraction

    Embeddings dict format:
        {protein_id: {"embeddings": np.ndarray[max_len, D], "length": int, "_fixed_len": True}, ...}
    """

    def __init__(
        self,
        csv_path: str,
        embeddings_dict: Dict[str, Any],
        max_len: int,
        dtype: str,
    ):
        """
        Initialize dataset.

        Args:
            csv_path: Path to CSV with protein pairs
            embeddings_dict: Pre-loaded embeddings dictionary
            max_len: Expected fixed sequence length for embeddings
            dtype: Embedding dtype string ("bf16", "fp32", "fp16")
        """
        super().__init__()

        self.embeddings_dict = embeddings_dict
        self.max_len = max_len

        # Map dtype string to torch dtype
        self.dtype_map = {
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "fp16": torch.float16,
        }
        if dtype not in self.dtype_map:
            raise ValueError(
                f"Unsupported dtype: {dtype}. Must be one of {list(self.dtype_map.keys())}"
            )
        self.torch_dtype = self.dtype_map[dtype]

        # Load CSV
        csv_path_obj = Path(csv_path)
        if not csv_path_obj.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.df = pd.read_csv(
            csv_path,
            dtype={
                "uniprotID_A": "category",
                "uniprotID_B": "category",
                "isInteraction": "float32",
            },
        )

        # Validate required columns
        required_cols = ["uniprotID_A", "uniprotID_B", "isInteraction"]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {missing_cols}")

        # Validate that all protein IDs in CSV exist in embeddings
        all_protein_ids = set(self.df["uniprotID_A"].unique()) | set(
            self.df["uniprotID_B"].unique()
        )
        missing_ids = all_protein_ids - set(self.embeddings_dict.keys())
        if missing_ids:
            sample_missing = list(missing_ids)[:10]
            if is_main_process():
                logging.warning(
                    f"Found {len(missing_ids)} protein IDs in CSV that are missing from embeddings. "
                    f"Sample missing IDs: {sample_missing}. "
                    f"These will cause KeyError during data loading. "
                    f"Consider filtering the CSV or updating embeddings."
                )
            # Optionally raise error instead of warning
            # raise ValueError(
            #     f"Missing embeddings for {len(missing_ids)} proteins. "
            #     f"Sample: {sample_missing}"
            # )

        # Log dataset stats
        n_samples = len(self.df)
        n_positive = int(self.df["isInteraction"].sum())
        if is_main_process():
            logging.info(
                f"Dataset: {n_samples} pairs ({n_positive} pos, {n_samples - n_positive} neg)"
            )

    def __len__(self) -> int:
        """Return number of protein pairs."""
        return len(self.df)

    def _process_embedding(self, protein_id: str) -> tuple[torch.Tensor, int]:
        """
        Load a single protein embedding.

        Args:
            protein_id: UniProt ID

        Returns:
            embedding: Tensor[max_len, D] (fixed-length)
            actual_length: Original sequence length (from index)

        Raises:
            KeyError: If protein_id is not found in embeddings_dict
        """
        # Fetch embedding from dict with better error message
        if protein_id not in self.embeddings_dict:
            available_keys = list(self.embeddings_dict.keys())[:10]
            raise KeyError(
                f"Protein ID '{protein_id}' not found in embeddings. "
                f"Total embeddings: {len(self.embeddings_dict)}. "
                f"Sample keys: {available_keys}. "
                f"Check that your CSV protein IDs match the embedding keys."
            )

        protein_data = self.embeddings_dict[protein_id]

        # Sharded embeddings: fixed-length with stored lengths.
        if isinstance(protein_data, dict) and protein_data.get("_fixed_len"):
            embedding = protein_data["embeddings"]
            actual_length = int(protein_data.get("length", self.max_len))

            if isinstance(embedding, np.ndarray):
                if not embedding.flags.writeable:
                    embedding = np.array(embedding, copy=True)
                embedding = torch.from_numpy(embedding)
            if embedding.dim() == 3:
                embedding = embedding.squeeze(0)

            if embedding.size(0) != self.max_len:
                raise ValueError(
                    f"Embedding length {embedding.size(0)} does not match expected "
                    f"max_len {self.max_len}. Regenerate sharded embeddings or "
                    "update config to match."
                )

            embedding = embedding.to(dtype=self.torch_dtype)
            return embedding, actual_length
        raise ValueError(
            f"Unexpected embedding format for protein '{protein_id}'. "
            "Expected sharded embeddings with _fixed_len metadata."
        )

    def __getitem__(self, idx: int | tuple[int, str] | tuple[int, str, int]):
        """
        Get a single protein pair sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with keys:
                - emb_a: Tensor[max_len, D]
                - emb_b: Tensor[max_len, D]
                - len_a: int (actual sequence length)
                - len_b: int (actual sequence length)
                - label: float (0.0 or 1.0)
                - role: Optional[str] (for OHEM batches)
                - hard_count: Optional[int] (for OHEM batches)
        """
        role = None
        hard_count = None
        if isinstance(idx, tuple):
            if len(idx) == 2:
                idx, role = idx
            elif len(idx) == 3:
                idx, role, hard_count = idx
            else:
                raise ValueError("Index tuple must be (idx, role[, hard_count])")

        if not isinstance(idx, (int, np.integer)):
            raise TypeError(f"Index must be int, got {type(idx)}")

        row = self.df.iloc[int(idx)]

        protein_a = str(row["uniprotID_A"])
        protein_b = str(row["uniprotID_B"])
        label = float(row["isInteraction"])

        # Process both proteins
        emb_a, len_a = self._process_embedding(protein_a)
        emb_b, len_b = self._process_embedding(protein_b)

        sample = {
            "emb_a": emb_a,
            "emb_b": emb_b,
            "len_a": len_a,
            "len_b": len_b,
            "label": label,
        }
        if role is not None:
            sample["role"] = role
        if hard_count is not None:
            sample["hard_count"] = int(hard_count)
        return sample


def build_loader(
    csv_path: str,
    embeddings_path: str,
    batch_size: int,
    max_len: int,
    dtype: str,
    ddp: bool = False,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    sampling_cfg: Optional[Dict[str, Any]] = None,
    dataloader_cfg: Optional[Dict[str, Any]] = None,
) -> DataLoader:
    """
    Build a DataLoader for protein pair data.

    This is the main public API called by run.py to create train/val/test loaders.

    Args:
        csv_path: Path to CSV with protein pairs
        embeddings_path: Path to sharded embeddings directory
        batch_size: Batch size
        max_len: Expected fixed sequence length for embeddings
        dtype: Embedding dtype string ("bf16", "fp32", "fp16")
        ddp: Whether DDP is enabled (will use DistributedSampler)
        shuffle: Whether to shuffle data (ignored if ddp=True; sampler handles it)
        num_workers: Optional override for dataloader workers
        pin_memory: Optional override for pin_memory
        sampling_cfg: Optional sampling configuration (e.g., {"strategy": "imbalanced", "pos_neg_ratio": 8})
        dataloader_cfg: Optional dataloader configuration dict:
            {
                "num_workers": int,
                "pin_memory": bool,
                "prefetch_factor": int,
                "persistent_workers": bool,
                "drop_last": bool,
            }

    Returns:
        DataLoader instance

    Example:
        train_loader = build_loader(
            csv_path="data/splits/train.csv",
            embeddings_path="data/embeddings_shards",
            batch_size=32,
            max_len=1024,
            dtype="bf16",
            ddp=False,
            shuffle=True,
        )
    """
    # Load embeddings (cached if already loaded)
    embeddings_dict = _load_embeddings(embeddings_path)

    # Create dataset
    dataset = ProteinPairDataset(
        csv_path=csv_path,
        embeddings_dict=embeddings_dict,
        max_len=max_len,
        dtype=dtype,
    )

    # Dataloader settings
    dl_cfg = dataloader_cfg.copy() if dataloader_cfg else {}
    if num_workers is not None:
        dl_cfg["num_workers"] = num_workers
    if pin_memory is not None:
        dl_cfg["pin_memory"] = pin_memory

    num_workers = int(dl_cfg.get("num_workers", 0))
    pin_memory = bool(dl_cfg.get("pin_memory", torch.cuda.is_available()))
    drop_last = bool(dl_cfg.get("drop_last", False))

    prefetch_factor = dl_cfg.get("prefetch_factor")
    persistent_workers = dl_cfg.get("persistent_workers")

    # Prefetch/persistent only valid when workers > 0
    if num_workers <= 0:
        prefetch_factor = None
        persistent_workers = False
    else:
        if prefetch_factor is None:
            prefetch_factor = 2  # PyTorch default
        if persistent_workers is None:
            persistent_workers = True

    # Configure sampler / batch_sampler
    batch_sampler = None
    sampler = None
    sampling_cfg = sampling_cfg or {}
    sampling_strategy = sampling_cfg.get("strategy")

    if sampling_strategy == "imbalanced":
        ratio = float(sampling_cfg.get("pos_neg_ratio", 3.0))
        base_sampler = ImbalancedBatchSampler(
            labels=list(dataset.df["isInteraction"].astype(int)),
            batch_size=batch_size,
            pos_neg_ratio=ratio,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=None,
        )
        if ddp and get_world_size() > 1:
            batch_sampler = DistributedBatchSampler(base_sampler, pad=True)
        else:
            batch_sampler = base_sampler
        shuffle = False  # Batch sampler controls ordering
    elif sampling_strategy == "staged_hard":
        ratio = float(sampling_cfg.get("pos_neg_ratio", 16.0))
        hard_ratio = float(sampling_cfg.get("hard_ratio", 0.7))
        warmup_epochs = int(sampling_cfg.get("warmup_epochs", 2))
        if "hard_start_epoch" in sampling_cfg:
            warmup_epochs = int(sampling_cfg["hard_start_epoch"])
        base_sampler = OnlineHardNegativeBatchSampler(
            labels=list(dataset.df["isInteraction"].astype(int)),
            batch_size=batch_size,
            pos_neg_ratio=ratio,
            warmup_epochs=warmup_epochs,
            hard_ratio=hard_ratio,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=None,
        )
        if ddp and get_world_size() > 1:
            batch_sampler = DistributedBatchSampler(base_sampler, pad=True)
        else:
            batch_sampler = base_sampler
        shuffle = False  # Batch sampler controls ordering
    elif ddp:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # Sampler controls ordering

    # Create DataLoader
    loader_kwargs = {
        "num_workers": num_workers,
        "collate_fn": _collate_protein_pairs,
        "pin_memory": pin_memory and torch.cuda.is_available(),
    }
    if prefetch_factor is not None and num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    if persistent_workers:
        loader_kwargs["persistent_workers"] = True

    # batch_sampler is mutually exclusive with batch_size, shuffle, sampler, drop_last
    # Only pass the appropriate set of arguments
    if batch_sampler is not None:
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            **loader_kwargs,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=drop_last,
            **loader_kwargs,
        )

    if is_main_process():
        sampler_name = (
            batch_sampler.__class__.__name__
            if batch_sampler is not None
            else sampler.__class__.__name__
            if sampler is not None
            else "None"
        )
        logging.info(
            "DataLoader built for %s: num_workers=%d, pin_memory=%s, prefetch_factor=%s, "
            "persistent_workers=%s, sampler=%s",
            Path(csv_path).name,
            num_workers,
            loader_kwargs["pin_memory"],
            prefetch_factor if prefetch_factor is not None else "None",
            bool(persistent_workers),
            sampler_name,
        )

    return loader
