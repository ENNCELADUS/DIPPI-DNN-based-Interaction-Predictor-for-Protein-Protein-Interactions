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

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.data.dataloader import (
    DistributedBatchSampler,
    build_dataloader_from_components,
    build_sampler_components,
    resolve_dataloader_settings,
    sampler_name,
)
from src.utils.data.embedding_store import (
    ShardedEmbeddingStore,
    load_sharded_embeddings,
)
from src.utils.distributed import is_main_process

__all__ = [
    "ShardedEmbeddingStore",
    "DistributedBatchSampler",
    "ProteinPairDataset",
    "build_loader",
]


def _load_embeddings(embeddings_path: str) -> Dict[str, Any]:
    """Wrapper kept for test patching/backward compatibility."""
    return load_sharded_embeddings(embeddings_path)


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

    Supports standard batches and OHEM pool batches containing "role" metadata.
    """
    if not batch:
        return {}

    has_roles = any("role" in item for item in batch)
    if not has_roles:
        return _stack_protein_pairs(batch)

    pool_items: List[Dict[str, Any]] = []
    batch_sizes = set()
    cap_values = set()

    for item in batch:
        role = item.get("role")
        if role != "ohem_pool":
            raise ValueError(f"Unknown sampling role: {role}")
        pool_items.append(item)
        if item.get("ohem_batch_size") is not None:
            batch_sizes.add(int(item["ohem_batch_size"]))
        if item.get("cap_protein") is not None:
            cap_values.add(int(item["cap_protein"]))

    if len(batch_sizes) > 1:
        raise ValueError(
            f"Inconsistent ohem_batch_size values within pool: {sorted(batch_sizes)}"
        )
    if len(cap_values) > 1:
        raise ValueError(
            f"Inconsistent cap_protein values within pool: {sorted(cap_values)}"
        )

    pool = _stack_protein_pairs(pool_items)
    pool["protein_a"] = [item["protein_a"] for item in pool_items]
    pool["protein_b"] = [item["protein_b"] for item in pool_items]

    return {
        "_ohem": True,
        "pool": pool,
        "ohem_batch_size": int(next(iter(batch_sizes))) if batch_sizes else 0,
        "cap_protein": int(next(iter(cap_values))) if cap_values else 0,
    }


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

    def __getitem__(
        self,
        idx: int | tuple[int, str] | tuple[int, str, int] | tuple[int, str, int, int],
    ):
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
                - ohem_batch_size: Optional[int] (for OHEM pool batches)
                - cap_protein: Optional[int] (for OHEM pool batches)
        """
        role = None
        ohem_batch_size = None
        cap_protein = None
        if isinstance(idx, tuple):
            if len(idx) == 2:
                idx, role = idx
            elif len(idx) == 3:
                idx, role, ohem_batch_size = idx
            elif len(idx) == 4:
                idx, role, ohem_batch_size, cap_protein = idx
            else:
                raise ValueError(
                    "Index tuple must be (idx, role[, ohem_batch_size[, cap_protein]])"
                )

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
            sample["protein_a"] = protein_a
            sample["protein_b"] = protein_b
            if ohem_batch_size is not None:
                sample["ohem_batch_size"] = int(ohem_batch_size)
            if cap_protein is not None:
                sample["cap_protein"] = int(cap_protein)
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

    settings = resolve_dataloader_settings(
        dataloader_cfg,
        num_workers_override=num_workers,
        pin_memory_override=pin_memory,
    )

    batch_sampler, sampler, shuffle = build_sampler_components(
        dataset=dataset,
        labels=list(dataset.df["isInteraction"].astype(int)),
        batch_size=batch_size,
        sampling_cfg=sampling_cfg,
        ddp=ddp,
        shuffle=shuffle,
        drop_last=bool(settings["drop_last"]),
    )

    loader = build_dataloader_from_components(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        drop_last=bool(settings["drop_last"]),
        collate_fn=_collate_protein_pairs,
        settings=settings,
    )

    if is_main_process():
        active_sampler_name = sampler_name(batch_sampler, sampler)
        logging.info(
            "DataLoader built for %s: num_workers=%d, pin_memory=%s, prefetch_factor=%s, "
            "persistent_workers=%s, sampler=%s",
            Path(csv_path).name,
            int(settings["num_workers"]),
            bool(settings["pin_memory"]) and torch.cuda.is_available(),
            settings["prefetch_factor"]
            if settings["prefetch_factor"] is not None
            else "None",
            bool(settings["persistent_workers"]),
            active_sampler_name,
        )

    return loader
