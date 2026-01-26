"""
Sequence-based data loader for V6 (raw sequence pairs, no embeddings).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from src.utils.distributed import get_world_size, is_main_process
from src.utils.samplers import ImbalancedBatchSampler, StagedOHEMBatchSampler
from src.utils.data_io import DistributedBatchSampler


def _clean_sequence(sequence: str) -> str:
    cleaned = sequence.upper()
    return cleaned.replace("-", "").replace(".", "").replace("*", "")


@lru_cache(maxsize=4)
def _load_sequences(sequence_path: str) -> Dict[str, str]:
    path = Path(sequence_path)
    if not path.exists():
        raise FileNotFoundError(f"Sequence CSV not found: {sequence_path}")

    df = pd.read_csv(path, dtype={"uniprotID": "category", "sequence": "string"})
    if "uniprotID" not in df.columns or "sequence" not in df.columns:
        raise ValueError("Sequence CSV must contain 'uniprotID' and 'sequence' columns")

    seq_dict = {str(row["uniprotID"]): str(row["sequence"]) for _, row in df.iterrows()}
    if is_main_process():
        logging.info("Loaded %d sequences from %s", len(seq_dict), path.name)
    return seq_dict


class SequencePairDataset(Dataset):
    """
    Dataset for protein pair interactions using raw sequences.

    CSV format:
        uniprotID_A,uniprotID_B,isInteraction

    Sequence CSV format:
        uniprotID,sequence
    """

    def __init__(
        self,
        csv_path: str,
        sequence_path: str,
        max_len: int,
    ) -> None:
        super().__init__()
        self.max_len = int(max_len)
        self.sequence_dict = _load_sequences(sequence_path)

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
        required_cols = ["uniprotID_A", "uniprotID_B", "isInteraction"]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {missing_cols}")

        all_ids = set(self.df["uniprotID_A"].unique()) | set(
            self.df["uniprotID_B"].unique()
        )
        missing_ids = all_ids - set(self.sequence_dict.keys())
        if missing_ids and is_main_process():
            sample_missing = list(missing_ids)[:10]
            logging.warning(
                "Found %d protein IDs missing sequences. Sample: %s",
                len(missing_ids),
                sample_missing,
            )

        n_samples = len(self.df)
        n_positive = int(self.df["isInteraction"].sum())
        if is_main_process():
            logging.info(
                "Sequence dataset: %d pairs (%d pos, %d neg)",
                n_samples,
                n_positive,
                n_samples - n_positive,
            )

    def __len__(self) -> int:
        return len(self.df)

    def _process_sequence(self, protein_id: str) -> str:
        if protein_id not in self.sequence_dict:
            available = list(self.sequence_dict.keys())[:10]
            raise KeyError(
                f"Protein ID '{protein_id}' not found in sequences. "
                f"Total sequences: {len(self.sequence_dict)}. "
                f"Sample keys: {available}."
            )
        sequence = _clean_sequence(self.sequence_dict[protein_id])
        if self.max_len > 0 and len(sequence) > self.max_len:
            sequence = sequence[: self.max_len]
        return sequence

    def __getitem__(
        self,
        idx: int | tuple[int, str] | tuple[int, str, int] | tuple[int, str, int, int],
    ):
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

        seq_a = self._process_sequence(protein_a)
        seq_b = self._process_sequence(protein_b)

        sample = {"seq_a": seq_a, "seq_b": seq_b, "label": label}
        if role is not None:
            sample["role"] = role
            sample["protein_a"] = protein_a
            sample["protein_b"] = protein_b
            if ohem_batch_size is not None:
                sample["ohem_batch_size"] = int(ohem_batch_size)
            if cap_protein is not None:
                sample["cap_protein"] = int(cap_protein)
        return sample


def _stack_sequence_pairs(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    seq_a = [item["seq_a"] for item in batch]
    seq_b = [item["seq_b"] for item in batch]
    labels = torch.tensor([[item["label"]] for item in batch], dtype=torch.float32)
    return {"seq_a": seq_a, "seq_b": seq_b, "label": labels}


def _collate_sequence_pairs(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        return {}

    has_roles = any("role" in item for item in batch)
    if not has_roles:
        return _stack_sequence_pairs(batch)

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

    pool = _stack_sequence_pairs(pool_items)
    pool["protein_a"] = [item["protein_a"] for item in pool_items]
    pool["protein_b"] = [item["protein_b"] for item in pool_items]

    return {
        "_ohem": True,
        "pool": pool,
        "ohem_batch_size": int(next(iter(batch_sizes))) if batch_sizes else 0,
        "cap_protein": int(next(iter(cap_values))) if cap_values else 0,
    }


def build_sequence_loader(
    csv_path: str,
    sequence_path: str,
    batch_size: int,
    max_len: int,
    ddp: bool = False,
    shuffle: bool = True,
    sampling_cfg: Optional[Dict[str, Any]] = None,
    dataloader_cfg: Optional[Dict[str, Any]] = None,
) -> DataLoader:
    dataset = SequencePairDataset(
        csv_path=csv_path,
        sequence_path=sequence_path,
        max_len=max_len,
    )

    dl_cfg = dataloader_cfg.copy() if dataloader_cfg else {}
    num_workers = int(dl_cfg.get("num_workers", 0))
    pin_memory = bool(dl_cfg.get("pin_memory", torch.cuda.is_available()))
    drop_last = bool(dl_cfg.get("drop_last", False))

    prefetch_factor = dl_cfg.get("prefetch_factor")
    persistent_workers = dl_cfg.get("persistent_workers")

    if num_workers <= 0:
        prefetch_factor = None
        persistent_workers = False
    else:
        if prefetch_factor is None:
            prefetch_factor = 2
        if persistent_workers is None:
            persistent_workers = True

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
        shuffle = False
    elif sampling_strategy == "staged_hard":
        warmup_ratio = float(sampling_cfg.get("warmup_pos_neg_ratio", 7.0))
        warmup_epochs = int(sampling_cfg.get("warmup_epochs", 2))
        if "hard_start_epoch" in sampling_cfg:
            warmup_epochs = int(sampling_cfg["hard_start_epoch"])
        pool_multiplier = int(sampling_cfg.get("pool_multiplier", 16))
        cap_protein = int(
            sampling_cfg.get(
                "cap_protein", max(2, int(round(0.05 * float(batch_size))))
            )
        )
        base_sampler = StagedOHEMBatchSampler(
            labels=list(dataset.df["isInteraction"].astype(int)),
            batch_size=batch_size,
            warmup_pos_neg_ratio=warmup_ratio,
            warmup_epochs=warmup_epochs,
            pool_multiplier=pool_multiplier,
            cap_protein=cap_protein,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=None,
        )
        if ddp and get_world_size() > 1:
            batch_sampler = DistributedBatchSampler(base_sampler, pad=True)
        else:
            batch_sampler = base_sampler
        shuffle = False
    elif ddp:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False

    loader_kwargs = {
        "num_workers": num_workers,
        "collate_fn": _collate_sequence_pairs,
        "pin_memory": pin_memory and torch.cuda.is_available(),
    }
    if prefetch_factor is not None and num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    if persistent_workers:
        loader_kwargs["persistent_workers"] = True

    if batch_sampler is not None:
        loader = DataLoader(dataset, batch_sampler=batch_sampler, **loader_kwargs)
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
            "Sequence DataLoader built for %s: num_workers=%d, pin_memory=%s, prefetch_factor=%s, "
            "persistent_workers=%s, sampler=%s",
            Path(csv_path).name,
            num_workers,
            loader_kwargs["pin_memory"],
            prefetch_factor if prefetch_factor is not None else "None",
            bool(persistent_workers),
            sampler_name,
        )

    return loader
