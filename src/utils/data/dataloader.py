"""Common dataloader construction helpers shared by embedding/sequence loaders."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, DistributedSampler

from src.utils.distributed import get_rank, get_world_size
from src.utils.samplers import ImbalancedBatchSampler, StagedOHEMBatchSampler


class DistributedBatchSampler:
    """
    Evenly shard a batch sampler across DDP ranks while keeping step counts aligned.
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
        """Forward epoch to wrapped sampler and refresh computed lengths."""
        if hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)
        self._total_batches = len(self.batch_sampler)
        self._max_per_rank = (
            math.ceil(self._total_batches / self.num_replicas)
            if self.num_replicas > 0
            else 0
        )


def resolve_dataloader_settings(
    dataloader_cfg: Optional[Dict[str, Any]],
    *,
    num_workers_override: Optional[int] = None,
    pin_memory_override: Optional[bool] = None,
) -> Dict[str, Any]:
    """Normalize dataloader settings from config + explicit overrides."""
    cfg = dataloader_cfg.copy() if dataloader_cfg else {}
    if num_workers_override is not None:
        cfg["num_workers"] = int(num_workers_override)
    if pin_memory_override is not None:
        cfg["pin_memory"] = bool(pin_memory_override)

    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", torch.cuda.is_available()))
    drop_last = bool(cfg.get("drop_last", False))
    prefetch_factor = cfg.get("prefetch_factor")
    persistent_workers = cfg.get("persistent_workers")

    if num_workers <= 0:
        prefetch_factor = None
        persistent_workers = False
    else:
        if prefetch_factor is None:
            prefetch_factor = 2
        if persistent_workers is None:
            persistent_workers = True

    return {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "prefetch_factor": prefetch_factor,
        "persistent_workers": bool(persistent_workers),
    }


def build_sampler_components(
    *,
    dataset: Any,
    labels: List[int],
    batch_size: int,
    sampling_cfg: Optional[Dict[str, Any]],
    ddp: bool,
    shuffle: bool,
    drop_last: bool,
) -> Tuple[Optional[Any], Optional[DistributedSampler], bool]:
    """Build batch sampler / sampler tuple for standard, imbalanced, or OHEM modes."""
    batch_sampler: Optional[Any] = None
    sampler: Optional[DistributedSampler] = None
    sampling_cfg = sampling_cfg or {}
    sampling_strategy = sampling_cfg.get("strategy")

    if sampling_strategy == "imbalanced":
        ratio = float(sampling_cfg.get("pos_neg_ratio", 3.0))
        base_sampler = ImbalancedBatchSampler(
            labels=labels,
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
            labels=labels,
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

    return batch_sampler, sampler, shuffle


def build_dataloader_from_components(
    *,
    dataset: Any,
    batch_size: int,
    shuffle: bool,
    sampler: Optional[DistributedSampler],
    batch_sampler: Optional[Any],
    drop_last: bool,
    collate_fn: Any,
    settings: Dict[str, Any],
) -> DataLoader:
    """Construct torch DataLoader from resolved samplers and settings."""
    loader_kwargs: Dict[str, Any] = {
        "num_workers": int(settings["num_workers"]),
        "collate_fn": collate_fn,
        "pin_memory": bool(settings["pin_memory"]) and torch.cuda.is_available(),
    }
    if settings["prefetch_factor"] is not None and int(settings["num_workers"]) > 0:
        loader_kwargs["prefetch_factor"] = int(settings["prefetch_factor"])
    if settings["persistent_workers"]:
        loader_kwargs["persistent_workers"] = True

    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, **loader_kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=drop_last,
        **loader_kwargs,
    )


def sampler_name(batch_sampler: Optional[Any], sampler: Optional[Any]) -> str:
    """Return friendly sampler name for logging."""
    if batch_sampler is not None:
        return batch_sampler.__class__.__name__
    if sampler is not None:
        return sampler.__class__.__name__
    return "None"
