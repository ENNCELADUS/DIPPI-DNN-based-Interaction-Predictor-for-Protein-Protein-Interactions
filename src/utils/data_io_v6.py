"""Sequence-native dataloaders for V6 (ESM3+LoRA) model runs."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.utils.config import ConfigDict, as_bool, as_float, as_int, as_str, get_section
from src.utils.data_io import TrainingStage, resolve_split_paths
from src.utils.data_samplers import StagedOHEMBatchSampler
from src.utils.pair_io import (
    PairRecord,
    collect_required_protein_ids,
    read_pair_records,
)
from src.utils.sequence_io import load_required_sequences


class SequencePairDataset(Dataset[dict[str, object]]):
    """PPI dataset that returns raw sequence pairs for sequence-native models."""

    def __init__(self, file_path: Path, sequences: dict[str, str]) -> None:
        self._records: list[PairRecord] = read_pair_records(file_path=file_path)
        self._sequences = dict(sequences)
        proteins = sorted(
            {record.protein_a for record in self._records}
            | {record.protein_b for record in self._records}
        )
        self._protein_to_id = {protein: index for index, protein in enumerate(proteins)}

        required_ids = {record.protein_a for record in self._records}
        required_ids.update(record.protein_b for record in self._records)
        missing_ids = sorted(
            protein_id
            for protein_id in required_ids
            if protein_id not in self._sequences
        )
        if missing_ids:
            preview = ", ".join(missing_ids[:10])
            raise FileNotFoundError(
                f"Sequence lookup is missing proteins required by {file_path}: "
                f"{preview} (missing={len(missing_ids)})"
            )

    def __len__(self) -> int:
        return len(self._records)

    def labels(self) -> list[int]:
        return [int(record.label) for record in self._records]

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self._records[index]
        label = torch.tensor(float(record.label), dtype=torch.float32)
        protein_a_id = torch.tensor(
            self._protein_to_id[record.protein_a], dtype=torch.long
        )
        protein_b_id = torch.tensor(
            self._protein_to_id[record.protein_b], dtype=torch.long
        )
        return {
            "seq_a": self._sequences[record.protein_a],
            "seq_b": self._sequences[record.protein_b],
            "label": label,
            "protein_a_id": protein_a_id,
            "protein_b_id": protein_b_id,
        }


def _collate_sequence_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    """Collate sequence-pair examples into a batched dictionary."""
    seq_a = [str(sample["seq_a"]) for sample in batch]
    seq_b = [str(sample["seq_b"]) for sample in batch]

    labels_list: list[torch.Tensor] = []
    protein_a_ids_list: list[torch.Tensor] = []
    protein_b_ids_list: list[torch.Tensor] = []
    for sample in batch:
        label_value = sample.get("label")
        protein_a_id_value = sample.get("protein_a_id")
        protein_b_id_value = sample.get("protein_b_id")
        if not isinstance(label_value, torch.Tensor):
            raise TypeError("label must be a torch.Tensor")
        if not isinstance(protein_a_id_value, torch.Tensor):
            raise TypeError("protein_a_id must be a torch.Tensor")
        if not isinstance(protein_b_id_value, torch.Tensor):
            raise TypeError("protein_b_id must be a torch.Tensor")
        labels_list.append(label_value)
        protein_a_ids_list.append(protein_a_id_value)
        protein_b_ids_list.append(protein_b_id_value)

    labels = torch.stack(labels_list, dim=0)
    protein_a_ids = torch.stack(protein_a_ids_list, dim=0)
    protein_b_ids = torch.stack(protein_b_ids_list, dim=0)
    return {
        "seq_a": seq_a,
        "seq_b": seq_b,
        "label": labels,
        "protein_a_id": protein_a_ids,
        "protein_b_id": protein_b_ids,
    }


def _build_split_loader_v6(
    split_path: Path,
    config: ConfigDict,
    sequences: dict[str, str],
    seed: int,
    shuffle: bool,
    distributed: bool,
    rank: int,
    world_size: int,
) -> DataLoader[dict[str, object]]:
    """Build one split dataloader for sequence-native training/evaluation."""
    training_cfg = get_section(config, "training_config")
    data_cfg = get_section(config, "data_config")
    dataloader_cfg = get_section(data_cfg, "dataloader")

    batch_size = as_int(training_cfg.get("batch_size", 8), "training_config.batch_size")
    num_workers = as_int(
        dataloader_cfg.get("num_workers", 0), "data_config.dataloader.num_workers"
    )
    pin_memory = as_bool(
        dataloader_cfg.get("pin_memory", False),
        "data_config.dataloader.pin_memory",
    )
    drop_last = as_bool(
        dataloader_cfg.get("drop_last", False), "data_config.dataloader.drop_last"
    )

    dataset = SequencePairDataset(file_path=split_path, sequences=sequences)
    sampler: DistributedSampler[dict[str, object]] | None = None
    batch_sampler: StagedOHEMBatchSampler | None = None
    should_shuffle = shuffle

    sampling_raw = dataloader_cfg.get("sampling", {})
    if not isinstance(sampling_raw, dict):
        raise ValueError("data_config.dataloader.sampling must be a mapping")
    sampling_cfg = sampling_raw
    sampling_strategy = as_str(
        sampling_cfg.get("strategy", "none"),
        "data_config.dataloader.sampling.strategy",
    ).lower()
    is_train_loader = shuffle

    if is_train_loader and sampling_strategy == "ohem":
        labels = dataset.labels()
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        natural_ratio = float(neg_count) / float(max(1, pos_count))
        batch_sampler = StagedOHEMBatchSampler(
            labels=labels,
            batch_size=batch_size,
            warmup_pos_neg_ratio=as_float(
                sampling_cfg.get("warmup_pos_neg_ratio", natural_ratio),
                "data_config.dataloader.sampling.warmup_pos_neg_ratio",
            ),
            warmup_epochs=as_int(
                sampling_cfg.get("warmup_epochs", 0),
                "data_config.dataloader.sampling.warmup_epochs",
            ),
            pool_multiplier=as_int(
                sampling_cfg.get("pool_multiplier", 32),
                "data_config.dataloader.sampling.pool_multiplier",
            ),
            cap_protein=as_int(
                sampling_cfg.get("cap_protein", 4),
                "data_config.dataloader.sampling.cap_protein",
            ),
            rank=rank if distributed else 0,
            world_size=world_size if distributed else 1,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )
        should_shuffle = False

    if distributed and batch_sampler is None:
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        should_shuffle = False

    if batch_sampler is not None:
        return DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_collate_sequence_batch,
        )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        sampler=sampler,
        collate_fn=_collate_sequence_batch,
    )


def build_dataloaders_v6(
    config: ConfigDict,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    train_stage: TrainingStage | str = "pretrain",
) -> dict[str, DataLoader[dict[str, object]]]:
    """Build V6 sequence-native train/valid/test dataloaders."""
    run_cfg = get_section(config, "run_config")
    split_path_map = resolve_split_paths(config=config, train_stage=train_stage)
    required_ids = collect_required_protein_ids(split_path_map.values())
    sequences = load_required_sequences(config=config, required_ids=required_ids)
    seed = as_int(run_cfg.get("seed", 0), "run_config.seed")

    return {
        "train": _build_split_loader_v6(
            split_path=split_path_map["train"],
            config=config,
            sequences=sequences,
            seed=seed,
            shuffle=True,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        ),
        "valid": _build_split_loader_v6(
            split_path=split_path_map["valid"],
            config=config,
            sequences=sequences,
            seed=seed + 1,
            shuffle=False,
            distributed=False,
            rank=rank,
            world_size=world_size,
        ),
        "test": _build_split_loader_v6(
            split_path=split_path_map["test"],
            config=config,
            sequences=sequences,
            seed=seed + 2,
            shuffle=False,
            distributed=False,
            rank=rank,
            world_size=world_size,
        ),
    }
