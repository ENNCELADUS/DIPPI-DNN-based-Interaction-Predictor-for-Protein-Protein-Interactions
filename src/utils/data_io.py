"""Data input utilities for embedding-cache-backed protein pair training."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.embed import (
    EmbeddingCacheManifest,
    ensure_embeddings_ready,
    load_cached_embedding,
)
from src.utils.config import (
    ConfigDict,
    as_bool,
    as_float,
    as_int,
    as_str,
    as_str_list,
    get_section,
)
from src.utils.pair_io import PairRecord, read_pair_records
from src.utils.data_samplers import StagedOHEMBatchSampler

TrainingStage = Literal["pretrain", "finetune"]


class PPIPairDataset(Dataset[dict[str, torch.Tensor]]):
    """PPI dataset that loads precomputed protein embeddings from cache."""

    def __init__(
        self,
        file_path: Path,
        input_dim: int,
        max_sequence_length: int,
        cache_dir: Path,
        embedding_index: dict[str, str],
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")

        self._records: list[PairRecord] = read_pair_records(file_path=file_path)
        self._input_dim = int(input_dim)
        self._max_sequence_length = int(max_sequence_length)
        self._cache_dir = cache_dir
        self._embedding_index = dict(embedding_index)
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
            if protein_id not in self._embedding_index
        )
        if missing_ids:
            preview = ", ".join(missing_ids[:10])
            raise FileNotFoundError(
                f"Embedding index is missing proteins required by {file_path}: "
                f"{preview} (missing={len(missing_ids)})"
            )

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self._records)

    def labels(self) -> list[int]:
        """Return binary labels for all records."""
        return [int(record.label) for record in self._records]

    def _load_embedding(self, protein_id: str) -> torch.Tensor:
        """Load one cached embedding tensor for a protein ID."""
        return load_cached_embedding(
            cache_dir=self._cache_dir,
            index=self._embedding_index,
            protein_id=protein_id,
            expected_input_dim=self._input_dim,
            max_sequence_length=self._max_sequence_length,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one dataset example."""
        record = self._records[index]
        emb_a = self._load_embedding(record.protein_a)
        emb_b = self._load_embedding(record.protein_b)
        len_a = torch.tensor(emb_a.size(0), dtype=torch.long)
        len_b = torch.tensor(emb_b.size(0), dtype=torch.long)
        label = torch.tensor(float(record.label), dtype=torch.float32)
        protein_a_id = torch.tensor(
            self._protein_to_id[record.protein_a], dtype=torch.long
        )
        protein_b_id = torch.tensor(
            self._protein_to_id[record.protein_b], dtype=torch.long
        )
        return {
            "emb_a": emb_a,
            "emb_b": emb_b,
            "len_a": len_a,
            "len_b": len_b,
            "label": label,
            "protein_a_id": protein_a_id,
            "protein_b_id": protein_b_id,
        }


def _collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate dataset samples with independent pair-wise sequence padding."""
    emb_a = pad_sequence([sample["emb_a"] for sample in batch], batch_first=True)
    emb_b = pad_sequence([sample["emb_b"] for sample in batch], batch_first=True)
    return {
        "emb_a": emb_a,
        "emb_b": emb_b,
        "len_a": torch.stack([sample["len_a"] for sample in batch], dim=0),
        "len_b": torch.stack([sample["len_b"] for sample in batch], dim=0),
        "label": torch.stack([sample["label"] for sample in batch], dim=0),
        "protein_a_id": torch.stack(
            [sample["protein_a_id"] for sample in batch], dim=0
        ),
        "protein_b_id": torch.stack(
            [sample["protein_b_id"] for sample in batch], dim=0
        ),
    }


def _build_split_loader(
    split_path: Path,
    config: ConfigDict,
    embedding_cache: EmbeddingCacheManifest,
    input_dim: int,
    max_sequence_length: int,
    seed: int,
    shuffle: bool,
    distributed: bool,
    rank: int,
    world_size: int,
    train_stage: TrainingStage,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Build a split-specific data loader."""
    training_cfg = get_section(config, "training_config")
    data_cfg = get_section(config, "data_config")
    dataloader_cfg = get_section(data_cfg, "dataloader")

    batch_size = as_int(training_cfg.get("batch_size", 8), "training_config.batch_size")
    num_workers = as_int(
        dataloader_cfg.get("num_workers", 0), "data_config.dataloader.num_workers"
    )
    prefetch_factor = dataloader_cfg.get("prefetch_factor")
    if prefetch_factor is not None:
        prefetch_factor = as_int(
            prefetch_factor, "data_config.dataloader.prefetch_factor"
        )
        if prefetch_factor <= 0:
            raise ValueError("data_config.dataloader.prefetch_factor must be positive")
    persistent_workers = as_bool(
        dataloader_cfg.get("persistent_workers", False),
        "data_config.dataloader.persistent_workers",
    )
    pin_memory = as_bool(
        dataloader_cfg.get("pin_memory", False),
        "data_config.dataloader.pin_memory",
    )
    drop_last = as_bool(
        dataloader_cfg.get("drop_last", False), "data_config.dataloader.drop_last"
    )

    dataset = PPIPairDataset(
        file_path=split_path,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
        cache_dir=embedding_cache.cache_dir,
        embedding_index=embedding_cache.index,
    )

    sampler: DistributedSampler[dict[str, torch.Tensor]] | None = None
    batch_sampler: StagedOHEMBatchSampler | None = None
    should_shuffle = shuffle
    sampling_cfg = _sampling_config_for_stage(
        dataloader_cfg=dataloader_cfg, train_stage=train_stage
    )
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
        loader_kwargs: dict[str, object] = {}
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = persistent_workers
            if prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_collate_batch,
            **loader_kwargs,
        )
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        sampler=sampler,
        collate_fn=_collate_batch,
        **loader_kwargs,
    )


def _distributed_barrier_if_initialized() -> None:
    """Run ``dist.barrier`` when a process group is active."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _resolve_training_stage(stage: str) -> TrainingStage:
    """Validate and normalize training-stage name."""
    stage_name = stage.lower()
    if stage_name == "pretrain":
        return "pretrain"
    if stage_name == "finetune":
        return "finetune"
    raise ValueError(f"Unsupported training stage: {stage}")


def _sampling_config_for_stage(
    dataloader_cfg: ConfigDict, train_stage: TrainingStage
) -> ConfigDict:
    """Resolve stage-specific sampling with fallback to global defaults."""
    sampling_raw = dataloader_cfg.get("sampling", {})
    if not isinstance(sampling_raw, dict):
        raise ValueError("data_config.dataloader.sampling must be a mapping")
    stage_key = f"{train_stage}_sampling"
    stage_sampling_raw = dataloader_cfg.get(stage_key)
    if stage_sampling_raw is None:
        return sampling_raw
    if not isinstance(stage_sampling_raw, dict):
        raise ValueError(f"data_config.dataloader.{stage_key} must be a mapping")
    merged = dict(sampling_raw)
    merged.update(stage_sampling_raw)
    return merged


def _configured_split_path_from_key(
    dataloader_cfg: ConfigDict,
    *,
    key: str,
    field_name: str,
) -> Path:
    """Resolve one configured dataset path from dataloader config."""
    return Path(as_str(dataloader_cfg.get(key, ""), field_name))


def _validated_existing_path(path: Path) -> Path:
    """Validate one resolved dataset path."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")
    return path


def resolve_split_paths(
    config: ConfigDict,
    train_stage: TrainingStage | str = "pretrain",
) -> dict[str, Path]:
    """Resolve train/valid/test paths for one training stage.

    Args:
        config: Root configuration mapping.
        train_stage: Training stage used to resolve train/valid datasets.

    Returns:
        Mapping with ``train``, ``valid``, and ``test`` split paths.
    """
    data_cfg = get_section(config, "data_config")
    dataloader_cfg = get_section(data_cfg, "dataloader")
    stage_name = _resolve_training_stage(train_stage)

    if stage_name == "finetune":
        train_key = (
            "finetune_train_dataset"
            if "finetune_train_dataset" in dataloader_cfg
            else "train_dataset"
        )
        valid_key = (
            "finetune_val_dataset"
            if "finetune_val_dataset" in dataloader_cfg
            else "valid_dataset"
        )
        train_field = "data_config.dataloader.finetune_train_dataset"
        valid_field = "data_config.dataloader.finetune_val_dataset"
    else:
        train_key = "train_dataset"
        valid_key = "valid_dataset"
        train_field = "data_config.dataloader.train_dataset"
        valid_field = "data_config.dataloader.valid_dataset"

    train_path = _configured_split_path_from_key(
        dataloader_cfg,
        key=train_key,
        field_name=train_field,
    )
    valid_path = _configured_split_path_from_key(
        dataloader_cfg,
        key=valid_key,
        field_name=valid_field,
    )
    test_path = _configured_split_path_from_key(
        dataloader_cfg,
        key="test_dataset",
        field_name="data_config.dataloader.test_dataset",
    )
    train_path = _validated_existing_path(train_path)
    valid_path = _validated_existing_path(valid_path)
    test_path = _validated_existing_path(test_path)
    return {"train": train_path, "valid": valid_path, "test": test_path}


def _configured_pipeline_stages(config: ConfigDict) -> list[str]:
    """Return normalized pipeline stages requested by config."""
    run_cfg = get_section(config, "run_config")
    raw_stages = run_cfg.get("stages")
    if raw_stages is None:
        raise ValueError("run_config.stages is required")
    stages = [stage.lower() for stage in as_str_list(raw_stages, "run_config.stages")]
    if not stages:
        raise ValueError("run_config.stages must not be empty")
    return stages


def collect_embedding_split_paths(config: ConfigDict) -> list[Path]:
    """Collect split paths required for embedding-cache preparation.

    This includes only the splits required by the configured pipeline stages.
    """
    configured_stages = _configured_pipeline_stages(config=config)
    data_cfg = get_section(config, "data_config")
    dataloader_cfg = get_section(data_cfg, "dataloader")
    ordered_candidates: list[Path] = []

    if "pretrain" in configured_stages:
        pretrain_split_paths = resolve_split_paths(config=config, train_stage="pretrain")
        ordered_candidates.append(pretrain_split_paths["train"])
        ordered_candidates.append(pretrain_split_paths["valid"])

    finetune_train = dataloader_cfg.get("finetune_train_dataset")
    finetune_valid = dataloader_cfg.get("finetune_val_dataset")
    needs_finetune_splits = "finetune" in configured_stages or "evaluate" in configured_stages
    if needs_finetune_splits and (finetune_train is not None or finetune_valid is not None):
        finetune_split_paths = resolve_split_paths(config=config, train_stage="finetune")
        ordered_candidates.append(finetune_split_paths["train"])
        ordered_candidates.append(finetune_split_paths["valid"])

    if not ordered_candidates or "evaluate" in configured_stages:
        test_path = _configured_split_path_from_key(
            dataloader_cfg,
            key="test_dataset",
            field_name="data_config.dataloader.test_dataset",
        )
        ordered_candidates.append(_validated_existing_path(test_path))

    deduplicated: list[Path] = []
    seen: set[str] = set()
    for path in ordered_candidates:
        normalized = str(path.resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        deduplicated.append(path)
    return deduplicated


def ensure_embedding_cache(
    config: ConfigDict,
    split_paths: Sequence[Path],
    input_dim: int,
    max_sequence_length: int,
    distributed: bool = False,
    rank: int = 0,
) -> EmbeddingCacheManifest:
    """Ensure required embeddings exist and return cache manifest."""
    if distributed:
        distributed_initialized = dist.is_available() and dist.is_initialized()
        allow_generation = True if distributed_initialized else rank == 0
        _distributed_barrier_if_initialized()
        embedding_cache = ensure_embeddings_ready(
            config=config,
            split_paths=split_paths,
            input_dim=input_dim,
            max_sequence_length=max_sequence_length,
            allow_generation=allow_generation,
        )
        _distributed_barrier_if_initialized()
        return embedding_cache

    return ensure_embeddings_ready(
        config=config,
        split_paths=split_paths,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
        allow_generation=True,
    )


def build_dataloaders(
    config: ConfigDict,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    train_stage: TrainingStage | str = "pretrain",
    embedding_cache: EmbeddingCacheManifest | None = None,
    embedding_split_paths: Sequence[Path] | None = None,
) -> dict[str, DataLoader[dict[str, torch.Tensor]]]:
    """Build train/valid/test loaders from the global config.

    Args:
        config: Root configuration mapping.
        distributed: Whether distributed data loading is enabled.
        rank: Global rank for distributed loading.
        world_size: Number of distributed processes.
        train_stage: Training stage used to select train/valid datasets.
        embedding_cache: Optional precomputed embedding cache manifest.
        embedding_split_paths: Optional split files used for embedding generation.

    Returns:
        Split dataloader mapping with keys ``train``, ``valid``, and ``test``.

    Raises:
        FileNotFoundError: If benchmark root, split files, or embedding files are missing.
        ValueError: If embeddings are invalid for configured model input dimension.
    """
    run_cfg = get_section(config, "run_config")
    data_cfg = get_section(config, "data_config")
    model_cfg = get_section(config, "model_config")
    benchmark_cfg = get_section(data_cfg, "benchmark")

    benchmark_root = Path(str(benchmark_cfg.get("root_dir", "")))
    if not benchmark_root.exists():
        raise FileNotFoundError(f"Benchmark root does not exist: {benchmark_root}")

    split_path_map = resolve_split_paths(config=config, train_stage=train_stage)
    train_path = split_path_map["train"]
    valid_path = split_path_map["valid"]
    test_path = split_path_map["test"]
    required_embedding_paths = (
        list(embedding_split_paths)
        if embedding_split_paths is not None
        else collect_embedding_split_paths(config=config)
    )

    input_dim = as_int(model_cfg.get("input_dim", 0), "model_config.input_dim")
    max_sequence_length = as_int(
        data_cfg.get("max_sequence_length", 64),
        "data_config.max_sequence_length",
    )
    seed = as_int(run_cfg.get("seed", 0), "run_config.seed")
    stage_name = _resolve_training_stage(train_stage)

    resolved_embedding_cache = embedding_cache
    if resolved_embedding_cache is None:
        resolved_embedding_cache = ensure_embedding_cache(
            config=config,
            split_paths=required_embedding_paths,
            input_dim=input_dim,
            max_sequence_length=max_sequence_length,
            distributed=distributed,
            rank=rank,
        )

    return {
        "train": _build_split_loader(
            split_path=train_path,
            config=config,
            embedding_cache=resolved_embedding_cache,
            input_dim=input_dim,
            max_sequence_length=max_sequence_length,
            seed=seed,
            shuffle=True,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            train_stage=stage_name,
        ),
        "valid": _build_split_loader(
            split_path=valid_path,
            config=config,
            embedding_cache=resolved_embedding_cache,
            input_dim=input_dim,
            max_sequence_length=max_sequence_length,
            seed=seed + 1,
            shuffle=False,
            distributed=False,
            rank=rank,
            world_size=world_size,
            train_stage=stage_name,
        ),
        "test": _build_split_loader(
            split_path=test_path,
            config=config,
            embedding_cache=resolved_embedding_cache,
            input_dim=input_dim,
            max_sequence_length=max_sequence_length,
            seed=seed + 2,
            shuffle=False,
            distributed=False,
            rank=rank,
            world_size=world_size,
            train_stage=stage_name,
        ),
    }
