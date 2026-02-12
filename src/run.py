"""Centralized, config-driven pipeline runner."""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.evaluate import Evaluator
from src.model import V3, V4, V5
from src.train import NoOpStrategy, StagedUnfreezeStrategy, Trainer
from src.train.base import OptimizerConfig, SchedulerConfig
from src.utils.config import (
    ConfigDict,
    as_bool,
    as_float,
    as_int,
    as_str,
    as_str_list,
    extract_model_kwargs,
    get_section,
    load_config,
)
from src.utils.data_io import (
    TrainingStage,
    build_dataloaders,
    collect_embedding_split_paths,
    ensure_embedding_cache,
)
from src.utils.device import resolve_device
from src.utils.distributed import (
    DistributedContext,
    cleanup_distributed,
    distributed_barrier,
    initialize_distributed,
)
from src.utils.early_stop import EarlyStopping
from src.utils.logging import (
    append_csv_row,
    generate_run_id,
    log_stage_event,
    prepare_stage_directories,
    setup_stage_logger,
)
from src.utils.losses import LossConfig
from src.utils.ohem_sample_strategy import OHEMSampleStrategy

ROOT_LOGGER = logging.getLogger(__name__)
AnnealStrategy = Literal["cos", "linear"]
PipelineStage = Literal["pretrain", "finetune", "evaluate"]
ModelFactory = Callable[[ConfigDict], nn.Module]
DEFAULT_TRAINING_VAL_METRICS = ["auprc", "auroc"]
DEFAULT_HEARTBEAT_EVERY_N_STEPS = 20
TRAINING_STAGE_SEQUENCE: tuple[TrainingStage, ...] = ("pretrain", "finetune")
PIPELINE_STAGE_SEQUENCE: tuple[PipelineStage, ...] = (
    "pretrain",
    "finetune",
    "evaluate",
)
TRAINING_STAGE_NAMES: frozenset[str] = frozenset(TRAINING_STAGE_SEQUENCE)
PIPELINE_STAGE_NAMES: frozenset[str] = frozenset(PIPELINE_STAGE_SEQUENCE)
EVAL_CSV_COLUMNS = [
    "split",
    "auroc",
    "auprc",
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "recall",
    "f1",
    "mcc",
]


def _stage_logger_name(model_name: str, stage: str, run_id: str, rank: int) -> str:
    """Return stable logger name for one stage/run/rank tuple."""
    return f"relic.{model_name}.{stage}.{run_id}.rank{rank}"


def _training_logging_config(training_cfg: ConfigDict) -> ConfigDict:
    """Return ``training_config.logging`` mapping with validation."""
    logging_cfg = training_cfg.get("logging", {})
    if not isinstance(logging_cfg, dict):
        raise ValueError("training_config.logging must be a mapping")
    return cast(ConfigDict, logging_cfg)


def _build_v3_model(model_kwargs: ConfigDict) -> nn.Module:
    """Build V3 model instance."""
    return V3(**model_kwargs)


def _build_v4_model(model_kwargs: ConfigDict) -> nn.Module:
    """Build V4 model instance."""
    return V4(**model_kwargs)


def _build_v5_model(model_kwargs: ConfigDict) -> nn.Module:
    """Build V5 model instance."""
    return V5(**model_kwargs)


MODEL_FACTORIES: dict[str, ModelFactory] = {
    "v3": _build_v3_model,
    "v4": _build_v4_model,
    "v5": _build_v5_model,
}


def _parse_anneal_strategy(value: object) -> AnnealStrategy:
    """Parse OneCycle anneal strategy."""
    anneal_strategy = as_str(value, "training_config.scheduler.anneal_strategy").lower()
    if anneal_strategy not in {"cos", "linear"}:
        raise ValueError(
            "training_config.scheduler.anneal_strategy must be 'cos' or 'linear'"
        )
    return cast(AnnealStrategy, anneal_strategy)


def _metrics_from_config(eval_cfg: ConfigDict) -> list[str]:
    """Extract configured metric names."""
    metrics = eval_cfg.get("metrics", [])
    if not isinstance(metrics, Sequence) or isinstance(metrics, (str, bytes)):
        raise ValueError("evaluate.metrics must be a sequence")
    return as_str_list(metrics, "evaluate.metrics")


def _build_loss_config(training_cfg: ConfigDict) -> LossConfig:
    """Build loss configuration from ``training_config.loss``."""
    loss_cfg = training_cfg.get("loss", {})
    if not isinstance(loss_cfg, dict):
        raise ValueError("training_config.loss must be a mapping")
    return LossConfig(
        loss_type=as_str(
            loss_cfg.get("type", "bce_with_logits"), "training_config.loss.type"
        ),
        pos_weight=as_float(
            loss_cfg.get("pos_weight", 1.0), "training_config.loss.pos_weight"
        ),
        label_smoothing=as_float(
            loss_cfg.get("label_smoothing", 0.0), "training_config.loss.label_smoothing"
        ),
    )


def _training_validation_metrics(training_cfg: ConfigDict) -> list[str]:
    """Parse metrics to persist in ``training_step.csv``."""
    logging_cfg = _training_logging_config(training_cfg)
    raw_metrics = logging_cfg.get("validation_metrics", DEFAULT_TRAINING_VAL_METRICS)
    if not isinstance(raw_metrics, Sequence) or isinstance(raw_metrics, (str, bytes)):
        raise ValueError(
            "training_config.logging.validation_metrics must be a sequence"
        )
    metrics = [
        metric.lower()
        for metric in as_str_list(
            raw_metrics, "training_config.logging.validation_metrics"
        )
    ]
    if not metrics:
        raise ValueError("training_config.logging.validation_metrics must not be empty")
    return metrics


def _training_heartbeat_every_n_steps(training_cfg: ConfigDict) -> int:
    """Parse heartbeat interval for trainer progress logs."""
    logging_cfg = _training_logging_config(training_cfg)
    heartbeat_every_n_steps = as_int(
        logging_cfg.get("heartbeat_every_n_steps", DEFAULT_HEARTBEAT_EVERY_N_STEPS),
        "training_config.logging.heartbeat_every_n_steps",
    )
    if heartbeat_every_n_steps < 0:
        raise ValueError("training_config.logging.heartbeat_every_n_steps must be >= 0")
    return heartbeat_every_n_steps


def _build_stage_logger(name: str, log_file: Path, enabled: bool) -> logging.Logger:
    """Create stage logger for rank-aware logging behavior."""
    if enabled:
        return setup_stage_logger(name=name, log_file=log_file)
    logger = logging.getLogger(name)
    logger.propagate = False
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def _rank_from_env() -> int:
    """Parse global rank from environment, defaulting to zero."""
    rank_raw = os.environ.get("RANK", "0")
    try:
        return int(rank_raw)
    except ValueError:
        return 0


def _configure_root_logging() -> None:
    """Configure process-level logging; suppress non-main rank noise."""
    logging.captureWarnings(True)
    if _rank_from_env() == 0:
        logging.basicConfig(level=logging.INFO, force=True)
        return
    logging.basicConfig(level=logging.CRITICAL, force=True)


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return underlying model when wrapped by DDP."""
    if isinstance(model, DistributedDataParallel):
        return cast(nn.Module, model.module)
    return model


def _merged_config(
    base_cfg: Mapping[str, object],
    override_cfg: Mapping[str, object],
) -> ConfigDict:
    """Return a shallow+one-level merged configuration mapping."""
    merged: ConfigDict = dict(base_cfg)
    for key, value in override_cfg.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(value, Mapping):
            nested = dict(base_value)
            nested.update(value)
            merged[key] = nested
            continue
        merged[key] = value
    return merged


def _stage_training_config(config: ConfigDict, stage: TrainingStage) -> ConfigDict:
    """Resolve stage-specific training config with defaults fallback."""
    training_cfg = get_section(config, "training_config")
    default_cfg: ConfigDict = {
        key: value
        for key, value in training_cfg.items()
        if key not in TRAINING_STAGE_NAMES
    }
    stage_override = training_cfg.get(stage)
    if stage_override is None:
        return default_cfg
    if not isinstance(stage_override, Mapping):
        raise ValueError(f"training_config.{stage} must be a mapping")
    return _merged_config(default_cfg, stage_override)


def _config_for_training_stage(config: ConfigDict, stage: TrainingStage) -> ConfigDict:
    """Build effective config for a given training stage."""
    stage_config = dict(config)
    stage_config["training_config"] = _stage_training_config(config=config, stage=stage)
    return stage_config


def _normalize_stage_name(value: object) -> PipelineStage:
    """Parse one stage name from run config."""
    stage_name = as_str(value, "run_config.stages[]").lower()
    if stage_name not in PIPELINE_STAGE_NAMES:
        raise ValueError(f"Unsupported stage name in run_config.stages: {stage_name}")
    return cast(PipelineStage, stage_name)


def _ordered_unique_stages(stages: Sequence[PipelineStage]) -> list[PipelineStage]:
    """Return stage sequence without duplicates while preserving order."""
    deduplicated: list[PipelineStage] = []
    seen: set[str] = set()
    for stage in stages:
        if stage in seen:
            continue
        seen.add(stage)
        deduplicated.append(stage)
    return deduplicated


def _validate_stage_order(stages: Sequence[PipelineStage]) -> None:
    """Validate stage ordering constraints."""
    positions = {stage: index for index, stage in enumerate(stages)}
    if "finetune" in positions and "pretrain" in positions:
        if positions["finetune"] < positions["pretrain"]:
            raise ValueError(
                "run_config.stages must order 'pretrain' before 'finetune'"
            )
    if "evaluate" in positions:
        for training_stage in TRAINING_STAGE_SEQUENCE:
            if (
                training_stage in positions
                and positions["evaluate"] < positions[training_stage]
            ):
                raise ValueError(
                    "run_config.stages must place 'evaluate' after training stages"
                )


def _resolve_stage_sequence(run_cfg: ConfigDict) -> list[PipelineStage]:
    """Resolve execution stages from config."""
    raw_stages = run_cfg.get("stages")
    if raw_stages is not None:
        if not isinstance(raw_stages, Sequence) or isinstance(raw_stages, (str, bytes)):
            raise ValueError("run_config.stages must be a sequence")
        parsed = [_normalize_stage_name(stage) for stage in raw_stages]
        if not parsed:
            raise ValueError("run_config.stages must not be empty")
        stages = _ordered_unique_stages(parsed)
        _validate_stage_order(stages)
        return stages

    mode = as_str(run_cfg.get("mode", "full_pipeline"), "run_config.mode").lower()
    if mode == "train_only":
        return ["pretrain"]
    if mode == "full_pipeline":
        return ["pretrain", "finetune", "evaluate"]
    if mode == "eval_only":
        return ["evaluate"]
    raise ValueError(f"Unsupported run mode: {mode}")


def _stage_run_ids(run_cfg: ConfigDict) -> dict[PipelineStage, str]:
    """Resolve run IDs for each pipeline stage."""
    pretrain_seed_id = run_cfg.get("pretrain_run_id")
    if pretrain_seed_id is None:
        pretrain_seed_id = run_cfg.get("train_run_id")
    return {
        "pretrain": generate_run_id(pretrain_seed_id),
        "finetune": generate_run_id(run_cfg.get("finetune_run_id")),
        "evaluate": generate_run_id(run_cfg.get("eval_run_id")),
    }


def _load_checkpoint_if_provided(
    model: nn.Module,
    checkpoint_path: Path | None,
    device: torch.device,
) -> bool:
    """Load checkpoint when provided, returning whether load occurred."""
    if checkpoint_path is None:
        return False
    _load_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device)
    return True


def _ddp_find_unused_parameters(config: ConfigDict) -> bool:
    """Return DDP ``find_unused_parameters`` setting from config."""
    device_cfg = get_section(config, "device_config")
    explicit_find_unused = device_cfg.get("find_unused_parameters")
    if explicit_find_unused is not None:
        return as_bool(explicit_find_unused, "device_config.find_unused_parameters")

    training_cfg = get_section(config, "training_config")
    candidate_configs: list[ConfigDict] = [training_cfg]
    for stage in TRAINING_STAGE_SEQUENCE:
        stage_cfg = training_cfg.get(stage)
        if isinstance(stage_cfg, dict):
            candidate_configs.append(cast(ConfigDict, stage_cfg))

    for candidate in candidate_configs:
        strategy_cfg = candidate.get("strategy")
        if not isinstance(strategy_cfg, dict):
            continue
        strategy_type = as_str(
            strategy_cfg.get("type", "none"),
            "training_config.strategy.type",
        ).lower()
        if strategy_type == "staged_unfreeze":
            return True
    return False


def _build_stage_runtime(
    model_name: str,
    stage: str,
    run_id: str,
    distributed_context: DistributedContext,
) -> tuple[Path, Path, logging.Logger]:
    """Create artifact directories and stage logger for one stage."""
    log_dir, model_dir = prepare_stage_directories(
        model_name=model_name,
        stage=stage,
        run_id=run_id,
    )
    stage_logger = _build_stage_logger(
        name=_stage_logger_name(
            model_name=model_name,
            stage=stage,
            run_id=run_id,
            rank=distributed_context.rank,
        ),
        log_file=log_dir / "log.log",
        enabled=distributed_context.is_main_process,
    )
    return log_dir, model_dir, stage_logger


def _len_or_unknown(value: object) -> int | str:
    """Return ``len(value)`` when available, otherwise ``'unknown'``."""
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return "unknown"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed CLI namespace with the config path.
    """
    parser = argparse.ArgumentParser(
        description="Run RELIC training/evaluation pipeline."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML."
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Global random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: ConfigDict) -> nn.Module:
    """Build model from ``model_config``.

    Args:
        config: Global run configuration dictionary.

    Returns:
        Instantiated PyTorch model.

    Raises:
        ValueError: If the model name is not supported.
    """
    model_name, model_kwargs = extract_model_kwargs(config)
    factory = MODEL_FACTORIES.get(model_name)
    if factory is not None:
        return factory(model_kwargs)
    raise ValueError(f"Unknown model: {model_name}")


def build_trainer(
    config: ConfigDict,
    model: nn.Module,
    device: torch.device,
    steps_per_epoch: int,
    logger: logging.Logger | None = None,
) -> tuple[Trainer, LossConfig]:
    """Instantiate trainer with optimizer/scheduler configs.

    Args:
        config: Global run configuration dictionary.
        model: Instantiated model.
        device: Target torch device.
        steps_per_epoch: Number of training steps per epoch.
        logger: Optional stage logger for heartbeat messages.

    Returns:
        Configured trainer instance and loss configuration.
    """
    training_cfg = get_section(config, "training_config")
    data_cfg = get_section(config, "data_config")
    dataloader_cfg = get_section(data_cfg, "dataloader")
    optimizer_cfg = get_section(training_cfg, "optimizer")
    scheduler_cfg = get_section(training_cfg, "scheduler")
    sampling_raw = dataloader_cfg.get("sampling", {})
    if not isinstance(sampling_raw, dict):
        raise ValueError("data_config.dataloader.sampling must be a mapping")
    sampling_cfg = sampling_raw

    optimizer_config = OptimizerConfig(
        optimizer_type=as_str(
            optimizer_cfg.get("type", "adamw"), "training_config.optimizer.type"
        ),
        lr=as_float(optimizer_cfg.get("lr", 1e-4), "training_config.optimizer.lr"),
        beta1=as_float(
            optimizer_cfg.get("beta1", 0.9), "training_config.optimizer.beta1"
        ),
        beta2=as_float(
            optimizer_cfg.get("beta2", 0.999), "training_config.optimizer.beta2"
        ),
        eps=as_float(optimizer_cfg.get("eps", 1e-8), "training_config.optimizer.eps"),
        weight_decay=as_float(
            optimizer_cfg.get("weight_decay", 0.0),
            "training_config.optimizer.weight_decay",
        ),
    )
    scheduler_config = SchedulerConfig(
        scheduler_type=as_str(
            scheduler_cfg.get("type", "none"), "training_config.scheduler.type"
        ),
        max_lr=as_float(
            scheduler_cfg.get("max_lr", optimizer_config.lr),
            "training_config.scheduler.max_lr",
        ),
        pct_start=as_float(
            scheduler_cfg.get("pct_start", 0.2),
            "training_config.scheduler.pct_start",
        ),
        div_factor=as_float(
            scheduler_cfg.get("div_factor", 25.0),
            "training_config.scheduler.div_factor",
        ),
        final_div_factor=as_float(
            scheduler_cfg.get("final_div_factor", 10000.0),
            "training_config.scheduler.final_div_factor",
        ),
        anneal_strategy=_parse_anneal_strategy(
            scheduler_cfg.get("anneal_strategy", "cos")
        ),
    )

    sampling_strategy = as_str(
        sampling_cfg.get("strategy", "none"), "data_config.dataloader.sampling.strategy"
    ).lower()
    ohem_strategy = None
    if sampling_strategy == "ohem":
        batch_size = as_int(
            training_cfg.get("batch_size", 8), "training_config.batch_size"
        )
        ohem_strategy = OHEMSampleStrategy(
            target_batch_size=batch_size,
            cap_protein=as_int(
                sampling_cfg.get("cap_protein", 4),
                "data_config.dataloader.sampling.cap_protein",
            ),
            warmup_epochs=as_int(
                sampling_cfg.get("warmup_epochs", 0),
                "data_config.dataloader.sampling.warmup_epochs",
            ),
        )

    device_cfg = get_section(config, "device_config")
    total_epochs = as_int(training_cfg.get("epochs", 1), "training_config.epochs")
    loss_config = _build_loss_config(training_cfg)
    trainer = Trainer(
        model=model,
        device=device,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        loss_config=loss_config,
        use_amp=as_bool(
            device_cfg.get("use_mixed_precision", False),
            "device_config.use_mixed_precision",
        ),
        total_epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        ohem_strategy=ohem_strategy,
        logger=logger,
        heartbeat_every_n_steps=_training_heartbeat_every_n_steps(training_cfg),
    )
    return trainer, loss_config


def build_strategy(config: ConfigDict) -> NoOpStrategy | StagedUnfreezeStrategy:
    """Build optional training strategy from config.

    Args:
        config: Global run configuration dictionary.

    Returns:
        Strategy implementation for training lifecycle hooks.
    """
    training_cfg = get_section(config, "training_config")
    strategy_cfg = training_cfg.get("strategy")
    if not isinstance(strategy_cfg, dict):
        return NoOpStrategy()
    strategy_type = str(strategy_cfg.get("type", "none")).lower()
    if strategy_type == "staged_unfreeze":
        prefixes_value = strategy_cfg.get("initial_trainable_prefixes", ["output_head"])
        if not isinstance(prefixes_value, list):
            raise ValueError("strategy.initial_trainable_prefixes must be a list")
        prefixes = tuple(str(prefix) for prefix in prefixes_value)
        return StagedUnfreezeStrategy(
            unfreeze_epoch=as_int(
                strategy_cfg.get("unfreeze_epoch", 1),
                "training_config.strategy.unfreeze_epoch",
            ),
            initial_trainable_prefixes=prefixes,
        )
    return NoOpStrategy()


def _save_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    """Persist model weights to disk.

    Args:
        model: Model to serialize.
        checkpoint_path: Destination checkpoint path.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_unwrap_model(model).state_dict(), checkpoint_path)


def _load_checkpoint(
    model: nn.Module, checkpoint_path: Path, device: torch.device
) -> None:
    """Load model weights from disk.

    Args:
        model: Model receiving loaded weights.
        checkpoint_path: Source checkpoint path.
        device: Device map target.
    """
    state_dict = torch.load(checkpoint_path, map_location=device)
    _unwrap_model(model).load_state_dict(state_dict)


def run_training_stage(
    stage: str,
    config: ConfigDict,
    model: nn.Module,
    device: torch.device,
    dataloaders: dict[str, torch.utils.data.DataLoader[dict[str, torch.Tensor]]],
    run_id: str,
    distributed_context: DistributedContext,
) -> Path:
    """Run stage training loop.

    Args:
        stage: Training stage name (`pretrain` or `finetune`).
        config: Global run configuration dictionary.
        model: Model to train.
        device: Target torch device.
        dataloaders: Split dataloaders keyed by `train`, `valid`, and `test`.
        run_id: Stage run identifier.
        distributed_context: Distributed process metadata.

    Returns:
        Path to the best checkpoint produced during the stage.
    """
    model_name, _ = extract_model_kwargs(config)
    log_dir, model_dir, stage_logger = _build_stage_runtime(
        model_name=model_name,
        stage=stage,
        run_id=run_id,
        distributed_context=distributed_context,
    )
    if distributed_context.is_main_process:
        log_stage_event(
            stage_logger,
            "stage_start",
            run_id=run_id,
        )
    training_cfg = get_section(config, "training_config")
    validation_metrics = _training_validation_metrics(training_cfg)

    trainer, loss_config = build_trainer(
        config=config,
        model=model,
        device=device,
        steps_per_epoch=len(dataloaders["train"]),
        logger=stage_logger,
    )
    strategy = build_strategy(config)

    monitor_metric = as_str(
        training_cfg.get("monitor_metric", "auprc"), "training_config.monitor_metric"
    ).lower()
    evaluator_metrics = sorted(set(validation_metrics + [monitor_metric]))
    evaluator = Evaluator(metrics=evaluator_metrics, loss_config=loss_config)
    monitor_key = f"val_{monitor_metric}"
    patience = as_int(
        training_cfg.get("early_stopping_patience", 5),
        "training_config.early_stopping_patience",
    )
    early_stopping = EarlyStopping(patience=patience, mode="max")
    epochs = as_int(training_cfg.get("epochs", 1), "training_config.epochs")
    save_best_only = as_bool(
        get_section(config, "run_config").get("save_best_only", True),
        "run_config.save_best_only",
    )

    best_checkpoint_path = model_dir / "best_model.pth"
    csv_path = log_dir / "training_step.csv"
    csv_headers = [
        "Epoch",
        "Epoch Time",
        "Train Loss",
        "Val Loss",
        *[f"Val {metric}" for metric in validation_metrics],
        "Learning Rate",
    ]
    if distributed_context.is_main_process:
        log_stage_event(
            stage_logger,
            "train_config",
            epochs=epochs,
            monitor=monitor_metric,
            patience=patience,
        )
    strategy.on_train_begin(trainer)

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        if distributed_context.is_main_process:
            log_stage_event(stage_logger, "epoch_start", epoch=epoch + 1)
        train_sampler = dataloaders["train"].sampler
        train_batch_sampler = getattr(dataloaders["train"], "batch_sampler", None)
        set_epoch_fn = getattr(train_batch_sampler, "set_epoch", None)
        if callable(set_epoch_fn):
            set_epoch_fn(epoch)
        elif distributed_context.is_distributed and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        strategy.on_epoch_begin(trainer, epoch)
        train_stats = trainer.train_one_epoch(dataloaders["train"], epoch_index=epoch)
        model.eval()
        with torch.no_grad():
            val_stats = evaluator.evaluate(
                model=model,
                data_loader=dataloaders["valid"],
                device=device,
                prefix="val",
            )
        model.train()
        strategy.on_epoch_end(trainer, epoch)
        epoch_seconds = time.perf_counter() - epoch_start

        row: dict[str, float | int | str] = {
            "Epoch": epoch + 1,
            "Epoch Time": epoch_seconds,
            "Train Loss": train_stats["loss"],
            "Val Loss": float(val_stats.get("val_loss", 0.0)),
            "Learning Rate": train_stats["lr"],
        }
        for metric in validation_metrics:
            row[f"Val {metric}"] = float(val_stats.get(f"val_{metric}", 0.0))
        if distributed_context.is_main_process:
            append_csv_row(csv_path=csv_path, row=row, fieldnames=csv_headers)
            log_stage_event(stage_logger, "csv_written", epoch=epoch + 1)

        monitor_value = float(val_stats.get(monitor_key, 0.0))
        should_stop = False
        if distributed_context.is_main_process:
            improved, should_stop = early_stopping.update(monitor_value)
            if improved:
                _save_checkpoint(model=model, checkpoint_path=best_checkpoint_path)
                log_stage_event(
                    stage_logger,
                    "best_saved",
                    epoch=epoch + 1,
                    monitor=monitor_key,
                    value=monitor_value,
                )
            if not save_best_only:
                epoch_checkpoint_path = (
                    model_dir / f"checkpoint_epoch_{epoch + 1:03d}.pth"
                )
                _save_checkpoint(
                    model=model,
                    checkpoint_path=epoch_checkpoint_path,
                )
                log_stage_event(
                    stage_logger,
                    "checkpoint_saved",
                    epoch=epoch + 1,
                )

        if distributed_context.is_main_process:
            val_metric_fields = {
                f"val_{m}": float(val_stats.get(f"val_{m}", 0.0))
                for m in validation_metrics
            }
            log_stage_event(
                stage_logger,
                "epoch_done",
                epoch=epoch + 1,
                time=epoch_seconds,
                train_loss=train_stats["loss"],
                val_loss=float(val_stats.get("val_loss", 0.0)),
                **val_metric_fields,
            )
        if distributed_context.is_distributed:
            stop_flag = torch.tensor(
                [1 if should_stop else 0], device=device, dtype=torch.int64
            )
            dist.broadcast(stop_flag, src=0)
            should_stop = bool(int(stop_flag.item()))
        if should_stop:
            if distributed_context.is_main_process:
                log_stage_event(stage_logger, "early_stop", epoch=epoch + 1)
            break

    if distributed_context.is_main_process and not best_checkpoint_path.exists():
        _save_checkpoint(model=model, checkpoint_path=best_checkpoint_path)
        log_stage_event(stage_logger, "fallback_saved")
    if distributed_context.is_main_process:
        log_stage_event(
            stage_logger,
            "stage_done",
            run_id=run_id,
        )
    distributed_barrier(distributed_context)
    return best_checkpoint_path


def run_evaluation_stage(
    config: ConfigDict,
    model: nn.Module,
    device: torch.device,
    dataloaders: dict[str, torch.utils.data.DataLoader[dict[str, torch.Tensor]]],
    run_id: str,
    checkpoint_path: Path,
    distributed_context: DistributedContext,
) -> dict[str, float]:
    """Run test evaluation and persist ``evaluate.csv``.

    Args:
        config: Global run configuration dictionary.
        model: Model to evaluate.
        device: Target torch device.
        dataloaders: Split dataloaders keyed by `train`, `valid`, and `test`.
        run_id: Evaluation run identifier.
        checkpoint_path: Checkpoint to evaluate.
        distributed_context: Distributed process metadata.

    Returns:
        Dictionary of computed test metrics.
    """
    model_name, _ = extract_model_kwargs(config)
    log_dir, _, logger = _build_stage_runtime(
        model_name=model_name,
        stage="evaluate",
        run_id=run_id,
        distributed_context=distributed_context,
    )
    if distributed_context.is_main_process:
        log_stage_event(
            logger,
            "stage_start",
            run_id=run_id,
            checkpoint=checkpoint_path,
        )
    _load_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device)
    if distributed_context.is_main_process:
        log_stage_event(logger, "checkpoint_loaded", path=checkpoint_path)
    model.eval()
    eval_cfg = get_section(config, "evaluate")
    training_cfg = get_section(config, "training_config")
    configured_metrics = _metrics_from_config(eval_cfg)
    metrics_to_compute = sorted(set(configured_metrics + EVAL_CSV_COLUMNS[1:]))
    evaluator = Evaluator(
        metrics=metrics_to_compute, loss_config=_build_loss_config(training_cfg)
    )
    with torch.no_grad():
        metrics = evaluator.evaluate(
            model=model,
            data_loader=dataloaders["test"],
            device=device,
            prefix=None,
        )
    if distributed_context.is_main_process:
        csv_row: dict[str, float | int | str] = {"split": "test"}
        for metric_name in EVAL_CSV_COLUMNS[1:]:
            csv_row[metric_name] = float(metrics.get(metric_name, 0.0))
        append_csv_row(
            csv_path=log_dir / "evaluate.csv",
            row=csv_row,
            fieldnames=EVAL_CSV_COLUMNS,
        )
        log_stage_event(logger, "evaluation_metrics", **csv_row)
        log_stage_event(logger, "csv_written", path=log_dir / "evaluate.csv")
        log_stage_event(
            logger,
            "stage_done",
            run_id=run_id,
        )
    distributed_barrier(distributed_context)
    return metrics


def execute_pipeline(config: ConfigDict) -> None:
    """Execute pipeline according to configured run mode.

    Args:
        config: Global run configuration dictionary.

    Raises:
        ValueError: If mode is unsupported or required checkpoint is missing.
    """
    run_cfg = get_section(config, "run_config")
    device_cfg = get_section(config, "device_config")
    seed = as_int(run_cfg.get("seed", 0), "run_config.seed")
    set_global_seed(seed=seed)
    distributed_context = initialize_distributed(
        ddp_enabled=as_bool(
            device_cfg.get("ddp_enabled", False), "device_config.ddp_enabled"
        )
    )
    ddp_find_unused_parameters = _ddp_find_unused_parameters(config)
    try:
        mode_value = run_cfg.get("mode", "full_pipeline")
        mode_label = mode_value if isinstance(mode_value, str) else "custom_stages"
        selected_stages = _resolve_stage_sequence(run_cfg=run_cfg)
        stage_run_map = _stage_run_ids(run_cfg=run_cfg)
        load_checkpoint_value = run_cfg.get("load_checkpoint_path")
        load_checkpoint_path = (
            Path(str(load_checkpoint_value))
            if isinstance(load_checkpoint_value, str) and load_checkpoint_value
            else None
        )
        model_name, _ = extract_model_kwargs(config)
        stage_loggers: dict[PipelineStage, logging.Logger] = {}
        for stage in selected_stages:
            _, _, stage_logger = _build_stage_runtime(
                model_name=model_name,
                stage=stage,
                run_id=stage_run_map[stage],
                distributed_context=distributed_context,
            )
            stage_loggers[stage] = stage_logger
            if distributed_context.is_main_process:
                log_stage_event(
                    stage_logger,
                    "startup",
                    mode=mode_label,
                    run_id=stage_run_map[stage],
                    seed=seed,
                    rank=distributed_context.rank,
                    world_size=distributed_context.world_size,
                    stages=",".join(selected_stages),
                )
        requested_device = as_str(
            device_cfg.get("device", "cpu"), "device_config.device"
        )
        device = resolve_device(requested_device)
        if distributed_context.is_distributed and device.type == "cuda":
            device = torch.device("cuda", distributed_context.local_rank)
        if distributed_context.is_main_process:
            for stage in selected_stages:
                log_stage_event(
                    stage_loggers[stage],
                    "device",
                    resolved_device=device,
                )

        data_cfg = get_section(config, "data_config")
        model_cfg = get_section(config, "model_config")
        embedding_split_paths = collect_embedding_split_paths(config=config)
        embedding_cache = ensure_embedding_cache(
            config=config,
            split_paths=embedding_split_paths,
            input_dim=as_int(model_cfg.get("input_dim", 0), "model_config.input_dim"),
            max_sequence_length=as_int(
                data_cfg.get("max_sequence_length", 64),
                "data_config.max_sequence_length",
            ),
            distributed=distributed_context.is_distributed,
            rank=distributed_context.rank,
        )

        model = build_model(config=config).to(device)
        if distributed_context.is_main_process:
            first_stage = selected_stages[0]
            log_stage_event(
                stage_loggers[first_stage],
                "model_init",
                model=model_name,
                params=sum(parameter.numel() for parameter in model.parameters()),
            )
        if distributed_context.is_distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[distributed_context.local_rank]
                if device.type == "cuda"
                else None,
                find_unused_parameters=ddp_find_unused_parameters,
            )
        if distributed_context.is_main_process:
            for stage in selected_stages:
                log_stage_event(
                    stage_loggers[stage],
                    "ddp_ready",
                    wrapped=distributed_context.is_distributed,
                )

        active_checkpoint = load_checkpoint_path
        last_training_stage: TrainingStage | None = None

        for stage in selected_stages:
            if stage not in TRAINING_STAGE_NAMES:
                continue
            training_stage = cast(TrainingStage, stage)
            stage_config = _config_for_training_stage(
                config=config, stage=training_stage
            )

            if training_stage == "finetune":
                if active_checkpoint is None:
                    raise ValueError(
                        "load_checkpoint_path is required when finetune runs without pretrain"
                    )
                if (
                    _load_checkpoint_if_provided(
                        model=model,
                        checkpoint_path=active_checkpoint,
                        device=device,
                    )
                    and distributed_context.is_main_process
                ):
                    log_stage_event(
                        stage_loggers[training_stage],
                        "checkpoint_loaded",
                        path=active_checkpoint,
                    )

            dataloaders = build_dataloaders(
                config=stage_config,
                distributed=distributed_context.is_distributed,
                rank=distributed_context.rank,
                world_size=distributed_context.world_size,
                train_stage=training_stage,
                embedding_cache=embedding_cache,
                embedding_split_paths=embedding_split_paths,
            )
            if distributed_context.is_main_process:
                log_stage_event(
                    stage_loggers[training_stage],
                    "data_ready",
                    train=_len_or_unknown(dataloaders["train"].dataset),
                    valid=_len_or_unknown(dataloaders["valid"].dataset),
                    test=_len_or_unknown(dataloaders["test"].dataset),
                )
                log_stage_event(stage_loggers[training_stage], "begin_training")

            active_checkpoint = run_training_stage(
                stage=training_stage,
                config=stage_config,
                model=model,
                device=device,
                dataloaders=dataloaders,
                run_id=stage_run_map[training_stage],
                distributed_context=distributed_context,
            )
            last_training_stage = training_stage
            if distributed_context.is_main_process:
                log_stage_event(
                    stage_loggers[training_stage],
                    "end_training",
                    checkpoint=active_checkpoint,
                )

        if "evaluate" in selected_stages:
            if active_checkpoint is None:
                raise ValueError(
                    "load_checkpoint_path is required for evaluate stage when no training stage ran"
                )
            eval_base_stage: TrainingStage = (
                last_training_stage if last_training_stage is not None else "pretrain"
            )
            eval_config = _config_for_training_stage(
                config=config, stage=eval_base_stage
            )
            eval_dataloaders = build_dataloaders(
                config=eval_config,
                distributed=distributed_context.is_distributed,
                rank=distributed_context.rank,
                world_size=distributed_context.world_size,
                train_stage=eval_base_stage,
                embedding_cache=embedding_cache,
                embedding_split_paths=embedding_split_paths,
            )
            if distributed_context.is_main_process:
                log_stage_event(
                    stage_loggers["evaluate"],
                    "data_ready",
                    train=_len_or_unknown(eval_dataloaders["train"].dataset),
                    valid=_len_or_unknown(eval_dataloaders["valid"].dataset),
                    test=_len_or_unknown(eval_dataloaders["test"].dataset),
                )
                log_stage_event(stage_loggers["evaluate"], "begin_evaluation")
            run_evaluation_stage(
                config=eval_config,
                model=model,
                device=device,
                dataloaders=eval_dataloaders,
                run_id=stage_run_map["evaluate"],
                checkpoint_path=active_checkpoint,
                distributed_context=distributed_context,
            )
            if distributed_context.is_main_process:
                log_stage_event(stage_loggers["evaluate"], "end_evaluation")
        return
    finally:
        cleanup_distributed(distributed_context)


def main() -> None:
    """Run CLI entrypoint."""
    _configure_root_logging()
    args = parse_args()
    config = load_config(args.config)
    ROOT_LOGGER.info("Loaded config: %s", args.config)
    execute_pipeline(config=config)


if __name__ == "__main__":
    main()
