"""
Finetune stage runner for DIPPI pipeline.

This module implements the finetune stage orchestration:
- Loads pretrained checkpoint
- Applies freeze/unfreeze strategy (e.g., StagedUnfreeze)
- Builds Trainer and Evaluator from config
- Runs training loop with validation every epoch
- Handles strategy lifecycle hooks (on_epoch_begin, on_epoch_end)
- Handles checkpointing (best model + optional per-epoch)
- Manages early stopping based on monitored metric
- Logs metrics to CSV and log files

Called by: run.py main orchestrator
"""

import logging
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader

from src.train.base import Trainer
from src.train.strategies import StagedUnfreeze
from src.finetune.distribution_alignment import DistributionAligner
from src.evaluate.base import Evaluator
from src.utils.checkpoint import load_checkpoint, save_checkpoint, maybe_save_best
from src.utils.early_stop import check_early_stop
from src.utils.logging import append_row
from src.utils.distributed import barrier, get_world_size, is_main_process
from src.utils.samplers import StagedHardNegativeBatchSampler


def _move_batch_to_device(
    batch: Dict[str, Any], device: torch.device
) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _normalize_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 2:
        if logits.size(1) == 2:
            return logits[:, 1]
        if logits.size(1) == 1:
            return logits.squeeze(1)
    return logits


def _extract_logits(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, dict) and "logits" in outputs:
        return outputs["logits"]
    if isinstance(outputs, torch.Tensor):
        return outputs
    raise ValueError("Model outputs must include logits for hard-negative mining.")


def _iter_chunks(items: list[int], chunk_size: int):
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _get_base_batch_sampler(
    loader: DataLoader,
) -> tuple[Optional[Any], Optional[Any]]:
    batch_sampler = getattr(loader, "batch_sampler", None)
    if batch_sampler is None:
        return None, None
    inner = batch_sampler
    if hasattr(batch_sampler, "batch_sampler"):
        inner = batch_sampler.batch_sampler
    return inner, batch_sampler


def _set_sampler_epoch(loader: DataLoader, epoch: int) -> None:
    batch_sampler = getattr(loader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)


def _broadcast_hard_scores(
    scores: list[float],
    device: torch.device,
) -> list[float]:
    if not dist.is_available() or not dist.is_initialized():
        return scores

    backend = dist.get_backend()
    tensor_device = device if backend == "nccl" else torch.device("cpu")
    scores_tensor = torch.tensor(scores, dtype=torch.float32, device=tensor_device)
    dist.broadcast(scores_tensor, src=0)
    if scores_tensor.device.type != "cpu":
        scores_tensor = scores_tensor.cpu()
    return scores_tensor.tolist()


def _sync_hard_scores_via_file(
    *,
    scores: list[float],
    log_dir: Path,
    epoch: int,
    timeout_sec: int,
) -> list[float]:
    sync_dir = log_dir / "hard_scores"
    sync_dir.mkdir(parents=True, exist_ok=True)
    scores_path = sync_dir / f"hard_scores_epoch_{epoch}.npy"
    ready_path = sync_dir / f"hard_scores_epoch_{epoch}.ready"

    if is_main_process():
        tmp_path = sync_dir / f"hard_scores_epoch_{epoch}.tmp.npy"
        np.save(tmp_path, np.array(scores, dtype=np.float32))
        tmp_path.replace(scores_path)
        ready_path.write_text("ready", encoding="utf-8")
        return scores

    start = time.time()
    while not ready_path.exists():
        if time.time() - start > timeout_sec:
            raise TimeoutError(
                f"Timed out waiting for hard scores file after {timeout_sec} seconds"
            )
        time.sleep(5)

    loaded = np.load(scores_path)
    return loaded.astype(np.float32).tolist()


def _refresh_hard_scores_epoch(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    sampler: StagedHardNegativeBatchSampler,
    device: torch.device,
    epoch: int,
    sampling_cfg: Dict[str, Any],
    amp_dtype: Optional[torch.dtype],
) -> dict:
    dataset = train_loader.dataset
    collate_fn = train_loader.collate_fn
    if collate_fn is None:
        raise ValueError("train_loader must define a collate_fn for mining")

    candidate_multiplier = float(sampling_cfg.get("candidate_multiplier", 8.0))
    ema_alpha = float(sampling_cfg.get("ema_alpha", 0.2))
    score_batch_size = int(sampling_cfg.get("score_batch_size", 256))
    if score_batch_size <= 0:
        score_batch_size = 256
    max_candidates_per_epoch = sampling_cfg.get("max_candidates_per_epoch", 2_000_000)
    max_candidates_per_epoch = int(max_candidates_per_epoch)
    clear_cuda_cache_every = sampling_cfg.get("clear_cuda_cache_every", 200)
    clear_cuda_cache_every = int(clear_cuda_cache_every)

    pos_indices = sampler.get_pos_indices()
    neg_indices = sampler.get_neg_indices()

    rng_seed = int(sampling_cfg.get("mining_seed", 0)) + epoch
    rng = random.Random(rng_seed)
    if sampler.shuffle:
        rng.shuffle(pos_indices)

    model_was_training = model.training
    model.eval()

    total_scored = 0
    chunk_counter = 0
    with torch.inference_mode():
        for start in range(0, len(pos_indices), sampler.pos_per_batch):
            pos_batch = pos_indices[start : start + sampler.pos_per_batch]
            if not pos_batch:
                continue

            neg_count = sampler.negatives_for_batch(len(pos_batch))
            if neg_count <= 0:
                continue

            candidate_count = int(math.ceil(candidate_multiplier * neg_count))
            if candidate_count <= 0:
                continue

            remaining = candidate_count
            while remaining > 0:
                if (
                    max_candidates_per_epoch is not None
                    and total_scored >= max_candidates_per_epoch
                ):
                    break

                step = min(score_batch_size, remaining)
                if (
                    max_candidates_per_epoch is not None
                    and total_scored + step > max_candidates_per_epoch
                ):
                    step = max_candidates_per_epoch - total_scored
                if step <= 0:
                    break

                chunk = rng.choices(neg_indices, k=step)
                batch_items = [dataset[idx] for idx in chunk]
                batch = collate_fn(batch_items)
                batch = _move_batch_to_device(batch, device)

                if amp_dtype is not None:
                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                        outputs = model(batch)
                else:
                    outputs = model(batch)

                logits = _normalize_logits(_extract_logits(outputs))
                scores = F.softplus(logits).detach().float().cpu().tolist()
                sampler.update_hard_scores(chunk, scores, ema_alpha=ema_alpha)
                total_scored += len(chunk)
                remaining -= len(chunk)

                chunk_counter += 1
                if (
                    clear_cuda_cache_every is not None
                    and device.type == "cuda"
                    and chunk_counter % clear_cuda_cache_every == 0
                ):
                    torch.cuda.empty_cache()
                del batch_items, batch, outputs, logits, scores

            if (
                max_candidates_per_epoch is not None
                and total_scored >= max_candidates_per_epoch
            ):
                break

    if model_was_training:
        model.train()

    return {"candidates_scored": total_scored}


def prepare_scheduler_config(
    scheduler_cfg: Dict[str, Any], num_epochs: int, steps_per_epoch: int
) -> Dict[str, Any]:
    """
    Prepare scheduler config by converting epoch-based to step-based parameters.

    Args:
        scheduler_cfg: Raw scheduler config from YAML
        num_epochs: Total number of epochs
        steps_per_epoch: Number of steps (batches) per epoch

    Returns:
        Prepared scheduler config with step-based parameters
    """
    if not scheduler_cfg:
        return None

    cfg = scheduler_cfg.copy()
    scheduler_type = cfg.get("type", "").lower()
    total_steps = num_epochs * steps_per_epoch

    # OneCycleLR: requires total_steps
    if scheduler_type in ["onecycle", "onecyclelr"]:
        cfg.setdefault("total_steps", total_steps)

    # warmup_cosine: convert epochs to steps
    elif scheduler_type in ["warmup_cosine", "cosine_warmup"]:
        if "warmup_epochs" in cfg:
            cfg["num_warmup_steps"] = cfg.pop("warmup_epochs") * steps_per_epoch
        cfg.setdefault("num_training_steps", total_steps)

    return cfg


def run_finetune(
    cfg: Dict[str, Any],
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    finetune_run_id: str,
    log_dir: Path,
    checkpoint_dir: Path,
    load_checkpoint_path: Optional[str] = None,
) -> None:
    """
    Execute finetune stage: load checkpoint, apply strategy, train with staged unfreeze.

    Args:
        cfg: Full parsed config
        model: Initialized model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        device: Target device
        finetune_run_id: Run identifier
        log_dir: Directory for logs
        checkpoint_dir: Directory for checkpoints
        load_checkpoint_path: Path to pretrain checkpoint (required)
    """
    finetune_cfg = cfg["finetune_config"]
    strategy_cfg = finetune_cfg["strategy"]
    num_epochs = finetune_cfg["epochs"]
    data_cfg = cfg["data_config"]
    sampling_cfg = data_cfg.get("finetune", {}).get("sampling", {})
    sampling_strategy = sampling_cfg.get("strategy")
    online_mining = bool(
        sampling_cfg.get("online_mining", sampling_strategy == "staged_hard")
    )
    warmup_epochs = int(
        sampling_cfg.get("warmup_epochs", sampling_cfg.get("hard_start_epoch", 2))
    )
    hard_ratio = float(sampling_cfg.get("hard_ratio", 0.7))
    hard_score_sync = sampling_cfg.get("hard_score_sync")
    if not hard_score_sync:
        hard_score_sync = "file" if get_world_size() > 1 else "broadcast"
    hard_score_sync_timeout = int(sampling_cfg.get("hard_score_sync_timeout_sec", 7200))
    hard_score_quantile_low = float(sampling_cfg.get("hard_score_quantile_low", 0.9))
    hard_score_quantile_high = float(
        sampling_cfg.get("hard_score_quantile_high", 0.995)
    )

    # Use monitor_metric from finetune_config (not from DA config)
    monitor_metric = finetune_cfg.get("monitor_metric", "auprc")
    patience = finetune_cfg["early_stopping_patience"]
    save_best_only = cfg["run_config"]["save_best_only"]

    # Check if DA is configured (optional section in data_config.finetune)
    da_cfg = data_cfg.get("finetune", {}).get("distribution_alignment", None)
    use_da = da_cfg is not None

    if is_main_process():
        logging.info(f"Starting finetune for {num_epochs} epochs")
        logging.info(f"Strategy: {strategy_cfg['type']}")
        logging.info(f"Monitor metric: {monitor_metric}, Patience: {patience}")

    # Load checkpoint
    if load_checkpoint_path is None:
        raise ValueError("finetune requires load_checkpoint_path")

    # Load checkpoint (weights only, no optimizer state for fresh finetune)
    ckpt_metadata = load_checkpoint(
        model=model,
        ckpt_path=load_checkpoint_path,
        map_location=device,
        strict=True,
        load_optim=False,
    )
    if is_main_process():
        logging.info(
            f"Loaded checkpoint from: {load_checkpoint_path} "
            f"(pretrain epoch {ckpt_metadata.get('epoch', 'unknown')})"
        )

    # Instantiate strategy from config
    strategy = None
    if strategy_cfg["type"] == "staged_unfreeze":
        schedule = strategy_cfg.get("schedule", [])
        if schedule:
            strategy = StagedUnfreeze(schedule=schedule)
            if is_main_process():
                logging.info(
                    f"Created StagedUnfreeze strategy with {len(schedule)} schedule entries"
                )
        else:
            if is_main_process():
                logging.warning(
                    "staged_unfreeze strategy specified but no schedule provided"
                )

    # Prepare scheduler config
    scheduler_cfg = prepare_scheduler_config(
        finetune_cfg.get("scheduler"), num_epochs, len(train_loader)
    )

    # Instantiate Trainer with strategy
    trainer = Trainer(
        model=model,
        device=device,
        optimizer_cfg=finetune_cfg["optimizer"],
        scheduler_cfg=scheduler_cfg,
        amp_cfg={
            "enabled": finetune_cfg.get("use_mixed_precision", False),
            "dtype": cfg["data_config"].get("embedding_dtype", "fp16"),
        },
        strategy=strategy,
        max_norm=finetune_cfg.get("max_grad_norm"),
        loss_cfg=finetune_cfg.get("loss"),
    )

    # Call strategy on_train_begin hook
    if strategy:
        strategy.on_train_begin(trainer)
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if is_main_process():
            logging.info(f"Trainable parameters: {num_trainable:,}")

    # Optionally instantiate DistributionAligner or Evaluator based on config
    aligner = None
    evaluator = None
    if use_da:
        da_search_metric = da_cfg.get("threshold_search_metric", monitor_metric)
        aligner = DistributionAligner(
            target_prior=da_cfg.get("target_prior", 0.5),
            search_metric=da_search_metric,
            search_steps=da_cfg.get("search_steps", 100),
        )
        if is_main_process():
            logging.info(
                "Distribution Alignment enabled: "
                f"target_prior={da_cfg.get('target_prior', 0.5)}, "
                f"search_metric={da_search_metric}"
            )
    else:
        # No DA configured: use standard Evaluator (same as pretrain validation)
        # Include primary and secondary metrics to align CSV with pretrain
        primary_metric = finetune_cfg["logging_metrics"]["primary"]
        secondary_metric = finetune_cfg["logging_metrics"]["secondary"]
        eval_metrics = list(
            set([monitor_metric, primary_metric, secondary_metric, "loss"])
        )
        evaluator = Evaluator(
            metrics_list=eval_metrics,
            threshold=0.5,
        )
        if is_main_process():
            logging.info(
                f"No DA configured; using standard validation with monitor_metric={monitor_metric}"
            )

    primary = finetune_cfg["logging_metrics"]["primary"]
    secondary = finetune_cfg["logging_metrics"]["secondary"]

    # Early stopping: track metric history for check_early_stop
    monitor_mode = "max"
    metric_history = []

    # Setup CSV path and columns for logging (adapt based on DA mode)
    csv_path = log_dir / "training_step.csv"
    if use_da:
        columns = [
            "Epoch",
            "Epoch Time",
            "Train Loss",
            "Val Loss",
            f"Val {primary}",
            f"Val {secondary}",
            "DA Bias",
            "DA Threshold",
            "Predicted Prior",
            "Calibrated Prior",
            "Learning Rate",
        ]
    else:
        columns = [
            "Epoch",
            "Epoch Time",
            "Train Loss",
            "Val Loss",
            f"Val {primary}",
            f"Val {secondary}",
            "Learning Rate",
        ]

    # Training loop
    best_metric = float("-inf")

    dtype_str = str(data_cfg.get("embedding_dtype", "fp32")).lower()
    amp_dtype = None
    if finetune_cfg.get("use_mixed_precision", False) and device.type == "cuda":
        if dtype_str == "bf16":
            amp_dtype = torch.bfloat16
        elif dtype_str in {"fp16", "float16", "half"}:
            amp_dtype = torch.float16

    for epoch in range(num_epochs):
        if is_main_process():
            logging.info(f"\n{'=' * 60}")
            logging.info(f"Finetune Epoch {epoch}/{num_epochs - 1}")
            logging.info(f"{'=' * 60}")

        _set_sampler_epoch(train_loader, epoch)

        # Call strategy on_epoch_begin hook
        if strategy:
            strategy.on_epoch_begin(trainer, epoch)

        if (
            online_mining
            and sampling_strategy == "staged_hard"
            and epoch >= warmup_epochs
            and hard_ratio > 0.0
        ):
            base_sampler, _ = _get_base_batch_sampler(train_loader)
            if isinstance(base_sampler, StagedHardNegativeBatchSampler):
                mining_stats = {}
                if is_main_process():
                    mining_stats = _refresh_hard_scores_epoch(
                        model=model,
                        train_loader=train_loader,
                        sampler=base_sampler,
                        device=device,
                        epoch=epoch,
                        sampling_cfg=sampling_cfg,
                        amp_dtype=amp_dtype,
                    )

                if get_world_size() > 1:
                    score_len = len(base_sampler.get_hard_scores())
                    if is_main_process():
                        scores = base_sampler.get_hard_scores()
                    else:
                        scores = [math.nan for _ in range(score_len)]

                    try:
                        if hard_score_sync == "broadcast":
                            scores = _broadcast_hard_scores(scores, device)
                        elif hard_score_sync == "file":
                            scores = _sync_hard_scores_via_file(
                                scores=scores,
                                log_dir=log_dir,
                                epoch=epoch,
                                timeout_sec=hard_score_sync_timeout,
                            )
                        else:
                            raise ValueError(
                                f"Unknown hard_score_sync method: {hard_score_sync}"
                            )
                    except TimeoutError:
                        if is_main_process():
                            logging.warning(
                                "Hard-score sync timed out; using local scores for this epoch."
                            )
                    base_sampler.set_hard_scores(scores)

                pool_stats = base_sampler.refresh_hard_pool(
                    quantile_low=hard_score_quantile_low,
                    quantile_high=hard_score_quantile_high,
                )
                barrier()

                if is_main_process():
                    logging.info(
                        "Hard-score refresh: candidates_scored=%d, scored_negatives=%d, "
                        "hard_pool=%d, score_low=%s, score_high=%s",
                        mining_stats.get("candidates_scored", 0),
                        pool_stats.get("num_scored_negatives", 0),
                        pool_stats.get("hard_pool_size", 0),
                        pool_stats.get("score_low", "n/a"),
                        pool_stats.get("score_high", "n/a"),
                    )

        # Start epoch timer
        epoch_start_time = datetime.now()

        # Get batch logging configuration
        log_every_n_batches = finetune_cfg.get("log_every_n_batches", 50)
        batch_log_path = log_dir / "finetune_batches.log"

        # Train one epoch - consume generator for batch-level logging
        train_metrics = None
        for batch_metrics in trainer.train_one_epoch(train_loader):
            # Check if this is the final epoch summary
            if batch_metrics.get("_epoch_end", False):
                train_metrics = batch_metrics
                break

            # Log batch progress every N batches
            batch_idx = batch_metrics["batch_idx"]
            if is_main_process() and (batch_idx + 1) % log_every_n_batches == 0:
                total_batches = len(train_loader)
                log_msg = (
                    f"[FINETUNE] Epoch {epoch}/{num_epochs - 1} | "
                    f"Batch {batch_idx + 1}/{total_batches} | "
                    f"Loss: {batch_metrics['loss']:.6f} | "
                    f"LR: {batch_metrics['lr']:.2e}\n"
                )
                # Append to batch log file
                with open(batch_log_path, "a", encoding="utf-8") as f:
                    f.write(log_msg)

        # Ensure we got the epoch summary
        if train_metrics is None:
            raise RuntimeError("Trainer did not yield epoch summary")

        # Validation: branch based on DA mode
        model.eval()
        val_loss = 0.0
        current_metric = 0.0
        extra_payload = {}  # Checkpoint payload

        if use_da:
            # DA validation path
            with torch.no_grad():
                da_result = aligner.calibrate_and_search(
                    model, val_loader, device, amp_dtype=amp_dtype
                )

            da_metrics = da_result["metrics"]
            if monitor_metric not in da_metrics:
                raise KeyError(
                    f"Monitor metric '{monitor_metric}' not found in DA metrics: {da_metrics.keys()}"
                )
            current_metric = da_metrics[monitor_metric]
            val_loss = da_result["loss"]

            # Compute epoch time
            epoch_time = (datetime.now() - epoch_start_time).total_seconds()

            # Log DA validation results
            if is_main_process():
                logging.info(
                    f"Epoch {epoch}: avg_loss={train_metrics['loss']:.6f}, "
                    f"lr={train_metrics['lr']:.2e}"
                )
                logging.info(
                    "DA validation: loss=%.6f, bias=%.4f, threshold=%.4f, "
                    "pred_prior=%.4f, cal_prior=%.4f",
                    da_result["loss"],
                    da_result["bias"],
                    da_result["threshold"],
                    da_result["predicted_prior"],
                    da_result["calibrated_prior"],
                )
                logging.info(
                    f"DA metrics: {monitor_metric}={current_metric:.6f}, "
                    f"{primary}={da_metrics.get(primary, 0.0):.6f}, "
                    f"{secondary}={da_metrics.get(secondary, 0.0):.6f}"
                )

            # Append to CSV (DA columns)
            row = {
                "Epoch": epoch,
                "Epoch Time": epoch_time,
                "Train Loss": train_metrics["loss"],
                "Val Loss": da_result["loss"],
                f"Val {primary}": da_metrics.get(primary, 0.0),
                f"Val {secondary}": da_metrics.get(secondary, 0.0),
                "DA Bias": da_result["bias"],
                "DA Threshold": da_result["threshold"],
                "Predicted Prior": da_result["predicted_prior"],
                "Calibrated Prior": da_result["calibrated_prior"],
                "Learning Rate": train_metrics["lr"],
            }

            # DA checkpoint payload
            extra_payload = {
                "da_bias": da_result["bias"],
                "da_threshold": da_result["threshold"],
                "da_metrics": da_metrics,
                "predicted_prior": da_result["predicted_prior"],
                "calibrated_prior": da_result["calibrated_prior"],
            }

        else:
            # Raw validation path (no DA)
            val_metrics = None
            with torch.no_grad():
                if amp_dtype is not None:
                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                        for batch_metrics in evaluator.evaluate(
                            model, val_loader, device
                        ):
                            if batch_metrics.get("_evaluation_end", False):
                                val_metrics = batch_metrics
                                break
                else:
                    for batch_metrics in evaluator.evaluate(model, val_loader, device):
                        if batch_metrics.get("_evaluation_end", False):
                            val_metrics = batch_metrics
                            break

            if val_metrics is None:
                raise RuntimeError("Evaluator did not yield evaluation summary")

            val_metrics.pop("_evaluation_end", None)
            val_loss = val_metrics.get("loss", 0.0)
            current_metric = val_metrics.get(monitor_metric, 0.0)

            # Compute epoch time
            epoch_time = (datetime.now() - epoch_start_time).total_seconds()

            # Log raw validation results
            if is_main_process():
                logging.info(
                    f"Epoch {epoch}: avg_loss={train_metrics['loss']:.6f}, "
                    f"lr={train_metrics['lr']:.2e}"
                )
                logging.info(
                    f"Validation: loss={val_loss:.6f}, "
                    f"{primary}={val_metrics.get(primary, 0.0):.6f}, "
                    f"{secondary}={val_metrics.get(secondary, 0.0):.6f}"
                )

            # Append to CSV (standard columns, aligned with pretrain)
            row = {
                "Epoch": epoch,
                "Epoch Time": epoch_time,
                "Train Loss": train_metrics["loss"],
                "Val Loss": val_loss,
                f"Val {primary}": val_metrics.get(primary, 0.0),
                f"Val {secondary}": val_metrics.get(secondary, 0.0),
                "Learning Rate": train_metrics["lr"],
            }

            # No DA payload for raw validation
            extra_payload = {}

        model.train()

        if is_main_process():
            append_row(csv_path, row, columns)

        # Save best checkpoint
        improved, best_metric = maybe_save_best(
            model=model,
            epoch=epoch,
            current_metric=current_metric,
            best_so_far=best_metric,
            mode="max",
            best_path=str(checkpoint_dir / "best_model.pth"),
            include_optim=True,
            optimizer=trainer.optimizer,
            extra=extra_payload,
        )

        if improved and is_main_process():
            logging.info(
                f"Saved best finetune checkpoint: {checkpoint_dir / 'best_model.pth'} "
                f"({monitor_metric}={current_metric:.6f})"
            )

        # Save per-epoch checkpoint if not save_best_only
        if not save_best_only:
            saved_path = save_checkpoint(
                model=model,
                epoch=epoch,
                path=str(checkpoint_dir),
                include_optim=True,
                optimizer=trainer.optimizer,
                extra=extra_payload,
            )
            if saved_path and is_main_process():
                logging.info(f"Saved epoch checkpoint: {saved_path}")

        # Call strategy on_epoch_end hook (handles unfreezing/optimizer rebuild)
        if strategy:
            strategy.on_epoch_end(trainer, epoch)

        # Early stopping: append current metric and check
        metric_history.append(current_metric)
        if check_early_stop(
            metrics=metric_history,
            patience=patience,
            monitor=monitor_metric,
            mode=monitor_mode,
        ):
            if is_main_process():
                logging.info(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(no improvement in {monitor_metric} for {patience} epochs)"
                )
            break

    if is_main_process():
        logging.info(f"Finetune completed. Best {monitor_metric}: {best_metric:.6f}")
