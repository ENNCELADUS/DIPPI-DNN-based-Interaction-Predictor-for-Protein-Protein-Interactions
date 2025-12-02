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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.train.base import Trainer
from src.train.strategies import StagedUnfreeze
from src.finetune.distribution_alignment import DistributionAligner
from src.utils.checkpoint import load_checkpoint, save_checkpoint, maybe_save_best
from src.utils.early_stop import check_early_stop
from src.utils.logging import append_row
from src.utils.distributed import is_main_process


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
    da_cfg = data_cfg.get("finetune", {}).get("distribution_alignment", {})
    da_metric_name = da_cfg.get("threshold_search_metric", "f1")
    monitor_metric = da_metric_name
    patience = finetune_cfg["early_stopping_patience"]
    save_best_only = cfg["run_config"]["save_best_only"]

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

    # Instantiate DistributionAligner for finetune validation
    aligner = DistributionAligner(
        target_prior=da_cfg.get("target_prior", 0.5),
        search_metric=da_metric_name,
        search_steps=da_cfg.get("search_steps", 100),
    )
    if is_main_process():
        logging.info(
            "Distribution Alignment enabled: "
            f"target_prior={da_cfg.get('target_prior', 0.5)}, "
            f"search_metric={da_metric_name}"
        )

    primary = finetune_cfg["logging_metrics"]["primary"]
    secondary = finetune_cfg["logging_metrics"]["secondary"]

    # Early stopping: track metric history for check_early_stop
    monitor_mode = "max"
    metric_history = []

    # Setup CSV path and columns for logging
    csv_path = log_dir / "training_step.csv"
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

        # Call strategy on_epoch_begin hook
        if strategy:
            strategy.on_epoch_begin(trainer, epoch)

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

        # Distribution alignment validation
        model.eval()
        with torch.no_grad():
            da_result = aligner.calibrate_and_search(
                model, val_loader, device, amp_dtype=amp_dtype
            )
        model.train()

        da_metrics = da_result["metrics"]
        if monitor_metric not in da_metrics:
            raise KeyError(
                f"Monitor metric '{monitor_metric}' not found in DA metrics: {da_metrics.keys()}"
            )
        current_metric = da_metrics[monitor_metric]

        # Compute epoch time
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()

        # Log to file (same pattern as pretrain)
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

        # Append to CSV
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
        if is_main_process():
            append_row(csv_path, row, columns)

        # Checkpointing (same pattern as pretrain)
        extra_payload = {
            "da_bias": da_result["bias"],
            "da_threshold": da_result["threshold"],
            "da_metrics": da_metrics,
            "predicted_prior": da_result["predicted_prior"],
            "calibrated_prior": da_result["calibrated_prior"],
        }

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
