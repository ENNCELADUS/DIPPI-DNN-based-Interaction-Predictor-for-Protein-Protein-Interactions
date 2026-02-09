"""
Pretrain stage runner for DIPPI pipeline.

This module implements the pretrain stage orchestration:
- Builds Trainer and Evaluator from config
- Runs training loop with validation every epoch
- Handles checkpointing (best model + optional per-epoch)
- Manages early stopping based on monitored metric
- Logs metrics to CSV and log files

Called by: run.py main orchestrator
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.train.base import Trainer
from src.evaluate.base import Evaluator
from src.utils.distributed import is_main_process
from src.utils.checkpoint import save_checkpoint, maybe_save_best
from src.utils.early_stop import check_early_stop
from src.utils.logging import append_row
from src.stages.stage_common import (
    broadcast_early_stop_flag,
    evaluate_to_summary,
    log_epoch_header,
    prepare_scheduler_config,
    resolve_amp_dtype,
    train_one_epoch_with_batch_logging,
)


def run_pretrain(
    cfg: Dict[str, Any],
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    log_dir: Path,
    checkpoint_dir: Path,
) -> None:
    """
    Execute pretrain stage: train for N epochs with validation and checkpointing.

    Args:
        cfg: Full parsed config
        model: Initialized model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        device: Target device
        log_dir: Directory for logs
        checkpoint_dir: Directory for checkpoints
    """
    pretrain_cfg = cfg["pretrain_config"]
    num_epochs = pretrain_cfg["epochs"]
    monitor_metric = pretrain_cfg["monitor_metric"]
    patience = pretrain_cfg["early_stopping_patience"]
    save_best_only = cfg["run_config"]["save_best_only"]

    if is_main_process():
        logging.info(f"Starting pretrain for {num_epochs} epochs")
        logging.info(f"Monitor metric: {monitor_metric}, Patience: {patience}")

    # Prepare scheduler config
    scheduler_cfg = prepare_scheduler_config(
        pretrain_cfg.get("scheduler"), num_epochs, len(train_loader)
    )

    # Instantiate Trainer
    trainer = Trainer(
        model=model,
        device=device,
        optimizer_cfg=pretrain_cfg["optimizer"],
        scheduler_cfg=scheduler_cfg,
        amp_cfg={
            "enabled": pretrain_cfg.get("use_mixed_precision", False),
            "dtype": cfg["data_config"].get("embedding_dtype", "fp16"),
        },
        max_norm=pretrain_cfg.get("max_grad_norm"),
        loss_cfg=pretrain_cfg.get("loss"),
    )

    eval_cfg = cfg.get("evaluate", {}) or {}
    curve_thresholds = eval_cfg.get("curve_thresholds")
    if curve_thresholds is None:
        curve_thresholds = cfg.get("data_config", {}).get("curve_thresholds")
    if curve_thresholds is None:
        curve_thresholds = Evaluator.DEFAULT_CURVE_THRESHOLDS
    else:
        curve_thresholds = int(curve_thresholds)

    # Instantiate Evaluator
    primary = pretrain_cfg["logging_metrics"]["primary"]
    secondary = pretrain_cfg["logging_metrics"]["secondary"]
    evaluator = Evaluator(
        metrics_list=["loss", primary, secondary],
        threshold=pretrain_cfg.get("classification_threshold", 0.5),
        curve_thresholds=curve_thresholds,
    )

    # Early stopping: track metric history for check_early_stop
    monitor_mode = "min" if "loss" in monitor_metric else "max"
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
        "Learning Rate",
    ]

    # Training loop
    best_metric = float("inf") if "loss" in monitor_metric else float("-inf")

    for epoch in range(num_epochs):
        log_epoch_header("Pretrain", epoch, num_epochs)

        # Start epoch timer
        epoch_start_time = datetime.now()

        # Get batch logging configuration
        log_every_n_batches = pretrain_cfg.get("log_every_n_batches", 50)
        batch_log_path = log_dir / "pretrain_batches.log"

        train_metrics = train_one_epoch_with_batch_logging(
            trainer=trainer,
            train_loader=train_loader,
            epoch=epoch,
            num_epochs=num_epochs,
            stage_tag="PRETRAIN",
            log_every_n_batches=log_every_n_batches,
            batch_log_path=batch_log_path,
        )

        # Validate
        model.eval()
        amp_dtype = resolve_amp_dtype(
            use_mixed_precision=pretrain_cfg.get("use_mixed_precision", False),
            device=device,
            dtype_name=cfg["data_config"].get("embedding_dtype", "fp32"),
        )
        val_metrics = evaluate_to_summary(
            evaluator=evaluator,
            model=model,
            loader=val_loader,
            device=device,
            amp_dtype=amp_dtype,
        )
        model.train()  # Back to training mode

        # Compute epoch time
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()

        # Log to file
        if is_main_process():
            logging.info(
                f"Epoch {epoch}: avg_loss={train_metrics['loss']:.6f}, "
                f"lr={train_metrics['lr']:.2e}"
            )
            logging.info(
                f"Validation metrics: val_loss={val_metrics['loss']:.6f}, "
                f"val_{primary}={val_metrics.get(primary, 0.0):.6f}, "
                f"val_{secondary}={val_metrics.get(secondary, 0.0):.6f}"
            )

        # Append to CSV
        row = {
            "Epoch": epoch,
            "Epoch Time": epoch_time,
            "Train Loss": train_metrics["loss"],
            "Val Loss": val_metrics["loss"],
            f"Val {primary}": val_metrics.get(primary, 0.0),
            f"Val {secondary}": val_metrics.get(secondary, 0.0),
            "Learning Rate": train_metrics["lr"],
        }
        if is_main_process():
            append_row(csv_path, row, columns)

        # Checkpointing
        current_metric = val_metrics.get(monitor_metric, val_metrics.get("loss"))
        mode = "min" if "loss" in monitor_metric else "max"

        # Save best checkpoint
        improved, best_metric = maybe_save_best(
            model=model,
            epoch=epoch,
            current_metric=current_metric,
            best_so_far=best_metric,
            mode=mode,
            best_path=str(checkpoint_dir / "best_model.pth"),
            include_optim=True,
            optimizer=trainer.optimizer,
            extra={"monitor_metric": monitor_metric, "val_metrics": val_metrics},
        )

        if improved and is_main_process():
            logging.info(
                f"Saved best pretrain checkpoint: {checkpoint_dir / 'best_model.pth'} "
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
                extra={"val_metrics": val_metrics},
            )
            if saved_path and is_main_process():
                logging.info(f"Saved epoch checkpoint: {saved_path}")

        # Early stopping: append current metric and check
        metric_history.append(current_metric)
        should_stop = False
        if is_main_process():
            should_stop = check_early_stop(
                metrics=metric_history,
                patience=patience,
                monitor=monitor_metric,
                mode=monitor_mode,
            )

        should_stop = broadcast_early_stop_flag(should_stop, device)

        if should_stop:
            if is_main_process():
                logging.info(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(no improvement in {monitor_metric} for {patience} epochs)"
                )
            break

    if is_main_process():
        logging.info(f"Pretrain completed. Best {monitor_metric}: {best_metric:.6f}")
