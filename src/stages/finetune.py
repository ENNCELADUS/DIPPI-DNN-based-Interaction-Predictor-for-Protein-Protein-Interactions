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
from src.evaluate.base import Evaluator
from src.utils.checkpoint import load_checkpoint, save_checkpoint, maybe_save_best
from src.utils.early_stop import check_early_stop
from src.utils.logging import append_row
from src.utils.distributed import is_main_process


def _set_sampler_epoch(loader: DataLoader, epoch: int) -> None:
    if hasattr(loader, "set_epoch"):
        loader.set_epoch(epoch)
    batch_sampler = getattr(loader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)


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
    Execute finetune stage: optionally load checkpoint, apply strategy, train.

    Args:
        cfg: Full parsed config
        model: Initialized model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        device: Target device
        finetune_run_id: Run identifier
        log_dir: Directory for logs
        checkpoint_dir: Directory for checkpoints
        load_checkpoint_path: Optional checkpoint path to initialize weights.
    """
    finetune_cfg = cfg["finetune_config"]
    strategy_cfg = finetune_cfg["strategy"]
    num_epochs = finetune_cfg["epochs"]
    data_cfg = cfg["data_config"]

    # Use monitor_metric from finetune_config
    monitor_metric = finetune_cfg.get("monitor_metric", "auprc")
    patience = finetune_cfg["early_stopping_patience"]
    save_best_only = cfg["run_config"]["save_best_only"]

    if is_main_process():
        logging.info(f"Starting finetune for {num_epochs} epochs")
        logging.info(f"Strategy: {strategy_cfg['type']}")
        logging.info(f"Monitor metric: {monitor_metric}, Patience: {patience}")

    # Load checkpoint if provided
    if load_checkpoint_path:
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
    else:
        if is_main_process():
            logging.info("No checkpoint provided; finetuning from scratch")

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

    primary = finetune_cfg["logging_metrics"]["primary"]
    secondary = finetune_cfg["logging_metrics"]["secondary"]

    # Standard Evaluator (same as pretrain validation).
    # Use a stable order to keep DDP metric synchronization deterministic.
    eval_metrics: list[str] = []
    for metric_name in ["loss", monitor_metric, primary, secondary]:
        if metric_name not in eval_metrics:
            eval_metrics.append(metric_name)
    evaluator = Evaluator(
        metrics_list=eval_metrics,
        threshold=0.5,
    )
    if is_main_process():
        logging.info(f"Using standard validation with monitor_metric={monitor_metric}")

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

        # Start epoch timer
        epoch_start_time = datetime.now()

        # Get batch logging configuration
        log_every_n_batches = finetune_cfg.get("log_every_n_batches", 50)
        batch_log_path = log_dir / "finetune_batches.log"

        # Train one epoch - consume generator for batch-level logging
        train_metrics = None
        for batch_metrics in trainer.train_one_epoch_iter(train_loader):
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

        # Validation
        model.eval()
        val_metrics = None
        val_loss = 0.0
        current_metric = 0.0
        extra_payload = {}  # Checkpoint payload
        with torch.no_grad():
            if amp_dtype is not None:
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                    for batch_metrics in evaluator.evaluate(model, val_loader, device):
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

        # Log validation results
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
        should_stop = False
        if is_main_process():
            should_stop = check_early_stop(
                metrics=metric_history,
                patience=patience,
                monitor=monitor_metric,
                mode=monitor_mode,
            )

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            stop_tensor = torch.tensor(
                1 if should_stop else 0, device=device, dtype=torch.int
            )
            torch.distributed.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())

        if should_stop:
            if is_main_process():
                logging.info(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(no improvement in {monitor_metric} for {patience} epochs)"
                )
            break

    if is_main_process():
        logging.info(f"Finetune completed. Best {monitor_metric}: {best_metric:.6f}")
