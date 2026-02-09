"""Shared helpers for pretrain/finetune/evaluate stage runners."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import torch
import torch.nn as nn

from src.evaluate.base import Evaluator
from src.train.base import Trainer
from src.utils.distributed import is_main_process


def prepare_scheduler_config(
    scheduler_cfg: Optional[Dict[str, Any]],
    num_epochs: int,
    steps_per_epoch: int,
) -> Optional[Dict[str, Any]]:
    """Convert scheduler config epoch fields to step-based values."""
    if not scheduler_cfg:
        return None

    cfg = scheduler_cfg.copy()
    scheduler_type = str(cfg.get("type", "")).lower()
    total_steps = num_epochs * steps_per_epoch

    if scheduler_type in {"onecycle", "onecyclelr"}:
        cfg.setdefault("total_steps", total_steps)
    elif scheduler_type in {"warmup_cosine", "cosine_warmup"}:
        if "warmup_epochs" in cfg:
            cfg["num_warmup_steps"] = int(cfg.pop("warmup_epochs")) * steps_per_epoch
        cfg.setdefault("num_training_steps", total_steps)

    return cfg


def resolve_amp_dtype(
    *,
    use_mixed_precision: bool,
    device: torch.device,
    dtype_name: str,
) -> Optional[torch.dtype]:
    """Resolve AMP dtype from config; return None when AMP should be disabled."""
    if not use_mixed_precision or device.type != "cuda":
        return None

    normalized = str(dtype_name).lower()
    if normalized == "bf16":
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    return None


def log_epoch_header(stage_name: str, epoch: int, num_epochs: int) -> None:
    """Emit a consistent epoch header on main process."""
    if not is_main_process():
        return
    logging.info("\n%s", "=" * 60)
    logging.info("%s Epoch %d/%d", stage_name, epoch, num_epochs - 1)
    logging.info("%s", "=" * 60)


def train_one_epoch_with_batch_logging(
    *,
    trainer: Trainer,
    train_loader: Iterable[Dict[str, Any]],
    epoch: int,
    num_epochs: int,
    stage_tag: str,
    log_every_n_batches: int,
    batch_log_path: Path,
) -> Dict[str, Any]:
    """Run one trainer epoch and append periodic batch metrics to a log file."""
    train_metrics: Optional[Dict[str, Any]] = None

    for batch_metrics in trainer.train_one_epoch_iter(train_loader):
        if batch_metrics.get("_epoch_end", False):
            train_metrics = batch_metrics
            break

        batch_idx = int(batch_metrics["batch_idx"])
        if is_main_process() and (batch_idx + 1) % log_every_n_batches == 0:
            total_batches = len(train_loader)  # type: ignore[arg-type]
            log_msg = (
                f"[{stage_tag}] Epoch {epoch}/{num_epochs - 1} | "
                f"Batch {batch_idx + 1}/{total_batches} | "
                f"Loss: {batch_metrics['loss']:.6f} | "
                f"LR: {batch_metrics['lr']:.2e}\n"
            )
            with open(batch_log_path, "a", encoding="utf-8") as handle:
                handle.write(log_msg)

    if train_metrics is None:
        raise RuntimeError("Trainer did not yield epoch summary")

    return dict(train_metrics)


def evaluate_to_summary(
    *,
    evaluator: Evaluator,
    model: nn.Module,
    loader: Iterable[Dict[str, Any]],
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    on_batch: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run evaluator loop and return final summary metrics."""
    summary: Optional[Dict[str, Any]] = None

    autocast_ctx: contextlib.AbstractContextManager[Any]
    if amp_dtype is None:
        autocast_ctx = contextlib.nullcontext()
    else:
        autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype)

    with torch.no_grad():
        with autocast_ctx:
            for batch_metrics in evaluator.evaluate(model, loader, device):
                if batch_metrics.get("_evaluation_end", False):
                    summary = dict(batch_metrics)
                    break
                if on_batch is not None:
                    on_batch(batch_metrics)

    if summary is None:
        raise RuntimeError("Evaluator did not yield evaluation summary")

    summary.pop("_evaluation_end", None)
    return summary


def broadcast_early_stop_flag(should_stop: bool, device: torch.device) -> bool:
    """Broadcast main-process early-stop decision across DDP ranks."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        stop_tensor = torch.tensor(
            1 if should_stop else 0, device=device, dtype=torch.int
        )
        torch.distributed.broadcast(stop_tensor, src=0)
        return bool(stop_tensor.item())
    return should_stop
