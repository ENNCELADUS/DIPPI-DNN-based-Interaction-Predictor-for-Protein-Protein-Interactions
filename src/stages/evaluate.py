"""
Evaluation stage runner for DIPPI pipeline.

This module implements the evaluation stage orchestration:
- Loads trained checkpoint (from pretrain or finetune)
- Builds test dataloaders
- Instantiates Evaluator with requested metrics
- Runs evaluation on test sets (test_balanced, test_realistic)
- Logs results to CSV

Called by: run.py main orchestrator
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.evaluate.base import Evaluator
from src.utils.checkpoint import load_checkpoint
from src.utils.data_io import build_loader
from src.utils.logging import append_row


def run_evaluation(
    cfg: Dict[str, Any],
    model: nn.Module,
    device: torch.device,
    eval_run_id: str,
    log_dir: Path,
    load_checkpoint_path: Optional[str] = None,
) -> None:
    """
    Execute evaluation stage: load checkpoint and evaluate on test sets.

    Args:
        cfg: Full parsed config
        model: Initialized model
        device: Target device
        eval_run_id: Run identifier
        log_dir: Directory for logs
        load_checkpoint_path: Path to checkpoint (required)
    """
    eval_cfg = cfg["evaluate"]
    metrics_list = eval_cfg["metrics"]

    logging.info(f"Starting evaluation with metrics: {metrics_list}")

    # Load checkpoint
    if load_checkpoint_path is None:
        raise ValueError("eval_only mode requires load_checkpoint_path")

    # Load checkpoint (weights only, no optimizer state for eval)
    ckpt_metadata = load_checkpoint(
        model=model,
        ckpt_path=load_checkpoint_path,
        map_location=device,
        strict=True,
        load_optim=False,
    )
    logging.info(
        f"Loaded checkpoint from: {load_checkpoint_path} "
        f"(epoch {ckpt_metadata.get('epoch', 'unknown')})"
    )
    ckpt_extra = ckpt_metadata.get("extra", {}) or {}
    da_bias = ckpt_extra.get("da_bias")
    da_threshold = ckpt_extra.get("da_threshold")
    use_da = da_bias is not None and da_threshold is not None
    if use_da:
        logging.info(
            "Applying distribution alignment parameters from checkpoint: "
            f"bias={da_bias:.4f}, threshold={da_threshold:.4f}"
        )
    else:
        logging.info("No DA parameters found in checkpoint; using raw logits")

    # Build test loaders
    data_cfg = cfg["data_config"]
    eval_data_cfg = data_cfg["evaluate"]
    dataloader_cfg = data_cfg.get("dataloader", {})

    test_balanced_loader = build_loader(
        csv_path=eval_data_cfg["test_balanced"],
        embeddings_path=data_cfg["embeddings_path"],
        batch_size=32,  # Fixed for eval
        max_len=data_cfg["max_sequence_length"],
        dtype=data_cfg["embedding_dtype"],
        ddp=False,  # No DDP for eval
        shuffle=False,
        dataloader_cfg=dataloader_cfg,
    )

    test_realistic_loader = build_loader(
        csv_path=eval_data_cfg["test_realistic"],
        embeddings_path=data_cfg["embeddings_path"],
        batch_size=32,
        max_len=data_cfg["max_sequence_length"],
        dtype=data_cfg["embedding_dtype"],
        ddp=False,
        shuffle=False,
        dataloader_cfg=dataloader_cfg,
    )

    # Instantiate Evaluator
    evaluator = Evaluator(
        metrics_list=metrics_list,
        threshold=eval_cfg.get("classification_threshold", 0.5),
    )

    # Evaluate on each test set
    csv_path = log_dir / "evaluate.csv"
    columns = ["split"] + metrics_list

    # Batch logging configuration
    log_every_n_batches = 50  # Match pretrain/finetune default
    batch_log_path = log_dir / "evaluate_batches.log"

    amp_dtype = None
    if device.type == "cuda":
        dtype_str = str(data_cfg.get("embedding_dtype", "fp32")).lower()
        if dtype_str == "bf16":
            amp_dtype = torch.bfloat16
        elif dtype_str in {"fp16", "float16", "half"}:
            amp_dtype = torch.float16

    for split_name, test_loader in [
        ("test_balanced", test_balanced_loader),
        ("test_realistic", test_realistic_loader),
    ]:
        logging.info(f"Evaluating on {split_name}...")
        model.eval()

        metrics = None
        with torch.no_grad():
            if amp_dtype is not None:
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                    for batch_metrics in evaluator.evaluate(
                        model,
                        test_loader,
                        device,
                        logit_bias=da_bias if use_da else 0.0,
                        threshold_override=da_threshold if use_da else None,
                    ):
                        # Check if this is the final evaluation summary
                        if batch_metrics.get("_evaluation_end", False):
                            metrics = batch_metrics
                            break

                        # Log batch progress every N batches
                        batch_idx = batch_metrics["batch_idx"]
                        if (batch_idx + 1) % log_every_n_batches == 0:
                            total_batches = len(test_loader)
                            log_msg = (
                                f"[EVALUATE] Split: {split_name} | "
                                f"Batch {batch_idx + 1}/{total_batches} | "
                                f"Loss: {batch_metrics['loss']:.6f}\n"
                            )
                            # Append to batch log file
                            with open(batch_log_path, "a", encoding="utf-8") as f:
                                f.write(log_msg)
            else:
                for batch_metrics in evaluator.evaluate(
                    model,
                    test_loader,
                    device,
                    logit_bias=da_bias if use_da else 0.0,
                    threshold_override=da_threshold if use_da else None,
                ):
                    # Check if this is the final evaluation summary
                    if batch_metrics.get("_evaluation_end", False):
                        metrics = batch_metrics
                        break

                    # Log batch progress every N batches
                    batch_idx = batch_metrics["batch_idx"]
                    if (batch_idx + 1) % log_every_n_batches == 0:
                        total_batches = len(test_loader)
                        log_msg = (
                            f"[EVALUATE] Split: {split_name} | "
                            f"Batch {batch_idx + 1}/{total_batches} | "
                            f"Loss: {batch_metrics['loss']:.6f}\n"
                        )
                        # Append to batch log file
                        with open(batch_log_path, "a", encoding="utf-8") as f:
                            f.write(log_msg)

        # Ensure we got the evaluation summary
        if metrics is None:
            raise RuntimeError("Evaluator did not yield evaluation summary")

        # Remove sentinel key before logging
        metrics.pop("_evaluation_end", None)

        logging.info(f"{split_name} results: {metrics}")

        row = {"split": split_name, **metrics}
        append_row(csv_path, row, columns)

    logging.info("Evaluation completed")
