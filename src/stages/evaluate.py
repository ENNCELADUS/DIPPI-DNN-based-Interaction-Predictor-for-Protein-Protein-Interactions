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
from src.utils.data.io import build_loader
from src.utils.data.sequence_io import build_sequence_loader
from src.utils.logging import append_row
from src.stages.stage_common import evaluate_to_summary, resolve_amp_dtype


def run_evaluation(
    cfg: Dict[str, Any],
    model: nn.Module,
    device: torch.device,
    log_dir: Path,
    load_checkpoint_path: Optional[str] = None,
) -> None:
    """
    Execute evaluation stage: load checkpoint and evaluate on test sets.

    Args:
        cfg: Full parsed config
        model: Initialized model
        device: Target device
        log_dir: Directory for logs
        load_checkpoint_path: Path to checkpoint (required)
    """
    eval_cfg = cfg["evaluate"]
    metrics_list = eval_cfg["metrics"]

    logging.info(f"Starting evaluation with metrics: {metrics_list}")

    curve_thresholds = eval_cfg.get("curve_thresholds")
    if curve_thresholds is None:
        curve_thresholds = cfg.get("data_config", {}).get("curve_thresholds")
    if curve_thresholds is None:
        curve_thresholds = Evaluator.DEFAULT_CURVE_THRESHOLDS
    else:
        curve_thresholds = int(curve_thresholds)

    logging.info(
        "Using %s thresholds for AUROC/AUPRC accumulation to bound host memory",
        curve_thresholds,
    )

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

    # Build test loaders
    data_cfg = cfg["data_config"]
    eval_data_cfg = data_cfg["evaluate"]
    dataloader_cfg = data_cfg.get("dataloader", {})

    test_loaders = []

    # Balanced split is expected for evaluation; realistic split is optional.
    split_paths = {
        "test_balanced": eval_data_cfg.get("test_balanced"),
        "test_realistic": eval_data_cfg.get("test_realistic"),
    }

    model_name = cfg["model_config"]["model"]

    for split_name, csv_path in split_paths.items():
        if not csv_path:
            if split_name == "test_realistic":
                logging.info("No test_realistic split configured; skipping")
                continue
            raise ValueError(f"Missing CSV path for required split: {split_name}")

        if model_name == "v6":
            loader = build_sequence_loader(
                csv_path=csv_path,
                sequence_path=data_cfg["sequence_path"],
                batch_size=32,
                max_len=data_cfg["max_sequence_length"],
                ddp=False,
                shuffle=False,
                dataloader_cfg=dataloader_cfg,
            )
        else:
            loader = build_loader(
                csv_path=csv_path,
                embeddings_path=data_cfg["embeddings_path"],
                batch_size=32,  # Fixed for eval
                max_len=data_cfg["max_sequence_length"],
                dtype=data_cfg["embedding_dtype"],
                ddp=False,  # No DDP for eval
                shuffle=False,
                dataloader_cfg=dataloader_cfg,
            )
        test_loaders.append((split_name, loader))

    if not test_loaders:
        raise ValueError("No evaluation splits configured in data_config.evaluate")

    # Instantiate Evaluator
    evaluator = Evaluator(
        metrics_list=metrics_list,
        threshold=eval_cfg.get("classification_threshold", 0.5),
        curve_thresholds=curve_thresholds,
    )

    # Evaluate on each test set
    csv_path = log_dir / "evaluate.csv"
    columns = ["split"] + metrics_list

    # Batch logging configuration
    log_every_n_batches = 50  # Match pretrain/finetune default
    batch_log_path = log_dir / "evaluate_batches.log"

    amp_dtype = None
    if model_name != "v6":
        amp_dtype = resolve_amp_dtype(
            use_mixed_precision=True,
            device=device,
            dtype_name=data_cfg.get("embedding_dtype", "fp32"),
        )

    for split_name, test_loader in test_loaders:
        logging.info(f"Evaluating on {split_name}...")
        model.eval()

        def _on_batch(batch_metrics: Dict[str, Any]) -> None:
            batch_idx = int(batch_metrics["batch_idx"])
            if (batch_idx + 1) % log_every_n_batches != 0:
                return
            total_batches = len(test_loader)
            log_msg = (
                f"[EVALUATE] Split: {split_name} | "
                f"Batch {batch_idx + 1}/{total_batches} | "
                f"Loss: {batch_metrics['loss']:.6f}\n"
            )
            with open(batch_log_path, "a", encoding="utf-8") as handle:
                handle.write(log_msg)

        metrics = evaluate_to_summary(
            evaluator=evaluator,
            model=model,
            loader=test_loader,
            device=device,
            amp_dtype=amp_dtype,
            on_batch=_on_batch,
        )

        logging.info(f"{split_name} results: {metrics}")

        row = {"split": split_name, **metrics}
        append_row(csv_path, row, columns)

    logging.info("Evaluation completed")
