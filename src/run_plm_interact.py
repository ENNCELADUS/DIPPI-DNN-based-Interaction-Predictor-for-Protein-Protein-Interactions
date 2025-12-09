"""
Standalone runner for PLM-interact style models (no changes to src/run.py).

Features:
- Finetune or eval-only on raw sequence CSVs (query, text, label)
- Uses HF ESM backbone loaded from local path
- Accepts PLM-interact checkpoint (.bin) to load weights
"""

import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer

# Ensure project root is on sys.path when running as a script (python src/run_plm_interact.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate.base import Evaluator
from src.model.plm_interact import PLMInteractModel
from src.utils.config import load_config
from src.utils.data_io import build_sequence_loader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logging(run_id: str, stage: str, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "log.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logging.info("=" * 80)
    logging.info(f"Starting {stage} stage, run_id={run_id}")
    logging.info("=" * 80)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }


def load_weights(model: nn.Module, ckpt_path: str) -> None:
    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state.get("state_dict") if isinstance(state, dict) else None
    if state_dict is None:
        state_dict = state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logging.warning(f"Missing keys when loading checkpoint: {missing}")
    if unexpected:
        logging.warning(f"Unexpected keys when loading checkpoint: {unexpected}")
    logging.info(f"Loaded checkpoint: {ckpt_path}")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    grad_accum_steps: int,
    max_grad_norm: float,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_steps = 0

    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(batch)
            loss = out["loss"] / grad_accum_steps

        scaler.scale(loss).backward()
        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * grad_accum_steps
        total_steps += 1

    return total_loss / max(total_steps, 1)


def evaluate_model(
    model: nn.Module,
    loaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    metrics: list[str],
    threshold: float,
    curve_thresholds: int | None,
    amp_dtype: torch.dtype | None = None,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    evaluator = Evaluator(
        metrics_list=metrics,
        threshold=threshold,
        curve_thresholds=curve_thresholds,
    )

    results: Dict[str, Dict[str, float]] = {}
    for name, loader in loaders.items():
        with torch.no_grad():
            if amp_dtype is not None and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    metrics_out = evaluator.evaluate(model, loader, device)
            else:
                metrics_out = evaluator.evaluate(model, loader, device)
        results[name] = metrics_out
        logging.info("%s: %s", name, metrics_out)
    return results


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    run_cfg = cfg["run_config"]
    data_cfg = cfg["data_config"]
    model_cfg = cfg["model_config"]
    train_cfg = cfg.get("train_config", {})
    eval_cfg = cfg["evaluate"]

    mode = run_cfg.get("mode", "eval_only")
    seed = run_cfg.get("seed", 42)
    device_str = run_cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    run_id = run_cfg.get("run_id") or generate_run_id()

    # Allow config to omit or set log_dir to null; fall back to default path in that case.
    log_dir = Path(run_cfg.get("log_dir") or f"logs/plm_interact/{mode}/{run_id}")
    setup_logging(run_id, mode, log_dir)
    set_seed(seed)
    logging.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model_path"])
    model = PLMInteractModel(
        base_model_path=model_cfg["base_model_path"],
        embedding_size=model_cfg.get("embedding_size", 1280),
        classifier_dropout=model_cfg.get("classifier_dropout", 0.1),
        pos_weight=model_cfg.get("pos_weight"),
    ).to(device)

    if model_cfg.get("checkpoint_path"):
        load_weights(model, model_cfg["checkpoint_path"])

    max_length = model_cfg.get("max_length", 1603)

    # Build loaders
    loaders: Dict[str, Any] = {}
    if mode != "eval_only":
        loaders["train"] = build_sequence_loader(
            csv_path=data_cfg["train_csv"],
            tokenizer=tokenizer,
            batch_size=train_cfg.get("batch_size_train", 2),
            max_length=max_length,
            shuffle=True,
            ddp=False,
            num_workers=train_cfg.get("num_workers", 0),
            pin_memory=True,
            require_labels=True,
        )
        loaders["val"] = build_sequence_loader(
            csv_path=data_cfg["val_csv"],
            tokenizer=tokenizer,
            batch_size=train_cfg.get("batch_size_eval", 8),
            max_length=max_length,
            shuffle=False,
            ddp=False,
            num_workers=train_cfg.get("num_workers", 0),
            pin_memory=True,
            require_labels=True,
        )

    eval_loaders = {
        "test_balanced": build_sequence_loader(
            csv_path=data_cfg["test_balanced_csv"],
            tokenizer=tokenizer,
            batch_size=train_cfg.get("batch_size_eval", 8),
            max_length=max_length,
            shuffle=False,
            ddp=False,
            num_workers=train_cfg.get("num_workers", 0),
            pin_memory=True,
            require_labels="label"
            in Path(data_cfg["test_balanced_csv"]).read_text(errors="ignore"),
        ),
        "test_realistic": build_sequence_loader(
            csv_path=data_cfg["test_realistic_csv"],
            tokenizer=tokenizer,
            batch_size=train_cfg.get("batch_size_eval", 8),
            max_length=max_length,
            shuffle=False,
            ddp=False,
            num_workers=train_cfg.get("num_workers", 0),
            pin_memory=True,
            require_labels="label"
            in Path(data_cfg["test_realistic_csv"]).read_text(errors="ignore"),
        ),
    }

    metrics = eval_cfg["metrics"]
    threshold = eval_cfg.get("classification_threshold", 0.5)
    curve_thresholds = eval_cfg.get(
        "curve_thresholds", Evaluator.DEFAULT_CURVE_THRESHOLDS
    )
    amp_dtype = (
        torch.float16
        if device.type == "cuda" and train_cfg.get("use_amp", True)
        else None
    )

    if mode == "eval_only":
        evaluate_model(
            model,
            eval_loaders,
            device,
            metrics,
            threshold,
            curve_thresholds,
            amp_dtype,
        )
        return

    # Finetune then eval
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 2e-5),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scaler = GradScaler(
        enabled=train_cfg.get("use_amp", True) and device.type == "cuda"
    )
    grad_accum = train_cfg.get("grad_accum_steps", 1)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    epochs = train_cfg.get("epochs", 1)

    best_metric = -1.0
    best_path = Path(run_cfg.get("checkpoint_dir") or f"models/plm_interact/{run_id}")
    best_path.mkdir(parents=True, exist_ok=True)
    best_file = best_path / "best_model.pth"

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            scaler,
            grad_accum,
            max_grad_norm,
            use_amp=train_cfg.get("use_amp", True) and device.type == "cuda",
        )
        logging.info(f"Epoch {epoch} train_loss={loss:.4f}")
        val_metrics = evaluate_model(
            model,
            {"val": loaders["val"]},
            device,
            metrics,
            threshold,
            curve_thresholds,
            amp_dtype,
        )["val"]
        val_score = val_metrics.get(eval_cfg.get("monitor_metric", "auroc"), 0.0)
        if val_score > best_metric:
            best_metric = val_score
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, best_file)
            logging.info(f"New best checkpoint at {best_file} ({best_metric:.4f})")

    # Load best and run final eval
    if best_file.exists():
        load_weights(model, str(best_file))
    evaluate_model(
        model,
        eval_loaders,
        device,
        metrics,
        threshold,
        curve_thresholds,
        amp_dtype,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_plm_interact.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
