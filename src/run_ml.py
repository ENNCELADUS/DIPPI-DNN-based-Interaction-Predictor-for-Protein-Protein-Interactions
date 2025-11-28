"""
DIPPI ML Pipeline Orchestrator (run_ml.py)

Minimal orchestrator for classical ML models (Random Forest, XGBoost).
Separate from run.py to keep DL and ML pipelines decoupled.

This module owns:
- Config parsing and validation
- Seed setup
- Run directory creation
- Data loading (mean-pooled features)
- Model training and evaluation
- Checkpointing (joblib) and logging

Invocation:
    python -m src.run_ml configs/ml.yaml
    python -m src.run_ml configs/ml.yaml --model xgboost
"""

import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.model.ml import build_ml_model
from src.utils.config import load_config, extract_keys
from src.utils.data_io import load_ml_features
from src.utils.logging import append_row


def setup_logging(run_id: str, model_name: str, log_dir: Path) -> None:
    """Configure Python's root logger for this run."""
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

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"Starting ML training for model '{model_name}' with run_id '{run_id}'")
    logger.info(f"Logs will be written to: {log_file}")
    logger.info("=" * 80)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    logging.info(f"Random seed set to {seed}")


def generate_run_id() -> str:
    """Generate timestamp-based run ID: YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_directories(model_name: str, run_id: str) -> Tuple[Path, Path]:
    """Create log and checkpoint directories for a run."""
    log_dir = Path(f"logs/ml/{model_name}/{run_id}")
    checkpoint_dir = Path(f"models/ml/{model_name}/{run_id}")

    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Created directories: {log_dir}, {checkpoint_dir}")
    return log_dir, checkpoint_dir


def balance_dataset(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    split_name: str = "dataset",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Undersample negatives to achieve 1:1 positive/negative ratio.

    Args:
        X: Feature matrix.
        y: Labels.
        seed: Random seed for reproducibility.
        split_name: Name of split for logging.

    Returns:
        Balanced (X, y) arrays.
    """
    pos_mask = y == 1
    neg_mask = y == 0

    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    logging.info(f"{split_name} before balancing: {n_pos} pos, {n_neg} neg")

    if n_neg <= n_pos:
        logging.info(f"{split_name} already balanced or more positives, skipping")
        return X, y

    # Get indices
    pos_indices = np.where(pos_mask)[0]
    neg_indices = np.where(neg_mask)[0]

    # Sample negatives to match positives
    rng = np.random.RandomState(seed)
    sampled_neg_indices = rng.choice(neg_indices, size=n_pos, replace=False)

    # Combine and shuffle
    balanced_indices = np.concatenate([pos_indices, sampled_neg_indices])
    rng.shuffle(balanced_indices)

    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]

    logging.info(
        f"{split_name} after balancing: {(y_balanced == 1).sum()} pos, "
        f"{(y_balanced == 0).sum()} neg"
    )

    return X_balanced, y_balanced


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    metrics_list: list,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities for positive class.
        metrics_list: List of metric names to compute.
        threshold: Decision threshold (not used if y_pred already binary).

    Returns:
        Dict mapping metric name to value.
    """
    results: Dict[str, float] = {}

    metric_funcs = {
        "auroc": lambda: roc_auc_score(y_true, y_proba),
        "auprc": lambda: average_precision_score(y_true, y_proba),
        "accuracy": lambda: accuracy_score(y_true, y_pred),
        "precision": lambda: precision_score(y_true, y_pred, zero_division=0),
        "recall": lambda: recall_score(y_true, y_pred, zero_division=0),
        "sensitivity": lambda: recall_score(y_true, y_pred, zero_division=0),
        "specificity": lambda: recall_score(
            1 - y_true, 1 - y_pred, zero_division=0
        ),  # TN / (TN + FP)
        "f1": lambda: f1_score(y_true, y_pred, zero_division=0),
        "mcc": lambda: matthews_corrcoef(y_true, y_pred),
    }

    for metric_name in metrics_list:
        if metric_name in metric_funcs:
            try:
                results[metric_name] = float(metric_funcs[metric_name]())
            except Exception as e:
                logging.warning(f"Failed to compute {metric_name}: {e}")
                results[metric_name] = 0.0

    return results


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    metrics_list: list,
    threshold: float,
    split_name: str,
) -> Dict[str, float]:
    """Evaluate model on a dataset split."""
    logging.info(f"Evaluating on {split_name}: {len(y)} samples")

    # Get predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]  # Probability of positive class

    # Compute metrics
    metrics = compute_metrics(y, y_pred, y_proba, metrics_list, threshold)

    # Log results
    logging.info(f"{split_name} results:")
    for name, value in metrics.items():
        logging.info(f"  {name}: {value:.4f}")

    return metrics


def main(config_path: str, model_override: str | None = None) -> None:
    """
    Main orchestrator: parse config, train model, evaluate.

    Args:
        config_path: Path to YAML config file.
        model_override: Optional model name to override config (e.g., "xgboost").
    """
    # ============================================================
    # 1. Load config
    # ============================================================
    cfg = load_config(config_path)
    print(f"Loaded config from: {config_path}")

    # ============================================================
    # 2. Run config setup
    # ============================================================
    run_cfg = extract_keys(cfg, "run_config")
    mode = run_cfg["mode"]
    seed = run_cfg["seed"]

    set_seed(seed)

    run_id = run_cfg.get("run_id") or generate_run_id()

    # Get model name (from override or config)
    model_name = model_override or cfg.get("model_config.model")
    logging.info(f"Model: {model_name}")

    # ============================================================
    # 3. Create directories and setup logging
    # ============================================================
    log_dir, checkpoint_dir = create_run_directories(model_name, run_id)
    setup_logging(run_id, model_name, log_dir)

    # ============================================================
    # 4. Load data
    # ============================================================
    data_cfg = extract_keys(cfg, "data_config")
    embeddings_path = data_cfg["embeddings_path"]
    pooling = data_cfg.get("pooling", "mean")

    logging.info("Loading training data...")
    X_train, y_train = load_ml_features(
        csv_path=data_cfg["train_csv"],
        embeddings_path=embeddings_path,
        pooling=pooling,
    )

    logging.info("Loading validation data...")
    X_val, y_val = load_ml_features(
        csv_path=data_cfg["val_csv"],
        embeddings_path=embeddings_path,
        pooling=pooling,
    )

    logging.info("Loading test data...")
    X_test, y_test = load_ml_features(
        csv_path=data_cfg["test_csv"],
        embeddings_path=embeddings_path,
        pooling=pooling,
    )

    logging.info(
        f"Data loaded: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}"
    )

    # ============================================================
    # 4.5 Balance train/val sets (undersample negatives to 1:1)
    # ============================================================
    balance = data_cfg.get("balance", True)
    if balance:
        X_train, y_train = balance_dataset(X_train, y_train, seed, "train")
        X_val, y_val = balance_dataset(X_val, y_val, seed, "val")

    # ============================================================
    # 5. Build model
    # ============================================================
    model_cfg = extract_keys(cfg, "model_config")

    # Get model-specific params
    if model_name in model_cfg:
        model_params = model_cfg[model_name]
    else:
        model_params = {}

    # Add seed to model params
    model_params["random_state"] = seed

    # Calculate scale_pos_weight for XGBoost if not set
    if model_name == "xgboost" and model_params.get("scale_pos_weight") is None:
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        if n_pos > 0:
            model_params["scale_pos_weight"] = float(n_neg / n_pos)
            logging.info(
                f"Auto-set scale_pos_weight: {model_params['scale_pos_weight']:.2f}"
            )

    logging.info(f"Building model: {model_name}")
    logging.info(f"Model params: {model_params}")

    model = build_ml_model(model_name, **model_params)

    # ============================================================
    # 6. Train model
    # ============================================================
    if mode == "train_eval":
        logging.info("Starting training...")

        if model_name == "xgboost":
            # XGBoost with early stopping on validation set
            # early_stopping_rounds is now passed via constructor
            early_stopping = model_params.get("early_stopping_rounds")
            if early_stopping:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            else:
                model.fit(X_train, y_train)
        else:
            # Random Forest
            model.fit(X_train, y_train)

        # Save model
        model_path = checkpoint_dir / f"{model_name}_model.joblib"
        joblib.dump(model, model_path)
        logging.info(f"Model saved to: {model_path}")

    elif mode == "eval_only":
        # Load existing model
        model_path = checkpoint_dir / f"{model_name}_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = joblib.load(model_path)
        logging.info(f"Loaded model from: {model_path}")

    # ============================================================
    # 7. Evaluate
    # ============================================================
    eval_cfg = cfg.get("evaluate", {})
    metrics_list = eval_cfg.get("metrics", ["auroc", "auprc", "accuracy", "f1"])
    threshold = eval_cfg.get("threshold", 0.5)

    # Evaluate on all splits
    val_metrics = evaluate_model(
        model, X_val, y_val, metrics_list, threshold, "validation"
    )
    test_metrics = evaluate_model(
        model, X_test, y_test, metrics_list, threshold, "test"
    )

    # Also evaluate on train for reference (overfitting check)
    train_metrics = evaluate_model(
        model, X_train, y_train, metrics_list, threshold, "train"
    )

    # ============================================================
    # 8. Save results
    # ============================================================
    # Save metrics to CSV
    results_csv = log_dir / "evaluation_results.csv"

    for split_name, metrics in [
        ("train", train_metrics),
        ("validation", val_metrics),
        ("test", test_metrics),
    ]:
        row = {"split": split_name, **metrics}
        append_row(results_csv, row)

    logging.info(f"Results saved to: {results_csv}")

    # Save full results as JSON
    results_json = log_dir / "results.json"
    full_results = {
        "run_id": run_id,
        "model": model_name,
        "seed": seed,
        "data": {
            "train_samples": len(y_train),
            "val_samples": len(y_val),
            "test_samples": len(y_test),
            "train_positive_ratio": float(y_train.mean()),
            "feature_dim": X_train.shape[1],
        },
        "model_params": model_params,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with open(results_json, "w") as f:
        json.dump(full_results, f, indent=2)

    logging.info(f"Full results saved to: {results_json}")
    logging.info("ML pipeline completed successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DIPPI ML Pipeline")
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default="configs/ml.yaml",
        help="Path to config file (default: configs/ml.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["random_forest", "xgboost"],
        default=None,
        help="Override model from config",
    )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    main(args.config, model_override=args.model)
