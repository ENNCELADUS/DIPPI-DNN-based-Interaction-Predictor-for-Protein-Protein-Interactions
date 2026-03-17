"""Classical ML training and evaluation pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

from src.utils.config import (
    ConfigDict,
    as_bool,
    as_int,
    as_str,
    get_section,
    load_config,
)
from src.utils.data_io import ensure_embedding_cache, resolve_split_paths
from src.utils.logging import append_csv_row, generate_run_id

from src_ml.data import load_ml_features
from src_ml.model import build_ml_model


@dataclass(frozen=True)
class MLRuntimeInputs:
    """Resolved runtime inputs for the classical ML pipeline."""

    train_path: Path
    valid_path: Path
    test_path: Path
    embeddings_path: Path
    pooling: str
    balance: bool
    input_dim: int
    max_sequence_length: int


def setup_logging(run_id: str, model_name: str, log_dir: Path) -> None:
    """Configure the root logger for one ML run."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "log.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    logging.info("=" * 80)
    logging.info(
        "Starting ML training for model '%s' with run_id '%s'",
        model_name,
        run_id,
    )
    logging.info("Logs will be written to: %s", log_file)
    logging.info("=" * 80)


def set_seed(seed: int) -> None:
    """Set process-local RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    logging.info("Random seed set to %d", seed)


def create_run_directories(model_name: str, run_id: str) -> tuple[Path, Path]:
    """Create output directories for logs and model artifacts."""
    log_dir = Path("logs") / "ml" / model_name / run_id
    checkpoint_dir = Path("models") / "ml" / model_name / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return log_dir, checkpoint_dir


def _deduplicate_paths(paths: list[Path]) -> list[Path]:
    """Return paths in original order without duplicates."""
    deduplicated: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        normalized = str(path.resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        deduplicated.append(path)
    return deduplicated


def resolve_ml_runtime_inputs(config: ConfigDict) -> MLRuntimeInputs:
    """Resolve dataset paths and validate the shared embedding cache."""
    data_cfg = get_section(config, "data_config")
    embeddings_cfg = get_section(data_cfg, "embeddings")
    model_cfg = get_section(config, "model_config")

    split_paths = resolve_split_paths(config=config, train_stage="finetune")
    required_split_paths = _deduplicate_paths(
        [split_paths["train"], split_paths["valid"], split_paths["test"]]
    )

    input_dim = as_int(model_cfg.get("input_dim", 0), "model_config.input_dim")
    if input_dim <= 0:
        raise ValueError("model_config.input_dim must be positive")
    max_sequence_length = as_int(
        data_cfg.get("max_sequence_length", 0),
        "data_config.max_sequence_length",
    )
    if max_sequence_length <= 0:
        raise ValueError("data_config.max_sequence_length must be positive")

    pooling = as_str(data_cfg.get("pooling", "mean"), "data_config.pooling")
    balance = as_bool(data_cfg.get("balance", False), "data_config.balance")
    configured_cache_dir = Path(
        as_str(embeddings_cfg.get("cache_dir", ""), "data_config.embeddings.cache_dir")
    )

    embedding_cache = ensure_embedding_cache(
        config=config,
        split_paths=required_split_paths,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
    )
    cache_dir = embedding_cache.cache_dir
    if cache_dir.resolve() != configured_cache_dir.resolve():
        logging.warning(
            "Embedding cache helper resolved %s, config requested %s",
            cache_dir,
            configured_cache_dir,
        )

    return MLRuntimeInputs(
        train_path=split_paths["train"],
        valid_path=split_paths["valid"],
        test_path=split_paths["test"],
        embeddings_path=cache_dir,
        pooling=pooling,
        balance=balance,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
    )


def balance_dataset(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    split_name: str = "dataset",
) -> tuple[np.ndarray, np.ndarray]:
    """Undersample negatives to a 1:1 positive/negative ratio."""
    pos_mask = y == 1
    neg_mask = y == 0

    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    logging.info("%s before balancing: %d pos, %d neg", split_name, n_pos, n_neg)

    if n_neg <= n_pos:
        logging.info("%s already balanced or more positives, skipping", split_name)
        return X, y

    pos_indices = np.where(pos_mask)[0]
    neg_indices = np.where(neg_mask)[0]

    rng = np.random.RandomState(seed)
    sampled_neg_indices = rng.choice(neg_indices, size=n_pos, replace=False)

    balanced_indices = np.concatenate([pos_indices, sampled_neg_indices])
    rng.shuffle(balanced_indices)

    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]

    logging.info(
        "%s after balancing: %d pos, %d neg",
        split_name,
        int((y_balanced == 1).sum()),
        int((y_balanced == 0).sum()),
    )
    return X_balanced, y_balanced


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    metrics_list: list[str],
) -> dict[str, float]:
    """Compute configured evaluation metrics for one split."""
    results: dict[str, float] = {}
    metric_funcs = {
        "auroc": lambda: roc_auc_score(y_true, y_proba),
        "auprc": lambda: average_precision_score(y_true, y_proba),
        "accuracy": lambda: accuracy_score(y_true, y_pred),
        "precision": lambda: precision_score(y_true, y_pred, zero_division=0),
        "recall": lambda: recall_score(y_true, y_pred, zero_division=0),
        "sensitivity": lambda: recall_score(y_true, y_pred, zero_division=0),
        "specificity": lambda: recall_score(1 - y_true, 1 - y_pred, zero_division=0),
        "f1": lambda: f1_score(y_true, y_pred, zero_division=0),
        "mcc": lambda: matthews_corrcoef(y_true, y_pred),
    }

    for metric_name in metrics_list:
        if metric_name not in metric_funcs:
            continue
        try:
            results[metric_name] = float(metric_funcs[metric_name]())
        except Exception as error:  # pragma: no cover - defensive logging
            logging.warning("Failed to compute %s: %s", metric_name, error)
            results[metric_name] = 0.0

    return results


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    metrics_list: list[str],
    split_name: str,
) -> dict[str, float]:
    """Evaluate one fitted model on a dataset split."""
    logging.info("Evaluating on %s: %d samples", split_name, len(y))
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    metrics = compute_metrics(y, y_pred, y_proba, metrics_list)

    logging.info("%s results:", split_name)
    for name, value in metrics.items():
        logging.info("  %s: %.4f", name, value)
    return metrics


def _model_name_from_config(
    model_cfg: ConfigDict,
    model_override: str | None,
) -> str:
    """Resolve the selected classical model name."""
    if model_override is not None:
        return model_override.lower()
    return as_str(model_cfg.get("model", ""), "model_config.model").lower()


def _model_params_from_config(model_cfg: ConfigDict, model_name: str) -> dict[str, Any]:
    """Resolve model-specific hyperparameters for the selected model."""
    model_params_raw = model_cfg.get(model_name, {})
    if not isinstance(model_params_raw, dict):
        raise ValueError(f"model_config.{model_name} must be a mapping")
    return dict(model_params_raw)


def main(config_path: str, model_override: str | None = None) -> None:
    """Run the classical ML training or evaluation pipeline."""
    config = load_config(config_path)
    run_cfg = get_section(config, "run_config")
    model_cfg = get_section(config, "model_config")

    mode = as_str(run_cfg.get("mode", "train_eval"), "run_config.mode").lower()
    if mode not in {"train_eval", "eval_only"}:
        raise ValueError("run_config.mode must be 'train_eval' or 'eval_only'")

    seed = as_int(run_cfg.get("seed", 0), "run_config.seed")
    set_seed(seed)

    run_id = generate_run_id(run_cfg.get("run_id"))
    model_name = _model_name_from_config(model_cfg=model_cfg, model_override=model_override)

    log_dir, checkpoint_dir = create_run_directories(model_name=model_name, run_id=run_id)
    setup_logging(run_id=run_id, model_name=model_name, log_dir=log_dir)
    logging.info("Loaded config from: %s", Path(config_path).resolve())

    runtime_inputs = resolve_ml_runtime_inputs(config)

    logging.info("Loading training data...")
    X_train, y_train = load_ml_features(
        csv_path=str(runtime_inputs.train_path),
        embeddings_path=str(runtime_inputs.embeddings_path),
        pooling=runtime_inputs.pooling,
    )
    logging.info("Loading validation data...")
    X_val, y_val = load_ml_features(
        csv_path=str(runtime_inputs.valid_path),
        embeddings_path=str(runtime_inputs.embeddings_path),
        pooling=runtime_inputs.pooling,
    )
    logging.info("Loading test data...")
    X_test, y_test = load_ml_features(
        csv_path=str(runtime_inputs.test_path),
        embeddings_path=str(runtime_inputs.embeddings_path),
        pooling=runtime_inputs.pooling,
    )
    logging.info(
        "Data loaded: train=%d, val=%d, test=%d",
        len(y_train),
        len(y_val),
        len(y_test),
    )

    if runtime_inputs.balance:
        X_train, y_train = balance_dataset(X_train, y_train, seed, "train")
        X_val, y_val = balance_dataset(X_val, y_val, seed, "val")

    model_params = _model_params_from_config(model_cfg=model_cfg, model_name=model_name)
    model_params["random_state"] = seed
    if model_name == "xgboost" and model_params.get("scale_pos_weight") is None:
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        if n_pos > 0:
            model_params["scale_pos_weight"] = float(n_neg / n_pos)
            logging.info(
                "Auto-set scale_pos_weight: %.2f",
                model_params["scale_pos_weight"],
            )

    logging.info("Building model: %s", model_name)
    logging.info("Model params: %s", model_params)
    model = build_ml_model(model_name, **model_params)

    model_path = checkpoint_dir / f"{model_name}_model.joblib"
    if mode == "train_eval":
        logging.info("Starting training...")
        if model_name == "xgboost" and model_params.get("early_stopping_rounds"):
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        else:
            model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        logging.info("Model saved to: %s", model_path)
    else:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = joblib.load(model_path)
        logging.info("Loaded model from: %s", model_path)

    eval_cfg_raw = config.get("evaluate", {})
    if eval_cfg_raw is None:
        eval_cfg_raw = {}
    if not isinstance(eval_cfg_raw, dict):
        raise ValueError("evaluate must be a mapping")
    eval_cfg = eval_cfg_raw
    metrics_list = [
        as_str(metric, "evaluate.metrics[]")
        for metric in eval_cfg.get("metrics", ["auroc", "auprc", "accuracy", "f1"])
    ]

    train_metrics = evaluate_model(model, X_train, y_train, metrics_list, "train")
    val_metrics = evaluate_model(model, X_val, y_val, metrics_list, "validation")
    test_metrics = evaluate_model(model, X_test, y_test, metrics_list, "test")

    results_csv = log_dir / "evaluation_results.csv"
    for split_name, metrics in (
        ("train", train_metrics),
        ("validation", val_metrics),
        ("test", test_metrics),
    ):
        append_csv_row(results_csv, {"split": split_name, **metrics})
    logging.info("Results saved to: %s", results_csv)

    results_json = log_dir / "results.json"
    full_results = {
        "run_id": run_id,
        "model": model_name,
        "seed": seed,
        "data": {
            "train_samples": len(y_train),
            "val_samples": len(y_val),
            "test_samples": len(y_test),
            "train_positive_ratio": float(y_train.mean()) if len(y_train) > 0 else 0.0,
            "feature_dim": int(X_train.shape[1]) if X_train.ndim == 2 else 0,
        },
        "model_params": model_params,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with results_json.open("w", encoding="utf-8") as handle:
        json.dump(full_results, handle, indent=2)

    logging.info("Full results saved to: %s", results_json)
    logging.info("ML pipeline completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIPPI classical ML pipeline")
    parser.add_argument(
        "config_path",
        nargs="?",
        default=None,
        help="Optional config path. Defaults to src_ml/ml.yaml.",
    )
    parser.add_argument(
        "--config",
        dest="config_flag",
        default=None,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["random_forest", "xgboost"],
        default=None,
        help="Override model from config",
    )

    args = parser.parse_args()
    resolved_config_path = args.config_flag or args.config_path or "src_ml/ml.yaml"
    if not Path(resolved_config_path).exists():
        raise FileNotFoundError(f"Config file not found: {resolved_config_path}")
    main(resolved_config_path, model_override=args.model)
