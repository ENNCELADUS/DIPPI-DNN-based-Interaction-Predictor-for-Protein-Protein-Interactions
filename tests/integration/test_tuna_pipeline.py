"""
End-to-End Integration Test for TUnA Model Pipeline

Tests the complete pretrain→finetune→evaluate workflow for TUnA using minimal
test data and network architecture. Validates artifacts and logs.
"""

import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Dict

import pytest
import yaml

from src.utils.config import load_config, extract_keys


# Fixed run IDs for predictable artifact paths
PRETRAIN_RUN_ID = "test_tuna_pretrain"
FINETUNE_RUN_ID = "test_tuna_finetune"
EVAL_RUN_ID = "test_tuna_eval"

# Artifact paths
LOGS_DIR = Path("logs/tuna")
MODELS_DIR = Path("models/tuna")
CONFIG_PATH = Path("configs/test_tuna.yaml")


def get_artifact_paths(stage: str, run_id: str) -> Dict[str, Path]:
    """Get expected artifact paths for a stage."""
    log_dir = LOGS_DIR / stage / run_id
    checkpoint_dir = MODELS_DIR / stage / run_id

    paths = {
        "log_dir": log_dir,
        "checkpoint_dir": checkpoint_dir,
        "log_file": log_dir / "log.log",
    }

    # Training stages have training logs and checkpoints
    if stage in ["pretrain", "finetune"]:
        paths.update(
            {
                "training_step_csv": log_dir / "training_step.csv",
                "best_checkpoint": checkpoint_dir / "best_model.pth",
            }
        )

    # Evaluate stage has single CSV with all test splits
    if stage == "evaluate":
        paths.update(
            {
                "evaluate_csv": log_dir / "evaluate.csv",
            }
        )

    return paths


def verify_file_exists(
    filepath: Path, description: str, allow_empty: bool = False
) -> None:
    """Verify a file exists and optionally check if non-empty."""
    assert filepath.exists(), f"{description} not found: {filepath}"
    if not allow_empty:
        assert filepath.stat().st_size > 0, f"{description} is empty: {filepath}"
    print(f"✓ {description}: {filepath}")


def verify_csv_structure(
    csv_path: Path,
    expected_columns: List[str],
    min_rows: int = 1,
    description: str = "CSV",
) -> None:
    """Verify CSV has expected columns and minimum row count."""
    verify_file_exists(csv_path, description)

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        # Check columns
        actual_columns = reader.fieldnames
        assert actual_columns is not None, f"{description} has no header"

        missing = set(expected_columns) - set(actual_columns)
        assert not missing, f"{description} missing columns: {missing}"

        # Check row count
        assert len(rows) >= min_rows, (
            f"{description} has {len(rows)} rows, expected >= {min_rows}"
        )

        print(
            f"✓ {description} structure valid: {len(rows)} rows, {len(actual_columns)} columns"
        )


def verify_training_artifacts(stage: str, run_id: str, num_epochs: int) -> None:
    """Verify all training artifacts for pretrain or finetune stage."""
    print(f"\nVerifying {stage} artifacts...")
    paths = get_artifact_paths(stage, run_id)

    # 1. Log file (allow empty since logging gets reconfigured between stages)
    verify_file_exists(paths["log_file"], f"{stage} log", allow_empty=True)

    # 2. Training step CSV (epoch-level metrics)
    # Note: primary=auroc, secondary=recall from test config
    expected_step_cols = ["Epoch", "Train Loss", "Val Loss"]
    verify_csv_structure(
        paths["training_step_csv"],
        expected_step_cols,
        min_rows=num_epochs,  # One row per epoch
        description=f"{stage} training_step.csv",
    )

    # 4. Best model checkpoint
    verify_file_exists(paths["best_checkpoint"], f"{stage} best_model.pth")

    print(f"✓ All {stage} artifacts verified")


def verify_evaluation_artifacts(run_id: str, expected_metrics: List[str]) -> None:
    """Verify all evaluation artifacts."""
    print("\nVerifying evaluate artifacts...")
    paths = get_artifact_paths("evaluate", run_id)

    # 1. Log file (allow empty)
    verify_file_exists(paths["log_file"], "evaluate log", allow_empty=True)

    # 2. Evaluate CSV with both test splits
    expected_cols = ["split"] + expected_metrics
    verify_csv_structure(
        paths["evaluate_csv"],
        expected_cols,
        min_rows=2,  # Two test splits: test_balanced and test_realistic
        description="evaluate.csv",
    )

    # Verify both splits are present
    with open(paths["evaluate_csv"], "r") as f:
        reader = csv.DictReader(f)
        splits = {row["split"] for row in reader}

    expected_splits = {"test_balanced", "test_realistic"}
    missing_splits = expected_splits - splits
    assert not missing_splits, f"evaluate.csv missing splits: {missing_splits}"
    print(
        f"✓ evaluate.csv has both test splits with all {len(expected_metrics)} metrics"
    )

    print("✓ All evaluate artifacts verified")


def cleanup_test_artifacts() -> None:
    """Remove all test artifacts."""
    print("\nCleaning up test artifacts...")

    stages = ["pretrain", "finetune", "evaluate"]
    run_ids = [PRETRAIN_RUN_ID, FINETUNE_RUN_ID, EVAL_RUN_ID]

    for stage in stages:
        for run_id in run_ids:
            log_dir = LOGS_DIR / stage / run_id
            checkpoint_dir = MODELS_DIR / stage / run_id

            if log_dir.exists():
                shutil.rmtree(log_dir)
                print(f"✓ Removed {log_dir}")

            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
                print(f"✓ Removed {checkpoint_dir}")


@pytest.fixture(scope="module")
def cleanup_after_tests():
    """Cleanup fixture that runs after all tests in this module."""
    yield
    cleanup_test_artifacts()


def test_tuna_full_pipeline(cleanup_after_tests):
    """
    Test full_pipeline mode for TUnA: pretrain → finetune.

    Verifies:
    - Pipeline completes without errors
    - Pretrain artifacts (logs, CSVs, checkpoint)
    - Finetune artifacts (logs, CSVs, checkpoint)
    - Cross-epoch flow (2 epochs each)
    """
    print("\n" + "=" * 80)
    print("TEST: TUnA E2E Full Pipeline (pretrain → finetune)")
    print("=" * 80)

    # Ensure config is in full_pipeline mode
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    assert config["run_config"]["mode"] == "full_pipeline", (
        "Config must be in full_pipeline mode for this test"
    )

    # Run the pipeline
    print("\nRunning pipeline...")
    result = subprocess.run(
        [sys.executable, "-m", "src.run", str(CONFIG_PATH)],
        capture_output=True,
        text=True,
    )

    # Print output for debugging
    if result.stdout:
        print("\n--- STDOUT ---")
        print(result.stdout)
    if result.stderr:
        print("\n--- STDERR ---")
        print(result.stderr)

    # Check exit code
    assert result.returncode == 0, f"Pipeline failed with exit code {result.returncode}"
    print("✓ Pipeline completed successfully")

    # Verify pretrain artifacts
    verify_training_artifacts("pretrain", PRETRAIN_RUN_ID, num_epochs=2)

    # Verify finetune artifacts
    verify_training_artifacts("finetune", FINETUNE_RUN_ID, num_epochs=2)

    print("\n" + "=" * 80)
    print("✓ TEST PASSED: TUnA E2E Full Pipeline")
    print("=" * 80)


def test_tuna_eval_only(cleanup_after_tests):
    """
    Test eval_only mode for TUnA: evaluate using finetune checkpoint.

    Verifies:
    - Evaluation completes without errors
    - Evaluation artifacts (logs, metrics CSVs)
    - All metrics from config are computed

    Note: Depends on test_tuna_full_pipeline to have run first.
    """
    print("\n" + "=" * 80)
    print("TEST: TUnA E2E Eval Only")
    print("=" * 80)

    # Load config and get expected metrics
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    expected_metrics = config["evaluate"]["metrics"]
    print(f"Expected metrics: {expected_metrics}")

    # Get finetune checkpoint path
    finetune_checkpoint = MODELS_DIR / "finetune" / FINETUNE_RUN_ID / "best_model.pth"
    assert finetune_checkpoint.exists(), (
        f"Finetune checkpoint not found: {finetune_checkpoint}. "
        "Ensure test_tuna_full_pipeline ran first."
    )

    # Create temporary config for eval_only mode
    eval_config = config.copy()
    eval_config["run_config"]["mode"] = "eval_only"
    eval_config["run_config"]["load_checkpoint_path"] = str(finetune_checkpoint)

    eval_config_path = CONFIG_PATH.parent / "test_tuna_eval_temp.yaml"
    with open(eval_config_path, "w") as f:
        yaml.dump(eval_config, f)

    print(f"Created temporary eval config: {eval_config_path}")

    try:
        # Run evaluation
        print("\nRunning evaluation...")
        result = subprocess.run(
            [sys.executable, "-m", "src.run", str(eval_config_path)],
            capture_output=True,
            text=True,
        )

        # Print output for debugging
        if result.stdout:
            print("\n--- STDOUT ---")
            print(result.stdout)
        if result.stderr:
            print("\n--- STDERR ---")
            print(result.stderr)

        # Check exit code
        assert result.returncode == 0, (
            f"Evaluation failed with exit code {result.returncode}"
        )
        print("✓ Evaluation completed successfully")

        # Verify evaluation artifacts
        verify_evaluation_artifacts(EVAL_RUN_ID, expected_metrics)

        print("\n" + "=" * 80)
        print("✓ TEST PASSED: TUnA E2E Eval Only")
        print("=" * 80)

    finally:
        # Cleanup temporary config
        if eval_config_path.exists():
            eval_config_path.unlink()
            print(f"✓ Removed temporary config: {eval_config_path}")


def test_tuna_config_all_keys_used():
    """
    Test that all keys in test_tuna.yaml config are actually used.

    This catches typos and ensures no unnecessary config keys remain.
    """
    print("\n" + "=" * 80)
    print("TEST: TUnA Config Key Usage Validation")
    print("=" * 80)

    # Load config with tracking
    cfg = load_config(CONFIG_PATH)

    # Simulate what run.py does - extract all sections
    _ = extract_keys(cfg, "run_config")
    _ = extract_keys(cfg, "top_level_config")
    _ = extract_keys(cfg, "data_config")
    _ = extract_keys(cfg, "model_config")
    _ = extract_keys(cfg, "pretrain_config")
    _ = extract_keys(cfg, "finetune_config")
    _ = extract_keys(cfg, "evaluate")  # Extract evaluate section with nested keys

    # Get unused keys
    unused_keys = cfg.get_unused_keys()

    # Report results
    if unused_keys:
        print(f"\n⚠ Found {len(unused_keys)} unused config keys:")
        for key in unused_keys:
            print(f"  - {key}")
        print("\nThese keys may be typos or unnecessary parameters.")
        print("Consider removing them or verifying they should be used.")
    else:
        print("✓ All config keys are used - no unused keys found!")

    # Assert no unused keys (strict validation)
    assert not unused_keys, (
        f"Config validation failed: {len(unused_keys)} unused keys found. "
        f"Unused keys: {unused_keys}"
    )

    print("\n" + "=" * 80)
    print("✓ TEST PASSED: TUnA Config Key Usage Validation")
    print("=" * 80)


if __name__ == "__main__":
    # Allow running tests directly for debugging
    pytest.main([__file__, "-v", "-s"])
