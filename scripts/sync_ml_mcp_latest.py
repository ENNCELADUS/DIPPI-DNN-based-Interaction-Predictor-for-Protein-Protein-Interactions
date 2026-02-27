#!/usr/bin/env python3
"""Sync latest XPI model artifacts into ml-lab experiment storage.

This script reads:
1) architecture/hyperparameters from configs/<model>.yaml
2) latest stage results from logs/<model>/{pretrain,finetune,evaluate}

Then it upserts records into ~/.cache/ml-lab/experiments.db so the data is
available for ml-mcp-backed workflows.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import yaml

RUN_ID_PATTERN = re.compile(r"^\d{8}_\d{6}$")
TRAIN_STAGES: tuple[str, ...] = ("pretrain", "finetune")
ALL_STAGES: tuple[str, ...] = ("pretrain", "finetune", "evaluate")


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Sync latest model snapshots to ml-lab."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["v3", "v4", "v5"],
        help="Model names to sync (default: v3 v4 v5).",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("configs"),
        help="Directory containing model yaml configs.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing per-model log artifacts.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path.home() / ".cache" / "ml-lab" / "experiments.db",
        help="Path to ml-lab SQLite database.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs") / "ml-mcp-sync",
        help="Directory to write sync snapshots.",
    )
    return parser.parse_args()


def latest_run_id(stage_dir: Path) -> str | None:
    """Return latest timestamp-like run_id in one stage directory."""
    if not stage_dir.exists():
        return None
    run_ids = [
        child.name
        for child in stage_dir.iterdir()
        if child.is_dir() and RUN_ID_PATTERN.match(child.name)
    ]
    if not run_ids:
        return None
    return sorted(run_ids)[-1]


def read_yaml_dict(path: Path) -> dict[str, object]:
    """Load yaml and guarantee mapping payload."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV into row dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def parse_float(value: str | None) -> float | None:
    """Best-effort float parser."""
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def parse_int(value: str | None) -> int | None:
    """Best-effort integer parser."""
    numeric = parse_float(value)
    if numeric is None:
        return None
    return int(numeric)


def summarize_training_rows(rows: Sequence[dict[str, str]]) -> dict[str, object]:
    """Summarize a training_step.csv payload."""
    if not rows:
        return {}

    last = rows[-1]
    summary: dict[str, object] = {
        "num_epochs_logged": len(rows),
        "last_epoch": parse_int(last.get("Epoch")),
        "last_train_loss": parse_float(last.get("Train Loss")),
        "last_val_loss": parse_float(last.get("Val Loss")),
        "last_learning_rate": parse_float(last.get("Learning Rate")),
    }

    val_auprc_values: list[tuple[float, dict[str, str]]] = []
    for row in rows:
        val_auprc = parse_float(row.get("Val auprc"))
        if val_auprc is not None:
            val_auprc_values.append((val_auprc, row))
    if val_auprc_values:
        best_auprc, best_row = max(val_auprc_values, key=lambda item: item[0])
        summary["best_val_auprc"] = best_auprc
        summary["best_epoch_by_auprc"] = parse_int(best_row.get("Epoch"))

    val_auroc = parse_float(last.get("Val auroc"))
    if val_auroc is not None:
        summary["last_val_auroc"] = val_auroc

    return summary


def summarize_evaluate_rows(rows: Sequence[dict[str, str]]) -> dict[str, object]:
    """Summarize evaluate.csv payload."""
    if not rows:
        return {}

    unique_rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        normalized = json.dumps(row, sort_keys=True)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_rows.append(row)

    preferred = unique_rows[0]
    for row in unique_rows:
        split = row.get("split", "").strip().lower()
        if split in {"test", "test_balanced"}:
            preferred = row
            break

    metrics: dict[str, object] = {
        "split": preferred.get("split", ""),
        "num_unique_rows": len(unique_rows),
    }
    for key, value in preferred.items():
        if key == "split":
            continue
        numeric = parse_float(value)
        metrics[key] = numeric if numeric is not None else value
    return metrics


def build_model_snapshot(
    model_name: str,
    configs_dir: Path,
    logs_dir: Path,
) -> dict[str, object]:
    """Build one model snapshot from config + latest stage artifacts."""
    config_path = configs_dir / f"{model_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    config_payload = read_yaml_dict(config_path)
    model_logs_root = logs_dir / model_name

    latest_runs: dict[str, str | None] = {}
    results: dict[str, object] = {}

    for stage in ALL_STAGES:
        stage_dir = model_logs_root / stage
        run_id = latest_run_id(stage_dir)
        latest_runs[stage] = run_id
        if run_id is None:
            results[stage] = {"missing": True}
            continue

        if stage in TRAIN_STAGES:
            csv_path = stage_dir / run_id / "training_step.csv"
            if not csv_path.exists():
                results[stage] = {"missing": True, "run_id": run_id}
                continue
            rows = read_csv_rows(csv_path)
            stage_summary = summarize_training_rows(rows)
            stage_summary["run_id"] = run_id
            stage_summary["csv_path"] = str(csv_path)
            results[stage] = stage_summary
            continue

        eval_csv = stage_dir / run_id / "evaluate.csv"
        if not eval_csv.exists():
            results[stage] = {"missing": True, "run_id": run_id}
            continue
        eval_rows = read_csv_rows(eval_csv)
        eval_summary = summarize_evaluate_rows(eval_rows)
        eval_summary["run_id"] = run_id
        eval_summary["csv_path"] = str(eval_csv)
        results[stage] = eval_summary

    return {
        "model_name": model_name,
        "synced_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "config_path": str(config_path),
            "logs_root": str(model_logs_root),
        },
        "latest_runs": latest_runs,
        "architecture": config_payload.get("model_config", {}),
        "hyperparameters": {
            "run_config": config_payload.get("run_config", {}),
            "training_config": config_payload.get("training_config", {}),
            "device_config": config_payload.get("device_config", {}),
            "data_config": config_payload.get("data_config", {}),
        },
        "results": results,
    }


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create ml-lab schema if absent."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            base_model TEXT NOT NULL,
            method TEXT NOT NULL,
            status TEXT NOT NULL,
            description TEXT,
            tags TEXT,
            config TEXT,
            metrics TEXT,
            best_checkpoint TEXT,
            parent_experiment_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            experiment_id TEXT NOT NULL,
            status TEXT NOT NULL,
            config TEXT,
            metrics TEXT,
            artifacts TEXT,
            error_message TEXT,
            started_at TEXT NOT NULL,
            ended_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            metrics TEXT NOT NULL
        )
        """
    )


def upsert_snapshot(conn: sqlite3.Connection, snapshot: dict[str, object]) -> None:
    """Upsert one snapshot into experiments/runs/metrics_log."""
    model_name = str(snapshot["model_name"])
    experiment_name = f"xpi_{model_name}_latest"
    now_iso = datetime.now(timezone.utc).isoformat()

    row = conn.execute(
        "SELECT id, created_at FROM experiments WHERE name = ? ORDER BY created_at DESC LIMIT 1",
        (experiment_name,),
    ).fetchone()

    config_json = json.dumps(
        {
            "sources": snapshot["sources"],
            "latest_runs": snapshot["latest_runs"],
            "architecture": snapshot["architecture"],
            "hyperparameters": snapshot["hyperparameters"],
        },
        sort_keys=True,
    )
    metrics_json = json.dumps(snapshot["results"], sort_keys=True)
    tags_json = json.dumps(["xpi", "auto-sync", model_name])

    if row is None:
        experiment_id = uuid.uuid4().hex[:8]
        conn.execute(
            """
            INSERT INTO experiments (
                id, name, base_model, method, status, description, tags, config,
                metrics, best_checkpoint, parent_experiment_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                experiment_name,
                model_name,
                "full",
                "synced",
                "Auto-synced from XPI configs/logs",
                tags_json,
                config_json,
                metrics_json,
                None,
                None,
                now_iso,
                now_iso,
            ),
        )
    else:
        experiment_id = str(row[0])
        conn.execute(
            """
            UPDATE experiments
            SET base_model = ?, method = ?, status = ?, description = ?, tags = ?,
                config = ?, metrics = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                model_name,
                "full",
                "synced",
                "Auto-synced from XPI configs/logs",
                tags_json,
                config_json,
                metrics_json,
                now_iso,
                experiment_id,
            ),
        )

    run_id = uuid.uuid4().hex[:8]
    conn.execute(
        """
        INSERT INTO runs (
            id, experiment_id, status, config, metrics, artifacts,
            error_message, started_at, ended_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            experiment_id,
            "completed",
            config_json,
            metrics_json,
            json.dumps([]),
            None,
            now_iso,
            now_iso,
        ),
    )

    results = snapshot["results"]
    if isinstance(results, dict):
        step = 1
        for stage_name in ALL_STAGES:
            stage_metrics = results.get(stage_name, {})
            conn.execute(
                """
                INSERT INTO metrics_log (run_id, step, timestamp, metrics)
                VALUES (?, ?, ?, ?)
                """,
                (
                    run_id,
                    step,
                    now_iso,
                    json.dumps({"stage": stage_name, "metrics": stage_metrics}),
                ),
            )
            step += 1


def main() -> None:
    """Entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.db_path.parent.mkdir(parents=True, exist_ok=True)

    snapshots: list[dict[str, object]] = []
    for model_name in args.models:
        snapshot = build_model_snapshot(
            model_name=model_name,
            configs_dir=args.configs_dir,
            logs_dir=args.logs_dir,
        )
        snapshots.append(snapshot)
        logging.info("Prepared snapshot for %s", model_name)

    with sqlite3.connect(args.db_path) as conn:
        ensure_schema(conn)
        for snapshot in snapshots:
            upsert_snapshot(conn, snapshot)
        conn.commit()

    latest_path = args.output_dir / "latest_snapshots.json"
    latest_path.write_text(
        json.dumps(snapshots, indent=2, sort_keys=True), encoding="utf-8"
    )
    stamped_path = (
        args.output_dir
        / f"snapshots_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    stamped_path.write_text(
        json.dumps(snapshots, indent=2, sort_keys=True), encoding="utf-8"
    )

    logging.info("Synced %d models into %s", len(snapshots), args.db_path)
    logging.info("Snapshot files written: %s, %s", latest_path, stamped_path)


if __name__ == "__main__":
    main()
