"""Integration tests for the classical ML pipeline using DL-style config keys."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import src.embed.embed as embed_module
import src_ml.run as run_module
import torch
import yaml


class FakeModel:
    """Minimal sklearn-compatible model used for integration tests."""

    def __init__(self, **kwargs: object) -> None:
        del kwargs
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> "FakeModel":
        del y, eval_set
        self._is_fitted = True
        self._feature_count = int(X.shape[1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("model must be fitted")
        scores = X.mean(axis=1)
        max_score = float(np.max(scores)) if scores.size > 0 else 1.0
        if max_score <= 0.0:
            max_score = 1.0
        positive = np.clip(scores / max_score, 0.0, 1.0)
        negative = 1.0 - positive
        return np.stack([negative, positive], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)


def _write_pairs_csv(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("uniprotID_A,uniprotID_B,isInteraction\n")
        for protein_a, protein_b, label in rows:
            handle.write(f"{protein_a},{protein_b},{label}\n")


def _write_embedding_cache(
    cache_dir: Path,
    embeddings: dict[str, torch.Tensor],
    input_dim: int,
    max_sequence_length: int,
) -> None:
    index: dict[str, str] = {}
    for protein_id, tensor in embeddings.items():
        relative_path = embed_module._embedding_relative_path(protein_id)
        output_path = cache_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, output_path)
        index[protein_id] = relative_path

    metadata = {
        "schema_version": 1,
        "source": "esm3",
        "model_name": "esm3_sm_open_v1",
        "input_dim": input_dim,
        "max_sequence_length": max_sequence_length,
        "format": "torch_pt_per_protein",
    }
    (cache_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")
    (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_ml_run_pipeline_reuses_dl_cache_and_datasets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = tmp_path / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)

    finetune_train = processed_dir / "finetune_train_tppni.csv"
    finetune_valid = processed_dir / "finetune_val_tppni.csv"
    test_path = processed_dir / "test.csv"
    _write_pairs_csv(finetune_train, [("P1", "P2", 1), ("P3", "P4", 0)])
    _write_pairs_csv(finetune_valid, [("P1", "P3", 1), ("P2", "P4", 0)])
    _write_pairs_csv(test_path, [("P1", "P4", 1), ("P2", "P3", 0)])

    _write_embedding_cache(
        cache_dir=cache_dir,
        embeddings={
            "P1": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            "P2": torch.tensor([[2.0, 1.0], [4.0, 3.0]], dtype=torch.float32),
            "P3": torch.tensor([[10.0, 10.0]], dtype=torch.float32),
            "P4": torch.tensor([[20.0, 20.0]], dtype=torch.float32),
        },
        input_dim=2,
        max_sequence_length=8,
    )
    cache_files_before = sorted(
        path.relative_to(cache_dir).as_posix()
        for path in cache_dir.rglob("*")
        if path.is_file()
    )

    config = {
        "run_config": {
            "mode": "train_eval",
            "stages": ["finetune", "evaluate"],
            "seed": 47,
            "run_id": "integration_ml_run",
        },
        "data_config": {
            "benchmark": {
                "name": "TMP",
                "root_dir": str(tmp_path),
                "processed_dir": str(processed_dir),
            },
            "embeddings": {
                "source": "esm3",
                "cache_dir": str(cache_dir),
            },
            "max_sequence_length": 8,
            "preprocessing": {
                "tppni": {
                    "enabled": True,
                    "force_rebuild": False,
                    "candidate_limit": 1000,
                    "pretrain": {
                        "source_dataset": str(processed_dir / "pretrain.csv"),
                        "train_ratio": 0.95,
                        "train_dataset": str(processed_dir / "pretrain_train_tppni.csv"),
                        "valid_dataset": str(processed_dir / "pretrain_val_tppni.csv"),
                    },
                    "finetune": {
                        "source_dataset": str(processed_dir / "finetune.csv"),
                        "train_ratio": 0.90,
                        "train_dataset": str(finetune_train),
                        "valid_dataset": str(finetune_valid),
                    },
                }
            },
            "dataloader": {
                "train_dataset": str(processed_dir / "pretrain_train_tppni.csv"),
                "valid_dataset": str(processed_dir / "pretrain_val_tppni.csv"),
                "finetune_train_dataset": str(finetune_train),
                "finetune_val_dataset": str(finetune_valid),
                "test_dataset": str(test_path),
            },
            "pooling": "mean",
            "balance": False,
        },
        "model_config": {
            "model": "random_forest",
            "input_dim": 2,
            "random_forest": {
                "n_estimators": 10,
                "max_depth": 3,
            },
        },
        "evaluate": {
            "metrics": ["accuracy", "f1", "auprc", "auroc"],
            "threshold": 0.5,
        },
    }
    config_path = tmp_path / "ml.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        run_module,
        "build_ml_model",
        lambda model_name, **kwargs: FakeModel(**kwargs),
    )

    run_module.main(str(config_path))

    model_dir = tmp_path / "models" / "ml" / "random_forest" / "integration_ml_run"
    log_dir = tmp_path / "logs" / "ml" / "random_forest" / "integration_ml_run"
    assert (model_dir / "random_forest_model.joblib").exists()
    assert (log_dir / "results.json").exists()
    assert (log_dir / "evaluation_results.csv").exists()

    results = json.loads((log_dir / "results.json").read_text(encoding="utf-8"))
    assert results["run_id"] == "integration_ml_run"
    assert results["data"]["feature_dim"] == 4
    assert results["data"]["train_samples"] == 2
    assert not (processed_dir / "TMP_embeddings.npz").exists()

    cache_files_after = sorted(
        path.relative_to(cache_dir).as_posix()
        for path in cache_dir.rglob("*")
        if path.is_file()
    )
    assert cache_files_after == cache_files_before
