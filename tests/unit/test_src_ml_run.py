"""Unit tests for classical ML runtime input resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
import src_ml.run as run_module

from src.embed import EmbeddingCacheManifest
from src.utils.config import ConfigDict


def _build_config(cache_dir: Path) -> ConfigDict:
    return {
        "run_config": {
            "mode": "train_eval",
            "stages": ["finetune", "evaluate"],
            "seed": 47,
            "run_id": "unit_run",
        },
        "data_config": {
            "benchmark": {
                "name": "TMP",
                "root_dir": str(cache_dir.parent),
                "processed_dir": str(cache_dir.parent),
            },
            "embeddings": {
                "source": "esm3",
                "cache_dir": str(cache_dir),
            },
            "max_sequence_length": 1024,
            "preprocessing": {
                "tppni": {
                    "enabled": True,
                    "force_rebuild": False,
                    "candidate_limit": 1000,
                    "pretrain": {
                        "source_dataset": str(cache_dir.parent / "pretrain.csv"),
                        "train_ratio": 0.95,
                        "train_dataset": str(cache_dir.parent / "pretrain_train_tppni.csv"),
                        "valid_dataset": str(cache_dir.parent / "pretrain_val_tppni.csv"),
                    },
                    "finetune": {
                        "source_dataset": str(cache_dir.parent / "finetune.csv"),
                        "train_ratio": 0.90,
                        "train_dataset": str(cache_dir.parent / "finetune_train_tppni.csv"),
                        "valid_dataset": str(cache_dir.parent / "finetune_val_tppni.csv"),
                    },
                }
            },
            "dataloader": {
                "train_dataset": str(cache_dir.parent / "pretrain_train_tppni.csv"),
                "valid_dataset": str(cache_dir.parent / "pretrain_val_tppni.csv"),
                "finetune_train_dataset": str(cache_dir.parent / "finetune_train_tppni.csv"),
                "finetune_val_dataset": str(cache_dir.parent / "finetune_val_tppni.csv"),
                "test_dataset": str(cache_dir.parent / "test.csv"),
            },
            "pooling": "mean",
            "balance": False,
        },
        "model_config": {
            "model": "random_forest",
            "input_dim": 1536,
            "random_forest": {
                "n_estimators": 10,
            },
        },
        "evaluate": {
            "metrics": ["accuracy"],
            "threshold": 0.5,
        },
    }


def test_resolve_ml_runtime_inputs_uses_finetune_splits_and_cache_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    config = _build_config(cache_dir=cache_dir)
    train_path = tmp_path / "finetune_train_tppni.csv"
    valid_path = tmp_path / "finetune_val_tppni.csv"
    test_path = tmp_path / "test.csv"

    captured: dict[str, object] = {}

    def _fake_resolve_split_paths(
        config: ConfigDict,
        train_stage: str = "pretrain",
    ) -> dict[str, Path]:
        del config
        captured["train_stage"] = train_stage
        return {"train": train_path, "valid": valid_path, "test": test_path}

    def _fake_ensure_embedding_cache(
        config: ConfigDict,
        split_paths: list[Path],
        input_dim: int,
        max_sequence_length: int,
        distributed: bool = False,
        rank: int = 0,
    ) -> EmbeddingCacheManifest:
        del config, distributed, rank
        captured["split_paths"] = split_paths
        captured["input_dim"] = input_dim
        captured["max_sequence_length"] = max_sequence_length
        return EmbeddingCacheManifest(
            cache_dir=cache_dir,
            index={"P1": "embeddings/p1.pt"},
            required_ids=frozenset({"P1"}),
        )

    monkeypatch.setattr(run_module, "resolve_split_paths", _fake_resolve_split_paths)
    monkeypatch.setattr(
        run_module, "ensure_embedding_cache", _fake_ensure_embedding_cache
    )

    runtime_inputs = run_module.resolve_ml_runtime_inputs(config)

    assert captured["train_stage"] == "finetune"
    assert captured["split_paths"] == [train_path, valid_path, test_path]
    assert captured["input_dim"] == 1536
    assert captured["max_sequence_length"] == 1024
    assert runtime_inputs.train_path == train_path
    assert runtime_inputs.valid_path == valid_path
    assert runtime_inputs.test_path == test_path
    assert runtime_inputs.embeddings_path == cache_dir
    assert runtime_inputs.pooling == "mean"
    assert runtime_inputs.balance is False
