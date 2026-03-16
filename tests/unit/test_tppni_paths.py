"""Unit tests for TPPNI dataset path handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.data_io import resolve_split_paths


def test_resolve_split_paths_uses_exact_configured_paths_when_tppni_enabled(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "pretrain_train_tppni.csv"
    valid_path = tmp_path / "pretrain_val_tppni.csv"
    test_path = tmp_path / "test.csv"
    for path in (train_path, valid_path, test_path):
        path.write_text("uniprotID_A,uniprotID_B,isInteraction\n", encoding="utf-8")

    config = {
        "data_config": {
            "dataloader": {
                "train_dataset": str(train_path),
                "valid_dataset": str(valid_path),
                "test_dataset": str(test_path),
            },
            "preprocessing": {
                "tppni": {
                    "enabled": True,
                }
            },
        }
    }

    resolved = resolve_split_paths(config=config, train_stage="pretrain")

    assert resolved["train"] == train_path
    assert resolved["valid"] == valid_path
    assert resolved["test"] == test_path


def test_resolve_split_paths_fails_without_exact_tppni_files(tmp_path: Path) -> None:
    base_train = tmp_path / "pretrain_train.csv"
    base_valid = tmp_path / "pretrain_val.csv"
    test_path = tmp_path / "test.csv"
    test_path.write_text("uniprotID_A,uniprotID_B,isInteraction\n", encoding="utf-8")

    config = {
        "data_config": {
            "dataloader": {
                "train_dataset": str(base_train),
                "valid_dataset": str(base_valid),
                "test_dataset": str(test_path),
            },
            "preprocessing": {
                "tppni": {
                    "enabled": True,
                }
            },
        }
    }

    with pytest.raises(FileNotFoundError, match=str(base_train)):
        resolve_split_paths(config=config, train_stage="pretrain")
