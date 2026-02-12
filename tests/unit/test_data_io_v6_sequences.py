"""Unit tests for V6 sequence-native dataloader construction."""

from __future__ import annotations

from pathlib import Path

import pytest
import src.utils.data_io_v6 as data_io_v6
import torch
from src.utils.config import ConfigDict


def _write_split(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("uniprotID_A,uniprotID_B,isInteraction\n")
        for protein_a, protein_b, label in rows:
            handle.write(f"{protein_a},{protein_b},{label}\n")


def _write_sequences(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("uniprotID,sequence\n")
        for protein_id, sequence in rows:
            handle.write(f"{protein_id},{sequence}\n")


def _build_config(
    sequence_path: Path,
    train_path: Path,
    valid_path: Path,
    test_path: Path,
    finetune_train_path: Path | None = None,
    finetune_valid_path: Path | None = None,
) -> ConfigDict:
    dataloader_cfg: dict[str, object] = {
        "train_dataset": str(train_path),
        "valid_dataset": str(valid_path),
        "test_dataset": str(test_path),
        "num_workers": 0,
        "pin_memory": False,
        "drop_last": False,
    }
    if finetune_train_path is not None:
        dataloader_cfg["finetune_train_dataset"] = str(finetune_train_path)
    if finetune_valid_path is not None:
        dataloader_cfg["finetune_val_dataset"] = str(finetune_valid_path)

    return {
        "run_config": {"seed": 11},
        "data_config": {
            "sequences": {
                "source_file": str(sequence_path),
                "id_column": "uniprotID",
                "sequence_column": "sequence",
            },
            "dataloader": dataloader_cfg,
        },
        "training_config": {"batch_size": 2},
    }


def test_build_dataloaders_v6_returns_sequence_batches(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    valid_path = tmp_path / "valid.csv"
    test_path = tmp_path / "test.csv"
    _write_split(train_path, [("P1", "P2", 1), ("P3", "P4", 0)])
    _write_split(valid_path, [("P1", "P3", 1)])
    _write_split(test_path, [("P2", "P4", 0)])

    sequence_path = tmp_path / "all_proteins.csv"
    _write_sequences(
        sequence_path,
        [
            ("P1", "AAAA"),
            ("P2", "BBBBB"),
            ("P3", "CC"),
            ("P4", "DDDD"),
        ],
    )

    config = _build_config(
        sequence_path=sequence_path,
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
    )
    dataloaders = data_io_v6.build_dataloaders_v6(config=config)
    valid_batch = next(iter(dataloaders["valid"]))

    seq_a = valid_batch["seq_a"]
    seq_b = valid_batch["seq_b"]
    labels = valid_batch["label"]
    assert isinstance(seq_a, list)
    assert isinstance(seq_b, list)
    assert seq_a == ["AAAA"]
    assert seq_b == ["CC"]
    assert isinstance(labels, torch.Tensor)
    assert labels.tolist() == [1.0]


def test_build_dataloaders_v6_missing_sequence_raises(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    valid_path = tmp_path / "valid.csv"
    test_path = tmp_path / "test.csv"
    _write_split(train_path, [("P1", "P2", 1)])
    _write_split(valid_path, [("P1", "P2", 1)])
    _write_split(test_path, [("P1", "P2", 1)])

    sequence_path = tmp_path / "all_proteins.csv"
    _write_sequences(sequence_path, [("P1", "AAAA")])

    config = _build_config(
        sequence_path=sequence_path,
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
    )
    with pytest.raises(FileNotFoundError, match="missing"):
        data_io_v6.build_dataloaders_v6(config=config)
