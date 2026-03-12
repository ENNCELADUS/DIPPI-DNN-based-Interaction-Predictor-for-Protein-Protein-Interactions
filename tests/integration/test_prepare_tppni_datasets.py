"""Integration tests for config-driven TPPNI dataset generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data_preprocess.prepare_tppni_datasets import main as prepare_tppni_main
from src.utils.config import ConfigDict
from src.utils.data_io_v6 import build_dataloaders_v6


def _write_pairs(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("uniprotID_A,uniprotID_B,isInteraction\n")
        for protein_a, protein_b, label in rows:
            handle.write(f"{protein_a},{protein_b},{label}\n")


def _write_sequences(path: Path, proteins: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("uniprotID,sequence\n")
        for protein in proteins:
            handle.write(f"{protein},AAAA\n")


def _write_config(path: Path, config: ConfigDict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def test_prepare_tppni_datasets_disabled_is_no_op(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    _write_pairs(processed_dir / "pretrain.csv", [("A", "B", 1), ("C", "D", 1)])
    _write_pairs(processed_dir / "finetune.csv", [("E", "F", 1), ("G", "H", 0)])
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        {
            "run_config": {"stages": ["pretrain", "finetune"]},
            "data_config": {
                "preprocessing": {
                    "tppni": {
                        "enabled": False,
                        "force_rebuild": False,
                        "candidate_limit": 100,
                        "pretrain": {
                            "source_dataset": str(processed_dir / "pretrain.csv"),
                            "train_ratio": 0.5,
                            "train_dataset": str(processed_dir / "pretrain_train.csv"),
                            "valid_dataset": str(processed_dir / "pretrain_val.csv"),
                        },
                        "finetune": {
                            "source_dataset": str(processed_dir / "finetune.csv"),
                            "train_ratio": 0.5,
                            "train_dataset": str(processed_dir / "finetune_train.csv"),
                            "valid_dataset": str(processed_dir / "finetune_val.csv"),
                        },
                    }
                }
            },
        },
    )

    exit_code = prepare_tppni_main(["--config", str(config_path)])

    assert exit_code == 0
    assert not (processed_dir / "pretrain_train.csv").exists()
    assert not (processed_dir / "tppni_preprocess_manifest.json").exists()


def test_prepare_tppni_datasets_rejects_cleaning_config_override(
    tmp_path: Path,
) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    _write_pairs(processed_dir / "pretrain.csv", [("A", "B", 1), ("C", "D", 1)])
    _write_pairs(processed_dir / "finetune.csv", [("E", "F", 1), ("G", "H", 0)])
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        {
            "run_config": {"stages": ["pretrain", "finetune"]},
            "data_config": {
                "preprocessing": {
                    "tppni": {
                        "enabled": True,
                        "force_rebuild": False,
                        "candidate_limit": 100,
                        "cleaning": {"drop_missing": False},
                        "pretrain": {
                            "source_dataset": str(processed_dir / "pretrain.csv"),
                            "train_ratio": 0.5,
                            "train_dataset": str(processed_dir / "pretrain_train.csv"),
                            "valid_dataset": str(processed_dir / "pretrain_val.csv"),
                            "negative_ratio_mode": "balanced",
                        },
                        "finetune": {
                            "source_dataset": str(processed_dir / "finetune.csv"),
                            "train_ratio": 0.5,
                            "train_dataset": str(processed_dir / "finetune_train.csv"),
                            "valid_dataset": str(processed_dir / "finetune_val.csv"),
                            "negative_ratio_mode": "preserve_input",
                        },
                    }
                }
            },
        },
    )

    with pytest.raises(ValueError, match="must be removed from config files"):
        prepare_tppni_main(["--config", str(config_path)])


def test_prepare_tppni_datasets_rejects_negative_ratio_mode_override(
    tmp_path: Path,
) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    _write_pairs(processed_dir / "pretrain.csv", [("A", "B", 1), ("C", "D", 1)])
    _write_pairs(processed_dir / "finetune.csv", [("E", "F", 1), ("G", "H", 0)])
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        {
            "run_config": {"stages": ["pretrain", "finetune"]},
            "data_config": {
                "preprocessing": {
                    "tppni": {
                        "enabled": True,
                        "force_rebuild": False,
                        "candidate_limit": 100,
                        "pretrain": {
                            "source_dataset": str(processed_dir / "pretrain.csv"),
                            "train_ratio": 0.5,
                            "train_dataset": str(processed_dir / "pretrain_train.csv"),
                            "valid_dataset": str(processed_dir / "pretrain_val.csv"),
                            "negative_ratio_mode": "balanced",
                        },
                        "finetune": {
                            "source_dataset": str(processed_dir / "finetune.csv"),
                            "train_ratio": 0.5,
                            "train_dataset": str(processed_dir / "finetune_train.csv"),
                            "valid_dataset": str(processed_dir / "finetune_val.csv"),
                        },
                    }
                }
            },
        },
    )

    with pytest.raises(ValueError, match="no longer supported"):
        prepare_tppni_main(["--config", str(config_path)])


def test_prepare_tppni_datasets_only_builds_requested_stage(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    _write_pairs(
        processed_dir / "finetune.csv",
        [
            ("I", "J", 1),
            ("K", "L", 1),
            ("M", "N", 1),
            ("O", "P", 1),
            ("I", "K", 0),
            ("I", "L", 0),
            ("J", "K", 0),
            ("J", "L", 0),
            ("M", "O", 0),
            ("M", "P", 0),
            ("N", "O", 0),
            ("N", "P", 0),
        ],
    )
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        {
            "run_config": {"stages": ["finetune"], "seed": 13},
            "data_config": {
                "preprocessing": {
                    "tppni": {
                        "enabled": True,
                        "force_rebuild": False,
                        "candidate_limit": 100,
                        "pretrain": {
                            "source_dataset": str(
                                processed_dir / "missing_pretrain.csv"
                            ),
                            "train_ratio": 0.5,
                            "train_dataset": str(processed_dir / "pretrain_train.csv"),
                            "valid_dataset": str(processed_dir / "pretrain_val.csv"),
                        },
                        "finetune": {
                            "source_dataset": str(processed_dir / "finetune.csv"),
                            "train_ratio": 0.5,
                            "train_dataset": str(processed_dir / "finetune_train.csv"),
                            "valid_dataset": str(processed_dir / "finetune_val.csv"),
                        },
                    }
                }
            },
        },
    )

    exit_code = prepare_tppni_main(["--config", str(config_path)])

    assert exit_code == 0
    assert not (processed_dir / "pretrain_train.csv").exists()
    assert (processed_dir / "finetune_train.csv").exists()
    assert (processed_dir / "finetune_val.csv").exists()
    manifest = yaml.safe_load(
        (processed_dir / "tppni_preprocess_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["payload"]["stages"] == ["finetune"]
    assert "pretrain" not in manifest["payload"]["stages_config"]


def test_prepare_tppni_datasets_generates_outputs_manifest_and_loader_inputs(
    tmp_path: Path,
) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    pretrain_rows = [
        ("A", "B", 1),
        ("C", "D", 1),
        ("E", "F", 1),
        ("G", "H", 1),
        ("A", "A", 1),
        ("", "Z", 0),
        ("H", "G", 1),
    ]
    finetune_rows = [
        ("I", "J", 1),
        ("K", "L", 1),
        ("M", "N", 1),
        ("O", "P", 1),
        ("Q", "R", 1),
        ("S", "T", 1),
        ("I", "K", 0),
        ("I", "L", 0),
        ("J", "K", 0),
        ("J", "L", 0),
        ("M", "O", 0),
        ("M", "P", 0),
        ("N", "O", 0),
        ("N", "P", 0),
        ("Q", "S", 0),
        ("Q", "T", 0),
        ("R", "S", 0),
        ("R", "T", 0),
    ]
    _write_pairs(processed_dir / "pretrain.csv", pretrain_rows)
    _write_pairs(processed_dir / "finetune.csv", finetune_rows)
    _write_pairs(processed_dir / "test.csv", [("A", "C", 0)])
    proteins = sorted(
        {item for row in pretrain_rows + finetune_rows for item in row[:2] if item}
    )
    _write_sequences(processed_dir / "all_proteins.csv", proteins)

    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        {
            "run_config": {"stages": ["pretrain", "finetune"]},
            "data_config": {
                "sequences": {
                    "source_file": str(processed_dir / "all_proteins.csv"),
                    "id_column": "uniprotID",
                    "sequence_column": "sequence",
                },
                "dataloader": {
                    "train_dataset": str(processed_dir / "pretrain_train.csv"),
                    "valid_dataset": str(processed_dir / "pretrain_val.csv"),
                    "finetune_train_dataset": str(processed_dir / "finetune_train.csv"),
                    "finetune_val_dataset": str(processed_dir / "finetune_val.csv"),
                    "test_dataset": str(processed_dir / "test.csv"),
                    "num_workers": 0,
                    "pin_memory": False,
                    "drop_last": False,
                    "sampling": {"strategy": "none"},
                },
                "preprocessing": {
                    "tppni": {
                        "enabled": True,
                        "force_rebuild": False,
                        "candidate_limit": 100,
                        "pretrain": {
                            "source_dataset": str(processed_dir / "pretrain.csv"),
                            "train_ratio": 0.5,
                            "train_dataset": str(processed_dir / "pretrain_train.csv"),
                            "valid_dataset": str(processed_dir / "pretrain_val.csv"),
                        },
                        "finetune": {
                            "source_dataset": str(processed_dir / "finetune.csv"),
                            "train_ratio": 0.5,
                            "train_dataset": str(processed_dir / "finetune_train.csv"),
                            "valid_dataset": str(processed_dir / "finetune_val.csv"),
                        },
                    }
                },
            },
            "training_config": {"batch_size": 2},
        },
    )

    exit_code = prepare_tppni_main(["--config", str(config_path)])

    assert exit_code == 0
    manifest_path = processed_dir / "tppni_preprocess_manifest.json"
    assert manifest_path.exists()

    pretrain_train = pd.read_csv(processed_dir / "pretrain_train.csv")
    pretrain_val = pd.read_csv(processed_dir / "pretrain_val.csv")
    finetune_train = pd.read_csv(processed_dir / "finetune_train.csv")
    finetune_val = pd.read_csv(processed_dir / "finetune_val.csv")

    assert pretrain_train["isInteraction"].value_counts().to_dict() == {0: 4, 1: 2}
    assert pretrain_val["isInteraction"].value_counts().to_dict() == {0: 4, 1: 2}
    assert finetune_train["isInteraction"].value_counts().to_dict() == {0: 12, 1: 3}
    assert finetune_val["isInteraction"].value_counts().to_dict() == {0: 12, 1: 3}

    dataloaders = build_dataloaders_v6(
        config=yaml.safe_load(config_path.read_text(encoding="utf-8")),
        train_stage="pretrain",
    )
    train_batch = next(iter(dataloaders["train"]))
    assert len(train_batch["label"]) == 2
    assert set(train_batch["label"].tolist()).issubset({0.0, 1.0})
