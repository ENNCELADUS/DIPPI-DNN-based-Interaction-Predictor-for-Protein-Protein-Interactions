"""Integration tests for config-driven TPPNI dataset generation."""

from __future__ import annotations

from itertools import combinations
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


def _clique_rows(prefix: str, clique_count: int, clique_size: int) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    for clique_index in range(clique_count):
        proteins = [f"{prefix}{clique_index}_{member}" for member in range(clique_size)]
        for protein_a, protein_b in combinations(proteins, 2):
            rows.append((protein_a, protein_b, 1))
    return rows


def _ratio(frame: pd.DataFrame) -> float:
    positives = int((frame["isInteraction"] == 1).sum())
    negatives = int((frame["isInteraction"] == 0).sum())
    if positives == 0:
        raise AssertionError("split unexpectedly has zero positives")
    return float(negatives) / float(positives)


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
    _write_pairs(processed_dir / "finetune.csv", _clique_rows("F", clique_count=4, clique_size=4))
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
                            "source_dataset": str(processed_dir / "missing_pretrain.csv"),
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
    pretrain_rows = _clique_rows("P", clique_count=4, clique_size=4)
    finetune_rows = _clique_rows("Q", clique_count=6, clique_size=4)
    test_rows = [("TEST_A", "TEST_B", 0), ("TEST_C", "TEST_D", 1)]
    _write_pairs(processed_dir / "pretrain.csv", pretrain_rows)
    _write_pairs(processed_dir / "finetune.csv", finetune_rows)
    _write_pairs(processed_dir / "test.csv", test_rows)
    test_before = (processed_dir / "test.csv").read_text(encoding="utf-8")
    proteins = sorted(
        {
            protein
            for row in pretrain_rows + finetune_rows + test_rows
            for protein in row[:2]
        }
    )
    _write_sequences(processed_dir / "all_proteins.csv", proteins)

    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        {
            "run_config": {"stages": ["pretrain", "finetune"], "seed": 11},
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
                        "candidate_limit": 1000,
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

    assert _ratio(pretrain_train) == pytest.approx(1.0)
    assert _ratio(pretrain_val) == pytest.approx(1.0)
    assert _ratio(finetune_train) == pytest.approx(_ratio(finetune_val))
    assert "score" not in pretrain_train.columns
    assert "score" not in finetune_train.columns
    assert (processed_dir / "test.csv").read_text(encoding="utf-8") == test_before

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["payload"]["test_dataset_unchanged"] is True
    assert manifest["stage_stats"]["pretrain"]["global_positive_count"] > 0
    assert manifest["stage_stats"]["pretrain"]["global_tppni_count"] > 0
    assert manifest["stage_stats"]["pretrain"]["post_downsample"]["train"]["negatives"] == manifest["stage_stats"]["pretrain"]["post_downsample"]["train"]["positives"]
    assert manifest["stage_stats"]["finetune"]["target_neg_pos_ratio"] == pytest.approx(
        manifest["stage_stats"]["finetune"]["post_downsample"]["train"]["negatives"]
        / manifest["stage_stats"]["finetune"]["post_downsample"]["train"]["positives"]
    )

    dataloaders = build_dataloaders_v6(
        config=yaml.safe_load(config_path.read_text(encoding="utf-8")),
        train_stage="pretrain",
    )
    train_batch = next(iter(dataloaders["train"]))
    assert len(train_batch["label"]) == 2
    assert set(train_batch["label"].tolist()).issubset({0.0, 1.0})

