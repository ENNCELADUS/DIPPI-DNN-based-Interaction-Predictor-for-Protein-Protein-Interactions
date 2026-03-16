"""Config-driven CLI for TPPNI dataset preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from src.data_preprocess.tppni import (
    CleaningConfig,
    PairCleaningStats,
    build_global_tppni_pool,
    build_stage_datasets_from_pools,
    clean_pairs,
    split_proteins_inductively,
)
from src.utils.config import (
    ConfigDict,
    as_bool,
    as_float,
    as_int,
    as_str,
    get_section,
    load_config,
)

LOGGER = logging.getLogger(__name__)
MANIFEST_NAME = "tppni_preprocess_manifest.json"
ENFORCED_CLEANING_CONFIG = CleaningConfig(
    drop_missing=True,
    drop_self_loops=True,
    canonicalize_undirected_pairs=True,
    deduplicate_within_label=True,
    conflict_policy="error",
)


@dataclass(frozen=True)
class StageDatasetConfig:
    """Dataset generation settings for one stage."""

    source_dataset: Path
    train_ratio: float
    train_dataset: Path
    valid_dataset: Path


@dataclass(frozen=True)
class TPPNIConfig:
    """Root preprocessing configuration."""

    enabled: bool
    force_rebuild: bool
    candidate_limit: int
    pretrain: StageDatasetConfig
    finetune: StageDatasetConfig


def _stage_items_for_names(
    tppni_config: TPPNIConfig,
    stage_names: list[str],
) -> list[tuple[str, StageDatasetConfig]]:
    stage_items: list[tuple[str, StageDatasetConfig]] = []
    if "pretrain" in stage_names:
        stage_items.append(("pretrain", tppni_config.pretrain))
    if "finetune" in stage_names:
        stage_items.append(("finetune", tppni_config.finetune))
    return stage_items


def _parse_stage_config(raw: ConfigDict, field_name: str) -> StageDatasetConfig:
    if "negative_ratio_mode" in raw:
        raise ValueError(
            f"{field_name}.negative_ratio_mode is no longer supported; "
            "TPPNI outputs now use the full post-CL3 negative set"
        )
    return StageDatasetConfig(
        source_dataset=Path(
            as_str(raw.get("source_dataset"), f"{field_name}.source_dataset")
        ),
        train_ratio=as_float(raw.get("train_ratio"), f"{field_name}.train_ratio"),
        train_dataset=Path(
            as_str(raw.get("train_dataset"), f"{field_name}.train_dataset")
        ),
        valid_dataset=Path(
            as_str(raw.get("valid_dataset"), f"{field_name}.valid_dataset")
        ),
    )


def _load_tppni_config(config: ConfigDict) -> TPPNIConfig | None:
    data_cfg = get_section(config, "data_config")
    preprocessing_raw = data_cfg.get("preprocessing")
    if preprocessing_raw is None:
        return None
    if not isinstance(preprocessing_raw, dict):
        raise ValueError("data_config.preprocessing must be a mapping")
    preprocessing_cfg = cast(ConfigDict, preprocessing_raw)

    tppni_raw = preprocessing_cfg.get("tppni")
    if tppni_raw is None:
        return None
    if not isinstance(tppni_raw, dict):
        raise ValueError("data_config.preprocessing.tppni must be a mapping")
    tppni_cfg = cast(ConfigDict, tppni_raw)

    if "cleaning" in tppni_cfg:
        raise ValueError(
            "data_config.preprocessing.tppni.cleaning is enforced in code and "
            "must be removed from config files"
        )

    pretrain_raw = tppni_cfg.get("pretrain")
    finetune_raw = tppni_cfg.get("finetune")
    if not isinstance(pretrain_raw, dict) or not isinstance(finetune_raw, dict):
        raise ValueError(
            "data_config.preprocessing.tppni.pretrain and .finetune must be mappings"
        )

    return TPPNIConfig(
        enabled=as_bool(
            tppni_cfg.get("enabled", False), "data_config.preprocessing.tppni.enabled"
        ),
        force_rebuild=as_bool(
            tppni_cfg.get("force_rebuild", False),
            "data_config.preprocessing.tppni.force_rebuild",
        ),
        candidate_limit=as_int(
            tppni_cfg.get("candidate_limit", 10_000_000),
            "data_config.preprocessing.tppni.candidate_limit",
        ),
        pretrain=_parse_stage_config(
            cast(ConfigDict, pretrain_raw),
            "data_config.preprocessing.tppni.pretrain",
        ),
        finetune=_parse_stage_config(
            cast(ConfigDict, finetune_raw),
            "data_config.preprocessing.tppni.finetune",
        ),
    )


def _manifest_payload(
    tppni_config: TPPNIConfig,
    stage_items: list[tuple[str, StageDatasetConfig]],
) -> dict[str, object]:
    sources = {}
    stage_settings = {}
    for stage_name, stage_config in stage_items:
        source_path = stage_config.source_dataset
        if source_path.exists():
            stat_result = source_path.stat()
            sources[stage_name] = {
                "path": str(source_path),
                "size": stat_result.st_size,
                "mtime_ns": stat_result.st_mtime_ns,
            }
        stage_settings[stage_name] = {
            "source_dataset": str(stage_config.source_dataset),
            "train_ratio": stage_config.train_ratio,
            "train_dataset": str(stage_config.train_dataset),
            "valid_dataset": str(stage_config.valid_dataset),
        }

    return {
        "stages": [stage_name for stage_name, _ in stage_items],
        "candidate_limit": tppni_config.candidate_limit,
        "cleaning": asdict(ENFORCED_CLEANING_CONFIG),
        "stages_config": stage_settings,
        "source_files": sources,
        "test_dataset_unchanged": True,
    }


def _manifest_path(stage_items: list[tuple[str, StageDatasetConfig]]) -> Path:
    return stage_items[0][1].train_dataset.parent / MANIFEST_NAME


def _outputs_exist(stage_items: list[tuple[str, StageDatasetConfig]]) -> bool:
    required_outputs = []
    for _, stage_config in stage_items:
        required_outputs.extend(
            [stage_config.train_dataset, stage_config.valid_dataset]
        )
    return all(path.exists() for path in required_outputs)


def _manifest_matches(
    manifest_path: Path,
    payload: dict[str, object],
) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        manifest.get("fingerprint")
        == hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
    )


def _write_manifest(
    manifest_path: Path,
    payload: dict[str, object],
    cleaning_stats: dict[str, PairCleaningStats],
    stage_stats: dict[str, dict[str, object]],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "fingerprint": hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "payload": payload,
        "cleaning_stats": {
            stage_name: asdict(stage_stats)
            for stage_name, stage_stats in cleaning_stats.items()
        },
        "stage_stats": stage_stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _prepare_stage(
    stage_name: str,
    stage_config: StageDatasetConfig,
    tppni_config: TPPNIConfig,
    seed: int,
) -> tuple[PairCleaningStats, dict[str, object]]:
    LOGGER.info(
        "Preparing %s datasets from %s", stage_name, stage_config.source_dataset
    )
    raw_frame = pd.read_csv(stage_config.source_dataset)
    cleaned_frame, cleaning_stats = clean_pairs(raw_frame, ENFORCED_CLEANING_CONFIG)

    positive_pairs = cleaned_frame.loc[
        cleaned_frame["isInteraction"] == 1,
        ["uniprotID_A", "uniprotID_B", "isInteraction"],
    ].reset_index(drop=True)
    graph, tppni_candidates = build_global_tppni_pool(
        positive_pairs=positive_pairs,
        candidate_limit=tppni_config.candidate_limit,
    )
    protein_split = split_proteins_inductively(
        positive_pairs=positive_pairs,
        train_ratio=stage_config.train_ratio,
        seed=seed,
    )
    negative_pool = tppni_candidates.to_frame(graph)
    negative_pool["isInteraction"] = 0
    stage_datasets = build_stage_datasets_from_pools(
        stage_name=stage_name,
        positive_pairs=positive_pairs,
        negative_pool=negative_pool,
        protein_split=protein_split,
        seed=seed,
    )

    stage_config.train_dataset.parent.mkdir(parents=True, exist_ok=True)
    stage_config.valid_dataset.parent.mkdir(parents=True, exist_ok=True)
    stage_datasets.train_dataset.to_csv(stage_config.train_dataset, index=False)
    stage_datasets.valid_dataset.to_csv(stage_config.valid_dataset, index=False)
    LOGGER.info(
        "Wrote %s train=%s valid=%s",
        stage_name,
        stage_config.train_dataset,
        stage_config.valid_dataset,
    )
    return cleaning_stats, {
        "global_positive_count": len(positive_pairs),
        "global_tppni_count": len(negative_pool),
        "train_protein_count": len(protein_split.train_proteins),
        "valid_protein_count": len(protein_split.valid_proteins),
        **stage_datasets.summary,
    }


def run_from_config_path(config_path: Path) -> int:
    """Execute TPPNI preprocessing from one YAML config path."""
    config = load_config(config_path)
    run_cfg = get_section(config, "run_config")
    stages_raw = run_cfg.get("stages", [])
    if not isinstance(stages_raw, list):
        raise ValueError("run_config.stages must be a sequence")
    stages = [str(stage).lower() for stage in stages_raw]
    if not any(stage in {"pretrain", "finetune"} for stage in stages):
        LOGGER.info("No training stage requested; skipping TPPNI preprocessing")
        return 0

    tppni_config = _load_tppni_config(config)
    if tppni_config is None:
        LOGGER.info("No data_config.preprocessing.tppni section found; skipping")
        return 0
    if not tppni_config.enabled:
        LOGGER.info("TPPNI preprocessing is disabled; skipping")
        return 0

    stage_items = _stage_items_for_names(tppni_config, stages)
    if not stage_items:
        LOGGER.info("No TPPNI-enabled stages requested; skipping")
        return 0

    payload = _manifest_payload(tppni_config, stage_items)
    manifest_path = _manifest_path(stage_items)
    if (
        not tppni_config.force_rebuild
        and _outputs_exist(stage_items)
        and _manifest_matches(manifest_path, payload)
    ):
        LOGGER.info("TPPNI outputs are up to date; skipping rebuild")
        return 0

    seed = as_int(run_cfg.get("seed", 0), "run_config.seed")
    cleaning_stats = {}
    stage_stats = {}
    for offset, (stage_name, stage_config) in enumerate(stage_items):
        stage_cleaning_stats, stage_summary = _prepare_stage(
            stage_name,
            stage_config,
            tppni_config,
            seed=seed + (offset * 1000),
        )
        cleaning_stats[stage_name] = stage_cleaning_stats
        stage_stats[stage_name] = stage_summary
    _write_manifest(manifest_path, payload, cleaning_stats, stage_stats)
    LOGGER.info("Wrote TPPNI preprocessing manifest to %s", manifest_path)
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Prepare TPPNI-based dataset splits")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = build_argument_parser().parse_args(argv)
    return run_from_config_path(Path(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
