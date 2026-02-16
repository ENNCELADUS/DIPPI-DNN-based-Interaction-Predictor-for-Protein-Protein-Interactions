"""Optuna-based finetune hyperparameter search runner."""

from __future__ import annotations

import argparse
import copy
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import optuna
import yaml
from optuna.pruners import BasePruner, MedianPruner
from optuna.samplers import BaseSampler, TPESampler
from optuna.trial import Trial

import src.run as run_module
from src.utils.config import (
    ConfigDict,
    as_bool,
    as_float,
    as_int,
    as_str,
    extract_model_kwargs,
    get_section,
    load_config,
)

LOGGER = logging.getLogger(__name__)
ScalarChoice = str | int | float | bool


@dataclass(frozen=True)
class ParameterSpec:
    """Search-space description for one logical hyperparameter."""

    name: str
    distribution: ConfigDict
    apply_to: tuple[str, ...]


@dataclass(frozen=True)
class OptunaConfig:
    """Validated Optuna settings for finetune search."""

    enabled: bool
    study_name: str
    storage: str
    direction: str
    metric: str
    n_trials: int
    timeout_sec: int | None
    sampler_type: str
    sampler_seed: int
    pruner_type: str
    pruner_n_startup_trials: int
    pruner_n_warmup_steps: int
    pruner_interval_steps: int
    parameters: tuple[ParameterSpec, ...]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Optuna finetune search."""
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter search for finetune stage."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional override for finetune seed checkpoint path.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optional override for Optuna study name.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Optional override for number of trials.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=None,
        help="Optional override for optimization timeout in seconds.",
    )
    return parser.parse_args()


def _as_config_dict(value: object, field_name: str) -> ConfigDict:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return cast(ConfigDict, value)


def _path_exists(root: ConfigDict, path: str) -> bool:
    current: object = root
    for token in path.split("."):
        if not isinstance(current, dict):
            return False
        if token not in current:
            return False
        current = current[token]
    return True


def _set_nested_value(root: ConfigDict, path: str, value: object) -> None:
    tokens = path.split(".")
    current: object = root
    for token in tokens[:-1]:
        if not isinstance(current, dict):
            raise ValueError(f"Cannot set nested path '{path}': parent is not mapping")
        next_value = current.get(token)
        if not isinstance(next_value, dict):
            raise ValueError(
                f"Cannot set nested path '{path}': missing mapping '{token}'"
            )
        current = next_value
    if not isinstance(current, dict):
        raise ValueError(f"Cannot set nested path '{path}': parent is not mapping")
    leaf = tokens[-1]
    if leaf not in current:
        raise ValueError(f"Cannot set nested path '{path}': missing key '{leaf}'")
    current[leaf] = value


def _read_scalar_choices(values: object, field_name: str) -> list[ScalarChoice]:
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must be a list")
    choices: list[ScalarChoice] = []
    for index, value in enumerate(values):
        if isinstance(value, bool):
            choices.append(value)
            continue
        if isinstance(value, (str, int, float)):
            choices.append(value)
            continue
        raise ValueError(f"{field_name}[{index}] must be scalar")
    if not choices:
        raise ValueError(f"{field_name} must not be empty")
    return choices


def _validate_distribution(distribution: ConfigDict, field_prefix: str) -> ConfigDict:
    distribution_type = as_str(
        distribution.get("type", ""), f"{field_prefix}.type"
    ).lower()
    if distribution_type == "float":
        low = as_float(distribution.get("low"), f"{field_prefix}.low")
        high = as_float(distribution.get("high"), f"{field_prefix}.high")
        if low >= high:
            raise ValueError(f"{field_prefix} requires low < high")
        result: ConfigDict = {"type": "float", "low": low, "high": high}
        if "log" in distribution:
            result["log"] = as_bool(distribution.get("log"), f"{field_prefix}.log")
        if "step" in distribution:
            result["step"] = as_float(distribution.get("step"), f"{field_prefix}.step")
        if result.get("log") and "step" in result:
            raise ValueError(f"{field_prefix} cannot set both log=true and step")
        return result

    if distribution_type == "int":
        low = as_int(distribution.get("low"), f"{field_prefix}.low")
        high = as_int(distribution.get("high"), f"{field_prefix}.high")
        if low >= high:
            raise ValueError(f"{field_prefix} requires low < high")
        result = {"type": "int", "low": low, "high": high}
        if "log" in distribution:
            result["log"] = as_bool(distribution.get("log"), f"{field_prefix}.log")
        if "step" in distribution:
            step = as_int(distribution.get("step"), f"{field_prefix}.step")
            if step <= 0:
                raise ValueError(f"{field_prefix}.step must be > 0")
            result["step"] = step
        if result.get("log") and "step" in result:
            raise ValueError(f"{field_prefix} cannot set both log=true and step")
        return result

    if distribution_type == "categorical":
        choices = _read_scalar_choices(
            distribution.get("choices"),
            f"{field_prefix}.choices",
        )
        return {
            "type": "categorical",
            "choices": choices,
        }

    raise ValueError(f"{field_prefix}.type must be one of: float, int, categorical")


def parse_optuna_config(
    config: ConfigDict,
    study_name_override: str | None = None,
    n_trials_override: int | None = None,
    timeout_sec_override: int | None = None,
) -> OptunaConfig:
    """Validate and normalize ``training_config.finetune.optuna`` settings."""
    model_name, _ = extract_model_kwargs(config)
    run_cfg = get_section(config, "run_config")
    run_seed = as_int(run_cfg.get("seed", 0), "run_config.seed")
    training_cfg = get_section(config, "training_config")
    finetune_cfg = _as_config_dict(
        training_cfg.get("finetune"),
        "training_config.finetune",
    )
    raw_optuna_cfg = _as_config_dict(
        finetune_cfg.get("optuna", {}),
        "training_config.finetune.optuna",
    )
    enabled = as_bool(
        raw_optuna_cfg.get("enabled", False),
        "training_config.finetune.optuna.enabled",
    )

    study_name = (
        study_name_override
        if study_name_override is not None
        else as_str(
            raw_optuna_cfg.get("study_name", f"{model_name}_finetune"),
            "training_config.finetune.optuna.study_name",
        )
    )
    default_storage = f"sqlite:///logs/optuna/{model_name}_finetune.db"
    storage = as_str(
        raw_optuna_cfg.get("storage", default_storage),
        "training_config.finetune.optuna.storage",
    )
    direction = as_str(
        raw_optuna_cfg.get("direction", "maximize"),
        "training_config.finetune.optuna.direction",
    ).lower()
    if direction not in {"maximize", "minimize"}:
        raise ValueError("training_config.finetune.optuna.direction must be valid")

    fallback_metric = as_str(
        finetune_cfg.get("monitor_metric", training_cfg.get("monitor_metric", "auprc")),
        "training_config.finetune.monitor_metric",
    ).lower()
    metric = as_str(
        raw_optuna_cfg.get("metric", fallback_metric),
        "training_config.finetune.optuna.metric",
    ).lower()

    n_trials = (
        n_trials_override
        if n_trials_override is not None
        else as_int(
            raw_optuna_cfg.get("n_trials", 20),
            "training_config.finetune.optuna.n_trials",
        )
    )
    if n_trials <= 0:
        raise ValueError("training_config.finetune.optuna.n_trials must be > 0")
    timeout_sec: int | None
    if timeout_sec_override is not None:
        timeout_sec = timeout_sec_override
    else:
        timeout_raw = raw_optuna_cfg.get("timeout_sec")
        timeout_sec = (
            None
            if timeout_raw is None
            else as_int(timeout_raw, "training_config.finetune.optuna.timeout_sec")
        )
    if timeout_sec is not None and timeout_sec <= 0:
        raise ValueError("training_config.finetune.optuna.timeout_sec must be > 0")

    raw_sampler_cfg = _as_config_dict(
        raw_optuna_cfg.get("sampler", {}),
        "training_config.finetune.optuna.sampler",
    )
    sampler_type = as_str(
        raw_sampler_cfg.get("type", "tpe"),
        "training_config.finetune.optuna.sampler.type",
    ).lower()
    sampler_seed = as_int(
        raw_sampler_cfg.get("seed", run_seed),
        "training_config.finetune.optuna.sampler.seed",
    )

    raw_pruner_cfg = _as_config_dict(
        raw_optuna_cfg.get("pruner", {}),
        "training_config.finetune.optuna.pruner",
    )
    pruner_type = as_str(
        raw_pruner_cfg.get("type", "median"),
        "training_config.finetune.optuna.pruner.type",
    ).lower()
    pruner_n_startup_trials = as_int(
        raw_pruner_cfg.get("n_startup_trials", 5),
        "training_config.finetune.optuna.pruner.n_startup_trials",
    )
    pruner_n_warmup_steps = as_int(
        raw_pruner_cfg.get("n_warmup_steps", 1),
        "training_config.finetune.optuna.pruner.n_warmup_steps",
    )
    pruner_interval_steps = as_int(
        raw_pruner_cfg.get("interval_steps", 1),
        "training_config.finetune.optuna.pruner.interval_steps",
    )
    if pruner_n_startup_trials < 0:
        raise ValueError("n_startup_trials must be >= 0")
    if pruner_n_warmup_steps < 0:
        raise ValueError("n_warmup_steps must be >= 0")
    if pruner_interval_steps <= 0:
        raise ValueError("interval_steps must be > 0")

    raw_parameters = _as_config_dict(
        raw_optuna_cfg.get("parameters", {}),
        "training_config.finetune.optuna.parameters",
    )
    parameter_specs: list[ParameterSpec] = []
    for parameter_name, parameter_value in raw_parameters.items():
        if not isinstance(parameter_name, str) or not parameter_name:
            raise ValueError("optuna.parameters keys must be non-empty strings")
        parameter_cfg = _as_config_dict(
            parameter_value,
            f"training_config.finetune.optuna.parameters.{parameter_name}",
        )
        raw_distribution = _as_config_dict(
            parameter_cfg.get("distribution"),
            (
                f"training_config.finetune.optuna.parameters.{parameter_name}"
                ".distribution"
            ),
        )
        distribution = _validate_distribution(
            distribution=raw_distribution,
            field_prefix=(
                f"training_config.finetune.optuna.parameters.{parameter_name}"
                ".distribution"
            ),
        )
        raw_apply_to = parameter_cfg.get("apply_to")
        if not isinstance(raw_apply_to, list) or not raw_apply_to:
            raise ValueError(
                f"training_config.finetune.optuna.parameters.{parameter_name}.apply_to "
                "must be a non-empty list"
            )
        apply_paths: list[str] = []
        for path_index, raw_path in enumerate(raw_apply_to):
            apply_path = as_str(
                raw_path,
                (
                    f"training_config.finetune.optuna.parameters.{parameter_name}"
                    f".apply_to[{path_index}]"
                ),
            )
            if not _path_exists(finetune_cfg, apply_path):
                raise ValueError(
                    f"Unknown finetune path '{apply_path}' for parameter "
                    f"'{parameter_name}'"
                )
            apply_paths.append(apply_path)
        parameter_specs.append(
            ParameterSpec(
                name=parameter_name,
                distribution=distribution,
                apply_to=tuple(apply_paths),
            )
        )

    if enabled and not parameter_specs:
        raise ValueError("Enabled optuna search requires at least one parameter")

    return OptunaConfig(
        enabled=enabled,
        study_name=study_name,
        storage=storage,
        direction=direction,
        metric=metric,
        n_trials=n_trials,
        timeout_sec=timeout_sec,
        sampler_type=sampler_type,
        sampler_seed=sampler_seed,
        pruner_type=pruner_type,
        pruner_n_startup_trials=pruner_n_startup_trials,
        pruner_n_warmup_steps=pruner_n_warmup_steps,
        pruner_interval_steps=pruner_interval_steps,
        parameters=tuple(parameter_specs),
    )


def _build_sampler(optuna_cfg: OptunaConfig) -> BaseSampler:
    if optuna_cfg.sampler_type != "tpe":
        raise ValueError("Only sampler.type='tpe' is supported")
    return TPESampler(seed=optuna_cfg.sampler_seed)


def _build_pruner(optuna_cfg: OptunaConfig) -> BasePruner:
    if optuna_cfg.pruner_type != "median":
        raise ValueError("Only pruner.type='median' is supported")
    return MedianPruner(
        n_startup_trials=optuna_cfg.pruner_n_startup_trials,
        n_warmup_steps=optuna_cfg.pruner_n_warmup_steps,
        interval_steps=optuna_cfg.pruner_interval_steps,
    )


def _suggest_from_distribution(trial: Trial, spec: ParameterSpec) -> object:
    distribution = spec.distribution
    distribution_type = cast(str, distribution["type"])
    if distribution_type == "float":
        kwargs: dict[str, object] = {
            "low": cast(float, distribution["low"]),
            "high": cast(float, distribution["high"]),
        }
        if "log" in distribution:
            kwargs["log"] = cast(bool, distribution["log"])
        if "step" in distribution:
            kwargs["step"] = cast(float, distribution["step"])
        return trial.suggest_float(spec.name, **kwargs)
    if distribution_type == "int":
        kwargs = {
            "low": cast(int, distribution["low"]),
            "high": cast(int, distribution["high"]),
        }
        if "log" in distribution:
            kwargs["log"] = cast(bool, distribution["log"])
        if "step" in distribution:
            kwargs["step"] = cast(int, distribution["step"])
        return trial.suggest_int(spec.name, **kwargs)
    if distribution_type == "categorical":
        return trial.suggest_categorical(
            spec.name,
            cast(list[ScalarChoice], distribution["choices"]),
        )
    raise ValueError(f"Unsupported distribution type: {distribution_type}")


def _apply_trial_parameters(
    trial: Trial,
    config: ConfigDict,
    parameter_specs: tuple[ParameterSpec, ...],
) -> dict[str, object]:
    training_cfg = get_section(config, "training_config")
    finetune_cfg = _as_config_dict(
        training_cfg.get("finetune"),
        "training_config.finetune",
    )
    sampled_params: dict[str, object] = {}
    for spec in parameter_specs:
        sampled_value = _suggest_from_distribution(trial=trial, spec=spec)
        sampled_params[spec.name] = sampled_value
        for apply_path in spec.apply_to:
            _set_nested_value(finetune_cfg, apply_path, sampled_value)
    return sampled_params


def _ensure_sqlite_storage_parent(storage: str) -> None:
    if not storage.startswith("sqlite:///"):
        return
    sqlite_path = Path(storage.removeprefix("sqlite:///"))
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)


def _study_output_dir(model_name: str, study_name: str) -> Path:
    output_dir = Path("logs") / model_name / "finetune_search" / study_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _write_trials_csv(study: optuna.study.Study, output_dir: Path) -> Path:
    trial_path = output_dir / "trials.csv"
    trials = list(study.trials)
    parameter_names = sorted(
        {param_name for trial in trials for param_name in trial.params.keys()}
    )
    user_attr_names = sorted(
        {attr_name for trial in trials for attr_name in trial.user_attrs.keys()}
    )
    headers = [
        "trial_number",
        "state",
        "value",
        *[f"param_{name}" for name in parameter_names],
        *[f"attr_{name}" for name in user_attr_names],
    ]
    with trial_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for trial in trials:
            row: dict[str, object] = {
                "trial_number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
            }
            for name in parameter_names:
                row[f"param_{name}"] = trial.params.get(name)
            for name in user_attr_names:
                row[f"attr_{name}"] = trial.user_attrs.get(name)
            writer.writerow(row)
    return trial_path


def _write_best_params(study: optuna.study.Study, output_dir: Path) -> Path:
    best_path = output_dir / "best_params.yaml"
    payload: dict[str, object] = {
        "study_name": study.study_name,
        "direction": study.direction.name.lower(),
    }
    try:
        payload["best_trial_number"] = study.best_trial.number
        payload["best_value"] = study.best_value
        payload["best_params"] = dict(study.best_params)
        payload["best_user_attrs"] = dict(study.best_trial.user_attrs)
    except ValueError:
        payload["best_trial_number"] = None
        payload["best_value"] = None
        payload["best_params"] = {}
        payload["best_user_attrs"] = {}
    with best_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True)
    return best_path


def _resolve_checkpoint_path(
    config: ConfigDict, checkpoint_override: str | None
) -> Path:
    if checkpoint_override is not None:
        return Path(checkpoint_override)
    run_cfg = get_section(config, "run_config")
    raw_path = run_cfg.get("load_checkpoint_path")
    if isinstance(raw_path, str) and raw_path:
        return Path(raw_path)
    raise ValueError(
        "A fixed checkpoint is required. Set run_config.load_checkpoint_path "
        "or pass --checkpoint."
    )


def run_finetune_optuna_search(
    config: ConfigDict,
    checkpoint_override: str | None = None,
    study_name_override: str | None = None,
    n_trials_override: int | None = None,
    timeout_sec_override: int | None = None,
) -> optuna.study.Study:
    """Run Optuna search for finetune stage and persist study summaries."""
    model_name, _ = extract_model_kwargs(config)
    optuna_cfg = parse_optuna_config(
        config=config,
        study_name_override=study_name_override,
        n_trials_override=n_trials_override,
        timeout_sec_override=timeout_sec_override,
    )
    if not optuna_cfg.enabled:
        raise ValueError("training_config.finetune.optuna.enabled must be true")

    checkpoint_path = _resolve_checkpoint_path(
        config=config, checkpoint_override=checkpoint_override
    )
    _ensure_sqlite_storage_parent(optuna_cfg.storage)
    output_dir = _study_output_dir(
        model_name=model_name, study_name=optuna_cfg.study_name
    )
    sampler = _build_sampler(optuna_cfg)
    pruner = _build_pruner(optuna_cfg)

    study = optuna.create_study(
        study_name=optuna_cfg.study_name,
        storage=optuna_cfg.storage,
        direction=optuna_cfg.direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    def _objective(trial: Trial) -> float:
        trial_run_id = f"{optuna_cfg.study_name}_trial_{trial.number:04d}"
        trial_config = copy.deepcopy(config)
        run_cfg = get_section(trial_config, "run_config")
        run_cfg["stages"] = ["finetune"]
        run_cfg["finetune_run_id"] = trial_run_id
        run_cfg["load_checkpoint_path"] = str(checkpoint_path)

        training_cfg = get_section(trial_config, "training_config")
        finetune_cfg = _as_config_dict(
            training_cfg.get("finetune"),
            "training_config.finetune",
        )
        finetune_cfg["monitor_metric"] = optuna_cfg.metric
        sampled_params = _apply_trial_parameters(
            trial=trial,
            config=trial_config,
            parameter_specs=optuna_cfg.parameters,
        )
        best_metric: float | None = None
        prune_requested = False

        def _epoch_callback(
            stage: run_module.TrainingStage, epoch: int, metric: float
        ) -> bool:
            nonlocal best_metric, prune_requested
            if stage != "finetune":
                return False
            trial.report(metric, step=epoch + 1)
            best_metric = metric if best_metric is None else max(best_metric, metric)
            if trial.should_prune():
                prune_requested = True
                return True
            return False

        run_module.execute_pipeline(
            config=trial_config,
            training_epoch_callback=_epoch_callback,
        )
        if best_metric is None:
            raise ValueError("No finetune monitor metric reported during trial")

        checkpoint_output = (
            Path("models") / model_name / "finetune" / trial_run_id / "best_model.pth"
        )
        trial.set_user_attr("checkpoint_path", str(checkpoint_output))
        trial.set_user_attr("run_id", trial_run_id)
        trial.set_user_attr("metric", best_metric)
        trial.set_user_attr("sampled_params", sampled_params)
        if prune_requested:
            raise optuna.TrialPruned(f"Trial {trial.number} pruned")
        return best_metric

    study.optimize(
        _objective,
        n_trials=optuna_cfg.n_trials,
        timeout=optuna_cfg.timeout_sec,
        n_jobs=1,
    )
    trials_path = _write_trials_csv(study=study, output_dir=output_dir)
    best_path = _write_best_params(study=study, output_dir=output_dir)
    LOGGER.info("Study complete: %s", study.study_name)
    LOGGER.info("Best params saved to: %s", best_path)
    LOGGER.info("Trials saved to: %s", trials_path)
    return study


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    config = load_config(args.config)
    run_finetune_optuna_search(
        config=config,
        checkpoint_override=args.checkpoint,
        study_name_override=args.study_name,
        n_trials_override=args.n_trials,
        timeout_sec_override=args.timeout_sec,
    )


if __name__ == "__main__":
    main()
