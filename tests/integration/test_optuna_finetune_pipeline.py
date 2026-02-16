"""Integration tests for Optuna finetune search orchestration."""

from __future__ import annotations

from pathlib import Path

import optuna
import pytest
from src.tune import finetune_optuna
from src.utils.config import ConfigDict


def _base_config(study_name: str, n_trials: int) -> ConfigDict:
    return {
        "run_config": {
            "seed": 47,
            "load_checkpoint_path": "artifacts/pretrain_seed.pth",
            "save_best_only": True,
        },
        "model_config": {"model": "v3"},
        "training_config": {
            "monitor_metric": "auprc",
            "finetune": {
                "batch_size": 16,
                "monitor_metric": "auprc",
                "optimizer": {"lr": 1.0e-4, "weight_decay": 0.01},
                "scheduler": {"max_lr": 1.0e-4, "pct_start": 0.2},
                "loss": {"pos_weight": 7.0, "label_smoothing": 0.01},
                "strategy": {
                    "type": "staged_unfreeze",
                    "unfreeze_epoch": 3,
                    "initial_trainable_prefixes": ["output_head"],
                },
                "optuna": {
                    "enabled": True,
                    "study_name": study_name,
                    "storage": "sqlite:///logs/optuna/v3_finetune.db",
                    "direction": "maximize",
                    "metric": "auprc",
                    "n_trials": n_trials,
                    "timeout_sec": None,
                    "sampler": {"type": "tpe", "seed": 47},
                    "pruner": {
                        "type": "median",
                        "n_startup_trials": 0,
                        "n_warmup_steps": 0,
                        "interval_steps": 1,
                    },
                    "parameters": {
                        "lr": {
                            "distribution": {
                                "type": "float",
                                "low": 1.0e-5,
                                "high": 1.0e-3,
                                "log": True,
                            },
                            "apply_to": ["optimizer.lr", "scheduler.max_lr"],
                        }
                    },
                },
            },
        },
        "device_config": {"device": "cpu", "ddp_enabled": False},
        "data_config": {"dataloader": {"sampling": {"strategy": "none"}}},
        "evaluate": {"metrics": ["auprc"]},
    }


def test_optuna_objective_runs_finetune_only_and_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _base_config(study_name="integration_case", n_trials=1)
    calls: list[tuple[object, object]] = []

    def fake_execute_pipeline(
        config: ConfigDict,
        training_epoch_callback: finetune_optuna.run_module.TrainingEpochCallback
        | None = None,
    ) -> None:
        run_cfg = config["run_config"]
        assert isinstance(run_cfg, dict)
        calls.append((run_cfg.get("stages"), run_cfg.get("load_checkpoint_path")))
        assert run_cfg.get("stages") == ["finetune"]
        run_id = run_cfg.get("finetune_run_id")
        assert isinstance(run_id, str)
        checkpoint_path = Path("models") / "v3" / "finetune" / run_id / "best_model.pth"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.touch()
        if training_epoch_callback is None:
            return
        should_stop = training_epoch_callback("finetune", 0, 0.35)
        if not should_stop:
            training_epoch_callback("finetune", 1, 0.55)

    monkeypatch.setattr(
        finetune_optuna.run_module,
        "execute_pipeline",
        fake_execute_pipeline,
    )

    study = finetune_optuna.run_finetune_optuna_search(config=config)

    assert len(calls) == 1
    assert len(study.trials) == 1
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[0].value == pytest.approx(0.55)
    output_dir = Path("logs") / "v3" / "finetune_search" / "integration_case"
    assert (output_dir / "best_params.yaml").exists()
    assert (output_dir / "trials.csv").exists()


def test_optuna_pruned_trial_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _base_config(study_name="prune_case", n_trials=2)

    def fake_execute_pipeline(
        config: ConfigDict,
        training_epoch_callback: finetune_optuna.run_module.TrainingEpochCallback
        | None = None,
    ) -> None:
        run_cfg = config["run_config"]
        assert isinstance(run_cfg, dict)
        run_id = run_cfg.get("finetune_run_id")
        assert isinstance(run_id, str)
        if training_epoch_callback is None:
            return
        if run_id.endswith("0000"):
            metrics = [0.95, 0.96]
        else:
            metrics = [0.05, 0.04]
        for epoch, metric in enumerate(metrics):
            if training_epoch_callback("finetune", epoch, metric):
                break

    monkeypatch.setattr(
        finetune_optuna.run_module,
        "execute_pipeline",
        fake_execute_pipeline,
    )

    study = finetune_optuna.run_finetune_optuna_search(config=config)

    assert len(study.trials) == 2
    states = {trial.state for trial in study.trials}
    assert optuna.trial.TrialState.COMPLETE in states
    assert optuna.trial.TrialState.PRUNED in states


def test_optuna_study_resume_with_sqlite_storage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _base_config(study_name="resume_case", n_trials=1)

    def fake_execute_pipeline(
        config: ConfigDict,
        training_epoch_callback: finetune_optuna.run_module.TrainingEpochCallback
        | None = None,
    ) -> None:
        del config
        if training_epoch_callback is not None:
            training_epoch_callback("finetune", 0, 0.4)

    monkeypatch.setattr(
        finetune_optuna.run_module,
        "execute_pipeline",
        fake_execute_pipeline,
    )

    first_study = finetune_optuna.run_finetune_optuna_search(config=config)
    second_study = finetune_optuna.run_finetune_optuna_search(config=config)

    assert first_study.study_name == "resume_case"
    assert second_study.study_name == "resume_case"
    assert len(second_study.trials) >= 2
    assert (Path("logs") / "optuna" / "v3_finetune.db").exists()
    output_dir = Path("logs") / "v3" / "finetune_search" / "resume_case"
    assert (output_dir / "best_params.yaml").exists()
    assert (output_dir / "trials.csv").exists()
