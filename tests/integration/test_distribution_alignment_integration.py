"""
Integration-style tests for Distribution Alignment wiring in finetune and eval stages.
"""

from pathlib import Path
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn


if "torchmetrics" not in sys.modules:
    torchmetrics_module = types.ModuleType("torchmetrics")
    classification_module = types.ModuleType("torchmetrics.classification")

    class _MetricStub:
        def __init__(self, *args, **kwargs):
            self._value = torch.tensor(0.5)

        def to(self, device):
            return self

        def reset(self):
            return None

        def update(self, preds, targets):
            return None

        def compute(self):
            return self._value

    for name in [
        "BinaryAccuracy",
        "BinaryAUROC",
        "BinaryF1Score",
        "BinaryMatthewsCorrCoef",
        "BinaryPrecision",
        "BinaryRecall",
        "BinarySpecificity",
        "BinaryAveragePrecision",
    ]:
        setattr(classification_module, name, _MetricStub)

    torchmetrics_module.classification = classification_module
    sys.modules["torchmetrics"] = torchmetrics_module
    sys.modules["torchmetrics.classification"] = classification_module


if "src.train.base" not in sys.modules:
    train_base_stub = types.ModuleType("src.train.base")

    class Trainer:  # pragma: no cover - compatibility shim
        pass

    train_base_stub.Trainer = Trainer
    sys.modules["src.train.base"] = train_base_stub
else:
    train_base_module = sys.modules["src.train.base"]
    if not hasattr(train_base_module, "Trainer"):

        class Trainer:  # pragma: no cover - compatibility shim
            pass

        train_base_module.Trainer = Trainer

import src.utils.data_io as data_io_module

if not hasattr(data_io_module, "build_loader"):

    def build_loader(**kwargs):  # pragma: no cover - placeholder for tests
        raise RuntimeError("build_loader not implemented in data_io")

    data_io_module.build_loader = build_loader


def _load_stage_module(name: str, relative_path: str):
    root = Path(__file__).resolve().parents[2]
    module_path = root / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if "." in name:
        parent_name, attr = name.rsplit(".", 1)
        parent_module = sys.modules.setdefault(
            parent_name, types.ModuleType(parent_name)
        )
        setattr(parent_module, attr, module)
    return module


run_finetune = _load_stage_module(
    "tests.stage_finetune", "src/stages/finetune.py"
).run_finetune
run_evaluation = _load_stage_module(
    "tests.stage_evaluate", "src/stages/evaluate.py"
).run_evaluation


class DummyLoader:
    """Simple iterable with deterministic length."""

    def __len__(self):
        return 1

    def __iter__(self):
        yield {"dummy": torch.tensor(0.0)}


class DummyTrainer:
    """Minimal Trainer stub used for finetune integration test."""

    def __init__(self, model, device, optimizer_cfg, **kwargs):
        self.model = model
        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=optimizer_cfg.get("lr", 0.01)
        )

    def train_one_epoch(self, loader):
        return {"loss": 0.1, "lr": self.optimizer.param_groups[0]["lr"]}


def test_finetune_stage_saves_da_params(tmp_path, monkeypatch):
    """Finetune runner should store DA bias/threshold in checkpoint metadata."""
    monkeypatch.setattr("tests.stage_finetune.Trainer", DummyTrainer)

    da_return = {
        "bias": 0.25,
        "threshold": 0.6,
        "loss": 0.4,
        "predicted_prior": 0.2,
        "calibrated_prior": 0.5,
        "metrics": {"f1": 0.7, "auroc": 0.8, "recall": 0.6},
    }
    call_counter = {"count": 0}

    def fake_calibrate(self, model, loader, device, amp_dtype=None):
        call_counter["count"] += 1
        return da_return

    monkeypatch.setattr(
        "src.finetune.distribution_alignment.DistributionAligner.calibrate_and_search",
        fake_calibrate,
    )

    model = nn.Linear(1, 1)
    pretrain_ckpt = tmp_path / "pretrain.pth"
    torch.save({"epoch": 0, "state_dict": model.state_dict()}, pretrain_ckpt)

    cfg = {
        "run_config": {"save_best_only": True},
        "data_config": {
            "embedding_dtype": "fp32",
            "finetune": {"distribution_alignment": {"target_prior": 0.5}},
        },
        "finetune_config": {
            "strategy": {"type": "staged_unfreeze", "schedule": []},
            "epochs": 1,
            "batch_size": 2,
            "logging_metrics": {"primary": "auroc", "secondary": "recall"},
            "use_mixed_precision": False,
            "early_stopping_patience": 1,
            "monitor_metric": "f1",
            "optimizer": {"type": "sgd", "lr": 0.01},
            "loss": {"type": "bce_with_logits"},
        },
    }

    log_dir = tmp_path / "logs"
    ckpt_dir = tmp_path / "ckpts"
    log_dir.mkdir()
    ckpt_dir.mkdir()

    run_finetune(
        cfg=cfg,
        model=model,
        train_loader=DummyLoader(),
        val_loader=DummyLoader(),
        device=torch.device("cpu"),
        finetune_run_id="test",
        log_dir=log_dir,
        checkpoint_dir=ckpt_dir,
        load_checkpoint_path=str(pretrain_ckpt),
    )

    assert call_counter["count"] == 1
    best_ckpt = ckpt_dir / "best_model.pth"
    assert best_ckpt.exists()
    payload = torch.load(best_ckpt)
    assert payload["extra"]["da_bias"] == pytest.approx(da_return["bias"])
    assert payload["extra"]["da_threshold"] == pytest.approx(da_return["threshold"])
    csv_contents = (log_dir / "training_step.csv").read_text()
    assert "DA Bias" in csv_contents
    assert "DA Threshold" in csv_contents


def test_evaluation_applies_loaded_da_params(tmp_path, monkeypatch):
    """Evaluation runner should pass DA bias/threshold to Evaluator."""
    model = nn.Linear(1, 1)
    ckpt_path = tmp_path / "finetune_best.pth"
    torch.save(
        {
            "epoch": 0,
            "state_dict": model.state_dict(),
            "extra": {"da_bias": 0.3, "da_threshold": 0.55},
        },
        ckpt_path,
    )

    cfg = {
        "data_config": {
            "embeddings_path": "unused",
            "max_sequence_length": 32,
            "embedding_dtype": "fp32",
            "evaluate": {
                "test_balanced": "balanced.csv",
                "test_realistic": "realistic.csv",
            },
        },
        "evaluate": {"metrics": ["auroc"]},
    }

    loaders = [DummyLoader(), DummyLoader()]

    def fake_build_loader(**kwargs):
        return loaders.pop(0)

    monkeypatch.setattr("tests.stage_evaluate.build_loader", fake_build_loader)

    eval_calls = []

    def fake_evaluate(
        self, model, loader, device, logit_bias=0.0, threshold_override=None
    ):
        eval_calls.append((logit_bias, threshold_override))
        return {"auroc": 0.5}

    monkeypatch.setattr("tests.stage_evaluate.Evaluator.evaluate", fake_evaluate)

    log_dir = tmp_path / "eval_logs"
    log_dir.mkdir()

    run_evaluation(
        cfg=cfg,
        model=model,
        device=torch.device("cpu"),
        eval_run_id="eval",
        log_dir=log_dir,
        load_checkpoint_path=str(ckpt_path),
    )

    assert eval_calls == [(0.3, 0.55), (0.3, 0.55)]
