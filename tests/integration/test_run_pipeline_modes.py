"""Integration tests for pipeline modes using fixtures and mocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
import src.run as run_module
import torch
from src.embed import EmbeddingCacheManifest
from src.utils.config import ConfigDict
from src.utils.distributed import DistributedContext
from torch import nn
from torch.utils.data import DataLoader, Dataset


class _EmptyDataset(Dataset[dict[str, torch.Tensor]]):
    """Empty dataset used for mocked dataloader wiring."""

    def __len__(self) -> int:
        """Return dataset length."""
        return 0

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Raise because no sample retrieval is expected."""
        raise IndexError(index)


class _DummyModel(nn.Module):
    """Simple model that satisfies ``nn.Module`` contract for orchestration tests."""

    def forward(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return fixed output dictionary."""
        del kwargs
        return {"logits": torch.zeros((1, 1), dtype=torch.float32)}


@dataclass
class PipelineCalls:
    """Recorded mocked pipeline calls."""

    training: list[tuple[str, str]] = field(default_factory=list)
    evaluation: list[tuple[Path, str]] = field(default_factory=list)
    checkpoint_loads: list[Path] = field(default_factory=list)


@pytest.fixture
def base_config() -> ConfigDict:
    """Build minimal valid config for execute_pipeline orchestration."""
    return {
        "run_config": {
            "mode": "full_pipeline",
            "seed": 7,
            "train_run_id": "train_run",
            "finetune_run_id": "finetune_run",
            "eval_run_id": "eval_run",
            "load_checkpoint_path": "artifacts/input_checkpoint.pth",
            "save_best_only": True,
        },
        "device_config": {
            "device": "cpu",
            "ddp_enabled": False,
            "use_mixed_precision": False,
        },
        "data_config": {},
        "model_config": {"model": "v3"},
        "training_config": {},
        "evaluate": {"metrics": ["accuracy"]},
    }


@pytest.fixture
def patched_pipeline(monkeypatch: pytest.MonkeyPatch) -> PipelineCalls:
    """Patch side-effectful pipeline dependencies and capture call sequence."""
    calls = PipelineCalls()

    def fake_build_dataloaders(
        config: ConfigDict,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        train_stage: str = "pretrain",
        embedding_cache: object | None = None,
        embedding_split_paths: object | None = None,
    ) -> dict[str, DataLoader[dict[str, torch.Tensor]]]:
        del (
            config,
            distributed,
            rank,
            world_size,
            train_stage,
            embedding_cache,
            embedding_split_paths,
        )
        loader = DataLoader(_EmptyDataset(), batch_size=1)
        return {"train": loader, "valid": loader, "test": loader}

    def fake_build_model(config: ConfigDict) -> nn.Module:
        del config
        return _DummyModel()

    def fake_run_training_stage(
        stage: str,
        config: ConfigDict,
        model: nn.Module,
        device: torch.device,
        dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
        run_id: str,
        distributed_context: DistributedContext,
    ) -> Path:
        del config, model, device, dataloaders, distributed_context
        calls.training.append((stage, run_id))
        return Path(f"artifacts/{stage}_best_model.pth")

    def fake_collect_embedding_split_paths(config: ConfigDict) -> list[Path]:
        del config
        return [Path("train.txt"), Path("valid.txt"), Path("test.txt")]

    def fake_ensure_embedding_cache(
        config: ConfigDict,
        split_paths: list[Path],
        input_dim: int,
        max_sequence_length: int,
        distributed: bool = False,
        rank: int = 0,
    ) -> EmbeddingCacheManifest:
        del config, split_paths, input_dim, max_sequence_length, distributed, rank
        return EmbeddingCacheManifest(
            cache_dir=Path("cache"),
            index={},
            required_ids=frozenset(),
        )

    def fake_load_checkpoint(
        model: nn.Module, checkpoint_path: Path, device: torch.device
    ) -> None:
        del model, device
        calls.checkpoint_loads.append(checkpoint_path)

    def fake_run_evaluation_stage(
        config: ConfigDict,
        model: nn.Module,
        device: torch.device,
        dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
        run_id: str,
        checkpoint_path: Path,
        distributed_context: DistributedContext,
    ) -> dict[str, float]:
        del config, model, device, dataloaders, distributed_context
        calls.evaluation.append((checkpoint_path, run_id))
        return {"accuracy": 1.0}

    def fake_initialize_distributed(ddp_enabled: bool) -> DistributedContext:
        del ddp_enabled
        return DistributedContext(ddp_enabled=False, is_distributed=False)

    def fake_cleanup_distributed(context: DistributedContext) -> None:
        del context

    def fake_resolve_device(device_name: str) -> torch.device:
        del device_name
        return torch.device("cpu")

    monkeypatch.setattr(run_module, "build_dataloaders", fake_build_dataloaders)
    monkeypatch.setattr(run_module, "build_model", fake_build_model)
    monkeypatch.setattr(run_module, "run_training_stage", fake_run_training_stage)
    monkeypatch.setattr(run_module, "run_evaluation_stage", fake_run_evaluation_stage)
    monkeypatch.setattr(
        run_module, "initialize_distributed", fake_initialize_distributed
    )
    monkeypatch.setattr(run_module, "cleanup_distributed", fake_cleanup_distributed)
    monkeypatch.setattr(run_module, "resolve_device", fake_resolve_device)
    monkeypatch.setattr(
        run_module, "collect_embedding_split_paths", fake_collect_embedding_split_paths
    )
    monkeypatch.setattr(
        run_module, "ensure_embedding_cache", fake_ensure_embedding_cache
    )
    monkeypatch.setattr(run_module, "_load_checkpoint", fake_load_checkpoint)
    return calls


def test_execute_pipeline_full_pipeline(
    base_config: ConfigDict,
    patched_pipeline: PipelineCalls,
) -> None:
    run_module.execute_pipeline(base_config)

    assert patched_pipeline.training == [
        ("pretrain", "train_run"),
        ("finetune", "finetune_run"),
    ]
    assert patched_pipeline.evaluation == [
        (Path("artifacts/finetune_best_model.pth"), "eval_run"),
    ]
    assert patched_pipeline.checkpoint_loads == [
        Path("artifacts/pretrain_best_model.pth")
    ]


def test_execute_pipeline_train_only(
    base_config: ConfigDict,
    patched_pipeline: PipelineCalls,
) -> None:
    run_cfg = base_config["run_config"]
    assert isinstance(run_cfg, dict)
    run_cfg["mode"] = "train_only"

    run_module.execute_pipeline(base_config)

    assert patched_pipeline.training == [("pretrain", "train_run")]
    assert patched_pipeline.evaluation == []


def test_execute_pipeline_eval_only(
    base_config: ConfigDict,
    patched_pipeline: PipelineCalls,
) -> None:
    run_cfg = base_config["run_config"]
    assert isinstance(run_cfg, dict)
    run_cfg["mode"] = "eval_only"
    run_cfg["load_checkpoint_path"] = "artifacts/eval_input_model.pth"

    run_module.execute_pipeline(base_config)

    assert patched_pipeline.training == []
    assert patched_pipeline.evaluation == [
        (Path("artifacts/eval_input_model.pth"), "eval_run")
    ]


def test_execute_pipeline_custom_stages_finetune_then_evaluate(
    base_config: ConfigDict,
    patched_pipeline: PipelineCalls,
) -> None:
    run_cfg = base_config["run_config"]
    assert isinstance(run_cfg, dict)
    run_cfg["stages"] = ["finetune", "evaluate"]
    run_cfg["load_checkpoint_path"] = "artifacts/seed_model.pth"

    run_module.execute_pipeline(base_config)

    assert patched_pipeline.training == [("finetune", "finetune_run")]
    assert patched_pipeline.evaluation == [
        (Path("artifacts/finetune_best_model.pth"), "eval_run")
    ]
    assert patched_pipeline.checkpoint_loads == [Path("artifacts/seed_model.pth")]


def test_execute_pipeline_finetune_without_checkpoint_raises(
    base_config: ConfigDict,
    patched_pipeline: PipelineCalls,
) -> None:
    del patched_pipeline
    run_cfg = base_config["run_config"]
    assert isinstance(run_cfg, dict)
    run_cfg["stages"] = ["finetune"]
    run_cfg["load_checkpoint_path"] = None

    with pytest.raises(
        ValueError, match="load_checkpoint_path is required when finetune"
    ):
        run_module.execute_pipeline(base_config)


@pytest.mark.parametrize("deprecated_mode", ["pretrain_only", "finetune_from_pretrain"])
def test_execute_pipeline_removed_modes_raise(
    base_config: ConfigDict,
    patched_pipeline: PipelineCalls,
    deprecated_mode: str,
) -> None:
    run_cfg = base_config["run_config"]
    assert isinstance(run_cfg, dict)
    run_cfg["mode"] = deprecated_mode
    with pytest.raises(ValueError, match="Unsupported run mode"):
        run_module.execute_pipeline(base_config)


def test_execute_pipeline_staged_unfreeze_enables_ddp_find_unused(
    monkeypatch: pytest.MonkeyPatch,
    base_config: ConfigDict,
) -> None:
    run_cfg = base_config["run_config"]
    assert isinstance(run_cfg, dict)
    run_cfg["mode"] = "train_only"

    device_cfg = base_config["device_config"]
    assert isinstance(device_cfg, dict)
    device_cfg["ddp_enabled"] = True
    device_cfg["device"] = "cpu"

    training_cfg = base_config["training_config"]
    assert isinstance(training_cfg, dict)
    training_cfg["strategy"] = {
        "type": "staged_unfreeze",
        "unfreeze_epoch": 1,
        "initial_trainable_prefixes": ["output_head"],
    }

    ddp_call: dict[str, object] = {}

    class _FakeDDP(nn.Module):
        def __init__(
            self,
            module: nn.Module,
            device_ids: list[int] | None = None,
            find_unused_parameters: bool = False,
        ) -> None:
            super().__init__()
            self.module = module
            ddp_call["device_ids"] = device_ids
            ddp_call["find_unused_parameters"] = find_unused_parameters

        def forward(self, *args: object, **kwargs: object) -> object:
            return self.module(*args, **kwargs)

    def fake_build_dataloaders(
        config: ConfigDict,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        train_stage: str = "pretrain",
        embedding_cache: object | None = None,
        embedding_split_paths: object | None = None,
    ) -> dict[str, DataLoader[dict[str, torch.Tensor]]]:
        del (
            config,
            distributed,
            rank,
            world_size,
            train_stage,
            embedding_cache,
            embedding_split_paths,
        )
        loader = DataLoader(_EmptyDataset(), batch_size=1)
        return {"train": loader, "valid": loader, "test": loader}

    def fake_build_model(config: ConfigDict) -> nn.Module:
        del config
        return _DummyModel()

    def fake_initialize_distributed(ddp_enabled: bool) -> DistributedContext:
        del ddp_enabled
        return DistributedContext(
            ddp_enabled=True,
            is_distributed=True,
            rank=0,
            local_rank=0,
            world_size=2,
        )

    def fake_cleanup_distributed(context: DistributedContext) -> None:
        del context

    def fake_resolve_device(device_name: str) -> torch.device:
        del device_name
        return torch.device("cpu")

    def fake_run_training_stage(
        stage: str,
        config: ConfigDict,
        model: nn.Module,
        device: torch.device,
        dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
        run_id: str,
        distributed_context: DistributedContext,
    ) -> Path:
        del stage, config, model, device, dataloaders, run_id, distributed_context
        return Path("artifacts/train_best_model.pth")

    def fake_collect_embedding_split_paths(config: ConfigDict) -> list[Path]:
        del config
        return [Path("train.txt"), Path("valid.txt"), Path("test.txt")]

    def fake_ensure_embedding_cache(
        config: ConfigDict,
        split_paths: list[Path],
        input_dim: int,
        max_sequence_length: int,
        distributed: bool = False,
        rank: int = 0,
    ) -> EmbeddingCacheManifest:
        del config, split_paths, input_dim, max_sequence_length, distributed, rank
        return EmbeddingCacheManifest(
            cache_dir=Path("cache"),
            index={},
            required_ids=frozenset(),
        )

    monkeypatch.setattr(run_module, "build_dataloaders", fake_build_dataloaders)
    monkeypatch.setattr(run_module, "build_model", fake_build_model)
    monkeypatch.setattr(
        run_module, "initialize_distributed", fake_initialize_distributed
    )
    monkeypatch.setattr(run_module, "cleanup_distributed", fake_cleanup_distributed)
    monkeypatch.setattr(run_module, "resolve_device", fake_resolve_device)
    monkeypatch.setattr(run_module, "run_training_stage", fake_run_training_stage)
    monkeypatch.setattr(
        run_module, "collect_embedding_split_paths", fake_collect_embedding_split_paths
    )
    monkeypatch.setattr(
        run_module, "ensure_embedding_cache", fake_ensure_embedding_cache
    )
    monkeypatch.setattr(run_module, "DistributedDataParallel", _FakeDDP)

    run_module.execute_pipeline(base_config)

    assert ddp_call["device_ids"] is None
    assert ddp_call["find_unused_parameters"] is True


def test_execute_pipeline_v6_uses_sequence_dataloaders(
    monkeypatch: pytest.MonkeyPatch,
    base_config: ConfigDict,
) -> None:
    run_cfg = base_config["run_config"]
    assert isinstance(run_cfg, dict)
    run_cfg["stages"] = ["finetune", "evaluate"]
    run_cfg["load_checkpoint_path"] = "artifacts/v6_seed_model.pth"

    model_cfg = base_config["model_config"]
    assert isinstance(model_cfg, dict)
    model_cfg["model"] = "v6"

    calls = {"v6_dataloader_calls": 0}

    def fake_build_model(config: ConfigDict) -> nn.Module:
        del config
        return _DummyModel()

    def fake_build_dataloaders_v6(
        config: ConfigDict,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        train_stage: str = "pretrain",
    ) -> dict[str, DataLoader[dict[str, object]]]:
        del config, distributed, rank, world_size, train_stage
        calls["v6_dataloader_calls"] += 1
        loader = DataLoader(_EmptyDataset(), batch_size=1)
        return {"train": loader, "valid": loader, "test": loader}

    def _unexpected_embedding_call(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("embedding-cache path should not run for v6")

    def fake_initialize_distributed(ddp_enabled: bool) -> DistributedContext:
        del ddp_enabled
        return DistributedContext(ddp_enabled=False, is_distributed=False)

    def fake_cleanup_distributed(context: DistributedContext) -> None:
        del context

    def fake_resolve_device(device_name: str) -> torch.device:
        del device_name
        return torch.device("cpu")

    def fake_run_training_stage(
        stage: str,
        config: ConfigDict,
        model: nn.Module,
        device: torch.device,
        dataloaders: dict[str, DataLoader[dict[str, object]]],
        run_id: str,
        distributed_context: DistributedContext,
    ) -> Path:
        del stage, config, model, device, dataloaders, run_id, distributed_context
        return Path("artifacts/v6_finetune_best_model.pth")

    def fake_load_checkpoint(
        model: nn.Module, checkpoint_path: Path, device: torch.device
    ) -> None:
        del model, checkpoint_path, device

    def fake_run_evaluation_stage(
        config: ConfigDict,
        model: nn.Module,
        device: torch.device,
        dataloaders: dict[str, DataLoader[dict[str, object]]],
        run_id: str,
        checkpoint_path: Path,
        distributed_context: DistributedContext,
    ) -> dict[str, float]:
        del (
            config,
            model,
            device,
            dataloaders,
            run_id,
            checkpoint_path,
            distributed_context,
        )
        return {"accuracy": 1.0}

    monkeypatch.setattr(run_module, "build_model", fake_build_model)
    monkeypatch.setattr(run_module, "build_dataloaders_v6", fake_build_dataloaders_v6)
    monkeypatch.setattr(
        run_module, "collect_embedding_split_paths", _unexpected_embedding_call
    )
    monkeypatch.setattr(
        run_module, "ensure_embedding_cache", _unexpected_embedding_call
    )
    monkeypatch.setattr(run_module, "build_dataloaders", _unexpected_embedding_call)
    monkeypatch.setattr(
        run_module, "initialize_distributed", fake_initialize_distributed
    )
    monkeypatch.setattr(run_module, "cleanup_distributed", fake_cleanup_distributed)
    monkeypatch.setattr(run_module, "resolve_device", fake_resolve_device)
    monkeypatch.setattr(run_module, "run_training_stage", fake_run_training_stage)
    monkeypatch.setattr(run_module, "_load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(run_module, "run_evaluation_stage", fake_run_evaluation_stage)

    run_module.execute_pipeline(base_config)

    assert calls["v6_dataloader_calls"] == 2
