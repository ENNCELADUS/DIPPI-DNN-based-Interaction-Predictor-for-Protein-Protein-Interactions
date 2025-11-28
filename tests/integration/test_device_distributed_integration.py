"""
Integration test for device/distributed setup in run.py orchestration.

Tests the complete device/DDP initialization flow as called by run.py:
- Device selection from config
- DDP initialization with torchrun envs
- Model placement and DDP wrapping
- Cleanup
"""

import os
from unittest.mock import patch

import pytest
import torch.nn as nn

from src.utils.device import get_device
from src.utils.distributed import (
    init_if_enabled,
    is_main_process,
    get_rank,
    cleanup,
    barrier,
)


class DummyModel(nn.Module):
    """Minimal model for testing."""

    def __init__(self, input_dim: int = 10, output_dim: int = 2):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class TestDeviceDistributedOrchestration:
    """Test device/DDP setup as orchestrated by run.py."""

    def test_single_process_cpu_orchestration(self):
        """Test complete setup flow: single process, CPU device."""
        # Config (as would be parsed by run.py)
        device_cfg = {"strategy": "cpu"}
        ddp_cfg = {"enabled": False}

        # Step 1: Device selection
        device = get_device(device_cfg)
        assert device.type == "cpu"

        # Step 2: DDP initialization
        ddp_enabled = init_if_enabled(ddp_cfg, device)
        assert ddp_enabled is False
        assert is_main_process() is True  # Single process = main

        # Step 3: Model placement
        model = DummyModel()
        model.to(device)
        assert next(model.parameters()).device.type == "cpu"

        # Step 4: No DDP wrapping (disabled)
        # model remains unwrapped

        # Step 5: Cleanup (no-op when not initialized)
        barrier()  # Should not raise
        cleanup()  # Should not raise

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.set_device")
    def test_single_process_cuda_orchestration(
        self, mock_set_device, mock_cuda_available
    ):
        """Test complete setup flow: single process, CUDA device."""
        # Config
        device_cfg = {"strategy": "cuda", "gpu_id": 0}
        ddp_cfg = {"enabled": False}

        # Step 1: Device selection
        device = get_device(device_cfg)
        assert device.type == "cuda"
        assert device.index == 0
        mock_set_device.assert_called_once_with(0)

        # Step 2: DDP initialization
        ddp_enabled = init_if_enabled(ddp_cfg, device)
        assert ddp_enabled is False

        # Step 3: Model placement
        _model = DummyModel()
        # Note: In real scenario, model.to(device) would place on CUDA
        # In test, we just verify the flow
        # _model.to(device)  # Skip actual CUDA ops in test

        # Step 4: No DDP wrapping (disabled)
        # model remains unwrapped

        # Step 5: Cleanup (no-op)
        cleanup()

    @patch("torch.distributed.init_process_group")
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.set_device")
    @patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "2", "LOCAL_RANK": "0"})
    def test_ddp_enabled_cuda_orchestration(
        self, mock_set_device, mock_cuda_available, mock_init_pg
    ):
        """Test complete setup flow: DDP enabled, CUDA, torchrun envs."""
        # Config
        device_cfg = {"strategy": "cuda"}
        ddp_cfg = {"enabled": True, "backend": "nccl", "timeout_sec": 1800}

        # Step 1: Device selection (reads LOCAL_RANK=0 from env)
        device = get_device(device_cfg)
        assert device.type == "cuda"
        assert device.index == 0
        mock_set_device.assert_called_once_with(0)

        # Step 2: DDP initialization
        ddp_enabled = init_if_enabled(ddp_cfg, device)
        assert ddp_enabled is True
        mock_init_pg.assert_called_once()
        assert mock_init_pg.call_args.kwargs["backend"] == "nccl"

        # Step 3: Model placement
        _model = DummyModel()
        # _model.to(device)  # Skip actual CUDA ops in test

        # Step 4: DDP wrapping (mocked)
        # In real run.py:
        # if ddp_enabled:
        #     model = DDP(model, device_ids=[device.index], output_device=device.index)
        # We verify the logic without actual DDP wrapper (requires full torch.distributed setup)

        # Step 5: Cleanup
        with patch("torch.distributed.is_available", return_value=True):
            with patch("torch.distributed.is_initialized", return_value=True):
                with patch("torch.distributed.barrier") as mock_barrier:
                    with patch(
                        "torch.distributed.destroy_process_group"
                    ) as mock_destroy:
                        barrier()
                        cleanup()
                        mock_barrier.assert_called_once()
                        mock_destroy.assert_called_once()

    @patch("torch.distributed.init_process_group")
    @patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "2", "LOCAL_RANK": "0"})
    def test_ddp_enabled_cpu_orchestration(self, mock_init_pg):
        """Test complete setup flow: DDP enabled, CPU (gloo backend)."""
        # Config
        device_cfg = {"strategy": "cpu"}
        ddp_cfg = {"enabled": True}  # No explicit backend (auto-select gloo)

        # Step 1: Device selection
        device = get_device(device_cfg)
        assert device.type == "cpu"

        # Step 2: DDP initialization (auto-selects gloo for CPU)
        ddp_enabled = init_if_enabled(ddp_cfg, device)
        assert ddp_enabled is True
        mock_init_pg.assert_called_once()
        assert mock_init_pg.call_args.kwargs["backend"] == "gloo"

        # Step 3: Model placement
        model = DummyModel()
        model.to(device)

        # Step 4: DDP wrapping (mocked)
        # In real run.py for CPU:
        # if ddp_enabled:
        #     model = DDP(model)  # No device_ids for CPU

        # Step 5: Cleanup
        with patch("torch.distributed.is_available", return_value=True):
            with patch("torch.distributed.is_initialized", return_value=True):
                with patch("torch.distributed.destroy_process_group") as mock_destroy:
                    cleanup()
                    mock_destroy.assert_called_once()

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.set_device")
    def test_auto_device_selection_cuda_available(
        self, mock_set_device, mock_cuda_available
    ):
        """Test auto strategy selects CUDA when available."""
        # Config with auto strategy
        device_cfg = {"strategy": "auto"}
        ddp_cfg = {"enabled": False}

        # Device selection should pick CUDA
        device = get_device(device_cfg)
        assert device.type == "cuda"
        mock_set_device.assert_called_once()

        # Rest of orchestration
        ddp_enabled = init_if_enabled(ddp_cfg, device)
        assert ddp_enabled is False

        _model = DummyModel()
        # _model.to(device)  # Skip actual CUDA ops

        cleanup()

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_auto_device_selection_cpu_fallback(
        self, mock_mps_available, mock_cuda_available
    ):
        """Test auto strategy falls back to CPU when CUDA/MPS unavailable."""
        # Config with auto strategy
        device_cfg = {"strategy": "auto"}
        ddp_cfg = {"enabled": False}

        # Device selection should fall back to CPU
        device = get_device(device_cfg)
        assert device.type == "cpu"

        # Rest of orchestration
        ddp_enabled = init_if_enabled(ddp_cfg, device)
        assert ddp_enabled is False

        model = DummyModel()
        model.to(device)

        cleanup()

    def test_missing_ddp_envs_raises(self):
        """Test DDP enabled but torchrun envs missing raises clear error."""
        device_cfg = {"strategy": "cpu"}
        ddp_cfg = {"enabled": True}

        device = get_device(device_cfg)

        # DDP init should fail with clear error
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="RANK/WORLD_SIZE not set"):
                init_if_enabled(ddp_cfg, device)

    @patch("torch.distributed.init_process_group")
    @patch.dict(
        os.environ,
        {"RANK": "1", "WORLD_SIZE": "4", "LOCAL_RANK": "1"},
    )
    def test_non_main_process_rank(self, mock_init_pg):
        """Test rank detection for non-main process."""
        device_cfg = {"strategy": "cpu"}
        ddp_cfg = {"enabled": True}

        device = get_device(device_cfg)
        ddp_enabled = init_if_enabled(ddp_cfg, device)
        assert ddp_enabled is True

        # Mock distributed as initialized
        with patch("torch.distributed.is_available", return_value=True):
            with patch("torch.distributed.is_initialized", return_value=True):
                with patch("torch.distributed.get_rank", return_value=1):
                    assert is_main_process() is False
                    assert get_rank() == 1

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.set_device")
    @patch.dict(
        os.environ,
        {"RANK": "2", "WORLD_SIZE": "4", "LOCAL_RANK": "2"},
    )
    def test_multi_gpu_local_rank_binding(self, mock_set_device, mock_cuda_available):
        """Test LOCAL_RANK env correctly binds to specific GPU."""
        device_cfg = {"strategy": "cuda", "gpu_id": 0}  # gpu_id ignored

        # Device selection should use LOCAL_RANK=2
        device = get_device(device_cfg)
        assert device.type == "cuda"
        assert device.index == 2  # From LOCAL_RANK, not gpu_id
        mock_set_device.assert_called_once_with(2)


class TestRunPyIntegrationScenarios:
    """Test complete run.py orchestration scenarios."""

    def test_pretrain_only_mode_single_gpu(self):
        """Test device/DDP setup for pretrain_only mode, single GPU."""
        # Simulate run.py config parsing
        top_level_cfg = {
            "device": {"strategy": "cuda", "gpu_id": 0},
            "ddp": {"enabled": False},
        }

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.set_device"):
                # Device setup
                device = get_device(top_level_cfg["device"])
                assert device.type == "cuda"

                # DDP setup
                ddp_enabled = init_if_enabled(top_level_cfg["ddp"], device)
                assert ddp_enabled is False

                # Model setup (run.py pattern)
                _model = DummyModel()
                # _model.to(device)  # Skip actual CUDA

                # No DDP wrapping
                assert not isinstance(_model, nn.parallel.DistributedDataParallel)

                # Cleanup
                cleanup()

    @patch("torch.distributed.init_process_group")
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.set_device")
    @patch.dict(
        os.environ,
        {
            "RANK": "0",
            "WORLD_SIZE": "2",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12355",
        },
    )
    def test_full_pipeline_mode_multi_gpu(
        self, mock_set_device, mock_cuda_available, mock_init_pg
    ):
        """Test device/DDP setup for full_pipeline mode, multi-GPU."""
        # Simulate run.py config parsing
        top_level_cfg = {
            "device": {"strategy": "auto"},  # Should select CUDA
            "ddp": {"enabled": True, "backend": "nccl"},
        }

        # Device setup
        device = get_device(top_level_cfg["device"])
        assert device.type == "cuda"
        assert device.index == 0  # From LOCAL_RANK=0

        # DDP setup
        ddp_enabled = init_if_enabled(top_level_cfg["ddp"], device)
        assert ddp_enabled is True
        mock_init_pg.assert_called_once()

        # Model setup (run.py pattern)
        _model = DummyModel()
        # _model.to(device)  # Skip actual CUDA

        # DDP wrapping (mocked - full DDP requires complete torch.distributed setup)
        # In real run.py:
        # if ddp_enabled:
        #     from torch.nn.parallel import DistributedDataParallel as DDP
        #     model = DDP(model, device_ids=[device.index], output_device=device.index)

        # Cleanup
        with patch("torch.distributed.is_available", return_value=True):
            with patch("torch.distributed.is_initialized", return_value=True):
                with patch("torch.distributed.barrier") as mock_barrier:
                    with patch(
                        "torch.distributed.destroy_process_group"
                    ) as mock_destroy:
                        barrier()
                        cleanup()
                        mock_barrier.assert_called_once()
                        mock_destroy.assert_called_once()
