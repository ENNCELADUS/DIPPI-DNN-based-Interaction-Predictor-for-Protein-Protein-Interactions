"""
Unit tests for src/utils/distributed.py (MVP).

Tests DDP initialization, rank/world_size helpers, and cleanup.
"""

import os
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch

from src.utils.distributed import (
    init_if_enabled,
    is_main_process,
    get_rank,
    get_world_size,
    barrier,
    cleanup,
)


class TestInitIfEnabled:
    """Test DDP initialization logic."""

    def test_disabled_returns_false(self):
        """Test disabled DDP returns False without init."""
        cfg = {"enabled": False}
        device = torch.device("cpu")
        result = init_if_enabled(cfg, device)
        assert result is False

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_rank_env_raises(self):
        """Test missing RANK/WORLD_SIZE raises RuntimeError."""
        cfg = {"enabled": True}
        device = torch.device("cpu")
        with pytest.raises(RuntimeError, match="RANK/WORLD_SIZE not set"):
            init_if_enabled(cfg, device)

    @patch("torch.distributed.init_process_group")
    @patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "2"})
    def test_auto_backend_cuda(self, mock_init_pg):
        """Test backend auto-selection: nccl for CUDA."""
        cfg = {"enabled": True}
        device = torch.device("cuda:0")
        result = init_if_enabled(cfg, device)
        assert result is True
        mock_init_pg.assert_called_once_with(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=1800),
        )

    @patch("torch.distributed.init_process_group")
    @patch.dict(os.environ, {"RANK": "1", "WORLD_SIZE": "4"})
    def test_auto_backend_cpu(self, mock_init_pg):
        """Test backend auto-selection: gloo for CPU."""
        cfg = {"enabled": True}
        device = torch.device("cpu")
        result = init_if_enabled(cfg, device)
        assert result is True
        mock_init_pg.assert_called_once_with(
            backend="gloo",
            init_method="env://",
            timeout=timedelta(seconds=1800),
        )

    @patch("torch.distributed.init_process_group")
    @patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "2"})
    def test_explicit_backend(self, mock_init_pg):
        """Test explicit backend from config."""
        cfg = {"enabled": True, "backend": "gloo"}
        device = torch.device("cuda:0")
        result = init_if_enabled(cfg, device)
        assert result is True
        mock_init_pg.assert_called_once_with(
            backend="gloo",  # explicit, not auto nccl
            init_method="env://",
            timeout=timedelta(seconds=1800),
        )

    @patch("torch.distributed.init_process_group")
    @patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "2"})
    def test_custom_timeout(self, mock_init_pg):
        """Test custom timeout from config."""
        cfg = {"enabled": True, "timeout_sec": 3600}
        device = torch.device("cpu")
        result = init_if_enabled(cfg, device)
        assert result is True
        mock_init_pg.assert_called_once_with(
            backend="gloo",
            init_method="env://",
            timeout=timedelta(seconds=3600),
        )


class TestRankHelpers:
    """Test rank/world_size query helpers."""

    @patch("torch.distributed.is_available", return_value=False)
    def test_get_rank_not_available(self, mock_available):
        """Test get_rank returns 0 when dist not available."""
        assert get_rank() == 0

    @patch("torch.distributed.is_available", return_value=True)
    @patch("torch.distributed.is_initialized", return_value=False)
    def test_get_rank_not_initialized(self, mock_init, mock_available):
        """Test get_rank returns 0 when dist not initialized."""
        assert get_rank() == 0

    @patch("torch.distributed.is_available", return_value=True)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_rank", return_value=2)
    def test_get_rank_initialized(self, mock_get_rank, mock_init, mock_available):
        """Test get_rank returns actual rank when initialized."""
        assert get_rank() == 2
        mock_get_rank.assert_called_once()

    @patch("torch.distributed.is_available", return_value=False)
    def test_get_world_size_not_available(self, mock_available):
        """Test get_world_size returns 1 when dist not available."""
        assert get_world_size() == 1

    @patch("torch.distributed.is_available", return_value=True)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_world_size", return_value=4)
    def test_get_world_size_initialized(
        self, mock_get_world_size, mock_init, mock_available
    ):
        """Test get_world_size returns actual size when initialized."""
        assert get_world_size() == 4
        mock_get_world_size.assert_called_once()


class TestIsMainProcess:
    """Test main process detection."""

    @patch("torch.distributed.is_available", return_value=False)
    def test_not_available_is_main(self, mock_available):
        """Test single-process mode is treated as main."""
        assert is_main_process() is True

    @patch("torch.distributed.is_available", return_value=True)
    @patch("torch.distributed.is_initialized", return_value=False)
    def test_not_initialized_is_main(self, mock_init, mock_available):
        """Test uninitialized mode is treated as main."""
        assert is_main_process() is True

    @patch("torch.distributed.is_available", return_value=True)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_rank_zero_is_main(self, mock_get_rank, mock_init, mock_available):
        """Test rank 0 is main process."""
        assert is_main_process() is True

    @patch("torch.distributed.is_available", return_value=True)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_rank", return_value=1)
    def test_rank_nonzero_not_main(self, mock_get_rank, mock_init, mock_available):
        """Test non-zero rank is not main process."""
        assert is_main_process() is False


class TestBarrier:
    """Test barrier synchronization."""

    @patch("torch.distributed.is_available", return_value=False)
    def test_barrier_not_available_noop(self, mock_available):
        """Test barrier is no-op when dist not available."""
        barrier()  # should not raise

    @patch("torch.distributed.is_available", return_value=True)
    @patch("torch.distributed.is_initialized", return_value=False)
    def test_barrier_not_initialized_noop(self, mock_init, mock_available):
        """Test barrier is no-op when dist not initialized."""
        barrier()  # should not raise

    @patch("torch.distributed.is_available", return_value=True)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.barrier")
    def test_barrier_initialized_calls(self, mock_barrier, mock_init, mock_available):
        """Test barrier calls dist.barrier when initialized."""
        barrier()
        mock_barrier.assert_called_once()


class TestCleanup:
    """Test process group cleanup."""

    @patch("torch.distributed.is_available", return_value=False)
    def test_cleanup_not_available_noop(self, mock_available):
        """Test cleanup is no-op when dist not available."""
        cleanup()  # should not raise

    @patch("torch.distributed.is_available", return_value=True)
    @patch("torch.distributed.is_initialized", return_value=False)
    def test_cleanup_not_initialized_noop(self, mock_init, mock_available):
        """Test cleanup is no-op when dist not initialized."""
        cleanup()  # should not raise

    @patch("torch.distributed.is_available", return_value=True)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.destroy_process_group")
    def test_cleanup_initialized_destroys(
        self, mock_destroy, mock_init, mock_available
    ):
        """Test cleanup calls destroy_process_group when initialized."""
        cleanup()
        mock_destroy.assert_called_once()
