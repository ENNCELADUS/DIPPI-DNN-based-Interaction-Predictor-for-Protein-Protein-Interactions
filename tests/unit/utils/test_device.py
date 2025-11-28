"""
Unit tests for src/utils/device.py (MVP).

Tests device selection logic, auto-detection, and CUDA binding.
"""

import os
from unittest.mock import patch

import pytest
import torch

from src.utils.device import get_device, is_cuda, is_mps, is_cpu


class TestGetDevice:
    """Test device selection from config."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.set_device")
    def test_string_input_cuda(self, mock_set_device, mock_cuda_available):
        """Test string input: 'cuda'."""
        device = get_device("cuda")
        assert device.type == "cuda"
        mock_set_device.assert_called_once()

    def test_string_input_cpu(self):
        """Test string input: 'cpu'."""
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_dict_input_explicit_cpu(self):
        """Test dict input with explicit CPU strategy."""
        cfg = {"strategy": "cpu"}
        device = get_device(cfg)
        assert device.type == "cpu"

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.set_device")
    def test_cuda_auto_selection(self, mock_set_device, mock_cuda_available):
        """Test auto strategy selects CUDA when available."""
        cfg = {"strategy": "auto"}
        device = get_device(cfg)
        assert device.type == "cuda"
        mock_set_device.assert_called_once()

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_mps_auto_selection(self, mock_mps_available, mock_cuda_available):
        """Test auto strategy selects MPS when CUDA unavailable."""
        cfg = {"strategy": "auto"}
        device = get_device(cfg)
        assert device.type == "mps"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_cpu_fallback_auto(self, mock_mps_available, mock_cuda_available):
        """Test auto strategy falls back to CPU."""
        cfg = {"strategy": "auto"}
        device = get_device(cfg)
        assert device.type == "cpu"

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.set_device")
    def test_explicit_gpu_id(self, mock_set_device, mock_cuda_available):
        """Test explicit gpu_id in config."""
        cfg = {"strategy": "cuda", "gpu_id": 2}
        device = get_device(cfg)
        assert device.type == "cuda"
        assert device.index == 2
        mock_set_device.assert_called_once_with(2)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.set_device")
    @patch.dict(os.environ, {"LOCAL_RANK": "3"})
    def test_local_rank_precedence(self, mock_set_device, mock_cuda_available):
        """Test LOCAL_RANK env takes precedence over gpu_id."""
        cfg = {"strategy": "cuda", "gpu_id": 1}
        device = get_device(cfg, prefer_local_rank=True)
        assert device.type == "cuda"
        assert device.index == 3  # LOCAL_RANK=3 overrides gpu_id=1
        mock_set_device.assert_called_once_with(3)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.set_device")
    def test_local_rank_disabled(self, mock_set_device, mock_cuda_available):
        """Test prefer_local_rank=False uses gpu_id."""
        with patch.dict(os.environ, {"LOCAL_RANK": "3"}):
            cfg = {"strategy": "cuda", "gpu_id": 1}
            device = get_device(cfg, prefer_local_rank=False)
            assert device.type == "cuda"
            assert device.index == 1  # gpu_id used
            mock_set_device.assert_called_once_with(1)

    def test_invalid_strategy_raises(self):
        """Test unknown strategy raises ValueError."""
        cfg = {"strategy": "invalid"}
        with pytest.raises(ValueError, match="Unknown device strategy"):
            get_device(cfg)


class TestDeviceHelpers:
    """Test device type check helpers."""

    def test_is_cuda_true(self):
        """Test is_cuda returns True for CUDA device."""
        device = torch.device("cuda:0")
        assert is_cuda(device) is True

    def test_is_cuda_false(self):
        """Test is_cuda returns False for non-CUDA device."""
        device = torch.device("cpu")
        assert is_cuda(device) is False

    def test_is_mps_true(self):
        """Test is_mps returns True for MPS device."""
        device = torch.device("mps")
        assert is_mps(device) is True

    def test_is_mps_false(self):
        """Test is_mps returns False for non-MPS device."""
        device = torch.device("cpu")
        assert is_mps(device) is False

    def test_is_cpu_true(self):
        """Test is_cpu returns True for CPU device."""
        device = torch.device("cpu")
        assert is_cpu(device) is True

    def test_is_cpu_false(self):
        """Test is_cpu returns False for non-CPU device."""
        device = torch.device("cuda:0")
        assert is_cpu(device) is False
