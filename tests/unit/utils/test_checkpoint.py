"""Unit tests for src.utils.checkpoint module."""

import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    infer_resume_global_step,
)


class SimpleModel(nn.Module):
    """Simple test model."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class EMAModel(nn.Module):
    """Test model with EMA support."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.ema_model = SimpleModel()

    def forward(self, x):
        return self.linear(x)

    def ema_state_dict(self):
        return self.ema_model.state_dict()


class TestSaveCheckpoint:
    """Test save_checkpoint function."""

    @patch("src.utils.checkpoint.checkpoint_paths")
    @patch("src.utils.checkpoint.ensure_dir")
    @patch("torch.save")
    def test_save_best_checkpoint_basic(self, mock_save, mock_ensure, mock_paths):
        """Test saving best checkpoint without EMA/SWA."""
        # Setup mocks
        test_dir = Path("/fake/dir")
        test_model_path = test_dir / "best_model.pth"
        mock_paths.return_value = {"dir": test_dir, "model": test_model_path}

        # Create test objects
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        # Call function
        result = save_checkpoint(
            model,
            "finetune",
            "20240815_143022",
            best=True,
            model_name="v3",
            optimizer=optimizer,
        )

        # Verify behavior
        mock_paths.assert_called_once_with(
            "v3",
            "finetune",
            "20240815_143022",
            best=True,
            epoch=None,
            ema=False,
            swa=False,
        )
        mock_ensure.assert_called_once_with(test_dir)
        mock_save.assert_called_once()

        # Check saved data structure
        save_args = mock_save.call_args[0]
        saved_data = save_args[0]
        saved_path = save_args[1]

        assert "model_state" in saved_data
        assert "optimizer_state" in saved_data
        assert saved_path == test_model_path
        assert result == {"model": test_model_path}

    @patch("src.utils.checkpoint.checkpoint_paths")
    @patch("src.utils.checkpoint.ensure_dir")
    @patch("torch.save")
    def test_save_epoch_checkpoint_with_ema(self, mock_save, mock_ensure, mock_paths):
        """Test saving epoch checkpoint with EMA."""
        # Setup mocks
        test_dir = Path("/fake/dir")
        test_model_path = test_dir / "checkpoint_epoch_10.pth"
        test_ema_path = test_dir / "checkpoint_epoch_10_ema.pth"
        mock_paths.return_value = {
            "dir": test_dir,
            "model": test_model_path,
            "ema": test_ema_path,
        }

        # Create test objects
        model = EMAModel()

        # Call function
        result = save_checkpoint(
            model,
            "pretrain",
            "20240815_143022",
            best=False,
            epoch=10,
            ema=True,
            model_name="v3",
        )

        # Verify behavior
        mock_paths.assert_called_once_with(
            "v3",
            "pretrain",
            "20240815_143022",
            best=False,
            epoch=10,
            ema=True,
            swa=False,
        )

        # Should save both model and EMA checkpoints
        assert mock_save.call_count == 2
        assert result == {"model": test_model_path, "ema": test_ema_path}

    @patch("src.utils.checkpoint.checkpoint_paths")
    @patch("src.utils.checkpoint.ensure_dir")
    @patch("torch.save")
    def test_save_ema_missing_raises_error(self, mock_save, mock_ensure, mock_paths):
        """Test that missing EMA state raises ValueError."""
        mock_paths.return_value = {
            "dir": Path("/fake"),
            "model": Path("/fake/model.pth"),
            "ema": Path("/fake/ema.pth"),
        }

        model = SimpleModel()  # No EMA support

        with pytest.raises(
            ValueError, match="EMA requested but no ema_state_dict available"
        ):
            save_checkpoint(
                model,
                "finetune",
                "20240815_143022",
                best=True,
                ema=True,
                model_name="v3",
            )

    @patch("src.utils.checkpoint.checkpoint_paths")
    @patch("src.utils.checkpoint.ensure_dir")
    @patch("torch.save")
    def test_save_with_ema_kwarg(self, mock_save, mock_ensure, mock_paths):
        """Test saving with EMA state dict provided as kwarg."""
        test_dir = Path("/fake/dir")
        mock_paths.return_value = {
            "dir": test_dir,
            "model": test_dir / "best_model.pth",
            "ema": test_dir / "best_model_ema.pth",
        }

        model = SimpleModel()
        ema_dict = {"linear.weight": torch.randn(5, 10)}

        result = save_checkpoint(
            model,
            "finetune",
            "20240815_143022",
            best=True,
            ema=True,
            model_name="v3",
            ema_state_dict=ema_dict,
        )

        assert mock_save.call_count == 2
        assert "ema" in result

    @patch("src.utils.checkpoint.checkpoint_paths")
    @patch("src.utils.checkpoint.ensure_dir")
    @patch("torch.save")
    def test_save_swa_missing_raises_error(self, mock_save, mock_ensure, mock_paths):
        """Test that missing SWA state raises ValueError."""
        mock_paths.return_value = {
            "dir": Path("/fake"),
            "model": Path("/fake/model.pth"),
            "swa": Path("/fake/swa.pth"),
        }

        model = SimpleModel()  # No SWA support

        with pytest.raises(
            ValueError, match="SWA requested but no swa_state_dict available"
        ):
            save_checkpoint(
                model,
                "finetune",
                "20240815_143022",
                best=True,
                swa=True,
                model_name="v3",
            )

    @patch("src.utils.checkpoint.checkpoint_paths")
    @patch("src.utils.checkpoint.ensure_dir")
    @patch("torch.save")
    def test_save_with_swa_kwarg(self, mock_save, mock_ensure, mock_paths):
        """Test saving with SWA state dict provided as kwarg."""
        test_dir = Path("/fake/dir")
        mock_paths.return_value = {
            "dir": test_dir,
            "model": test_dir / "best_model.pth",
            "swa": test_dir / "best_model_swa.pth",
        }

        model = SimpleModel()
        swa_dict = {"linear.weight": torch.randn(5, 10), "linear.bias": torch.randn(5)}

        result = save_checkpoint(
            model,
            "finetune",
            "20240815_143022",
            best=True,
            swa=True,
            model_name="v3",
            swa_state_dict=swa_dict,
        )

        assert mock_save.call_count == 2
        assert "swa" in result

    @patch("src.utils.checkpoint.checkpoint_paths")
    @patch("src.utils.checkpoint.ensure_dir")
    @patch("torch.save")
    def test_save_with_scheduler_and_extra(self, mock_save, mock_ensure, mock_paths):
        """Test saving with scheduler and extra data."""
        test_dir = Path("/fake/dir")
        mock_paths.return_value = {
            "dir": test_dir,
            "model": test_dir / "best_model.pth",
        }

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        extra_data = {"experiment_id": "exp_123", "notes": "test run"}

        result = save_checkpoint(
            model,
            "finetune",
            "20240815_143022",
            best=True,
            model_name="v3",
            optimizer=optimizer,
            scheduler=scheduler,
            extra=extra_data,
        )

        # Check that save was called
        mock_save.assert_called_once()

        # Check saved data includes scheduler and extra
        save_args = mock_save.call_args[0]
        saved_data = save_args[0]

        assert "scheduler_state" in saved_data
        assert "extra" in saved_data
        assert saved_data["extra"] == extra_data
        assert result == {"model": test_dir / "best_model.pth"}


class TestInferResumeGlobalStep:
    """Tests for infer_resume_global_step utility."""

    def test_prefers_explicit_global_step(self):
        report = {
            "global_step": 123,
            "checkpoint": {"extra": {"global_step": 5}},
        }

        value, source = infer_resume_global_step(report)

        assert value == 123
        assert source == "report.global_step"

    def test_uses_scheduler_last_epoch_for_per_batch(self):
        param = torch.nn.Parameter(torch.randn(2, 2))
        optimizer = torch.optim.Adam([param], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-2, total_steps=100
        )
        scheduler.last_epoch = 42

        value, source = infer_resume_global_step(
            {"checkpoint": {}}, scheduler=scheduler
        )

        assert value == 42
        assert source == "scheduler.last_epoch"

    def test_falls_back_to_epoch_length(self):
        report = {"checkpoint": {"extra": {"current_epoch": 2}}}

        value, source = infer_resume_global_step(report, epoch_length=100)

        assert value == 300
        assert source == "epoch_length"


class TestLoadCheckpoint:
    """Test load_checkpoint function."""

    @patch("src.utils.checkpoint.checkpoint_paths")
    @patch("torch.load")
    @patch("pathlib.Path.exists")
    def test_load_best_checkpoint_basic(self, mock_exists, mock_load, mock_paths):
        """Test loading best checkpoint."""
        # Setup mocks
        test_path = Path("/fake/best_model.pth")
        mock_paths.return_value = {"model": test_path}
        mock_exists.return_value = True

        # Create test objects first to get proper optimizer structure
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        # Mock checkpoint data with proper optimizer structure
        mock_checkpoint = {
            "model_state": {
                "linear.weight": torch.randn(5, 10),
                "linear.bias": torch.randn(5),
            },
            "optimizer_state": optimizer.state_dict(),  # Use real optimizer structure
            "epoch": 100,
        }
        mock_load.return_value = mock_checkpoint

        # Call function - structured mode
        result = load_checkpoint(
            model,
            stage="finetune",
            run_id="20240815_143022",
            model_name="v3",
            best=True,
            optimizer=optimizer,
        )

        # Verify behavior
        mock_paths.assert_called_once_with(
            "v3",
            "finetune",
            "20240815_143022",
            best=True,
            epoch=None,
            ema=False,
            swa=False,
        )
        mock_load.assert_called_once_with(
            test_path, map_location="cpu", weights_only=False
        )

        # Check return value - new unified format
        assert result["path"] == test_path
        assert result["epoch"] == 100
        assert "model_state" in result["keys"]
        assert result["format"] == "checkpoint"
        assert "missing_keys" in result
        assert "unexpected_keys" in result
        assert "model_state_dict" in result
        assert "checkpoint" in result

    @patch("src.utils.checkpoint.checkpoint_paths")
    @patch("pathlib.Path.exists")
    def test_load_missing_file_error(self, mock_exists, mock_paths):
        """Test that missing checkpoint file raises FileNotFoundError with absolute path."""
        test_path = Path("/fake/missing_model.pth")
        mock_paths.return_value = {"model": test_path}
        mock_exists.return_value = False

        model = SimpleModel()

        with pytest.raises(FileNotFoundError) as exc_info:
            load_checkpoint(
                model,
                stage="finetune",
                run_id="20240815_143022",
                model_name="v3",
                best=True,
            )

        # Check that absolute path is in error message
        assert str(test_path.absolute()) in str(exc_info.value)

    @patch("src.utils.checkpoint.checkpoint_paths")
    @patch("torch.load")
    @patch("pathlib.Path.exists")
    def test_load_ema_checkpoint(self, mock_exists, mock_load, mock_paths):
        """Test loading EMA checkpoint instead of main model."""
        # Setup mocks
        test_model_path = Path("/fake/best_model.pth")
        test_ema_path = Path("/fake/best_model_ema.pth")
        mock_paths.return_value = {"model": test_model_path, "ema": test_ema_path}
        mock_exists.return_value = True

        mock_checkpoint = {
            "model_state": {
                "linear.weight": torch.randn(5, 10),
                "linear.bias": torch.randn(5),
            },
            "epoch": 50,
        }
        mock_load.return_value = mock_checkpoint

        model = SimpleModel()

        result = load_checkpoint(
            model,
            stage="finetune",
            run_id="20240815_143022",
            model_name="v3",
            best=True,
            ema=True,
        )

        # Should load from EMA path, not main model path
        mock_load.assert_called_once_with(
            test_ema_path, map_location="cpu", weights_only=False
        )
        assert result["path"] == test_ema_path

    def test_load_both_ema_swa_error(self):
        """Test that loading both EMA and SWA raises ValueError."""
        model = SimpleModel()

        with pytest.raises(
            ValueError, match="Cannot load both EMA and SWA simultaneously"
        ):
            load_checkpoint(
                model,
                stage="finetune",
                run_id="20240815_143022",
                model_name="v3",
                best=True,
                ema=True,
                swa=True,
            )

    @patch("src.utils.checkpoint.checkpoint_paths")
    @patch("torch.load")
    @patch("pathlib.Path.exists")
    def test_load_swa_checkpoint(self, mock_exists, mock_load, mock_paths):
        """Test loading SWA checkpoint instead of main model."""
        # Setup mocks
        test_model_path = Path("/fake/best_model.pth")
        test_swa_path = Path("/fake/best_model_swa.pth")
        mock_paths.return_value = {"model": test_model_path, "swa": test_swa_path}
        mock_exists.return_value = True

        mock_checkpoint = {
            "model_state": {
                "linear.weight": torch.randn(5, 10),
                "linear.bias": torch.randn(5),
            },
            "epoch": 75,
        }
        mock_load.return_value = mock_checkpoint

        model = SimpleModel()

        result = load_checkpoint(
            model,
            stage="finetune",
            run_id="20240815_143022",
            model_name="v3",
            best=True,
            swa=True,
        )

        # Should load from SWA path, not main model path
        mock_load.assert_called_once_with(
            test_swa_path, map_location="cpu", weights_only=False
        )
        assert result["path"] == test_swa_path

    @patch("src.utils.checkpoint.checkpoint_paths")
    @patch("torch.load")
    @patch("pathlib.Path.exists")
    def test_load_with_scheduler_restoration(self, mock_exists, mock_load, mock_paths):
        """Test loading checkpoint with scheduler state restoration."""
        # Setup mocks
        test_path = Path("/fake/best_model.pth")
        mock_paths.return_value = {"model": test_path}
        mock_exists.return_value = True

        # Create test objects first to get proper state dicts
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        mock_checkpoint = {
            "model_state": {
                "linear.weight": torch.randn(5, 10),
                "linear.bias": torch.randn(5),
            },
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": 50,
        }
        mock_load.return_value = mock_checkpoint

        # Load checkpoint
        result = load_checkpoint(
            model,
            stage="finetune",
            run_id="20240815_143022",
            model_name="v3",
            best=True,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        # Verify load was called
        mock_load.assert_called_once_with(
            test_path, map_location="cpu", weights_only=False
        )

        # Check return value
        assert result["path"] == test_path
        assert result["epoch"] == 50
        assert "scheduler_state" in result["keys"]


class TestIntegration:
    """Integration tests for save/load round-trip."""

    def test_save_load_roundtrip(self):
        """Test that saved checkpoint can be loaded correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Mock the paths module to use temp directory
            with (
                patch("src.utils.checkpoint.checkpoint_paths") as mock_paths,
                patch("src.utils.checkpoint.ensure_dir") as mock_ensure,
            ):
                # Setup real file paths in temp directory
                checkpoint_dir = (
                    Path(tmp_dir) / "models" / "v3" / "finetune" / "20240815_143022"
                )
                checkpoint_path = checkpoint_dir / "best_model.pth"

                mock_paths.return_value = {
                    "dir": checkpoint_dir,
                    "model": checkpoint_path,
                }
                mock_ensure.return_value = checkpoint_dir

                # Create and save model
                model = SimpleModel()
                optimizer = torch.optim.Adam(model.parameters())

                # Actually create the directory
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Save checkpoint
                save_result = save_checkpoint(
                    model,
                    "finetune",
                    "20240815_143022",
                    best=True,
                    model_name="v3",
                    optimizer=optimizer,
                    epoch=42,
                )

                # Verify file was created
                assert checkpoint_path.exists()
                assert save_result["model"] == checkpoint_path

                # Create new model and load
                new_model = SimpleModel()
                new_optimizer = torch.optim.Adam(new_model.parameters())

                # Mock paths for loading
                with patch("src.utils.checkpoint.checkpoint_paths") as mock_load_paths:
                    mock_load_paths.return_value = {"model": checkpoint_path}

                    load_result = load_checkpoint(
                        new_model,
                        stage="finetune",
                        run_id="20240815_143022",
                        model_name="v3",
                        best=True,
                        optimizer=new_optimizer,
                    )

                # Verify load results
                assert load_result["path"] == checkpoint_path
                assert load_result["epoch"] == 42
                assert "model_state" in load_result["keys"]


class TestLoadCheckpointDirectPath:
    """Test load_checkpoint direct path mode (new unified API)."""

    @patch("torch.load")
    @patch("pathlib.Path.exists")
    def test_load_direct_path_checkpoint_format(self, mock_exists, mock_load):
        """Test loading from direct path with checkpoint format detection."""
        mock_exists.return_value = True

        # Mock checkpoint with model_state_dict format
        mock_checkpoint = {
            "model_state_dict": {
                "linear.weight": torch.randn(5, 10),
                "linear.bias": torch.randn(5),
            },
            "epoch": 100,
            "optimizer_state": {"state": {}, "param_groups": []},
        }
        mock_load.return_value = mock_checkpoint

        model = SimpleModel()

        result = load_checkpoint(model, checkpoint_path="/path/to/checkpoint.pth")

        # Verify behavior
        mock_load.assert_called_once_with(
            Path("/path/to/checkpoint.pth"), map_location="cpu", weights_only=False
        )

        # Check enhanced return format
        assert result["path"] == Path("/path/to/checkpoint.pth")
        assert result["epoch"] == 100
        assert result["format"] == "checkpoint"
        assert "model_state_dict" in result
        assert "missing_keys" in result
        assert "unexpected_keys" in result
        assert "checkpoint" in result

    @patch("torch.load")
    @patch("pathlib.Path.exists")
    def test_load_direct_path_raw_state_dict(self, mock_exists, mock_load):
        """Test loading from direct path with raw state dict format."""
        mock_exists.return_value = True

        # Mock raw state dict (no model_state_dict wrapper)
        mock_state_dict = {
            "linear.weight": torch.randn(5, 10),
            "linear.bias": torch.randn(5),
        }
        mock_load.return_value = mock_state_dict

        model = SimpleModel()

        result = load_checkpoint(model, checkpoint_path="/path/to/state_dict.pth")

        # Check format detection
        assert result["format"] == "raw_state_dict"
        assert result["model_state_dict"] == mock_state_dict

    @patch("torch.load")
    @patch("pathlib.Path.exists")
    def test_load_direct_path_utils_format(self, mock_exists, mock_load):
        """Test loading from direct path with utils checkpoint format (model_state)."""
        mock_exists.return_value = True

        # Mock utils checkpoint format
        mock_checkpoint = {
            "model_state": {
                "linear.weight": torch.randn(5, 10),
                "linear.bias": torch.randn(5),
            },
            "epoch": 50,
        }
        mock_load.return_value = mock_checkpoint

        model = SimpleModel()

        result = load_checkpoint(model, checkpoint_path="/path/to/utils_checkpoint.pth")

        # Check format detection
        assert result["format"] == "checkpoint"
        assert result["model_state_dict"] == mock_checkpoint["model_state"]
        assert result["epoch"] == 50

    @patch("pathlib.Path.exists")
    def test_load_direct_path_missing_file(self, mock_exists):
        """Test that missing file in direct path mode raises FileNotFoundError."""
        mock_exists.return_value = False

        model = SimpleModel()

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            load_checkpoint(model, checkpoint_path="/nonexistent/checkpoint.pth")

    @patch("torch.load")
    @patch("pathlib.Path.exists")
    def test_load_direct_path_invalid_format(self, mock_exists, mock_load):
        """Test that invalid checkpoint format raises ValueError."""
        mock_exists.return_value = True
        mock_load.return_value = "invalid_format"  # Not a dict

        model = SimpleModel()

        with pytest.raises(ValueError, match="Invalid checkpoint format"):
            load_checkpoint(model, checkpoint_path="/path/to/invalid.pth")


class TestLoadCheckpointValidation:
    """Test load_checkpoint parameter validation."""

    def test_both_modes_specified_error(self):
        """Test that specifying both modes raises ValueError."""
        model = SimpleModel()

        with pytest.raises(
            ValueError,
            match="Cannot specify both checkpoint_path and structured parameters",
        ):
            load_checkpoint(
                model,
                checkpoint_path="/path/to/checkpoint.pth",
                stage="finetune",
                run_id="20240815_143022",
                model_name="v3",
            )

    def test_no_mode_specified_error(self):
        """Test that specifying neither mode raises ValueError."""
        model = SimpleModel()

        with pytest.raises(
            ValueError,
            match="Must specify either checkpoint_path OR structured parameters",
        ):
            load_checkpoint(model)

    def test_partial_structured_mode_error(self):
        """Test that incomplete structured parameters raise ValueError."""
        model = SimpleModel()

        # Missing model_name
        with pytest.raises(
            ValueError,
            match="Must specify either checkpoint_path OR structured parameters",
        ):
            load_checkpoint(model, stage="finetune", run_id="20240815_143022")

        # Missing stage
        with pytest.raises(
            ValueError,
            match="Must specify either checkpoint_path OR structured parameters",
        ):
            load_checkpoint(model, run_id="20240815_143022", model_name="v3")


class TestLoadCheckpointStateDict:
    """Test load_checkpoint state dict loading and verification."""

    @patch("torch.load")
    @patch("pathlib.Path.exists")
    def test_missing_keys_detection(self, mock_exists, mock_load):
        """Test detection and reporting of missing keys."""
        mock_exists.return_value = True

        # Create model with extra parameter that won't be in checkpoint
        model = SimpleModel()
        model.extra_param = nn.Parameter(torch.randn(3))

        # Mock checkpoint missing the extra parameter
        mock_checkpoint = {
            "model_state_dict": {
                "linear.weight": torch.randn(5, 10),
                "linear.bias": torch.randn(5),
                # Missing extra_param
            }
        }
        mock_load.return_value = mock_checkpoint

        result = load_checkpoint(
            model,
            checkpoint_path="/path/to/checkpoint.pth",
            strict=False,  # Allow missing keys
        )

        # Should detect missing keys
        assert "extra_param" in result["missing_keys"]

    @patch("torch.load")
    @patch("pathlib.Path.exists")
    def test_unexpected_keys_detection(self, mock_exists, mock_load):
        """Test detection and reporting of unexpected keys."""
        mock_exists.return_value = True

        model = SimpleModel()

        # Mock checkpoint with extra parameter not in model
        mock_checkpoint = {
            "model_state_dict": {
                "linear.weight": torch.randn(5, 10),
                "linear.bias": torch.randn(5),
                "extra_param": torch.randn(3),  # Not in model
            }
        }
        mock_load.return_value = mock_checkpoint

        result = load_checkpoint(
            model,
            checkpoint_path="/path/to/checkpoint.pth",
            strict=False,  # Allow unexpected keys
        )

        # Should detect unexpected keys
        assert "extra_param" in result["unexpected_keys"]

    @patch("torch.load")
    @patch("pathlib.Path.exists")
    def test_strict_loading_with_mismatch(self, mock_exists, mock_load):
        """Test that strict=True with key mismatch raises error."""
        mock_exists.return_value = True

        model = SimpleModel()
        model.extra_param = nn.Parameter(torch.randn(3))

        # Mock checkpoint missing the extra parameter
        mock_checkpoint = {
            "model_state_dict": {
                "linear.weight": torch.randn(5, 10),
                "linear.bias": torch.randn(5),
            }
        }
        mock_load.return_value = mock_checkpoint

        with pytest.raises(
            RuntimeError
        ):  # PyTorch raises RuntimeError for missing keys
            load_checkpoint(
                model,
                checkpoint_path="/path/to/checkpoint.pth",
                strict=True,  # Strict loading should fail
            )
