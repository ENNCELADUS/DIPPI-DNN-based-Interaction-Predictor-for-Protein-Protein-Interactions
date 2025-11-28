"""Unit tests for TUnA model (src/model/tuna.py).

Tests verify:
- Constructor accepts kwargs only (no config parsing)
- Forward API matches spec: batch dict -> {"logits": Tensor}
- No logging, checkpointing, or metrics computation
- Architecture matches original TUnA functionality
"""

import pytest
import torch

from src.model.tuna import TUnA


class TestTUnAConstructor:
    """Test TUnA constructor and configuration validation."""

    def test_constructor_with_minimal_required_fields(self):
        """Constructor should accept all required kwargs and build model."""
        model = TUnA(
            input_dim=1280,
            d_model=64,
            intra_layers=2,
            inter_layers=2,
            n_heads=8,
            ff_dim=256,
            dropout=0.1,
            activation="gelu",
        )
        assert isinstance(model, torch.nn.Module)
        assert model.input_dim == 1280
        assert model.d_model == 64
        assert model.intra_layers == 2
        assert model.inter_layers == 2
        assert model.n_heads == 8
        assert model.ff_dim == 256
        assert model.dropout == 0.1
        assert model.activation == "gelu"

    def test_constructor_raises_on_missing_required_fields(self):
        """Constructor should raise ValueError if required fields missing."""
        with pytest.raises(ValueError, match="Missing required model configuration"):
            TUnA(
                input_dim=1280,
                d_model=64,
                # Missing intra_layers, inter_layers, n_heads, ff_dim, dropout, activation
            )

    def test_constructor_accepts_unused_fields_gracefully(self):
        """Constructor should accept and store unused fields for config compatibility."""
        model = TUnA(
            input_dim=1280,
            d_model=64,
            intra_layers=2,
            inter_layers=2,
            n_heads=8,
            ff_dim=256,
            dropout=0.1,
            activation="gelu",
            spectral_norm=True,  # Unused field
            gp_layer={"some": "config"},  # Unused field
        )
        assert model._unused_spectral_norm is True
        assert model._unused_gp_layer == {"some": "config"}

    def test_constructor_validates_activation_function(self):
        """Constructor should raise ValueError for unsupported activation."""
        with pytest.raises(ValueError, match="Activation function .* not supported"):
            model = TUnA(
                input_dim=1280,
                d_model=64,
                intra_layers=1,
                inter_layers=1,
                n_heads=4,
                ff_dim=128,
                dropout=0.1,
                activation="invalid_activation",
            )
            # Trigger error by building the modules
            _ = model.intra_encoder


class TestTUnAForward:
    """Test TUnA forward pass and interface contract."""

    @pytest.fixture
    def model(self):
        """Create a small TUnA model for testing."""
        return TUnA(
            input_dim=128,
            d_model=64,
            intra_layers=1,
            inter_layers=1,
            n_heads=4,
            ff_dim=128,
            dropout=0.1,
            activation="gelu",
        )

    def test_forward_with_required_inputs(self, model):
        """Forward should process batch dict with emb_a and emb_b."""
        batch = {
            "emb_a": torch.randn(2, 10, 128),
            "emb_b": torch.randn(2, 12, 128),
        }
        output = model(batch)

        assert isinstance(output, dict)
        assert "logits" in output
        assert output["logits"].shape == (2, 1)

    def test_forward_with_sequence_lengths(self, model):
        """Forward should respect len_a and len_b for masking."""
        batch = {
            "emb_a": torch.randn(3, 20, 128),
            "emb_b": torch.randn(3, 15, 128),
            "len_a": torch.tensor([10, 15, 20]),
            "len_b": torch.tensor([8, 12, 15]),
        }
        output = model(batch)

        assert output["logits"].shape == (3, 1)

    def test_forward_with_different_sequence_lengths(self, model):
        """Forward should handle variable-length sequences in batch."""
        batch = {
            "emb_a": torch.randn(4, 25, 128),
            "emb_b": torch.randn(4, 30, 128),
            "len_a": torch.tensor([10, 15, 20, 25]),
            "len_b": torch.tensor([12, 18, 25, 30]),
        }
        output = model(batch)

        assert output["logits"].shape == (4, 1)

    def test_forward_raises_on_missing_emb_a(self, model):
        """Forward should raise KeyError if emb_a missing."""
        batch = {"emb_b": torch.randn(2, 10, 128)}
        with pytest.raises(KeyError, match="emb_a"):
            model(batch)

    def test_forward_raises_on_missing_emb_b(self, model):
        """Forward should raise KeyError if emb_b missing."""
        batch = {"emb_a": torch.randn(2, 10, 128)}
        with pytest.raises(KeyError, match="emb_b"):
            model(batch)

    def test_forward_raises_on_wrong_embedding_dim(self, model):
        """Forward should raise ValueError if embedding dim mismatches."""
        batch = {
            "emb_a": torch.randn(2, 10, 256),  # Wrong dim (should be 128)
            "emb_b": torch.randn(2, 10, 128),
        }
        with pytest.raises(ValueError, match="Input embedding dimension"):
            model(batch)

    def test_forward_raises_on_batch_size_mismatch(self, model):
        """Forward should raise ValueError if batch sizes don't match."""
        batch = {
            "emb_a": torch.randn(2, 10, 128),
            "emb_b": torch.randn(3, 10, 128),  # Different batch size
        }
        with pytest.raises(ValueError, match="matching batch dimension"):
            model(batch)

    def test_forward_raises_on_wrong_tensor_shape(self, model):
        """Forward should raise ValueError if tensors not 3D."""
        batch = {
            "emb_a": torch.randn(2, 128),  # 2D instead of 3D
            "emb_b": torch.randn(2, 10, 128),
        }
        with pytest.raises(ValueError, match="must be shaped"):
            model(batch)


class TestTUnADeviceHandling:
    """Test TUnA device handling and GPU compatibility."""

    def test_model_to_device(self):
        """Model should correctly transfer to device."""
        model = TUnA(
            input_dim=128,
            d_model=64,
            intra_layers=1,
            inter_layers=1,
            n_heads=4,
            ff_dim=128,
            dropout=0.1,
            activation="gelu",
        )

        # Test CPU
        model_cpu = model.to("cpu")
        assert model_cpu.device == torch.device("cpu")

        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to("cuda")
            assert model_cuda.device.type == "cuda"


class TestTUnAMVPBoundaries:
    """Test that TUnA adheres to MVP role boundaries."""

    def test_no_config_parsing_inside_model(self):
        """Model should not read configs, only accept kwargs."""
        # This is verified by constructor signature and implementation
        model = TUnA(
            input_dim=128,
            d_model=64,
            intra_layers=1,
            inter_layers=1,
            n_heads=4,
            ff_dim=128,
            dropout=0.1,
            activation="gelu",
        )
        # Model should not have any config parsing methods
        assert not hasattr(model, "load_config")
        assert not hasattr(model, "parse_config")

    def test_no_logging_methods(self):
        """Model should not have logging methods."""
        model = TUnA(
            input_dim=128,
            d_model=64,
            intra_layers=1,
            inter_layers=1,
            n_heads=4,
            ff_dim=128,
            dropout=0.1,
            activation="gelu",
        )
        assert not hasattr(model, "log")
        assert not hasattr(model, "logger")
        assert not hasattr(model, "log_metrics")

    def test_no_checkpoint_methods(self):
        """Model should not have checkpointing methods."""
        model = TUnA(
            input_dim=128,
            d_model=64,
            intra_layers=1,
            inter_layers=1,
            n_heads=4,
            ff_dim=128,
            dropout=0.1,
            activation="gelu",
        )
        # PyTorch's state_dict/load_state_dict are allowed (used by orchestrator)
        # but no custom checkpoint methods
        assert not hasattr(model, "save_checkpoint")
        assert not hasattr(model, "load_checkpoint")

    def test_no_training_loop_methods(self):
        """Model should not have training loop methods (Trainer's job)."""
        model = TUnA(
            input_dim=128,
            d_model=64,
            intra_layers=1,
            inter_layers=1,
            n_heads=4,
            ff_dim=128,
            dropout=0.1,
            activation="gelu",
        )
        assert not hasattr(model, "train_epoch")
        assert not hasattr(model, "train_step")
        assert not hasattr(model, "compute_loss")

    def test_no_metrics_computation(self):
        """Model should not compute metrics (Evaluator's job)."""
        model = TUnA(
            input_dim=128,
            d_model=64,
            intra_layers=1,
            inter_layers=1,
            n_heads=4,
            ff_dim=128,
            dropout=0.1,
            activation="gelu",
        )
        assert not hasattr(model, "compute_metrics")
        assert not hasattr(model, "evaluate")
        assert not hasattr(model, "compute_auroc")
