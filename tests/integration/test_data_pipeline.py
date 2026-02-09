"""
Integration tests for data loading pipeline with real data and models.
"""

import pytest
import torch
from pathlib import Path

from src.utils.data.io import build_loader
from src.model.v3 import V3
from src.model.tuna import TUnA


# Skip if data files don't exist (e.g., in CI without data)
DATA_DIR = Path("data/membrane_protein/splits")
EMBEDDINGS_PATH = Path("data/embedding/complete_soluble_proteins_embeddings.pkl")


@pytest.mark.skipif(
    not DATA_DIR.exists() or not EMBEDDINGS_PATH.exists(),
    reason="Real data files not available",
)
class TestDataPipelineIntegration:
    """Integration tests with real data files."""

    def test_load_real_data_small_batch(self):
        """Test loading a small batch from real data."""
        csv_path = DATA_DIR / "pretrain_train.csv"

        if not csv_path.exists():
            pytest.skip("Pretrain train CSV not available")

        loader = build_loader(
            csv_path=str(csv_path),
            embeddings_path=str(EMBEDDINGS_PATH),
            batch_size=2,
            max_len=128,
            dtype="fp32",
            ddp=False,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        # Fetch one batch
        batch = next(iter(loader))

        # Verify batch structure
        assert "emb_a" in batch
        assert "emb_b" in batch
        assert "len_a" in batch
        assert "len_b" in batch
        assert "label" in batch

        # Verify shapes
        assert batch["emb_a"].shape == (2, 128, 1536)
        assert batch["emb_b"].shape == (2, 128, 1536)
        assert batch["len_a"].shape == (2,)
        assert batch["len_b"].shape == (2,)
        assert batch["label"].shape == (2, 1)

    def test_v3_model_forward_with_real_batch(self):
        """Test V3 model forward pass with real data batch."""
        csv_path = DATA_DIR / "pretrain_train.csv"

        if not csv_path.exists():
            pytest.skip("Pretrain train CSV not available")

        # Create loader
        loader = build_loader(
            csv_path=str(csv_path),
            embeddings_path=str(EMBEDDINGS_PATH),
            batch_size=4,
            max_len=256,
            dtype="fp32",
            ddp=False,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        # Create V3 model
        model = V3(
            input_dim=1536,
            d_model=128,
            encoder_layers=1,
            cross_attn_layers=1,
            n_heads=4,
            mlp_head={
                "hidden_dims": [64],
                "dropout": 0.1,
                "activation": "gelu",
                "norm": "layernorm",
            },
            regularization={
                "dropout": 0.1,
                "token_dropout": 0.0,
                "cross_attention_dropout": 0.05,
            },
        )
        model.eval()

        # Fetch batch and run forward pass
        batch = next(iter(loader))

        with torch.no_grad():
            output = model(batch)

        # Verify output
        assert "logits" in output
        assert output["logits"].shape == (4, 1)
        assert torch.isfinite(output["logits"]).all()

    def test_tuna_model_forward_with_real_batch(self):
        """Test TUnA model forward pass with real data batch."""
        csv_path = DATA_DIR / "pretrain_train.csv"

        if not csv_path.exists():
            pytest.skip("Pretrain train CSV not available")

        # Create loader
        loader = build_loader(
            csv_path=str(csv_path),
            embeddings_path=str(EMBEDDINGS_PATH),
            batch_size=4,
            max_len=256,
            dtype="fp32",
            ddp=False,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        # Create TUnA model
        model = TUnA(
            input_dim=1536,
            d_model=128,
            intra_layers=1,
            inter_layers=1,
            n_heads=4,
            ff_dim=256,
            dropout=0.1,
            activation="gelu",
        )
        model.eval()

        # Fetch batch and run forward pass
        batch = next(iter(loader))

        with torch.no_grad():
            output = model(batch)

        # Verify output
        assert "logits" in output
        assert output["logits"].shape == (4, 1)
        assert torch.isfinite(output["logits"]).all()

    def test_bf16_dtype_with_model(self):
        """Test bf16 dtype conversion works with model forward."""
        csv_path = DATA_DIR / "pretrain_train.csv"

        if not csv_path.exists():
            pytest.skip("Pretrain train CSV not available")

        # Create loader with bf16
        loader = build_loader(
            csv_path=str(csv_path),
            embeddings_path=str(EMBEDDINGS_PATH),
            batch_size=2,
            max_len=128,
            dtype="bf16",
            ddp=False,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        batch = next(iter(loader))

        # Verify dtypes
        assert batch["emb_a"].dtype == torch.bfloat16
        assert batch["emb_b"].dtype == torch.bfloat16

        # Create model and convert to bf16
        model = V3(
            input_dim=1536,
            d_model=64,
            encoder_layers=1,
            cross_attn_layers=1,
            n_heads=4,
            mlp_head={
                "hidden_dims": [32],
                "dropout": 0.1,
            },
            regularization={
                "dropout": 0.1,
            },
        )
        model = model.to(dtype=torch.bfloat16)
        model.eval()

        # Forward pass should work
        with torch.no_grad():
            output = model(batch)

        assert output["logits"].dtype == torch.bfloat16
        assert torch.isfinite(output["logits"]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
