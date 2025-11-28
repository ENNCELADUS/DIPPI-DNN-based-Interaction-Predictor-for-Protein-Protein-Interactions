"""
Unit tests for src/utils/data_io.py
"""

import tempfile
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.utils.data_io import (
    ProteinPairDataset,
    build_loader,
    _clean_tokens,
    _collate_protein_pairs,
)


@pytest.fixture
def mock_embeddings_dict():
    """Create mock embeddings dict with CLS/EOS tokens."""
    return {
        "PROT_A": {
            "embeddings": np.random.randn(1, 52, 1536).astype(
                np.float32
            ),  # 50 + CLS + EOS
            "uniprot_id": "PROT_A",
        },
        "PROT_B": {
            "embeddings": np.random.randn(1, 102, 1536).astype(
                np.float32
            ),  # 100 + CLS + EOS
            "uniprot_id": "PROT_B",
        },
        "PROT_C": {
            "embeddings": np.random.randn(1, 202, 1536).astype(
                np.float32
            ),  # 200 + CLS + EOS
            "uniprot_id": "PROT_C",
        },
    }


@pytest.fixture
def mock_csv_file(tmp_path):
    """Create mock CSV file with protein pairs."""
    csv_path = tmp_path / "test_pairs.csv"
    df = pd.DataFrame(
        {
            "uniprotID_A": ["PROT_A", "PROT_B", "PROT_A"],
            "uniprotID_B": ["PROT_B", "PROT_C", "PROT_C"],
            "isInteraction": [1, 0, 1],
        }
    )
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_embeddings_file(tmp_path, mock_embeddings_dict):
    """Create mock pickle file with embeddings."""
    pkl_path = tmp_path / "embeddings.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(mock_embeddings_dict, f)
    return pkl_path


class TestCleanTokens:
    """Tests for _clean_tokens helper."""

    def test_strip_cls_eos(self):
        """Test CLS/EOS stripping."""
        # Simulate embeddings with CLS/EOS: (1, 52, 1536) for 50 actual tokens
        embeddings = torch.randn(1, 52, 1536)
        lengths = torch.tensor([50])

        cleaned, clean_lengths = _clean_tokens(embeddings, lengths, strip_cls_eos=True)

        # Should remove first and last: 52 -> 50
        assert cleaned.shape == (1, 50, 1536)
        assert clean_lengths[0].item() == 50

    def test_no_strip(self):
        """Test keeping embeddings as-is."""
        embeddings = torch.randn(1, 50, 1536)
        lengths = torch.tensor([50])

        cleaned, clean_lengths = _clean_tokens(embeddings, lengths, strip_cls_eos=False)

        # Should be unchanged
        assert cleaned.shape == (1, 50, 1536)
        assert clean_lengths[0].item() == 50

    def test_no_cls_eos_present(self):
        """Test heuristic when no CLS/EOS present."""
        # embeddings size matches lengths (no +2)
        embeddings = torch.randn(1, 50, 1536)
        lengths = torch.tensor([50])

        cleaned, clean_lengths = _clean_tokens(embeddings, lengths, strip_cls_eos=True)

        # Should not strip (heuristic detects no CLS/EOS)
        assert cleaned.shape == (1, 50, 1536)


class TestCollateFunction:
    """Tests for _collate_protein_pairs."""

    def test_collate_batch(self):
        """Test collating a batch of samples."""
        batch = [
            {
                "emb_a": torch.randn(100, 1536),
                "emb_b": torch.randn(100, 1536),
                "len_a": 50,
                "len_b": 75,
                "label": 1.0,
            },
            {
                "emb_a": torch.randn(100, 1536),
                "emb_b": torch.randn(100, 1536),
                "len_a": 80,
                "len_b": 60,
                "label": 0.0,
            },
        ]

        collated = _collate_protein_pairs(batch)

        assert collated["emb_a"].shape == (2, 100, 1536)
        assert collated["emb_b"].shape == (2, 100, 1536)
        assert collated["len_a"].shape == (2,)
        assert collated["len_b"].shape == (2,)
        assert collated["label"].shape == (2, 1)
        assert collated["label"].dtype == torch.float32


class TestProteinPairDataset:
    """Tests for ProteinPairDataset."""

    def test_dataset_init(self, mock_csv_file, mock_embeddings_dict):
        """Test dataset initialization."""
        dataset = ProteinPairDataset(
            csv_path=str(mock_csv_file),
            embeddings_dict=mock_embeddings_dict,
            max_len=128,
            dtype="fp32",
        )

        assert len(dataset) == 3
        assert dataset.max_len == 128
        assert dataset.torch_dtype == torch.float32

    def test_dataset_getitem(self, mock_csv_file, mock_embeddings_dict):
        """Test fetching a single sample."""
        dataset = ProteinPairDataset(
            csv_path=str(mock_csv_file),
            embeddings_dict=mock_embeddings_dict,
            max_len=128,
            dtype="fp32",
            strip_cls_eos=True,
        )

        sample = dataset[0]

        # Check keys
        assert set(sample.keys()) == {"emb_a", "emb_b", "len_a", "len_b", "label"}

        # Check shapes (padded to max_len)
        assert sample["emb_a"].shape == (128, 1536)
        assert sample["emb_b"].shape == (128, 1536)

        # Check dtypes
        assert sample["emb_a"].dtype == torch.float32
        assert sample["emb_b"].dtype == torch.float32

        # Check lengths (CLS/EOS removed: 52->50, 102->100)
        assert sample["len_a"] == 50
        assert sample["len_b"] == 100

        # Check label
        assert sample["label"] == 1.0

    def test_dataset_truncation(self, mock_csv_file, mock_embeddings_dict):
        """Test truncation when sequence exceeds max_len."""
        dataset = ProteinPairDataset(
            csv_path=str(mock_csv_file),
            embeddings_dict=mock_embeddings_dict,
            max_len=64,  # Shorter than some sequences
            dtype="fp32",
            strip_cls_eos=True,
        )

        sample = dataset[1]  # PROT_B (100 tokens) and PROT_C (200 tokens)

        # Both should be truncated to max_len
        assert sample["emb_a"].shape == (64, 1536)
        assert sample["emb_b"].shape == (64, 1536)
        assert sample["len_a"] == 64  # Truncated
        assert sample["len_b"] == 64  # Truncated

    def test_dataset_bf16_dtype(self, mock_csv_file, mock_embeddings_dict):
        """Test bf16 dtype conversion."""
        dataset = ProteinPairDataset(
            csv_path=str(mock_csv_file),
            embeddings_dict=mock_embeddings_dict,
            max_len=128,
            dtype="bf16",
        )

        sample = dataset[0]
        assert sample["emb_a"].dtype == torch.bfloat16
        assert sample["emb_b"].dtype == torch.bfloat16

    def test_invalid_dtype(self, mock_csv_file, mock_embeddings_dict):
        """Test error on invalid dtype."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            ProteinPairDataset(
                csv_path=str(mock_csv_file),
                embeddings_dict=mock_embeddings_dict,
                max_len=128,
                dtype="invalid",
            )

    def test_missing_csv_columns(self, tmp_path, mock_embeddings_dict):
        """Test error on missing CSV columns."""
        csv_path = tmp_path / "bad.csv"
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            ProteinPairDataset(
                csv_path=str(csv_path),
                embeddings_dict=mock_embeddings_dict,
                max_len=128,
                dtype="fp32",
            )


class TestBuildLoader:
    """Tests for build_loader function."""

    def test_build_loader_basic(self, mock_csv_file, mock_embeddings_file):
        """Test basic DataLoader creation."""
        loader = build_loader(
            csv_path=str(mock_csv_file),
            embeddings_path=str(mock_embeddings_file),
            batch_size=2,
            max_len=128,
            dtype="fp32",
            ddp=False,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        assert (
            len(loader) == 2
        )  # 3 samples / batch_size 2 = 2 batches (drop_last=False)

        # Fetch one batch
        batch = next(iter(loader))

        assert set(batch.keys()) == {"emb_a", "emb_b", "len_a", "len_b", "label"}
        assert batch["emb_a"].shape[0] == 2  # Batch size
        assert batch["emb_a"].shape[1] == 128  # max_len
        assert batch["emb_a"].shape[2] == 1536  # embedding_dim

    def test_build_loader_caching(self, mock_csv_file, mock_embeddings_file):
        """Test embeddings caching across multiple calls."""
        # First call
        loader1 = build_loader(
            csv_path=str(mock_csv_file),
            embeddings_path=str(mock_embeddings_file),
            batch_size=2,
            max_len=128,
            dtype="fp32",
            ddp=False,
            num_workers=0,
        )

        # Second call with same embeddings_path (should use cache)
        loader2 = build_loader(
            csv_path=str(mock_csv_file),
            embeddings_path=str(mock_embeddings_file),
            batch_size=2,
            max_len=128,
            dtype="fp32",
            ddp=False,
            num_workers=0,
        )

        # Both should work
        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))

        assert batch1["emb_a"].shape == batch2["emb_a"].shape

    def test_build_loader_file_not_found(self, mock_csv_file):
        """Test error on missing embeddings file."""
        with pytest.raises(FileNotFoundError):
            build_loader(
                csv_path=str(mock_csv_file),
                embeddings_path="nonexistent.pkl",
                batch_size=2,
                max_len=128,
                dtype="fp32",
                ddp=False,
                num_workers=0,
            )

    def test_build_loader_iteration(self, mock_csv_file, mock_embeddings_file):
        """Test iterating over full DataLoader."""
        loader = build_loader(
            csv_path=str(mock_csv_file),
            embeddings_path=str(mock_embeddings_file),
            batch_size=2,
            max_len=128,
            dtype="fp32",
            ddp=False,
            shuffle=False,
            num_workers=0,
        )

        batches = list(loader)
        assert len(batches) == 2  # 3 samples / 2 batch_size = 2 batches

        # First batch has 2 samples
        assert batches[0]["emb_a"].shape[0] == 2

        # Last batch has 1 sample (drop_last=False)
        assert batches[1]["emb_a"].shape[0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
