"""
Integration tests for data loader sampling strategies.
"""

import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.utils.data_io import build_dataloaders
from src.utils.samplers import ImbalancedBatchSampler


def _make_embeddings(proteins):
    """Create minimal embedding dict for provided protein IDs."""
    return {pid: torch.randn(1, 8, 1536) for pid in proteins}


@pytest.mark.parametrize("stage", ["pretrain", "finetune"])
def test_build_dataloaders_uses_imbalanced_sampler(tmp_path: Path, stage: str):
    """Ensure build_dataloaders wires ImbalancedBatchSampler when configured."""
    csv_path = tmp_path / f"{stage}_train.csv"
    val_path = tmp_path / f"{stage}_val.csv"
    proteins = [f"P{i:05d}" for i in range(10)]

    df = pd.DataFrame(
        {
            "uniprotID_A": proteins[:10],
            "uniprotID_B": proteins[::-1],
            "isInteraction": [1] * 6 + [0] * 4,
        }
    )
    df.to_csv(csv_path, index=False)
    df.to_csv(val_path, index=False)

    emb_index = _make_embeddings(proteins)

    cfg = {
        "run_config": {"seed": 99},
        "data_config": {
            "embeddings_path": "unused",
            "embedding_dtype": "fp32",
            "max_sequence_length": 64,
            stage: {
                "train_csv": str(csv_path),
                "valid_csv": str(val_path),
                "sampling": {"strategy": "imbalanced", "pos_neg_ratio": 3.0},
            },
            "dataloader": {"num_workers": 0, "pin_memory": False},
        },
        f"{stage}_config": {"batch_size": 8},
    }

    loaders = build_dataloaders(cfg, emb_index, {f"{stage}_train"})
    loader = loaders[f"{stage}_train"]

    assert isinstance(loader.batch_sampler, ImbalancedBatchSampler)
    sampler = loader.batch_sampler
    assert sampler.pos_per_batch == 2  # 8 / (1 + 3)
    assert len(sampler) == math.ceil(6 / sampler.pos_per_batch)
