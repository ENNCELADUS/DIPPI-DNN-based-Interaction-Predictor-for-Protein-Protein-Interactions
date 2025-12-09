#!/usr/bin/env python3
"""
Negative downsampling that only keeps negatives where both proteins appear in
positive samples. This is a lightweight alternative to the DDB sampler that
reduces shortcut-prone negatives (proteins never seen in positives).

Usage:
    python src/data_preprocess/sample_negatives_pos_only.py \\
        --input data/TMP/raw/finetune.csv \\
        --output data/TMP/processed/finetune.csv \\
        --seed 42

Optional:
    --target-ratio 1.0  # If provided, downsample negatives to ratio; otherwise keep all filtered negatives.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Set

import numpy as np
import pandas as pd


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a PPI CSV with expected columns."""
    df = pd.read_csv(path)
    missing_cols = {"uniprotID_A", "uniprotID_B", "isInteraction"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input file missing columns: {missing_cols}")
    return df


def filter_negatives_by_positive_proteins(
    negatives: pd.DataFrame, positive_proteins: Set[str]
) -> pd.DataFrame:
    """Keep only negatives where both proteins appeared in any positive pair."""
    mask = negatives["uniprotID_A"].isin(positive_proteins) & negatives[
        "uniprotID_B"
    ].isin(positive_proteins)
    return negatives[mask].reset_index(drop=True)


def sample_negatives(
    negatives: pd.DataFrame, target_count: int | None, rng: np.random.Generator
) -> pd.DataFrame:
    """Downsample negatives to the requested count if provided."""
    if target_count is None or target_count >= len(negatives):
        return negatives.reset_index(drop=True)

    # pandas sample requires an int seed; draw from Generator for reproducibility
    seed = int(rng.integers(low=0, high=2**32 - 1))
    return negatives.sample(n=target_count, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Downsample negatives by first filtering to proteins seen in positives."
        )
    )
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=None,
        help=(
            "Optional negative-to-positive ratio after filtering. "
            "If omitted, keeps all filtered negatives."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    if args.target_ratio is not None and args.target_ratio <= 0:
        raise ValueError("--target-ratio must be positive")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    print(f"Loading {input_path}...")
    df = load_dataset(input_path)

    positives = df[df["isInteraction"] == 1].reset_index(drop=True)
    negatives = df[df["isInteraction"] == 0].reset_index(drop=True)

    num_pos = len(positives)
    num_neg = len(negatives)
    print(f"Original: {num_pos:,} positives, {num_neg:,} negatives (1:{num_neg/num_pos:.1f})")

    print("\nFiltering negatives to proteins present in positives...")
    positive_proteins: Set[str] = set(positives["uniprotID_A"]).union(
        positives["uniprotID_B"]
    )
    filtered_negatives = filter_negatives_by_positive_proteins(
        negatives, positive_proteins
    )
    print(
        f"  Filtered negatives: {len(filtered_negatives):,} "
        f"(removed {num_neg - len(filtered_negatives):,})"
    )

    target_negatives = (
        int(num_pos * args.target_ratio) if args.target_ratio is not None else None
    )
    print(
        f"\nSampling negatives to target count: "
        f"{target_negatives if target_negatives is not None else 'all available'}"
    )
    sampled_negatives = sample_negatives(filtered_negatives, target_negatives, rng)
    print(f"  Sampled negatives: {len(sampled_negatives):,}")

    print("\nBuilding output dataset...")
    sampled_negatives = sampled_negatives.copy()
    sampled_negatives["isInteraction"] = 0

    output_df = pd.concat([positives, sampled_negatives], ignore_index=True)
    output_df = output_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    output_df.to_csv(output_path, index=False)

    final_pos = output_df["isInteraction"].sum()
    final_neg = len(output_df) - final_pos
    print(f"\nSaved to {output_path}")
    print(f"  Total: {len(output_df):,}")
    print(f"  Positives: {final_pos:,}")
    print(f"  Negatives: {final_neg:,}")
    print(f"  Ratio: 1:{final_neg / final_pos:.2f}")


if __name__ == "__main__":
    main()
