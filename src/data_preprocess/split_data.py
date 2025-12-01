#!/usr/bin/env python3
"""
Stratified train/val split for protein-protein interaction datasets.

Splits data by class (positive/negative) to preserve original label ratios.
Val set keeps true long-tail distribution for realistic evaluation.

Usage:
    python scripts/split_data.py --input data/TMP/processed/pretrain.csv --train-ratio 0.95
    python scripts/split_data.py --input data/TMP/processed/finetune.csv --train-ratio 0.90
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float,
    seed: int,
    label_col: str = "isInteraction",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified train/val split preserving class ratios.

    Args:
        df: DataFrame with protein pairs and labels.
        train_ratio: Fraction for training (e.g., 0.95).
        seed: Random seed for reproducibility.
        label_col: Column name for labels.

    Returns:
        (train_df, val_df) with shuffled rows.
    """
    # Separate by class and shuffle
    pos = (
        df[df[label_col] == 1].sample(frac=1, random_state=seed).reset_index(drop=True)
    )
    neg = (
        df[df[label_col] == 0].sample(frac=1, random_state=seed).reset_index(drop=True)
    )

    # Compute split indices
    pos_train_cut = int(len(pos) * train_ratio)
    neg_train_cut = int(len(neg) * train_ratio)

    # Split each class
    train_df = pd.concat(
        [pos.iloc[:pos_train_cut], neg.iloc[:neg_train_cut]], ignore_index=True
    )
    val_df = pd.concat(
        [pos.iloc[pos_train_cut:], neg.iloc[neg_train_cut:]], ignore_index=True
    )

    # Shuffle final splits
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return train_df, val_df


def print_stats(name: str, df: pd.DataFrame, label_col: str = "isInteraction") -> None:
    """Print dataset statistics."""
    n_total = len(df)
    n_pos = int(df[label_col].sum())
    n_neg = n_total - n_pos
    ratio = n_neg / n_pos if n_pos > 0 else float("inf")
    print(
        f"  {name}: {n_total:,} total | {n_pos:,} pos | {n_neg:,} neg | ratio 1:{ratio:.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified train/val split")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument(
        "--train-ratio", type=float, required=True, help="Train ratio (e.g., 0.95)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir", default=None, help="Output directory (default: same as input)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive output names from input stem
    stem = input_path.stem  # e.g., "pretrain" or "finetune"
    train_path = output_dir / f"{stem}_train.csv"
    val_path = output_dir / f"{stem}_val.csv"

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)

    print(f"Splitting with train_ratio={args.train_ratio}, seed={args.seed}")
    train_df, val_df = stratified_split(df, args.train_ratio, args.seed)

    # Print statistics
    print("\nDataset statistics:")
    print_stats("Original", df)
    print_stats("Train", train_df)
    print_stats("Val", val_df)

    # Save
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    print(f"\nSaved: {train_path}")
    print(f"Saved: {val_path}")


if __name__ == "__main__":
    main()
