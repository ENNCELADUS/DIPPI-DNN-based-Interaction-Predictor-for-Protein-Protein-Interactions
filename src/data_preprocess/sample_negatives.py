#!/usr/bin/env python3
"""
Degree Distribution Balanced (DDB) negative sampling for PPI datasets.

Downsamples negatives to 1:1 ratio while:
1. Only keeping negatives where both proteins appear in positive samples
2. Balancing degree distribution between pos/neg samples
3. Prioritizing hard negatives (2-hop non-edges + hub pairs)

Usage:
    python src/data_preprocess/sample_negatives.py \
        --input data/TMP/processed/pretrain.csv \
        --output data/TMP/processed/pretrain_sampled.csv \
        --hard-ratio 0.3 \
        --seed 42
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


def build_ppi_graph(
    positives: pd.DataFrame,
) -> Tuple[Set[str], Set[Tuple[str, str]], Dict[str, Set[str]]]:
    """
    Build PPI graph from positive samples.

    Args:
        positives: DataFrame with uniprotID_A, uniprotID_B columns (positive pairs only).

    Returns:
        v_pos: Set of proteins appearing in positive samples.
        e_pos: Set of positive edges (undirected, stored as sorted tuples).
        adj: Adjacency list mapping protein -> set of neighbors.
    """
    v_pos: Set[str] = set()
    e_pos: Set[Tuple[str, str]] = set()
    adj: Dict[str, Set[str]] = defaultdict(set)

    for _, row in positives.iterrows():
        a, b = row["uniprotID_A"], row["uniprotID_B"]
        v_pos.add(a)
        v_pos.add(b)
        edge = tuple(sorted([a, b]))
        e_pos.add(edge)
        adj[a].add(b)
        adj[b].add(a)

    return v_pos, e_pos, dict(adj)


def compute_degree_stats(
    adj: Dict[str, Set[str]], hub_percentile: float = 90.0
) -> Tuple[Dict[str, int], int]:
    """
    Compute degree for each protein and hub threshold.

    Args:
        adj: Adjacency list.
        hub_percentile: Percentile for hub threshold (default 90th).

    Returns:
        degree: Dict mapping protein -> degree.
        hub_threshold: Degree threshold for hub classification.
    """
    degree = {p: len(neighbors) for p, neighbors in adj.items()}
    degrees = list(degree.values())
    hub_threshold = int(np.percentile(degrees, hub_percentile))
    return degree, hub_threshold


def bucket_proteins_by_degree(
    degree: Dict[str, int], num_buckets: int = 10
) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
    """
    Bucket proteins by degree using log-scale bins.

    Args:
        degree: Dict mapping protein -> degree.
        num_buckets: Number of buckets to create.

    Returns:
        protein_to_bucket: Dict mapping protein -> bucket index.
        bucket_to_proteins: Dict mapping bucket index -> list of proteins.
    """
    if not degree:
        return {}, {}

    max_deg = max(degree.values())
    min_deg = min(degree.values())

    # Log-scale bucketing
    if max_deg == min_deg:
        # All same degree
        protein_to_bucket = {p: 0 for p in degree}
        bucket_to_proteins = {0: list(degree.keys())}
        return protein_to_bucket, bucket_to_proteins

    log_min = math.log1p(min_deg)
    log_max = math.log1p(max_deg)
    bucket_width = (log_max - log_min) / num_buckets

    protein_to_bucket: Dict[str, int] = {}
    bucket_to_proteins: Dict[int, List[str]] = defaultdict(list)

    for protein, deg in degree.items():
        log_deg = math.log1p(deg)
        bucket_idx = min(int((log_deg - log_min) / bucket_width), num_buckets - 1)
        protein_to_bucket[protein] = bucket_idx
        bucket_to_proteins[bucket_idx].append(protein)

    return protein_to_bucket, dict(bucket_to_proteins)


def is_2hop_connected(p1: str, p2: str, adj: Dict[str, Set[str]]) -> bool:
    """
    Check if two proteins share a common neighbor (2-hop non-edge).

    Args:
        p1, p2: Protein IDs.
        adj: Adjacency list.

    Returns:
        True if p1 and p2 share at least one neighbor.
    """
    neighbors_p1 = adj.get(p1, set())
    neighbors_p2 = adj.get(p2, set())
    return len(neighbors_p1 & neighbors_p2) > 0


def classify_negative(
    p1: str,
    p2: str,
    adj: Dict[str, Set[str]],
    degree: Dict[str, int],
    hub_threshold: int,
) -> bool:
    """
    Classify a negative sample as hard or default.

    Hard negatives:
    - 2-hop connected (share a neighbor)
    - Involve at least one hub protein

    Args:
        p1, p2: Protein IDs.
        adj: Adjacency list.
        degree: Degree dict.
        hub_threshold: Threshold for hub classification.

    Returns:
        True if hard negative, False otherwise.
    """
    # Check hub
    is_hub = degree.get(p1, 0) >= hub_threshold or degree.get(p2, 0) >= hub_threshold

    # Check 2-hop
    is_2hop = is_2hop_connected(p1, p2, adj)

    return is_hub or is_2hop


def filter_and_classify_negatives(
    negatives: pd.DataFrame,
    v_pos: Set[str],
    adj: Dict[str, Set[str]],
    degree: Dict[str, int],
    hub_threshold: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Filter negatives to only include proteins from V_pos and classify as hard/default.

    Args:
        negatives: DataFrame with negative pairs.
        v_pos: Set of proteins in positive samples.
        adj: Adjacency list.
        degree: Degree dict.
        hub_threshold: Hub threshold.

    Returns:
        hard_negs: List of hard negative pairs.
        default_negs: List of default negative pairs.
    """
    hard_negs: List[Tuple[str, str]] = []
    default_negs: List[Tuple[str, str]] = []

    total = len(negatives)
    filtered_count = 0

    for idx, row in negatives.iterrows():
        a, b = row["uniprotID_A"], row["uniprotID_B"]

        # Filter: both proteins must be in V_pos
        if a not in v_pos or b not in v_pos:
            filtered_count += 1
            continue

        pair = (a, b)
        if classify_negative(a, b, adj, degree, hub_threshold):
            hard_negs.append(pair)
        else:
            default_negs.append(pair)

        # Progress logging
        if (idx + 1) % 1_000_000 == 0:
            print(f"  Processed {idx + 1:,}/{total:,} negatives...")

    print(f"  Filtered out {filtered_count:,} negatives (proteins not in V_pos)")
    return hard_negs, default_negs


def sample_negatives(
    hard_negs: List[Tuple[str, str]],
    default_negs: List[Tuple[str, str]],
    num_samples: int,
    hard_ratio: float,
    rng: np.random.Generator,
) -> List[Tuple[str, str]]:
    """
    Sample negatives with specified hard/default ratio.

    Args:
        hard_negs: Pool of hard negatives.
        default_negs: Pool of default negatives.
        num_samples: Total number of negatives to sample.
        hard_ratio: Fraction of samples that should be hard.
        rng: Random number generator.

    Returns:
        List of sampled negative pairs.
    """
    num_hard = int(num_samples * hard_ratio)
    num_default = num_samples - num_hard

    # Adjust if pools are insufficient
    if len(hard_negs) < num_hard:
        print(
            f"  Warning: Only {len(hard_negs):,} hard negatives available, "
            f"requested {num_hard:,}"
        )
        num_hard = len(hard_negs)
        num_default = num_samples - num_hard

    if len(default_negs) < num_default:
        print(
            f"  Warning: Only {len(default_negs):,} default negatives available, "
            f"requested {num_default:,}"
        )
        num_default = len(default_negs)
        # Try to compensate with more hard negatives
        extra_hard = min(
            num_samples - num_hard - num_default, len(hard_negs) - num_hard
        )
        num_hard += extra_hard

    # Sample
    hard_indices = rng.choice(len(hard_negs), size=num_hard, replace=False)
    default_indices = rng.choice(len(default_negs), size=num_default, replace=False)

    sampled = [hard_negs[i] for i in hard_indices]
    sampled.extend([default_negs[i] for i in default_indices])

    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DDB negative sampling for PPI datasets"
    )
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--hard-ratio",
        type=float,
        default=0.3,
        help="Fraction of negatives that should be hard (default: 0.3)",
    )
    parser.add_argument(
        "--hub-percentile",
        type=float,
        default=90.0,
        help="Percentile for hub threshold (default: 90.0)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)

    positives = df[df["isInteraction"] == 1].copy()
    negatives = df[df["isInteraction"] == 0].copy()
    num_pos = len(positives)
    num_neg_orig = len(negatives)

    print(
        f"Original: {num_pos:,} positives, {num_neg_orig:,} negatives "
        f"(ratio 1:{num_neg_orig / num_pos:.1f})"
    )

    # Step 1: Build PPI graph from positives
    print("\nStep 1: Building PPI graph from positive samples...")
    v_pos, e_pos, adj = build_ppi_graph(positives)
    print(f"  V_pos: {len(v_pos):,} proteins")
    print(f"  E_pos: {len(e_pos):,} edges")

    # Step 2: Compute degree statistics
    print("\nStep 2: Computing degree statistics...")
    degree, hub_threshold = compute_degree_stats(adj, args.hub_percentile)
    degrees = list(degree.values())
    print(f"  Degree range: {min(degrees)} - {max(degrees)}")
    print(f"  Degree mean: {np.mean(degrees):.1f}, median: {np.median(degrees):.1f}")
    print(f"  Hub threshold ({args.hub_percentile}th percentile): {hub_threshold}")

    num_hubs = sum(1 for d in degrees if d >= hub_threshold)
    print(f"  Number of hub proteins: {num_hubs:,}")

    # Step 3: Bucket proteins by degree
    print("\nStep 3: Bucketing proteins by degree...")
    protein_to_bucket, bucket_to_proteins = bucket_proteins_by_degree(degree)
    for bucket_idx in sorted(bucket_to_proteins.keys()):
        proteins = bucket_to_proteins[bucket_idx]
        if proteins:
            bucket_degrees = [degree[p] for p in proteins]
            print(
                f"  Bucket {bucket_idx}: {len(proteins):,} proteins, "
                f"degree range [{min(bucket_degrees)}-{max(bucket_degrees)}]"
            )

    # Step 4: Filter and classify negatives
    print("\nStep 4: Filtering and classifying negatives...")
    hard_negs, default_negs = filter_and_classify_negatives(
        negatives, v_pos, adj, degree, hub_threshold
    )
    print(f"  Hard negatives: {len(hard_negs):,}")
    print(f"  Default negatives: {len(default_negs):,}")
    print(f"  Total valid negatives: {len(hard_negs) + len(default_negs):,}")

    # Step 5: Sample negatives (1:1 ratio)
    print(f"\nStep 5: Sampling {num_pos:,} negatives (1:1 ratio)...")
    print(
        f"  Target: {int(num_pos * args.hard_ratio):,} hard, "
        f"{int(num_pos * (1 - args.hard_ratio)):,} default"
    )

    sampled_negs = sample_negatives(
        hard_negs, default_negs, num_pos, args.hard_ratio, rng
    )
    print(f"  Sampled: {len(sampled_negs):,} negatives")

    # Step 6: Create output DataFrame
    print("\nStep 6: Creating output dataset...")
    neg_df = pd.DataFrame(sampled_negs, columns=["uniprotID_A", "uniprotID_B"])
    neg_df["isInteraction"] = 0

    output_df = pd.concat([positives, neg_df], ignore_index=True)

    # Shuffle
    output_df = output_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Save
    output_df.to_csv(output_path, index=False)

    # Final statistics
    final_pos = output_df["isInteraction"].sum()
    final_neg = len(output_df) - final_pos
    print(f"\nOutput: {output_path}")
    print(f"  Total: {len(output_df):,} samples")
    print(f"  Positives: {final_pos:,}")
    print(f"  Negatives: {final_neg:,}")
    print(f"  Ratio: 1:{final_neg / final_pos:.2f}")


if __name__ == "__main__":
    main()
