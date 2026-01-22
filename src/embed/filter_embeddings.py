#!/usr/bin/env python3
"""
Filter and extract embeddings for a specific set of proteins.

Reuses embeddings from archive .npz files, extracting only proteins
that match a reference protein list (e.g., from unique_proteins.csv).

Converts embedding shape from (1, L, 1536) to (L, 1536) to match embed.py output.

Usage:
    python src/embed/filter_embeddings.py \
        --protein-list data/TMP/processed/unique_proteins.csv \
        --embedding-file archives/logs/embed/complete_TMP_embeddings.npz \
        --output data/TMP/processed/TMP_embeddings.npz

Input:
    - protein-list: CSV file with 'uniprotID' column
    - embedding-file: .npz file with 'ids' and 'embeddings' keys

Output:
    .npz file with only 'ids' and 'embeddings' keys.
    Embeddings shape: (L, 1536)
    Only includes proteins that exist in both files.
"""

import argparse
import csv
import sys
import numpy as np
from pathlib import Path
from typing import Set, Tuple


def load_protein_list(csv_file: str) -> Set[str]:
    """Load protein IDs from CSV file."""
    protein_ids = set()
    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                print("Error: CSV is empty or missing headers.", file=sys.stderr)
                sys.exit(1)

            if "uniprotID" not in reader.fieldnames:
                print(
                    "Error: CSV must have 'uniprotID' column.",
                    file=sys.stderr,
                )
                sys.exit(1)

            for row in reader:
                uid = row.get("uniprotID", "").strip()
                if uid:
                    protein_ids.add(uid)

        print(f"Loaded {len(protein_ids)} protein IDs from {csv_file}", file=sys.stderr)
        return protein_ids
    except Exception as e:
        print(f"Error loading protein list: {e}", file=sys.stderr)
        sys.exit(1)


def load_embeddings(npz_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load ids and embeddings from a .npz file."""
    try:
        data = np.load(npz_file, allow_pickle=True)
        if "ids" not in data.files or "embeddings" not in data.files:
            print(
                f"Error: {npz_file} missing 'ids' or 'embeddings' key.",
                file=sys.stderr,
            )
            sys.exit(1)

        ids = data["ids"]
        embeddings = data["embeddings"]

        print(f"Loaded {len(ids)} embeddings from {npz_file}", file=sys.stderr)
        return ids, embeddings
    except Exception as e:
        print(f"Error loading embeddings: {e}", file=sys.stderr)
        sys.exit(1)


def squeeze_embedding(emb: np.ndarray) -> np.ndarray:
    """
    Normalize embedding shape.

    Converts (1, L, 1536) to (L, 1536).
    """
    if isinstance(emb, np.ndarray):
        if emb.ndim == 3 and emb.shape[0] == 1:
            emb = emb.squeeze(0)  # (1, L, 1536) -> (L, 1536)
    return emb


def filter_embeddings(
    ids: np.ndarray, embeddings: np.ndarray, target_proteins: Set[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter ids and embeddings to only include target proteins.

    Returns matching ids and embeddings, and reports statistics.
    """
    filtered_ids = []
    filtered_embeddings = []

    for uid, emb in zip(ids, embeddings):
        if uid in target_proteins:
            filtered_ids.append(uid)
            filtered_embeddings.append(squeeze_embedding(emb))

    print(f"Matched {len(filtered_ids)} proteins from embeddings", file=sys.stderr)
    return (
        np.array(filtered_ids, dtype=object),
        np.array(filtered_embeddings, dtype=object),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Filter embeddings to match a protein list."
    )
    parser.add_argument(
        "--protein-list",
        required=True,
        help="CSV file with 'uniprotID' column",
    )
    parser.add_argument(
        "--embedding-file",
        required=True,
        help="Input .npz file with 'ids' and 'embeddings' keys",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .npz file",
    )

    args = parser.parse_args()

    # Load protein list
    target_proteins = load_protein_list(args.protein_list)

    # Load embeddings
    ids, embeddings = load_embeddings(args.embedding_file)

    # Filter
    print("Filtering embeddings...", file=sys.stderr)
    filtered_ids, filtered_embeddings = filter_embeddings(
        ids, embeddings, target_proteins
    )

    # Validate
    if len(filtered_ids) == 0:
        print("Error: No proteins matched!", file=sys.stderr)
        sys.exit(1)

    print(f"Filtered to {len(filtered_ids)} embeddings", file=sys.stderr)

    # Check first embedding shape
    if len(filtered_embeddings) > 0:
        first_emb = filtered_embeddings[0]
        if isinstance(first_emb, np.ndarray):
            print(f"  First embedding shape: {first_emb.shape}", file=sys.stderr)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        ids=filtered_ids,
        embeddings=filtered_embeddings,
    )
    print(f"Saved {len(filtered_ids)} embeddings to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
