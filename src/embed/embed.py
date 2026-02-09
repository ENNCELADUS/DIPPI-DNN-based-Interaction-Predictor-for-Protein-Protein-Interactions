#!/usr/bin/env python3
"""
Lightweight ESM3 embedding generator.
Replaces the previous complex embed module.

Usage:
    python src/embed.py --input data/sequences.csv --output data/embeddings.npz

Input:
    CSV file with columns 'uniprotID' and 'sequence'.

Output:
    .npz file with 'ids' and 'embeddings' arrays.
    Compatible with src.utils.data.io.ProteinPairDataset.
"""

import argparse
import csv
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Any

try:
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, LogitsConfig
except ImportError:
    print("Error: 'esm' package not installed.", file=sys.stderr)
    sys.exit(1)


def clean_protein_sequence(sequence: str) -> str:
    """Canonicalize protein sequence."""
    if not sequence:
        return ""
    cleaned = sequence.upper()
    cleaned = cleaned.replace("-", "").replace(".", "").replace("*", "")
    return cleaned


def load_model(model_name: str = "esm3_sm_open_v1", device: str = "auto") -> Any:
    """Load the ESM3 model."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if (
            device == "cpu"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = "mps"

    print(f"Loading model {model_name} on {device}...", file=sys.stderr)
    model = ESM3.from_pretrained(model_name).to(device)
    model.eval()
    return model, device


def embed_sequence(model: Any, sequence: str, device: str) -> np.ndarray:
    """Embed a single sequence using ESM3."""
    protein = ESMProtein(sequence=sequence)
    logits_config = LogitsConfig(sequence=True, return_embeddings=True)

    with torch.no_grad():
        protein_tensor = model.encode(protein)
        sequence_output = model.logits(protein_tensor, logits_config)
        embeddings = sequence_output.embeddings
        if embeddings is None:
            raise ValueError("Model did not return embeddings.")

    return embeddings.detach().cpu().numpy().squeeze(0)  # (L, D)


def main():
    parser = argparse.ArgumentParser(description="Generate ESM3 embeddings.")
    parser.add_argument(
        "--input", required=True, help="Input CSV file (uniprotID, sequence)"
    )
    parser.add_argument("--output", required=True, help="Output .npz file")
    parser.add_argument("--model", default="esm3_sm_open_v1", help="ESM3 model name")
    parser.add_argument(
        "--device", default="auto", help="Device (cpu, cuda, mps, auto)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Max sequence length (truncate if longer)",
    )

    args = parser.parse_args()

    # Load inputs
    items = []
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found.", file=sys.stderr)
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("Error: CSV is empty or missing headers.", file=sys.stderr)
            sys.exit(1)

        if "uniprotID" not in reader.fieldnames or "sequence" not in reader.fieldnames:
            print(
                "Error: CSV must have 'uniprotID' and 'sequence' columns.",
                file=sys.stderr,
            )
            sys.exit(1)

        for row in reader:
            uid = row.get("uniprotID", "").strip()
            seq = row.get("sequence", "").strip()
            if uid and seq:
                items.append((uid, seq))

    print(f"Processing {len(items)} sequences...", file=sys.stderr)

    # Load model
    model, device = load_model(args.model, args.device)

    # Process
    ids = []
    embeddings = []

    for uid, seq in items:
        clean_seq = clean_protein_sequence(seq)
        if not clean_seq:
            continue

        if len(clean_seq) > args.max_length:
            clean_seq = clean_seq[: args.max_length]

        try:
            emb = embed_sequence(model, clean_seq, device)
            ids.append(uid)
            embeddings.append(emb)

            if len(ids) % 10 == 0:
                print(f"Processed {len(ids)}/{len(items)}...", file=sys.stderr)
        except Exception as e:
            print(f"Error embedding {uid}: {e}", file=sys.stderr)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        ids=np.array(ids, dtype=object),
        embeddings=np.array(embeddings, dtype=object),
    )
    print(f"Saved {len(ids)} embeddings to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
