#!/usr/bin/env python3
"""
Convert structured ESM embedding .npz into sharded, fixed-length .npy files.

Input format:
  - .npz with arrays: ids, embeddings
  - embeddings may be object dtype or fixed-shape (N, L, D)

Output directory layout:
  - manifest.json
  - index.npz (ids, shard_idx, row_idx, lengths)
  - shard_00000.npy, shard_00001.npy, ...
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Tuple

import numpy as np


def _normalize_id(raw: Any) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    if isinstance(raw, np.str_):
        return str(raw)
    return str(raw)


def _normalize_embedding(
    emb: Any,
    *,
    max_len: int,
    strip_cls_eos: bool,
    target_dtype: np.dtype,
) -> Tuple[np.ndarray, int]:
    if isinstance(emb, np.ndarray) and emb.dtype == object:
        emb = np.array(emb, dtype=np.float32)

    if isinstance(emb, np.ndarray):
        arr = emb
    else:
        arr = np.array(emb, dtype=np.float32)

    if arr.ndim == 3:
        arr = arr.squeeze(0)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embedding array, got shape {arr.shape}")

    if strip_cls_eos and arr.shape[0] > 2:
        arr = arr[1:-1, :]

    length = int(arr.shape[0])
    if length > max_len:
        arr = arr[:max_len, :]
        length = max_len

    if length < max_len:
        pad = np.zeros((max_len - length, arr.shape[1]), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=0)

    if arr.dtype != target_dtype:
        arr = arr.astype(target_dtype, copy=False)

    return arr, length


def _resolve_storage_dtype(name: str) -> Tuple[str, np.dtype]:
    name = name.lower()
    if name == "fp16":
        return "fp16", np.float16
    if name == "fp32":
        return "fp32", np.float32
    if name == "bf16":
        return "fp16", np.float16
    raise ValueError("storage_dtype must be one of: fp16, fp32, bf16")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert embeddings .npz into sharded, fixed-length .npy format."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input .npz with ids + embeddings",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for shards (omit when using --inplace)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Replace the input .npz with a sharded directory in-place",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        required=True,
        help="Fixed max sequence length to pad/truncate to",
    )
    parser.add_argument(
        "--storage-dtype",
        default="fp16",
        help="Storage dtype: fp16 (default) or fp32; bf16 maps to fp16",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=5000,
        help="Number of proteins per shard",
    )
    parser.add_argument(
        "--strip-cls-eos",
        action="store_true",
        help="Remove first/last token before padding",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it exists",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if args.inplace and args.output_dir:
        raise ValueError("--output-dir must be omitted when using --inplace")
    if not args.inplace and not args.output_dir:
        raise ValueError("--output-dir is required unless --inplace is used")

    temp_dir: Path | None = None
    if args.inplace:
        if input_path.is_dir():
            raise ValueError("--inplace expects a .npz file, not a directory")
        temp_dir = Path(
            tempfile.mkdtemp(
                dir=input_path.parent,
                prefix=f"{input_path.name}.sharded_tmp.",
            )
        )
        output_dir = temp_dir
        overwrite = True
    else:
        output_dir = Path(args.output_dir)
        overwrite = args.overwrite

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dir exists: {output_dir} (use --overwrite to replace)"
            )
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    storage_name, storage_dtype = _resolve_storage_dtype(args.storage_dtype)
    if args.storage_dtype.lower() == "bf16":
        print("Note: bf16 storage is not supported by numpy; using fp16 instead.")

    try:
        with np.load(input_path, allow_pickle=True) as data:
            if "ids" not in data or "embeddings" not in data:
                raise ValueError("Input .npz must contain 'ids' and 'embeddings' arrays")

            ids = data["ids"]
            embeddings = data["embeddings"]
            num_items = int(len(ids))
    except Exception:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise

    ids_out = np.empty(num_items, dtype=object)
    shard_idx = np.empty(num_items, dtype=np.int32)
    row_idx = np.empty(num_items, dtype=np.int32)
    lengths = np.empty(num_items, dtype=np.int32)

    shard_size = int(args.shard_size)
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")

    num_shards = int(math.ceil(num_items / shard_size)) if num_items else 0
    embed_dim = None

    for shard_id in range(num_shards):
        start = shard_id * shard_size
        end = min(num_items, start + shard_size)
        count = end - start
        shard_path = output_dir / f"shard_{shard_id:05d}.npy"
        shard_mmap = None

        for local_idx in range(count):
            global_idx = start + local_idx
            protein_id = _normalize_id(ids[global_idx])
            ids_out[global_idx] = protein_id

            emb_raw = embeddings[global_idx]
            emb, length = _normalize_embedding(
                emb_raw,
                max_len=args.max_len,
                strip_cls_eos=args.strip_cls_eos,
                target_dtype=storage_dtype,
            )

            if embed_dim is None:
                embed_dim = int(emb.shape[1])
            if shard_mmap is None:
                shard_mmap = np.lib.format.open_memmap(
                    shard_path,
                    mode="w+",
                    dtype=storage_dtype,
                    shape=(count, args.max_len, embed_dim),
                )
            elif emb.shape[1] != embed_dim:
                raise ValueError(
                    f"Embedding dim mismatch at {protein_id}: {emb.shape[1]} != {embed_dim}"
                )

            shard_mmap[local_idx] = emb
            shard_idx[global_idx] = shard_id
            row_idx[global_idx] = local_idx
            lengths[global_idx] = length

        if shard_mmap is not None:
            del shard_mmap
        print(f"Wrote shard {shard_id + 1}/{num_shards} ({count} proteins)")

    if embed_dim is None:
        raise ValueError("No embeddings found in input")

    np.savez(
        output_dir / "index.npz",
        ids=ids_out,
        shard_idx=shard_idx,
        row_idx=row_idx,
        lengths=lengths,
    )

    manifest = {
        "format": "dippi_sharded_embeddings_v1",
        "max_len": int(args.max_len),
        "embedding_dim": int(embed_dim),
        "storage_dtype": storage_name,
        "shard_size": int(shard_size),
        "num_shards": int(num_shards),
        "strip_cls_eos": bool(args.strip_cls_eos),
        "index_file": "index.npz",
        "shards": [{"file": f"shard_{i:05d}.npy"} for i in range(num_shards)],
    }

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote manifest + index to {output_dir}")

    if args.inplace:
        try:
            input_path.unlink()
            output_dir.rename(input_path)
        except Exception:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise


if __name__ == "__main__":
    main()
