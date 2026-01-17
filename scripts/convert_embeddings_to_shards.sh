#!/bin/bash
#SBATCH -J PREPROCESS
#SBATCH -p critical
#SBATCH -A hexm-critical
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=ai_gpu28
#SBATCH --output=logs/preprocess/slurm_%j.out
#SBATCH --error=logs/preprocess/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

set -e

INPUT=${1:-data/TMP/processed/TMP_embeddings.npz}
OUTPUT_DIR=${2:-__INPLACE__}
MAX_LEN=${3:-768}
DTYPE=${4:-fp16}
SHARD_SIZE=${5:-5000}

if [ -d "$INPUT" ]; then
  echo "Input is already a directory: $INPUT" >&2
  exit 1
fi

PARENT_DIR=$(dirname "$INPUT")
BASENAME=$(basename "$INPUT")
TMP_DIR=""

cleanup() {
  if [ -n "$TMP_DIR" ] && [ -d "$TMP_DIR" ]; then
    rm -rf "$TMP_DIR"
  fi
}
trap cleanup EXIT

if [ "$OUTPUT_DIR" = "__INPLACE__" ]; then
  TMP_DIR=$(mktemp -d "${PARENT_DIR}/${BASENAME}.sharded_tmp.XXXXXX")
  python scripts/convert_embeddings_to_shards.py \
    --input "$INPUT" \
    --output-dir "$TMP_DIR" \
    --max-len "$MAX_LEN" \
    --storage-dtype "$DTYPE" \
    --shard-size "$SHARD_SIZE" \
    --strip-cls-eos

  rm -f "$INPUT"
  mv "$TMP_DIR" "$INPUT"
  TMP_DIR=""
else
  python scripts/convert_embeddings_to_shards.py \
    --input "$INPUT" \
    --output-dir "$OUTPUT_DIR" \
    --max-len "$MAX_LEN" \
    --storage-dtype "$DTYPE" \
    --shard-size "$SHARD_SIZE" \
    --strip-cls-eos
fi
