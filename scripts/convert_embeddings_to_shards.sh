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
MAX_LEN=${2:-768}
DTYPE=${3:-fp16}
SHARD_SIZE=${4:-5000}

python scripts/convert_embeddings_to_shards.py \
  --input "$INPUT" \
  --inplace \
  --max-len "$MAX_LEN" \
  --storage-dtype "$DTYPE" \
  --shard-size "$SHARD_SIZE" \
  --strip-cls-eos
