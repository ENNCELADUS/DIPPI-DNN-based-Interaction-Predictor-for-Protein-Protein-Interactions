#!/bin/bash
#SBATCH -J EMBED
#SBATCH -p hexm
#SBATCH -A hexm
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA40:4
#SBATCH --output=logs/embed/slurm_%j.out
#SBATCH --error=logs/embed/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

ROOT_DIR="/public/home/wangar2023/DIPPI-DNN-based-Interaction-Predictor-for-Protein-Protein-Interactions"
cd "$ROOT_DIR" || { echo "Error: Cannot access project root: $ROOT_DIR" >&2; exit 1; }

if [[ -f "$HOME/.bashrc" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.bashrc"
fi
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate esm

export EMBED_WORKSPACE="$ROOT_DIR"
export EMBED_DATA_ROOT="${EMBED_DATA_ROOT:-$ROOT_DIR/data/embed}"
export EMBED_CACHE_ROOT="${EMBED_CACHE_ROOT:-$ROOT_DIR/.cache}"
export EMBED_MODEL_CACHE="${EMBED_MODEL_CACHE:-$EMBED_CACHE_ROOT/models}"
export EMBED_MODEL_NAME="${EMBED_MODEL_NAME:-esm3_sm_open_v1}"
export EMBED_USE_LOCAL_MODEL="${EMBED_USE_LOCAL_MODEL:-true}"
export EMBED_DEVICE="${EMBED_DEVICE:-cuda}"
export EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-1}"
export EMBED_MAX_SEQUENCE_LENGTH="${EMBED_MAX_SEQUENCE_LENGTH:-4096}"
TRUNCATE_LENGTH="${TRUNCATE_LENGTH:-2048}"

INPUT_CSV="${INPUT_CSV:-$ROOT_DIR/data/TMP_protein/processed/unique_proteins.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/data/embedding}"
OUTPUT_FILE="${OUTPUT_FILE:-complete_TMP_embeddings.npz}"
OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT_FILE"

mkdir -p "$OUTPUT_DIR" logs/embed

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "Error: CSV input not found: $INPUT_CSV" >&2
  exit 1
fi

echo "Running sequence-only embedding:"
echo "  Input CSV : $INPUT_CSV"
echo "  Output NPZ: $OUTPUT_PATH"
echo "  Data root : $EMBED_DATA_ROOT"

set -euo pipefail

python src/embed/run.py \
  "$INPUT_CSV" \
  "$OUTPUT_PATH" \
  --input-format csv \
  --csv-id-column uniprotID \
  --csv-sequence-column sequence \
  --mode sequence \
  --retry-truncate-errors \
  --truncate-retry-length "$TRUNCATE_LENGTH"

echo "Embedding complete: $OUTPUT_PATH"
