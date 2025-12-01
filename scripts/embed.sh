#!/bin/bash
#SBATCH -J EMBED
#SBATCH -p hexm_l40
#SBATCH -A hexm
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAL40:4
#SBATCH --output=logs/embed/slurm_%j.out
#SBATCH --error=logs/embed/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

ROOT_DIR="/public/home/wangar2023/DIPPI-DNN-based-Interaction-Predictor-for-Protein-Protein-Interactions"

source ~/.bashrc
conda activate esm

cd "$ROOT_DIR"

# File paths
TMP_CSV="data/TMP/processed/_missing_proteins.csv"
TMP_NPZ="data/TMP/processed/_new_embeddings.npz"

# Step 1: Find missing proteins, create temp CSV
echo "Step 1: Finding missing proteins..."
python scripts/_find_missing.py
STEP1_EXIT=$?

if [ $STEP1_EXIT -ne 0 ]; then
    echo "No missing proteins or error in step 1. Cleaning up..."
    rm -f scripts/_find_missing.py scripts/_merge_embeddings.py "$TMP_CSV" "$TMP_NPZ"
    exit 0
fi

# Check if temp CSV was created (has missing proteins)
if [ ! -f "$TMP_CSV" ]; then
    echo "No missing proteins found. Cleaning up..."
    rm -f scripts/_find_missing.py scripts/_merge_embeddings.py "$TMP_CSV" "$TMP_NPZ"
    exit 0
fi

# Step 2: Run embed.py on missing proteins
echo "Step 2: Embedding missing proteins..."
python src/embed/embed.py --input "$TMP_CSV" --output "$TMP_NPZ"

# Step 3: Merge new embeddings into original NPZ
echo "Step 3: Merging embeddings..."
python scripts/_merge_embeddings.py

# Step 4: Cleanup all temp files
echo "Step 4: Cleaning up temp files..."
rm -f scripts/_find_missing.py scripts/_merge_embeddings.py "$TMP_CSV" "$TMP_NPZ"