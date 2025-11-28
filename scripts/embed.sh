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

python src/embed/embed.py --input data/TMP/processed/unique_proteins.csv --output data/TMP/processed/esm3_embedding_2048.npz --max-length 2048
