#!/bin/bash
#SBATCH -J TEST
#SBATCH -p critical
#SBATCH -A hexm-critical
#SBATCH -N 1
#SBATCH -t 4-00:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIATITANRTX:4
#SBATCH --output=tests/e2e/artifacts/slurm_%j.out
#SBATCH --error=tests/e2e/artifacts/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

set -euo pipefail

# cd /public/home/wangar2023/DIPPI-DNN-based-Interaction-Predictor-for-Protein-Protein-Interactions
# source ~/.bashrc
# conda activate esm

# # Automatically detect number of GPUs from SLURM allocation
# NGPUS=$(nvidia-smi -L | wc -l)
# echo "Detected $NGPUS GPUs"

# export PYTHONPATH="$PWD:${PYTHONPATH:-}"
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

torchrun --standalone --nproc_per_node=$NGPUS -m src.run tests/e2e/config/test.yaml