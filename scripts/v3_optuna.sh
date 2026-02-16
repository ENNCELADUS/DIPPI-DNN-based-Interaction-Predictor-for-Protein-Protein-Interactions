#!/bin/bash
#SBATCH -J V3_OPTUNA
#SBATCH -p hexm_l40
#SBATCH -A hexm
#SBATCH -N 1
#SBATCH -t 4-00:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAL40:1
#SBATCH --output=logs/v3_optuna/slurm_%j.out
#SBATCH --error=logs/v3_optuna/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <pretrain-checkpoint-path> [extra-optuna-args...]"
  exit 1
fi

CHECKPOINT_PATH="$1"
shift

cd /public/home/wangar2023/DIPPI-DNN-based-Interaction-Predictor-for-Protein-Protein-Interactions
source ~/.bashrc
conda activate esm

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python -m src.tune.finetune_optuna --config configs/v3.yaml --checkpoint "${CHECKPOINT_PATH}" "$@"
