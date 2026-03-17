#!/bin/bash
#SBATCH -J ML
#SBATCH -p critical
#SBATCH -A hexm-critical
#SBATCH -N 1
#SBATCH -t 1-00:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/ml/slurm_%j.out
#SBATCH --error=logs/ml/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

set -euo pipefail

CONFIG_PATH="${1:-src_ml/ml.yaml}"

cd /public/home/wangar2023/DIPPI-DNN-based-Interaction-Predictor-for-Protein-Protein-Interactions
source ~/.bashrc
conda activate esm

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m src.data_preprocess.prepare_tppni_datasets --config "${CONFIG_PATH}"
python -m src_ml.run --config "${CONFIG_PATH}"
