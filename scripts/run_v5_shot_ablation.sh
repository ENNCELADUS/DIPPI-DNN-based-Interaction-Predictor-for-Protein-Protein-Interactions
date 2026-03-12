#!/bin/bash
#SBATCH -J V5
#SBATCH -p hexm
#SBATCH -A hexm
#SBATCH -N 1
#SBATCH -t 4-00:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA40:4
#SBATCH --output=logs/v5/slurm_%j.out
#SBATCH --error=logs/v5/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

set -euo pipefail

cd /public/home/wangar2023/DIPPI-DNN-based-Interaction-Predictor-for-Protein-Protein-Interactions
source ~/.bashrc
conda activate esm

if command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_GPUS="$(nvidia-smi -L | wc -l | tr -d '[:space:]')"
else
  DETECTED_GPUS="1"
fi
NGPUS="${NGPUS:-${DETECTED_GPUS}}"

CONFIGS=(
  "configs/experiments/v5_shot/v5_shot_lr_5e-5.yaml"
  "configs/experiments/v5_shot/v5_shot_lr_5e-6.yaml"
  "configs/experiments/v5_shot/v5_shot_align_0p3_prior_0p023_lr_5e-5.yaml"
  "configs/experiments/v5_shot/v5_shot_align_3p0_prior_0p023_lr_5e-5.yaml"
  "configs/experiments/v5_shot/v5_shot_tau_0p93_0p995_lr_5e-5.yaml"
  "configs/experiments/v5_shot/v5_shot_ent_0p3_prior_0p023_lr_5e-5.yaml"
  "configs/experiments/v5_shot/v5_shot_ent_0p1_align_3p0_prior_0p023_lr_5e-5.yaml"
)

for config_path in "${CONFIGS[@]}"; do
  if [[ ! -f "${config_path}" ]]; then
    echo "missing config: ${config_path}" >&2
    exit 1
  fi
done

echo "Running ${#CONFIGS[@]} v5 SHOT ablations with NGPUS=${NGPUS}"
for config_path in "${CONFIGS[@]}"; do
  run_name="$(basename "${config_path}" .yaml)"
  echo "[start] ${run_name}"
  python -m src.data_preprocess.prepare_tppni_datasets --config "${config_path}"
  torchrun --standalone --nproc_per_node="${NGPUS}" -m src.run -- --config "${config_path}"
  echo "[done]  ${run_name}"
done
