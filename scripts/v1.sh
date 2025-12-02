#!/bin/bash
#SBATCH -J V1
#SBATCH -p critical
#SBATCH -A hexm-critical
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIATITANRTX:3
#SBATCH --exclude=ai_gpu28
#SBATCH --output=logs/v1/slurm_%j.out
#SBATCH --error=logs/v1/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

set -euo pipefail

ROOT_DIR="/public/home/wangar2023/DIPPI-DNN-based-Interaction-Predictor-for-Protein-Protein-Interactions"
CONFIG_PATH="$ROOT_DIR/configs/v1.yaml"

echo "[INFO] Using config: $CONFIG_PATH"
echo "[INFO] Project root: $ROOT_DIR"

if [[ -f "$HOME/.bashrc" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.bashrc"
fi
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate esm

cd "$ROOT_DIR"

# Export environment variables
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ============================================================
# Detect GPUs and check DDP configuration
# ============================================================
# Detect number of GPUs
if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  NUM_GPUS="$SLURM_GPUS_ON_NODE"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a gpu_arr <<< "$CUDA_VISIBLE_DEVICES"
  NUM_GPUS="${#gpu_arr[@]}"
else
  NUM_GPUS=1
fi
echo "GPUs available: $NUM_GPUS"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE:-<not set>}"

# Quick CUDA sanity check before proceeding
if ! python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')" 2>/dev/null; then
  echo "[WARN] CUDA sanity check failed - GPUs may not be properly initialized"
fi

# Check if DDP is enabled in config
# NOTE: Use CUDA_VISIBLE_DEVICES="" to prevent accidental CUDA init during config parsing
DDP_ENABLED=$(CUDA_VISIBLE_DEVICES="" python3 - "$CONFIG_PATH" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
ddp = cfg.get("top_level_config", {}).get("ddp", {}).get("enabled", False)
print("true" if ddp else "false")
PY
)

# Get pipeline mode from config
PIPELINE_MODE=$(CUDA_VISIBLE_DEVICES="" python3 - "$CONFIG_PATH" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
mode = cfg.get("run_config", {}).get("mode", "full_pipeline")
print(mode)
PY
)
echo "Pipeline mode: $PIPELINE_MODE"

# ============================================================
# Launch pipeline
# ============================================================
if [[ "$DDP_ENABLED" == "true" ]]; then
  if [[ "$NUM_GPUS" -lt 2 ]]; then
    echo "Error: DDP enabled but only $NUM_GPUS GPU(s) available. Need >=2 GPUs." >&2
    exit 1
  fi
  
  echo "Launching with DDP (torchrun, $NUM_GPUS GPUs)..."
  
  # Setup torchrun environment
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MASTER_PORT:-29500}"
  
  torchrun --standalone --nproc_per_node="$NUM_GPUS" -m src.run "$CONFIG_PATH"
else
  echo "Launching single-process (python)..."
  python3 -m src.run "$CONFIG_PATH"
fi

echo "Pipeline completed successfully!"