#!/bin/bash
#SBATCH -J V4
#SBATCH -p hexm
#SBATCH -A hexm
#SBATCH -N 1
#SBATCH -t 4-00:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA40:4
#SBATCH --output=logs/v4/slurm_%j.out
#SBATCH --error=logs/v4/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

set -euo pipefail

# Default config path (relative to project root)
CONFIG_PATH="configs/v4.yaml"

# ============================================================
# Setup environment
# ============================================================
# Initialize conda for bash shell
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  # Properly initialize conda function for this shell
  eval "$(conda shell.bash hook)"
elif [[ -f "/public/software/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "/public/software/anaconda3/etc/profile.d/conda.sh"
  eval "$(conda shell.bash hook)"
else
  echo "Error: Cannot find conda.sh (checked: $HOME/miniconda3/etc/profile.d/conda.sh and /public/software/anaconda3/etc/profile.d/conda.sh)" >&2
  exit 1
fi

conda activate esm || { echo "Error: Failed to activate 'esm' environment" >&2; exit 1; }

# Get project root (assume script is in scripts/ subdirectory)
ROOT_DIR="/public/home/wangar2023/DIPPI-DNN-based-Interaction-Predictor-for-Protein-Protein-Interactions"
cd "$ROOT_DIR" || { echo "Error: Cannot access project root: $ROOT_DIR" >&2; exit 1; }

# Resolve config path (make absolute if relative)
if [[ ! "$CONFIG_PATH" = /* ]]; then
  CONFIG_PATH="$ROOT_DIR/$CONFIG_PATH"
fi

# Validate config exists
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

echo "Config: $CONFIG_PATH"
echo "Working directory: $ROOT_DIR"

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

# Check if DDP is enabled in config
DDP_ENABLED=$(python3 - "$CONFIG_PATH" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
ddp = cfg.get("top_level_config", {}).get("ddp", {}).get("enabled", False)
print("true" if ddp else "false")
PY
)

# Get pipeline mode from config
PIPELINE_MODE=$(python3 - "$CONFIG_PATH" <<'PY'
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
