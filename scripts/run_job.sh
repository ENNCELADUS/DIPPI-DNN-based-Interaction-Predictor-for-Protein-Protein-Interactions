#!/bin/bash
#SBATCH -J TeslaM40-DIPPI-Training
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH --mem=64G          # Reasonable RAM for Tesla M40 node
#SBATCH --cpus-per-task=12 # CPUs for data loading (node has 56 total)
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:TeslaM4024GB:1  # Request 1 Tesla M40 GPU (24GB VRAM each)
#SBATCH --mail-user=2162352828@qq.com
# sleep 9999999
source ~/.bashrc
conda activate esm
cd /public/home/wangar2023/DIPPI-DNN-based-Interaction-Predictor-for-Protein-Protein-Interactions/ || { echo "目录不存在"; exit 1; }

# ⚡ TESLA M40 PERFORMANCE OPTIMIZATIONS
echo "=== SETTING TESLA M40 OPTIMIZATIONS ==="

# CUDA Performance Settings for Tesla M40
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0  # Async CUDA ops

# PyTorch Performance Settings for Tesla M40 (Maxwell architecture)
export TORCH_CUDA_ARCH_LIST="5.2"  # Tesla M40 Maxwell architecture
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"  # Better memory management for 24GB VRAM

# CPU Performance Settings
export OMP_NUM_THREADS=12  # Match cpus-per-task
export MKL_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12

# Data Loading Optimization
export PYTHONUNBUFFERED=1  # Faster stdout
export TORCH_CUDNN_V8_API_ENABLED=1  # Use cuDNN v8 API

# Memory Optimization for Tesla M40
ulimit -n 65536  # Increase file descriptor limit
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"  # Reduce memory fragmentation

echo "Tesla M40 optimizations applied ✅"
echo ""

# ✅ TESLA M40 GPU DIAGNOSTICS
echo "=== TESLA M40 GPU DIAGNOSTICS ==="
nvidia-smi
echo ""
echo "=== CUDA VISIBLE DEVICES ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""
echo "=== PYTORCH GPU CHECK ==="
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    try:
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        # Test memory allocation on Tesla M40
        print(f'  - Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
        test = torch.randn(1000, 1000).cuda(i)
        print(f'  - Memory test: OK')
        del test
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'  - Memory test: FAILED - {e}')
        
# Performance info
print(f'\\nPerformance Settings:')
print(f'OMP_NUM_THREADS: {torch.get_num_threads()}')
print(f'cuDNN enabled: {torch.backends.cudnn.enabled}')
print(f'cuDNN benchmark: {torch.backends.cudnn.benchmark}')
print(f'Tesla M40 setup ready! 🚀')
"
echo ""
echo "=== STARTING TRAINING ON TESLA M40 ==="

# ⚡ RUN WITH TESLA M40 OPTIMIZED FLAGS
python -O -u src/training/v2_train.py