#!/bin/bash
#SBATCH -J RTX2080Ti4x-V2Training
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH --mem=128G         # RAM for 4x RTX 2080 Ti setup
#SBATCH --cpus-per-task=16 # CPUs for 4-GPU setup
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:4  # Request 4x RTX 2080 Ti GPUs (44GB total VRAM)
#SBATCH --mail-user=2162352828@qq.com
# sleep 9999999
source ~/.bashrc
conda activate esm
cd /public/home/wangar2023/DIPPI-DNN-based-Interaction-Predictor-for-Protein-Protein-Interactions/ || { echo "目录不存在"; exit 1; }

# ⚡ RTX 2080 Ti 4x PERFORMANCE OPTIMIZATIONS
echo "=== SETTING 4x RTX 2080 Ti OPTIMIZATIONS ==="

# CUDA Performance Settings for 4x RTX 2080 Ti
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use all 4 RTX 2080 Ti GPUs
export CUDA_LAUNCH_BLOCKING=0  # Async CUDA ops

# PyTorch Performance Settings for RTX 2080 Ti (Turing architecture)
export TORCH_CUDA_ARCH_LIST="7.5"  # RTX 2080 Ti Turing architecture
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"  # Better memory management for 11GB VRAM

# NCCL Settings for Multi-GPU Communication
export NCCL_DEBUG=INFO  # Enable NCCL debugging
export NCCL_SOCKET_IFNAME=^lo,docker0  # Exclude loopback and docker interfaces
export NCCL_IB_DISABLE=1  # Disable InfiniBand (use Ethernet)
export NCCL_P2P_DISABLE=1  # Disable P2P to avoid communication issues

# CPU Performance Settings
export OMP_NUM_THREADS=16  # Match cpus-per-task
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

# Data Loading Optimization
export PYTHONUNBUFFERED=1  # Faster stdout
export TORCH_CUDNN_V8_API_ENABLED=1  # Use cuDNN v8 API

# Memory Optimization for RTX 2080 Ti
ulimit -n 65536  # Increase file descriptor limit
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"  # Reduce memory fragmentation

echo "4x RTX 2080 Ti optimizations applied ✅"
echo ""

# ✅ RTX 2080 Ti GPU DIAGNOSTICS
echo "=== 4x RTX 2080 Ti GPU DIAGNOSTICS ==="
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
        # Test memory allocation on RTX 2080 Ti
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
print(f'4x RTX 2080 Ti setup ready! 🚀')
"
echo ""
echo "=== STARTING V2 TRAINING ON 4x RTX 2080 Ti ==="

# ⚡ RUN WITH 4x RTX 2080 Ti OPTIMIZED FLAGS
python -O -u src/training/v2_train.py