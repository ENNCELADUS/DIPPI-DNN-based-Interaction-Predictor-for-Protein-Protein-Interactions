#!/bin/bash
#SBATCH -J PREPROCESS
#SBATCH -p critical
#SBATCH -A hexm-critical
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=ai_gpu28
#SBATCH --output=logs/preprocess/slurm_%j.out
#SBATCH --error=logs/preprocess/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

source ~/.bashrc && conda activate esm

python src/data_preprocess/sample_negatives.py --input data/TMP/processed/pretrain_train.csv --output data/TMP/processed/pretrain_train_balanced.csv --hard-ratio 0.3 --seed 47

python src/data_preprocess/sample_negatives.py --input data/TMP/processed/pretrain_val.csv --output data/TMP/processed/pretrain_val_balanced.csv --hard-ratio 0.3 --seed 47
