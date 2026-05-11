#!/bin/bash
#SBATCH --job-name=matteo-act-full
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/slurm-logs/%u_%x_%j.out
#SBATCH --error=/home/slurm-logs/%u_%x_%j.err

set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/intern_matteo_vulliez/envs/lerobot-act

export PYTHONUNBUFFERED=1
export HF_HOME="$HOME/intern_matteo_vulliez/.hf-cache"

cd ~/intern_matteo_vulliez/widowx_act

echo "=== ACT full train start at $(date) ==="
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv

lerobot-train \
  --dataset.repo_id=matteo/widowx-push-cube-local \
  --dataset.root=$HOME/intern_matteo_vulliez/widowx_act/lerobot_datasets/widowx_push_cube_full \
  --policy.type=act \
  --output_dir=$HOME/intern_matteo_vulliez/widowx_act/outputs/train/act_widowx_full \
  --job_name=act_widowx_full \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --steps=10000 \
  --batch_size=8 \
  --save_freq=1000

echo "=== ACT full train end at $(date) ==="
