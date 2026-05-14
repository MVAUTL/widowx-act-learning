#!/bin/bash
#SBATCH --job-name=matteo-act-smoke
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/slurm-logs/%u_%x_%j.out
#SBATCH --error=/home/slurm-logs/%u_%x_%j.err

set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/intern_matteo_vulliez/envs/lerobot-act

export PYTHONUNBUFFERED=1
export HF_HOME="$HOME/intern_matteo_vulliez/.hf-cache"

cd ~/intern_matteo_vulliez/widowx_act

echo "=== ACT smoke start at $(date) ==="
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

lerobot-train \
  --dataset.repo_id=matteo/widowx-push-cube-local \
  --dataset.root=$HOME/intern_matteo_vulliez/widowx_act/lerobot_datasets/widowx_push_cube_smoke \
  --policy.type=act \
  --output_dir=$HOME/intern_matteo_vulliez/widowx_act/outputs/train/act_widowx_smoke \
  --job_name=act_widowx_smoke \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --steps=200 \
  --batch_size=8 \
  --save_freq=100

echo "=== ACT smoke end at $(date) ==="
