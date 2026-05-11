#!/bin/bash
#SBATCH --job-name=matteo-convert-lerobot-full
#SBATCH --time=04:00:00
#SBATCH --output=/home/slurm-logs/%u_%x_%j.out
#SBATCH --error=/home/slurm-logs/%u_%x_%j.err

set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/intern_matteo_vulliez/envs/lerobot-act

export PYTHONUNBUFFERED=1
export HF_HOME="$HOME/intern_matteo_vulliez/.hf-cache"

cd ~/intern_matteo_vulliez/widowx_act

python scripts/convert_widowx_to_lerobot.py \
  --source-root widowx_ai/recordings \
  --output-root lerobot_datasets/widowx_push_cube_full \
  --repo-id matteo/widowx-push-cube-local \
  --robot-type widowx_ai \
  --fps 15 \
  --cameras top_view,wrist_rgb \
  --overwrite
