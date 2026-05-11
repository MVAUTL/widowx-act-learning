#!/bin/bash
#SBATCH --job-name=matteo-act-test-ui
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/slurm-logs/%u_%x_%j.out
#SBATCH --error=/home/slurm-logs/%u_%x_%j.err

set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/intern_matteo_vulliez/envs/lerobot-act

export PYTHONUNBUFFERED=1
export HF_HOME="$HOME/intern_matteo_vulliez/.hf-cache"

cd ~/intern_matteo_vulliez/widowx_act

python -u scripts/lerobot_act_test_interface.py \
  --host 0.0.0.0 \
  --port 7866 \
  --device cuda \
  --dataset-root "$HOME/intern_matteo_vulliez/widowx_act/lerobot_datasets/widowx_push_cube_full" \
  --checkpoint "$HOME/intern_matteo_vulliez/widowx_act/outputs/train/act_widowx_full/checkpoints/010000/pretrained_model"
