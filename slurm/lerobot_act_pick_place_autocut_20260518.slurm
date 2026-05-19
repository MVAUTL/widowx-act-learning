#!/bin/bash
#SBATCH --job-name=matteo-act-pick
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/slurm-logs/%u_%x_%j.out
#SBATCH --error=/home/slurm-logs/%u_%x_%j.err

set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/intern_matteo_vulliez/envs/lerobot-act

export PYTHONUNBUFFERED=1
export HF_HOME="$HOME/intern_matteo_vulliez/.hf-cache"

cd ~/intern_matteo_vulliez/widowx_act_current

DATASET_NAME="dataset_teach_20260512_042931_20260514_111259_autocut_autocut"
DATASET_DIR="$HOME/intern_matteo_vulliez/lerobot_datasets/$DATASET_NAME"
OUTPUT_DIR="$HOME/intern_matteo_vulliez/widowx_act_current/outputs/train/act_dataset_20260512_autocut_20260518"
LIVE_LOG="$HOME/intern_matteo_vulliez/widowx_act_current/outputs/train/act_dataset_20260512_autocut_20260518_live.log"

mkdir -p "$(dirname "$LIVE_LOG")"
rm -f "$LIVE_LOG"

cleanup() {
  if [ -n "${MONITOR_PID:-}" ]; then
    kill "$MONITOR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

python -u scripts/lerobot_act_monitor.py \
  --host 0.0.0.0 \
  --port 7865 \
  --log-path "$LIVE_LOG" \
  --output-dir "$OUTPUT_DIR" \
  --dataset-dir "$DATASET_DIR" &
MONITOR_PID=$!

echo "=== ACT pick-place train start at $(date) ===" | tee -a "$LIVE_LOG"
echo "Monitor: http://192.168.100.36:7865" | tee -a "$LIVE_LOG"
hostname | tee -a "$LIVE_LOG"
nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a "$LIVE_LOG"

if [ ! -f "$DATASET_DIR/meta/info.json" ]; then
  echo "ERROR: LeRobot dataset not found at $DATASET_DIR" | tee -a "$LIVE_LOG"
  exit 2
fi

lerobot-train \
  --dataset.repo_id=matteo/dataset-20260512-autocut-local \
  --dataset.root="$DATASET_DIR" \
  --policy.type=act \
  --output_dir="$OUTPUT_DIR" \
  --job_name=act_dataset_20260512_autocut_20260518 \
  --policy.device=cuda \
  --policy.use_amp=true \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --steps=1000 \
  --batch_size=8 \
  --num_workers=2 \
  --save_freq=250 2>&1 | tee -a "$LIVE_LOG"

echo "=== ACT pick-place train end at $(date) ===" | tee -a "$LIVE_LOG"
