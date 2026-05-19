#!/bin/bash
#SBATCH --job-name=matteo-act-tape
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/slurm-logs/%u_%x_%j.out
#SBATCH --error=/home/slurm-logs/%u_%x_%j.err

set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/intern_matteo_vulliez/envs/lerobot-act

export PYTHONUNBUFFERED=1
export HF_HOME="$HOME/intern_matteo_vulliez/.hf-cache"

PROJECT_DIR="$HOME/intern_matteo_vulliez/widowx_act_current"
DATASET_NAME="widowx_push_tape_front_20260512"
DATASET_DIR="$PROJECT_DIR/lerobot_datasets/$DATASET_NAME"
OUTPUT_DIR="$PROJECT_DIR/outputs/train/act_$DATASET_NAME"
LIVE_LOG="$PROJECT_DIR/outputs/train/act_${DATASET_NAME}_live.log"

cd "$PROJECT_DIR"
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

echo "=== ACT push_tape front train start at $(date) ===" | tee -a "$LIVE_LOG"
echo "Monitor: http://192.168.100.36:7865" | tee -a "$LIVE_LOG"
hostname | tee -a "$LIVE_LOG"
nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a "$LIVE_LOG"

echo "=== Convert WidowX dataset to LeRobot ===" | tee -a "$LIVE_LOG"
python scripts/convert_widowx_to_lerobot.py \
  --source-root widowx_ai/recordings \
  --output-root "$DATASET_DIR" \
  --repo-id matteo/$DATASET_NAME \
  --robot-type widowx_ai \
  --fps 30 \
  --cameras front \
  --action-offset 1 \
  --no-use-videos \
  --overwrite 2>&1 | tee -a "$LIVE_LOG"

echo "=== Train LeRobot ACT ===" | tee -a "$LIVE_LOG"
lerobot-train \
  --dataset.repo_id=matteo/$DATASET_NAME \
  --dataset.root="$DATASET_DIR" \
  --policy.type=act \
  --output_dir="$OUTPUT_DIR" \
  --job_name=act_$DATASET_NAME \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --steps=10000 \
  --batch_size=8 \
  --save_freq=1000 2>&1 | tee -a "$LIVE_LOG"

echo "=== ACT push_tape front train end at $(date) ===" | tee -a "$LIVE_LOG"
