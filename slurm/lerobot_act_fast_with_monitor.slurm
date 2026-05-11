#!/bin/bash
#SBATCH --job-name=matteo-act-fast
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/home/slurm-logs/%u_%x_%j.out
#SBATCH --error=/home/slurm-logs/%u_%x_%j.err

set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/intern_matteo_vulliez/envs/lerobot-act

export PYTHONUNBUFFERED=1
export HF_HOME="$HOME/intern_matteo_vulliez/.hf-cache"

cd ~/intern_matteo_vulliez/widowx_act

DATASET_DIR="$HOME/intern_matteo_vulliez/widowx_act/lerobot_datasets/widowx_push_cube_full"
OUTPUT_DIR="$HOME/intern_matteo_vulliez/widowx_act/outputs/train/act_widowx_fast_b32"
LIVE_LOG="$HOME/intern_matteo_vulliez/widowx_act/outputs/train/act_widowx_fast_b32_live.log"

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

echo "=== ACT fast train start at $(date) ===" | tee -a "$LIVE_LOG"
echo "Monitor: http://192.168.100.36:7865" | tee -a "$LIVE_LOG"
hostname | tee -a "$LIVE_LOG"
nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a "$LIVE_LOG"

if [ ! -f "$DATASET_DIR/meta/info.json" ]; then
  echo "ERROR: full LeRobot dataset not found at $DATASET_DIR" | tee -a "$LIVE_LOG"
  exit 2
fi

lerobot-train \
  --dataset.repo_id=matteo/widowx-push-cube-local \
  --dataset.root="$DATASET_DIR" \
  --policy.type=act \
  --output_dir="$OUTPUT_DIR" \
  --job_name=act_widowx_fast_b32 \
  --policy.device=cuda \
  --policy.use_amp=true \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --steps=2500 \
  --batch_size=32 \
  --num_workers=4 \
  --save_freq=500 2>&1 | tee -a "$LIVE_LOG"

echo "=== ACT fast train end at $(date) ===" | tee -a "$LIVE_LOG"
