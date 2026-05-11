# LeRobot local setup

Environment:

```bash
source Lerobot/.venv-lerobot/bin/activate
```

Installed stack:

- `lerobot==0.4.4`
- `torch==2.10.0+cpu`
- `torchvision==0.25.0+cpu`

This local environment is CPU-only to fit the workstation disk. Use the DGX/SLURM scripts for real GPU training.

## Quick checks

```bash
Lerobot/.venv-lerobot/bin/python - <<'PY'
import torch
import lerobot
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("lerobot", lerobot.__version__)
PY

Lerobot/.venv-lerobot/bin/lerobot-train --help
```

## Recording and visualization tools

LeRobot's native recording tools are installed, but they target robots supported
directly by LeRobot (`so100`, `so101`, `koch`, `omx`, etc.). For this WidowX
setup, record with the local `/teach` interface, then convert the captures with
`scripts/convert_widowx_to_lerobot.py`.

Useful non-training tools:

```bash
Lerobot/.venv-lerobot/bin/lerobot-record --help
Lerobot/.venv-lerobot/bin/lerobot-replay --help
Lerobot/.venv-lerobot/bin/lerobot-dataset-viz --help
Lerobot/.venv-lerobot/bin/lerobot-find-cameras opencv --output-dir /tmp/lerobot_camera_test --record-time-s 1
```

On this machine, `lerobot-find-cameras opencv` currently reports no OpenCV
cameras. Use the project control interface camera panel for the D405/top-camera
workflow.

## Convert WidowX recordings to LeRobot

The `/teach` page now has a **LeRobot Dataset export** panel. Record with:

1. `Record source motion`
2. `Replay movement + record camera`
3. `Export LeRobotDataset`

That export calls `scripts/convert_widowx_to_lerobot.py` and writes a standard
LeRobotDataset folder with `meta/`, parquet data, and either `images/` or
`videos/` depending on the selected storage mode.

Use `/tmp` for smoke tests because the project disk is nearly full.

```bash
Lerobot/.venv-lerobot/bin/python scripts/convert_widowx_to_lerobot.py \
  --source-root widowx_ai/recordings \
  --output-root /tmp/widowx_lerobot_smoke \
  --repo-id local/widowx-smoke \
  --robot-type widowx_ai \
  --fps 30 \
  --cameras top_view,wrist_rgb \
  --max-episodes 1 \
  --no-use-videos \
  --overwrite
```

Inspect the converted dataset:

```bash
HF_HOME=/tmp/lerobot_hf_cache \
HF_DATASETS_CACHE=/tmp/lerobot_hf_cache/datasets \
Lerobot/.venv-lerobot/bin/lerobot-edit-dataset \
  --repo_id local/widowx-smoke \
  --root /tmp/widowx_lerobot_smoke \
  --operation.type info \
  --operation.show_features true
```

Create a Rerun visualization file without launching a GUI:

```bash
HF_HOME=/tmp/lerobot_hf_cache \
HF_DATASETS_CACHE=/tmp/lerobot_hf_cache/datasets \
Lerobot/.venv-lerobot/bin/lerobot-dataset-viz \
  --repo-id local/widowx-smoke \
  --root /tmp/widowx_lerobot_smoke \
  --episode-index 0 \
  --save 1 \
  --output-dir /tmp/lerobot_viz_smoke \
  --batch-size 4 \
  --num-workers 0
```

Open the generated file with:

```bash
Lerobot/.venv-lerobot/bin/rerun /tmp/lerobot_viz_smoke/local_widowx-smoke_episode_0.rrd
```

## ACT smoke train

The Hugging Face cache must be outside `/home/mvautl/.cache` in this session.

```bash
HF_HOME=/tmp/lerobot_hf_cache \
HF_DATASETS_CACHE=/tmp/lerobot_hf_cache/datasets \
Lerobot/.venv-lerobot/bin/lerobot-train \
  --dataset.repo_id=local/widowx-smoke \
  --dataset.root=/tmp/widowx_lerobot_smoke \
  --policy.type=act \
  --output_dir=/tmp/widowx_act_train_smoke \
  --job_name=widowx_act_train_smoke \
  --policy.device=cpu \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --steps=1 \
  --batch_size=1 \
  --num_workers=0 \
  --save_freq=1
```
