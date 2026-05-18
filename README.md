# WidowX ACT Learning

Local tools for controlling a Trossen Robotics WidowX AI arm, recording teaching sessions, training a compact ACT-style policy, monitoring training runs, and replaying checkpoints with conservative safety gates.

Full documentation:

- English: [docs/README.en.md](docs/README.en.md)
- French: [docs/README.fr.md](docs/README.fr.md)
- HAMSTER/VILA on DGX Spark: [docs/hamster_dgx_spark.fr.md](docs/hamster_dgx_spark.fr.md)
- Research links and papers: [docs/research/README.md](docs/research/README.md)

## Project Layout

```text
.
├── README.md
├── docs/
│   ├── README.en.md
│   └── README.fr.md
├── requirements.txt
└── widowx_ai/
    ├── apps/          # Local web UI
    ├── config/        # Local robot configuration
    ├── policies/      # Safe policy replay
    ├── robot/         # Robot connection and motion utilities
    ├── tools/         # Dataset inspection utilities
    └── training/      # ACT training and training monitor
```

Large local artifacts are intentionally ignored by Git:

- `widowx_ai/recordings*/`
- `widowx_ai/models/`
- `.venv*/`
- `*.pdf`

## Setup

### Install on a new PC

Clone the repository and enter the project:

```bash
git clone https://github.com/MVAUTL/widowx-act-learning.git
cd widowx-act-learning
```

Install or select Python 3.10. The control environment is expected to be named `.venv-trossen-ui`:

```bash
python3.10 -m venv .venv-trossen-ui
.venv-trossen-ui/bin/python -m pip install --upgrade pip setuptools wheel
.venv-trossen-ui/bin/python -m pip install -r requirements.txt
```

If `python3.10` is not available but `pyenv` is installed:

```bash
pyenv install 3.10.12
pyenv local 3.10.12
python -m venv .venv-trossen-ui
.venv-trossen-ui/bin/python -m pip install --upgrade pip setuptools wheel
.venv-trossen-ui/bin/python -m pip install -r requirements.txt
```

Verify the environment before connecting to the robot:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.robot.widowx_demo --check-import
.venv-trossen-ui/bin/python -m py_compile widowx_ai/apps/interface.py
```

Start the interface in dry-run mode first:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.apps.interface --port 7862
```

Open `http://127.0.0.1:7862/`. If the page loads, stop the server with `Ctrl+C`, connect the robot Ethernet, then start real mode:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.apps.interface --real --ip 192.168.1.2 --variant base --timeout 15 --port 7862
```

Useful pages after startup:

- Control: `http://127.0.0.1:7862/`
- Hamster camera planner: `http://127.0.0.1:7862/hamster`
- Model test: `http://127.0.0.1:7862/model-test`
- Teaching/data collection: `http://127.0.0.1:7862/teach`

Notes for a fresh PC:

- Large local assets such as `widowx_ai/models/` and `widowx_ai/recordings*/` are not stored in Git.
- The real robot should be tested in dry-run first, then with `--real` only when the workspace is clear.
- If USB cameras do not appear, unplug/replug the camera and use `Refresh cameras` in the web UI.

### Minimal local setup

Create the local Python environment used by the control UI:

```bash
python3.10 -m venv .venv-trossen-ui
.venv-trossen-ui/bin/python -m pip install --upgrade pip setuptools wheel
.venv-trossen-ui/bin/python -m pip install -r requirements.txt
```

Verify that the Trossen driver installed correctly:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.robot.widowx_demo --check-import
```

Expected output includes:

```text
trossen_arm import OK
Model wxai_v0 OK: True
```

If `python3.10` is not available but `pyenv` is installed:

```bash
pyenv install 3.10.12
pyenv local 3.10.12
python -m venv .venv-trossen-ui
.venv-trossen-ui/bin/python -m pip install --upgrade pip setuptools wheel
.venv-trossen-ui/bin/python -m pip install -r requirements.txt
```

## Common Commands

Verify the driver import:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.robot.widowx_demo --check-import
```

Start the local web interface in dry-run mode:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.apps.interface --port 7862
```

Start the web interface for the real arm:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.apps.interface --real --ip 192.168.1.2 --variant base --port 7862
```

Train an ACT-style policy from local recordings:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.training.train_act --epochs 10 --batch-size 16 --chunk-size 8 --view wrist_rgb
```

Monitor training runs:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.training.act_monitor --port 7865
```

Check dataset timing:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.tools.check_dataset_sync
```

## NVIDIA DGX Spark ACT Training

Use the DGX Spark for GPU training. Do not start GPU training directly on the
login shell with `python` or `lerobot-train`; submit it through SLURM with
`sbatch`.

### Connect to the DGX Spark

From the lab network:

```bash
ssh -4 guest@dgx-spark.local
```

If the local hostname does not resolve, use the Ethernet IP:

```bash
ssh -4 guest@192.168.100.36
```

### Enter or update the project

The current DGX working directory used by the SLURM scripts is:

```bash
cd ~/intern_matteo_vulliez/widowx_act_current
```

If the project is already cloned there, update it:

```bash
git fetch origin
git checkout lerobot-teach-dataset-export
git pull origin lerobot-teach-dataset-export
```

If you need a fresh clone:

```bash
cd ~/intern_matteo_vulliez
git clone https://github.com/MVAUTL/widowx-act-learning.git widowx_act_current
cd widowx_act_current
git checkout lerobot-teach-dataset-export
```

### Check GPU and SLURM

Before submitting a job:

```bash
squeue
nvidia-smi
```

Quick interpretation:

- empty `squeue`: no SLURM job is currently running;
- `ST=R`: a job is already running;
- `ST=PD`: a job is waiting in the queue;
- a Python process using a lot of VRAM in `nvidia-smi`: the GPU is busy.

Do not kill another user's job.

### Activate the LeRobot ACT environment

The SLURM scripts activate this environment automatically, but you can check it
manually:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/intern_matteo_vulliez/envs/lerobot-act
python - <<'PY'
import torch
import lerobot
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("lerobot", lerobot.__version__)
PY
```

`cuda` must be `True` for GPU training.

### Launch ACT training

For the current front-camera tape dataset pipeline, submit:

```bash
cd ~/intern_matteo_vulliez/widowx_act_current
sbatch slurm/lerobot_act_push_tape_front_20260512.slurm
```

This job converts `widowx_ai/recordings` to a LeRobot dataset, then trains ACT
on CUDA. It writes:

```text
lerobot_datasets/widowx_push_tape_front_20260512/
outputs/train/act_widowx_push_tape_front_20260512/
outputs/train/act_widowx_push_tape_front_20260512_live.log
```

Open the live monitor from a browser on the lab network:

```text
http://192.168.100.36:7865
```

For a smaller smoke pipeline:

```bash
sbatch slurm/lerobot_act_small_pipeline.slurm
```

For the older full push-cube pipeline:

```bash
sbatch slurm/lerobot_act_full_with_monitor.slurm
```

### Monitor or stop a job

List jobs:

```bash
squeue
```

Follow SLURM logs:

```bash
tail -f /home/slurm-logs/guest_matteo-act-tape_<JOB_ID>.out
```

Cancel only your own job:

```bash
scancel <JOB_ID>
```

More DGX/HAMSTER notes are in
[`docs/hamster_dgx_spark.fr.md`](docs/hamster_dgx_spark.fr.md).

## NVIDIA DGX Spark HAMSTER Server

HAMSTER/VILA also runs on the DGX Spark through SLURM. Do not launch
`server.py` directly on the login shell; it loads the GPU and must be submitted
with `sbatch`.

### Connect to the DGX Spark

From the lab network:

```bash
ssh -4 guest@dgx-spark.local
```

Fallback if mDNS does not resolve:

```bash
ssh -4 guest@192.168.100.36
```

### Launch HAMSTER

Use the existing DGX install:

```bash
cd ~/intern_matteo_vulliez
squeue -u $USER
sbatch slurm/hamster_full_pipeline.slurm
```

The job starts both:

- backend/API on `http://192.168.100.36:8000/docs`
- Gradio web UI on `http://192.168.100.36:7860`

Check readiness from the DGX:

```bash
curl -s -o /dev/null -w 'backend:%{http_code}\n' http://127.0.0.1:8000/docs
curl -s -o /dev/null -w 'web:%{http_code}\n' http://127.0.0.1:7860
```

Expected result:

```text
backend:200
web:200
```

Follow logs, replacing `<JOB_ID>` with the value returned by `sbatch`:

```bash
tail -f /home/slurm-logs/guest_matteo-hamster-full_<JOB_ID>.out
tail -f /home/slurm-logs/guest_matteo-hamster-full_<JOB_ID>.err
```

Stop HAMSTER:

```bash
scancel <JOB_ID>
```

Once HAMSTER is running, the local control UI can call it from:

```text
