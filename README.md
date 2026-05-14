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
