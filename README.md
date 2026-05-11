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

```bash
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
