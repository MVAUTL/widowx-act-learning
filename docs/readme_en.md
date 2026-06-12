# WidowX ACT Learning - English Guide

This project provides local tools to control a Trossen Robotics WidowX AI arm, record demonstrations, train a small ACT-style policy, and test checkpoints with conservative safety gates.

French documentation: [readme_fr.md](readme_fr.md)

Research links and downloaded papers: [research/README.md](research/README.md)

## Layout

```text
widowx_ai/
├── apps/       # Local web interface
├── config/     # Local robot configuration
├── policies/   # Safe policy replay
├── robot/      # Robot connection and motion utilities
├── tools/      # Dataset checks
└── training/   # ACT training and training monitor
```

Large local artifacts are ignored by Git:

- `widowx_ai/recordings*/`
- `widowx_ai/models/`
- `.venv*/`
- `*.pdf`

## Setup

From the project root:

```bash
.venv-trossen-ui/bin/python -m pip install -r requirements.txt
```

Check that the Trossen driver is available:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.robot.widowx_demo --check-import
```

## Arm Connection

Typical controller network settings:

- Arm IP: `192.168.1.2`
- PC IP: `192.168.1.1`
- Netmask: `255.255.255.0`

Check connectivity:

```bash
ping 192.168.1.2
```

Read the arm state without motion:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.robot.widowx_demo --ip 192.168.1.2 --variant base
```

Run a small motion test only when the workspace is clear:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.robot.widowx_demo --ip 192.168.1.2 --variant base --move
```

## Web Interface

Dry-run mode, with no real motion:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.apps.interface --port 7862
```

Real-arm mode:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.apps.interface --real --ip 192.168.1.2 --variant base --port 7862
```

Open:

```text
http://127.0.0.1:7862
```

In the interface, enable `enable motion` before every command that moves the arm. Use `Emergency stop` if the arm must be stopped immediately.

## Gravity Compensation

To guide the arm by hand:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.robot.gravity_compensation --ip 192.168.1.2 --variant base
```

With a RealSense D405 mounted on the gripper, the `d405-follower` profile may better compensate the payload:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.robot.gravity_compensation --ip 192.168.1.2 --variant base --payload-profile d405-follower --camera-wrist-effort 0.10
```

## Recording and Datasets

The web interface can:

- record a hand-guided movement;
- replay the movement;
- capture camera frames during replay;
- produce `motor_samples.jsonl`, `samples.jsonl`, and `metadata.json`.

Check synchronization for the latest dataset:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.tools.check_dataset_sync
```

Check a specific dataset:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.tools.check_dataset_sync widowx_ai/recordings/DATASET_NAME
```

Practical targets:

- camera near 30 Hz;
- motor samples near 100 Hz;
- image/motor sync delta below 50 ms;
- no missing images.

## ACT Training

Train a local model:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.training.train_act --epochs 10 --batch-size 16 --chunk-size 8 --view wrist_rgb
```

Checkpoints are saved under:

```text
widowx_ai/models/
```

Training monitor:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.training.act_monitor --port 7865
```

Then open:

```text
http://127.0.0.1:7865
```

## Safety

Before any real motion:

- secure the arm on a stable surface;
- clear the workspace;
- start with no payload in the gripper;
- test first without `--move` or without `--real`;
- keep emergency stop access available.
