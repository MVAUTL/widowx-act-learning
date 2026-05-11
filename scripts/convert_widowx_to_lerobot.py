#!/usr/bin/env python3
"""Convert local WidowX recordings to a LeRobotDataset.

Input format:
  widowx_ai/recordings/dataset_*/samples.jsonl
  widowx_ai/recordings/dataset_*/motor_samples.jsonl
  JPEG files referenced by each sample row under images/

Output format:
  A standard LeRobotDataset directory created with LeRobotDataset.create().

This script intentionally keeps the feature names simple:
  observation.state
  observation.images.top_view
  observation.images.wrist_rgb
  action

The action at each camera frame is the nearest future motor sample after the
camera-linked motor_index. That is the simplest one-step imitation target and
lets LeRobot ACT build its own action chunks during training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image


STATE_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
    "gripper",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _state(row: dict[str, Any]) -> np.ndarray:
    qpos = row.get("qpos")
    if not isinstance(qpos, list) or len(qpos) != 6:
        raise ValueError(f"Expected qpos with 6 joints, got: {qpos!r}")
    return np.asarray([*qpos, row.get("gripper_position", 0.0)], dtype=np.float32)


def _load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _find_sessions(root: Path, limit: int | None) -> list[Path]:
    sessions = [
        path
        for path in sorted(root.glob("dataset_*"))
        if path.is_dir() and (path / "samples.jsonl").exists() and (path / "motor_samples.jsonl").exists()
    ]
    if limit is not None:
        sessions = sessions[:limit]
    if not sessions:
        raise RuntimeError(f"No WidowX dataset sessions found in {root}")
    return sessions


def _first_image_shape(sessions: list[Path], camera: str) -> tuple[int, int, int]:
    for session in sessions:
        for row in _read_jsonl(session / "samples.jsonl"):
            image_rel = (row.get("images") or {}).get(camera)
            if image_rel:
                image_path = session / str(image_rel)
                if image_path.exists():
                    image = _load_image(image_path)
                    return tuple(int(x) for x in image.shape)
    raise RuntimeError(f"No image found for camera {camera!r}")


def _import_lerobot_dataset():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "LeRobot is not importable. Install it first, e.g. `pip install lerobot` "
            "or use the source install from the official docs."
        ) from exc
    return LeRobotDataset


def convert(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    sessions = _find_sessions(source_root, args.max_episodes)

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    cameras = [camera.strip() for camera in args.cameras.split(",") if camera.strip()]
    if not cameras:
        raise RuntimeError("At least one camera is required.")

    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": STATE_NAMES,
        },
    }
    for camera in cameras:
        features[f"observation.images.{camera}"] = {
            "dtype": "video" if args.use_videos else "image",
            "shape": _first_image_shape(sessions, camera),
            "names": ["height", "width", "channels"],
        }

    LeRobotDataset = _import_lerobot_dataset()
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=output_root,
        features=features,
        robot_type=args.robot_type,
        use_videos=args.use_videos,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )

    converted = 0
    skipped = 0
    for session in sessions:
        samples = _read_jsonl(session / "samples.jsonl")
        motor = _read_jsonl(session / "motor_samples.jsonl")
        episode_frames = 0
        metadata_path = session / "metadata.json"
        task = args.task
        if metadata_path.exists() and not task:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            task = str(metadata.get("task_name") or "widowx task")
        task = task or "widowx task"

        for row in samples:
            images = row.get("images") or {}
            if any(camera not in images for camera in cameras):
                skipped += 1
                continue
            motor_index = int(row.get("motor_index", -1))
            action_index = min(max(motor_index + args.action_offset, 0), len(motor) - 1)
            if motor_index < 0 or not motor:
                skipped += 1
                continue

            frame: dict[str, Any] = {
                "observation.state": _state(row),
                "action": _state(motor[action_index]),
                "task": task,
            }
            missing = False
            for camera in cameras:
                image_path = session / str(images[camera])
                if not image_path.exists():
                    missing = True
                    break
                frame[f"observation.images.{camera}"] = _load_image(image_path)
            if missing:
                skipped += 1
                continue
            dataset.add_frame(frame)
            episode_frames += 1

        if episode_frames:
            dataset.save_episode()
            converted += 1
            print(f"converted {session.name}: {episode_frames} frames", flush=True)
        else:
            print(f"skipped empty episode {session.name}", flush=True)

    if hasattr(dataset, "finalize"):
        dataset.finalize()

    print(f"done: episodes={converted} skipped_frames={skipped} output={output_root}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="widowx_ai/recordings")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-id", default="matteo/widowx-push-cube-local")
    parser.add_argument("--robot-type", default="widowx_ai")
    parser.add_argument("--task", default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--cameras", default="top_view,wrist_rgb")
    parser.add_argument("--action-offset", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--use-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    convert(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
