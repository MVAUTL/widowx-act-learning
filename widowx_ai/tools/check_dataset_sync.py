#!/usr/bin/env python3
"""Check timing quality for a WidowX teaching dataset session."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _interval_stats(timestamps: list[float]) -> dict[str, float | None]:
    if len(timestamps) < 2:
        return {"fps": None, "median_dt_ms": None, "max_dt_ms": None}
    intervals = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
    if not intervals:
        return {"fps": None, "median_dt_ms": None, "max_dt_ms": None}
    median_dt = statistics.median(intervals)
    return {
        "fps": 1.0 / median_dt if median_dt > 0 else None,
        "median_dt_ms": median_dt * 1000.0,
        "max_dt_ms": max(intervals) * 1000.0,
    }


def _latest_dataset(root: Path) -> Path:
    sessions = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and (path / "samples.jsonl").exists()
        and (path / "motor_samples.jsonl").exists()
    ]
    if not sessions:
        raise RuntimeError(f"No dataset session found in {root}")
    return max(sessions, key=lambda path: path.stat().st_mtime)


def _fmt(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.2f}{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check WidowX teaching dataset timing.")
    parser.add_argument(
        "session",
        nargs="?",
        help="Dataset session directory. Defaults to latest session with samples + motor samples.",
    )
    parser.add_argument(
        "--root",
        default=str(PACKAGE_ROOT / "recordings"),
        help="Recordings root used when no session is provided.",
    )
    parser.add_argument("--max-sync-ms", type=float, default=50.0)
    parser.add_argument("--max-camera-spread-ms", type=float, default=100.0)
    args = parser.parse_args()

    session = Path(args.session).expanduser().resolve() if args.session else _latest_dataset(Path(args.root))
    samples = _read_jsonl(session / "samples.jsonl")
    motor_samples = _read_jsonl(session / "motor_samples.jsonl")
    if not samples:
        raise RuntimeError("No camera samples found.")
    if not motor_samples:
        raise RuntimeError("No motor samples found.")

    camera_times = [float(row["timestamp"]) for row in samples if row.get("timestamp") is not None]
    motor_times = [float(row["timestamp"]) for row in motor_samples if row.get("timestamp") is not None]
    camera_stats = _interval_stats(camera_times)
    motor_stats = _interval_stats(motor_times)
    sync_deltas = []
    for row in samples:
        if row.get("sync_delta_seconds") is not None:
            sync_deltas.append(abs(float(row["sync_delta_seconds"])) * 1000.0)
        elif row.get("timestamp") is not None and row.get("motor_timestamp") is not None:
            sync_deltas.append(abs(float(row["timestamp"]) - float(row["motor_timestamp"])) * 1000.0)

    missing_images = 0
    roles: set[str] = set()
    camera_spreads = []
    for row in samples:
        images = row.get("images") or {}
        roles.update(str(role) for role in images)
        for image in images.values():
            if not (session / str(image)).exists():
                missing_images += 1
        frame_timestamps = row.get("frame_timestamps") or {}
        if len(frame_timestamps) >= 2:
            times = [float(value) for value in frame_timestamps.values()]
            camera_spreads.append((max(times) - min(times)) * 1000.0)

    max_sync = max(sync_deltas) if sync_deltas else None
    median_sync = statistics.median(sync_deltas) if sync_deltas else None
    max_camera_spread = max(camera_spreads) if camera_spreads else None
    median_camera_spread = statistics.median(camera_spreads) if camera_spreads else None
    ok = (
        missing_images == 0
        and camera_stats["fps"] is not None
        and 20.0 <= float(camera_stats["fps"]) <= 50.0
        and max_sync is not None
        and max_sync <= args.max_sync_ms
        and (max_camera_spread is None or max_camera_spread <= args.max_camera_spread_ms)
    )

    print(f"Session: {session}")
    print(f"Camera samples: {len(samples)}")
    print(f"Motor samples: {len(motor_samples)}")
    print(f"Camera FPS median: {_fmt(camera_stats['fps'])} ({_fmt(camera_stats['median_dt_ms'], ' ms')} dt)")
    print(f"Motor FPS median: {_fmt(motor_stats['fps'])} ({_fmt(motor_stats['median_dt_ms'], ' ms')} dt)")
    print(f"Roles: {', '.join(sorted(roles)) if roles else 'none'}")
    print(f"Missing image files: {missing_images}")
    print(f"Median image/motor sync delta: {_fmt(median_sync, ' ms')}")
    print(f"Max image/motor sync delta: {_fmt(max_sync, ' ms')}")
    print(f"Median inter-camera spread: {_fmt(median_camera_spread, ' ms')}")
    print(f"Max inter-camera spread: {_fmt(max_camera_spread, ' ms')}")
    print(f"Verdict: {'OK for first ACT dataset' if ok else 'CHECK before training'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
