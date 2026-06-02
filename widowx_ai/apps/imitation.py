from __future__ import annotations

import base64
from contextlib import suppress
from datetime import datetime
import json
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np


class ImitationTrajectoryRunner:
    def __init__(self, arm: Any, recorder: Any) -> None:
        self.arm = arm
        self.recorder = recorder
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.running = False
        self.point_index = 0
        self.point_count = 0
        self.session_dir: str | None = None
        self.last_message = "Imitation trajectory ready"
        self.last_error: str | None = None

    def status(self) -> dict[str, Any]:
        with self.lock:
            return self.status_unlocked()

    def status_unlocked(self) -> dict[str, Any]:
        return {
            "ok": True,
            "running": self.running,
            "point_index": self.point_index,
            "point_count": self.point_count,
            "session_dir": self.session_dir,
            "message": self.last_error or self.last_message,
        }

    @staticmethod
    def _safe_session_name(raw_name: Any) -> str:
        name = str(raw_name or "").strip() or f"hamster_imitation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        cleaned = "".join(ch if ch in allowed else "_" for ch in name)
        return cleaned.strip("._-") or f"hamster_imitation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    @staticmethod
    def _bounded_float(payload: dict[str, Any], name: str, default: float, low: float, high: float) -> float:
        value = float(payload.get(name, default))
        if not low <= value <= high:
            raise RuntimeError(f"{name} must be between {low} and {high}.")
        return value

    @staticmethod
    def _validated_expected_count(payload: dict[str, Any]) -> int | None:
        raw_count = payload.get("expected_count")
        if raw_count in (None, ""):
            return None
        count = int(raw_count)
        if not 3 <= count <= 30:
            raise RuntimeError("Expected point count must be between 3 and 30.")
        return count

    def _validate_waypoints(self, raw_waypoints: Any, expected_count: int | None) -> list[dict[str, Any]]:
        if not isinstance(raw_waypoints, list) or len(raw_waypoints) < 3:
            raise RuntimeError("Save at least 3 trajectory points before running imitation.")
        if expected_count is not None and len(raw_waypoints) != expected_count:
            raise RuntimeError(f"Hamster expects exactly {expected_count} points, but {len(raw_waypoints)} are saved.")
        waypoints: list[dict[str, Any]] = []
        for index, point in enumerate(raw_waypoints, start=1):
            raw_positions = point.get("positions") if isinstance(point, dict) else point
            try:
                gripper_position = (
                    self.arm._validated_gripper_position(point.get("gripper_position"))
                    if isinstance(point, dict)
                    else None
                )
                waypoints.append(
                    {
                        "positions": self.arm._validated_positions(raw_positions),
                        "gripper_position": gripper_position,
                    }
                )
            except Exception as exc:
                raise RuntimeError(f"Point {index} is invalid: {exc}") from exc
        return waypoints

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.arm._require_motion(payload)
        expected_count = self._validated_expected_count(payload)
        waypoints = self._validate_waypoints(payload.get("waypoints"), expected_count)
        move_time = self._bounded_float(payload, "move_time", 2.0, 0.2, 20.0)
        pause_time = self._bounded_float(payload, "pause_time", 0.15, 0.0, 10.0)
        record = bool(payload.get("record", True))
        if record and not str(payload.get("video_source", "")).strip():
            raise RuntimeError("Select a camera source before recording imitation.")
        with self.lock:
            if self.running:
                raise RuntimeError("Imitation trajectory is already running.")
            self.stop_event.clear()
            self.running = True
            self.point_index = 0
            self.point_count = len(waypoints)
            self.session_dir = None
            self.last_error = None
            self.last_message = "Starting imitation trajectory"
            self.thread = threading.Thread(
                target=self._run,
                args=(waypoints, move_time, pause_time, record, dict(payload)),
                daemon=True,
            )
            self.thread.start()
            return self.status_unlocked()

    def stop(self) -> dict[str, Any]:
        with self.lock:
            self.stop_event.set()
            if not self.running:
                return self.status_unlocked()
            self.last_message = "Stopping imitation trajectory"
            return self.status_unlocked()

    def force_stop(self) -> None:
        self.stop_event.set()
        with suppress(Exception):
            self.recorder.stop()
        with self.lock:
            self.running = False
            self.last_message = "Imitation trajectory stopped by emergency stop"

    def _record_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        video_source = str(payload.get("video_source", "")).strip()
        if not video_source:
            raise RuntimeError("Select a camera source before recording imitation.")
        hamster_answer = str(payload.get("hamster_answer", "")).strip()
        hamster_prompt = str(payload.get("hamster_prompt", "")).strip()
        return {
            "fps": self.recorder.HIGH_SMOOTH_CAMERA_FPS,
            "video_source": video_source,
            "capture_sources": [{"source": video_source, "role": "hamster_view", "crop": None}],
            "session_name": self._safe_session_name(payload.get("session_name")),
            "high_smooth": True,
            "with_camera": True,
            "capture_type": "hamster_imitation",
            "task_name": payload.get("task_name") or hamster_prompt or "hamster_imitation",
            "dataset_profile": payload.get("dataset_profile"),
            "action_offset": payload.get("action_offset", 1),
            "lerobot_features": {
                "hamster_prompt": hamster_prompt,
                "hamster_answer": hamster_answer,
                "trajectory_waypoints": payload.get("waypoints"),
            },
        }

    def _save_reference_assets(self, payload: dict[str, Any], session_dir: str | None) -> None:
        if not session_dir:
            return
        session = Path(session_dir)
        reference = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "hamster_prompt": payload.get("hamster_prompt"),
            "hamster_answer": payload.get("hamster_answer"),
            "waypoints": payload.get("waypoints"),
        }
        (session / "hamster_imitation_plan.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
        overlay = str(payload.get("hamster_overlay", "")).strip()
        if not overlay:
            return
        if "," in overlay:
            overlay = overlay.split(",", 1)[1]
        try:
            (session / "hamster_overlay.jpg").write_bytes(base64.b64decode(overlay, validate=False))
        except Exception:
            pass

    def _run(
        self,
        waypoints: list[dict[str, Any]],
        move_time: float,
        pause_time: float,
        record: bool,
        payload: dict[str, Any],
    ) -> None:
        recording_started = False
        try:
            if record:
                record_status = self.recorder.start(self._record_payload(payload))
                recording_started = True
                with self.lock:
                    self.session_dir = record_status.get("session_dir")
                self._save_reference_assets(payload, self.session_dir)
                time.sleep(0.25)
            self.arm.prepare_replay()
            for index, point in enumerate(waypoints, start=1):
                if self.stop_event.is_set():
                    break
                with self.lock:
                    self.point_index = index
                    self.last_message = f"Moving to imitation point {index}/{len(waypoints)}"
                actual_move_time = self.arm.replay_position(
                    point["positions"],
                    move_time,
                    point.get("gripper_position"),
                )
                deadline = time.monotonic() + actual_move_time + pause_time
                while time.monotonic() < deadline:
                    if self.stop_event.is_set():
                        break
                    time.sleep(0.02)
            if recording_started:
                record_status = self.recorder.stop()
                with self.lock:
                    self.session_dir = record_status.get("session_dir") or self.session_dir
            with self.lock:
                self.running = False
                self.thread = None
                self.last_message = (
                    f"Imitation trajectory stopped at point {self.point_index}/{self.point_count}"
                    if self.stop_event.is_set()
                    else f"Imitation trajectory complete: {self.session_dir or 'not recorded'}"
                )
        except Exception as exc:  # noqa: BLE001 - surfaced in UI status.
            if recording_started:
                with suppress(Exception):
                    record_status = self.recorder.stop()
                    with self.lock:
                        self.session_dir = record_status.get("session_dir") or self.session_dir
            with self.lock:
                self.running = False
                self.thread = None
                self.last_error = f"Imitation trajectory failed: {exc}"
