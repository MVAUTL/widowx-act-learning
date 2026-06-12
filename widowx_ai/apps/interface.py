#!/usr/bin/env python3
"""Local web interface for a Trossen WidowX AI arm.

The server starts in dry-run mode unless --real is passed explicitly.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from contextlib import suppress
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import trossen_arm

from .config import (
    DEFAULT_GRAVITY_PAYLOAD,
    DEFAULT_MAX_SPEED,
    DEMO,
    END_EFFECTORS,
    GRAVITY_PAYLOADS,
    HOME,
    JOINT_LIMITS,
    JOINT_LIMIT_TOLERANCE,
    MAX_CAMERA_WRIST_EFFORT,
    MAX_MAX_SPEED,
    MIN_CAMERA_WRIST_EFFORT,
    MIN_MAX_SPEED,
    PACKAGE_ROOT,
    PROJECT_ROOT,
    REPLAY_GRIPPER_MAX_SPEED,
    REST,
    START_POSITION_MIN_TIME,
)
from .hamster import HamsterService
from .imitation import ImitationTrajectoryRunner
from .lerobot_export import ActDatasetPlanner, LeRobotExportRunner
from .trajectory_overlay import TrajectoryOverlay

try:
    import cv2
    import pyrealsense2 as rs
except ImportError:
    cv2 = None
    rs = None


INDEX_HTML = (Path(__file__).resolve().parent / "pages" / "control.html").read_text(encoding="utf-8")


MODEL_TEST_HTML = (Path(__file__).resolve().parent / "pages" / "model_test.html").read_text(encoding="utf-8")


TEACH_HTML = (Path(__file__).resolve().parent / "pages" / "teach.html").read_text(encoding="utf-8")


HAMSTER_HTML = (Path(__file__).resolve().parent / "pages" / "hamster.html").read_text(encoding="utf-8")


DATASET_TRIM_HTML = (Path(__file__).resolve().parent / "pages" / "dataset_trim.html").read_text(encoding="utf-8")


IMITATION_HTML = (Path(__file__).resolve().parent / "pages" / "imitation.html").read_text(encoding="utf-8")

IMITATION_REVIEW_JS = (Path(__file__).resolve().parent / "static" / "imitation_review.js").read_text(encoding="utf-8")

IMITATION_TRAJECTORY_JS = (Path(__file__).resolve().parent / "static" / "imitation_trajectory.js").read_text(encoding="utf-8")


class CameraController:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.pipeline: Any | None = None
        self.mode = "color"
        self.serial: str | None = None
        self.last_message = "Camera ready"
        self.usb_capture: Any | None = None
        self.usb_index: int | None = None
        self.usb_label: str | None = None
        self.usb_captures: dict[int, Any] = {}
        self.usb_labels: dict[int, str] = {}
        self.usb_last_message = "USB camera ready"

    @staticmethod
    def _require_deps() -> None:
        if rs is None or cv2 is None:
            raise RuntimeError("pyrealsense2 and opencv-python are required for camera preview.")

    @staticmethod
    def _validate_mode(mode: str) -> str:
        if mode not in {"color", "depth"}:
            raise RuntimeError("Camera mode must be 'color' or 'depth'.")
        return mode

    @staticmethod
    def validate_crop(raw_crop: Any) -> dict[str, Any] | None:
        if not raw_crop:
            return None
        if not isinstance(raw_crop, dict):
            raise RuntimeError("Crop configuration must be an object.")
        if not bool(raw_crop.get("enabled", False)):
            return None
        aspect = str(raw_crop.get("aspect", "source"))
        if aspect not in {"source", "1:1", "4:3", "16:9", "3:2", "9:16"}:
            raise RuntimeError("Unknown crop aspect ratio.")
        zoom = float(raw_crop.get("zoom", 1.0))
        offset_x = float(raw_crop.get("offset_x", 0.0))
        offset_y = float(raw_crop.get("offset_y", 0.0))
        if not 1.0 <= zoom <= 4.0:
            raise RuntimeError("Crop zoom must be between 1.00 and 4.00.")
        if not -1.0 <= offset_x <= 1.0 or not -1.0 <= offset_y <= 1.0:
            raise RuntimeError("Crop offsets must be between -1.00 and 1.00.")
        return {
            "enabled": True,
            "aspect": aspect,
            "zoom": zoom,
            "offset_x": offset_x,
            "offset_y": offset_y,
        }

    @staticmethod
    def _aspect_value(aspect: str, width: int, height: int) -> float:
        if aspect == "source":
            return width / height
        left, right = aspect.split(":", 1)
        return float(left) / float(right)

    @classmethod
    def _apply_crop(cls, image: Any, raw_crop: Any) -> Any:
        crop = cls.validate_crop(raw_crop)
        if crop is None:
            return image
        height, width = image.shape[:2]
        if width < 2 or height < 2:
            return image

        target_aspect = cls._aspect_value(str(crop["aspect"]), width, height)
        crop_width = width / float(crop["zoom"])
        crop_height = height / float(crop["zoom"])
        if crop_width / crop_height > target_aspect:
            crop_width = crop_height * target_aspect
        else:
            crop_height = crop_width / target_aspect
        crop_width = max(1, min(width, int(round(crop_width))))
        crop_height = max(1, min(height, int(round(crop_height))))

        max_left = width - crop_width
        max_top = height - crop_height
        center_left = max_left / 2.0
        center_top = max_top / 2.0
        left = int(round(center_left + float(crop["offset_x"]) * center_left))
        top = int(round(center_top + float(crop["offset_y"]) * center_top))
        left = max(0, min(max_left, left))
        top = max(0, min(max_top, top))
        return image[top : top + crop_height, left : left + crop_width]

    def _first_device_unlocked(self) -> tuple[Any | None, str | None, str | None]:
        self._require_deps()
        ctx = rs.context()
        devices = list(ctx.query_devices())
        if not devices:
            return None, None, None
        device = devices[0]
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        return device, name, serial

    @staticmethod
    def _video_device_indices() -> list[int]:
        indices: list[int] = []
        for candidate in sorted(Path("/dev").glob("video*")):
            suffix = candidate.name.removeprefix("video")
            if suffix.isdigit():
                indices.append(int(suffix))
        return indices

    @staticmethod
    def _sys_video_name(index: int) -> str:
        name_path = Path(f"/sys/class/video4linux/video{index}/name")
        try:
            return name_path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    @staticmethod
    def _usb_device_label(index: int) -> str:
        device_path = Path(f"/dev/video{index}")
        if not device_path.exists():
            return ""
        for directory in (Path("/dev/v4l/by-id"), Path("/dev/v4l/by-path")):
            if not directory.exists():
                continue
            for candidate in directory.iterdir():
                try:
                    if candidate.resolve() == device_path:
                        label = candidate.name
                        lower = label.lower()
                        if "realsense" in lower or "intel" in lower:
                            return ""
                        return label
                except OSError:
                    continue
        sys_name = CameraController._sys_video_name(index)
        if sys_name:
            lower = sys_name.lower()
            if "realsense" in lower or "intel" in lower:
                return ""
            return sys_name
        return f"USB camera {index}"

    @staticmethod
    def _is_builtin_laptop_camera_label(label: str) -> bool:
        lower = label.lower()
        keywords = (
            "integrated_camera",
            "integrated camera",
            "chicony",
            "built-in",
            "builtin",
            "internal",
            "pci",
        )
        return any(keyword in lower for keyword in keywords)

    @classmethod
    def _probe_usb_capture(cls, source: int | str, width: int = 640, height: int = 480) -> Any | None:
        cls._require_deps()
        backends = []
        if hasattr(cv2, "CAP_V4L2"):
            backends.append(cv2.CAP_V4L2)
        backends.append(None)
        for backend in backends:
            capture = cv2.VideoCapture(source, backend) if backend is not None else cv2.VideoCapture(source)
            if not capture or not capture.isOpened():
                if capture:
                    capture.release()
                continue
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, _ = capture.read()
            if ok:
                return capture
            capture.release()
        return None

    def _usb_cameras_unlocked(self) -> list[dict[str, Any]]:
        self._require_deps()
        cameras: list[dict[str, Any]] = []
        for index in self._video_device_indices():
            if index in self.usb_captures and self.usb_labels.get(index):
                cameras.append(
                    {
                        "index": index,
                        "label": self.usb_labels[index],
                        "device": f"/dev/video{index}",
                    }
                )
                continue
            label = self._usb_device_label(index)
            if not label:
                continue
            if self._is_builtin_laptop_camera_label(label):
                continue
            capture = self._probe_usb_capture(index)
            if capture is None:
                continue
            capture.release()
            cameras.append(
                {
                    "index": index,
                    "label": label,
                    "device": f"/dev/video{index}",
                }
            )
        return cameras

    def _video_sources_unlocked(self) -> list[dict[str, str]]:
        sources: list[dict[str, str]] = []
        try:
            _, name, serial = self._first_device_unlocked()
        except Exception:
            name = None
            serial = None
        if serial is not None:
            base = name or "Intel RealSense D405"
            sources.append(
                {
                    "id": "d405:color",
                    "label": f"{base} RGB",
                    "detail": f"S/N {serial} · color",
                }
            )
            sources.append(
                {
                    "id": "d405:depth",
                    "label": f"{base} Depth",
                    "detail": f"S/N {serial} · depth",
                }
            )
        for camera in self._usb_cameras_unlocked():
            sources.append(
                {
                    "id": f"usb:{camera['index']}",
                    "label": camera["label"],
                    "detail": camera["device"],
                }
            )
        return sources

    @staticmethod
    def _parse_video_source(raw_source: Any) -> tuple[str, str | int]:
        source = str(raw_source or "").strip()
        if source in {"d405:color", "d405:depth"}:
            return ("d405", source.split(":", 1)[1])
        if source.startswith("usb:"):
            return ("usb", CameraController._validate_usb_index(source.split(":", 1)[1]))
        raise RuntimeError("Unknown camera source.")

    def status(self) -> dict[str, Any]:
        with self.lock:
            try:
                _, name, serial = self._first_device_unlocked()
                available = serial is not None
                if available:
                    self.serial = serial
                    message = self.last_message
                else:
                    message = "No RealSense camera detected"
            except Exception as exc:  # noqa: BLE001 - sent to UI.
                available = False
                name = None
                serial = None
                message = str(exc)
            return {
                "ok": True,
                "available": available,
                "running": self.pipeline is not None,
                "mode": self.mode,
                "name": name,
                "serial": serial or self.serial,
                "message": message,
            }

    def usb_status(self) -> dict[str, Any]:
        with self.lock:
            try:
                cameras = self._usb_cameras_unlocked()
                message = self.usb_last_message if cameras else "No USB camera detected"
            except Exception as exc:  # noqa: BLE001 - sent to UI.
                cameras = []
                message = str(exc)
            return {
                "ok": True,
                "cameras": cameras,
                "running": bool(self.usb_captures),
                "active_index": self.usb_index,
                "active_label": self.usb_label,
                "active_device": f"/dev/video{self.usb_index}" if self.usb_index is not None else "",
                "active_indices": sorted(self.usb_captures),
                "message": message,
            }

    def video_status(self) -> dict[str, Any]:
        with self.lock:
            sources = self._video_sources_unlocked()
            active_source = ""
            active_label = ""
            active_detail = ""
            if self.pipeline is not None:
                active_source = f"d405:{self.mode}"
            elif self.usb_captures and self.usb_index is not None:
                active_source = f"usb:{self.usb_index}"
            for source in sources:
                if source["id"] == active_source:
                    active_label = source["label"]
                    active_detail = source["detail"]
                    break
            if not active_source and sources:
                active_detail = "Preview stopped"
            message = self.last_message if self.pipeline is not None else self.usb_last_message
            if not sources:
                message = "No camera detected"
            return {
                "ok": True,
                "sources": sources,
                "running": bool(active_source),
                "active_source": active_source,
                "active_label": active_label,
                "active_detail": active_detail,
                "message": message,
            }

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode = self._validate_mode(str(payload.get("mode", self.mode)))
        width = int(payload.get("width", 640))
        height = int(payload.get("height", 480))
        with self.lock:
            self._require_deps()
            if self.pipeline is None:
                _, name, serial = self._first_device_unlocked()
                if serial is None:
                    raise RuntimeError("No Intel RealSense camera detected on USB.")
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
                config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
                pipeline = rs.pipeline()
                pipeline.start(config)
                self.pipeline = pipeline
                self.serial = serial
                self.last_message = f"Camera started: {name} ({serial}) at {width}x{height}"
            else:
                self.last_message = "Camera already running"
            self.mode = mode
            return self.status_unlocked(self.last_message)

    def stop(self) -> dict[str, Any]:
        with self.lock:
            if self.pipeline is not None:
                self.pipeline.stop()
                self.pipeline = None
            self.last_message = "Camera stopped"
            return self.status_unlocked(self.last_message)

    @staticmethod
    def _validate_usb_index(raw_index: Any) -> int:
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("USB camera index must be an integer.") from exc
        if index < 0 or index > 31:
            raise RuntimeError("USB camera index out of range.")
        return index

    def usb_start(self, payload: dict[str, Any]) -> dict[str, Any]:
        index = self._validate_usb_index(payload.get("index"))
        width = int(payload.get("width", 640))
        height = int(payload.get("height", 480))
        with self.lock:
            label = self._usb_device_label(index)
            if not label:
                raise RuntimeError("Selected camera is reserved for RealSense and not shown here.")
            if index in self.usb_captures:
                self.usb_last_message = f"USB camera already running: {label}"
                self.usb_index = index
                self.usb_label = label
                return self.usb_status_unlocked(self.usb_last_message)
            capture = self._probe_usb_capture(f"/dev/video{index}", width, height)
            if capture is None:
                raise RuntimeError(
                    f"Unable to open /dev/video{index}. Another app may own it, or this node is not a capture stream."
                )
            self.usb_captures[index] = capture
            self.usb_labels[index] = label
            self.usb_capture = capture
            self.usb_index = index
            self.usb_label = label
            self.usb_last_message = f"USB camera started: {label} (/dev/video{index}) at {width}x{height}"
            return self.usb_status_unlocked(self.usb_last_message)

    def usb_stop(self) -> dict[str, Any]:
        with self.lock:
            for capture in self.usb_captures.values():
                capture.release()
            self.usb_captures.clear()
            self.usb_labels.clear()
            self.usb_capture = None
            self.usb_index = None
            self.usb_label = None
            self.usb_last_message = "USB camera stopped"
            return self.usb_status_unlocked(self.usb_last_message)

    def video_start(self, payload: dict[str, Any]) -> dict[str, Any]:
        kind, value = self._parse_video_source(payload.get("source"))
        width = int(payload.get("width", 640))
        height = int(payload.get("height", 480))
        if kind == "d405":
            self.start({"mode": value, "width": width, "height": height})
        else:
            if self.pipeline is not None:
                self.pipeline.stop()
                self.pipeline = None
            self.usb_start({"index": value, "width": width, "height": height})
        return self.video_status()

    def video_stop(self) -> dict[str, Any]:
        with self.lock:
            if self.pipeline is not None:
                self.pipeline.stop()
                self.pipeline = None
                self.last_message = "Camera preview stopped"
            for capture in self.usb_captures.values():
                capture.release()
            self.usb_captures.clear()
            self.usb_labels.clear()
            self.usb_capture = None
            self.usb_index = None
            self.usb_label = None
            self.usb_last_message = "Camera preview stopped"
        return self.video_status()

    def set_mode(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode = self._validate_mode(str(payload.get("mode", self.mode)))
        with self.lock:
            self.mode = mode
            self.last_message = f"Camera mode: {mode}"
            return self.status_unlocked(self.last_message)

    def status_unlocked(self, message: str) -> dict[str, Any]:
        return {
            "ok": True,
            "available": self.serial is not None,
            "running": self.pipeline is not None,
            "mode": self.mode,
            "serial": self.serial,
            "message": message,
        }

    def usb_status_unlocked(self, message: str) -> dict[str, Any]:
        cameras = self._usb_cameras_unlocked()
        if self.usb_index is not None and self.usb_index not in self.usb_captures and not any(camera["index"] == self.usb_index for camera in cameras):
            self.usb_index = None
            self.usb_label = None
        return {
            "ok": True,
            "cameras": cameras,
            "running": bool(self.usb_captures),
            "active_index": self.usb_index,
            "active_label": self.usb_label,
            "active_device": f"/dev/video{self.usb_index}" if self.usb_index is not None else "",
            "active_indices": sorted(self.usb_captures),
            "message": message,
        }

    def frame_jpeg(self, mode: str, crop: Any = None) -> bytes:
        mode = self._validate_mode(mode)
        with self.lock:
            self._require_deps()
            if self.pipeline is None:
                raise RuntimeError("Camera is not running.")
            frames = self.pipeline.wait_for_frames(1000)
            if mode == "color":
                frame = frames.get_color_frame()
                if not frame:
                    raise RuntimeError("No color frame received.")
                image = np.asanyarray(frame.get_data())
            else:
                frame = frames.get_depth_frame()
                if not frame:
                    raise RuntimeError("No depth frame received.")
                depth = np.asanyarray(frame.get_data())
                depth_8bit = cv2.convertScaleAbs(depth, alpha=0.03)
                image = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_TURBO)
            image = self._apply_crop(image, crop)
            ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                raise RuntimeError("Failed to encode camera frame.")
            self.mode = mode
            return encoded.tobytes()

    def usb_frame_jpeg(self, raw_index: Any, crop: Any = None) -> bytes:
        index = self._validate_usb_index(raw_index)
        with self.lock:
            self._require_deps()
            capture = self.usb_captures.get(index)
            if capture is None:
                raise RuntimeError("USB camera preview is not running for the selected device.")
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"No frame received from USB camera {index}.")
            frame = self._apply_crop(frame, crop)
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                raise RuntimeError("Failed to encode USB camera frame.")
            return encoded.tobytes()

    def video_frame_jpeg(self, raw_source: Any, crop: Any = None) -> bytes:
        kind, value = self._parse_video_source(raw_source)
        if kind == "d405":
            return self.frame_jpeg(str(value), crop)
        return self.usb_frame_jpeg(value, crop)


class TeachRecorder:
    HIGH_SMOOTH_MOTOR_FPS = 100.0
    HIGH_SMOOTH_CAMERA_FPS = 30.0

    def __init__(self, arm: ArmController, camera: CameraController, root: Path) -> None:
        self.arm = arm
        self.camera = camera
        self.root = root
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.threads: list[threading.Thread] = []
        self.stop_event = threading.Event()
        self.running = False
        self.samples = 0
        self.motor_samples = 0
        self.session_dir: Path | None = None
        self.camera_mode = "color"
        self.fps = 20.0
        self.motor_fps = 20.0
        self.recording_mode = "standard"
        self.with_camera = True
        self.video_source: str | None = None
        self.capture_sources: list[dict[str, Any]] = []
        self.trajectory_overlay: TrajectoryOverlay | None = None
        self.latest_qpos: list[float] | None = None
        self.latest_gripper_position: float | None = None
        self.latest_motor_index = -1
        self.latest_motor_timestamp: float | None = None
        self.motor_history: deque[dict[str, Any]] = deque(maxlen=500)
        self.last_message = "Teaching recorder ready"
        self.last_error: str | None = None

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "ok": True,
                "running": self.running,
                "samples": self.samples,
                "motor_samples": self.motor_samples,
                "session_dir": str(self.session_dir) if self.session_dir else None,
                "fps": self.fps,
                "motor_fps": self.motor_fps,
                "camera_mode": self.camera_mode,
                "with_camera": self.with_camera,
                "video_source": self.video_source,
                "capture_sources": self.capture_sources,
                "trajectory_overlay": self.trajectory_overlay.metadata() if self.trajectory_overlay else None,
                "mode": self.recording_mode,
                "message": self.last_error or self.last_message,
            }

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        high_smooth = bool(payload.get("high_smooth", False))
        fps = self.HIGH_SMOOTH_CAMERA_FPS if high_smooth else float(payload.get("fps", 20))
        motor_fps = self.HIGH_SMOOTH_MOTOR_FPS if high_smooth else fps
        if not 1 <= fps <= 50:
            raise RuntimeError("Recording camera fps must be between 1 and 50.")
        if not 1 <= motor_fps <= 200:
            raise RuntimeError("Recording motor fps must be between 1 and 200.")
        with_camera = bool(payload.get("with_camera", True))
        camera_mode = CameraController._validate_mode(str(payload.get("camera_mode", "color")))
        video_source = str(payload.get("video_source", "")).strip() or None
        raw_capture_sources = payload.get("capture_sources")
        capture_sources: list[dict[str, Any]] = []
        if isinstance(raw_capture_sources, list):
            for entry in raw_capture_sources:
                if not isinstance(entry, dict):
                    continue
                source = str(entry.get("source", "")).strip()
                role = str(entry.get("role", "")).strip()
                if not source or not role:
                    continue
                capture_sources.append(
                    {
                        "source": source,
                        "role": role,
                        "crop": CameraController.validate_crop(entry.get("crop")),
                    }
                )
        if not capture_sources and video_source:
            capture_sources = [{"source": video_source, "role": "main", "crop": None}]
        trajectory_overlay = TrajectoryOverlay.from_payload(payload.get("trajectory_overlay"))
        session_name = str(payload.get("session_name", "")).strip()
        if not session_name:
            session_name = datetime.now().strftime("teach_%Y%m%d_%H%M%S")
        if any(ch in session_name for ch in "/\\"):
            raise RuntimeError("Session name cannot contain path separators.")

        # Validate devices before creating a session directory.
        self.arm.positions_for_recording()
        if with_camera:
            for entry in capture_sources:
                source = entry["source"]
                kind, value = CameraController._parse_video_source(source)
                if kind == "d405":
                    self.camera.start({"mode": str(value)})
                else:
                    self.camera.usb_start({"index": value})

        with self.lock:
            if self.running:
                raise RuntimeError("Recording is already running.")
            self.root.mkdir(parents=True, exist_ok=True)
            session_dir = self.root / session_name
            if session_dir.exists():
                session_dir = self.root / f"{session_name}_{int(time.time())}"
            (session_dir / "images").mkdir(parents=True, exist_ok=True)
            metadata = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "fps": fps,
                "motor_fps": motor_fps,
                "camera_mode": camera_mode,
                "with_camera": with_camera,
                "video_source": video_source,
                "capture_sources": capture_sources,
                "trajectory_overlay": trajectory_overlay.metadata() if trajectory_overlay else None,
                "capture_type": str(payload.get("capture_type", "manual")),
                "source_recording": payload.get("source_recording"),
                "task_name": str(payload.get("task_name", "")).strip() or None,
                "dataset_profile": str(payload.get("dataset_profile", "")).strip() or None,
                "action_offset": int(payload.get("action_offset", 1)),
                "lerobot_features": payload.get("lerobot_features"),
                "robot_ip": self.arm.args.ip,
                "variant": self.arm.args.variant,
                "mode": "high_smooth" if high_smooth else "standard",
                "format": (
                    "motor_samples.jsonl 100 Hz + JPEG frames 30 Hz, each image row linked to nearest motor timestamp"
                    if high_smooth and with_camera
                    else "motor_samples.jsonl 100 Hz, no camera"
                    if high_smooth
                    else "jsonl qpos + JPEG frames"
                    if with_camera
                    else "jsonl qpos, no camera"
                ),
                "sync": {
                    "camera_timestamp": "midpoint between frame capture start/end",
                    "motor_match": "nearest motor sample by timestamp",
                    "sync_delta_seconds": "timestamp - motor_timestamp on each camera row",
                    "frame_timestamps": "per captured camera role, measured immediately after JPEG capture",
                },
            }
            (session_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            self.stop_event.clear()
            self.samples = 0
            self.motor_samples = 0
            self.session_dir = session_dir
            self.camera_mode = camera_mode
            self.fps = fps
            self.motor_fps = motor_fps
            self.recording_mode = "high_smooth" if high_smooth else "standard"
            self.with_camera = with_camera
            self.video_source = video_source
            self.capture_sources = capture_sources
            self.trajectory_overlay = trajectory_overlay
            self.latest_qpos = None
            self.latest_gripper_position = None
            self.latest_motor_index = -1
            self.latest_motor_timestamp = None
            self.motor_history.clear()
            self.last_error = None
            self.running = True
            if high_smooth:
                self.threads = [threading.Thread(target=self._run_high_motor, daemon=True)]
                if with_camera:
                    self.threads.append(threading.Thread(target=self._run_high_camera, daemon=True))
                for thread in self.threads:
                    thread.start()
                self.thread = None
            else:
                self.thread = threading.Thread(target=self._run, daemon=True)
                self.thread.start()
                self.threads = [self.thread]
            self.last_message = f"Recording started in {session_dir}"
            return self.status_unlocked(self.last_message)

    def stop(self) -> dict[str, Any]:
        threads: list[threading.Thread]
        with self.lock:
            threads = list(self.threads)
            if not self.running:
                return self.status_unlocked("Recording is not running.")
            self.stop_event.set()
        for thread in threads:
            thread.join(timeout=3.0)
        with self.lock:
            self.running = False
            self.thread = None
            self.threads = []
            self.last_message = (
                f"Recording stopped with {self.motor_samples} motor samples and {self.samples} images"
                if self.recording_mode == "high_smooth"
                else f"Recording stopped with {self.samples} samples"
            )
            return self.status_unlocked(self.last_message)

    def status_unlocked(self, message: str) -> dict[str, Any]:
        return {
            "ok": True,
            "running": self.running,
            "samples": self.samples,
            "motor_samples": self.motor_samples,
            "session_dir": str(self.session_dir) if self.session_dir else None,
            "fps": self.fps,
            "motor_fps": self.motor_fps,
            "camera_mode": self.camera_mode,
            "with_camera": self.with_camera,
            "video_source": self.video_source,
            "capture_sources": self.capture_sources,
            "trajectory_overlay": self.trajectory_overlay.metadata() if self.trajectory_overlay else None,
            "mode": self.recording_mode,
            "message": message,
        }

    def _capture_frame_jpeg(self) -> bytes:
        if self.video_source:
            return self.camera.video_frame_jpeg(self.video_source)
        return self.camera.frame_jpeg(self.camera_mode)

    def _capture_frame_set(self) -> dict[str, bytes]:
        if not self.capture_sources:
            return {"main": self._capture_frame_jpeg()}
        frames: dict[str, bytes] = {}
        for entry in self.capture_sources:
            role = entry["role"]
            frames[role] = self._apply_trajectory_overlay(role, self.camera.video_frame_jpeg(entry["source"], entry.get("crop")))
        return frames

    def _apply_trajectory_overlay(self, role: str, frame: bytes) -> bytes:
        overlay = self.trajectory_overlay
        return overlay.apply_jpeg(frame) if overlay and overlay.role == role else frame

    def _capture_frame_set_with_timestamps(self) -> tuple[dict[str, bytes], dict[str, float], float, float]:
        capture_start = time.time()
        if not self.capture_sources:
            frame = self._apply_trajectory_overlay("main", self._capture_frame_jpeg())
            capture_end = time.time()
            return {"main": frame}, {"main": capture_end}, capture_start, capture_end
        frames: dict[str, bytes] = {}
        frame_timestamps: dict[str, float] = {}
        for entry in self.capture_sources:
            role = entry["role"]
            frames[role] = self._apply_trajectory_overlay(
                role,
                self.camera.video_frame_jpeg(entry["source"], entry.get("crop")),
            )
            frame_timestamps[role] = time.time()
        capture_end = time.time()
        return frames, frame_timestamps, capture_start, capture_end

    def _nearest_motor_sample_unlocked(self, timestamp: float) -> dict[str, Any] | None:
        if not self.motor_history:
            return None
        return min(self.motor_history, key=lambda sample: abs(float(sample["timestamp"]) - timestamp))

    def _run(self) -> None:
        assert self.session_dir is not None
        sample_path = self.session_dir / "samples.jsonl"
        period = 1.0 / self.fps
        next_tick = time.monotonic()
        try:
            with sample_path.open("a", encoding="utf-8") as file:
                while not self.stop_event.is_set():
                    timestamp = time.time()
                    state = self.arm.state_for_recording()
                    qpos = state["qpos"]
                    gripper_position = state["gripper_position"]
                    with self.lock:
                        index = self.samples
                        self.samples += 1
                        with_camera = self.with_camera
                    image_ref = None
                    images_ref = None
                    frame_timestamps = None
                    capture_start = None
                    capture_end = None
                    if with_camera:
                        frames, frame_timestamps, capture_start, capture_end = self._capture_frame_set_with_timestamps()
                        timestamp = (capture_start + capture_end) / 2.0
                        images_ref = {}
                        for role, frame in frames.items():
                            image_name = f"frame_{index:06d}_{role}.jpg"
                            image_path = self.session_dir / "images" / image_name
                            image_path.write_bytes(frame)
                            images_ref[role] = f"images/{image_name}"
                        image_ref = images_ref.get("top_view") or next(iter(images_ref.values()), None)
                    record = {
                        "index": index,
                        "timestamp": timestamp,
                        "qpos": qpos,
                        "gripper_position": gripper_position,
                        "image": image_ref,
                        "images": images_ref,
                        "camera_mode": self.camera_mode,
                        "frame_timestamps": frame_timestamps,
                        "capture_start_timestamp": capture_start,
                        "capture_end_timestamp": capture_end,
                    }
                    file.write(json.dumps(record) + "\n")
                    file.flush()

                    next_tick += period
                    sleep_time = next_tick - time.monotonic()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        next_tick = time.monotonic()
        except Exception as exc:  # noqa: BLE001 - surfaced in UI status.
            with self.lock:
                self.last_error = f"Recording failed: {exc}"
                self.running = False

    def _run_high_motor(self) -> None:
        assert self.session_dir is not None
        motor_path = self.session_dir / "motor_samples.jsonl"
        period = 1.0 / self.motor_fps
        next_tick = time.monotonic()
        try:
            with motor_path.open("a", encoding="utf-8") as file:
                while not self.stop_event.is_set():
                    timestamp = time.time()
                    state = self.arm.state_for_recording()
                    qpos = state["qpos"]
                    gripper_position = state["gripper_position"]
                    with self.lock:
                        index = self.motor_samples
                        self.motor_samples += 1
                        self.latest_qpos = qpos
                        self.latest_gripper_position = gripper_position
                        self.latest_motor_index = index
                        self.latest_motor_timestamp = timestamp
                        self.motor_history.append(
                            {
                                "index": index,
                                "timestamp": timestamp,
                                "qpos": qpos,
                                "gripper_position": gripper_position,
                            }
                        )
                    file.write(
                        json.dumps(
                            {
                                "index": index,
                                "timestamp": timestamp,
                                "qpos": qpos,
                                "gripper_position": gripper_position,
                            }
                        )
                        + "\n"
                    )
                    file.flush()

                    next_tick += period
                    sleep_time = next_tick - time.monotonic()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        next_tick = time.monotonic()
        except Exception as exc:  # noqa: BLE001 - surfaced in UI status.
            with self.lock:
                self.last_error = f"Motor recording failed: {exc}"
                self.running = False
                self.stop_event.set()

    def _run_high_camera(self) -> None:
        assert self.session_dir is not None
        sample_path = self.session_dir / "samples.jsonl"
        period = 1.0 / self.fps
        next_tick = time.monotonic()
        try:
            with sample_path.open("a", encoding="utf-8") as file:
                while not self.stop_event.is_set():
                    frames, frame_timestamps, capture_start, capture_end = self._capture_frame_set_with_timestamps()
                    timestamp = (capture_start + capture_end) / 2.0
                    with self.lock:
                        nearest_motor = self._nearest_motor_sample_unlocked(timestamp)
                        if nearest_motor is None:
                            qpos = self.latest_qpos or self.arm.positions.tolist()
                            gripper_position = self.latest_gripper_position
                            motor_index = self.latest_motor_index
                            motor_timestamp = self.latest_motor_timestamp
                        else:
                            qpos = nearest_motor["qpos"]
                            gripper_position = nearest_motor["gripper_position"]
                            motor_index = nearest_motor["index"]
                            motor_timestamp = nearest_motor["timestamp"]
                        index = self.samples
                        self.samples += 1
                    sync_delta = timestamp - motor_timestamp if motor_timestamp is not None else None
                    images_ref = {}
                    for role, frame in frames.items():
                        image_name = f"frame_{index:06d}_{role}.jpg"
                        image_path = self.session_dir / "images" / image_name
                        image_path.write_bytes(frame)
                        images_ref[role] = f"images/{image_name}"
                    record = {
                        "index": index,
                        "timestamp": timestamp,
                        "qpos": qpos,
                        "gripper_position": gripper_position,
                        "motor_index": motor_index,
                        "motor_timestamp": motor_timestamp,
                        "sync_delta_seconds": sync_delta,
                        "frame_timestamps": frame_timestamps,
                        "capture_start_timestamp": capture_start,
                        "capture_end_timestamp": capture_end,
                        "image": images_ref.get("top_view") or next(iter(images_ref.values()), None),
                        "images": images_ref,
                        "camera_mode": self.camera_mode,
                    }
                    file.write(json.dumps(record) + "\n")
                    file.flush()

                    next_tick += period
                    sleep_time = next_tick - time.monotonic()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        next_tick = time.monotonic()
        except Exception as exc:  # noqa: BLE001 - surfaced in UI status.
            with self.lock:
                self.last_error = f"Camera recording failed: {exc}"
                self.running = False
                self.stop_event.set()


class RecordingLibrary:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _resolve_root(self, raw_root: str | None = None) -> Path:
        if not raw_root:
            return self.root
        path = Path(raw_root).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        resolved = path.resolve()
        home_root = Path.home().resolve()
        inside_allowed_root = any(
            resolved == allowed_root or allowed_root in resolved.parents
            for allowed_root in (PROJECT_ROOT, home_root)
        )
        if not inside_allowed_root:
            raise RuntimeError("Dataset source folder must be inside your home or project directory.")
        if not resolved.exists() or not resolved.is_dir():
            raise RuntimeError("Dataset source folder not found.")
        return resolved

    def _resolve_session(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.root / path
        resolved = path.resolve()
        inside_default_root = resolved == self.root or self.root in resolved.parents
        home_root = Path.home().resolve()
        inside_allowed_root = any(
            resolved == allowed_root or allowed_root in resolved.parents
            for allowed_root in (PROJECT_ROOT, home_root)
        )
        if not inside_default_root and not inside_allowed_root:
            raise RuntimeError("Recording path is outside your home or project directory.")
        if not resolved.exists() or not resolved.is_dir():
            raise RuntimeError("Recording session not found.")
        return resolved

    @staticmethod
    def _fps_from_timestamps(timestamps: list[float]) -> float | None:
        valid = [value for value in timestamps if math.isfinite(value)]
        if len(valid) < 2:
            return None
        duration = valid[-1] - valid[0]
        if duration <= 0:
            return None
        return (len(valid) - 1) / duration

    @staticmethod
    def _jsonl_count_and_timestamps(path: Path) -> tuple[int, list[float]]:
        count = 0
        timestamps: list[float] = []
        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                count += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    timestamp = float(record["timestamp"])
                except (KeyError, TypeError, ValueError):
                    continue
                timestamps.append(timestamp)
        return count, timestamps

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not path.exists():
            return rows
        with path.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")

    @staticmethod
    def _safe_session_name(raw_name: Any, fallback: str) -> str:
        name = str(raw_name or fallback).strip()
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        cleaned = "".join(ch if ch in allowed else "_" for ch in name)
        return cleaned.strip("._-") or fallback

    def list(self, root: str | None = None) -> dict[str, Any]:
        source_root = self._resolve_root(root)
        source_root.mkdir(parents=True, exist_ok=True)
        recordings = []
        root_is_session = (source_root / "samples.jsonl").exists() or (source_root / "motor_samples.jsonl").exists()
        sessions = [source_root] if root_is_session else sorted(
            source_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
        )
        for session in sessions:
            if not session.is_dir():
                continue
            samples_path = session / "samples.jsonl"
            motor_path = session / "motor_samples.jsonl"
            metadata_path = session / "metadata.json"
            if not samples_path.exists() and not motor_path.exists():
                continue
            samples = 0
            sample_timestamps: list[float] = []
            if samples_path.exists():
                samples, sample_timestamps = self._jsonl_count_and_timestamps(samples_path)
            motor_samples = 0
            motor_timestamps: list[float] = []
            if motor_path.exists():
                motor_samples, motor_timestamps = self._jsonl_count_and_timestamps(motor_path)
            metadata = {}
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    metadata = {}
            actual_camera_fps = self._fps_from_timestamps(sample_timestamps)
            actual_motor_fps = self._fps_from_timestamps(motor_timestamps)
            recordings.append(
                {
                    "name": session.name,
                    "path": str(session),
                    "samples": samples,
                    "motor_samples": motor_samples or samples,
                    "actual_camera_fps": actual_camera_fps,
                    "actual_motor_fps": actual_motor_fps,
                    "nominal_camera_fps": metadata.get("fps"),
                    "nominal_motor_fps": metadata.get("motor_fps"),
                    "created_at": metadata.get("created_at"),
                    "camera_mode": metadata.get("camera_mode"),
                    "mode": metadata.get("mode", "standard"),
                    "with_camera": metadata.get("with_camera", True),
                    "video_source": metadata.get("video_source"),
                    "capture_sources": metadata.get("capture_sources", []),
                    "task_name": metadata.get("task_name"),
                    "capture_type": metadata.get("capture_type", "manual"),
                    "source_recording": metadata.get("source_recording"),
                }
            )
        return {"ok": True, "root": str(source_root), "default_root": str(self.root), "recordings": recordings}

    def load(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = self._resolve_session(str(payload.get("path", "")))
        frames = self.load_frames(session)
        frame_timestamps: list[float] = []
        for frame in frames:
            try:
                frame_timestamps.append(float(frame["timestamp"]))
            except (KeyError, TypeError, ValueError):
                continue
        motor_timestamps: list[float] = []
        motor_path = session / "motor_samples.jsonl"
        if motor_path.exists():
            _, motor_timestamps = self._jsonl_count_and_timestamps(motor_path)
        metadata_path = session / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}
        return {
            "ok": True,
            "name": session.name,
            "path": str(session),
            "frames": frames,
            "actual_camera_fps": self._fps_from_timestamps(frame_timestamps),
            "actual_motor_fps": self._fps_from_timestamps(motor_timestamps),
            "nominal_camera_fps": metadata.get("fps"),
            "nominal_motor_fps": metadata.get("motor_fps"),
            "metadata": metadata,
        }

    def load_frames(self, session: Path) -> list[dict[str, Any]]:
        samples_path = session / "samples.jsonl"
        if not samples_path.exists():
            motor_path = session / "motor_samples.jsonl"
            if not motor_path.exists():
                raise RuntimeError("samples.jsonl not found for this recording.")
            frames = []
            with motor_path.open(encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    frames.append(
                        {
                            "index": record.get("index"),
                            "timestamp": record.get("timestamp"),
                            "qpos": record.get("qpos", []),
                            "gripper_position": record.get("gripper_position"),
                            "motor_index": record.get("motor_index"),
                            "motor_timestamp": record.get("motor_timestamp"),
                            "image": None,
                            "images": {},
                            "camera_mode": None,
                            "sync_delta_seconds": None,
                            "frame_timestamps": {},
                            "capture_start_timestamp": None,
                            "capture_end_timestamp": None,
                        }
                    )
            return frames
        frames = []
        with samples_path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                frames.append(
                    {
                        "index": record.get("index"),
                        "timestamp": record.get("timestamp"),
                        "qpos": record.get("qpos", []),
                        "gripper_position": record.get("gripper_position"),
                        "motor_index": record.get("motor_index"),
                        "motor_timestamp": record.get("motor_timestamp"),
                        "image": record.get("image"),
                        "images": record.get("images", {}),
                        "camera_mode": record.get("camera_mode"),
                        "sync_delta_seconds": record.get("sync_delta_seconds"),
                        "frame_timestamps": record.get("frame_timestamps", {}),
                        "capture_start_timestamp": record.get("capture_start_timestamp"),
                        "capture_end_timestamp": record.get("capture_end_timestamp"),
                    }
                )
        return frames

    def replay_frames(self, raw_path: str) -> list[dict[str, Any]]:
        session = self._resolve_session(raw_path)
        motor_path = session / "motor_samples.jsonl"
        if not motor_path.exists():
            return self.load_frames(session)
        frames = []
        with motor_path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                frames.append(
                    {
                        "index": record.get("index"),
                        "timestamp": record.get("timestamp"),
                        "qpos": record.get("qpos", []),
                        "gripper_position": record.get("gripper_position"),
                        "image": None,
                        "images": {},
                        "camera_mode": None,
                    }
                )
        return frames

    def image(self, session_path: str, image_path: str) -> bytes:
        session = self._resolve_session(session_path)
        image = (session / image_path).resolve()
        if image != session and session not in image.parents:
            raise RuntimeError("Image path is outside the recording session.")
        if not image.exists() or not image.is_file():
            raise RuntimeError("Recording image not found.")
        return image.read_bytes()

    def trim(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = self._resolve_session(str(payload.get("path", "")))
        output_root = self._resolve_root(str(payload.get("source_root") or session.parent))
        samples_path = session / "samples.jsonl"
        motor_path = session / "motor_samples.jsonl"
        samples = self._read_jsonl(samples_path)
        motor = self._read_jsonl(motor_path)
        source_rows = samples or motor
        if not source_rows:
            raise RuntimeError("Selected recording has no samples to trim.")

        cut_start = int(payload.get("cut_start") or 0)
        cut_end = int(payload.get("cut_end") or 0)
        if cut_start < 0 or cut_end < 0:
            raise RuntimeError("Trim values must be >= 0.")
        keep_start = cut_start
        keep_end = len(source_rows) - cut_end
        if keep_start >= keep_end:
            raise RuntimeError(
                f"Trim removes all frames: total={len(source_rows)}, first={cut_start}, last={cut_end}."
            )

        default_name = f"{session.name}_trim_{cut_start}_{cut_end}"
        output_name = self._safe_session_name(payload.get("output_name"), default_name)
        output = (output_root / output_name).resolve()
        if output.exists():
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = (output_root / f"{output_name}_{suffix}").resolve()
        if output != output_root and output_root not in output.parents:
            raise RuntimeError("Trim output path is outside the selected source folder.")
        output.mkdir(parents=True)

        metadata_path = session / "metadata.json"
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}
        metadata.update(
            {
                "source_recording": str(session),
                "trim_source_recording": str(session),
                "trim_first_frames": cut_start,
                "trim_last_frames": cut_end,
                "trim_original_samples": len(samples),
                "trim_original_motor_samples": len(motor),
                "trim_created_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if samples:
            kept_samples = [dict(row) for row in samples[keep_start:keep_end]]
            old_motor_indices = []
            for row in kept_samples:
                try:
                    motor_index = int(row.get("motor_index"))
                except (TypeError, ValueError):
                    continue
                if motor_index >= 0:
                    old_motor_indices.append(motor_index)
            motor_offset = min(old_motor_indices) if old_motor_indices else 0
            motor_keep_end = max(old_motor_indices) + 2 if old_motor_indices else len(motor)
            motor_keep_end = min(max(motor_keep_end, motor_offset), len(motor))
            kept_motor = [dict(row) for row in motor[motor_offset:motor_keep_end]] if motor else []
            old_to_new_motor = {}
            for new_index, row in enumerate(kept_motor):
                old_index = int(row.get("index", motor_offset + new_index))
                old_to_new_motor[old_index] = new_index
                row["index"] = new_index
            for new_index, row in enumerate(kept_samples):
                row["index"] = new_index
                old_motor = row.get("motor_index")
                if old_motor is not None:
                    try:
                        row["motor_index"] = old_to_new_motor.get(int(old_motor), max(0, int(old_motor) - motor_offset))
                    except (TypeError, ValueError):
                        pass
                self._copy_sample_images(session, output, row)
            self._write_jsonl(output / "samples.jsonl", kept_samples)
            if kept_motor:
                self._write_jsonl(output / "motor_samples.jsonl", kept_motor)
        else:
            kept_motor = [dict(row) for row in motor[keep_start:keep_end]]
            for new_index, row in enumerate(kept_motor):
                row["index"] = new_index
            self._write_jsonl(output / "motor_samples.jsonl", kept_motor)

        return {
            "ok": True,
            "source": str(session),
            "session_dir": str(output),
            "message": f"Created trimmed dataset {output.name}",
            "kept_frames": keep_end - keep_start,
            "removed_start": cut_start,
            "removed_end": cut_end,
        }

    @staticmethod
    def _copy_sample_images(source_session: Path, output_session: Path, row: dict[str, Any]) -> None:
        image_refs = set()
        image = row.get("image")
        if image:
            image_refs.add(str(image))
        images = row.get("images")
        if isinstance(images, dict):
            image_refs.update(str(value) for value in images.values() if value)
        for rel_path in image_refs:
            source = (source_session / rel_path).resolve()
            if not source.exists() or source_session not in source.parents:
                continue
            dest = output_session / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(source, dest)

    def delete(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = self._resolve_session(str(payload.get("path", "")))
        name = session.name
        shutil.rmtree(session)
        return {"ok": True, "message": f"Deleted recording {name}"}

    def clear(self) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        count = 0
        for session in self.root.iterdir():
            if not session.is_dir():
                continue
            samples_path = session / "samples.jsonl"
            motor_path = session / "motor_samples.jsonl"
            if not samples_path.exists() and not motor_path.exists():
                continue
            shutil.rmtree(session)
            count += 1
        return {"ok": True, "message": f"Deleted {count} recording(s)"}


class ArmController:
    def __init__(self, args: argparse.Namespace, start_position_path: Path) -> None:
        self.args = args
        self.start_position_path = start_position_path
        self.lock = threading.Lock()
        self.driver: trossen_arm.TrossenArmDriver | None = None
        self.positions = HOME.copy()
        self.gripper_position: float | None = None
        self.connected = False
        self.gravity_compensation_enabled = False
        self.hold_enabled = False
        self.max_speed = DEFAULT_MAX_SPEED
        self.gravity_payload_profile = DEFAULT_GRAVITY_PAYLOAD
        self.camera_wrist_effort = 0.0
        self.start_position: np.ndarray | None = None
        self.start_gripper_position: float | None = None
        self.last_message = "Ready"
        self._load_start_position()

    def _load_start_position(self) -> None:
        if not self.start_position_path.exists():
            return
        try:
            payload = json.loads(self.start_position_path.read_text(encoding="utf-8"))
            positions = self._validated_positions(payload.get("positions"))
            gripper_position = self._validated_gripper_position(payload.get("gripper_position"))
        except Exception:
            return
        self.start_position = positions
        self.start_gripper_position = gripper_position

    def _save_start_position_file(self) -> None:
        if self.start_position is None:
            return
        payload = {
            "positions": self.start_position.tolist(),
            "gripper_position": self.start_gripper_position,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.start_position_path.parent.mkdir(parents=True, exist_ok=True)
        self.start_position_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def config(self) -> dict[str, Any]:
        return {
            "real": self.args.real,
            "ip": self.args.ip,
            "port": self.args.arm_port,
            "variant": self.args.variant,
        }

    def status(self, message: str | None = None) -> dict[str, Any]:
        with self.lock:
            if self.connected and self.driver is not None and self.args.real:
                try:
                    positions = np.asarray(self.driver.get_arm_positions(), dtype=float)
                    if positions.size >= 6:
                        self.positions = positions[:6]
                    self.gripper_position = float(self.driver.get_gripper_position())
                except Exception as exc:  # noqa: BLE001 - driver exceptions are pybind runtime errors.
                    self.connected = False
                    self.last_message = f"Read failed: {exc}"
            return {
                "ok": True,
                "connected": self.connected,
                "positions": self.positions.tolist(),
                "gripper_position": self.gripper_position,
                "config": self.config(),
                "gravity_compensation": self.gravity_compensation_enabled,
                "hold": self.hold_enabled,
                "max_speed": self.max_speed,
                "gravity_payload_profile": self.gravity_payload_profile,
                "camera_wrist_effort": self.camera_wrist_effort,
                "start_position_saved": self.start_position is not None,
                "start_position_label": self._start_position_label_unlocked(),
                "message": message or self.last_message,
            }

    def _preflight_tcp(self) -> None:
        try:
            with socket.create_connection((self.args.ip, self.args.arm_port), timeout=self.args.timeout):
                return
        except OSError as exc:
            raise RuntimeError(
                f"Cannot reach {self.args.ip}:{self.args.arm_port}. Check Ethernet static IP and arm power."
            ) from exc

    def connect(self) -> dict[str, Any]:
        with self.lock:
            if self.connected:
                return self.status_unlocked("Already connected")
            if not self.args.real:
                self.connected = True
                self.last_message = "Dry-run connected"
                return self.status_unlocked(self.last_message)

            self._preflight_tcp()
            driver = trossen_arm.TrossenArmDriver()
            end_effector = END_EFFECTORS[self.args.variant]
            try:
                driver.configure(trossen_arm.Model.wxai_v0, end_effector, self.args.ip, True, self.args.timeout)
            except TypeError:
                driver.configure(trossen_arm.Model.wxai_v0, end_effector, self.args.ip, True)
            self.driver = driver
            self.connected = True
            self.gravity_compensation_enabled = False
            self.hold_enabled = False
            self.last_message = "Arm connected"
            return self.status_unlocked(self.last_message)

    def disconnect(self) -> dict[str, Any]:
        with self.lock:
            self.driver = None
            self.connected = False
            self.gravity_compensation_enabled = False
            self.hold_enabled = False
            self.last_message = "Disconnected"
            return self.status_unlocked(self.last_message)

    def status_unlocked(self, message: str) -> dict[str, Any]:
        return {
            "ok": True,
            "connected": self.connected,
            "positions": self.positions.tolist(),
            "gripper_position": self.gripper_position,
            "config": self.config(),
            "gravity_compensation": self.gravity_compensation_enabled,
            "hold": self.hold_enabled,
            "max_speed": self.max_speed,
            "gravity_payload_profile": self.gravity_payload_profile,
            "camera_wrist_effort": self.camera_wrist_effort,
            "start_position_saved": self.start_position is not None,
            "start_position_label": self._start_position_label_unlocked(),
            "message": message,
        }

    def _start_position_label_unlocked(self) -> str | None:
        if self.start_position is None:
            return None
        return ", ".join(f"{value:.2f}" for value in self.start_position.tolist())

    def positions_for_recording(self) -> list[float]:
        return self.state_for_recording()["qpos"]

    def state_for_recording(self) -> dict[str, Any]:
        with self.lock:
            if not self.connected or self.driver is None:
                raise RuntimeError("Arm must be connected before recording.")
            if self.args.real:
                positions = np.asarray(self.driver.get_arm_positions(), dtype=float)
                if positions.size >= 6:
                    self.positions = positions[:6]
                self.gripper_position = float(self.driver.get_gripper_position())
            return {
                "qpos": self.positions.tolist(),
                "gripper_position": self.gripper_position,
            }

    def set_max_speed(self, payload: dict[str, Any]) -> dict[str, Any]:
        max_speed = float(payload.get("max_speed", self.max_speed))
        if not MIN_MAX_SPEED <= max_speed <= MAX_MAX_SPEED:
            raise RuntimeError(
                f"Max speed must be between {MIN_MAX_SPEED:.2f} and {MAX_MAX_SPEED:.2f} rad/s."
            )
        with self.lock:
            self.max_speed = max_speed
            self.last_message = f"Max speed set to {self.max_speed:.2f} rad/s"
            return self.status_unlocked(self.last_message)

    @staticmethod
    def _validate_gravity_payload_profile(raw_profile: Any) -> str:
        profile = str(raw_profile or DEFAULT_GRAVITY_PAYLOAD)
        if profile not in GRAVITY_PAYLOADS:
            raise RuntimeError("Unknown gravity payload profile.")
        return profile

    @staticmethod
    def _validate_camera_wrist_effort(raw_effort: Any) -> float:
        effort = float(raw_effort or 0.0)
        if not MIN_CAMERA_WRIST_EFFORT <= effort <= MAX_CAMERA_WRIST_EFFORT:
            raise RuntimeError(
                f"Camera wrist effort must be between {MIN_CAMERA_WRIST_EFFORT:.2f} "
                f"and {MAX_CAMERA_WRIST_EFFORT:.2f} Nm."
            )
        return effort

    def _apply_gravity_payload_profile(self, profile: str) -> None:
        if not self.args.real or self.driver is None:
            return
        end_effector = GRAVITY_PAYLOADS[profile] or END_EFFECTORS[self.args.variant]
        self.driver.set_end_effector(end_effector)

    @staticmethod
    def _camera_compensated_efforts(camera_wrist_effort: float) -> list[float]:
        efforts = [0.0] * 7
        efforts[4] = camera_wrist_effort
        return efforts

    def _enable_gravity_compensation_unlocked(self, profile: str, camera_wrist_effort: float) -> None:
        if self.args.real and self.driver is not None:
            self._apply_gravity_payload_profile(profile)
            self.driver.set_all_modes(trossen_arm.Mode.external_effort)
            self.driver.set_all_external_efforts(
                self._camera_compensated_efforts(camera_wrist_effort),
                0.0,
                False,
            )
        self.gravity_compensation_enabled = True
        self.hold_enabled = False
        self.gravity_payload_profile = profile
        self.camera_wrist_effort = camera_wrist_effort

    def _require_motion(self, payload: dict[str, Any]) -> None:
        if not payload.get("armed"):
            raise RuntimeError("Motion disabled. Check 'enable motion' first.")
        if not self.connected:
            raise RuntimeError("Arm is not connected.")

    @staticmethod
    def _validated_positions(raw_positions: Any) -> np.ndarray:
        positions = np.asarray(raw_positions, dtype=float)
        if positions.shape != (6,):
            raise RuntimeError("Expected 6 joint positions.")
        for idx, (value, (low, high)) in enumerate(zip(positions, JOINT_LIMITS)):
            value = float(value)
            if value < low and value >= low - JOINT_LIMIT_TOLERANCE:
                positions[idx] = low
                continue
            if value > high and value <= high + JOINT_LIMIT_TOLERANCE:
                positions[idx] = high
                continue
            if not low <= value <= high:
                raise RuntimeError(f"Joint {idx} out of range: {value:.3f} not in [{low:.3f}, {high:.3f}]")
        return positions

    @staticmethod
    def _validated_gripper_position(raw_position: Any) -> float | None:
        if raw_position is None:
            return None
        position = float(raw_position)
        if not math.isfinite(position):
            raise RuntimeError("Gripper position must be finite.")
        return position

    def move(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_motion(payload)
        positions = self._validated_positions(payload.get("positions"))
        self.move_positions(positions)
        return self.status_unlocked(self.last_message)

    def move_positions(self, positions: np.ndarray, requested_time: float | None = None) -> None:
        with self.lock:
            delta = float(np.max(np.abs(positions - self.positions)))
            base_time = requested_time if requested_time is not None else self.args.move_time
            move_time = max(base_time, delta / self.max_speed if delta > 0 else base_time)
            self.positions = positions
            if self.args.real and self.driver is not None:
                self.driver.set_arm_modes(trossen_arm.Mode.position)
                self.driver.set_arm_positions(positions, move_time, True)
            self.gravity_compensation_enabled = False
            self.hold_enabled = False
            self.last_message = f"Moved joints at max {self.max_speed:.2f} rad/s ({move_time:.2f}s)"

    def save_start_position(self) -> dict[str, Any]:
        with self.lock:
            if not self.connected or self.driver is None:
                raise RuntimeError("Arm must be connected before saving the start position.")
            if self.args.real:
                positions = np.asarray(self.driver.get_arm_positions(), dtype=float)
                if positions.size >= 6:
                    self.positions = positions[:6]
                self.gripper_position = float(self.driver.get_gripper_position())
            self.start_position = self.positions.copy()
            self.start_gripper_position = self.gripper_position
            self._save_start_position_file()
            self.last_message = "Saved current pose as start position"
            return self.status_unlocked(self.last_message)

    def go_to_start_position(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_motion(payload)
        with self.lock:
            if self.start_position is None:
                raise RuntimeError("No saved start position. Use 'Save current as start' first.")
            target = self.start_position.copy()
            target_gripper = self.start_gripper_position
            if self.args.real and self.driver is not None:
                live_positions = np.asarray(self.driver.get_arm_positions(), dtype=float)
                if live_positions.size >= 6:
                    self.positions = live_positions[:6]
                self.gripper_position = float(self.driver.get_gripper_position())
            current = self.positions.copy()
            delta = float(np.max(np.abs(target - current)))
            safe_speed = self.max_speed
            gripper_delta = (
                abs(target_gripper - self.gripper_position)
                if target_gripper is not None and self.gripper_position is not None
                else 0.0
            )
            move_time = max(
                START_POSITION_MIN_TIME,
                delta / safe_speed if delta > 0 else START_POSITION_MIN_TIME,
                gripper_delta / REPLAY_GRIPPER_MAX_SPEED if gripper_delta > 0 else START_POSITION_MIN_TIME,
            )
            self.positions = target
            if target_gripper is not None:
                self.gripper_position = target_gripper
            if self.args.real and self.driver is not None:
                self.driver.set_arm_modes(trossen_arm.Mode.position)
                self.driver.set_gripper_mode(trossen_arm.Mode.position)
                self.driver.set_arm_positions(target, move_time, True)
                if target_gripper is not None:
                    self.driver.set_gripper_position(target_gripper, move_time, True)
            self.gravity_compensation_enabled = False
            self.hold_enabled = False
            self.last_message = (
                f"Moved to saved start position at max {safe_speed:.2f} rad/s ({move_time:.2f}s)"
            )
            return self.status_unlocked(self.last_message)

    def replay_position(
        self,
        positions: np.ndarray,
        requested_time: float,
        gripper_position: float | None = None,
    ) -> float:
        with self.lock:
            delta = float(np.max(np.abs(positions - self.positions)))
            replay_speed = self.max_speed
            gripper_delta = (
                abs(gripper_position - self.gripper_position)
                if gripper_position is not None and self.gripper_position is not None
                else 0.0
            )
            move_time = max(
                requested_time,
                delta / replay_speed if delta > 0 else requested_time,
                gripper_delta / REPLAY_GRIPPER_MAX_SPEED if gripper_delta > 0 else requested_time,
            )
            self.positions = positions
            if gripper_position is not None:
                self.gripper_position = gripper_position
            if self.args.real and self.driver is not None:
                self.driver.set_arm_positions(positions, move_time, False)
                if gripper_position is not None:
                    self.driver.set_gripper_position(gripper_position, move_time, False)
            self.gravity_compensation_enabled = False
            self.hold_enabled = False
            self.last_message = f"Replaying safely at max {replay_speed:.2f} rad/s"
            return move_time

    def replay_move_to_start(
        self,
        positions: np.ndarray,
        gripper_position: float | None,
        stop_event: threading.Event,
    ) -> float:
        with self.lock:
            current = self.positions.copy()
            current_gripper = self.gripper_position
            if self.args.real and self.driver is not None:
                live_positions = np.asarray(self.driver.get_arm_positions(), dtype=float)
                if live_positions.size >= 6:
                    current = live_positions[:6]
                    self.positions = current.copy()
                current_gripper = float(self.driver.get_gripper_position())
                self.gripper_position = current_gripper

            delta = float(np.max(np.abs(positions - current)))
            replay_speed = self.max_speed
            gripper_delta = (
                abs(gripper_position - current_gripper)
                if gripper_position is not None and current_gripper is not None
                else 0.0
            )
            move_time = max(
                1.5,
                delta / replay_speed if delta > 0 else 1.5,
                gripper_delta / REPLAY_GRIPPER_MAX_SPEED if gripper_delta > 0 else 1.5,
            )
            self.gravity_compensation_enabled = False
            self.hold_enabled = False
            self.last_message = f"Moving to replay start at max {replay_speed:.2f} rad/s ({move_time:.2f}s)"

            if self.args.real and self.driver is not None:
                self.driver.set_arm_modes(trossen_arm.Mode.position)
                self.driver.set_gripper_mode(trossen_arm.Mode.position)
                self.driver.set_arm_positions(positions, move_time, False)
                if gripper_position is not None:
                    self.driver.set_gripper_position(gripper_position, move_time, False)

            self.positions = positions.copy()
            if gripper_position is not None:
                self.gripper_position = gripper_position

        deadline = time.monotonic() + move_time
        while time.monotonic() < deadline:
            if stop_event.is_set():
                break
            time.sleep(0.02)
        return move_time

    def replay_current_positions(self) -> np.ndarray:
        with self.lock:
            if self.args.real and self.driver is not None:
                positions = np.asarray(self.driver.get_arm_positions(), dtype=float)
                if positions.size >= 6:
                    self.positions = positions[:6]
                self.gripper_position = float(self.driver.get_gripper_position())
            return self.positions.copy()

    def replay_current_gripper_position(self) -> float | None:
        with self.lock:
            if self.args.real and self.driver is not None:
                self.gripper_position = float(self.driver.get_gripper_position())
            return self.gripper_position

    def replay_speed_limit(self) -> float:
        with self.lock:
            return self.max_speed

    def prepare_replay(self) -> None:
        with self.lock:
            if self.args.real and self.driver is not None:
                self.driver.set_arm_modes(trossen_arm.Mode.position)
                self.driver.set_gripper_mode(trossen_arm.Mode.position)
            self.gravity_compensation_enabled = False
            self.hold_enabled = False

    def emergency_stop(self) -> dict[str, Any]:
        with self.lock:
            if self.args.real and self.driver is not None:
                self.driver.set_all_modes(trossen_arm.Mode.idle)
            self.gravity_compensation_enabled = False
            self.hold_enabled = False
            self.last_message = "Emergency stop: replay stopped and motors braked"
            return self.status_unlocked(self.last_message)

    def home(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        payload["positions"] = HOME.tolist()
        data = self.move(payload)
        data["message"] = "Moved to home"
        return data

    def rest(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        payload["positions"] = REST.tolist()
        data = self.move(payload)
        data["message"] = "Moved to rest position"
        return data

    def demo(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_motion(payload)
        with self.lock:
            demo_speed = self.max_speed
            if self.args.real and self.driver is not None:
                self.driver.set_arm_modes(trossen_arm.Mode.position)
                delta_home = float(np.max(np.abs(HOME - self.positions)))
                home_time = max(3.0, delta_home / demo_speed if delta_home > 0 else 3.0)
                self.driver.set_arm_positions(HOME, home_time, True)
                time.sleep(0.25)
                delta_demo = float(np.max(np.abs(DEMO - HOME)))
                demo_time = max(2.0, delta_demo / demo_speed if delta_demo > 0 else 2.0)
                self.driver.set_arm_positions(DEMO, demo_time, True)
                time.sleep(0.25)
                self.driver.set_arm_positions(HOME, demo_time, True)
            self.positions = HOME.copy()
            self.gravity_compensation_enabled = False
            self.hold_enabled = False
            self.last_message = f"Demo complete at max {demo_speed:.2f} rad/s"
            return self.status_unlocked(self.last_message)

    def gripper(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_motion(payload)
        effort = float(payload.get("effort", 0.0))
        if abs(effort) > 20:
            raise RuntimeError("Gripper effort is limited to +/-20 for this interface.")
        with self.lock:
            if self.args.real and self.driver is not None:
                self.driver.set_gripper_mode(trossen_arm.Mode.external_effort)
                self.driver.set_gripper_external_effort(effort, 2.0, True)
            self.last_message = f"Gripper effort {effort:.1f}"
            return self.status_unlocked(self.last_message)

    def gravity_compensation(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_motion(payload)
        profile = self._validate_gravity_payload_profile(payload.get("payload_profile"))
        camera_wrist_effort = self._validate_camera_wrist_effort(payload.get("camera_wrist_effort"))
        with self.lock:
            self._enable_gravity_compensation_unlocked(profile, camera_wrist_effort)
            label = "D405/follower" if profile == "d405_follower" else self.args.variant
            self.last_message = (
                f"Gravity compensation enabled ({label}, wrist {camera_wrist_effort:.2f} Nm)"
            )
            return self.status_unlocked(self.last_message)

    def hold(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_motion(payload)
        enabled = bool(payload.get("enabled", not self.hold_enabled))
        with self.lock:
            if enabled:
                if self.args.real and self.driver is not None:
                    positions = np.asarray(self.driver.get_arm_positions(), dtype=float)
                    if positions.size >= 6:
                        self.positions = positions[:6]
                    self.driver.set_arm_modes(trossen_arm.Mode.position)
                    self.driver.set_arm_positions(self.positions, self.args.move_time, True)
                    self.driver.set_gripper_mode(trossen_arm.Mode.idle)
                self.gravity_compensation_enabled = False
                self.hold_enabled = True
                self.last_message = "Hold enabled"
            else:
                profile = self._validate_gravity_payload_profile(
                    payload.get("payload_profile", self.gravity_payload_profile)
                )
                camera_wrist_effort = self._validate_camera_wrist_effort(
                    payload.get("camera_wrist_effort", self.camera_wrist_effort)
                )
                self._enable_gravity_compensation_unlocked(profile, camera_wrist_effort)
                self.last_message = "Hold off, gravity compensation enabled"
            return self.status_unlocked(self.last_message)


class ReplayRunner:
    REPLAY_HZ = 100.0
    COMMAND_HZ = 20.0

    def __init__(self, arm: ArmController, library: RecordingLibrary) -> None:
        self.arm = arm
        self.library = library
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.running = False
        self.name = ""
        self.frame_index = 0
        self.frame_count = 0
        self.last_message = "Replay ready"
        self.last_error: str | None = None

    def status(self) -> dict[str, Any]:
        with self.lock:
            return self.status_unlocked()

    def status_unlocked(self) -> dict[str, Any]:
        return {
            "ok": True,
            "running": self.running,
            "name": self.name,
            "frame_index": self.frame_index,
            "frame_count": self.frame_count,
            "message": self.last_error or self.last_message,
        }

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.arm._require_motion(payload)
        raw_path = str(payload.get("path", ""))
        if not raw_path:
            raise RuntimeError("Select a recording before replay.")
        speed = float(payload.get("speed", 1.0))
        if not 0.25 <= speed <= 1.0:
            raise RuntimeError("Replay speed must be between 0.25 and 1.0.")
        frames = self.library.replay_frames(raw_path)
        qpos_frames: list[dict[str, Any]] = []
        for frame in frames:
            qpos = frame.get("qpos", [])
            if len(qpos) != 6:
                continue
            qpos_frames.append(
                {
                    "timestamp": float(frame.get("timestamp") or 0.0),
                    "qpos": self.arm._validated_positions(qpos),
                    "gripper_position": self.arm._validated_gripper_position(
                        frame.get("gripper_position")
                    ),
                }
            )
        if not qpos_frames:
            raise RuntimeError("This recording has no replayable qpos frames.")
        trajectory = self._interpolate_trajectory(qpos_frames, speed)

        with self.lock:
            if self.running:
                raise RuntimeError("Replay is already running.")
            self.stop_event.clear()
            self.running = True
            self.frame_index = 0
            self.frame_count = len(trajectory)
            self.name = Path(raw_path).name
            self.last_error = None
            self.last_message = (
                f"Replay started: {self.name} interpolated at {self.REPLAY_HZ:.0f} Hz, "
                f"commanded at {self.COMMAND_HZ:.0f} Hz"
            )
            self.thread = threading.Thread(target=self._run, args=(trajectory,), daemon=True)
            self.thread.start()
            return self.status_unlocked()

    def stop(self) -> dict[str, Any]:
        with self.lock:
            if not self.running:
                return self.status_unlocked()
            self.stop_event.set()
            self.last_message = "Stopping replay"
            return self.status_unlocked()

    def force_stop(self) -> None:
        with self.lock:
            self.stop_event.set()
            self.running = False
            self.last_message = "Replay stopped by emergency stop"

    def _interpolate_trajectory(self, frames: list[dict[str, Any]], speed: float) -> list[dict[str, Any]]:
        if len(frames) == 1:
            return [
                {
                    "qpos": frames[0]["qpos"],
                    "gripper_position": frames[0].get("gripper_position"),
                }
            ]
        timestamps = np.asarray([frame["timestamp"] for frame in frames], dtype=float)
        qpos = np.asarray([frame["qpos"] for frame in frames], dtype=float)
        start = float(timestamps[0])
        times = timestamps - start
        duration = float(times[-1])
        if duration <= 0:
            return [
                {"qpos": row, "gripper_position": frame.get("gripper_position")}
                for row, frame in zip(qpos, frames)
            ]
        replay_duration = duration / speed
        count = max(2, int(math.ceil(replay_duration * self.REPLAY_HZ)) + 1)
        wall_times = np.arange(count, dtype=float) / self.REPLAY_HZ
        source_times = np.clip(wall_times * speed, 0.0, duration)
        interpolated = np.empty((count, 6), dtype=float)
        for joint_index in range(6):
            interpolated[:, joint_index] = np.interp(source_times, times, qpos[:, joint_index])
        gripper_values = [frame.get("gripper_position") for frame in frames]
        if all(value is not None for value in gripper_values):
            gripper_array = np.asarray(gripper_values, dtype=float)
            interpolated_gripper = np.interp(source_times, times, gripper_array)
            return [
                {"qpos": row, "gripper_position": float(gripper)}
                for row, gripper in zip(interpolated, interpolated_gripper)
            ]
        return [{"qpos": row, "gripper_position": None} for row in interpolated]

    def _run(self, trajectory: list[dict[str, Any]]) -> None:
        try:
            self.arm.prepare_replay()
            period = 1.0 / self.COMMAND_HZ
            stride = max(1, round(self.REPLAY_HZ / self.COMMAND_HZ))
            self._move_to_start(trajectory[0], period)
            next_tick = time.monotonic()
            for index in range(0, len(trajectory), stride):
                if self.stop_event.is_set():
                    break
                point = trajectory[index]
                self.arm.replay_position(
                    point["qpos"],
                    period * 1.8,
                    point.get("gripper_position"),
                )
                with self.lock:
                    self.frame_index = index + 1
                next_tick += period
                sleep_time = next_tick - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_tick = time.monotonic()
            with self.lock:
                self.running = False
                self.thread = None
                self.last_message = (
                    f"Replay stopped at frame {self.frame_index}/{self.frame_count}"
                    if self.stop_event.is_set()
                    else f"Replay complete: {self.name}"
                )
        except Exception as exc:  # noqa: BLE001 - surfaced in UI.
            with self.lock:
                self.running = False
                self.thread = None
                self.last_error = f"Replay failed: {exc}"

    def _move_to_start(self, target: dict[str, Any], period: float) -> None:
        target_qpos = target["qpos"]
        target_gripper = target.get("gripper_position")
        with self.lock:
            self.last_message = "Moving to replay start"
        self.arm.replay_move_to_start(target_qpos, target_gripper, self.stop_event)


class DatasetCaptureRunner:
    def __init__(self, recorder: TeachRecorder, replay: ReplayRunner) -> None:
        self.recorder = recorder
        self.replay = replay
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.running = False
        self.session_dir: str | None = None
        self.source_path = ""
        self.last_message = "Dataset capture ready"
        self.last_error: str | None = None

    def status(self) -> dict[str, Any]:
        with self.lock:
            return self.status_unlocked()

    def status_unlocked(self) -> dict[str, Any]:
        return {
            "ok": True,
            "running": self.running,
            "source_path": self.source_path,
            "session_dir": self.session_dir,
            "message": self.last_error or self.last_message,
        }

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw_path = str(payload.get("path", ""))
        if not raw_path:
            raise RuntimeError("Select a source movement before dataset capture.")
        with self.lock:
            if self.running:
                raise RuntimeError("Dataset capture is already running.")
            self.running = True
            self.source_path = raw_path
            self.session_dir = None
            self.last_error = None
            self.last_message = "Starting dataset capture"

        session_name = f"dataset_{Path(raw_path).name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        record_payload = {
            "fps": TeachRecorder.HIGH_SMOOTH_CAMERA_FPS,
            "camera_mode": payload.get("camera_mode", "color"),
            "video_source": payload.get("video_source"),
            "capture_sources": payload.get("capture_sources"),
            "session_name": session_name,
            "high_smooth": True,
            "with_camera": True,
            "capture_type": "dataset_replay",
            "source_recording": raw_path,
            "task_name": payload.get("task_name"),
            "dataset_profile": payload.get("dataset_profile"),
            "action_offset": payload.get("action_offset", 1),
            "lerobot_features": payload.get("lerobot_features"),
        }
        replay_payload = {
            "path": raw_path,
            "speed": payload.get("speed", 0.75),
            "armed": payload.get("armed", False),
        }

        try:
            self.replay.arm._require_motion(replay_payload)
            start_point = self._source_start_point(raw_path)
            self.replay.stop_event.clear()
            self.replay.arm.prepare_replay()
            with self.lock:
                self.last_message = f"Moving to source start: {Path(raw_path).name}"
            self.replay._move_to_start(start_point, 1.0 / self.replay.COMMAND_HZ)
            if self.replay.stop_event.is_set():
                raise RuntimeError("Dataset capture was stopped before recording.")
            record_status = self.recorder.start(record_payload)
            with self.lock:
                self.session_dir = record_status.get("session_dir")
            self.replay.start(replay_payload)
        except Exception:
            with self.lock:
                self.running = False
            try:
                self.recorder.stop()
            except Exception:
                pass
            raise

        self.thread = threading.Thread(target=self._watch_replay, daemon=True)
        self.thread.start()
        with self.lock:
            self.last_message = f"Dataset capture started from {Path(raw_path).name}"
            return self.status_unlocked()

    def _source_start_point(self, raw_path: str) -> dict[str, Any]:
        frames = self.replay.library.replay_frames(raw_path)
        for frame in frames:
            qpos = frame.get("qpos", [])
            if len(qpos) != 6:
                continue
            return {
                "qpos": self.replay.arm._validated_positions(qpos),
                "gripper_position": self.replay.arm._validated_gripper_position(
                    frame.get("gripper_position")
                ),
            }
        raise RuntimeError("This source movement has no replayable first frame.")

    def stop(self) -> dict[str, Any]:
        self.replay.stop()
        record_status = self.recorder.stop()
        with self.lock:
            self.running = False
            self.session_dir = record_status.get("session_dir") or self.session_dir
            self.last_message = "Dataset capture stopped"
            return self.status_unlocked()

    def force_stop(self) -> None:
        self.replay.force_stop()
        try:
            record_status = self.recorder.stop()
            session_dir = record_status.get("session_dir")
        except Exception:
            session_dir = None
        with self.lock:
            self.running = False
            if session_dir:
                self.session_dir = session_dir
            self.last_message = "Dataset capture stopped by emergency stop"

    def _watch_replay(self) -> None:
        try:
            while True:
                if not self.status()["running"]:
                    return
                replay_status = self.replay.status()
                if not replay_status.get("running"):
                    break
                time.sleep(0.1)
            record_status = self.recorder.stop()
            with self.lock:
                self.running = False
                self.session_dir = record_status.get("session_dir") or self.session_dir
                self.last_message = f"Dataset capture complete: {self.session_dir}"
        except Exception as exc:  # noqa: BLE001 - surfaced in UI status.
            with self.lock:
                self.running = False
                self.last_error = f"Dataset capture failed: {exc}"


class TrossenDataCollectionUIRunner:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.executable = project_root / ".venv-trossen-ui" / "bin" / "trossen_ai_data_collection_ui"
        self.lock = threading.Lock()
        self.process: subprocess.Popen[bytes] | None = None
        self.last_message = "Trossen AI Data Collection UI ready"

    def _is_running_unlocked(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def status(self) -> dict[str, Any]:
        with self.lock:
            if self.process is not None and self.process.poll() is not None:
                self.process = None
            return self.status_unlocked(self.last_message)

    def status_unlocked(self, message: str) -> dict[str, Any]:
        running = self._is_running_unlocked()
        return {
            "ok": True,
            "available": self.executable.exists(),
            "running": running,
            "pid": self.process.pid if running and self.process is not None else None,
            "command": str(self.executable),
            "message": message,
        }

    def start(self) -> dict[str, Any]:
        with self.lock:
            if not self.executable.exists():
                raise RuntimeError(f"Trossen UI executable not found: {self.executable}")
            if self._is_running_unlocked():
                return self.status_unlocked("Trossen AI Data Collection UI is already running.")
            self.process = subprocess.Popen(
                [str(self.executable)],
                cwd=str(self.project_root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self.last_message = f"Trossen AI Data Collection UI started (PID {self.process.pid})."
            return self.status_unlocked(self.last_message)

    def stop(self) -> dict[str, Any]:
        with self.lock:
            if not self._is_running_unlocked():
                self.process = None
                return self.status_unlocked("Trossen AI Data Collection UI is not running.")
            assert self.process is not None
            pid = self.process.pid
            self.process.terminate()
            self.last_message = f"Trossen AI Data Collection UI stop requested (PID {pid})."
            return self.status_unlocked(self.last_message)


class ModelTestRunner:
    def __init__(self, project_root: Path, controller: ArmController) -> None:
        self.project_root = project_root
        self.controller = controller
        self.python = project_root / ".venv-trossen-ui" / "bin" / "python"
        self.lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None

    @staticmethod
    def _bounded_float(payload: dict[str, Any], name: str, default: float, low: float, high: float) -> float:
        value = float(payload.get(name, default))
        if not low <= value <= high:
            raise RuntimeError(f"{name} must be between {low} and {high}.")
        return value

    @staticmethod
    def _float_value(payload: dict[str, Any], name: str, default: float, low: float | None = None) -> float:
        value = float(payload.get(name, default))
        if not math.isfinite(value):
            raise RuntimeError(f"{name} must be a finite number.")
        if low is not None and value < low:
            raise RuntimeError(f"{name} must be >= {low}.")
        return value

    @staticmethod
    def _bounded_int(payload: dict[str, Any], name: str, default: int, low: int, high: int) -> int:
        value = int(payload.get(name, default))
        if not low <= value <= high:
            raise RuntimeError(f"{name} must be between {low} and {high}.")
        return value

    def checkpoints(self) -> dict[str, Any]:
        models_root = self.project_root / "widowx_ai" / "models"
        candidates: dict[Path, dict[str, Any]] = {}
        if models_root.exists():
            for path in models_root.rglob("*"):
                checkpoint = self._checkpoint_candidate(path)
                if checkpoint is None:
                    continue
                stat_path = checkpoint if checkpoint.is_file() else checkpoint
                try:
                    modified = stat_path.stat().st_mtime
                except OSError:
                    modified = 0.0
                existing = candidates.get(checkpoint)
                if existing is None or modified > existing["modified_ts"]:
                    candidates[checkpoint] = {
                        "path": str(checkpoint),
                        "label": self._checkpoint_label(models_root, checkpoint),
                        "kind": self._checkpoint_kind(checkpoint),
                        "modified_ts": modified,
                        "modified_at": datetime.fromtimestamp(modified).isoformat(timespec="seconds") if modified else None,
                    }
        items = sorted(candidates.values(), key=lambda item: item["modified_ts"], reverse=True)
        for item in items:
            item.pop("modified_ts", None)
        return {"ok": True, "models_root": str(models_root), "checkpoints": items[:30]}

    @staticmethod
    def _checkpoint_candidate(path: Path) -> Path | None:
        if path.is_dir():
            if path.name == "pretrained_model" and (path.parent / "pretrained_model" / "config.json").exists():
                return None
            if (path / "pretrained_model" / "config.json").exists():
                return path
            if (path / "config.json").exists():
                return path
            if (path / "best.pt").exists():
                return path / "best.pt"
            return None
        if path.name in {"best.pt", "last.pt"}:
            return path
        return None

    @staticmethod
    def _checkpoint_kind(path: Path) -> str:
        if path.is_dir() and (path / "pretrained_model" / "config.json").exists():
            return "LeRobot policy"
        if path.is_dir() and (path / "config.json").exists():
            return "LeRobot checkpoint"
        if path.is_file():
            return path.suffix.lstrip(".") or "file"
        return "checkpoint"

    @staticmethod
    def _checkpoint_label(models_root: Path, checkpoint: Path) -> str:
        try:
            rel = checkpoint.relative_to(models_root)
        except ValueError:
            rel = checkpoint
        return str(rel)

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.python.exists():
            raise RuntimeError(f"Python environment not found: {self.python}")
        real = bool(payload.get("real"))
        armed = bool(payload.get("armed"))
        if real and not armed:
            raise RuntimeError("Real model test blocked: check 'enable real model motion' first.")
        if real and not self.controller.args.real:
            raise RuntimeError("Real model test blocked: restart the control interface with --real.")
        if real and self.controller.connected:
            raise RuntimeError("Real model test blocked: disconnect this interface from the arm first.")
        if real:
            with suppress(Exception):
                RequestHandler.camera_controller.video_stop()
            with suppress(Exception):
                RequestHandler.camera_controller.usb_stop()
            with suppress(Exception):
                RequestHandler.camera_controller.stop()

        checkpoint = str(payload.get("checkpoint") or "widowx_ai/models/act_20260428_084937/best.pt")
        primary_camera_source = str(payload.get("primary_camera_source") or "").strip()
        cameras = RequestHandler.camera_controller.usb_status().get("cameras", [])
        valid_usb_sources = {f"usb:{camera['index']}" for camera in cameras}
        if primary_camera_source.startswith("usb:") and primary_camera_source not in valid_usb_sources:
            primary_camera_source = next(iter(sorted(valid_usb_sources)), "")
        steps = self._bounded_int(payload, "steps", 1, 1, 800)
        period = self._float_value(payload, "period", 1.0, 0.0)
        command_move_time = self._float_value(payload, "command_move_time", 0.5, 0.0)
        movement_speed = self._bounded_float(payload, "movement_speed", 100.0, 50.0, 100.0)
        wait_after_command = self._float_value(payload, "wait_after_command", 0.5, 0.0)
        speed_scale = movement_speed / 100.0
        effective_step_time = wait_after_command
        if speed_scale < 0.999 and effective_step_time <= 0:
            base_step_time = command_move_time if command_move_time > 0 else period
            effective_step_time = max(0.0, (base_step_time / speed_scale) - base_step_time)
        max_speed = self._float_value(payload, "max_speed", 0.05, 1e-6)
        max_step_rad = self._bounded_float(payload, "max_step_rad", 0.20, 0.005, 1.00)
        envelope_margin = self._bounded_float(payload, "envelope_margin", 3.14, 0.0, 3.14)
        collision_action = str(payload.get("collision_action") or "gravity")
        if collision_action not in {"idle", "gravity"}:
            raise RuntimeError("collision_action must be 'idle' or 'gravity'.")
        software_safety = bool(payload.get("software_safety", False))
        stall_error_rad = self._bounded_float(payload, "stall_error_rad", 3.14, 0.015, 3.14)
        stall_velocity_rad_s = self._bounded_float(payload, "stall_velocity_rad_s", 0.0, 0.0, 1.00)
        stall_seconds = self._bounded_float(payload, "stall_seconds", 60.0, 0.1, 60.0)

        command = [
            str(self.python),
            "-m",
            "widowx_ai.policies.run_act_safe",
            "--checkpoint",
            checkpoint,
            "--steps",
            str(steps),
            "--period",
            str(period),
            "--command-move-time",
            str(command_move_time),
            "--movement-speed-scale",
            str(speed_scale),
            "--wait-after-command",
            str(wait_after_command),
            "--max-speed",
            str(max_speed),
            "--max-step-rad",
            str(max_step_rad),
            "--envelope-margin",
            str(envelope_margin),
            "--collision-action",
            collision_action,
            "--stall-error-rad",
            str(stall_error_rad),
            "--stall-velocity-rad-s",
            str(stall_velocity_rad_s),
            "--stall-seconds",
            str(stall_seconds),
            "--ip",
            self.controller.args.ip,
            "--arm-port",
            str(self.controller.args.arm_port),
            "--variant",
            self.controller.args.variant,
            "--timeout",
            str(self.controller.args.timeout),
            "--max-runtime",
            str(max(10.0, steps * max(effective_step_time, 0.05) + 10.0)),
        ]
        if not software_safety:
            command.append("--disable-software-safety")
        if primary_camera_source:
            command.extend(["--primary-camera-source", primary_camera_source])
        
        if real:
            command.extend(["--real", "--armed"])

        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError("A model test is already running.")
            self.process = subprocess.Popen(
                command,
                cwd=str(self.project_root),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            process = self.process
        try:
            stdout, stderr = process.communicate(timeout=max(30.0, steps * max(effective_step_time, period, 0.05) + 30.0))
        except subprocess.TimeoutExpired:
            self.force_stop()
            stdout, stderr = process.communicate(timeout=5.0)
            stderr = f"{stderr}\nSAFETY WARNING: model test timed out and was interrupted.".strip()
        finally:
            with self.lock:
                if self.process is process:
                    self.process = None

        output = stdout
        if stderr:
            output = f"{output}\n{stderr}".strip()
        if process.returncode != 0:
            if any(
                marker in output
                for marker in (
                    "Force/stall guard triggered",
                    "Target outside training envelope",
                    "Max runtime reached",
                    "Target outside robot limit",
                )
            ):
                output = f"SAFETY WARNING: model run stopped by a guard.\n\n{output}"
            raise RuntimeError(output or f"Model test failed with code {process.returncode}.")
        return {
            "ok": True,
            "real": real,
            "command": " ".join(command),
            "output": output,
            "message": "Model test complete.",
        }

    def force_stop(self) -> str | None:
        with self.lock:
            process = self.process
            if process is None or process.poll() is not None:
                self.process = None
                return None
            pid = process.pid
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return None
        return f"Model test emergency stop requested (PID {pid})."


class ActReviewRunner:
    def __init__(self, project_root: Path, planner: ActDatasetPlanner) -> None:
        self.project_root = project_root
        self.planner = planner
        self.python = project_root / "Lerobot" / ".venv-lerobot" / "bin" / "python"
        self.script = project_root / "scripts" / "lerobot_act_test_interface.py"
        self.default_checkpoint = project_root / "widowx_ai" / "models" / "checkpoint_last"
        self.lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None
        self.command: list[str] = []
        self.last_message = "ACT review ready."
        self.url = "http://127.0.0.1:7866"

    def _is_running_unlocked(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def status(self) -> dict[str, Any]:
        with self.lock:
            if self.process is not None and self.process.poll() is not None:
                self.process = None
            return self._status_unlocked(self.last_message)

    def _status_unlocked(self, message: str) -> dict[str, Any]:
        running = self._is_running_unlocked()
        return {
            "ok": True,
            "running": running,
            "pid": self.process.pid if running and self.process is not None else None,
            "url": self.url,
            "command": ActDatasetPlanner._quote_args(self.command) if self.command else "",
            "message": message,
        }

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.python.exists():
            raise RuntimeError(f"LeRobot Python environment not found: {self.python}")
        if not self.script.exists():
            raise RuntimeError(f"ACT review script not found: {self.script}")
        checkpoint = Path(str(payload.get("checkpoint") or self.default_checkpoint)).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = self.project_root / checkpoint
        checkpoint = checkpoint.resolve()
        if not checkpoint.exists():
            raise RuntimeError(f"ACT checkpoint not found: {checkpoint}")
        plan = self.planner.plan(payload)
        dataset_root = Path(str(plan["dataset"]["output_root"])).resolve()
        if not dataset_root.exists():
            fallback = Path("/tmp/widowx_push_tape_front_lerobot")
            if fallback.exists():
                dataset_root = fallback
            else:
                raise RuntimeError(
                    f"LeRobot dataset not found: {dataset_root}. Export LeRobotDataset first."
                )
        repo_id = str(payload.get("repo_id") or "local/widowx-push-tape-front")
        command = [
            str(self.python),
            str(self.script),
            "--dataset-root",
            str(dataset_root),
            "--repo-id",
            repo_id,
            "--checkpoint",
            str(checkpoint),
            "--host",
            "127.0.0.1",
            "--port",
            "7866",
            "--device",
            "cpu",
        ]
        env = os.environ.copy()
        env.setdefault("HF_HOME", "/tmp/lerobot_hf_cache")
        env.setdefault("HF_DATASETS_CACHE", "/tmp/lerobot_hf_cache/datasets")
        env["PYTHONUNBUFFERED"] = "1"
        with self.lock:
            if self._is_running_unlocked():
                return self._status_unlocked("ACT review is already running.")
            self.command = command
            self.process = subprocess.Popen(
                command,
                cwd=str(self.project_root),
                env=env,
                text=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self.last_message = f"ACT review started at {self.url} (PID {self.process.pid})."
            return self._status_unlocked(self.last_message)


class RequestHandler(BaseHTTPRequestHandler):
    controller: ArmController
    camera_controller: CameraController
    teach_recorder: TeachRecorder
    recording_library: RecordingLibrary
    replay_runner: ReplayRunner
    dataset_capture_runner: DatasetCaptureRunner
    trossen_ui_runner: TrossenDataCollectionUIRunner
    model_test_runner: ModelTestRunner
    act_dataset_planner: ActDatasetPlanner
    lerobot_export_runner: LeRobotExportRunner
    act_review_runner: ActReviewRunner
    hamster_service: HamsterService
    imitation_runner: ImitationTrajectoryRunner

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    @staticmethod
    def crop_from_query(query: dict[str, list[str]]) -> dict[str, Any] | None:
        if query.get("crop_enabled", ["0"])[0] != "1":
            return None
        return {
            "enabled": True,
            "aspect": query.get("crop_aspect", ["source"])[0],
            "zoom": query.get("crop_zoom", ["1"])[0],
            "offset_x": query.get("crop_x", ["0"])[0],
            "offset_y": query.get("crop_y", ["0"])[0],
        }

    def send_mjpeg_stream(self, source: str) -> None:
        boundary = "frame"
        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
        self.end_headers()

        while True:
            frame = self.camera_controller.video_frame_jpeg(source)
            try:
                self.wfile.write(f"--{boundary}\r\n".encode("ascii"))
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break
            time.sleep(0.06)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            html = INDEX_HTML.replace("LIMITS_PLACEHOLDER", json.dumps(JOINT_LIMITS))
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
            return
        if parsed.path == "/teach":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(TEACH_HTML.encode("utf-8"))
            return
        if parsed.path == "/model-test":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(MODEL_TEST_HTML.encode("utf-8"))
            return
        if parsed.path == "/hamster":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HAMSTER_HTML.encode("utf-8"))
            return
        if parsed.path == "/imitation":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(IMITATION_HTML.encode("utf-8"))
            return
        if parsed.path == "/static/imitation_review.js":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/javascript; charset=utf-8")
            self.end_headers()
            self.wfile.write(IMITATION_REVIEW_JS.encode("utf-8"))
            return
        if parsed.path == "/static/imitation_trajectory.js":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/javascript; charset=utf-8")
            self.end_headers()
            self.wfile.write(IMITATION_TRAJECTORY_JS.encode("utf-8"))
            return
        if parsed.path == "/dataset-trim":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DATASET_TRIM_HTML.encode("utf-8"))
            return
        if parsed.path == "/lerobot-export":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DATASET_TRIM_HTML.encode("utf-8"))
            return
        if parsed.path == "/api/status":
            self.send_json(self.controller.status())
            return
        if parsed.path == "/api/camera/status":
            self.send_json(self.camera_controller.status())
            return
        if parsed.path == "/api/video/status":
            self.send_json(self.camera_controller.video_status())
            return
        if parsed.path == "/api/usb_cameras/status":
            self.send_json(self.camera_controller.usb_status())
            return
        if parsed.path == "/api/teach/status":
            self.send_json(self.teach_recorder.status())
            return
        if parsed.path == "/api/replay/status":
            self.send_json(self.replay_runner.status())
            return
        if parsed.path == "/api/dataset_capture/status":
            self.send_json(self.dataset_capture_runner.status())
            return
        if parsed.path == "/api/trossen_ui/status":
            self.send_json(self.trossen_ui_runner.status())
            return
        if parsed.path == "/api/model_test/checkpoints":
            self.send_json(self.model_test_runner.checkpoints())
            return
        if parsed.path == "/api/recordings":
            query = parse_qs(parsed.query)
            root = query.get("root", [""])[0]
            self.send_json(self.recording_library.list(root or None))
            return
        if parsed.path == "/api/act_dataset/plan":
            self.send_json(self.act_dataset_planner.plan())
            return
        if parsed.path == "/api/lerobot_export/status":
            self.send_json(self.lerobot_export_runner.status())
            return
        if parsed.path == "/api/act_review/status":
            self.send_json(self.act_review_runner.status())
            return
        if parsed.path == "/api/hamster/status":
            self.send_json(self.hamster_service.status())
            return
        if parsed.path == "/api/imitation/status":
            self.send_json(self.imitation_runner.status())
            return
        if parsed.path == "/api/camera/frame":
            try:
                query = parse_qs(parsed.query)
                mode = query.get("mode", [self.camera_controller.mode])[0]
                frame = self.camera_controller.frame_jpeg(mode)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)
            except Exception as exc:  # noqa: BLE001 - errors are sent to the browser.
                self.send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/video/frame":
            try:
                query = parse_qs(parsed.query)
                source = query.get("source", [""])[0]
                frame = self.camera_controller.video_frame_jpeg(source, self.crop_from_query(query))
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)
            except Exception as exc:  # noqa: BLE001 - errors are sent to the browser.
                self.send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/video/stream":
            try:
                query = parse_qs(parsed.query)
                source = query.get("source", [""])[0]
                self.send_mjpeg_stream(source)
            except Exception as exc:  # noqa: BLE001 - stream setup errors are sent to the browser.
                with suppress(BrokenPipeError, ConnectionResetError):
                    self.send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/usb_cameras/frame":
            try:
                query = parse_qs(parsed.query)
                index = query.get("index", [""])[0]
                frame = self.camera_controller.usb_frame_jpeg(index)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)
            except Exception as exc:  # noqa: BLE001 - errors are sent to the browser.
                self.send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/recording/image":
            try:
                query = parse_qs(parsed.query)
                session_path = query.get("path", [""])[0]
                image_path = query.get("image", [""])[0]
                image = self.recording_library.image(session_path, image_path)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(image)))
                self.end_headers()
                self.wfile.write(image)
            except Exception as exc:  # noqa: BLE001 - errors are sent to the browser.
                self.send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            payload = self.read_json()
            routes = {
                "/api/connect": self.controller.connect,
                "/api/disconnect": self.controller.disconnect,
                "/api/max_speed": lambda: self.controller.set_max_speed(payload),
                "/api/move": lambda: self.controller.move(payload),
                "/api/home": lambda: self.controller.home(payload),
                "/api/rest": lambda: self.controller.rest(payload),
                "/api/demo": lambda: self.controller.demo(payload),
                "/api/gripper": lambda: self.controller.gripper(payload),
                "/api/gravity_compensation": lambda: self.controller.gravity_compensation(payload),
                "/api/hold": lambda: self.controller.hold(payload),
                "/api/start_position/save": self.controller.save_start_position,
                "/api/start_position/go": lambda: self.controller.go_to_start_position(payload),
                "/api/camera/start": lambda: self.camera_controller.start(payload),
                "/api/camera/stop": self.camera_controller.stop,
                "/api/camera/mode": lambda: self.camera_controller.set_mode(payload),
                "/api/video/start": lambda: self.camera_controller.video_start(payload),
                "/api/video/stop": self.camera_controller.video_stop,
                "/api/usb_cameras/start": lambda: self.camera_controller.usb_start(payload),
                "/api/usb_cameras/stop": self.camera_controller.usb_stop,
                "/api/teach/start": lambda: self.teach_recorder.start(payload),
                "/api/teach/stop": self.teach_recorder.stop,
                "/api/replay/start": lambda: self.replay_runner.start(payload),
                "/api/replay/stop": self.replay_runner.stop,
                "/api/dataset_capture/start": lambda: self.dataset_capture_runner.start(payload),
                "/api/dataset_capture/stop": self.dataset_capture_runner.stop,
                "/api/trossen_ui/start": self.trossen_ui_runner.start,
                "/api/trossen_ui/stop": self.trossen_ui_runner.stop,
                "/api/model_test/run": lambda: self.model_test_runner.run(payload),
                "/api/act_dataset/plan": lambda: self.act_dataset_planner.plan(payload),
                "/api/lerobot_export/start": lambda: self.lerobot_export_runner.start(payload),
                "/api/act_review/start": lambda: self.act_review_runner.start(payload),
                "/api/hamster/send_camera": lambda: self.hamster_service.send_camera(payload),
                "/api/hamster/start": lambda: self.hamster_service.start(payload),
                "/api/hamster/stop": lambda: self.hamster_service.stop(payload),
                "/api/hamster/status": lambda: self.hamster_service.status(payload),
                "/api/imitation/start": lambda: self.imitation_runner.start(payload),
                "/api/imitation/stop": self.imitation_runner.stop,
                "/api/recording/load": lambda: self.recording_library.load(payload),
                "/api/recording/trim": lambda: self.recording_library.trim(payload),
                "/api/recording/delete": lambda: self.recording_library.delete(payload),
                "/api/recordings/clear": self.recording_library.clear,
                "/api/emergency_stop": self.emergency_stop,
            }
            handler = routes.get(self.path)
            if handler is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_json(handler())
        except Exception as exc:  # noqa: BLE001 - errors are sent to the browser.
            self.send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def emergency_stop(self) -> dict[str, Any]:
        self.dataset_capture_runner.force_stop()
        self.replay_runner.force_stop()
        self.imitation_runner.force_stop()
        model_message = self.model_test_runner.force_stop()
        data = self.controller.emergency_stop()
        if model_message:
            data["message"] = f"{data['message']}; {model_message}"
        return data

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def send_json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local web interface for WidowX AI control.")
    parser.add_argument("--host", default="127.0.0.1", help="Web server host.")
    parser.add_argument("--port", type=int, default=7862, help="Web server port.")
    parser.add_argument("--ip", default="192.168.1.2", help="Arm controller IP address.")
    parser.add_argument("--arm-port", type=int, default=50001, help="Arm controller TCP port.")
    parser.add_argument("--variant", choices=sorted(END_EFFECTORS), default="base")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--move-time", type=float, default=1.0)
    parser.add_argument("--real", action="store_true", help="Enable real arm connection and movement.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = PROJECT_ROOT
    controller = ArmController(args, PACKAGE_ROOT / "config" / "start_position.json")
    camera_controller = CameraController()
    recordings_root = PACKAGE_ROOT / "recordings"
    teach_recorder = TeachRecorder(
        controller,
        camera_controller,
        recordings_root,
    )
    recording_library = RecordingLibrary(recordings_root)
    replay_runner = ReplayRunner(controller, recording_library)
    dataset_capture_runner = DatasetCaptureRunner(teach_recorder, replay_runner)
    trossen_ui_runner = TrossenDataCollectionUIRunner(project_root)
    model_test_runner = ModelTestRunner(project_root, controller)
    act_dataset_planner = ActDatasetPlanner(project_root, recording_library)
    lerobot_export_runner = LeRobotExportRunner(project_root, act_dataset_planner)
    act_review_runner = ActReviewRunner(project_root, act_dataset_planner)
    hamster_service = HamsterService(camera_controller)
    imitation_runner = ImitationTrajectoryRunner(controller, teach_recorder)
    RequestHandler.controller = controller
    RequestHandler.camera_controller = camera_controller
    RequestHandler.teach_recorder = teach_recorder
    RequestHandler.recording_library = recording_library
    RequestHandler.replay_runner = replay_runner
    RequestHandler.dataset_capture_runner = dataset_capture_runner
    RequestHandler.trossen_ui_runner = trossen_ui_runner
    RequestHandler.model_test_runner = model_test_runner
    RequestHandler.act_dataset_planner = act_dataset_planner
    RequestHandler.lerobot_export_runner = lerobot_export_runner
    RequestHandler.act_review_runner = act_review_runner
    RequestHandler.hamster_service = hamster_service
    RequestHandler.imitation_runner = imitation_runner
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    mode = "REAL ARM ENABLED" if args.real else "dry-run"
    print(f"WidowX AI interface running at http://{args.host}:{args.port} ({mode})")
    print("Use --real only after Ethernet is configured and the workspace is clear.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
