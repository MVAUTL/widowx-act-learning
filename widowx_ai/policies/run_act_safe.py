#!/usr/bin/env python3
"""Run a trained ACT-style policy with conservative safety gates.

Default mode is dry-run: no robot connection and no motion. Real motion requires
both --real and --armed.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import trossen_arm

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

try:
    import cv2
except ImportError:
    cv2 = None

from widowx_ai.training.train_act import STATE_DIM, TinyActPolicy


STOP_REQUESTED = False


def _request_stop(signum: int, frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    raise KeyboardInterrupt


JOINT_LIMITS = [
    (-math.pi, math.pi),
    (0.0, math.pi),
    (0.0, 2.3562),
    (-math.pi / 2, math.pi / 2),
    (-math.pi / 2, math.pi / 2),
    (-math.pi, math.pi),
]
END_EFFECTORS = {
    "base": trossen_arm.StandardEndEffector.wxai_v0_base,
    "leader": trossen_arm.StandardEndEffector.wxai_v0_leader,
    "follower": trossen_arm.StandardEndEffector.wxai_v0_follower,
}
GRAVITY_PAYLOAD = trossen_arm.StandardEndEffector.wxai_v0_follower


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _state_from_row(row: dict[str, Any]) -> np.ndarray:
    return np.asarray([*row["qpos"], row.get("gripper_position", 0.0)], dtype=np.float32)


def _preflight_tcp(ip: str, port: int, timeout: float) -> None:
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return
    except OSError as exc:
        raise RuntimeError(f"Cannot reach arm controller at {ip}:{port}.") from exc


def _configure_driver(args: argparse.Namespace) -> trossen_arm.TrossenArmDriver:
    _preflight_tcp(args.ip, args.arm_port, args.timeout)
    driver = trossen_arm.TrossenArmDriver()
    end_effector = END_EFFECTORS[args.variant]
    try:
        driver.configure(trossen_arm.Model.wxai_v0, end_effector, args.ip, True, args.timeout)
    except TypeError:
        driver.configure(trossen_arm.Model.wxai_v0, end_effector, args.ip, True)
    return driver


def _load_policy(checkpoint_path: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    if checkpoint_path.is_dir():
        if (checkpoint_path / "pretrained_model").is_dir():
            checkpoint_path = checkpoint_path / "pretrained_model"
        try:
            from lerobot.common.policies.act.modeling_act import ACTPolicy
        except ImportError:
            from lerobot.policies.act.modeling_act import ACTPolicy
        model = ACTPolicy.from_pretrained(checkpoint_path, local_files_only=True)
        _load_lerobot_processor_stats(model, checkpoint_path)
        model.eval()
        return model, {}, {}

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    normalization = checkpoint["normalization"]
    model = TinyActPolicy(
        chunk_size=int(config["chunk_size"]),
        action_dim=STATE_DIM,
        hidden_dim=int(config["hidden_dim"]),
        heads=int(config["heads"]),
        layers=int(config["layers"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, normalization


def _load_lerobot_processor_stats(model: Any, checkpoint_path: Path) -> None:
    stats_files = [
        checkpoint_path / "policy_preprocessor_step_3_normalizer_processor.safetensors",
        checkpoint_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ]
    if not any(path.exists() for path in stats_files):
        return

    try:
        from safetensors.torch import load_file
    except ImportError:
        return

    state_dict = model.state_dict()
    updates = {}
    for stats_file in stats_files:
        if not stats_file.exists():
            continue
        stats = load_file(str(stats_file))
        for key, tensor in stats.items():
            if not (key.endswith(".mean") or key.endswith(".std")):
                continue
            feature, stat_name = key.rsplit(".", 1)
            buffer_name = f"buffer_{feature.replace('.', '_')}.{stat_name}"
            for prefix in ("normalize_inputs", "normalize_targets", "unnormalize_outputs"):
                model_key = f"{prefix}.{buffer_name}"
                if model_key in state_dict and state_dict[model_key].shape == tensor.shape:
                    updates[model_key] = tensor.to(dtype=state_dict[model_key].dtype)
    if updates:
        state_dict.update(updates)
        model.load_state_dict(state_dict, strict=False)


def _lerobot_visual_keys(model: Any) -> list[str]:
    try:
        features = model.config.input_features
    except AttributeError:
        return ["observation.images.wrist_rgb"]
    keys = []
    for key, feature in features.items():
        feature_type = str(getattr(feature, "type", "")).upper()
        if key.startswith("observation.images.") or feature_type.endswith("VISUAL"):
            keys.append(key)
    return keys or ["observation.images.wrist_rgb"]


def _latest_dataset_sample(root: Path, view: str) -> tuple[Path, dict[str, Any]]:
    sessions = sorted(
        [
            path
            for path in root.glob("dataset_*")
            if path.is_dir() and (path / "samples.jsonl").exists()
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for session in sessions:
        samples = _read_jsonl(session / "samples.jsonl")
        for sample in reversed(samples):
            if view in (sample.get("images") or {}):
                return session, sample
    raise RuntimeError(f"No sample image found for view '{view}'.")


def _crop_center_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))

def _load_image_tensor(image_path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = _crop_center_square(image)
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).unsqueeze(0)


class RealSenseColor:
    def __init__(self, *, serial: str | None, width: int, height: int, fps: int) -> None:
        if rs is None:
            raise RuntimeError("pyrealsense2 is not installed in this Python environment.")
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.started = False
        if serial:
            self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

    def __enter__(self) -> "RealSenseColor":
        self.pipeline.start(self.config)
        self.started = True
        for _ in range(15):
            self.pipeline.wait_for_frames()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.started:
            self.pipeline.stop()
            self.started = False

    def image_tensor(self, image_size: int) -> torch.Tensor:
        frames = self.pipeline.wait_for_frames()
        frame = frames.get_color_frame()
        if not frame:
            raise RuntimeError("No RealSense color frame received.")
        image = Image.fromarray(np.asanyarray(frame.get_data())).convert("RGB")
        image = _crop_center_square(image)
        image = image.resize(image_size if isinstance(image_size, tuple) else (image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array).unsqueeze(0)


class USBCamera:
    def __init__(self, source: str, width: int, height: int) -> None:
        if cv2 is None:
            raise RuntimeError("cv2 is not installed in this Python environment.")
        parsed_source: str | int = source
        if source.startswith("usb:"):
            parsed_source = int(source.split(":", 1)[1])
        if isinstance(parsed_source, int) and hasattr(cv2, "CAP_V4L2"):
            self.cap = cv2.VideoCapture(parsed_source, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(parsed_source)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Cannot open USB camera source {source}.")
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.started = False

    def __enter__(self) -> "USBCamera":
        self.started = True
        for _ in range(15):
            self.cap.read()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.started:
            self.cap.release()
            self.started = False

    def image_tensor(self, image_size: tuple[int, int] | int) -> torch.Tensor:
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("No USB camera frame received.")
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = _crop_center_square(image)
        image = image.resize(image_size if isinstance(image_size, tuple) else (image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array).unsqueeze(0)


def _training_envelope(config: dict[str, Any], checkpoint_path: Path, margin: float) -> tuple[np.ndarray, np.ndarray]:
    if "root" not in config:
        low = np.asarray([float(limit[0]) for limit in JOINT_LIMITS] + [0.0], dtype=np.float32)
        high = np.asarray([float(limit[1]) for limit in JOINT_LIMITS] + [0.15], dtype=np.float32)
        return low, high
    root = Path(config["root"])
    if not root.is_absolute():
        root = checkpoint_path.parent / root
    states: list[np.ndarray] = []
    for name in [*config.get("train_sessions", []), *config.get("val_sessions", [])]:
        motor_path = root / name / "motor_samples.jsonl"
        if not motor_path.exists():
            continue
        states.extend(_state_from_row(row) for row in _read_jsonl(motor_path))
    if not states:
        raise RuntimeError("Cannot build safety envelope from training motor samples.")
    array = np.stack(states)
    low = array.min(axis=0)
    high = array.max(axis=0)
    low[:6] -= margin
    high[:6] += margin
    low[6] -= min(margin, 0.002)
    high[6] += min(margin, 0.002)
    return low, high


def _validate_target(
    *,
    current: np.ndarray,
    target: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    max_step_rad: float,
    max_gripper_step: float,
    disable_software_safety: bool,
) -> np.ndarray:
    if target.shape != (STATE_DIM,) or not np.all(np.isfinite(target)):
        raise RuntimeError("Policy produced an invalid target.")

    if disable_software_safety:
        clipped = target.copy()
        joint_low = np.asarray([limit[0] for limit in JOINT_LIMITS], dtype=np.float32)
        joint_high = np.asarray([limit[1] for limit in JOINT_LIMITS], dtype=np.float32)
        clipped[:6] = np.minimum(np.maximum(clipped[:6], joint_low), joint_high)
        clipped[6] = float(np.clip(clipped[6], 0.0, 0.15))
        return clipped

    clipped = target.copy()
    arm_delta = np.clip(clipped[:6] - current[:6], -max_step_rad, max_step_rad)
    clipped[:6] = current[:6] + arm_delta
    if current[6] and math.isfinite(float(current[6])):
        gripper_delta = float(np.clip(clipped[6] - current[6], -max_gripper_step, max_gripper_step))
        clipped[6] = current[6] + gripper_delta
    clipped = np.minimum(np.maximum(clipped, low), high)

    for idx, (value, (joint_low, joint_high)) in enumerate(zip(clipped[:6], JOINT_LIMITS)):
        if not joint_low <= float(value) <= joint_high:
            raise RuntimeError(
                f"Joint {idx} target {value:.3f} outside robot limit [{joint_low:.3f}, {joint_high:.3f}]."
            )

    tolerance = max(max_step_rad, max_gripper_step)
    far_outside = np.where((clipped < low - tolerance) | (clipped > high + tolerance))[0]
    if far_outside.size:
        details = ", ".join(
            f"{idx}: {clipped[idx]:.4f} not in [{low[idx]:.4f}, {high[idx]:.4f}]"
            for idx in far_outside.tolist()
        )
        raise RuntimeError(f"Target outside training envelope: {details}")
    return clipped


def _predict(
    *,
    model: Any,
    image: torch.Tensor,
    top_image: torch.Tensor | None,
    state: np.ndarray,
    normalization: dict[str, Any],
    image_key: str = "observation.images.wrist_rgb",
    top_image_key: str | None = "observation.images.top_view",
) -> np.ndarray:
    if hasattr(model, "select_action"):
        device = next(model.parameters()).device
        batch = {
            image_key: image.to(device),
            "observation.state": torch.from_numpy(state).unsqueeze(0).to(device)
        }
        if top_image is not None and top_image_key is not None:
            batch[top_image_key] = top_image.to(device)
            
        with torch.inference_mode():
            pred = model.select_action(batch)[0]
        return pred.detach().cpu().numpy()

    state_mean = np.asarray(normalization["state_mean"], dtype=np.float32)
    state_std = np.asarray(normalization["state_std"], dtype=np.float32)
    action_mean = np.asarray(normalization["action_mean"], dtype=np.float32)
    action_std = np.asarray(normalization["action_std"], dtype=np.float32)
    norm_state = torch.from_numpy(((state - state_mean) / state_std).astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        pred = model(image, norm_state)[0, 0].cpu().numpy()
    return pred * action_std + action_mean


def _read_robot_state(driver: trossen_arm.TrossenArmDriver) -> np.ndarray:
    qpos = np.asarray(driver.get_arm_positions(), dtype=np.float32)[:6]
    gripper = float(driver.get_gripper_position())
    return np.asarray([*qpos, gripper], dtype=np.float32)


def _set_safety_mode(driver: trossen_arm.TrossenArmDriver, action: str) -> None:
    if action == "gravity":
        driver.set_end_effector(GRAVITY_PAYLOAD)
        driver.set_all_modes(trossen_arm.Mode.external_effort)
        driver.set_all_external_efforts([0.0] * 7, 0.0, False)
        return
    driver.set_all_modes(trossen_arm.Mode.idle)


def _command_robot(
    driver: trossen_arm.TrossenArmDriver,
    target: np.ndarray,
    move_time: float,
    *,
    wait_after_command: float | None,
    disable_stall_guard: bool,
    collision_action: str,
    stall_error_rad: float,
    stall_velocity_rad_s: float,
    stall_seconds: float,
) -> None:
    driver.set_arm_modes(trossen_arm.Mode.position)
    driver.set_gripper_mode(trossen_arm.Mode.position)
    driver.set_arm_positions(target[:6].astype(float), move_time, False)
    driver.set_gripper_position(float(target[6]), move_time, False)

    wait_time = move_time if wait_after_command is None else max(0.0, wait_after_command)
    if wait_time <= 0:
        return
    if disable_stall_guard:
        time.sleep(wait_time)
        return

    deadline = time.monotonic() + wait_time + 0.25
    last_qpos = np.asarray(driver.get_arm_positions(), dtype=np.float32)[:6]
    last_time = time.monotonic()
    stalled_since: float | None = None
    while time.monotonic() < deadline:
        time.sleep(0.05)
        now = time.monotonic()
        qpos = np.asarray(driver.get_arm_positions(), dtype=np.float32)[:6]
        dt = max(now - last_time, 1e-6)
        velocity = float(np.max(np.abs(qpos - last_qpos)) / dt)
        error = float(np.max(np.abs(target[:6] - qpos)))

        if error > stall_error_rad and velocity < stall_velocity_rad_s:
            if stalled_since is None:
                stalled_since = now
            elif now - stalled_since >= stall_seconds:
                _set_safety_mode(driver, collision_action)
                raise RuntimeError(
                    "Force/stall guard triggered: arm is not moving toward target. "
                    f"max_error={error:.4f} rad, max_velocity={velocity:.4f} rad/s. "
                    f"Safety action={collision_action}."
                )
        else:
            stalled_since = None

        if error <= max(0.01, stall_error_rad * 0.5):
            return
        last_qpos = qpos
        last_time = now


def _print_train_info(checkpoint_path: Path, config: dict[str, Any]) -> None:
    run_dir = checkpoint_path.parent
    history_path = run_dir / "history.json"
    if not history_path.exists():
        print(f"Loaded LeRobot policy from: {checkpoint_path}")
        return
        
    history = _read_json(history_path)
    best = min(history, key=lambda row: row["val_l1"] if row.get("val_l1") is not None else row["train_l1"])
    print("Training run")
    print(f"  run_dir: {run_dir}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  view: {config.get('view')}")
    print(f"  examples: {config.get('train_examples')} train / {config.get('val_examples')} val")
    print(f"  epochs: {config.get('epochs')}")
    print(f"  best epoch: {best.get('epoch')} val_l1={best.get('val_l1'):.5f}")
    print(f"  last epoch: {history[-1].get('epoch')} val_l1={history[-1].get('val_l1'):.5f}")


def main() -> int:
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    parser = argparse.ArgumentParser(description="Safely test an ACT policy on WidowX AI.")
    parser.add_argument("--checkpoint", default="widowx_ai/models/act_20260428_084937/best.pt")
    parser.add_argument("--ip", default="192.168.1.2")
    parser.add_argument("--arm-port", type=int, default=50001)
    parser.add_argument("--variant", choices=sorted(END_EFFECTORS), default="base")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--real", action="store_true", help="Connect to the real arm.")
    parser.add_argument("--armed", action="store_true", help="Required together with --real to command motion.")
    parser.add_argument("--steps", type=int, default=1, help="Number of policy control steps.")
    parser.add_argument("--period", type=float, default=1.0, help="Seconds per command step.")
    parser.add_argument("--command-move-time", type=float, default=None, help="Override motor command duration in seconds.")
    parser.add_argument("--wait-after-command", type=float, default=None, help="Seconds to wait/monitor after sending each command.")
    parser.add_argument("--movement-speed-scale", type=float, default=1.0, help="Motor speed scale between 0.5 and 1.0.")
    parser.add_argument("--disable-software-safety", action="store_true", help="Disable model-test software guards except arming and physical joint clipping.")
    parser.add_argument("--max-runtime", type=float, default=10.0)
    parser.add_argument("--max-step-rad", type=float, default=0.035)
    parser.add_argument("--max-speed", type=float, default=0.05)
    parser.add_argument("--max-gripper-step", type=float, default=0.001)
    parser.add_argument("--envelope-margin", type=float, default=0.08)
    parser.add_argument("--collision-action", choices=["idle", "gravity"], default="gravity")
    parser.add_argument("--stall-error-rad", type=float, default=0.045)
    parser.add_argument("--stall-velocity-rad-s", type=float, default=0.008)
    parser.add_argument("--stall-seconds", type=float, default=0.35)
    parser.add_argument("--realsense-serial", default=None)
    parser.add_argument("--primary-camera-source", default=None)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--top-camera-source", default=None)
    parser.add_argument("--top-camera-resolution", default="640x480")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    model, config, normalization = _load_policy(checkpoint_path)
    _print_train_info(checkpoint_path, config)
    low, high = _training_envelope(config, checkpoint_path, args.envelope_margin)
    view = str(config.get("view", "wrist_rgb"))
    
    is_lerobot = hasattr(model, "select_action")
    if is_lerobot:
        visual_keys = _lerobot_visual_keys(model)
        primary_image_key = visual_keys[0]
        top_image_key = "observation.images.top_view" if "observation.images.top_view" in visual_keys else None
        try:
            shape = model.config.input_features[primary_image_key].shape
            image_size = (shape[2], shape[1])
        except (AttributeError, KeyError):
            image_size = (480, 480)
    else:
        visual_keys = []
        primary_image_key = "observation.images.wrist_rgb"
        top_image_key = None
        image_size = int(config.get("image_size", 96))

    print("Safety gates")
    print(f"  real motion: {args.real and args.armed}")
    print(f"  steps: {args.steps}")
    print(f"  max_step_rad: {args.max_step_rad}")
    print(f"  max_speed: {args.max_speed} rad/s")
    print(f"  envelope_margin: {args.envelope_margin} rad")
    print(f"  software safety: {'disabled' if args.disable_software_safety else 'enabled'}")
    print(f"  force/stall guard: {'disabled' if args.disable_software_safety else f'{args.collision_action}, error>{args.stall_error_rad} rad, velocity<{args.stall_velocity_rad_s} rad/s'}")

    if args.real and not args.armed:
        raise RuntimeError("Real arm requested, but --armed is missing. No motion will be sent.")
    speed_scale = min(max(float(args.movement_speed_scale), 0.5), 1.0)

    driver: trossen_arm.TrossenArmDriver | None = None
    camera: RealSenseColor | USBCamera | None = None
    top_camera: USBCamera | None = None
    try:
        if args.real:
            driver = _configure_driver(args)
            if args.primary_camera_source and args.primary_camera_source.startswith("usb:"):
                camera = USBCamera(args.primary_camera_source, args.camera_width, args.camera_height)
            else:
                camera = RealSenseColor(
                    serial=args.realsense_serial,
                    width=args.camera_width,
                    height=args.camera_height,
                    fps=args.camera_fps,
                )
            camera_ctx = camera.__enter__()
            top_camera_ctx = None
            if is_lerobot and top_image_key is not None and args.top_camera_source:
                w, h = map(int, args.top_camera_resolution.split("x"))
                top_camera = USBCamera(args.top_camera_source, w, h)
                top_camera_ctx = top_camera.__enter__()
            current = _read_robot_state(driver)
        else:
            if is_lerobot:
                current = np.asarray([0.0]*7, dtype=np.float32)
                camera_ctx = None
                top_camera_ctx = None
                image_path = Path("dummy")
                print("Dry-run: LeRobot checkpoint detected, using dummy inputs.")
            else:
                root = Path(config["root"])
                session, sample = _latest_dataset_sample(root, view)
                image_path = session / str((sample.get("images") or {})[view])
                current = _state_from_row(sample)
                camera_ctx = None
                top_camera_ctx = None
                print(f"Dry-run sample: {image_path}")

        deadline = time.monotonic() + args.max_runtime
        for step in range(1, args.steps + 1):
            if STOP_REQUESTED:
                raise KeyboardInterrupt
            if time.monotonic() > deadline:
                raise RuntimeError("Max runtime reached.")
            if args.real:
                assert driver is not None
                assert camera_ctx is not None
                current = _read_robot_state(driver)
                image = camera_ctx.image_tensor(image_size)
                top_image = top_camera_ctx.image_tensor(image_size) if top_camera_ctx else None
            else:
                if is_lerobot:
                    c, h, w = (3, image_size[1], image_size[0]) if isinstance(image_size, tuple) else (3, image_size, image_size)
                    image = torch.zeros((1, c, h, w), dtype=torch.float32)
                    top_image = torch.zeros((1, c, h, w), dtype=torch.float32)
                else:
                    image = _load_image_tensor(image_path, image_size)
                    top_image = None
            raw_target = _predict(
                model=model,
                image=image,
                top_image=top_image,
                state=current,
                normalization=normalization,
                image_key=primary_image_key,
                top_image_key=top_image_key,
            )
            target = _validate_target(
                current=current,
                target=raw_target,
                low=low,
                high=high,
                max_step_rad=args.max_step_rad,
                max_gripper_step=args.max_gripper_step,
                disable_software_safety=args.disable_software_safety,
            )
            delta = float(np.max(np.abs(target[:6] - current[:6])))
            calculated_move_time = max(args.period, delta / args.max_speed if delta > 0 else args.period)
            base_move_time = args.command_move_time if args.command_move_time is not None else calculated_move_time
            move_time = base_move_time
            if move_time > 0:
                move_time = base_move_time / speed_scale
            effective_wait_after_command = args.wait_after_command
            if speed_scale < 0.999 and (effective_wait_after_command is None or effective_wait_after_command <= 0):
                effective_wait_after_command = max(0.0, move_time - base_move_time)
            print(
                f"step {step}: current={np.array2string(current, precision=4)} "
                f"target={np.array2string(target, precision=4)} move_time={move_time:.2f}s"
            )
            if args.real and args.armed:
                assert driver is not None
                _command_robot(
                    driver,
                    target,
                    move_time,
                    wait_after_command=effective_wait_after_command,
                    disable_stall_guard=args.disable_software_safety,
                    collision_action=args.collision_action,
                    stall_error_rad=args.stall_error_rad,
                    stall_velocity_rad_s=args.stall_velocity_rad_s,
                    stall_seconds=args.stall_seconds,
                )
            current = target

    finally:
        if top_camera is not None:
            top_camera.__exit__(None, None, None)
        if camera is not None:
            camera.__exit__(None, None, None)
        if driver is not None:
            _set_safety_mode(driver, args.collision_action)
            print(f"Arm set to safety mode: {args.collision_action}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Stopping.", file=sys.stderr)
        raise SystemExit(130)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
