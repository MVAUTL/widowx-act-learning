"""LeRobot dataset planning and export helpers for the WidowX web app."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import os
import shlex
import subprocess
import threading
from typing import Any


class ActDatasetPlanner:
    def __init__(self, project_root: Path, recordings: Any) -> None:
        self.project_root = project_root
        self.recordings = recordings

    @staticmethod
    def _safe_name(raw_name: Any, default: str) -> str:
        name = str(raw_name or default).strip()
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        cleaned = "".join(ch if ch in allowed else "_" for ch in name)
        return cleaned or default

    @staticmethod
    def _positive_int(payload: dict[str, Any], name: str, default: int, low: int, high: int) -> int:
        value = int(payload.get(name) or default)
        if not low <= value <= high:
            raise RuntimeError(f"{name} must be between {low} and {high}.")
        return value

    @staticmethod
    def _camera_list(raw_cameras: Any) -> list[str]:
        cameras = [item.strip() for item in str(raw_cameras or "top_view,wrist_rgb").split(",") if item.strip()]
        allowed = {"top_view", "front", "wrist_rgb", "wrist_depth"}
        unknown = [camera for camera in cameras if camera not in allowed]
        if unknown:
            raise RuntimeError(f"Unknown ACT camera(s): {', '.join(unknown)}.")
        if not cameras:
            raise RuntimeError("Select at least one ACT camera.")
        return cameras

    @staticmethod
    def _quote_args(args: list[str]) -> str:
        return " ".join(shlex.quote(str(arg)) for arg in args)

    @staticmethod
    def _is_relative_to(path: Path, base: Path) -> bool:
        try:
            path.relative_to(base)
        except ValueError:
            return False
        return True

    def _default_output_root(self, dataset_name: str) -> Path:
        return Path("/tmp") / "lerobot_datasets" / dataset_name

    def _safe_output_root(self, raw_root: Any, dataset_name: str) -> Path:
        raw = str(raw_root or "").strip()
        root = Path(raw).expanduser() if raw else self._default_output_root(dataset_name)
        if not root.is_absolute():
            root = self.project_root / root
        resolved = root.resolve()
        project_root = self.project_root.resolve()
        tmp_root = Path("/tmp").resolve()
        forbidden = {Path("/").resolve(), tmp_root, project_root, project_root.parent.resolve()}
        if resolved in forbidden:
            raise RuntimeError(f"Refusing unsafe LeRobot output root: {resolved}")
        if not (self._is_relative_to(resolved, tmp_root) or self._is_relative_to(resolved, project_root)):
            raise RuntimeError("LeRobot output root must be under /tmp or this project folder.")
        return resolved

    def plan(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        dataset_name = self._safe_name(payload.get("dataset_name"), "widowx_push_cube_full")
        repo_id = str(payload.get("repo_id") or "matteo/widowx-push-cube-local").strip()
        if "/" not in repo_id:
            raise RuntimeError("repo_id should look like hf_user/dataset_name.")
        cameras = self._camera_list(payload.get("cameras"))
        steps = self._positive_int(payload, "steps", 10000, 200, 500000)
        batch_size = self._positive_int(payload, "batch_size", 8, 1, 256)
        max_episodes = payload.get("max_episodes")
        max_episode_count = int(max_episodes) if max_episodes not in {None, "", 0} else None
        if max_episode_count is not None and max_episode_count < 1:
            raise RuntimeError("max_episodes must be empty or greater than 0.")
        output_root = self._safe_output_root(payload.get("output_root"), dataset_name)
        use_videos = bool(payload.get("use_videos"))
        overwrite = bool(payload.get("overwrite", True))
        source_root = self.recordings._resolve_root(str(payload.get("source_root") or ""))
        raw_recording_paths = payload.get("recording_paths", [])
        if isinstance(raw_recording_paths, str):
            raw_recording_paths = [raw_recording_paths]
        selected_paths = {
            str(Path(str(item)).expanduser().resolve())
            for item in raw_recording_paths
            if str(item).strip()
        }

        recordings = self.recordings.list(str(source_root))["recordings"]
        dataset_recordings = [
            item for item in recordings if item.get("capture_type") == "dataset_replay"
        ]
        if selected_paths:
            dataset_recordings = [
                item for item in dataset_recordings if str(Path(str(item.get("path"))).resolve()) in selected_paths
            ]
        selected_recordings = dataset_recordings[:max_episode_count] if max_episode_count else dataset_recordings

        total_frames = sum(int(item.get("samples") or 0) for item in selected_recordings)
        tasks = sorted({str(item.get("task_name")) for item in selected_recordings if item.get("task_name")})
        available_cameras: set[str] = set()
        missing_camera_sessions = 0
        for item in selected_recordings:
            roles = {
                str(entry.get("role"))
                for entry in item.get("capture_sources", [])
                if isinstance(entry, dict) and entry.get("role")
            }
            available_cameras.update(roles)
            if any(camera not in roles for camera in cameras):
                missing_camera_sessions += 1

        warnings = []
        if len(selected_recordings) < 50:
            warnings.append("ACT is data efficient, but the LeRobot docs recommend starting near 50 demonstrations when possible.")
        if total_frames < 500:
            warnings.append("Very few camera frames detected; record more replay+cameras episodes before a full train.")
        if not tasks:
            warnings.append("No task labels found; set a stable task label before dataset capture.")
        if missing_camera_sessions:
            warnings.append(f"{missing_camera_sessions} selected episode(s) do not contain all requested ACT cameras.")

        dataset_root = output_root
        output_dir = self.project_root / "outputs" / "train" / f"act_{dataset_name}"
        job_name = f"act_{dataset_name}"
        lerobot_python = self.project_root / "Lerobot" / ".venv-lerobot" / "bin" / "python"

        convert_command = [
            str(lerobot_python),
            "scripts/convert_widowx_to_lerobot.py",
            "--source-root",
            str(source_root),
            "--output-root",
            str(dataset_root),
            "--repo-id",
            repo_id,
            "--robot-type",
            "widowx_ai",
            "--fps",
            "30",
            "--cameras",
            ",".join(cameras),
            "--action-offset",
            "1",
            "--use-videos" if use_videos else "--no-use-videos",
        ]
        if overwrite:
            convert_command.append("--overwrite")
        if selected_recordings:
            session_names = ",".join(
                "."
                if Path(str(item["path"])).resolve() == source_root
                else Path(str(item["path"])).name
                for item in selected_recordings
            )
            convert_command.extend(["--sessions", session_names])
        if max_episode_count:
            convert_command.extend(["--max-episodes", str(max_episode_count)])

        train_command = [
            "lerobot-train",
            f"--dataset.repo_id={repo_id}",
            f"--dataset.root={dataset_root}",
            "--policy.type=act",
            f"--output_dir={output_dir}",
            f"--job_name={job_name}",
            "--policy.device=cuda",
            "--wandb.enable=false",
            "--policy.push_to_hub=false",
            f"--steps={steps}",
            f"--batch_size={batch_size}",
            "--save_freq=1000",
        ]

        slurm_command = "sbatch slurm/convert_widowx_lerobot_full.slurm && sbatch slurm/lerobot_act_train_full.slurm"
        return {
            "ok": True,
            "dataset": {
                "episodes": len(selected_recordings),
                "available_episodes": len(dataset_recordings),
                "frames": total_frames,
                "tasks": len(tasks),
                "task_names": tasks,
                "available_cameras": sorted(available_cameras),
                "requested_cameras": cameras,
                "ready": not warnings,
                "warnings": warnings,
                "output_root": str(dataset_root),
                "source_root": str(source_root),
                "selected_recordings": [item["path"] for item in selected_recordings],
                "media_storage": "videos" if use_videos else "images",
            },
            "commands": {
                "convert": self._quote_args(convert_command),
                "train": self._quote_args(train_command),
                "slurm": slurm_command,
            },
            "notes": [
                "ACT takes RGB images and robot state as observations, then predicts future action chunks.",
                "The converter uses observation.state, observation.images.<camera>, and action features for LeRobot.",
            ],
        }


class LeRobotExportRunner:
    def __init__(self, project_root: Path, planner: ActDatasetPlanner) -> None:
        self.project_root = project_root
        self.planner = planner
        self.lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None
        self.output: deque[str] = deque(maxlen=400)
        self.command: list[str] = []
        self.output_root: str | None = None
        self.returncode: int | None = None
        self.last_message = "LeRobot export ready."

    def _is_running_unlocked(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def status(self) -> dict[str, Any]:
        with self.lock:
            if self.process is not None and self.process.poll() is not None:
                self.returncode = self.process.returncode
            return self._status_unlocked(self.last_message)

    def _status_unlocked(self, message: str) -> dict[str, Any]:
        running = self._is_running_unlocked()
        return {
            "ok": True,
            "running": running,
            "pid": self.process.pid if running and self.process is not None else None,
            "returncode": self.returncode,
            "command": ActDatasetPlanner._quote_args(self.command) if self.command else "",
            "output_root": self.output_root,
            "output": "\n".join(self.output),
            "message": message,
        }

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        plan = self.planner.plan(payload)
        if int(plan["dataset"]["episodes"]) < 1:
            raise RuntimeError("No dataset capture episodes found. Run 'Replay movement + record camera' first.")
        command = shlex.split(plan["commands"]["convert"])
        python_path = Path(command[0])
        converter_path = self.project_root / "scripts" / "convert_widowx_to_lerobot.py"
        if not python_path.exists():
            raise RuntimeError(f"LeRobot Python environment not found: {python_path}")
        if not converter_path.exists():
            raise RuntimeError(f"LeRobot converter not found: {converter_path}")
        env = os.environ.copy()
        env.setdefault("HF_HOME", "/tmp/lerobot_hf_cache")
        env.setdefault("HF_DATASETS_CACHE", "/tmp/lerobot_hf_cache/datasets")
        env["PYTHONUNBUFFERED"] = "1"
        with self.lock:
            if self._is_running_unlocked():
                return self._status_unlocked("LeRobot export is already running.")
            self.command = command
            self.output_root = str(plan["dataset"]["output_root"])
            self.returncode = None
            self.output.clear()
            self.process = subprocess.Popen(
                command,
                cwd=str(self.project_root),
                env=env,
                text=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            process = self.process
            self.last_message = f"LeRobot export started (PID {process.pid})."
            threading.Thread(target=self._watch, args=(process,), daemon=True).start()
            return self._status_unlocked(self.last_message)

    def _watch(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            with self.lock:
                self.output.append(line.rstrip())
        returncode = process.wait()
        with self.lock:
            self.returncode = returncode
            if self.process is process:
                self.last_message = (
                    f"LeRobot export complete: {self.output_root}"
                    if returncode == 0
                    else f"LeRobot export failed with code {returncode}."
                )
