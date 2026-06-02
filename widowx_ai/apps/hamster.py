from __future__ import annotations

import base64
import json
import os
import pty
import re
import select
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import suppress
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class HamsterService:
    def __init__(self, camera_controller: Any) -> None:
        self.camera_controller = camera_controller
        self.dgx_host = os.environ.get("DGX_SPARK_HOST", "192.168.100.36")
        self.dgx_user = os.environ.get("DGX_SPARK_USER", "guest")
        self.dgx_workdir = os.environ.get("DGX_HAMSTER_WORKDIR", "~/intern_matteo_vulliez")
        self.dgx_slurm_script = os.environ.get("DGX_HAMSTER_SLURM", "slurm/hamster_full_pipeline.slurm")
        self.dgx_job_name = os.environ.get("DGX_HAMSTER_JOB_NAME", "matteo-hamster-full")
        self.dgx_backend_url = os.environ.get("DGX_HAMSTER_URL", f"http://{self.dgx_host}:8000")

    def send_camera(self, payload: dict[str, Any]) -> dict[str, Any]:
        source = str(payload.get("source") or "").strip()
        if not source:
            raise RuntimeError("No camera source selected.")
        quest = str(payload.get("prompt") or "").strip()
        if not quest:
            quest = "Move the visible object to the target area"
        base_url = str(payload.get("base_url") or "http://192.168.100.36:8000").strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise RuntimeError("Hamster URL must start with http:// or https://.")

        frame = self.camera_controller.video_frame_jpeg(source)
        raw_response = self._request_hamster(base_url, frame, quest)
        answer = self._extract_answer(raw_response)
        parsed_points = self._parse_points(answer)
        return {
            "ok": True,
            "answer": answer,
            "output_image": self._output_image(frame, answer),
            "point_count": len(parsed_points),
            "points": [
                {"x": x, "y": y, "gripper_action": action}
                for x, y, action in parsed_points
            ],
            "raw_response": raw_response,
            "message": f"Hamster answered from {source}",
        }

    def status(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        backend = self._backend_status()
        remote = self._run_dgx_command(
            "squeue -h -u $USER -n "
            f"{self._shell_quote(self.dgx_job_name)} -o '%i %T %M %j' || true",
            timeout=10,
            password=self._password_from_payload(payload),
        )
        running_jobs = remote["stdout"].strip().splitlines() if remote["ok"] and remote["stdout"].strip() else []
        return {
            "ok": True,
            "backend_url": self.dgx_backend_url,
            "backend_ready": backend["ready"],
            "backend_message": backend["message"],
            "dgx_host": self.dgx_host,
            "job_name": self.dgx_job_name,
            "jobs": running_jobs,
            "ssh_ok": remote["ok"],
            "ssh_message": remote["message"],
            "message": self._status_message(backend["ready"], running_jobs, remote),
        }

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        script = str(payload.get("slurm_script") or self.dgx_slurm_script).strip()
        password = self._password_from_payload(payload)
        command = (
            f"cd {self.dgx_workdir} && "
            "if squeue -h -u $USER -n "
            f"{self._shell_quote(self.dgx_job_name)} | grep -q .; then "
            f"echo 'Hamster job already queued/running: {self.dgx_job_name}'; "
            "else "
            f"sbatch {self._shell_quote(script)}; "
            "fi"
        )
        result = self._run_dgx_command(command, timeout=20, password=password)
        if not result["ok"]:
            raise RuntimeError(result["message"])
        status = self.status(payload)
        status["message"] = result["stdout"].strip() or "Hamster start command sent."
        return status

    def stop(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        password = self._password_from_payload(payload)
        command = (
            "if squeue -h -u $USER -n "
            f"{self._shell_quote(self.dgx_job_name)} | grep -q .; then "
            f"scancel -n {self._shell_quote(self.dgx_job_name)} && "
            f"echo 'Cancelled {self.dgx_job_name}'; "
            "else "
            f"echo 'No running {self.dgx_job_name} job'; "
            "fi"
        )
        result = self._run_dgx_command(command, timeout=20, password=password)
        if not result["ok"]:
            raise RuntimeError(result["message"])
        status = self.status(payload)
        status["message"] = result["stdout"].strip() or "Hamster stop command sent."
        return status

    def _backend_status(self) -> dict[str, Any]:
        try:
            request = urllib.request.Request(f"{self.dgx_backend_url.rstrip('/')}/docs", method="GET")
            with urllib.request.urlopen(request, timeout=3) as response:
                ready = 200 <= response.status < 400
                return {
                    "ready": ready,
                    "message": f"HTTP {response.status}",
                }
        except urllib.error.URLError as exc:
            return {"ready": False, "message": str(exc.reason)}
        except Exception as exc:  # noqa: BLE001 - surfaced to the browser.
            return {"ready": False, "message": str(exc)}

    def _run_dgx_command(self, remote_command: str, *, timeout: int, password: str = "") -> dict[str, Any]:
        ssh_target = f"{self.dgx_user}@{self.dgx_host}"
        command = ["ssh", "-F", "/dev/null", "-4", "-o", "ConnectTimeout=8", ssh_target, remote_command]
        password = password or os.environ.get("DGX_SPARK_PASSWORD", "")
        if password:
            sshpass = shutil.which("sshpass")
            if not sshpass:
                return self._run_dgx_command_with_askpass(command, password, timeout=timeout)
            command = [sshpass, "-p", password, *command]
        else:
            command[2:2] = ["-o", "BatchMode=yes"]
        try:
            process = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "message": "Timed out while contacting DGX Spark over SSH.",
            }
        except FileNotFoundError as exc:
            return {
                "ok": False,
                "stdout": "",
                "stderr": "",
                "message": f"Required command not found: {exc.filename}",
            }

        ok = process.returncode == 0
        stderr = process.stderr.strip()
        stdout = process.stdout.strip()
        if ok:
            message = stdout or "DGX command completed."
        elif "Permission denied" in stderr and not password:
            message = (
                "SSH authentication failed. Configure an SSH key, or start the local UI with "
                "DGX_SPARK_PASSWORD set and sshpass installed."
            )
        else:
            message = stderr or stdout or f"DGX SSH command failed with exit code {process.returncode}."
        return {
            "ok": ok,
            "stdout": stdout,
            "stderr": stderr,
            "message": message,
        }

    def _run_dgx_command_with_askpass(self, command: list[str], password: str, *, timeout: int) -> dict[str, Any]:
        setsid = shutil.which("setsid")
        if not setsid:
            return self._run_dgx_command_with_pty(command, password, timeout=timeout)

        with tempfile.TemporaryDirectory(prefix="dgx_askpass_") as temp_dir:
            askpass = os.path.join(temp_dir, "askpass.sh")
            with open(askpass, "w", encoding="utf-8") as file:
                file.write("#!/bin/sh\n")
                file.write("printf '%s\\n' \"$DGX_ASKPASS_PASSWORD\"\n")
            os.chmod(askpass, 0o700)
            env = os.environ.copy()
            env["SSH_ASKPASS"] = askpass
            env["SSH_ASKPASS_REQUIRE"] = "force"
            env["DGX_ASKPASS_PASSWORD"] = password
            env.setdefault("DISPLAY", "localhost:0")
            try:
                process = subprocess.run(
                    [setsid, *command],
                    text=True,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                    check=False,
                    env=env,
                )
            except subprocess.TimeoutExpired as exc:
                return {
                    "ok": False,
                    "stdout": exc.stdout or "",
                    "stderr": exc.stderr or "",
                    "message": "Timed out while contacting DGX Spark over SSH.",
                }

        ok = process.returncode == 0
        stdout = process.stdout.strip()
        stderr = process.stderr.strip()
        if ok:
            message = stdout or "DGX command completed."
        elif "Permission denied" in stderr:
            message = "SSH authentication failed. Check the DGX SSH password."
        else:
            message = stderr or stdout or f"DGX SSH command failed with exit code {process.returncode}."
        return {
            "ok": ok,
            "stdout": stdout,
            "stderr": stderr,
            "message": message,
        }

    def _run_dgx_command_with_pty(self, command: list[str], password: str, *, timeout: int) -> dict[str, Any]:
        master_fd, slave_fd = pty.openpty()
        process = subprocess.Popen(
            command,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
        )
        os.close(slave_fd)

        output = bytearray()
        password_sent = False
        host_confirmed = False
        deadline = time.monotonic() + timeout
        timed_out = False

        try:
            while process.poll() is None:
                if time.monotonic() > deadline:
                    timed_out = True
                    process.kill()
                    break
                readable, _, _ = select.select([master_fd], [], [], 0.1)
                if not readable:
                    continue
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                output.extend(chunk)
                lowered = bytes(output).lower()
                if not host_confirmed and b"are you sure you want to continue connecting" in lowered:
                    os.write(master_fd, b"yes\n")
                    host_confirmed = True
                if not password_sent and b"password:" in lowered:
                    os.write(master_fd, f"{password}\n".encode("utf-8"))
                    password_sent = True

            while True:
                readable, _, _ = select.select([master_fd], [], [], 0)
                if not readable:
                    break
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                output.extend(chunk)
        finally:
            with suppress(OSError):
                os.close(master_fd)

        text = output.decode("utf-8", errors="replace").strip()
        if timed_out:
            return {
                "ok": False,
                "stdout": text,
                "stderr": "",
                "message": "Timed out while contacting DGX Spark over SSH.",
            }
        ok = process.returncode == 0
        if ok:
            message = self._clean_ssh_pty_output(text) or "DGX command completed."
        elif "permission denied" in text.lower():
            message = "SSH authentication failed. Check the DGX SSH password."
        else:
            message = text or f"DGX SSH command failed with exit code {process.returncode}."
        clean_output = self._clean_ssh_pty_output(text)
        return {
            "ok": ok,
            "stdout": clean_output,
            "stderr": "" if ok else text,
            "message": message,
        }

    @staticmethod
    def _clean_ssh_pty_output(text: str) -> str:
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.lower().endswith("'s password:"):
                continue
            if "permanently added" in stripped.lower():
                continue
            lines.append(stripped)
        return "\n".join(lines)

    @staticmethod
    def _password_from_payload(payload: dict[str, Any] | None) -> str:
        if not payload:
            return ""
        return str(payload.get("password") or payload.get("dgx_password") or "").strip()

    @staticmethod
    def _shell_quote(value: str) -> str:
        return "'" + value.replace("'", "'\\''") + "'"

    @staticmethod
    def _status_message(backend_ready: bool, jobs: list[str], remote: dict[str, Any]) -> str:
        if backend_ready:
            return "Hamster backend is ready."
        if jobs:
            return "Hamster SLURM job is queued/running; backend is not ready yet."
        if not remote["ok"]:
            return remote["message"]
        return "Hamster backend is stopped."

    def _request_hamster(self, base_url: str, frame_jpeg: bytes, quest: str) -> dict[str, Any]:
        image_b64 = base64.b64encode(frame_jpeg).decode("ascii")
        request_payload = {
            "model": "HAMSTER_dev",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                        {"type": "text", "text": self._trajectory_prompt(quest)},
                    ],
                }
            ],
            "max_tokens": 256,
            "num_beams": 1,
            "use_cache": False,
            "temperature": 0.0,
            "top_p": 0.95,
        }
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(request_payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer fake-key",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Hamster HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot reach Hamster at {base_url}: {exc.reason}") from exc

        try:
            return json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Hamster returned non-JSON response: {raw_body[:500]}") from exc

    @staticmethod
    def _trajectory_prompt(quest: str) -> str:
        return (
            f"\nIn the image, please execute the command described in <quest>{quest}</quest>.\n"
            "Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal.\n"
            "Format your answer as a list of tuples enclosed by <ans> and </ans> tags. For example:\n"
            "<ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), <action>Open Gripper</action>, "
            "(0.74, 0.21), <action>Close Gripper</action>, ...]</ans>\n"
            "The tuple denotes point x and y location of the end effector in the image. "
            "The action tags indicate gripper actions.\n"
            "Coordinates should be floats between 0 and 1, representing relative positions.\n"
            "Remember to provide points between <ans> and </ans> tags and think step by step."
        )

    @staticmethod
    def _extract_answer(raw_response: dict[str, Any]) -> str:
        try:
            content = raw_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return json.dumps(raw_response, indent=2)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
                elif item is not None:
                    parts.append(str(item))
            return "\n".join(parts) if parts else json.dumps(content, indent=2)
        return str(content)

    @staticmethod
    def _output_image(frame_jpeg: bytes, answer: str) -> str | None:
        if cv2 is None:
            return None
        image_array = np.frombuffer(frame_jpeg, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            return None
        points = HamsterService._parse_points(answer)
        if points:
            height, width = image.shape[:2]
            pixel_points = []
            actions = []
            current_action = "move"
            for x, y, action in points:
                current_action = action or current_action
                pixel_points.append((int(x * width), int(y * height)))
                actions.append(current_action)
            for index in range(1, len(pixel_points)):
                color = HamsterService._trajectory_color(index, max(1, len(pixel_points) - 1))
                cv2.line(image, pixel_points[index - 1], pixel_points[index], color, 3, cv2.LINE_AA)
            for index, point in enumerate(pixel_points):
                action = actions[index]
                color = (0, 0, 255) if action == "close" else (255, 0, 0) if action == "open" else (0, 255, 255)
                cv2.circle(image, point, 8, color, -1, cv2.LINE_AA)
                cv2.circle(image, point, 10, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(
                    image,
                    str(index + 1),
                    (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        HamsterService._draw_answer_label(image, answer)
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

    @staticmethod
    def _parse_points(answer: str) -> list[tuple[float, float, str | None]]:
        match = re.search(r"<ans>(.*?)</ans>", answer, re.DOTALL)
        body = match.group(1) if match else answer
        tokens = re.findall(
            r"\(([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\)|<action>(Open|Close) Gripper</action>",
            body,
            re.IGNORECASE,
        )
        points: list[tuple[float, float, str | None]] = []
        pending_action: str | None = None
        for x_raw, y_raw, action_raw in tokens:
            if action_raw:
                pending_action = action_raw.lower()
                if points:
                    x, y, _ = points[-1]
                    points[-1] = (x, y, pending_action)
                continue
            try:
                x = min(max(float(x_raw), 0.0), 1.0)
                y = min(max(float(y_raw), 0.0), 1.0)
            except ValueError:
                continue
            points.append((x, y, pending_action))
        return points

    @staticmethod
    def _trajectory_color(index: int, total: int) -> tuple[int, int, int]:
        ratio = index / max(1, total)
        red = int(255 * ratio)
        blue = int(255 * (1.0 - ratio))
        green = 180
        return (blue, green, red)

    @staticmethod
    def _draw_answer_label(image: Any, answer: str) -> None:
        label = answer.replace("\n", " ")
        if len(label) > 150:
            label = label[:147] + "..."
        cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(
            image,
            label,
            (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
