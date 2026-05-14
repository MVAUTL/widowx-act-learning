from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class HamsterService:
    def __init__(self, camera_controller: Any) -> None:
        self.camera_controller = camera_controller

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
        return {
            "ok": True,
            "answer": answer,
            "output_image": self._output_image(frame, answer),
            "raw_response": raw_response,
            "message": f"Hamster answered from {source}",
        }

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
