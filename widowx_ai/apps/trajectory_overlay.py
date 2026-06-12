from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass(frozen=True)
class TrajectoryPoint:
    x: float
    y: float
    action: str | None = None


class TrajectoryOverlay:
    def __init__(self, points: list[TrajectoryPoint], role: str = "hamster_view") -> None:
        self.points = points
        self.role = role

    @classmethod
    def from_payload(cls, payload: Any) -> TrajectoryOverlay | None:
        if not isinstance(payload, dict):
            return None
        points: list[TrajectoryPoint] = []
        for raw_point in payload.get("points", []):
            if not isinstance(raw_point, dict):
                continue
            try:
                x = min(max(float(raw_point["x"]), 0.0), 1.0)
                y = min(max(float(raw_point["y"]), 0.0), 1.0)
            except (KeyError, TypeError, ValueError):
                continue
            action = str(raw_point.get("gripper_action") or "").strip().lower() or None
            points.append(TrajectoryPoint(x=x, y=y, action=action))
        return cls(points, str(payload.get("role") or "hamster_view")) if len(points) >= 2 else None

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "role": self.role,
            "points": [
                {"x": point.x, "y": point.y, "gripper_action": point.action}
                for point in self.points
            ],
        }

    def apply_jpeg(self, jpeg: bytes) -> bytes:
        if cv2 is None:
            raise RuntimeError("OpenCV is required to render trajectory overlays.")
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Cannot decode camera frame for trajectory overlay.")
        self.draw(image)
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            raise RuntimeError("Cannot encode camera frame with trajectory overlay.")
        return encoded.tobytes()

    def draw(self, image: Any) -> None:
        height, width = image.shape[:2]
        pixel_points = [
            (min(width - 1, int(point.x * width)), min(height - 1, int(point.y * height)))
            for point in self.points
        ]
        scale = max(0.75, min(width, height) / 640.0)
        line_width = max(3, round(4 * scale))
        radius = max(7, round(9 * scale))
        for index in range(1, len(pixel_points)):
            cv2.line(
                image,
                pixel_points[index - 1],
                pixel_points[index],
                self._trajectory_color(index, len(pixel_points) - 1),
                line_width,
                cv2.LINE_AA,
            )
        for index, (point, pixel_point) in enumerate(zip(self.points, pixel_points), start=1):
            color = (0, 0, 255) if point.action == "close" else (255, 0, 0) if point.action == "open" else (0, 255, 255)
            cv2.circle(image, pixel_point, radius, color, -1, cv2.LINE_AA)
            cv2.circle(image, pixel_point, radius + max(2, round(2 * scale)), (255, 255, 255), max(2, round(2 * scale)), cv2.LINE_AA)
            cv2.putText(
                image,
                str(index),
                (pixel_point[0] + radius + 3, pixel_point[1] - radius),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55 * scale,
                (255, 255, 255),
                max(2, round(2 * scale)),
                cv2.LINE_AA,
            )

    @staticmethod
    def _trajectory_color(index: int, total: int) -> tuple[int, int, int]:
        ratio = index / max(1, total)
        return (int(255 * (1.0 - ratio)), 180, int(255 * ratio))
