from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trossen_arm


HOME = np.array([0.0, math.pi / 2, math.pi / 2, 0.0, 0.0, 0.0], dtype=float)
DEMO = HOME + np.array([0.0, 0.0, 0.0, 0.15, -0.12, 0.0], dtype=float)
REST = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
DEFAULT_MAX_SPEED = 0.3
MIN_MAX_SPEED = 0.05
MAX_MAX_SPEED = 1.5
REPLAY_GRIPPER_MAX_SPEED = 0.015
START_POSITION_MIN_TIME = 2.5
JOINT_LIMIT_TOLERANCE = 5e-3
DEFAULT_GRAVITY_PAYLOAD = "d405_follower"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
MIN_CAMERA_WRIST_EFFORT = -0.6
MAX_CAMERA_WRIST_EFFORT = 0.6
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
GRAVITY_PAYLOADS = {
    "variant": None,
    "d405_follower": trossen_arm.StandardEndEffector.wxai_v0_follower,
}
