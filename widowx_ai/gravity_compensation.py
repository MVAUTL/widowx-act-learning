# Copyright 2025 Trossen Robotics
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Purpose:
# This script demonstrates how to do gravity compensation, which is useful for
# manually moving the arm to teach a trajectory or record specific positions.

# Hardware setup:
# 1. A WXAI V0 arm at 192.168.1.2

# The script does the following:
# 1. Initializes the driver
# 2. Configures the driver
# 3. Sets the external efforts to 0s
# 4. Waits for the user to press enter
# 5. Sets the mode to idle
# 6. The driver automatically sets the mode to idle at the destructor

from __future__ import annotations

import argparse
import socket
import sys

import trossen_arm


END_EFFECTORS = {
    "base": trossen_arm.StandardEndEffector.wxai_v0_base,
    "leader": trossen_arm.StandardEndEffector.wxai_v0_leader,
    "follower": trossen_arm.StandardEndEffector.wxai_v0_follower,
}
GRAVITY_PAYLOADS = {
    "variant": None,
    "d405-follower": trossen_arm.StandardEndEffector.wxai_v0_follower,
}
MIN_CAMERA_WRIST_EFFORT = -0.6
MAX_CAMERA_WRIST_EFFORT = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enable gravity compensation on a WidowX AI arm for kinesthetic teaching."
    )
    parser.add_argument("--ip", default="192.168.1.2", help="Arm controller IP address.")
    parser.add_argument("--port", type=int, default=50001, help="Arm controller TCP port.")
    parser.add_argument(
        "--variant",
        choices=sorted(END_EFFECTORS),
        default="base",
        help="WidowX AI end-effector variant. Adding a USB camera does not change this.",
    )
    parser.add_argument(
        "--payload-profile",
        choices=sorted(GRAVITY_PAYLOADS),
        default="d405-follower",
        help="Mass profile used during gravity compensation. Use d405-follower for the D405 camera mount.",
    )
    parser.add_argument(
        "--camera-wrist-effort",
        type=float,
        default=0.0,
        help="Extra external effort on wrist pitch joint in Nm. Tune slowly if the camera still tilts the gripper.",
    )
    parser.add_argument("--timeout", type=float, default=2.0, help="TCP preflight timeout in seconds.")
    args = parser.parse_args()
    if not MIN_CAMERA_WRIST_EFFORT <= args.camera_wrist_effort <= MAX_CAMERA_WRIST_EFFORT:
        parser.error(
            f"--camera-wrist-effort must be between {MIN_CAMERA_WRIST_EFFORT:.2f} "
            f"and {MAX_CAMERA_WRIST_EFFORT:.2f} Nm"
        )
    return args


def preflight_tcp(args: argparse.Namespace) -> None:
    try:
        with socket.create_connection((args.ip, args.port), timeout=args.timeout):
            return
    except OSError as exc:
        raise RuntimeError(
            f"Cannot reach {args.ip}:{args.port}. Check Ethernet static IP, cable, and arm power."
        ) from exc


def gravity_external_efforts(camera_wrist_effort: float) -> list[float]:
    efforts = [0.0] * 7
    efforts[4] = camera_wrist_effort
    return efforts


def main() -> int:
    args = parse_args()
    print(f"Connecting to WidowX AI at {args.ip} with variant={args.variant}...")
    preflight_tcp(args)

    driver = trossen_arm.TrossenArmDriver()
    driver.configure(trossen_arm.Model.wxai_v0, END_EFFECTORS[args.variant], args.ip, True)
    payload_end_effector = GRAVITY_PAYLOADS[args.payload_profile]
    if payload_end_effector is not None:
        driver.set_end_effector(payload_end_effector)

    print(
        "Starting gravity compensation "
        f"(payload={args.payload_profile}, wrist={args.camera_wrist_effort:.2f} Nm). "
        "Support the arm before pressing Enter to stop."
    )
    driver.set_all_modes(trossen_arm.Mode.external_effort)
    driver.set_all_external_efforts(
        gravity_external_efforts(args.camera_wrist_effort),
        0.0,
        False
    )

    input("Press Enter to end gravity compensation...")
    driver.set_all_modes(trossen_arm.Mode.idle)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
