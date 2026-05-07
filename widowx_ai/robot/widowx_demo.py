#!/usr/bin/env python3
"""Minimal WidowX AI control demo using the official trossen-arm driver."""

from __future__ import annotations

import argparse
import socket
import sys
import time

import numpy as np
import trossen_arm


END_EFFECTORS = {
    "base": trossen_arm.StandardEndEffector.wxai_v0_base,
    "leader": trossen_arm.StandardEndEffector.wxai_v0_leader,
    "follower": trossen_arm.StandardEndEffector.wxai_v0_follower,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read state from a Trossen WidowX AI arm, optionally run a short safe joint-space move."
    )
    parser.add_argument("--ip", default="192.168.1.2", help="Arm controller IP address.")
    parser.add_argument("--port", type=int, default=50001, help="Arm controller TCP port.")
    parser.add_argument(
        "--variant",
        choices=sorted(END_EFFECTORS),
        default="base",
        help="WidowX AI end-effector variant.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Connection timeout in seconds if supported by the installed driver.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Actually move the arm through a small joint-space demo.",
    )
    parser.add_argument(
        "--check-import",
        action="store_true",
        help="Only verify that trossen_arm imports and exposes the WidowX AI API.",
    )
    return parser.parse_args()


def preflight_tcp(args: argparse.Namespace) -> None:
    print(f"Checking TCP reachability on {args.ip}:{args.port} with timeout={args.timeout}s...")
    try:
        with socket.create_connection((args.ip, args.port), timeout=args.timeout):
            print("TCP preflight OK.")
    except OSError as exc:
        raise RuntimeError(
            f"Cannot reach the arm controller at {args.ip}:{args.port}. "
            "Check Ethernet cabling, PC static IP, controller power, and the arm IP address."
        ) from exc


def configure_driver(driver: trossen_arm.TrossenArmDriver, args: argparse.Namespace) -> None:
    end_effector = END_EFFECTORS[args.variant]
    try:
        driver.configure(trossen_arm.Model.wxai_v0, end_effector, args.ip, True, args.timeout)
    except TypeError:
        driver.configure(trossen_arm.Model.wxai_v0, end_effector, args.ip, True)


def print_robot_state(driver: trossen_arm.TrossenArmDriver) -> None:
    output = driver.get_robot_output()
    print("Connected.")
    print(f"Number of joints including gripper: {driver.get_num_joints()}")

    joint = getattr(output, "joint", None)
    if joint is None:
        print(f"Robot output: {output}")
        return

    positions = np.asarray(getattr(joint, "positions", []), dtype=float)
    velocities = np.asarray(getattr(joint, "velocities", []), dtype=float)
    if positions.size:
        print("Joint positions [rad]:", np.array2string(positions, precision=4))
    if velocities.size:
        print("Joint velocities [rad/s]:", np.array2string(velocities, precision=4))


def run_motion_demo(driver: trossen_arm.TrossenArmDriver) -> None:
    arm_joint_count = driver.get_num_joints() - 1
    if arm_joint_count != 6:
        raise RuntimeError(f"Expected 6 arm joints plus gripper, got {driver.get_num_joints()} joints.")

    home = np.array([0.0, np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0], dtype=float)
    small_offset = home + np.array([0.0, 0.0, 0.0, 0.15, -0.12, 0.0], dtype=float)

    print("Setting arm to position mode.")
    driver.set_arm_modes(trossen_arm.Mode.position)

    print("Moving to demo home pose.")
    driver.set_arm_positions(home, 3.0, True)
    time.sleep(0.5)

    print("Moving through a small wrist offset.")
    driver.set_arm_positions(small_offset, 2.0, True)
    time.sleep(0.5)

    print("Returning to demo home pose.")
    driver.set_arm_positions(home, 2.0, True)

    print("Opening gripper lightly.")
    driver.set_gripper_mode(trossen_arm.Mode.external_effort)
    driver.set_gripper_external_effort(10.0, 2.0, True)


def main() -> int:
    args = parse_args()
    if args.check_import:
        print("trossen_arm import OK")
        print("Model wxai_v0 OK:", hasattr(trossen_arm.Model, "wxai_v0"))
        print("End-effectors:", ", ".join(sorted(END_EFFECTORS)))
        return 0

    print(f"Connecting to WidowX AI at {args.ip} with variant={args.variant}...")
    preflight_tcp(args)
    driver = trossen_arm.TrossenArmDriver()
    configure_driver(driver, args)
    print_robot_state(driver)

    if not args.move:
        print("No motion requested. Re-run with --move to execute the demo trajectory.")
        return 0

    run_motion_demo(driver)
    print("Demo complete.")
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
