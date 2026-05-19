#!/usr/bin/env python3
"""Small local UI for testing WidowX AI Cartesian end-effector targets."""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import trossen_arm


END_EFFECTORS = {
    "base": trossen_arm.StandardEndEffector.wxai_v0_base,
    "leader": trossen_arm.StandardEndEffector.wxai_v0_leader,
    "follower": trossen_arm.StandardEndEffector.wxai_v0_follower,
}

POSE_KEYS = ("x", "y", "z", "rx", "ry", "rz")


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def _as_float_list(values: Any, *, length: int, name: str) -> list[float]:
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"{name} must be a list of {length} numbers")
    result = [float(value) for value in values]
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


class CartesianController:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.RLock()
        self.driver: trossen_arm.TrossenArmDriver | None = None
        self.connected = False
        self.last_pose = np.array([0.35, 0.0, 0.25, 0.0, 0.0, 0.0], dtype=float)
        self.last_joint_positions: list[float] = []
        self.last_message = "Dry-run ready" if not args.real else "Not connected"
        self.last_error: str | None = None
        self.gravity_enabled = False

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "real": self.args.real,
                "connected": self.connected,
                "ip": self.args.ip,
                "variant": self.args.variant,
                "pose": self.last_pose.tolist(),
                "joint_positions": self.last_joint_positions,
                "gravity_enabled": self.gravity_enabled,
                "message": self.last_message,
                "error": self.last_error,
                "limits": {
                    "max_translation_delta_m": self.args.max_translation_delta,
                    "max_rotation_delta_rad": self.args.max_rotation_delta,
                    "min_goal_time_s": self.args.min_goal_time,
                    "max_goal_time_s": self.args.max_goal_time,
                    "max_camera_wrist_effort_nm": self.args.max_camera_wrist_effort,
                },
            }

    def preflight_tcp(self) -> None:
        with socket.create_connection((self.args.ip, self.args.arm_port), timeout=self.args.timeout):
            return

    def connect(self) -> dict[str, Any]:
        with self.lock:
            if self.connected:
                return self.read_state()
            self.last_error = None
            if not self.args.real:
                self.connected = True
                self.last_message = "Dry-run connected"
                return self.status()

            self.preflight_tcp()
            driver = trossen_arm.TrossenArmDriver()
            end_effector = END_EFFECTORS[self.args.variant]
            try:
                driver.configure(
                    trossen_arm.Model.wxai_v0,
                    end_effector,
                    self.args.ip,
                    True,
                    self.args.timeout,
                )
            except TypeError:
                driver.configure(trossen_arm.Model.wxai_v0, end_effector, self.args.ip, True)
            driver.set_arm_modes(trossen_arm.Mode.position)
            self.driver = driver
            self.connected = True
            self.last_message = "Connected to real arm"
            return self.read_state()

    def disconnect(self) -> dict[str, Any]:
        with self.lock:
            if self.driver is not None:
                self.driver.set_all_modes(trossen_arm.Mode.idle)
                self.driver.cleanup()
            self.driver = None
            self.connected = False
            self.gravity_enabled = False
            self.last_message = "Disconnected"
            return self.status()

    def idle(self) -> dict[str, Any]:
        with self.lock:
            if self.driver is not None:
                self.driver.set_all_modes(trossen_arm.Mode.idle)
            self.gravity_enabled = False
            self.last_message = "Motors set to idle"
            return self.status()

    def gravity_external_efforts(self, camera_wrist_effort: float) -> list[float]:
        joint_count = 7
        if self.driver is not None:
            joint_count = int(self.driver.get_num_joints())
        efforts = [0.0] * joint_count
        if joint_count > 4:
            efforts[4] = camera_wrist_effort
        return efforts

    def set_gravity_compensation(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if not self.connected:
                raise RuntimeError("Connect first")
            enabled = bool(payload.get("enabled", True))
            camera_wrist_effort = float(payload.get("camera_wrist_effort", 0.0))
            if abs(camera_wrist_effort) > self.args.max_camera_wrist_effort:
                raise ValueError(
                    f"camera_wrist_effort must be between "
                    f"{-self.args.max_camera_wrist_effort:.2f} and "
                    f"{self.args.max_camera_wrist_effort:.2f} Nm"
                )

            if self.driver is not None:
                if enabled:
                    self.driver.set_all_modes(trossen_arm.Mode.external_effort)
                    self.driver.set_all_external_efforts(
                        self.gravity_external_efforts(camera_wrist_effort),
                        0.0,
                        False,
                    )
                else:
                    self.driver.set_arm_modes(trossen_arm.Mode.position)
            self.gravity_enabled = enabled
            self.last_message = (
                f"Gravity compensation on (wrist {camera_wrist_effort:.2f} Nm)"
                if enabled
                else "Gravity compensation off"
            )
            self.last_error = None
            return self.status()

    def read_state(self) -> dict[str, Any]:
        with self.lock:
            if self.driver is not None:
                self.last_pose = np.asarray(self.driver.get_cartesian_positions(), dtype=float)
                self.last_joint_positions = [float(value) for value in self.driver.get_all_positions()]
            self.last_message = "State refreshed"
            return self.status()

    def validate_target(self, target: list[float], goal_time: float) -> tuple[np.ndarray, float]:
        pose = np.asarray(target, dtype=float)
        if pose.shape != (6,):
            raise ValueError("target must contain exactly 6 values")
        if not np.all(np.isfinite(pose)):
            raise ValueError("target contains non-finite values")
        if not self.args.min_goal_time <= goal_time <= self.args.max_goal_time:
            raise ValueError(
                f"goal_time must be between {self.args.min_goal_time:.2f}s and "
                f"{self.args.max_goal_time:.2f}s"
            )

        translation_delta = float(np.linalg.norm(pose[:3] - self.last_pose[:3]))
        rotation_delta = float(np.linalg.norm(pose[3:] - self.last_pose[3:]))
        if translation_delta > self.args.max_translation_delta:
            raise ValueError(
                f"translation delta {translation_delta:.3f}m exceeds "
                f"{self.args.max_translation_delta:.3f}m"
            )
        if rotation_delta > self.args.max_rotation_delta:
            raise ValueError(
                f"rotation delta {rotation_delta:.3f}rad exceeds "
                f"{self.args.max_rotation_delta:.3f}rad"
            )
        return pose, goal_time

    def move(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if not self.connected:
                raise RuntimeError("Connect first")
            target = _as_float_list(payload.get("target"), length=6, name="target")
            goal_time = float(payload.get("goal_time", 2.0))
            pose, goal_time = self.validate_target(target, goal_time)
            interpolation_name = str(payload.get("interpolation", "cartesian"))
            if interpolation_name not in {"cartesian", "joint"}:
                raise ValueError("interpolation must be 'cartesian' or 'joint'")

            if self.driver is not None:
                interpolation = (
                    trossen_arm.InterpolationSpace.cartesian
                    if interpolation_name == "cartesian"
                    else trossen_arm.InterpolationSpace.joint
                )
                self.driver.set_arm_modes(trossen_arm.Mode.position)
                self.gravity_enabled = False
                self.driver.set_cartesian_positions(
                    pose.tolist(),
                    interpolation,
                    goal_time,
                    bool(payload.get("blocking", True)),
                )
                if bool(payload.get("blocking", True)):
                    self.last_pose = np.asarray(self.driver.get_cartesian_positions(), dtype=float)
                else:
                    self.last_pose = pose
            else:
                time.sleep(min(goal_time, 0.2))
                self.last_pose = pose
                self.gravity_enabled = False

            self.last_message = f"Target sent ({interpolation_name}, {goal_time:.2f}s)"
            self.last_error = None
            return self.status()


HTML = """<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WidowX Cartesian Test</title>
  <style>
    :root { color-scheme: light; --bg:#f6f7f9; --panel:#ffffff; --text:#17202a; --muted:#657080; --line:#d7dce2; --accent:#0f766e; --danger:#b42318; --warn:#9a6700; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; background: var(--bg); color: var(--text); }
    main { max-width: 1180px; margin: 0 auto; padding: 24px; }
    header { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; margin-bottom: 18px; }
    h1 { margin: 0; font-size: 24px; letter-spacing: 0; }
    .sub { color: var(--muted); margin-top: 4px; font-size: 14px; }
    .grid { display: grid; grid-template-columns: 1.1fr .9fr; gap: 16px; align-items: start; }
    section, .status { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }
    h2 { margin: 0 0 14px; font-size: 16px; }
    .row { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
    label { display: grid; gap: 6px; color: var(--muted); font-size: 13px; }
    input, select { width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 9px 10px; font: inherit; background: #fff; color: var(--text); }
    input:focus, select:focus { outline: 2px solid rgba(15,118,110,.18); border-color: var(--accent); }
    .buttons { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }
    button { border: 1px solid var(--line); border-radius: 6px; padding: 9px 12px; background: #fff; color: var(--text); cursor: pointer; font: inherit; }
    button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
    button.danger { color: var(--danger); border-color: #f0b8b2; }
    button:disabled { opacity: .5; cursor: wait; }
    .pose { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 8px; }
    .pose div { border: 1px solid var(--line); border-radius: 6px; padding: 10px; background: #fbfcfd; }
    .pose span { display: block; color: var(--muted); font-size: 12px; }
    .pose strong { display: block; margin-top: 4px; font-variant-numeric: tabular-nums; font-size: 15px; }
    .nudge { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 14px; }
    .nudge-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
    pre { min-height: 150px; max-height: 260px; overflow: auto; background: #111827; color: #e5e7eb; border-radius: 8px; padding: 12px; font-size: 12px; }
    .pill { display: inline-flex; align-items: center; gap: 6px; border: 1px solid var(--line); border-radius: 999px; padding: 6px 10px; background: #fff; font-size: 13px; }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--warn); }
    .dot.on { background: var(--accent); }
    .dot.err { background: var(--danger); }
    @media (max-width: 860px) {
      main { padding: 16px; }
      header { display: block; }
      .grid { grid-template-columns: 1fr; }
      .row, .nudge { grid-template-columns: 1fr; }
      .pose { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>WidowX Cartesian Test</h1>
      <div class="sub">Cible effecteur: x, y, z en metres; rx, ry, rz en radians angle-axis.</div>
    </div>
    <div class="pill"><span id="dot" class="dot"></span><span id="conn">...</span></div>
  </header>

  <div class="grid">
    <section>
      <h2>Pose actuelle</h2>
      <div id="pose" class="pose"></div>
      <div class="buttons">
        <button onclick="connect()">Connecter</button>
        <button onclick="refreshState()">Lire pose</button>
        <button onclick="copyCurrent()">Copier vers cible</button>
        <button class="danger" onclick="idle()">Idle</button>
        <button class="danger" onclick="disconnect()">Deconnecter</button>
      </div>
    </section>

    <section>
      <h2>Cible</h2>
      <div class="row">
        <label>x <input id="x" type="number" step="0.005"></label>
        <label>y <input id="y" type="number" step="0.005"></label>
        <label>z <input id="z" type="number" step="0.005"></label>
      </div>
      <div class="row" style="margin-top:12px">
        <label>rx <input id="rx" type="number" step="0.02"></label>
        <label>ry <input id="ry" type="number" step="0.02"></label>
        <label>rz <input id="rz" type="number" step="0.02"></label>
      </div>
      <div class="row" style="margin-top:12px">
        <label>temps mouvement <input id="goal_time" type="number" min="0.2" max="10" step="0.1" value="2.0"></label>
        <label>interpolation
          <select id="interpolation">
            <option value="cartesian">cartesian</option>
            <option value="joint">joint</option>
          </select>
        </label>
        <label>pas nudge <input id="step" type="number" min="0.001" step="0.001" value="0.01"></label>
      </div>
      <div class="buttons">
        <button class="primary" onclick="sendTarget()">Envoyer cible</button>
      </div>
      <div class="nudge">
        <div><label>x</label><div class="nudge-pair"><button onclick="nudge('x', -1)">-</button><button onclick="nudge('x', 1)">+</button></div></div>
        <div><label>y</label><div class="nudge-pair"><button onclick="nudge('y', -1)">-</button><button onclick="nudge('y', 1)">+</button></div></div>
        <div><label>z</label><div class="nudge-pair"><button onclick="nudge('z', -1)">-</button><button onclick="nudge('z', 1)">+</button></div></div>
      </div>
    </section>
  </div>

  <section style="margin-top:16px">
    <h2>Gravity compensation</h2>
    <div class="row">
      <label>effort poignet camera Nm <input id="camera_wrist_effort" type="number" min="-0.6" max="0.6" step="0.05" value="0.0"></label>
      <label>etat <input id="gravity_state" type="text" readonly value="off"></label>
      <label>mode <input type="text" readonly value="external_effort"></label>
    </div>
    <div class="buttons">
      <button onclick="gravity(true)">Activer gravity</button>
      <button onclick="gravity(false)">Desactiver gravity</button>
    </div>
  </section>

  <section style="margin-top:16px">
    <h2>Journal</h2>
    <pre id="log"></pre>
  </section>
</main>

<script>
const keys = ["x", "y", "z", "rx", "ry", "rz"];
let state = null;

function log(message, data) {
  const el = document.getElementById("log");
  const line = `[${new Date().toLocaleTimeString()}] ${message}` + (data ? "\\n" + JSON.stringify(data, null, 2) : "");
  el.textContent = line + "\\n\\n" + el.textContent;
}

async function api(path, payload) {
  const options = payload === undefined ? {} : {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  };
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw data;
  update(data);
  return data;
}

function update(data) {
  state = data;
  const dot = document.getElementById("dot");
  dot.className = "dot" + (data.error ? " err" : data.connected ? " on" : "");
  document.getElementById("conn").textContent = `${data.real ? "REAL" : "DRY"} | ${data.connected ? "connecte" : "deconnecte"} | ${data.gravity_enabled ? "gravity" : "position"} | ${data.message}`;
  const pose = document.getElementById("pose");
  pose.innerHTML = keys.map((key, i) => `<div><span>${key}</span><strong>${Number(data.pose[i]).toFixed(4)}</strong></div>`).join("");
  document.getElementById("gravity_state").value = data.gravity_enabled ? "on" : "off";
  log(data.message, data.error ? {error: data.error} : null);
}

function currentTarget() {
  return keys.map(key => Number(document.getElementById(key).value));
}

function copyCurrent() {
  if (!state) return;
  keys.forEach((key, i) => document.getElementById(key).value = Number(state.pose[i]).toFixed(4));
}

function nudge(key, sign) {
  const input = document.getElementById(key);
  const step = Number(document.getElementById("step").value || "0.01");
  input.value = (Number(input.value || "0") + sign * step).toFixed(4);
}

async function connect() { try { await api("/api/connect", {}); copyCurrent(); } catch (e) { log("Erreur connexion", e); } }
async function refreshState() { try { await api("/api/state"); } catch (e) { log("Erreur lecture", e); } }
async function idle() { try { await api("/api/idle", {}); } catch (e) { log("Erreur idle", e); } }
async function disconnect() { try { await api("/api/disconnect", {}); } catch (e) { log("Erreur deconnexion", e); } }
async function gravity(enabled) {
  try {
    await api("/api/gravity", {
      enabled,
      camera_wrist_effort: Number(document.getElementById("camera_wrist_effort").value || "0"),
    });
  } catch (e) {
    log("Erreur gravity", e);
  }
}

async function sendTarget() {
  const target = currentTarget();
  if (target.some(v => !Number.isFinite(v))) {
    log("Cible invalide", {target});
    return;
  }
  try {
    await api("/api/move", {
      target,
      goal_time: Number(document.getElementById("goal_time").value || "2"),
      interpolation: document.getElementById("interpolation").value,
      blocking: true,
    });
  } catch (e) {
    log("Mouvement refuse", e);
  }
}

refreshState().then(copyCurrent).catch(e => log("Initialisation", e));
</script>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    controller: CartesianController

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        try:
            if self.path == "/" or self.path == "/index.html":
                body = HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/api/state":
                _json_response(self, HTTPStatus.OK, self.controller.read_state())
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
        except Exception as exc:
            _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def do_POST(self) -> None:
        try:
            payload = _read_json(self)
            routes = {
                "/api/connect": self.controller.connect,
                "/api/disconnect": self.controller.disconnect,
                "/api/gravity": lambda: self.controller.set_gravity_compensation(payload),
                "/api/idle": self.controller.idle,
                "/api/move": lambda: self.controller.move(payload),
            }
            route = routes.get(self.path)
            if route is None:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            _json_response(self, HTTPStatus.OK, route())
        except (RuntimeError, ValueError, OSError, trossen_arm.RuntimeError) as exc:
            with self.controller.lock:
                self.controller.last_error = str(exc)
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc), **self.controller.status()})
        except Exception as exc:
            with self.controller.lock:
                self.controller.last_error = str(exc)
            _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc), **self.controller.status()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local WidowX AI Cartesian target test UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7866)
    parser.add_argument("--real", action="store_true", help="Connect to the physical arm. Default is dry-run.")
    parser.add_argument("--ip", default="192.168.1.2", help="Arm controller IP address.")
    parser.add_argument("--arm-port", type=int, default=50001, help="Arm controller TCP preflight port.")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--variant", choices=sorted(END_EFFECTORS), default="base")
    parser.add_argument("--max-translation-delta", type=float, default=0.05)
    parser.add_argument("--max-rotation-delta", type=float, default=0.35)
    parser.add_argument("--max-camera-wrist-effort", type=float, default=0.6)
    parser.add_argument("--min-goal-time", type=float, default=0.5)
    parser.add_argument("--max-goal-time", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    RequestHandler.controller = CartesianController(args)
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    url = f"http://{args.host}:{args.port}"
    mode = "REAL ARM" if args.real else "dry-run"
    print(f"Serving WidowX Cartesian Test UI at {url} ({mode})")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        RequestHandler.controller.disconnect()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
