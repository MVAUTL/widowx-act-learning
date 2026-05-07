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

try:
    import cv2
    import pyrealsense2 as rs
except ImportError:
    cv2 = None
    rs = None


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


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WidowX AI Control</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101214;
      --panel: #181c20;
      --panel-2: #20262b;
      --text: #eef2f4;
      --muted: #a6b0b8;
      --accent: #37c48d;
      --danger: #e45858;
      --warning: #e6b450;
      --line: #2d363d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    main {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 36px;
    }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0;
      font-size: 30px;
      line-height: 1.05;
      font-weight: 720;
    }
    .sub {
      margin-top: 6px;
      color: var(--muted);
      font-size: 14px;
    }
    .status {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 10px 12px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 260px;
      justify-content: space-between;
      font-size: 14px;
    }
    .status-main {
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--danger);
    }
    .dot.ok { background: var(--accent); }
    .dot.warn { background: var(--warning); }
    .layout {
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 18px;
      align-items: start;
    }
    section, aside {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }
    .toolbar {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .tool-group {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: flex-start;
      align-content: flex-start;
      min-height: 100%;
      padding: 10px;
      background: #151a1e;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .tool-meta {
      width: 100%;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
    }
    .tool-label {
      min-width: 92px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    button {
      height: 40px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--panel-2);
      color: var(--text);
      padding: 0 14px;
      font-size: 14px;
      cursor: pointer;
    }
    button:hover { border-color: #52616b; }
    button.primary { background: #1f5f49; border-color: #2b8c67; }
    button.danger { background: #642828; border-color: #9b3d3d; }
    button.emergency {
      width: 100%;
      height: 64px;
      background: #a51616;
      border-color: #ff4d4d;
      color: white;
      font-size: 20px;
      font-weight: 800;
      text-transform: uppercase;
      margin-bottom: 14px;
    }
    button:disabled { opacity: .45; cursor: not-allowed; }
    .armed {
      display: flex;
      align-items: center;
      gap: 10px;
      min-height: 40px;
      padding: 0 12px;
      background: #171a1d;
      border: 1px solid var(--line);
      border-radius: 7px;
      color: var(--muted);
      font-size: 14px;
    }
    input[type="checkbox"] { width: 18px; height: 18px; }
    .joint {
      display: grid;
      grid-template-columns: 70px 1fr 92px;
      gap: 12px;
      align-items: center;
      padding: 13px 0;
      border-top: 1px solid var(--line);
    }
    .joint:first-child { border-top: 0; }
    .joint label {
      color: var(--muted);
      font-size: 14px;
    }
    input[type="range"] { width: 100%; }
    input[type="number"], input[type="text"] {
      width: 100%;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #111518;
      color: var(--text);
      padding: 0 10px;
      font-size: 14px;
    }
    select {
      width: 100%;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #111518;
      color: var(--text);
      padding: 0 10px;
      font-size: 14px;
    }
    .side-title {
      margin: 0 0 12px;
      font-size: 15px;
      color: var(--muted);
      font-weight: 650;
      text-transform: uppercase;
    }
    .kv {
      display: grid;
      grid-template-columns: 110px 1fr;
      gap: 8px 12px;
      margin-bottom: 18px;
      font-size: 14px;
    }
    .kv div:nth-child(odd) { color: var(--muted); }
    pre {
      min-height: 180px;
      max-height: 340px;
      overflow: auto;
      margin: 0;
      white-space: pre-wrap;
      background: #0b0d0f;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 12px;
      color: #d8dee3;
      font-size: 12px;
      line-height: 1.45;
    }
    .camera-panel {
      margin-top: 18px;
      padding-top: 18px;
      border-top: 1px solid var(--line);
    }
    .camera-controls {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto auto auto;
      gap: 10px;
      margin-bottom: 12px;
    }
    select {
      width: 100%;
      height: 40px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #111518;
      color: var(--text);
      padding: 0 10px;
      font-size: 14px;
    }
    .camera-view {
      position: relative;
      width: 100%;
      aspect-ratio: 4 / 3;
      background: #0b0d0f;
      border: 1px solid var(--line);
      border-radius: 7px;
      overflow: hidden;
      display: grid;
      place-items: center;
      color: var(--muted);
      font-size: 14px;
    }
    .camera-view img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: none;
    }
    .camera-meta {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
      margin-top: 8px;
    }
    .camera-note {
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }
    @media (max-width: 880px) {
      header { align-items: stretch; flex-direction: column; }
      .layout { grid-template-columns: 1fr; }
      .toolbar { grid-template-columns: 1fr; }
      .joint { grid-template-columns: 56px 1fr 82px; gap: 8px; }
      .camera-controls { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>WidowX AI Control</h1>
        <div class="sub" id="subtitle"></div>
      </div>
      <div class="status"><span class="status-main"><span class="dot warn" id="dot"></span> <span id="state">Initializing</span></span><span id="mode"></span></div>
    </header>
    <div class="layout">
      <section>
        <button class="emergency" onclick="emergencyStop()">Emergency stop</button>
        <div class="toolbar">
          <div class="tool-group">
            <span class="tool-label">Session</span>
            <button class="primary" onclick="connectArm()">Connect</button>
            <button class="danger" onclick="disconnectArm()">Disconnect</button>
            <button onclick="refreshStatus()">Refresh</button>
            <label class="armed"><input type="checkbox" id="armed"> enable motion</label>
          </div>
          <div class="tool-group">
            <span class="tool-label">Positions</span>
            <button onclick="home()">Home</button>
            <button onclick="goToStartPosition()">Start position</button>
            <button onclick="saveStartPosition()">Save start</button>
            <button onclick="rest()">Rest</button>
          </div>
          <div class="tool-group">
            <span class="tool-label">Gripper</span>
            <button onclick="gripper(10)">Open gripper</button>
            <button onclick="gripper(-10)">Close gripper</button>
            <div class="tool-meta">Opening: <span id="gripperStatus">n/a</span></div>
          </div>
          <div class="tool-group">
            <span class="tool-label">Tools</span>
            <button onclick="gravityCompensation()">Gravity comp</button>
            <button id="holdButton" onclick="toggleHold()">Hold</button>
            <button onclick="location.href='/teach'">Teaching</button>
            <button onclick="location.href='/model-test'">Model test</button>
          </div>
        </div>
        <div id="joints"></div>
        <div class="camera-panel">
          <h2 class="side-title">Camera Hub</h2>
          <div class="camera-controls">
            <select id="cameraSource" onchange="handleCameraSourceChange()"></select>
            <button onclick="refreshCameraHub(true)">Refresh cameras</button>
            <button onclick="startCameraPreview()">Start live</button>
            <button onclick="stopCameraPreview()">Stop live</button>
          </div>
          <div class="camera-view">
            <img id="cameraImage" alt="Camera preview">
            <span id="cameraPlaceholder">No camera selected</span>
          </div>
          <div class="camera-meta">
            <span id="cameraState">Scanning cameras</span>
            <span id="cameraDetail"></span>
          </div>
          <div class="camera-note">One control module manages the D405 and all detected USB cameras from the same selector with a continuous live stream.</div>
        </div>
      </section>
      <aside>
        <h2 class="side-title">Configuration</h2>
        <div class="kv">
          <div>IP</div><div id="ip"></div>
          <div>Variant</div><div id="variant"></div>
          <div>Port</div><div id="port"></div>
          <div>Gravity</div><div id="gravity"></div>
          <div>Max speed</div><div><span id="speedValue"></span> rad/s</div>
        </div>
        <input id="speedSlider" type="range" min="0.05" max="1.5" step="0.05" value="0.3" oninput="setMaxSpeed(this.value)">
        <p class="camera-note">Home, Rest, Demo, Return to start and replay follow this speed limit. Default is 0.30 rad/s.</p>
        <h2 class="side-title">Log</h2>
        <pre id="log"></pre>
      </aside>
    </div>
  </main>
  <script>
    const limits = LIMITS_PLACEHOLDER;
    let joints = [0, Math.PI / 2, Math.PI / 2, 0, 0, 0];
    let config = {};
    let cameraRunning = false;
    let cameraTimer = null;
    let cameraSources = [];
    let activeCameraSource = '';
    let resolutionState = { top_view: "640x480", d405: "640x480" };

    function log(msg) {
      const el = document.getElementById('log');
      const stamp = new Date().toLocaleTimeString();
      el.textContent = `[${stamp}] ${msg}\\n` + el.textContent;
    }

    async function api(path, payload = null) {
      const opts = payload ? {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      } : {};
      const res = await fetch(path, opts);
      const data = await res.json();
      if (!res.ok || data.ok === false) throw new Error(data.error || res.statusText);
      return data;
    }

    function renderJoints() {
      const root = document.getElementById('joints');
      root.innerHTML = '';
      joints.forEach((value, i) => {
        const row = document.createElement('div');
        row.className = 'joint';
        const min = limits[i][0], max = limits[i][1];
        row.innerHTML = `
          <label>Joint ${i}</label>
          <input type="range" min="${min}" max="${max}" step="0.001" value="${value}" oninput="setJoint(${i}, this.value)">
          <input type="number" min="${min}" max="${max}" step="0.001" value="${value.toFixed(3)}" onchange="setJoint(${i}, this.value)">
        `;
        root.appendChild(row);
      });
    }

    function setJoint(i, value) {
      joints[i] = Number(value);
      renderJoints();
      move();
    }

    async function move() {
      if (!document.getElementById('armed').checked) return;
      try {
        const data = await api('/api/move', {positions: joints, armed: true});
        log(data.message);
        await refreshStatus(false);
      } catch (err) {
        log(`ERROR: ${err.message}`);
      }
    }

    function updateStatus(data) {
      config = data.config;
      if (Array.isArray(data.positions) && data.positions.length >= 6) joints = data.positions.slice(0, 6);
      document.getElementById('subtitle').textContent = `${config.real ? 'Real mode' : 'Dry-run'} · ${config.ip}`;
      document.getElementById('mode').textContent = config.real ? 'REAL' : 'DRY';
      document.getElementById('ip').textContent = config.ip;
      document.getElementById('variant').textContent = config.variant;
      document.getElementById('port').textContent = config.port;
      document.getElementById('gravity').textContent = data.gravity_compensation ? 'on' : 'off';
      document.getElementById('gripperStatus').textContent = formatGripperStatus(data.gripper_position);
      document.getElementById('holdButton').textContent = data.hold ? 'Hold off' : 'Hold';
      document.getElementById('speedValue').textContent = Number(data.max_speed).toFixed(2);
      document.getElementById('speedSlider').value = data.max_speed;
      document.getElementById('state').textContent = data.connected ? 'Connected' : 'Disconnected';
      const dot = document.getElementById('dot');
      dot.className = 'dot ' + (data.connected ? 'ok' : (config.real ? '' : 'warn'));
      renderJoints();
    }

    function formatGripperStatus(position) {
      if (position == null || !Number.isFinite(Number(position))) return 'n/a';
      const meters = Number(position);
      const mm = meters * 1000;
      return `${mm.toFixed(1)} mm open`;
    }

    function setBackendOffline(message = 'No backend') {
      document.getElementById('subtitle').textContent = message;
      document.getElementById('mode').textContent = 'OFFLINE';
      document.getElementById('state').textContent = 'No backend';
      document.getElementById('dot').className = 'dot';
      document.getElementById('cameraState').textContent = 'No backend';
      document.getElementById('cameraDetail').textContent = 'Restart the control server';
      document.getElementById('gripperStatus').textContent = 'n/a';
      cameraRunning = false;
      stopCameraElement();
    }

    async function setMaxSpeed(value) {
      try {
        const speed = Number(value);
        document.getElementById('speedValue').textContent = speed.toFixed(2);
        const data = await api('/api/max_speed', {max_speed: speed});
        updateStatus(data);
      } catch (err) {
        log(`ERROR: ${err.message}`);
      }
    }

    function gravityPayload() {
      return {
        armed: document.getElementById('armed').checked,
        payload_profile: 'd405_follower',
        camera_wrist_effort: 0.0
      };
    }

    async function refreshStatus(writeLog = true) {
      try {
        const data = await api('/api/status');
        updateStatus(data);
        if (writeLog) log('Status updated');
      } catch (err) {
        setBackendOffline();
        log(`ERROR: ${err.message}`);
      }
    }

    async function emergencyStop() {
      try {
        const data = await api('/api/emergency_stop', {});
        updateStatus(data);
        log(data.message);
      } catch (err) {
        log(`ERROR emergency: ${err.message}`);
      }
    }

    function renderCameraSources(sources, activeSource) {
      const select = document.getElementById('cameraSource');
      const previous = select.value;
      cameraSources = Array.isArray(sources) ? sources : [];
      select.innerHTML = '';
      if (cameraSources.length === 0) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'No camera detected';
        select.appendChild(option);
        select.disabled = true;
        activeCameraSource = '';
        return;
      }
      select.disabled = false;
      cameraSources.forEach((source) => {
        const option = document.createElement('option');
        option.value = source.id;
        option.textContent = source.label;
        select.appendChild(option);
      });
      const preferred = activeSource || previous;
      const exists = cameraSources.some((source) => source.id === preferred);
      select.value = exists ? preferred : cameraSources[0].id;
      activeCameraSource = select.value;
    }

    function selectedCameraSource() {
      const value = document.getElementById('cameraSource').value;
      return value || '';
    }

    async function refreshCameraHub(writeLog = false) {
      try {
        const data = await api('/api/video/status');
        cameraRunning = data.running;
        renderCameraSources(data.sources || [], data.active_source);
        activeCameraSource = data.active_source || selectedCameraSource();
        document.getElementById('cameraState').textContent = data.sources.length
          ? (data.running ? `Live stream active · ${data.active_label}` : `${data.sources.length} camera(s) detected`)
          : 'No camera detected';
        document.getElementById('cameraDetail').textContent = data.active_detail || '';
        if (writeLog) log(data.message);
        updateCameraView();
      } catch (err) {
        setBackendOffline();
        log(`ERROR camera: ${err.message}`);
      }
    }

    function cameraFrameUrl() {
      const source = selectedCameraSource();
      if (!source) return '';
      return `/api/video/frame?source=${encodeURIComponent(source)}&t=${Date.now()}`;
    }

    function stopCameraElement() {
      const img = document.getElementById('cameraImage');
      const placeholder = document.getElementById('cameraPlaceholder');
      if (cameraTimer) {
        clearInterval(cameraTimer);
        cameraTimer = null;
      }
      img.src = '';
      img.style.display = 'none';
      placeholder.style.display = 'block';
    }

    function updateCameraFrame() {
      if (!cameraRunning) return;
      const img = document.getElementById('cameraImage');
      const placeholder = document.getElementById('cameraPlaceholder');
      const frameUrl = cameraFrameUrl();
      if (!frameUrl) {
        stopCameraElement();
        placeholder.textContent = 'No camera selected';
        return;
      }
      img.style.display = 'block';
      placeholder.style.display = 'none';
      img.src = frameUrl;
    }

    function updateCameraView() {
      const img = document.getElementById('cameraImage');
      const placeholder = document.getElementById('cameraPlaceholder');
      if (!cameraRunning) {
        stopCameraElement();
        placeholder.textContent = selectedCameraSource() ? 'Live stream stopped' : 'No camera selected';
        return;
      }
      const frameUrl = cameraFrameUrl();
      if (!frameUrl) {
        stopCameraElement();
        placeholder.textContent = 'No camera selected';
        return;
      }
      img.style.display = 'block';
      placeholder.style.display = 'none';
      if (!cameraTimer) {
        cameraTimer = setInterval(updateCameraFrame, 150);
      }
      img.src = frameUrl;
    }

    async function handleCameraSourceChange() {
      activeCameraSource = selectedCameraSource();
      if (!cameraRunning) return;
      await startCameraPreview();
    }

    async function startCameraPreview() {
      try {
        const source = selectedCameraSource();
        if (!source) {
          log('No camera selected');
          return;
        }
        let res = "640x480";
        if (source.includes('d405')) res = resolutionState.d405;
        else res = resolutionState.top_view;
        const [w, h] = res.split('x').map(Number);
        const data = await api('/api/video/start', {source, width: w, height: h});
        log(data.message);
        await refreshCameraHub(false);
      } catch (err) {
        log(`ERROR camera: ${err.message}`);
      }
    }

    async function stopCameraPreview() {
      try {
        const data = await api('/api/video/stop', {});
        log(data.message);
        stopCameraElement();
        await refreshCameraHub(false);
      } catch (err) {
        log(`ERROR camera: ${err.message}`);
      }
    }

    async function connectArm() {
      try {
        const data = await api('/api/connect', {});
        updateStatus(data);
        log(data.message);
      } catch (err) {
        log(`ERROR: ${err.message}`);
      }
    }

    async function disconnectArm() {
      try {
        const data = await api('/api/disconnect', {});
        updateStatus(data);
        log(data.message);
      } catch (err) {
        log(`ERROR: ${err.message}`);
      }
    }

    async function home() {
      try {
        const data = await api('/api/home', {armed: document.getElementById('armed').checked});
        updateStatus(data);
        log(data.message);
      } catch (err) {
        log(`ERROR: ${err.message}`);
      }
    }

    async function rest() {
      try {
        const data = await api('/api/rest', {armed: document.getElementById('armed').checked});
        updateStatus(data);
        log(data.message);
      } catch (err) {
        log(`ERROR: ${err.message}`);
      }
    }

    async function gripper(effort) {
      try {
        const data = await api('/api/gripper', {effort, armed: document.getElementById('armed').checked});
        updateStatus(data);
        log(data.message);
      } catch (err) {
        log(`ERROR: ${err.message}`);
      }
    }

    async function saveStartPosition() {
      try {
        const data = await api('/api/start_position/save', {});
        updateStatus(data);
        log(data.message);
      } catch (err) {
        log(`ERROR start position: ${err.message}`);
      }
    }

    async function goToStartPosition() {
      try {
        const data = await api('/api/start_position/go', {armed: document.getElementById('armed').checked});
        updateStatus(data);
        log(data.message);
      } catch (err) {
        log(`ERROR start position: ${err.message}`);
      }
    }

    async function gravityCompensation() {
      try {
        const data = await api('/api/gravity_compensation', gravityPayload());
        updateStatus(data);
        log(data.message);
      } catch (err) {
        log(`ERROR: ${err.message}`);
      }
    }

    async function toggleHold() {
      try {
        const payload = gravityPayload();
        payload.enabled = document.getElementById('holdButton').textContent === 'Hold';
        const data = await api('/api/hold', payload);
        updateStatus(data);
        log(data.message);
      } catch (err) {
        log(`ERROR: ${err.message}`);
      }
    }

    refreshStatus();
    refreshCameraHub();
    setInterval(() => refreshStatus(false), 2000);
    setInterval(() => refreshCameraHub(false), 4000);
  </script>
</body>
</html>
"""


MODEL_TEST_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WidowX AI Model Test</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101214;
      --panel: #181c20;
      --panel-2: #20262b;
      --text: #eef2f4;
      --muted: #a6b0b8;
      --accent: #37c48d;
      --danger: #e45858;
      --warning: #e6b450;
      --line: #2d363d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    main {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 36px;
    }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 18px;
    }
    h1 { margin: 0; font-size: 30px; line-height: 1.05; font-weight: 720; }
    a { color: var(--accent); text-decoration: none; }
    .sub { margin-top: 6px; color: var(--muted); font-size: 14px; }
    .layout {
      display: grid;
      grid-template-columns: 1fr 360px;
      gap: 18px;
      align-items: start;
    }
    section, aside {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .field label {
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 6px;
    }
    input[type="number"], input[type="text"] {
      width: 100%;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #111518;
      color: var(--text);
      padding: 0 10px;
      font-size: 14px;
    }
    button {
      height: 40px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--panel-2);
      color: var(--text);
      padding: 0 14px;
      font-size: 14px;
      cursor: pointer;
    }
    button:hover { border-color: #52616b; }
    button.primary { background: #1f5f49; border-color: #2b8c67; }
    button.danger { background: #642828; border-color: #9b3d3d; }
    button.emergency {
      width: 100%;
      height: 64px;
      background: #a51616;
      border-color: #ff4d4d;
      color: white;
      font-size: 20px;
      font-weight: 800;
      text-transform: uppercase;
      margin-bottom: 14px;
    }
    .armed {
      display: flex;
      align-items: center;
      gap: 10px;
      min-height: 40px;
      padding: 0 12px;
      background: #171a1d;
      border: 1px solid var(--line);
      border-radius: 7px;
      color: var(--muted);
      font-size: 14px;
    }
    input[type="checkbox"] { width: 18px; height: 18px; }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 14px 0 18px;
    }
    .note {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
      margin-bottom: 16px;
    }
    .warning {
      border: 1px solid #8a6a2a;
      background: #211a0e;
      color: #f0d490;
      border-radius: 8px;
      padding: 12px;
      font-size: 13px;
      line-height: 1.45;
      margin-bottom: 16px;
    }
    .runtime-warning {
      border: 1px solid #c07a2c;
      background: #2a1708;
      color: #ffd28a;
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 13px;
      line-height: 1.45;
      margin-bottom: 12px;
    }
    .runtime-warning.hidden { display: none; }
    .side-title {
      margin: 0 0 12px;
      font-size: 15px;
      color: var(--muted);
      font-weight: 650;
      text-transform: uppercase;
    }
    .kv {
      display: grid;
      grid-template-columns: 130px 1fr;
      gap: 8px 12px;
      margin-bottom: 18px;
      font-size: 14px;
    }
    .kv div:nth-child(odd) { color: var(--muted); }
    pre {
      min-height: 360px;
      max-height: 560px;
      overflow: auto;
      margin: 0;
      white-space: pre-wrap;
      background: #0b0d0f;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 12px;
      color: #d8dee3;
      font-size: 12px;
      line-height: 1.45;
    }
    pre.warning-output {
      border-color: #c07a2c;
      background: #120c06;
      color: #ffd28a;
    }
    canvas {
      width: 100%;
      height: 320px;
      display: block;
      background: #0b0d0f;
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-top: 14px;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .swatch {
      display: inline-block;
      width: 11px;
      height: 11px;
      border-radius: 2px;
      margin-right: 6px;
      vertical-align: -1px;
    }
    @media (max-width: 880px) {
      header { align-items: stretch; flex-direction: column; }
      .layout, .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Model test</h1>
        <div class="sub">Test tres limite du checkpoint ACT avec garde-fous logiciels</div>
      </div>
      <a href="/">Back to control</a>
    </header>
    <div class="layout">
      <section>
        <button class="emergency" onclick="emergencyStop()">Emergency stop</button>
        <div class="warning">
          Le mode reel n'est pas une garantie anti-collision. Pour reduire le risque: table degagee,
          bras deja proche de la pose de depart, main proche de l'arret d'urgence, vitesse basse,
          et premier test avec un seul pas.
        </div>
        <div class="grid">
          <div class="field">
            <label>Checkpoint</label>
            <input id="checkpoint" type="text" value="widowx_ai/models/act_100ep_20260430_0520/best.pt">
          </div>
          <div class="field">
            <label>Steps</label>
            <select id="steps">
              <option value="1" selected>1 step - first contact</option>
              <option value="3">3 steps - very short</option>
              <option value="5">5 steps - short</option>
              <option value="10">10 steps</option>
              <option value="20">20 steps</option>
              <option value="30">30 steps</option>
              <option value="50">50 steps - long</option>
            </select>
          </div>
          <div class="field">
            <label>Period seconds</label>
            <input id="period" type="number" min="0.5" max="5" step="0.1" value="1.0">
          </div>
          <div class="field">
            <label>Max speed rad/s</label>
            <input id="maxSpeed" type="number" min="0.02" max="0.20" step="0.01" value="0.05">
          </div>
          <div class="field">
            <label>Max joint step rad</label>
            <input id="maxStepRad" type="number" min="0.005" max="0.10" step="0.005" value="0.035">
          </div>
          <div class="field">
            <label>Envelope margin rad</label>
            <input id="envelopeMargin" type="number" min="0.0" max="0.20" step="0.01" value="0.08">
          </div>
          <div class="field">
            <label>Safety action</label>
            <select id="collisionAction">
              <option value="gravity" selected>Anti-gravity compensation</option>
              <option value="idle">Idle / brake motors</option>
            </select>
          </div>
          <div class="field">
            <label>Stall error rad</label>
            <input id="stallErrorRad" type="number" min="0.015" max="0.12" step="0.005" value="0.025">
          </div>
          <div class="field">
            <label>Stall velocity rad/s</label>
            <input id="stallVelocity" type="number" min="0.001" max="0.05" step="0.001" value="0.020">
          </div>
          <div class="field">
            <label>Stall seconds</label>
            <input id="stallSeconds" type="number" min="0.1" max="2.0" step="0.05" value="0.20">
          </div>
        </div>
        <label class="armed"><input type="checkbox" id="armed"> enable real model motion</label>
        <div class="actions">
          <button onclick="connectModelArm()">Connect arm</button>
          <button onclick="disconnectModelArm()">Disconnect arm</button>
          <button onclick="runModel(false)">Dry-run test</button>
          <button class="primary" onclick="runModel(true)">REAL run selected steps</button>
          <button onclick="goToStartPositionModel()">Return to start position</button>
          <button onclick="openMonitor()">Open training monitor</button>
        </div>
        <div class="note">
          Dry-run utilise une image d'un dataset existant et n'envoie aucun mouvement. Le test reel lance
          le nombre de pas indique dans Steps, puis remet le bras en mode securite. Il est bloque si la
          case de confirmation n'est pas cochee.
        </div>
        <canvas id="trajectoryChart" width="1000" height="320"></canvas>
        <div class="legend">
          <span><span class="swatch" style="background:#37c48d"></span>J0</span>
          <span><span class="swatch" style="background:#e6b450"></span>J1</span>
          <span><span class="swatch" style="background:#58a6ff"></span>J2</span>
          <span><span class="swatch" style="background:#ff7b72"></span>J3</span>
          <span><span class="swatch" style="background:#c297ff"></span>J4</span>
          <span><span class="swatch" style="background:#7ee787"></span>J5</span>
        </div>
      </section>
      <aside>
        <h2 class="side-title">Status</h2>
        <div class="kv">
          <div>Interface</div><div id="mode">-</div>
          <div>Connected</div><div id="connected">-</div>
          <div>Checkpoint</div><div>best.pt</div>
        </div>
        <h2 class="side-title">Output</h2>
        <div class="runtime-warning hidden" id="modelWarning"></div>
        <pre id="output">Ready.</pre>
      </aside>
    </div>
  </main>
  <script>
    async function api(path, payload = null) {
      const opts = payload ? {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      } : {};
      const res = await fetch(path, opts);
      const data = await res.json();
      if (!res.ok || data.ok === false) throw new Error(data.error || res.statusText);
      return data;
    }

    function isSafetyWarningText(text) {
      return /SAFETY WARNING|Force\\/stall guard triggered|Target outside training envelope|Max runtime reached/i.test(String(text || ''));
    }

    function output(text, level = 'info') {
      const outputElement = document.getElementById('output');
      const warning = document.getElementById('modelWarning');
      const warningLevel = level === 'warning' || isSafetyWarningText(text);
      outputElement.textContent = text;
      outputElement.classList.toggle('warning-output', warningLevel);
      if (warning) {
        warning.classList.toggle('hidden', !warningLevel);
        warning.textContent = warningLevel
          ? 'WARNING: a safety guard stopped or limited the model run. Check the details below before relaunching.'
          : '';
      }
    }

    function parseVector(raw) {
      return raw
        .replace(/,/g, ' ')
        .trim()
        .split(/\s+/)
        .map(Number)
        .filter((value) => Number.isFinite(value));
    }

    function parseTrajectory(text) {
      const rows = [];
      const lines = String(text || '').split('\\n');
      for (const line of lines) {
        const stepMatch = line.match(/^step\\s+(\\d+):/);
        const targetMatch = line.match(/target=\\[([^\\]]+)\\]/);
        if (!stepMatch || !targetMatch) continue;
        const target = parseVector(targetMatch[1]);
        if (target.length >= 6) {
          rows.push({step: Number(stepMatch[1]), target: target.slice(0, 6)});
        }
      }
      return rows;
    }

    function drawTrajectory(rows) {
      const canvas = document.getElementById('trajectoryChart');
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#0b0d0f';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      const pad = {left: 56, right: 18, top: 18, bottom: 40};
      const w = canvas.width - pad.left - pad.right;
      const h = canvas.height - pad.top - pad.bottom;
      ctx.strokeStyle = '#2d363d';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.left, pad.top);
      ctx.lineTo(pad.left, pad.top + h);
      ctx.lineTo(pad.left + w, pad.top + h);
      ctx.stroke();
      ctx.fillStyle = '#a6b0b8';
      ctx.font = '13px ui-sans-serif, system-ui';
      ctx.fillText('target qpos [rad]', pad.left, 14);
      ctx.fillText('step', pad.left + w - 28, canvas.height - 12);
      if (!rows.length) {
        ctx.fillText('Run a dry-run test to preview model targets.', pad.left + 10, pad.top + 32);
        return;
      }
      const values = rows.flatMap((row) => row.target);
      let minY = Math.min(...values);
      let maxY = Math.max(...values);
      if (Math.abs(maxY - minY) < 1e-6) {
        minY -= 0.05;
        maxY += 0.05;
      }
      const margin = Math.max(0.03, (maxY - minY) * 0.08);
      minY -= margin;
      maxY += margin;
      const yFor = (value) => pad.top + h - ((value - minY) / (maxY - minY)) * h;
      const xFor = (index) => pad.left + (rows.length === 1 ? 0 : (index / (rows.length - 1)) * w);
      const colors = ['#37c48d', '#e6b450', '#58a6ff', '#ff7b72', '#c297ff', '#7ee787'];
      for (let joint = 0; joint < 6; joint += 1) {
        ctx.strokeStyle = colors[joint];
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        rows.forEach((row, index) => {
          const x = xFor(index);
          const y = yFor(row.target[joint]);
          if (index === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
        rows.forEach((row, index) => {
          const x = xFor(index);
          const y = yFor(row.target[joint]);
          ctx.fillStyle = colors[joint];
          ctx.beginPath();
          ctx.arc(x, y, 3, 0, Math.PI * 2);
          ctx.fill();
        });
      }
      ctx.fillStyle = '#a6b0b8';
      ctx.fillText(maxY.toFixed(2), 8, pad.top + 5);
      ctx.fillText(minY.toFixed(2), 8, pad.top + h);
    }

    function payload(real) {
      return {
        real,
        armed: document.getElementById('armed').checked,
        checkpoint: document.getElementById('checkpoint').value.trim(),
        steps: Number(document.getElementById('steps').value),
        period: Number(document.getElementById('period').value),
        max_speed: Number(document.getElementById('maxSpeed').value),
        max_step_rad: Number(document.getElementById('maxStepRad').value),
        envelope_margin: Number(document.getElementById('envelopeMargin').value),
        collision_action: document.getElementById('collisionAction').value,
        stall_error_rad: Number(document.getElementById('stallErrorRad').value),
        stall_velocity_rad_s: Number(document.getElementById('stallVelocity').value),
        stall_seconds: Number(document.getElementById('stallSeconds').value)
      };
    }

    async function refreshStatus() {
      try {
        const data = await api('/api/status');
        document.getElementById('mode').textContent = data.config.real ? 'REAL' : 'DRY';
        document.getElementById('connected').textContent = data.connected ? 'yes' : 'no';
      } catch (err) {
        output(`ERROR status: ${err.message}`);
      }
    }

    async function connectModelArm() {
      try {
        output('Connecting arm...');
        const data = await api('/api/connect', {});
        output(data.message);
        await refreshStatus();
      } catch (err) {
        output(`ERROR connect: ${err.message}`);
      }
    }

    async function disconnectModelArm() {
      try {
        output('Disconnecting arm...');
        const data = await api('/api/disconnect', {});
        output(data.message);
        await refreshStatus();
      } catch (err) {
        output(`ERROR disconnect: ${err.message}`);
      }
    }

    async function runModel(real) {
      try {
        output(real ? 'Preparing real model test...' : 'Running dry-run model test...');
        if (real) {
          const status = await api('/api/status');
          if (status.connected) {
            output('Disconnecting interface from arm before model test...');
            await api('/api/disconnect', {});
            await refreshStatus();
          }
        }
        output('Running model test...');
        const data = await api('/api/model_test/run', payload(real));
        output(data.output || data.message, isSafetyWarningText(data.output || data.message) ? 'warning' : 'info');
        drawTrajectory(parseTrajectory(data.output || ''));
      } catch (err) {
        const message = `ERROR model test: ${err.message}`;
        output(message, isSafetyWarningText(message) ? 'warning' : 'error');
        drawTrajectory([]);
      }
    }

    async function goToStartPositionModel() {
      try {
        if (!document.getElementById('armed').checked) {
          output('ERROR start position: check enable real model motion first.');
          return;
        }
        output('Moving to saved start position...');
        const data = await api('/api/start_position/go', {armed: true});
        output(data.message);
        await refreshStatus();
      } catch (err) {
        output(`ERROR start position: ${err.message}`);
      }
    }

    async function emergencyStop() {
      try {
        const data = await api('/api/emergency_stop', {});
        output(data.message);
      } catch (err) {
        output(`ERROR emergency: ${err.message}`);
      }
    }

    function openMonitor() {
      window.open('http://127.0.0.1:7865', '_blank');
    }

    refreshStatus();
    drawTrajectory([]);
    setInterval(refreshStatus, 2000);
  </script>
</body>
</html>
"""


TEACH_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WidowX AI Teaching</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101214;
      --panel: #181c20;
      --panel-2: #20262b;
      --text: #eef2f4;
      --muted: #a6b0b8;
      --accent: #37c48d;
      --danger: #e45858;
      --warning: #e6b450;
      --line: #2d363d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    main {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 36px;
    }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 18px;
    }
    h1 { margin: 0; font-size: 30px; line-height: 1.05; font-weight: 720; }
    a { color: var(--accent); text-decoration: none; }
    .sub { margin-top: 6px; color: var(--muted); font-size: 14px; }
    .layout {
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 18px;
      align-items: start;
    }
    section, aside {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }
    .toolbar {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 18px;
    }
    .toolbar-block {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      padding: 12px;
      background: #111518;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .toolbar-title {
      min-width: 90px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    button {
      height: 40px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--panel-2);
      color: var(--text);
      padding: 0 14px;
      font-size: 14px;
      cursor: pointer;
    }
    button:hover { border-color: #52616b; }
    button.primary { background: #1f5f49; border-color: #2b8c67; }
    button.danger { background: #642828; border-color: #9b3d3d; }
    button.emergency {
      width: 100%;
      height: 64px;
      background: #a51616;
      border-color: #ff4d4d;
      color: white;
      font-size: 20px;
      font-weight: 800;
      text-transform: uppercase;
      margin-bottom: 14px;
    }
    .armed {
      display: flex;
      align-items: center;
      gap: 10px;
      min-height: 40px;
      padding: 0 12px;
      background: #171a1d;
      border: 1px solid var(--line);
      border-radius: 7px;
      color: var(--muted);
      font-size: 14px;
    }
    input[type="checkbox"] { width: 18px; height: 18px; }
    input[type="number"], input[type="text"], select {
      width: 100%;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #111518;
      color: var(--text);
      padding: 0 10px;
      font-size: 14px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .field label {
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 6px;
    }
    .camera-view {
      position: relative;
      width: 100%;
      aspect-ratio: 4 / 3;
      background: #0b0d0f;
      border: 1px solid var(--line);
      border-radius: 7px;
      overflow: hidden;
      display: grid;
      place-items: center;
      color: var(--muted);
      font-size: 14px;
    }
    .camera-view img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: none;
    }
    .side-title {
      margin: 0 0 12px;
      font-size: 15px;
      color: var(--muted);
      font-weight: 650;
      text-transform: uppercase;
    }
    .kv {
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 8px 12px;
      margin-bottom: 18px;
      font-size: 14px;
    }
    .kv div:nth-child(odd) { color: var(--muted); }
    pre {
      min-height: 260px;
      max-height: 460px;
      overflow: auto;
      margin: 0;
      white-space: pre-wrap;
      background: #0b0d0f;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 12px;
      color: #d8dee3;
      font-size: 12px;
      line-height: 1.45;
    }
    .note {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
      margin-bottom: 16px;
    }
    .teaching-shell {
      display: grid;
      gap: 16px;
    }
    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(300px, 0.8fr);
      gap: 14px;
      align-items: stretch;
    }
    .hero-card, .control-card, .review-shell {
      background: #111518;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 16px;
    }
    .hero-card h2, .control-card h3, .review-shell h3 {
      margin: 0;
    }
    .hero-title {
      font-size: 22px;
      line-height: 1.15;
      margin-bottom: 8px;
    }
    .hero-copy {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.55;
      margin-bottom: 14px;
    }
    .hero-points {
      display: grid;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }
    .hero-points strong {
      color: var(--text);
      font-weight: 650;
    }
    .hero-actions {
      display: grid;
      gap: 10px;
      align-content: start;
    }
    .emergency-inline {
      height: 52px;
      font-size: 17px;
      margin-bottom: 0;
    }
    .control-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    .step-card {
      background: #151a1e;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    .step-card.review {
      grid-column: 1 / -1;
    }
    .step-card.camera-capture-card {
      grid-column: 1 / -1;
    }
    .step-kicker {
      color: var(--accent);
      font-size: 12px;
      font-weight: 750;
      letter-spacing: 0;
      text-transform: uppercase;
      margin-bottom: 6px;
    }
    .step-card h3 {
      margin: 0 0 14px;
      font-size: 18px;
      line-height: 1.2;
    }
    .card-subtitle {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      margin: 6px 0 14px;
    }
    .action-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 14px;
    }
    .action-row button {
      min-width: 120px;
    }
    .capture-row {
      display: grid;
      grid-template-columns: 88px 88px minmax(160px, 0.9fr);
      gap: 10px;
      margin-bottom: 12px;
    }
    .capture-row button {
      min-width: 0;
      width: 100%;
      padding: 0 10px;
    }
    .return-start-button {
      max-width: 180px;
      justify-self: start;
    }
    .secondary-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .secondary-row button {
      height: 34px;
      padding: 0 10px;
      font-size: 13px;
    }
    .settings-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
    }
    .crop-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin: 12px 0;
    }
    .crop-grid .wide {
      grid-column: 1 / -1;
    }
    .capture-settings-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(320px, 0.9fr);
      gap: 14px;
      align-items: start;
      margin-top: 12px;
    }
    .capture-panel {
      background: #101417;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }
    .capture-panel h4 {
      margin: 0 0 12px;
      font-size: 14px;
      line-height: 1.2;
    }
    .capture-panel.hidden {
      display: none;
    }
    .preview-panel.selectable {
      cursor: pointer;
    }
    .preview-panel.selectable.active {
      border-color: var(--accent);
    }
    .capture-summary {
      display: grid;
      gap: 8px;
      margin-top: 12px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #0d1114;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }
    .capture-summary strong {
      color: var(--text);
      font-weight: 650;
    }
    .check-field {
      display: flex;
      align-items: center;
      gap: 10px;
      min-height: 38px;
      padding: 0 10px;
      background: #101417;
      border: 1px solid var(--line);
      border-radius: 7px;
      color: var(--muted);
      font-size: 13px;
    }
    .mode-summary {
      display: grid;
      gap: 8px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }
    .mode-summary strong {
      color: var(--text);
      font-weight: 650;
    }
    .preview-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 0;
    }
    .preview-panel {
      background: #151a1e;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }
    .preview-panel.compact {
      padding: 12px;
    }
    .preview-title {
      margin: 0 0 10px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .camera-control-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto auto;
      gap: 10px;
      margin-bottom: 12px;
    }
    .quick-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }
    .quick-actions button {
      flex: 1 1 160px;
    }
    .review-shell {
      margin-top: 0;
    }
    .review-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto auto;
      gap: 10px;
      align-items: end;
      margin-bottom: 12px;
    }
    .utility-note {
      margin-top: 12px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 34px;
      padding: 0 12px;
      background: #101417;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }
    .status-pill.live {
      color: #ffd8d8;
      border-color: #8a3a3a;
      background: #261414;
    }
    .status-led {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #5c6770;
      flex: 0 0 auto;
    }
    .status-pill.live .status-led {
      background: #ff5f5f;
      box-shadow: 0 0 0 4px rgba(255, 95, 95, 0.18);
    }
    .status-pill.ok {
      color: #d7f7ea;
      border-color: #2b8c67;
      background: #112019;
    }
    .status-pill.ok .status-led {
      background: var(--accent);
      box-shadow: 0 0 0 4px rgba(55, 196, 141, 0.18);
    }
    .status-pill.warn .status-led {
      background: var(--warning);
      box-shadow: 0 0 0 4px rgba(230, 180, 80, 0.18);
    }
    .inline-preview {
      margin-top: 14px;
    }
    .review-meta {
      min-height: 42px;
      margin-top: 10px;
      margin-bottom: 0;
      overflow-wrap: anywhere;
    }
    .review {
      margin-top: 0;
    }
    .review-controls {
      display: grid;
      grid-template-columns: 1fr repeat(6, auto);
      gap: 10px;
      margin-bottom: 12px;
    }
    .frame-controls {
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 10px;
      align-items: center;
      margin-top: 12px;
    }
    .video-controls {
      display: grid;
      grid-template-columns: auto auto 1fr;
      gap: 10px;
      align-items: center;
      margin-top: 12px;
    }
    .playback-state {
      color: var(--muted);
      font-size: 13px;
      text-align: right;
      overflow-wrap: anywhere;
    }
    input[type="range"] { width: 100%; }
    @media (max-width: 880px) {
      header { align-items: stretch; flex-direction: column; }
      .layout { grid-template-columns: 1fr; }
      .grid { grid-template-columns: 1fr; }
      .toolbar, .hero, .control-grid, .preview-grid, .settings-grid, .camera-control-row, .review-head, .capture-settings-grid { grid-template-columns: 1fr; }
      .capture-row { grid-template-columns: 1fr; }
      .review-controls { grid-template-columns: 1fr; }
      .frame-controls { grid-template-columns: 1fr; }
      .video-controls { grid-template-columns: 1fr; }
      .playback-state { text-align: left; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Teach</h1>
        <div class="sub">Collecte simple de demonstrations WidowX avec top cam + D405 RGB/depth</div>
      </div>
      <a href="/">Back to control</a>
    </header>
    <div class="layout">
      <section>
        <div class="toolbar">
          <div class="toolbar-block">
            <span class="toolbar-title">Session</span>
            <button class="primary" onclick="connectArm()">Connect</button>
            <button class="danger" onclick="disconnectArm()">Disconnect</button>
            <label class="armed"><input type="checkbox" id="armed"> enable motion</label>
            <div class="status-pill warn" id="teachConnectionPill">
              <span class="status-led"></span>
              <span id="teachConnectionText">Disconnected</span>
            </div>
          </div>
          <div class="toolbar-block">
            <span class="toolbar-title">Quick tools</span>
            <button class="danger emergency-inline" onclick="emergencyStop()">Emergency stop</button>
            <button onclick="gravityCompensation()">Gravity comp</button>
            <button id="holdButton" onclick="toggleHold()">Hold</button>
            <button onclick="gripper(10)">Open gripper</button>
            <button onclick="gripper(-10)">Close gripper</button>
          </div>
        </div>
        <div class="teaching-shell">
          <input id="cameraMode" type="hidden" value="color">
          <input id="payloadProfile" type="hidden" value="d405_follower">
          <input id="wristEffortSlider" type="hidden" value="0">
          <span id="wristEffortValue" hidden>0.00</span>

          <div class="control-grid">
            <div class="control-card">
              <div class="step-kicker">Prepare</div>
              <h3>Start position</h3>
              <div class="card-subtitle">Sauvegarde une pose de depart stable, puis ramene toujours le bras ici avant une nouvelle demo.</div>
              <div class="quick-actions">
                <button onclick="saveStartPosition()">Save current as start</button>
                <button class="primary" onclick="goToStartPosition()">Go to start position</button>
              </div>
            </div>

            <div class="step-card">
              <div class="step-kicker">Step 1</div>
              <h3>Record source motion</h3>
              <div class="card-subtitle">Passe en gravity compensation, guide le bras a la main, puis sauvegarde une demo source sans video.</div>
              <div class="field">
                <label>Source movement name</label>
                <input id="sourceSessionName" type="text" placeholder="push_cube_source_01">
              </div>
              <div class="action-row">
                <button class="primary" onclick="startRecording()">Start source record</button>
                <button class="danger" onclick="stopRecording()">Stop record</button>
              </div>
              <div class="mode-summary">
                <div><strong>Source mode:</strong> gravity compensation, motor state at 100 Hz, no camera.</div>
                <div><strong>Output:</strong> a replayable source movement for dataset capture.</div>
              </div>
            </div>

            <div class="step-card camera-capture-card">
              <div class="step-kicker">Camera</div>
              <h3>Camera capture</h3>
              <div class="card-subtitle">Regle les sources video et le crop avant de lancer la capture dataset. La D405 RGB et depth sont enregistrees ensemble.</div>
              <div class="capture-settings-grid">
                <div class="capture-panel">
                  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <h3 class="preview-title" style="margin-bottom: 0;">Live previews</h3>
                    <div style="display: flex; gap: 8px;">
                      <button style="height: 32px;" onclick="refreshTeachCameras(true)">Refresh</button>
                      <button class="primary" style="height: 32px;" onclick="startCamera()">Start preview</button>
                    </div>
                  </div>
                  <div class="preview-panel inline-preview" style="border: none; padding: 0; background: transparent; margin-top: 0;">
                    <div class="preview-grid">
                      <div class="preview-panel compact selectable" id="previewPanelTop" onclick="selectCropRole('top_view')">
                        <h3 class="preview-title">Top camera</h3>
                        <div class="camera-view">
                          <img id="cameraImageTop" alt="Top camera live preview" onload="recordFrame('top_view')" onerror="clearPendingFrame('top_view')">
                          <span class="fps-counter" id="fpsTop" style="position: absolute; top: 4px; right: 4px; background: rgba(0,0,0,0.5); padding: 2px 4px; border-radius: 4px; font-size: 10px;">0 FPS</span>
                          <span id="cameraPlaceholderTop">Camera inactive</span>
                        </div>
                      </div>
                      <div class="preview-panel compact selectable" id="previewPanelWristRgb" onclick="selectCropRole('wrist_rgb')">
                        <h3 class="preview-title">D405 RGB</h3>
                        <div class="camera-view">
                          <img id="cameraImageD405Rgb" alt="D405 RGB live preview" onload="recordFrame('wrist_rgb')" onerror="clearPendingFrame('wrist_rgb')">
                          <span class="fps-counter" id="fpsWristRgb" style="position: absolute; top: 4px; right: 4px; background: rgba(0,0,0,0.5); padding: 2px 4px; border-radius: 4px; font-size: 10px;">0 FPS</span>
                          <span id="cameraPlaceholderD405Rgb">Camera inactive</span>
                        </div>
                      </div>
                      <div class="preview-panel compact selectable" id="previewPanelWristDepth" onclick="selectCropRole('wrist_depth')">
                        <h3 class="preview-title">D405 depth</h3>
                        <div class="camera-view">
                          <img id="cameraImageD405Depth" alt="D405 depth live preview" onload="recordFrame('wrist_depth')" onerror="clearPendingFrame('wrist_depth')">
                          <span class="fps-counter" id="fpsWristDepth" style="position: absolute; top: 4px; right: 4px; background: rgba(0,0,0,0.5); padding: 2px 4px; border-radius: 4px; font-size: 10px;">0 FPS</span>
                          <span id="cameraPlaceholderD405Depth">Camera inactive</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                <div class="capture-panel hidden" id="cropOutputPanel">
                  <h4>Camera Settings</h4>
                  <div class="field">
                    <label><input id="cameraCaptureEnabled" type="checkbox" checked onchange="updateCameraCaptureEnabled()"> Enable recording for this camera</label>
                  </div>
                  <div class="field" id="resolutionSelectorField">
                    <label>Resolution</label>
                    <select id="cameraResolution" onchange="updateCameraResolution()"></select>
                  </div>
                  <div class="field" id="topCameraSelectorField" style="display: none;">
                    <label>Top camera source</label>
                    <select id="teachCameraSource" onchange="updateCaptureSummary(); updateCameraFrame();"></select>
                  </div>
                  <div class="field">
                    <label><input id="datasetCropEnabled" type="checkbox" onchange="updateCaptureSummary(); updateCameraFrame();"> Crop video flux</label>
                    <div class="crop-grid">
                      <div>
                        <label>Apply crop to</label>
                        <select id="datasetCropTarget" onchange="updateCaptureSummary(); updateCameraFrame();">
                          <option value="all" selected>All cameras</option>
                          <option value="top">Top camera</option>
                          <option value="d405">D405 RGB/depth</option>
                        </select>
                      </div>
                      <div>
                        <label>Ratio</label>
                        <select id="datasetCropAspect" onchange="updateCaptureSummary(); updateCameraFrame();">
                          <option value="source" selected>Keep source</option>
                          <option value="1:1">1:1 square</option>
                          <option value="4:3">4:3</option>
                          <option value="16:9">16:9</option>
                          <option value="3:2">3:2</option>
                          <option value="9:16">9:16 vertical</option>
                        </select>
                      </div>
                      <div>
                        <label>Zoom <span id="datasetCropZoomValue">1.00</span>x</label>
                        <input id="datasetCropZoom" type="range" min="1" max="4" step="0.05" value="1" oninput="document.getElementById('datasetCropZoomValue').textContent = Number(this.value).toFixed(2); updateCaptureSummary(); updateCameraFrame();">
                      </div>
                      <div>
                        <label>Offset X <span id="datasetCropXValue">0.00</span></label>
                        <input id="datasetCropX" type="range" min="-1" max="1" step="0.05" value="0" oninput="document.getElementById('datasetCropXValue').textContent = Number(this.value).toFixed(2); updateCaptureSummary(); updateCameraFrame();">
                      </div>
                      <div class="wide">
                        <label>Offset Y <span id="datasetCropYValue">0.00</span></label>
                        <input id="datasetCropY" type="range" min="-1" max="1" step="0.05" value="0" oninput="document.getElementById('datasetCropYValue').textContent = Number(this.value).toFixed(2); updateCaptureSummary(); updateCameraFrame();">
                      </div>
                    </div>
                  </div>
                  <div class="capture-summary" id="captureParamPreview"></div>
                </div>
              </div>
            </div>

            <div class="step-card review">
              <div class="step-kicker">Step 2</div>
              <h3>Replay and capture</h3>
              <div class="card-subtitle">Choisis une demo source, rejoue-la, puis capture simultanement la top cam et la D405 en RGB ou depth.</div>
              <div class="status-pill" id="datasetCaptureIndicator">
                <span class="status-led"></span>
                <span id="datasetCaptureIndicatorText">Capture idle</span>
              </div>
              <div class="note">
                Cette liste ne montre que les demos source rejouables, pas les datasets deja captures.
              </div>
              <div class="capture-row">
                <button onclick="previousSourceRecording()">Previous</button>
                <button onclick="nextSourceRecording()">Next</button>
                <button class="return-start-button" onclick="goToStartPosition()">Return to start</button>
              </div>
              <div class="field">
                <label>Recorded demo to replay</label>
                <select id="recordingSelect" onchange="loadSelectedRecording(); updateCaptureSummary();"></select>
              </div>
              <div class="field">
                <label>Task label</label>
                <input id="datasetTaskName" type="text" value="push_cube_5cm" placeholder="push_cube_5cm" oninput="updateCaptureSummary()">
              </div>
              <div class="field">
                <label>Replay speed <span id="replaySpeedValue">0.75</span>x</label>
                <input id="replaySpeed" type="range" min="0.25" max="1" step="0.05" value="0.75" oninput="document.getElementById('replaySpeedValue').textContent = Number(this.value).toFixed(2); updateCaptureSummary();">
              </div>
              <div class="action-row">
                <button onclick="startReplay()">Replay movement</button>
                <button class="primary" onclick="startDatasetCapture()">Replay movement + record camera</button>
              </div>
              <div class="mode-summary">
                <div><strong>Replay movement:</strong> robot motion only.</div>
                <div><strong>Replay movement + record camera:</strong> D405 RGB/depth + selected top camera at 30 Hz synchronized with motor samples at 100 Hz.</div>
              </div>
              <div class="secondary-row">
                <button class="danger" onclick="deleteRecording()">Delete selected</button>
                <button class="danger" onclick="clearRecordings()">Delete all</button>
              </div>
            </div>
          </div>

          <div class="review-shell">
            <h3>Review</h3>
            <div class="card-subtitle">Controle visuellement la capture avant de passer a la demo suivante.</div>
            <div class="review-head">
              <div class="field">
                <label>Recording to review</label>
                <select id="reviewRecordingSelect" onchange="loadReviewRecording()"></select>
              </div>
              <button onclick="previousRecording()">Previous</button>
              <button onclick="nextRecording()">Next</button>
            </div>
            <div class="preview-panel compact">
              <h3 class="preview-title">Selected recording</h3>
              <div class="preview-grid">
                <div class="preview-panel">
                  <h3 class="preview-title">Top camera</h3>
                  <div class="camera-view">
                    <img id="reviewImageTop" alt="Recorded top camera preview">
                    <span id="reviewPlaceholderTop">No recording selected</span>
                  </div>
                </div>
                <div class="preview-panel">
                  <h3 class="preview-title">Wrist RGB</h3>
                  <div class="camera-view">
                    <img id="reviewImageWristRgb" alt="Recorded wrist RGB preview">
                    <span id="reviewPlaceholderWristRgb">No recording selected</span>
                  </div>
                </div>
                <div class="preview-panel">
                  <h3 class="preview-title">Wrist depth</h3>
                  <div class="camera-view">
                    <img id="reviewImageWristDepth" alt="Recorded wrist depth preview">
                    <span id="reviewPlaceholderWristDepth">No recording selected</span>
                  </div>
                </div>
              </div>
              <div class="video-controls">
                <button id="reviewPlayButton" onclick="toggleReviewPlayback()">Play</button>
                <button onclick="stopReviewPlayback(true)">Stop</button>
                <div class="playback-state" id="reviewPlaybackState">0:00 / 0:00</div>
              </div>
              <div class="frame-controls">
                <button onclick="stepFrame(-1)">Frame -</button>
                <input id="frameSlider" type="range" min="0" max="0" step="1" value="0" oninput="showFrame(Number(this.value))">
                <button onclick="stepFrame(1)">Frame +</button>
              </div>
              <div class="note review-meta" id="reviewMeta"></div>
            </div>
          </div>
        </div>
      </section>
      <aside>
        <h2 class="side-title">Status</h2>
        <div class="kv">
          <div>Camera</div><div id="cameraState">-</div>
          <div>Start pos</div><div id="startPoseState">-</div>
          <div>Record</div><div id="recordState">-</div>
          <div>Replay</div><div id="replayState">-</div>
          <div>Dataset</div><div id="datasetState">-</div>
          <div>Samples</div><div id="samples">0</div>
          <div>Folder</div><div id="sessionDir">-</div>
        </div>
        <h2 class="side-title">Log</h2>
        <pre id="log"></pre>
      </aside>
    </div>
  </main>
  <script>
    let cameraTimer = null;
    let cameraRunning = false;
    let teachCameraSources = [];
    let activeTeachCameraSource = '';
    let recordings = [];
    let reviewRecordings = [];
    let selectedRecording = null;
    let reviewFrames = [];
    let reviewIndex = 0;
    let reviewPlaying = false;
    let reviewPlaybackTimer = null;
    let lastDatasetRunning = false;
    let activeCropRole = 'all';
    let cameraCaptureState = { top_view: true, wrist_rgb: true, wrist_depth: true };
    let fpsCounters = { top_view: { count: 0, lastTime: Date.now(), fps: 0 }, wrist_rgb: { count: 0, lastTime: Date.now(), fps: 0 }, wrist_depth: { count: 0, lastTime: Date.now(), fps: 0 } };
    let pendingFrames = { top_view: false, wrist_rgb: false, wrist_depth: false };
    let resolutionState = { top_view: "640x480", d405: "640x480" };

    function populateResolutions() {
      const resSelect = document.getElementById('cameraResolution');
      if (!resSelect || !activeCropRole) return;
      const isD405 = activeCropRole === 'wrist_rgb' || activeCropRole === 'wrist_depth';
      const options = isD405
        ? ["640x480", "848x480", "1280x720"]
        : ["640x480", "800x600", "1280x720", "1920x1080"];
      resSelect.innerHTML = '';
      options.forEach(opt => {
        const option = document.createElement('option');
        option.value = opt;
        option.textContent = opt;
        resSelect.appendChild(option);
      });
      resSelect.value = resolutionState[isD405 ? 'd405' : 'top_view'];
    }

    function updateCameraResolution() {
      if (!activeCropRole) return;
      const isD405 = activeCropRole === 'wrist_rgb' || activeCropRole === 'wrist_depth';
      resolutionState[isD405 ? 'd405' : 'top_view'] = document.getElementById('cameraResolution').value;
      updateCaptureSummary();
    }

    function clearPendingFrame(role) {
      pendingFrames[role] = false;
    }

    function recordFrame(role) {
      pendingFrames[role] = false;
      const now = Date.now();
      const counter = fpsCounters[role];
      counter.count++;
      if (now - counter.lastTime >= 1000) {
        counter.fps = counter.count;
        counter.count = 0;
        counter.lastTime = now;
        const elId = role === 'top_view' ? 'fpsTop' : (role === 'wrist_rgb' ? 'fpsWristRgb' : 'fpsWristDepth');
        const el = document.getElementById(elId);
        if (el) el.textContent = `${counter.fps} FPS`;
      }
    }

    function updateCameraCaptureEnabled() {
      if (activeCropRole) {
        cameraCaptureState[activeCropRole] = document.getElementById('cameraCaptureEnabled').checked;
        updateCaptureSummary();
        updateCameraLoop();
      }
    }

    function log(msg) {
      const el = document.getElementById('log');
      const stamp = new Date().toLocaleTimeString();
      el.textContent = `[${stamp}] ${msg}\\n` + el.textContent;
    }

    async function api(path, payload = null) {
      const opts = payload ? {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      } : {};
      const res = await fetch(path, opts);
      const data = await res.json();
      if (!res.ok || data.ok === false) throw new Error(data.error || res.statusText);
      return data;
    }

    async function refreshAll(writeLog = false) {
      try {
        const arm = await api('/api/status');
        const pill = document.getElementById('teachConnectionPill');
        document.getElementById('teachConnectionText').textContent = arm.connected ? 'Connected' : 'Disconnected';
        pill.className = 'status-pill ' + (arm.connected ? 'ok' : 'warn');
        document.getElementById('startPoseState').textContent = arm.start_position_saved
          ? `Saved · ${arm.start_position_label || 'ready'}`
          : 'Not saved';
        document.getElementById('payloadProfile').value = 'd405_follower';
        document.getElementById('wristEffortSlider').value = 0;
        document.getElementById('wristEffortValue').textContent = '0.00';
        const cam = await api('/api/video/status');
        cameraRunning = cam.running;
        renderTeachCameraSources(cam.sources || [], cam.active_source);
        activeTeachCameraSource = cam.active_source || selectedTeachCameraSource();
        document.getElementById('cameraState').textContent = cam.sources.length
          ? (cam.running ? `Active · ${cam.active_label}` : `${cam.sources.length} detected`)
          : 'Not detected';
        document.getElementById('cameraMode').value = 'color';
        const trossenUi = await api('/api/trossen_ui/status');
        const trossenUiState = document.getElementById('trossenUiState');
        if (trossenUiState) {
          trossenUiState.textContent = trossenUi.running
            ? `Running PID ${trossenUi.pid}`
            : (trossenUi.available ? 'Ready' : 'Not installed');
        }
        const rec = await api('/api/teach/status');
        document.getElementById('recordState').textContent = rec.running ? 'Recording' : 'Stop';
        document.getElementById('samples').textContent = rec.mode === 'high_smooth'
          ? `${rec.motor_samples} motor / ${rec.samples} img`
          : rec.samples;
        document.getElementById('sessionDir').textContent = rec.session_dir || '-';
        const replay = await api('/api/replay/status');
        document.getElementById('replayState').textContent = replay.running
          ? `${replay.frame_index}/${replay.frame_count}`
          : 'Stop';
        const dataset = await api('/api/dataset_capture/status');
        document.getElementById('datasetState').textContent = dataset.running ? 'Capturing' : 'Idle';
        const datasetIndicator = document.getElementById('datasetCaptureIndicator');
        const datasetIndicatorText = document.getElementById('datasetCaptureIndicatorText');
        if (datasetIndicator && datasetIndicatorText) {
          datasetIndicator.classList.toggle('live', Boolean(dataset.running));
          datasetIndicatorText.textContent = dataset.running ? 'Capture camera active' : 'Capture idle';
        }
        if (lastDatasetRunning && !dataset.running) {
          log(dataset.message);
          await loadRecordings(null, dataset.source_path, dataset.session_dir);
        }
        lastDatasetRunning = dataset.running;
        document.getElementById('holdButton').textContent = arm.hold ? 'Hold off' : 'Hold';
        updateCaptureSummary();
        updateCameraLoop();
        if (writeLog) log('Status updated');
      } catch (err) {
        log(`ERROR: ${err.message}`);
      }
    }

    async function emergencyStop() {
      try {
        const data = await api('/api/emergency_stop', {});
        log(data.message);
        await refreshAll(false);
      } catch (err) {
        log(`ERROR emergency: ${err.message}`);
      }
    }

    function updateCameraLoop() {
      const views = livePreviewViews();
      if (cameraRunning && selectedTeachCameraSource()) {
        views.forEach((view) => {
          const enabled = cameraCaptureState[view.role] !== false;
          view.img.style.display = enabled ? 'block' : 'none';
          view.placeholder.style.display = enabled ? 'none' : 'block';
          if (!enabled) view.placeholder.textContent = 'Disabled';
        });
        if (!cameraTimer) cameraTimer = setInterval(updateCameraFrame, 33);
        updateCameraFrame();
      } else {
        views.forEach((view) => {
          view.img.style.display = 'none';
          view.placeholder.style.display = 'block';
          view.placeholder.textContent = selectedTeachCameraSource() ? 'Camera inactive' : 'No camera selected';
        });
        if (cameraTimer) {
          clearInterval(cameraTimer);
          cameraTimer = null;
        }
      }
    }

    function preferredTopCameraSource() {
      const usb0 = teachCameraSources.find((source) => source.id === 'usb:0');
      if (usb0) return usb0.id;
      const brio = teachCameraSources.find((source) => String(source.label || '').toLowerCase().includes('brio'));
      if (brio) return brio.id;
      return teachCameraSources.length ? teachCameraSources[0].id : '';
    }

    function teachCameraOptionLabel(source) {
      if (source.id === 'usb:0') {
        return `${source.label} · top cam par defaut`;
      }
      return source.label;
    }

    function renderTeachCameraSources(sources, activeSource) {
      const select = document.getElementById('teachCameraSource');
      const previous = select.value;
      teachCameraSources = Array.isArray(sources) ? sources : [];
      select.innerHTML = '';
      if (teachCameraSources.length === 0) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'No camera detected';
        select.appendChild(option);
        select.disabled = true;
        activeTeachCameraSource = '';
        updateCaptureSummary();
        return;
      }
      select.disabled = false;
      teachCameraSources.forEach((source) => {
        const option = document.createElement('option');
        option.value = source.id;
        option.textContent = teachCameraOptionLabel(source);
        select.appendChild(option);
      });
      const preferred = preferredTopCameraSource() || previous || activeSource;
      const exists = teachCameraSources.some((source) => source.id === preferred);
      select.value = exists ? preferred : preferredTopCameraSource();
      activeTeachCameraSource = select.value;
      updateCaptureSummary();
    }

    function selectedTeachCameraSource() {
      const value = document.getElementById('teachCameraSource').value;
      return value || '';
    }

    function cropRoleLabel(role) {
      if (role === 'all') return 'All cameras';
      if (role === 'top_view') return 'Top camera';
      if (role === 'wrist_rgb') return 'D405 RGB';
      if (role === 'wrist_depth') return 'D405 depth';
      return 'Click a preview';
    }

    function selectCropRole(role) {
      activeCropRole = role;
      const panel = document.getElementById('cropOutputPanel');
      if (panel) panel.classList.remove('hidden');

      const topCamField = document.getElementById('topCameraSelectorField');
      if (topCamField) {
        topCamField.style.display = (role === 'top_view') ? 'block' : 'none';
      }

      populateResolutions();

      const captureEnabledCheckbox = document.getElementById('cameraCaptureEnabled');
      if (captureEnabledCheckbox) {
        captureEnabledCheckbox.checked = cameraCaptureState[role] !== false;
      }

      const enabled = document.getElementById('datasetCropEnabled');
      if (enabled) enabled.checked = true;
      ['top_view', 'wrist_rgb', 'wrist_depth'].forEach((candidate) => {
        const elementId = candidate === 'top_view'
          ? 'previewPanelTop'
          : candidate === 'wrist_rgb'
          ? 'previewPanelWristRgb'
          : 'previewPanelWristDepth';
        const element = document.getElementById(elementId);
        if (element) element.classList.toggle('active', candidate === role);
      });
      updateCaptureSummary();
      updateCameraFrame();
    }

    function livePreviewViews() {
      return [
        {
          source: selectedTeachCameraSource(),
          role: 'top_view',
          img: document.getElementById('cameraImageTop'),
          placeholder: document.getElementById('cameraPlaceholderTop')
        },
        {
          source: 'd405:color',
          role: 'wrist_rgb',
          img: document.getElementById('cameraImageD405Rgb'),
          placeholder: document.getElementById('cameraPlaceholderD405Rgb')
        },
        {
          source: 'd405:depth',
          role: 'wrist_depth',
          img: document.getElementById('cameraImageD405Depth'),
          placeholder: document.getElementById('cameraPlaceholderD405Depth')
        }
      ];
    }

    function cropQueryString(crop) {
      if (!crop) return '';
      const params = new URLSearchParams({
        crop_enabled: '1',
        crop_aspect: crop.aspect,
        crop_zoom: String(crop.zoom),
        crop_x: String(crop.offset_x),
        crop_y: String(crop.offset_y)
      });
      return `&${params.toString()}`;
    }

    function selectedTeachCameraLabel() {
      const sourceId = selectedTeachCameraSource();
      const source = teachCameraSources.find((item) => item.id === sourceId);
      return source ? source.label : (sourceId || 'None');
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, (char) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
      }[char]));
    }

    function updateCaptureSummary() {
      const preview = document.getElementById('captureParamPreview');
      if (!preview) return;
      const cropEnabled = Boolean(document.getElementById('datasetCropEnabled')?.checked);
      const cropTarget = document.getElementById('datasetCropTarget')?.value || 'all';
      const cropAspect = document.getElementById('datasetCropAspect')?.value || 'source';
      const cropZoom = Number(document.getElementById('datasetCropZoom')?.value || 1).toFixed(2);
      const cropX = Number(document.getElementById('datasetCropX')?.value || 0).toFixed(2);
      const cropY = Number(document.getElementById('datasetCropY')?.value || 0).toFixed(2);
      const replaySpeed = Number(document.getElementById('replaySpeed')?.value || 0.75).toFixed(2);
      const taskName = document.getElementById('datasetTaskName')?.value.trim() || 'no task label';
      const sourceName = selectedTeachCameraLabel();
      const cropText = cropEnabled
        ? `${cropTarget}, ratio ${cropAspect}, zoom ${cropZoom}x, x ${cropX}, y ${cropY}`
        : 'disabled';
      preview.innerHTML = `
        <div><strong>Top camera:</strong> ${cameraCaptureState.top_view ? escapeHtml(sourceName) + ' (' + resolutionState.top_view + ')' : 'disabled'}</div>
        <div><strong>D405:</strong> ${cameraCaptureState.wrist_rgb ? 'RGB ' : ''}${cameraCaptureState.wrist_depth ? 'Depth' : ''}${!cameraCaptureState.wrist_rgb && !cameraCaptureState.wrist_depth ? 'disabled' : ''} (${resolutionState.d405})</div>
        <div><strong>Selected crop preview:</strong> ${escapeHtml(cropRoleLabel(activeCropRole))}</div>
        <div><strong>Crop:</strong> ${escapeHtml(cropText)}</div>
        <div><strong>Replay:</strong> ${replaySpeed}x · task ${escapeHtml(taskName)}</div>
        <div><strong>Output:</strong> ${Object.keys(cameraCaptureState).filter(k => cameraCaptureState[k]).join(' + ')} at 30 Hz, motor at 100 Hz</div>
      `;
    }

    async function refreshTeachCameras(writeLog = false) {
      try {
        const data = await api('/api/video/status');
        cameraRunning = data.running;
        renderTeachCameraSources(data.sources || [], data.active_source);
        activeTeachCameraSource = data.active_source || selectedTeachCameraSource();
        document.getElementById('cameraState').textContent = data.sources.length
          ? (data.running ? `Active · ${data.active_label}` : `${data.sources.length} detected`)
          : 'Not detected';
        if (writeLog) log(data.message);
        updateCaptureSummary();
        updateCameraLoop();
      } catch (err) {
        log(`ERROR camera: ${err.message}`);
      }
    }

    function updateCameraFrame() {
      if (!cameraRunning) return;
      livePreviewViews().forEach((view) => {
        if (!view.source || !view.img) return;
        if (cameraCaptureState[view.role] === false) return;
        if (pendingFrames[view.role]) return;
        
        pendingFrames[view.role] = true;
        const cropQuery = cropQueryString(datasetCropConfig(view.role));
        view.img.src = `/api/video/frame?source=${encodeURIComponent(view.source)}${cropQuery}&t=${Date.now()}`;
      });
    }

    async function connectArm() {
      try {
        const data = await api('/api/connect', {});
        log(data.message);
        await refreshAll(false);
      } catch (err) {
        log(`ERROR: ${err.message}`);
      }
    }

    async function disconnectArm() {
      try {
        const data = await api('/api/disconnect', {});
        log(data.message);
        await refreshAll(false);
      } catch (err) {
        log(`ERROR: ${err.message}`);
      }
    }

    async function gravityCompensation() {
      try {
        const data = await api('/api/gravity_compensation', gravityPayload());
        log(data.message);
        await refreshAll(false);
      } catch (err) {
        log(`ERROR: ${err.message}`);
      }
    }

    async function saveStartPosition() {
      try {
        const data = await api('/api/start_position/save', {});
        log(data.message);
        await refreshAll(false);
      } catch (err) {
        log(`ERROR start position: ${err.message}`);
      }
    }

    async function goToStartPosition() {
      try {
        const replay = await api('/api/replay/status');
        if (replay.running) {
          const stop = await api('/api/replay/stop', {});
          log(stop.message);
          await new Promise((resolve) => setTimeout(resolve, 250));
        }
        const data = await api('/api/start_position/go', {armed: document.getElementById('armed').checked});
        log(data.message);
        await refreshAll(false);
      } catch (err) {
        log(`ERROR start position: ${err.message}`);
      }
    }

    async function toggleHold() {
      try {
        const payload = gravityPayload();
        payload.enabled = document.getElementById('holdButton').textContent === 'Hold';
        const data = await api('/api/hold', payload);
        log(data.message);
        await refreshAll(false);
      } catch (err) {
        log(`ERROR: ${err.message}`);
      }
    }

    async function gripper(effort) {
      try {
        const data = await api('/api/gripper', {effort, armed: document.getElementById('armed').checked});
        log(data.message);
        await refreshAll(false);
      } catch (err) {
        log(`ERROR: ${err.message}`);
      }
    }

    function gravityPayload() {
      return {
        armed: document.getElementById('armed').checked,
        payload_profile: 'd405_follower',
        camera_wrist_effort: 0
      };
    }

    async function refreshTrossenUI(writeLog = false) {
      try {
        const data = await api('/api/trossen_ui/status');
        const trossenUiState = document.getElementById('trossenUiState');
        if (trossenUiState) {
          trossenUiState.textContent = data.running
            ? `Running PID ${data.pid}`
            : (data.available ? 'Ready' : 'Not installed');
        }
        if (writeLog) log(data.message);
      } catch (err) {
        log(`ERROR Trossen UI: ${err.message}`);
      }
    }

    async function startTrossenUI() {
      try {
        await api('/api/video/stop', {});
        const data = await api('/api/trossen_ui/start', {});
        log(data.message);
        await refreshAll(false);
      } catch (err) {
        log(`ERROR Trossen UI: ${err.message}`);
      }
    }

    async function stopTrossenUI() {
      try {
        const data = await api('/api/trossen_ui/stop', {});
        log(data.message);
        await refreshAll(false);
      } catch (err) {
        log(`ERROR Trossen UI: ${err.message}`);
      }
    }

    function setGravityControls() {
      const effort = Number(document.getElementById('wristEffortSlider').value);
      document.getElementById('wristEffortValue').textContent = effort.toFixed(2);
    }

    async function startCamera() {
      try {
        const source = selectedTeachCameraSource();
        if (!source) {
          log('ERROR camera: select a camera first');
          return;
        }
        
        const topRes = resolutionState.top_view || "640x480";
        const d405Res = resolutionState.d405 || "640x480";
        const [topW, topH] = topRes.split('x').map(Number);
        const [d405W, d405H] = d405Res.split('x').map(Number);

        if (source.startsWith('usb:')) {
          await api('/api/usb_cameras/start', {index: source.split(':')[1], width: topW, height: topH});
        } else {
          await api('/api/video/start', {source, width: topW, height: topH});
        }
        await api('/api/camera/start', {mode: 'color', width: d405W, height: d405H});
        log(`Camera preview started. Top cam: ${topW}x${topH}, D405: ${d405W}x${d405H}`);
        await refreshAll(false);
      } catch (err) {
        log(`ERROR camera: ${err.message}`);
      }
    }

    async function setCameraMode() {
      if (!cameraRunning) return;
      try {
        const source = selectedTeachCameraSource();
        if (!source) return;
        await api('/api/video/start', {source});
        updateCameraFrame();
      } catch (err) {
        log(`ERROR camera: ${err.message}`);
      }
    }

    async function startRecording() {
      try {
        const gravityData = await api('/api/gravity_compensation', gravityPayload());
        log(gravityData.message);
        const payload = {
          fps: 30,
          camera_mode: 'color',
          session_name: document.getElementById('sourceSessionName').value.trim(),
          high_smooth: true,
          with_camera: false
        };
        const data = await api('/api/teach/start', payload);
        log(data.message);
        await refreshAll(false);
      } catch (err) {
        log(`ERROR record: ${err.message}`);
      }
    }

    async function stopRecording() {
      try {
        const data = await api('/api/teach/stop', {});
        log(data.message);
        await loadRecordings(data.session_dir, null, data.session_dir);
        await refreshAll(false);
      } catch (err) {
        log(`ERROR record: ${err.message}`);
      }
    }

    async function startReplay() {
      const select = document.getElementById('recordingSelect');
      if (!select.value) {
        log('ERROR replay: select a recording');
        return;
      }
      try {
        const data = await api('/api/replay/start', {
          path: select.value,
          speed: Number(document.getElementById('replaySpeed').value),
          armed: document.getElementById('armed').checked
        });
        log(data.message);
        await refreshAll(false);
      } catch (err) {
        log(`ERROR replay: ${err.message}`);
      }
    }

    async function stopReplay() {
      try {
        const data = await api('/api/replay/stop', {});
        log(data.message);
        await refreshAll(false);
      } catch (err) {
        log(`ERROR replay: ${err.message}`);
      }
    }

    function datasetCropConfig(role) {
      const enabled = document.getElementById('datasetCropEnabled').checked;
      const rawTarget = document.getElementById('datasetCropTarget').value;
      const target = ['all', 'top', 'd405'].includes(rawTarget) ? rawTarget : 'all';
      const applies = enabled && (target === 'all' || (target === 'top' && role === 'top_view') || (target === 'd405' && role !== 'top_view'));
      if (!applies) return null;
      return {
        enabled: true,
        aspect: document.getElementById('datasetCropAspect').value,
        zoom: Number(document.getElementById('datasetCropZoom').value),
        offset_x: Number(document.getElementById('datasetCropX').value),
        offset_y: Number(document.getElementById('datasetCropY').value)
      };
    }

    async function startDatasetCapture() {
      const select = document.getElementById('recordingSelect');
      if (!select.value) {
        log('ERROR dataset: select a source movement');
        return;
      }
      try {
        const source = selectedTeachCameraSource();
        if (!source) {
          log('ERROR dataset: select a capture camera');
          return;
        }
        const data = await api('/api/dataset_capture/start', {
          path: select.value,
          speed: Number(document.getElementById('replaySpeed').value),
          camera_mode: 'color',
          video_source: source,
          capture_sources: [
            cameraCaptureState.top_view ? {source, role: 'top_view', crop: datasetCropConfig('top_view')} : null,
            cameraCaptureState.wrist_rgb ? {source: 'd405:color', role: 'wrist_rgb', crop: datasetCropConfig('wrist_rgb')} : null,
            cameraCaptureState.wrist_depth ? {source: 'd405:depth', role: 'wrist_depth', crop: datasetCropConfig('wrist_depth')} : null
          ].filter(Boolean),
          task_name: document.getElementById('datasetTaskName').value.trim(),
          armed: document.getElementById('armed').checked
        });
        log(data.message);
        await refreshAll(false);
      } catch (err) {
        log(`ERROR dataset: ${err.message}`);
      }
    }

    async function stopDatasetCapture() {
      try {
        const data = await api('/api/dataset_capture/stop', {});
        log(data.message);
        await loadRecordings(null, null, data.session_dir);
        await refreshAll(false);
      } catch (err) {
        log(`ERROR dataset: ${err.message}`);
      }
    }

    function reviewOptionLabel(rec) {
      const label = rec.capture_type === 'dataset_replay' ? 'dataset' : 'source';
      const task = rec.task_name ? ` · ${rec.task_name}` : '';
      const camera = rec.video_source ? ` · ${rec.video_source}` : '';
      const fps = formatFpsLabel(rec.actual_camera_fps, rec.nominal_camera_fps);
      const fpsText = fps ? ` · ${fps}` : '';
      const motorFps = formatFpsLabel(rec.actual_motor_fps, rec.nominal_motor_fps);
      const motorFpsText = motorFps ? ` · motor ${motorFps}` : '';
      return rec.mode === 'high_smooth'
        ? `${rec.name} · ${label}${task}${camera}${fpsText}${motorFpsText} (${rec.motor_samples} motor / ${rec.samples} img)`
        : `${rec.name} · ${label}${task}${fpsText}${motorFpsText} (${rec.samples} samples)`;
    }

    async function loadRecordings(preferredDir = null, nextAfterSource = null, preferredReviewDir = null) {
      try {
        const data = await api('/api/recordings');
        const allRecordings = data.recordings || [];
        reviewRecordings = allRecordings;
        recordings = allRecordings.filter((rec) => rec.capture_type !== 'dataset_replay');
        const select = document.getElementById('recordingSelect');
        select.innerHTML = '';
        const reviewSelect = document.getElementById('reviewRecordingSelect');
        reviewSelect.innerHTML = '';
        let selectedIndex = 0;
        recordings.forEach((rec, i) => {
          const opt = document.createElement('option');
          opt.value = rec.path;
          opt.textContent = reviewOptionLabel(rec).replace('dataset', 'source');
          select.appendChild(opt);
          if (preferredDir && rec.path === preferredDir) selectedIndex = i;
        });
        let reviewIndex = 0;
        reviewRecordings.forEach((rec, i) => {
          const opt = document.createElement('option');
          opt.value = rec.path;
          opt.textContent = reviewOptionLabel(rec);
          reviewSelect.appendChild(opt);
          if (preferredReviewDir && rec.path === preferredReviewDir) reviewIndex = i;
        });
        if (recordings.length === 0) {
          select.value = '';
        } else {
          if (nextAfterSource) {
            const sourceIndex = recordings.findIndex(rec => rec.path === nextAfterSource);
            selectedIndex = sourceIndex >= 0 ? sourceIndex : 0;
          } else if (!preferredDir) {
            selectedIndex = 0;
          }
          select.selectedIndex = Math.min(selectedIndex, recordings.length - 1);
        }
        if (reviewRecordings.length === 0) {
          stopReviewPlayback(false);
          selectedRecording = null;
          reviewFrames = [];
          reviewSelect.value = '';
          renderReviewEmpty('No recording yet.');
          return;
        }
        if (!preferredReviewDir) {
          const latestDatasetIndex = reviewRecordings.findIndex((rec) => rec.capture_type === 'dataset_replay');
          reviewIndex = latestDatasetIndex >= 0 ? latestDatasetIndex : 0;
        }
        reviewSelect.selectedIndex = Math.min(reviewIndex, reviewRecordings.length - 1);
        await loadReviewRecording();
      } catch (err) {
        log(`ERROR review: ${err.message}`);
      }
    }

    async function loadSelectedRecording() {
      const select = document.getElementById('recordingSelect');
      if (!select.value) return;
      const reviewSelect = document.getElementById('reviewRecordingSelect');
      if (reviewSelect && reviewSelect.value !== select.value) {
        reviewSelect.value = select.value;
      }
      await loadReviewRecording();
    }

    async function loadReviewRecording() {
      const select = document.getElementById('reviewRecordingSelect');
      if (!select.value) return;
      try {
        stopReviewPlayback(false);
        const data = await api('/api/recording/load', {path: select.value});
        selectedRecording = data;
        reviewFrames = data.frames;
        reviewIndex = 0;
        const slider = document.getElementById('frameSlider');
        slider.max = Math.max(0, reviewFrames.length - 1);
        slider.value = 0;
        showFrame(0);
      } catch (err) {
        log(`ERROR review: ${err.message}`);
      }
    }

    function formatReviewTime(seconds) {
      if (!Number.isFinite(seconds) || seconds < 0) return '0:00';
      const minutes = Math.floor(seconds / 60);
      const wholeSeconds = Math.floor(seconds % 60);
      return `${minutes}:${String(wholeSeconds).padStart(2, '0')}`;
    }

    function formatFpsLabel(actual, nominal = null) {
      const actualNumber = Number(actual);
      const nominalNumber = Number(nominal);
      if (!Number.isFinite(actualNumber) || actualNumber <= 0) {
        return Number.isFinite(nominalNumber) && nominalNumber > 0 ? `${nominalNumber.toFixed(0)} FPS target` : '';
      }
      const actualText = actualNumber >= 10 ? actualNumber.toFixed(1) : actualNumber.toFixed(2);
      if (Number.isFinite(nominalNumber) && nominalNumber > 0) {
        return `${actualText}/${nominalNumber.toFixed(0)} FPS`;
      }
      return `${actualText} FPS`;
    }

    function frameTime(index) {
      if (!reviewFrames.length) return 0;
      const first = Number(reviewFrames[0].timestamp);
      const current = Number(reviewFrames[Math.max(0, Math.min(index, reviewFrames.length - 1))].timestamp);
      if (!Number.isFinite(first) || !Number.isFinite(current)) return index / 30;
      return Math.max(0, current - first);
    }

    function recordingDuration() {
      if (reviewFrames.length < 2) return 0;
      return frameTime(reviewFrames.length - 1);
    }

    function updateReviewPlaybackState() {
      const state = document.getElementById('reviewPlaybackState');
      const current = formatReviewTime(frameTime(reviewIndex));
      const total = formatReviewTime(recordingDuration());
      state.textContent = `${current} / ${total} · frame ${reviewFrames.length ? reviewIndex + 1 : 0}/${reviewFrames.length}`;
    }

    function toggleReviewPlayback() {
      if (reviewPlaying) {
        stopReviewPlayback(false);
        return;
      }
      startReviewPlayback();
    }

    function startReviewPlayback() {
      if (!selectedRecording || reviewFrames.length === 0) {
        log('ERROR review: no recording to play');
        return;
      }
      if (reviewIndex >= reviewFrames.length - 1) showFrame(0);
      reviewPlaying = true;
      document.getElementById('reviewPlayButton').textContent = 'Pause';
      scheduleNextReviewFrame();
    }

    function stopReviewPlayback(reset) {
      if (reviewPlaybackTimer) {
        clearTimeout(reviewPlaybackTimer);
        reviewPlaybackTimer = null;
      }
      reviewPlaying = false;
      const button = document.getElementById('reviewPlayButton');
      if (button) button.textContent = 'Play';
      if (reset && reviewFrames.length) showFrame(0);
      else updateReviewPlaybackState();
    }

    function scheduleNextReviewFrame() {
      if (!reviewPlaying) return;
      if (reviewIndex >= reviewFrames.length - 1) {
        stopReviewPlayback(false);
        return;
      }
      const current = Number(reviewFrames[reviewIndex].timestamp);
      const next = Number(reviewFrames[reviewIndex + 1].timestamp);
      let delayMs = Number.isFinite(current) && Number.isFinite(next)
        ? (next - current) * 1000
        : 33;
      if (!Number.isFinite(delayMs) || delayMs <= 0) delayMs = 33;
      delayMs = Math.max(15, Math.min(delayMs, 250));
      reviewPlaybackTimer = setTimeout(() => {
        showFrame(reviewIndex + 1);
        scheduleNextReviewFrame();
      }, delayMs);
    }

    function showFrame(index) {
      if (!selectedRecording || reviewFrames.length === 0) {
        renderReviewEmpty('No frame');
        return;
      }
      reviewIndex = Math.max(0, Math.min(index, reviewFrames.length - 1));
      const frame = reviewFrames[reviewIndex];
      const topImg = document.getElementById('reviewImageTop');
      const topPlaceholder = document.getElementById('reviewPlaceholderTop');
      const wristRgbImg = document.getElementById('reviewImageWristRgb');
      const wristRgbPlaceholder = document.getElementById('reviewPlaceholderWristRgb');
      const wristDepthImg = document.getElementById('reviewImageWristDepth');
      const wristDepthPlaceholder = document.getElementById('reviewPlaceholderWristDepth');
      const images = frame.images || {};
      const topImage = images.top_view || frame.image;
      const wristRgbImage = images.wrist_rgb || null;
      const wristDepthImage = images.wrist_depth || null;
      if (topImage) {
        topImg.src = `/api/recording/image?path=${encodeURIComponent(selectedRecording.path)}&image=${encodeURIComponent(topImage)}&t=${Date.now()}`;
        topImg.style.display = 'block';
        topPlaceholder.style.display = 'none';
      } else {
        topImg.style.display = 'none';
        topPlaceholder.style.display = 'block';
        topPlaceholder.textContent = 'No top-view image';
      }
      if (wristRgbImage) {
        wristRgbImg.src = `/api/recording/image?path=${encodeURIComponent(selectedRecording.path)}&image=${encodeURIComponent(wristRgbImage)}&t=${Date.now()}`;
        wristRgbImg.style.display = 'block';
        wristRgbPlaceholder.style.display = 'none';
      } else {
        wristRgbImg.style.display = 'none';
        wristRgbPlaceholder.style.display = 'block';
        wristRgbPlaceholder.textContent = 'No wrist-RGB image';
      }
      if (wristDepthImage) {
        wristDepthImg.src = `/api/recording/image?path=${encodeURIComponent(selectedRecording.path)}&image=${encodeURIComponent(wristDepthImage)}&t=${Date.now()}`;
        wristDepthImg.style.display = 'block';
        wristDepthPlaceholder.style.display = 'none';
      } else {
        wristDepthImg.style.display = 'none';
        wristDepthPlaceholder.style.display = 'block';
        wristDepthPlaceholder.textContent = 'No wrist-depth image';
      }
      document.getElementById('frameSlider').value = reviewIndex;
      const gripperText = frame.gripper_position == null
        ? 'gripper n/a'
        : `gripper ${Number(frame.gripper_position).toFixed(4)} m`;
      const syncText = frame.motor_index == null
        ? 'sync n/a'
        : `sync motor ${frame.motor_index}`;
      const metadata = selectedRecording.metadata || {};
      const taskText = metadata.task_name ? ` · task ${metadata.task_name}` : '';
      const cameraText = metadata.video_source ? ` · camera ${metadata.video_source}` : '';
      const fpsText = formatFpsLabel(selectedRecording.actual_camera_fps, selectedRecording.nominal_camera_fps);
      const fpsMeta = fpsText ? ` · camera ${fpsText}` : '';
      const motorFpsText = formatFpsLabel(selectedRecording.actual_motor_fps, selectedRecording.nominal_motor_fps);
      const motorFpsMeta = motorFpsText ? ` · motor ${motorFpsText}` : '';
      document.getElementById('reviewMeta').textContent =
        `${selectedRecording.name}${taskText}${cameraText}${fpsMeta}${motorFpsMeta} · frame ${reviewIndex + 1}/${reviewFrames.length} · ${syncText} · ${gripperText} · qpos ${frame.qpos.map(v => Number(v).toFixed(2)).join(', ')}`;
      updateReviewPlaybackState();
    }

    function renderReviewEmpty(message) {
      stopReviewPlayback(false);
      document.getElementById('reviewImageTop').style.display = 'none';
      document.getElementById('reviewPlaceholderTop').style.display = 'block';
      document.getElementById('reviewPlaceholderTop').textContent = message;
      document.getElementById('reviewImageWristRgb').style.display = 'none';
      document.getElementById('reviewPlaceholderWristRgb').style.display = 'block';
      document.getElementById('reviewPlaceholderWristRgb').textContent = message;
      document.getElementById('reviewImageWristDepth').style.display = 'none';
      document.getElementById('reviewPlaceholderWristDepth').style.display = 'block';
      document.getElementById('reviewPlaceholderWristDepth').textContent = message;
      document.getElementById('reviewMeta').textContent = '';
      updateReviewPlaybackState();
    }

    function stepFrame(delta) {
      stopReviewPlayback(false);
      showFrame(reviewIndex + delta);
    }

    async function nextSourceRecording() {
      const select = document.getElementById('recordingSelect');
      if (select.options.length === 0) return;
      select.selectedIndex = Math.min(select.selectedIndex + 1, select.options.length - 1);
      await loadSelectedRecording();
    }

    async function previousSourceRecording() {
      const select = document.getElementById('recordingSelect');
      if (select.options.length === 0) return;
      select.selectedIndex = Math.max(select.selectedIndex - 1, 0);
      await loadSelectedRecording();
    }

    async function nextRecording() {
      const select = document.getElementById('reviewRecordingSelect');
      if (select.options.length === 0) return;
      select.selectedIndex = Math.min(select.selectedIndex + 1, select.options.length - 1);
      await loadReviewRecording();
    }

    async function previousRecording() {
      const select = document.getElementById('reviewRecordingSelect');
      if (select.options.length === 0) return;
      select.selectedIndex = Math.max(select.selectedIndex - 1, 0);
      await loadReviewRecording();
    }

    async function deleteRecording() {
      const select = document.getElementById('reviewRecordingSelect');
      if (!select.value) return;
      if (!confirm('Delete this recording?')) return;
      try {
        const nextIndex = Math.max(0, select.selectedIndex - 1);
        const data = await api('/api/recording/delete', {path: select.value});
        log(data.message);
        selectedRecording = null;
        reviewFrames = [];
        renderReviewEmpty('Recording deleted');
        await loadRecordings();
        const refreshed = document.getElementById('reviewRecordingSelect');
        if (refreshed.options.length > 0) {
          refreshed.selectedIndex = Math.min(nextIndex, refreshed.options.length - 1);
          await loadReviewRecording();
        }
      } catch (err) {
        log(`ERROR review: ${err.message}`);
      }
    }

    async function clearRecordings() {
      if (!confirm('Delete all recordings?')) return;
      try {
        const data = await api('/api/recordings/clear', {});
        log(data.message);
        selectedRecording = null;
        reviewFrames = [];
        const select = document.getElementById('recordingSelect');
        select.innerHTML = '';
        renderReviewEmpty('No recording');
        await loadRecordings();
      } catch (err) {
        log(`ERROR review: ${err.message}`);
      }
    }

    refreshAll();
    loadRecordings();
    setInterval(() => refreshAll(false), 1500);
  </script>
</body>
</html>
"""


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
            if self.usb_capture is not None and self.usb_index == index and self.usb_label:
                cameras.append(
                    {
                        "index": index,
                        "label": self.usb_label,
                        "device": f"/dev/video{index}",
                    }
                )
                continue
            label = self._usb_device_label(index)
            if not label:
                continue
            if self._is_builtin_laptop_camera_label(label):
                continue
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
                "running": self.usb_capture is not None,
                "active_index": self.usb_index,
                "active_label": self.usb_label,
                "active_device": f"/dev/video{self.usb_index}" if self.usb_index is not None else "",
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
            elif self.usb_capture is not None and self.usb_index is not None:
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
            if self.usb_capture is not None and self.usb_index == index:
                self.usb_last_message = f"USB camera already running: {label}"
                return self.usb_status_unlocked(self.usb_last_message)
            if self.usb_capture is not None:
                self.usb_capture.release()
                self.usb_capture = None
            capture = self._probe_usb_capture(f"/dev/video{index}", width, height)
            if capture is None:
                raise RuntimeError(
                    f"Unable to open /dev/video{index}. Another app may own it, or this node is not a capture stream."
                )
            self.usb_capture = capture
            self.usb_index = index
            self.usb_label = label
            self.usb_last_message = f"USB camera started: {label} (/dev/video{index}) at {width}x{height}"
            return self.usb_status_unlocked(self.usb_last_message)

    def usb_stop(self) -> dict[str, Any]:
        with self.lock:
            if self.usb_capture is not None:
                self.usb_capture.release()
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
            if self.usb_capture is not None:
                self.usb_capture.release()
                self.usb_capture = None
                self.usb_index = None
                self.usb_label = None
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
            if self.usb_capture is not None:
                self.usb_capture.release()
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
        if self.usb_index is not None and not any(camera["index"] == self.usb_index for camera in cameras):
            self.usb_index = None
            self.usb_label = None
        return {
            "ok": True,
            "cameras": cameras,
            "running": self.usb_capture is not None,
            "active_index": self.usb_index,
            "active_label": self.usb_label,
            "active_device": f"/dev/video{self.usb_index}" if self.usb_index is not None else "",
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
            if self.usb_capture is None or self.usb_index != index:
                raise RuntimeError("USB camera preview is not running for the selected device.")
            ok, frame = self.usb_capture.read()
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
                "capture_type": str(payload.get("capture_type", "manual")),
                "source_recording": payload.get("source_recording"),
                "task_name": str(payload.get("task_name", "")).strip() or None,
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
            frames[entry["role"]] = self.camera.video_frame_jpeg(entry["source"], entry.get("crop"))
        return frames

    def _capture_frame_set_with_timestamps(self) -> tuple[dict[str, bytes], dict[str, float], float, float]:
        capture_start = time.time()
        if not self.capture_sources:
            frame = self._capture_frame_jpeg()
            capture_end = time.time()
            return {"main": frame}, {"main": capture_end}, capture_start, capture_end
        frames: dict[str, bytes] = {}
        frame_timestamps: dict[str, float] = {}
        for entry in self.capture_sources:
            role = entry["role"]
            frames[role] = self.camera.video_frame_jpeg(entry["source"], entry.get("crop"))
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

    def _resolve_session(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.root / path
        resolved = path.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise RuntimeError("Recording path is outside the recordings directory.")
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

    def list(self) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        recordings = []
        for session in sorted(self.root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
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
        return {"ok": True, "recordings": recordings}

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
    def _bounded_int(payload: dict[str, Any], name: str, default: int, low: int, high: int) -> int:
        value = int(payload.get(name, default))
        if not low <= value <= high:
            raise RuntimeError(f"{name} must be between {low} and {high}.")
        return value

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
        steps = self._bounded_int(payload, "steps", 1, 1, 50)
        period = self._bounded_float(payload, "period", 1.0, 0.5, 5.0)
        max_speed = self._bounded_float(payload, "max_speed", 0.05, 0.02, 0.20)
        max_step_rad = self._bounded_float(payload, "max_step_rad", 0.035, 0.005, 0.10)
        envelope_margin = self._bounded_float(payload, "envelope_margin", 0.08, 0.0, 0.20)
        collision_action = str(payload.get("collision_action") or "gravity")
        if collision_action not in {"idle", "gravity"}:
            raise RuntimeError("collision_action must be 'idle' or 'gravity'.")
        stall_error_rad = self._bounded_float(payload, "stall_error_rad", 0.045, 0.015, 0.12)
        stall_velocity_rad_s = self._bounded_float(payload, "stall_velocity_rad_s", 0.008, 0.001, 0.05)
        stall_seconds = self._bounded_float(payload, "stall_seconds", 0.35, 0.1, 2.0)

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
            str(max(10.0, steps * period + 10.0)),
        ]
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
            stdout, stderr = process.communicate(timeout=max(30.0, steps * period + 30.0))
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


class RequestHandler(BaseHTTPRequestHandler):
    controller: ArmController
    camera_controller: CameraController
    teach_recorder: TeachRecorder
    recording_library: RecordingLibrary
    replay_runner: ReplayRunner
    dataset_capture_runner: DatasetCaptureRunner
    trossen_ui_runner: TrossenDataCollectionUIRunner
    model_test_runner: ModelTestRunner

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
        if parsed.path == "/api/recordings":
            self.send_json(self.recording_library.list())
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
                "/api/recording/load": lambda: self.recording_library.load(payload),
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
    RequestHandler.controller = controller
    RequestHandler.camera_controller = camera_controller
    RequestHandler.teach_recorder = teach_recorder
    RequestHandler.recording_library = recording_library
    RequestHandler.replay_runner = replay_runner
    RequestHandler.dataset_capture_runner = dataset_capture_runner
    RequestHandler.trossen_ui_runner = trossen_ui_runner
    RequestHandler.model_test_runner = model_test_runner
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
