#!/usr/bin/env python3
"""Local dashboard for ACT training runs."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


INDEX_HTML = """<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ACT Training Monitor</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101214;
      --panel: #181c20;
      --panel-2: #20262b;
      --text: #eef2f4;
      --muted: #a6b0b8;
      --accent: #37c48d;
      --warning: #e6b450;
      --danger: #e45858;
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
      font-size: 28px;
      line-height: 1.08;
      font-weight: 720;
    }
    .sub {
      margin-top: 6px;
      color: var(--muted);
      font-size: 14px;
    }
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
      grid-template-columns: 1fr auto auto;
      gap: 10px;
      margin-bottom: 18px;
    }
    select, button {
      height: 40px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--panel-2);
      color: var(--text);
      padding: 0 12px;
      font-size: 14px;
    }
    button { cursor: pointer; }
    button:hover { border-color: #52616b; }
    .cards {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .metric {
      background: #111518;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-height: 82px;
    }
    .metric .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      font-weight: 680;
    }
    .metric .value {
      margin-top: 8px;
      font-size: 25px;
      font-weight: 760;
      overflow-wrap: anywhere;
    }
    .state {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 14px;
      color: var(--muted);
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--warning);
    }
    .dot.completed { background: var(--accent); }
    .dot.missing, .dot.error { background: var(--danger); }
    canvas {
      width: 100%;
      height: 360px;
      display: block;
      background: #0b0d0f;
      border: 1px solid var(--line);
      border-radius: 8px;
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
      min-height: 180px;
      max-height: 420px;
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
    .progress {
      width: 100%;
      height: 12px;
      background: #0b0d0f;
      border: 1px solid var(--line);
      border-radius: 999px;
      overflow: hidden;
      margin: 12px 0 18px;
    }
    .bar {
      width: 0%;
      height: 100%;
      background: var(--accent);
      transition: width .25s ease;
    }
    .legend {
      display: flex;
      gap: 16px;
      color: var(--muted);
      font-size: 13px;
      margin-top: 10px;
    }
    .swatch {
      display: inline-block;
      width: 11px;
      height: 11px;
      border-radius: 2px;
      margin-right: 6px;
      vertical-align: -1px;
    }
    @media (max-width: 900px) {
      header, .layout { grid-template-columns: 1fr; display: grid; }
      .toolbar, .cards { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 560px) {
      main { width: min(100vw - 20px, 1180px); padding-top: 14px; }
      .toolbar, .cards { grid-template-columns: 1fr; }
      canvas { height: 300px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>ACT Training Monitor</h1>
        <div class="sub">Suivi local des runs dans <span id="root"></span></div>
      </div>
      <div class="state"><span id="dot" class="dot"></span><span id="state">chargement</span></div>
    </header>
    <div class="layout">
      <section>
        <div class="toolbar">
          <select id="run"></select>
          <button id="latest">Dernier run</button>
          <button id="refresh">Rafraichir</button>
        </div>
        <div class="cards">
          <div class="metric"><div class="label">Epoch</div><div id="epoch" class="value">-</div></div>
          <div class="metric"><div class="label">Progression</div><div id="progressMetric" class="value">-</div></div>
          <div class="metric"><div class="label">Train L1</div><div id="train" class="value">-</div></div>
          <div class="metric"><div class="label">Val L1</div><div id="val" class="value">-</div></div>
          <div class="metric"><div class="label">Best</div><div id="best" class="value">-</div></div>
          <div class="metric"><div class="label">Best epoch</div><div id="bestEpoch" class="value">-</div></div>
          <div class="metric"><div class="label">Gap val-train</div><div id="gap" class="value">-</div></div>
          <div class="metric"><div class="label">Delta val</div><div id="delta" class="value">-</div></div>
          <div class="metric"><div class="label">Gain val</div><div id="improvement" class="value">-</div></div>
          <div class="metric"><div class="label">Temps/epoch</div><div id="epochTime" class="value">-</div></div>
          <div class="metric"><div class="label">ETA</div><div id="eta" class="value">-</div></div>
          <div class="metric"><div class="label">Train ex/s</div><div id="speed" class="value">-</div></div>
          <div class="metric"><div class="label">Grad norm</div><div id="gradNorm" class="value">-</div></div>
          <div class="metric"><div class="label">LR</div><div id="lr" class="value">-</div></div>
        </div>
        <div class="progress"><div id="bar" class="bar"></div></div>
        <canvas id="chart" width="1000" height="360"></canvas>
        <div class="legend">
          <span><span class="swatch" style="background:#37c48d"></span>train_l1</span>
          <span><span class="swatch" style="background:#e6b450"></span>val_l1</span>
        </div>
      </section>
      <aside>
        <h2 class="side-title">Run</h2>
        <div class="kv">
          <div>Nom</div><div id="runName">-</div>
          <div>Device</div><div id="device">-</div>
          <div>Vue</div><div id="view">-</div>
          <div>Exemples</div><div id="examples">-</div>
          <div>Batch</div><div id="batch">-</div>
          <div>Chunk</div><div id="chunk">-</div>
          <div>Trim fin</div><div id="trimEnd">-</div>
          <div>Checkpoint</div><div id="checkpoint">-</div>
          <div>MAJ</div><div id="updated">-</div>
        </div>
        <h2 class="side-title">Config</h2>
        <pre id="config">{}</pre>
      </aside>
    </div>
  </main>
  <script>
    const runSelect = document.getElementById('run');
    const rootEl = document.getElementById('root');
    const stateEl = document.getElementById('state');
    const dotEl = document.getElementById('dot');
    const ids = [
      'epoch', 'progressMetric', 'train', 'val', 'best', 'bestEpoch', 'gap', 'delta',
      'improvement', 'epochTime', 'eta', 'speed', 'gradNorm', 'lr', 'runName',
      'device', 'view', 'examples', 'batch', 'chunk', 'trimEnd', 'checkpoint',
      'updated', 'config'
    ];
    const el = Object.fromEntries(ids.map(id => [id, document.getElementById(id)]));
    const bar = document.getElementById('bar');
    const chart = document.getElementById('chart');
    const ctx = chart.getContext('2d');

    function fmt(value) {
      if (value === null || value === undefined || Number.isNaN(value)) return '-';
      if (typeof value === 'number') return value.toFixed(5);
      return String(value);
    }

    function fmtSigned(value) {
      if (value === null || value === undefined || Number.isNaN(value)) return '-';
      return `${value >= 0 ? '+' : ''}${value.toFixed(5)}`;
    }

    function fmtPct(value) {
      if (value === null || value === undefined || Number.isNaN(value)) return '-';
      return `${value.toFixed(1)}%`;
    }

    function fmtDuration(seconds) {
      if (!Number.isFinite(seconds) || seconds < 0) return '-';
      if (seconds < 60) return `${seconds.toFixed(0)}s`;
      const minutes = Math.floor(seconds / 60);
      const rest = Math.round(seconds % 60);
      if (minutes < 60) return `${minutes}m ${rest}s`;
      const hours = Math.floor(minutes / 60);
      return `${hours}h ${minutes % 60}m`;
    }

    function numberOrNull(value) {
      return typeof value === 'number' && Number.isFinite(value) ? value : null;
    }

    function isoSecondsBetween(start, end) {
      const startMs = Date.parse(start || '');
      const endMs = Date.parse(end || '');
      if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs <= startMs) return null;
      return (endMs - startMs) / 1000;
    }

    function average(values) {
      const filtered = values.filter(value => typeof value === 'number' && Number.isFinite(value) && value > 0);
      if (filtered.length === 0) return null;
      return filtered.reduce((a, b) => a + b, 0) / filtered.length;
    }

    function drawChart(history) {
      ctx.clearRect(0, 0, chart.width, chart.height);
      ctx.fillStyle = '#0b0d0f';
      ctx.fillRect(0, 0, chart.width, chart.height);
      const pad = { left: 54, right: 18, top: 18, bottom: 42 };
      const w = chart.width - pad.left - pad.right;
      const h = chart.height - pad.top - pad.bottom;
      ctx.strokeStyle = '#2d363d';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.left, pad.top);
      ctx.lineTo(pad.left, pad.top + h);
      ctx.lineTo(pad.left + w, pad.top + h);
      ctx.stroke();
      if (!history || history.length === 0) return;
      const vals = [];
      history.forEach(row => {
        if (typeof row.train_l1 === 'number') vals.push(row.train_l1);
        if (typeof row.val_l1 === 'number') vals.push(row.val_l1);
      });
      const maxY = Math.max(...vals, 1e-6);
      const minY = Math.min(...vals, 0);
      const spanY = Math.max(maxY - minY, 1e-6);
      const xFor = i => pad.left + (history.length === 1 ? 0 : (i / (history.length - 1)) * w);
      const yFor = v => pad.top + h - ((v - minY) / spanY) * h;
      function line(key, color) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.beginPath();
        let started = false;
        history.forEach((row, i) => {
          const v = row[key];
          if (typeof v !== 'number') return;
          const x = xFor(i);
          const y = yFor(v);
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        });
        ctx.stroke();
      }
      line('train_l1', '#37c48d');
      line('val_l1', '#e6b450');
      ctx.fillStyle = '#a6b0b8';
      ctx.font = '13px ui-sans-serif, system-ui';
      ctx.fillText(maxY.toFixed(3), 8, pad.top + 5);
      ctx.fillText(minY.toFixed(3), 8, pad.top + h);
      ctx.fillText('epoch', pad.left + w - 38, chart.height - 12);
    }

    async function fetchJson(url) {
      const response = await fetch(url, { cache: 'no-store' });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      return response.json();
    }

    async function loadRuns(selectLatest = false) {
      const data = await fetchJson('/api/runs');
      rootEl.textContent = data.root;
      const current = runSelect.value;
      runSelect.innerHTML = '';
      data.runs.forEach(run => {
        const option = document.createElement('option');
        option.value = run.name;
        option.textContent = run.name;
        runSelect.appendChild(option);
      });
      if (data.runs.length === 0) return;
      runSelect.value = selectLatest || !current ? data.runs[0].name : current;
      if (!runSelect.value) runSelect.value = data.runs[0].name;
    }

    async function loadRun() {
      if (!runSelect.value) {
        stateEl.textContent = 'aucun run';
        dotEl.className = 'dot missing';
        return;
      }
      const data = await fetchJson(`/api/run?name=${encodeURIComponent(runSelect.value)}`);
      const status = data.status || {};
      const config = data.config || {};
      const history = data.history || [];
      const latest = history[history.length - 1] || {};
      stateEl.textContent = status.state || 'inconnu';
      dotEl.className = `dot ${status.state || 'missing'}`;
      const currentEpoch = status.current_epoch ?? latest.epoch ?? 0;
      const totalEpochs = status.total_epochs ?? config.epochs ?? 0;
      const pct = totalEpochs > 0 ? Math.max(0, Math.min(100, currentEpoch * 100 / totalEpochs)) : 0;
      const latestTrain = numberOrNull(status.latest_train_l1) ?? numberOrNull(latest.train_l1);
      const latestVal = numberOrNull(status.latest_val_l1) ?? numberOrNull(latest.val_l1);
      const firstVal = numberOrNull((history.find(row => numberOrNull(row.val_l1) !== null) || {}).val_l1);
      const previousVal = history.length > 1 ? numberOrNull(history[history.length - 2].val_l1) : null;
      const gap = numberOrNull(status.latest_gap) ?? numberOrNull(latest.generalization_gap)
        ?? (latestTrain !== null && latestVal !== null ? latestVal - latestTrain : null);
      const delta = numberOrNull(status.latest_val_delta) ?? numberOrNull(latest.val_delta)
        ?? (latestVal !== null && previousVal !== null ? latestVal - previousVal : null);
      const improvement = latestVal !== null && firstVal !== null ? firstVal - latestVal : null;
      const improvementPct = improvement !== null && firstVal !== 0 ? improvement * 100 / firstVal : null;
      const recentEpochSeconds = average(history.slice(-5).map(row => row.epoch_seconds));
      const elapsedSeconds = isoSecondsBetween(status.started_at, status.updated_at);
      const estimatedEpochSeconds = recentEpochSeconds
        ?? numberOrNull(status.latest_epoch_seconds)
        ?? numberOrNull(latest.epoch_seconds)
        ?? (elapsedSeconds !== null && currentEpoch > 0 ? elapsedSeconds / currentEpoch : null);
      const etaSeconds = estimatedEpochSeconds !== null ? Math.max(0, totalEpochs - currentEpoch) * estimatedEpochSeconds : null;
      el.epoch.textContent = `${currentEpoch}/${totalEpochs}`;
      el.progressMetric.textContent = fmtPct(pct);
      el.train.textContent = fmt(latestTrain);
      el.val.textContent = fmt(latestVal);
      el.best.textContent = fmt(status.best_score);
      el.bestEpoch.textContent = status.best_epoch ?? '-';
      el.gap.textContent = fmtSigned(gap);
      el.delta.textContent = fmtSigned(delta);
      el.improvement.textContent = improvement === null ? '-' : `${fmtSigned(improvement)} (${fmtPct(improvementPct)})`;
      el.epochTime.textContent = fmtDuration(estimatedEpochSeconds);
      el.eta.textContent = fmtDuration(etaSeconds);
      el.speed.textContent = fmt(numberOrNull(status.latest_train_samples_per_sec) ?? numberOrNull(latest.train_samples_per_sec));
      el.gradNorm.textContent = fmt(numberOrNull(status.latest_grad_norm_avg) ?? numberOrNull(latest.grad_norm_avg));
      el.lr.textContent = fmt(numberOrNull(status.latest_lr) ?? numberOrNull(latest.lr));
      el.runName.textContent = data.name;
      el.device.textContent = config.device || '-';
      el.view.textContent = config.view || '-';
      el.examples.textContent = `${config.train_examples ?? '-'} train / ${config.val_examples ?? '-'} val`;
      el.batch.textContent = config.batch_size ?? '-';
      el.chunk.textContent = config.chunk_size ?? '-';
      el.trimEnd.textContent = config.trim_end_seconds === undefined ? '-' : `${config.trim_end_seconds}s`;
      el.checkpoint.textContent = status.best_checkpoint || '-';
      el.updated.textContent = status.updated_at || '-';
      el.config.textContent = JSON.stringify(config, null, 2);
      bar.style.width = `${pct}%`;
      drawChart(history);
    }

    async function refresh(selectLatest = false) {
      try {
        await loadRuns(selectLatest);
        await loadRun();
      } catch (error) {
        stateEl.textContent = error.message;
        dotEl.className = 'dot error';
      }
    }

    document.getElementById('refresh').addEventListener('click', () => refresh(false));
    document.getElementById('latest').addEventListener('click', () => refresh(true));
    runSelect.addEventListener('change', loadRun);
    refresh(true);
    setInterval(() => refresh(false), 2000);
  </script>
</body>
</html>
"""
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _status_for_run(run_dir: Path) -> dict[str, Any]:
    status = _read_json(run_dir / "status.json", {})
    if status:
        return status
    config = _read_json(run_dir / "config.json", {})
    history = _read_json(run_dir / "history.json", [])
    latest = history[-1] if history else {}
    best_epoch = None
    best_score = None
    for row in history:
        val = row.get("val_l1")
        score = row.get("train_l1") if val is None else val
        if score is not None and (best_score is None or score < best_score):
            best_score = score
            best_epoch = row.get("epoch")
    return {
        "state": "completed" if history else "missing",
        "updated_at": None,
        "current_epoch": latest.get("epoch", 0),
        "total_epochs": config.get("epochs"),
        "latest_train_l1": latest.get("train_l1"),
        "latest_val_l1": latest.get("val_l1"),
        "best_score": best_score,
        "best_epoch": best_epoch,
        "best_checkpoint": str(run_dir / "best.pt") if (run_dir / "best.pt").exists() else None,
        "last_checkpoint": str(run_dir / "last.pt") if (run_dir / "last.pt").exists() else None,
        "run_dir": str(run_dir),
    }


class MonitorHandler(BaseHTTPRequestHandler):
    root: Path

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _runs(self) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        if self.root.exists():
            for path in self.root.iterdir():
                if not path.is_dir():
                    continue
                status = _read_json(path / "status.json", {})
                config = _read_json(path / "config.json", {})
                if not status:
                    status = _status_for_run(path)
                runs.append(
                    {
                        "name": path.name,
                        "mtime": path.stat().st_mtime,
                        "state": status.get("state"),
                        "current_epoch": status.get("current_epoch"),
                        "total_epochs": status.get("total_epochs") or config.get("epochs"),
                    }
                )
        return sorted(runs, key=lambda run: float(run["mtime"]), reverse=True)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(INDEX_HTML)
            return
        if parsed.path == "/api/runs":
            self._send_json({"root": str(self.root), "runs": self._runs()})
            return
        if parsed.path == "/api/run":
            query = parse_qs(parsed.query)
            name = (query.get("name") or [""])[0]
            run_dir = (self.root / name).resolve()
            root = self.root.resolve()
            if root not in [run_dir, *run_dir.parents] or not run_dir.is_dir():
                self._send_json({"error": "run not found"}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(
                {
                    "name": run_dir.name,
                    "status": _status_for_run(run_dir),
                    "config": _read_json(run_dir / "config.json", {}),
                    "history": _read_json(run_dir / "history.json", []),
                    "normalization": _read_json(run_dir / "normalization.json", {}),
                }
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND)


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve a local ACT training monitor.")
    parser.add_argument("--root", default=str(PACKAGE_ROOT / "models"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7864)
    args = parser.parse_args()

    MonitorHandler.root = Path(args.root).expanduser().resolve()
    server = ThreadingHTTPServer((args.host, args.port), MonitorHandler)
    print(f"ACT monitor running at http://{args.host}:{args.port}")
    print(f"Watching: {MonitorHandler.root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping ACT monitor.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
