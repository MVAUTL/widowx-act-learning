#!/usr/bin/env python3
"""Small HTTP monitor for LeRobot ACT training on DGX Spark."""

from __future__ import annotations

import argparse
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import time
from typing import Any


STEP_RE = re.compile(r"step:(?P<step>\d+(?:K|M)?).*?loss:(?P<loss>[0-9.]+).*?lr:(?P<lr>[0-9.eE+-]+)")


def _tail(path: Path, max_lines: int) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(errors="replace").splitlines()
    return lines[-max_lines:]


def _latest_metrics(lines: list[str]) -> dict[str, str | None]:
    latest: dict[str, str | None] = {"step": None, "loss": None, "lr": None}
    for line in lines:
        match = STEP_RE.search(line)
        if match:
            latest = match.groupdict()
            latest["step"] = _expand_count(latest["step"])
    return latest


def _metric_history(lines: list[str], limit: int = 80) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for line in lines:
        match = STEP_RE.search(line)
        if not match:
            continue
        step = _expand_count(match.group("step"))
        try:
            points.append(
                {
                    "step": float(step or 0),
                    "loss": float(match.group("loss")),
                    "lr": float(match.group("lr")),
                }
            )
        except ValueError:
            continue
    return points[-limit:]


def _loss_svg(points: list[dict[str, float]]) -> str:
    width = 760
    height = 220
    pad_left = 54
    pad_right = 18
    pad_top = 18
    pad_bottom = 36
    if len(points) < 2:
        return f'<svg viewBox="0 0 {width} {height}" role="img"><text x="24" y="112" fill="#5c6670">waiting for loss points</text></svg>'
    steps = [point["step"] for point in points]
    losses = [point["loss"] for point in points]
    min_step, max_step = min(steps), max(steps)
    min_loss, max_loss = min(losses), max(losses)
    if max_step <= min_step:
        max_step = min_step + 1
    if max_loss <= min_loss:
        max_loss = min_loss + 1e-6

    def x(step: float) -> float:
        return pad_left + ((step - min_step) / (max_step - min_step)) * (width - pad_left - pad_right)

    def y(loss: float) -> float:
        return pad_top + (1 - ((loss - min_loss) / (max_loss - min_loss))) * (height - pad_top - pad_bottom)

    polyline = " ".join(f"{x(point['step']):.1f},{y(point['loss']):.1f}" for point in points)
    last = points[-1]
    first = points[0]
    return f"""
<svg viewBox="0 0 {width} {height}" role="img" aria-label="Loss over training steps">
  <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" stroke="#d8dde3"/>
  <line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" y2="{height - pad_bottom}" stroke="#d8dde3"/>
  <text x="8" y="{pad_top + 6}" fill="#5c6670" font-size="12">{max_loss:.4g}</text>
  <text x="8" y="{height - pad_bottom}" fill="#5c6670" font-size="12">{min_loss:.4g}</text>
  <text x="{pad_left}" y="{height - 10}" fill="#5c6670" font-size="12">{int(first['step'])}</text>
  <text x="{width - pad_right - 58}" y="{height - 10}" fill="#5c6670" font-size="12">{int(last['step'])}</text>
  <polyline points="{polyline}" fill="none" stroke="#2f80ed" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
  <circle cx="{x(last['step']):.1f}" cy="{y(last['loss']):.1f}" r="4" fill="#2f80ed"/>
</svg>"""


def _expand_count(value: str | None) -> str | None:
    if value is None:
        return None
    if value.endswith("K"):
        return str(int(float(value[:-1]) * 1000))
    if value.endswith("M"):
        return str(int(float(value[:-1]) * 1_000_000))
    return value


def _dataset_info(path: Path) -> dict[str, Any]:
    info_path = path / "meta" / "info.json"
    if not info_path.exists():
        return {}
    try:
        return json.loads(info_path.read_text())
    except Exception as exc:
        return {"error": str(exc)}


class Handler(BaseHTTPRequestHandler):
    log_path: Path
    output_dir: Path
    dataset_dir: Path
    started_at: float

    def do_GET(self) -> None:
        if self.path == "/status.json":
            self._send_json(self._status())
            return
        self._send_html(self._page())

    def _status(self) -> dict[str, Any]:
        lines = _tail(self.log_path, 240)
        checkpoints = sorted((self.output_dir / "checkpoints").glob("*")) if self.output_dir.exists() else []
        history = _metric_history(lines)
        return {
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "log_path": str(self.log_path),
            "output_dir": str(self.output_dir),
            "dataset_dir": str(self.dataset_dir),
            "latest": _latest_metrics(lines),
            "history": history,
            "checkpoints": [path.name for path in checkpoints if path.is_dir()],
            "dataset": _dataset_info(self.dataset_dir),
            "tail": lines[-80:],
        }

    def _page(self) -> str:
        status = self._status()
        latest = status["latest"]
        checkpoints = ", ".join(status["checkpoints"]) or "none yet"
        dataset = status["dataset"]
        chart = _loss_svg(status["history"])
        tail = "\n".join(html.escape(line) for line in status["tail"])
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="5">
  <title>ACT Training Monitor</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #f6f7f9; color: #14171a; }}
    header {{ padding: 18px 24px; background: #102033; color: white; }}
    main {{ padding: 20px 24px; max-width: 1200px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }}
    .card {{ background: white; border: 1px solid #d8dde3; border-radius: 8px; padding: 14px; }}
    .label {{ color: #5c6670; font-size: 13px; }}
    .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
    pre {{ background: #0d1117; color: #d6deeb; padding: 16px; border-radius: 8px; overflow: auto; max-height: 58vh; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    svg {{ display: block; width: 100%; height: 260px; }}
  </style>
</head>
<body>
  <header><h1>ACT Training Monitor</h1></header>
  <main>
    <div class="grid">
      <div class="card"><div class="label">Step</div><div class="value">{html.escape(str(latest.get("step") or "waiting"))}</div></div>
      <div class="card"><div class="label">Loss</div><div class="value">{html.escape(str(latest.get("loss") or "waiting"))}</div></div>
      <div class="card"><div class="label">Learning rate</div><div class="value">{html.escape(str(latest.get("lr") or "waiting"))}</div></div>
      <div class="card"><div class="label">Uptime</div><div class="value">{status["uptime_seconds"]}s</div></div>
    </div>
    <div class="card" style="margin-top:12px">
      <div class="label">Checkpoints</div>
      <div>{html.escape(checkpoints)}</div>
    </div>
    <div class="card" style="margin-top:12px">
      <div class="label">Loss graph</div>
      {chart}
    </div>
    <div class="card" style="margin-top:12px">
      <div class="label">Dataset</div>
      <code>{html.escape(json.dumps(dataset, indent=2)[:3000])}</code>
    </div>
    <h2>Live Log</h2>
    <pre>{tail}</pre>
  </main>
</body>
</html>"""

    def _send_json(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, indent=2).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, html_text: str) -> None:
        data = html_text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7865)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    args = parser.parse_args()

    Handler.log_path = Path(args.log_path)
    Handler.output_dir = Path(args.output_dir)
    Handler.dataset_dir = Path(args.dataset_dir)
    Handler.started_at = time.time()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ACT monitor running at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
