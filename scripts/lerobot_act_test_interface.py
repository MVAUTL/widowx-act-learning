#!/usr/bin/env python3
"""Web UI to test a trained LeRobot ACT policy on dataset frames."""

from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageDraw
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy


JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
    "gripper",
]


def _tensor_image_to_data_url(tensor: torch.Tensor, title: str) -> str:
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    image = Image.fromarray((array * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(image.width, 260), 30), fill=(0, 0, 0))
    draw.text((8, 8), title, fill=(255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def _round_list(values: torch.Tensor) -> list[float]:
    return [round(float(x), 4) for x in values.detach().cpu().flatten().tolist()]


class App:
    def __init__(self, dataset_root: Path, repo_id: str, checkpoint: Path, device: str) -> None:
        self.dataset = LeRobotDataset(repo_id, root=dataset_root)
        self.device = torch.device(device)
        self.policy = ACTPolicy.from_pretrained(checkpoint, local_files_only=True)
        self.policy.to(self.device)
        self.policy.eval()
        self.checkpoint = checkpoint

    def predict(self, index: int) -> dict:
        index = max(0, min(index, len(self.dataset) - 1))
        sample = self.dataset[index]
        batch = {
            key: value.unsqueeze(0).to(self.device)
            for key, value in sample.items()
            if hasattr(value, "unsqueeze") and key.startswith(("observation", "action"))
        }
        with torch.inference_mode():
            pred = self.policy.select_action(batch)[0]
        target = sample["action"]
        error = (pred.detach().cpu() - target.detach().cpu()).abs()
        return {
            "index": index,
            "count": len(self.dataset),
            "checkpoint": str(self.checkpoint),
            "device": str(self.device),
            "top_view": _tensor_image_to_data_url(sample["observation.images.top_view"], "top_view"),
            "wrist_rgb": _tensor_image_to_data_url(sample["observation.images.wrist_rgb"], "wrist_rgb"),
            "state": _round_list(sample["observation.state"]),
            "target": _round_list(target),
            "pred": _round_list(pred),
            "abs_error": _round_list(error),
            "mean_abs_error": round(float(error.mean()), 4),
            "episode_index": int(sample["episode_index"]),
            "frame_index": int(sample["frame_index"]),
            "timestamp": round(float(sample["timestamp"]), 3),
        }


def _table(row: dict) -> str:
    lines = []
    for i, name in enumerate(JOINT_NAMES):
        lines.append(
            "<tr>"
            f"<td>{name}</td>"
            f"<td>{row['state'][i]:.4f}</td>"
            f"<td>{row['target'][i]:.4f}</td>"
            f"<td>{row['pred'][i]:.4f}</td>"
            f"<td>{row['abs_error'][i]:.4f}</td>"
            "</tr>"
        )
    return "\n".join(lines)


def _page(row: dict) -> str:
    idx = row["index"]
    previous_idx = max(0, idx - 1)
    next_idx = min(row["count"] - 1, idx + 1)
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LeRobot ACT Test</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; color: #17202a; background: #f5f6f7; }}
    header {{ background: #102033; color: white; padding: 16px 22px; }}
    main {{ padding: 18px 22px; max-width: 1280px; }}
    .bar {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 14px; }}
    input {{ width: 110px; padding: 8px; border: 1px solid #b8c1cc; border-radius: 6px; }}
    button, a.btn {{ padding: 9px 12px; border: 1px solid #0f62fe; background: #0f62fe; color: white; border-radius: 6px; text-decoration: none; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; }}
    .card {{ background: white; border: 1px solid #d8dde3; border-radius: 8px; padding: 14px; }}
    img {{ width: 100%; border-radius: 6px; border: 1px solid #d8dde3; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ text-align: right; padding: 8px; border-bottom: 1px solid #e5e8eb; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .metric {{ font-size: 26px; font-weight: 700; }}
    code {{ word-break: break-all; }}
  </style>
</head>
<body>
  <header><h1>LeRobot ACT Test</h1></header>
  <main>
    <form class="bar" method="get">
      <a class="btn" href="/?i={previous_idx}">Prev</a>
      <label>Frame <input name="i" type="number" min="0" max="{row['count'] - 1}" value="{idx}"></label>
      <button type="submit">Tester</button>
      <a class="btn" href="/?i={next_idx}">Next</a>
      <span>{idx + 1} / {row['count']}</span>
    </form>
    <div class="grid">
      <div class="card"><img src="{row['top_view']}" alt="top_view"></div>
      <div class="card"><img src="{row['wrist_rgb']}" alt="wrist_rgb"></div>
    </div>
    <div class="grid" style="margin-top:14px">
      <div class="card">
        <div>Mean absolute error</div>
        <div class="metric">{row['mean_abs_error']:.4f}</div>
        <p>episode {row['episode_index']} | frame {row['frame_index']} | t={row['timestamp']}s | {row['device']}</p>
      </div>
      <div class="card">
        <div>Checkpoint</div>
        <code>{row['checkpoint']}</code>
      </div>
    </div>
    <div class="card" style="margin-top:14px">
      <table>
        <thead><tr><th>Joint</th><th>State</th><th>Target action</th><th>Pred action</th><th>Abs error</th></tr></thead>
        <tbody>{_table(row)}</tbody>
      </table>
    </div>
  </main>
</body>
</html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", default="matteo/widowx-push-cube-local")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7866)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    app = App(Path(args.dataset_root), args.repo_id, Path(args.checkpoint), args.device)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            query = parse_qs(urlparse(self.path).query)
            index = int(query.get("i", ["0"])[0])
            row = app.predict(index)
            if urlparse(self.path).path == "/status.json":
                data = json.dumps({k: v for k, v in row.items() if not str(k).endswith("view") and k != "wrist_rgb"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            data = _page(row).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ACT test UI running at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
