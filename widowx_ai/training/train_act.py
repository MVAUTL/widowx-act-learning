#!/usr/bin/env python3
"""Train a small ACT-style policy from WidowX teaching recordings.

This is intentionally compact and CPU-friendly. It is meant as a first
sanity-check trainer for the local recordings format, not as a production
robot policy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


STATE_DIM = 7
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _state(row: dict[str, Any]) -> np.ndarray:
    return np.array([*row["qpos"], row.get("gripper_position", 0.0)], dtype=np.float32)


def _latest_run_name() -> str:
    return datetime.now().strftime("act_%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class Example:
    session: Path
    sample_index: int
    action_indices: list[int]


class WidowXActDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        sessions: list[Path],
        *,
        view: str,
        chunk_size: int,
        image_size: int,
        trim_end_seconds: float,
        stats: dict[str, Any] | None = None,
    ) -> None:
        self.sessions = sessions
        self.view = view
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.trim_end_seconds = trim_end_seconds
        self.samples_by_session: dict[Path, list[dict[str, Any]]] = {}
        self.motor_by_session: dict[Path, list[dict[str, Any]]] = {}
        self.examples: list[Example] = []

        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        for session in sessions:
            samples = _read_jsonl(session / "samples.jsonl")
            motor = _read_jsonl(session / "motor_samples.jsonl")
            self.samples_by_session[session] = samples
            self.motor_by_session[session] = motor
            timestamp_values = [
                float(row["timestamp"])
                for row in [*samples, *motor]
                if row.get("timestamp") is not None
            ]
            cutoff_timestamp = max(timestamp_values) - trim_end_seconds if timestamp_values else None
            for sample in samples:
                images = sample.get("images") or {}
                if view not in images:
                    continue
                motor_index = int(sample.get("motor_index", -1))
                action_indices = list(range(motor_index + 1, motor_index + 1 + chunk_size))
                if motor_index < 0 or action_indices[-1] >= len(motor):
                    continue
                if cutoff_timestamp is not None:
                    sample_timestamp = float(sample.get("timestamp", 0.0))
                    last_action_timestamp = float(motor[action_indices[-1]].get("timestamp", sample_timestamp))
                    if sample_timestamp > cutoff_timestamp or last_action_timestamp > cutoff_timestamp:
                        continue
                image_path = session / str(images[view])
                if not image_path.exists():
                    continue
                self.examples.append(Example(session, int(sample["index"]), action_indices))
                states.append(_state(sample))
                actions.extend(_state(motor[index]) for index in action_indices)

        if not self.examples:
            raise RuntimeError(f"No usable examples found for view '{view}'.")

        if stats is None:
            state_array = np.stack(states)
            action_array = np.stack(actions)
            self.stats = {
                "state_mean": state_array.mean(axis=0).tolist(),
                "state_std": np.maximum(state_array.std(axis=0), 1e-6).tolist(),
                "action_mean": action_array.mean(axis=0).tolist(),
                "action_std": np.maximum(action_array.std(axis=0), 1e-6).tolist(),
            }
        else:
            self.stats = stats

        self.state_mean = np.asarray(self.stats["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(self.stats["state_std"], dtype=np.float32)
        self.action_mean = np.asarray(self.stats["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(self.stats["action_std"], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        sample = self.samples_by_session[example.session][example.sample_index]
        motor = self.motor_by_session[example.session]
        image_rel = (sample.get("images") or {})[self.view]
        image = Image.open(example.session / str(image_rel)).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = np.transpose(image_array, (2, 0, 1))

        state = (_state(sample) - self.state_mean) / self.state_std
        action = np.stack([_state(motor[i]) for i in example.action_indices])
        action = (action - self.action_mean) / self.action_std
        return {
            "image": torch.from_numpy(image_array),
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(action.astype(np.float32)),
        }


class TinyActPolicy(nn.Module):
    def __init__(self, *, chunk_size: int, action_dim: int, hidden_dim: int, heads: int, layers: int) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(96, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.state_encoder = nn.Sequential(nn.Linear(STATE_DIM, hidden_dim), nn.ReLU(inplace=True))
        self.context = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(inplace=True))
        self.query_embed = nn.Parameter(torch.randn(chunk_size, hidden_dim) / math.sqrt(hidden_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        image_feat = self.image_encoder(image)
        state_feat = self.state_encoder(state)
        memory = self.context(torch.cat([image_feat, state_feat], dim=-1)).unsqueeze(1)
        queries = self.query_embed.unsqueeze(0).expand(image.shape[0], -1, -1)
        decoded = self.decoder(queries, memory)
        return self.action_head(decoded)


def _find_sessions(root: Path) -> list[Path]:
    sessions = [
        path
        for path in sorted(root.glob("dataset_*"))
        if path.is_dir() and (path / "samples.jsonl").exists() and (path / "motor_samples.jsonl").exists()
    ]
    if not sessions:
        raise RuntimeError(f"No dataset sessions found in {root}.")
    return sessions


def _split_sessions(sessions: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    shuffled = sessions[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
    return shuffled[val_count:], shuffled[:val_count]


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float | int | None]:
    training = optimizer is not None
    model.train(training)
    started = time.perf_counter()
    total_loss = 0.0
    total_items = 0
    total_batches = 0
    grad_norm_sum = 0.0
    grad_norm_max: float | None = None
    grad_norm_count = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            image = batch["image"].to(device)
            state = batch["state"].to(device)
            action = batch["action"].to(device)
            pred = model(image, state)
            loss = nn.functional.l1_loss(pred, action)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                grad_norm_sum += grad_norm
                grad_norm_max = grad_norm if grad_norm_max is None else max(grad_norm_max, grad_norm)
                grad_norm_count += 1
                optimizer.step()
            total_loss += float(loss.item()) * image.shape[0]
            total_items += image.shape[0]
            total_batches += 1
    seconds = max(time.perf_counter() - started, 1e-9)
    return {
        "loss": total_loss / max(total_items, 1),
        "seconds": seconds,
        "items": total_items,
        "batches": total_batches,
        "samples_per_sec": total_items / seconds,
        "grad_norm_avg": grad_norm_sum / grad_norm_count if grad_norm_count else None,
        "grad_norm_max": grad_norm_max,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a tiny ACT-style policy from WidowX recordings.")
    parser.add_argument("--root", default=str(PACKAGE_ROOT / "recordings"))
    parser.add_argument("--output-dir", default=str(PACKAGE_ROOT / "models"))
    parser.add_argument("--run-name", default=None, help="Optional fixed run directory name.")
    parser.add_argument("--view", default="wrist_rgb", choices=["top_view", "wrist_rgb"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--trim-end-seconds", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    run_dir = output_root / (args.run_name or _latest_run_name())
    run_dir.mkdir(parents=True, exist_ok=True)

    sessions = _find_sessions(root)
    train_sessions, val_sessions = _split_sessions(sessions, args.val_ratio, args.seed)
    train_dataset = WidowXActDataset(
        train_sessions,
        view=args.view,
        chunk_size=args.chunk_size,
        image_size=args.image_size,
        trim_end_seconds=args.trim_end_seconds,
    )
    val_dataset = (
        WidowXActDataset(
            val_sessions,
            view=args.view,
            chunk_size=args.chunk_size,
            image_size=args.image_size,
            trim_end_seconds=args.trim_end_seconds,
            stats=train_dataset.stats,
        )
        if val_sessions
        else None
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        if val_dataset is not None
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyActPolicy(
        chunk_size=args.chunk_size,
        action_dim=STATE_DIM,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        layers=args.layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    config = vars(args) | {
        "device": str(device),
        "train_sessions": [path.name for path in train_sessions],
        "val_sessions": [path.name for path in val_sessions],
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset) if val_dataset is not None else 0,
        "state_dim": STATE_DIM,
    }
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "normalization.json", train_dataset.stats)
    _write_json(
        run_dir / "status.json",
        {
            "state": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "current_epoch": 0,
            "total_epochs": args.epochs,
            "best_score": None,
            "best_epoch": None,
            "run_dir": str(run_dir),
        },
    )

    history: list[dict[str, float | int | None]] = []
    best_val = float("inf")
    best_epoch: int | None = None
    best_path = run_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        previous_val = next(
            (float(row["val_l1"]) for row in reversed(history) if row.get("val_l1") is not None),
            None,
        )
        train_metrics = _run_epoch(model, train_loader, optimizer=optimizer, device=device)
        val_metrics = _run_epoch(model, val_loader, optimizer=None, device=device) if val_loader is not None else None
        train_loss = float(train_metrics["loss"])
        val_loss = float(val_metrics["loss"]) if val_metrics is not None else None
        epoch_seconds = float(train_metrics["seconds"]) + (float(val_metrics["seconds"]) if val_metrics else 0.0)
        generalization_gap = val_loss - train_loss if val_loss is not None else None
        val_delta = val_loss - previous_val if val_loss is not None and previous_val is not None else None
        latest_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_l1": train_loss,
                "val_l1": val_loss,
                "generalization_gap": generalization_gap,
                "val_delta": val_delta,
                "epoch_seconds": epoch_seconds,
                "train_seconds": train_metrics["seconds"],
                "val_seconds": val_metrics["seconds"] if val_metrics else None,
                "train_samples_per_sec": train_metrics["samples_per_sec"],
                "val_samples_per_sec": val_metrics["samples_per_sec"] if val_metrics else None,
                "train_batches": train_metrics["batches"],
                "val_batches": val_metrics["batches"] if val_metrics else None,
                "grad_norm_avg": train_metrics["grad_norm_avg"],
                "grad_norm_max": train_metrics["grad_norm_max"],
                "lr": latest_lr,
            }
        )
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_l1={train_loss:.5f} "
            f"val_l1={'n/a' if val_loss is None else f'{val_loss:.5f}'} "
            f"gap={'n/a' if generalization_gap is None else f'{generalization_gap:+.5f}'} "
            f"epoch_s={epoch_seconds:.1f} "
            f"train_ex_s={float(train_metrics['samples_per_sec']):.1f}",
            flush=True,
        )
        score = train_loss if val_loss is None else val_loss
        if score < best_val:
            best_val = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "normalization": train_dataset.stats,
                    "epoch": epoch,
                    "score": score,
                },
                best_path,
            )
        _write_json(run_dir / "history.json", history)
        _write_json(
            run_dir / "status.json",
            {
                "state": "running",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "current_epoch": epoch,
                "total_epochs": args.epochs,
                "latest_train_l1": train_loss,
                "latest_val_l1": val_loss,
                "latest_gap": generalization_gap,
                "latest_val_delta": val_delta,
                "latest_epoch_seconds": epoch_seconds,
                "latest_train_samples_per_sec": train_metrics["samples_per_sec"],
                "latest_val_samples_per_sec": val_metrics["samples_per_sec"] if val_metrics else None,
                "latest_grad_norm_avg": train_metrics["grad_norm_avg"],
                "latest_grad_norm_max": train_metrics["grad_norm_max"],
                "latest_lr": latest_lr,
                "best_score": best_val,
                "best_epoch": best_epoch,
                "best_checkpoint": str(best_path),
                "run_dir": str(run_dir),
            },
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "normalization": train_dataset.stats,
            "history": history,
        },
        run_dir / "last.pt",
    )
    _write_json(run_dir / "history.json", history)
    _write_json(
        run_dir / "status.json",
        {
            "state": "completed",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "current_epoch": args.epochs,
            "total_epochs": args.epochs,
            "latest_train_l1": history[-1]["train_l1"] if history else None,
            "latest_val_l1": history[-1]["val_l1"] if history else None,
            "latest_gap": history[-1].get("generalization_gap") if history else None,
            "latest_val_delta": history[-1].get("val_delta") if history else None,
            "latest_epoch_seconds": history[-1].get("epoch_seconds") if history else None,
            "latest_train_samples_per_sec": history[-1].get("train_samples_per_sec") if history else None,
            "latest_val_samples_per_sec": history[-1].get("val_samples_per_sec") if history else None,
            "latest_grad_norm_avg": history[-1].get("grad_norm_avg") if history else None,
            "latest_grad_norm_max": history[-1].get("grad_norm_max") if history else None,
            "latest_lr": history[-1].get("lr") if history else None,
            "best_score": best_val,
            "best_epoch": best_epoch,
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(run_dir / "last.pt"),
            "run_dir": str(run_dir),
        },
    )
    print(f"Saved run: {run_dir}")
    print(f"Best checkpoint: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
