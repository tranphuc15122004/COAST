#!/usr/bin/env python3
"""Greedy evaluation for in-memory or versioned COAST2 checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MODEL.coast2_checkpoint import load_coast2_model
from MODEL.model.coast2 import COAST2
from MODEL.train_coast2 import evaluate_costs
from problems import DVRPTW_Dataset


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_from_checkpoint(path, device="cpu"):
    return load_coast2_model(path, device)


def evaluate_dataset(model_or_checkpoint, dataset, device="cpu", batch_size=512,
                     pending_cost=2.0, late_cost=1.0):
    if len(dataset) <= 0:
        raise ValueError("Evaluation dataset is empty")
    model = (
        load_model_from_checkpoint(model_or_checkpoint, device)
        if isinstance(model_or_checkpoint, (str, Path))
        else model_or_checkpoint.to(torch.device(device))
    )
    if not isinstance(model, COAST2):
        raise TypeError("COAST2 evaluator accepts only COAST2 models/checkpoints")
    data = dataset
    data.normalize()
    return evaluate_costs(
        model, data, device=device, batch_size=batch_size,
        pending_cost=pending_cost, late_cost=late_cost,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--pending-cost", type=float, default=2.0)
    parser.add_argument("--late-cost", type=float, default=1.0)
    args = parser.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for COAST2 evaluation but is unavailable")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    if not checkpoint.is_file() or not data_path.is_file():
        raise FileNotFoundError("COAST2 checkpoint and evaluation data must exist")
    dataset = DVRPTW_Dataset.load(data_path)
    costs = evaluate_dataset(
        checkpoint, dataset, args.device, args.batch_size,
        args.pending_cost, args.late_cost,
    )
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "data": str(data_path),
        "data_sha256": _sha256(data_path),
        "count": int(costs.numel()),
        "mean_cost": float(costs.mean().item()),
        "std_cost": float(costs.std(unbiased=False).item()),
        "per_instance_cost": costs.cpu().tolist(),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("COAST2 evaluated {} instances; mean cost {:.5f}".format(report["count"], report["mean_cost"]))
    print("Saved {}".format(output))


if __name__ == "__main__":
    main()
