#!/usr/bin/env python3
"""Tiny COAST2 data→update→checkpoint→greedy-inference readiness check."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MODEL.coast2_checkpoint import load_coast2_model
from script.eval_coast2 import evaluate_dataset
from script.train_coast2 import build_training_args, run_training
from problems import DVRPTW_Dataset


def run_preflight(output_dir=None):
    temporary = None
    if output_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="coast2-preflight-")
        root = Path(temporary.name)
    else:
        root = Path(output_dir).expanduser().resolve()
        if root.exists() and any(root.iterdir()):
            raise FileExistsError("COAST2 preflight output directory must be empty: {}".format(root))
        root.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(20261105)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_data = DVRPTW_Dataset.generate(
        batch_size=2, cust_count=5, veh_count=2, dod=1.0, d_early_ratio=1.0
    )
    validation_data = DVRPTW_Dataset.generate(
        batch_size=4, cust_count=5, veh_count=2, dod=1.0, d_early_ratio=1.0
    )
    train_path = data_dir / "train.pyth"
    validation_path = data_dir / "validation.pyth"
    train_data.save(train_path)
    validation_data.save(validation_path)

    run_dir = root / "run"
    args = build_training_args(
        42, train_path, validation_path, run_dir,
        protocol="smoke", device="cpu",
    )
    run_training(args)

    checkpoint = run_dir / "chkpt_best.pyth"
    model = load_coast2_model(checkpoint, "cpu")
    costs_live = evaluate_dataset(
        model, DVRPTW_Dataset.load(validation_path), "cpu", batch_size=2
    )
    costs_checkpoint = evaluate_dataset(
        checkpoint, DVRPTW_Dataset.load(validation_path), "cpu", batch_size=2
    )
    if not torch.equal(costs_live, costs_checkpoint):
        raise RuntimeError("COAST2 in-memory/checkpoint greedy evaluation differs")

    predictor_params = sum(p.numel() for p in model.postdecision_lookahead.parameters())
    report = {
        "scope": "tiny CPU smoke; not a performance or efficacy result",
        "training_update_count": 1,
        "validation_instances": int(costs_live.numel()),
        "validation_mean_cost": float(costs_live.mean().item()),
        "greedy_checkpoint_parity": True,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "postdecision_predictor_parameters": predictor_params,
        "gate_parameters": model.gate_logit.numel(),
        "cuda_available": bool(torch.cuda.is_available()),
        "batch512_cuda_runtime_gate": "not run" if not torch.cuda.is_available() else "not run by preflight",
        "checkpoint": str(checkpoint),
    }
    report_path = root / "preflight_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if temporary is not None:
        temporary.cleanup()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None, help="Keep smoke artifacts here")
    args = parser.parse_args(argv)
    run_preflight(args.output_dir)


if __name__ == "__main__":
    main()
