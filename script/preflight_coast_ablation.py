#!/usr/bin/env python3
"""Run a tiny train→best-checkpoint→strict-load→evaluation pass for all variants."""
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

from problems import DVRPTW_Dataset
from script.eval_coast_ablation import evaluate_dataset, load_model_from_study_checkpoint
from script.prepare_coast_ablation_data import generate_data_pools
from script.train_coast_ablation import build_training_args, run_training


def run_preflight(output_dir=None):
    temporary = None
    if output_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="coast-ablation-preflight-")
        root = Path(temporary.name)
    else:
        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)

    data_dir = root / "data"
    manifest = generate_data_pools(
        data_dir,
        seeds=(42,),
        train_count=2,
        validation_count=4,
        test_count=4,
        validation_seed=202701,
        test_seed=202702,
        dataset_kwargs={"cust_count": 5, "veh_count": 2},
    )
    validation = DVRPTW_Dataset.load(manifest["primary_test"]["path"])
    validation.normalize()
    results = {}
    for variant in ("f11", "f10", "f01", "cm"):
        run_dir = root / "runs" / variant / "seed42"
        args = build_training_args(
            variant,
            42,
            manifest["train"]["42"]["path"],
            manifest["validation"]["path"],
            run_dir,
            protocol="smoke",
            device="cpu",
        )
        run_training(args)
        model = load_model_from_study_checkpoint(run_dir / "chkpt_best.pyth", "cpu")
        costs = evaluate_dataset(model, validation, "cpu", batch_size=2)
        if not torch.isfinite(costs).all():
            raise RuntimeError("{} produced non-finite primary-test costs".format(variant))
        results[variant] = {
            "label": model.study_label if hasattr(model, "study_label") else variant,
            "finite_costs": int(torch.isfinite(costs).sum()),
            "num_instances": int(costs.numel()),
            "mean_cost": float(costs.mean()),
            "strict_checkpoint_load": True,
        }
        print("{}: smoke training and strict evaluation passed; mean={:.4f}".format(
            variant, results[variant]["mean_cost"]
        ))

    report = {"protocol": "tiny CPU smoke only; not a performance or efficacy result", "variants": results}
    if output_dir is not None:
        report_path = root / "preflight_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Preflight report: {}".format(report_path))
    if temporary is not None:
        temporary.cleanup()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None, help="Keep smoke checkpoints and report here")
    args = parser.parse_args(argv)
    run_preflight(args.output_dir)


if __name__ == "__main__":
    main()
