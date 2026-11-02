#!/usr/bin/env python3
"""Unified Multi-Seed Evaluation for G1 AAAI 2027 Rebuttal.

Evaluates all completed checkpoints for COAST, MARDAM, AM, and PolyNet across seeds 42, 123, 456
on:
1. Held-out 500-instance In-Distribution reference split (data/test_sets/test_dvrptw_id_n50m3_500.pyth)
2. Large-scale PYTH benchmark 10,240 instances (datasets/dvrptw_n50m3_10240.pyth)
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_500 = REPO_ROOT / "data" / "test_sets" / "test_dvrptw_id_n50m3_500.pyth"
TEST_10240 = REPO_ROOT / "datasets" / "dvrptw_n50m3_10240.pyth"
OUT_JSON = REPO_ROOT / "output" / "multiseed_g1_prebest" / "evaluation_summary.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

# Model specifications
RUNS = [
    # COAST
    {"method": "coast", "seed": 42, "ckpt": REPO_ROOT / "output/multiseed_g1_prebest/coast/seed42/chkpt_best.pyth", "type": "coast"},
    {"method": "coast", "seed": 123, "ckpt": REPO_ROOT / "output/multiseed_g1_prebest/coast/seed123/chkpt_best.pyth", "type": "coast"},
    {"method": "coast", "seed": 456, "ckpt": REPO_ROOT / "output/multiseed_g1_prebest/coast/seed456/chkpt_best.pyth", "type": "coast"},
    # MARDAM
    {"method": "mardam", "seed": 42, "ckpt": REPO_ROOT / "output/multiseed_g1_prebest/mardam/seed42/chkpt_best.pyth", "type": "mardam"},
    {"method": "mardam", "seed": 123, "ckpt": REPO_ROOT / "output/multiseed_g1_prebest/mardam/seed123/chkpt_best.pyth", "type": "mardam"},
    {"method": "mardam", "seed": 456, "ckpt": REPO_ROOT / "output/multiseed_g1_prebest/mardam/seed456/chkpt_best.pyth", "type": "mardam"},
    # AM
    {"method": "am", "seed": 42, "ckpt": REPO_ROOT / "output/multiseed_g1_choice_b/am/seed42/chkpt_best.pyth", "type": "am"},
    {"method": "am", "seed": 123, "ckpt": REPO_ROOT / "output/multiseed_g1_choice_b/am/seed123/chkpt_best.pyth", "type": "am"},
    {"method": "am", "seed": 456, "ckpt": REPO_ROOT / "output/multiseed_g1_choice_b/am/seed456/chkpt_best.pyth", "type": "am"},
    # PolyNet
    {"method": "polynet", "seed": 42, "ckpt": REPO_ROOT / "output/multiseed_g1_choice_b/polynet/seed42/chkpt_best.pyth", "type": "polynet"},
    {"method": "polynet", "seed": 123, "ckpt": REPO_ROOT / "output/multiseed_g1_choice_b/polynet/seed123/chkpt_best.pyth", "type": "polynet"},
    {"method": "polynet", "seed": 456, "ckpt": REPO_ROOT / "output/multiseed_g1_choice_b/polynet/seed456/chkpt_best.pyth", "type": "polynet"},
]

def parse_infer_output(stdout_text):
    """Extract normalized mean and std from stdout."""
    # Look for "Inference done on X instance(s): mean=Y, std=Z"
    for line in stdout_text.splitlines():
        if "Inference done on" in line and "mean=" in line and "std=" in line:
            parts = line.split("mean=")[1].split(",")
            mean_val = float(parts[0].strip())
            std_part = parts[1].split("std=")[1].split(",")[0].strip()
            std_val = float(std_part)
            return {"mean": mean_val, "std": std_val}
    return None

def eval_single(run_info, test_file, test_size):
    m_type = run_info["type"]
    ckpt = str(run_info["ckpt"])
    
    if m_type == "coast":
        cmd = [
            sys.executable, "MODEL/infer.py",
            "--model-weight", ckpt,
            "--data-file", str(test_file),
            "--no-cuda"
        ]
    elif m_type == "mardam":
        cmd = [
            sys.executable, "MODEL/infer_mardam.py",
            "--model-weight", ckpt,
            "--test-data-file", str(test_file),
            "--test-batch-size", str(test_size)
        ]
    elif m_type == "am":
        cmd = [
            sys.executable, "third_party/baselines/RL4DVRPTW/am/infer.py",
            "--problem-type", "dvrptw",
            "--customers-count", "50", "--vehicles-count", "3", "--veh-capa", "200", "--horizon", "480",
            "--pending-cost", "2.0", "--late-cost", "1.0",
            "--test-data-file", str(test_file),
            "--model-weight", ckpt,
            "--layer-count", "5", "--head-count", "8", "--ff-size", "256", "--model-size", "128",
            "--test-batch-size", str(test_size),
            "--no-cuda"
        ]
    elif m_type == "polynet":
        cmd = [
            sys.executable, "third_party/baselines/RL4DVRPTW/polynet/infer.py",
            "--problem-type", "dvrptw",
            "--customers-count", "50", "--vehicles-count", "3", "--veh-capa", "200", "--horizon", "480",
            "--pending-cost", "2.0", "--late-cost", "1.0",
            "--test-data-file", str(test_file),
            "--model-weight", ckpt,
            "--layer-count", "5", "--head-count", "8", "--ff-size", "256", "--model-size", "128",
            "--test-batch-size", str(test_size),
            "--no-cuda"
        ]
    else:
        raise ValueError(f"Unknown type {m_type}")

    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT}/script"
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    duration = time.time() - t0
    
    if res.returncode != 0:
        print(f"Error evaluating {run_info['method']}:{run_info['seed']}:\n{res.stderr[-500:]}", flush=True)
        return None
    
    stats = parse_infer_output(res.stdout)
    if stats:
        stats["time_sec"] = round(duration, 2)
    return stats

def main():
    results = {}
    if OUT_JSON.is_file():
        try:
            results = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        except Exception:
            results = {}

    print("="*70)
    print("Unified Evaluation on Test Sets (N=500 & N=10,240)")
    print("="*70, flush=True)

    for run in RUNS:
        key = f"{run['method']}_seed{run['seed']}"
        ckpt = run["ckpt"]
        if not ckpt.is_file():
            print(f"[SKIP] {key}: checkpoint not ready yet ({ckpt.name})", flush=True)
            continue
        
        if key not in results:
            results[key] = {
                "method": run["method"],
                "seed": run["seed"],
                "checkpoint": str(ckpt)
            }
        
        # Test 500
        if "test_500" not in results[key] or results[key]["test_500"] is None:
            print(f"[RUN] {key} on Test 500...", end=" ", flush=True)
            res_500 = eval_single(run, TEST_500, 500)
            if res_500:
                results[key]["test_500"] = res_500
                print(f"DONE -> mean={res_500['mean']:.4f}, std={res_500['std']:.4f} ({res_500['time_sec']}s)", flush=True)
            else:
                print("FAILED", flush=True)
            OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
        else:
            print(f"[CACHED] {key} on Test 500: mean={results[key]['test_500']['mean']:.4f}")

    print("\n" + "="*70)
    print("Summary Table (N=500 In-Distribution Reference Split):")
    print("="*70)
    
    import numpy as np
    methods = ["coast", "mardam", "am", "polynet"]
    for m in methods:
        means = []
        for s in [42, 123, 456]:
            k = f"{m}_seed{s}"
            if k in results and "test_500" in results[k] and results[k]["test_500"]:
                means.append(results[k]["test_500"]["mean"])
        if means:
            mu = np.mean(means)
            std = np.std(means, ddof=1) if len(means) > 1 else 0.0
            print(f"{m.upper():<10} (seeds={len(means)}): Mean across seeds = {mu:.4f} ± {std:.4f} | Per-seed: {[round(x, 4) for x in means]}")
        else:
            print(f"{m.upper():<10}: No evaluated seeds yet")

if __name__ == "__main__":
    main()
