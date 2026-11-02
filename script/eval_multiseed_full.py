#!/usr/bin/env python3
"""Comprehensive Multi-Seed Evaluation Runner for G1 AAAI 2027 Rebuttal.

Evaluates checkpoints across seeds (42, 123, 456) on:
1. Held-out 500-instance In-Distribution reference split (data/test_sets/test_dvrptw_id_n50m3_500.pyth)
2. Large-scale PYTH benchmark 10,240 instances (datasets/dvrptw_n50m3_10240.pyth)

Extracts:
- normalized cost (mean +- std)
- raw replay cost (mean)
- skipped customers per instance
- time window violations per instance
"""

import os
import re
import sys
import json
import time
import subprocess
from pathlib import Path

def parse_infer_output(stdout_text):
    res = {}
    for line in stdout_text.splitlines():
        # Match: Inference done on 500 instance(s): mean=44.0749, std=3.8294, time=1.45s
        m_inf = re.search(r"Inference done on (\d+) instance\(s\):\s*mean=([0-9.-]+),\s*std=([0-9.-]+)", line)
        if m_inf:
            res["count"] = int(m_inf.group(1))
            res["mean"] = float(m_inf.group(2))
            res["std"] = float(m_inf.group(3))
        
        # Match: Cost summary: normalized_cost_mean=44.0749, raw_replay_cost_mean=1858.6404
        m_cost = re.search(r"normalized_cost_mean=([0-9.-]+),\s*raw_replay_cost_mean=([0-9.-]+)", line)
        if m_cost:
            res["norm_cost_mean"] = float(m_cost.group(1))
            res["raw_replay_cost_mean"] = float(m_cost.group(2))

        # Match: Skipped customers: total=8342 | per_instance=...
        m_skip = re.search(r"Skipped customers:\s*total=(\d+)", line)
        if m_skip:
            res["skipped_total"] = int(m_skip.group(1))

        # Match: Constraint violations: total_tw=4678 | total_appearance=0
        m_viol = re.search(r"Constraint violations:\s*total_tw=(\d+)", line)
        if m_viol:
            res["tw_viol_total"] = int(m_viol.group(1))

    if "count" in res and res["count"] > 0:
        if "skipped_total" in res:
            res["skipped_per_inst"] = round(res["skipped_total"] / res["count"], 4)
        if "tw_viol_total" in res:
            res["tw_viol_per_inst"] = round(res["tw_viol_total"] / res["count"], 4)
    return res

def run_cmd(cmd):
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - t0
    return proc.returncode, proc.stdout, elapsed

def get_infer_command(repo_root, method, ckpt, data_file, test_size):
    root = Path(repo_root)
    if method == "coast":
        return [
            sys.executable, str(root / "MODEL/infer.py"),
            "--model-weight", str(ckpt),
            "--data-file", str(data_file),
            "--no-verify-routes"
        ]
    elif method == "mardam":
        return [
            sys.executable, str(root / "MODEL/infer_mardam.py"),
            "--model-weight", str(ckpt),
            "--test-data-file", str(data_file),
            "--test-batch-size", str(test_size),
            "--no-verify-routes"
        ]
    elif method == "am":
        return [
            sys.executable, str(root / "third_party/baselines/RL4DVRPTW/am/infer.py"),
            "--problem-type", "dvrptw",
            "--customers-count", "50", "--vehicles-count", "3", "--veh-capa", "200", "--horizon", "480",
            "--pending-cost", "2.0", "--late-cost", "1.0",
            "--data-file", str(data_file),
            "--model-weight", str(ckpt),
            "--layer-count", "5", "--head-count", "8", "--ff-size", "256", "--model-size", "128",
            "--no-verify-routes"
        ]
    elif method == "polynet":
        return [
            sys.executable, str(root / "third_party/baselines/RL4DVRPTW/polynet/infer.py"),
            "--problem-type", "dvrptw",
            "--customers-count", "50", "--vehicles-count", "3", "--veh-capa", "200", "--horizon", "480",
            "--pending-cost", "2.0", "--late-cost", "1.0",
            "--data-file", str(data_file),
            "--model-weight", str(ckpt),
            "--layer-count", "5", "--head-count", "8", "--ff-size", "256", "--model-size", "128",
            "--no-verify-routes"
        ]
    else:
        raise ValueError(f"Unknown method {method}")

def main():
    repo_root = sys.argv[1] if len(sys.argv) > 1 else "/content"
    out_file = sys.argv[2] if len(sys.argv) > 2 else f"{repo_root}/output/multiseed_evaluation_full.json"
    root = Path(repo_root)

    test_500 = root / "data/test_sets/test_dvrptw_id_n50m3_500.pyth"
    test_10240 = root / "datasets/dvrptw_n50m3_10240.pyth"

    runs = [
        # COAST
        {"method": "coast", "seed": 42, "ckpt": root / "output/multiseed_g1_prebest/coast/seed42/chkpt_best.pyth"},
        {"method": "coast", "seed": 123, "ckpt": root / "output/multiseed_g1_prebest/coast/seed123/chkpt_best.pyth"},
        {"method": "coast", "seed": 456, "ckpt": root / "output/multiseed_g1_prebest/coast/seed456/chkpt_best.pyth"},
        # MARDAM
        {"method": "mardam", "seed": 42, "ckpt": root / "output/multiseed_g1_prebest/mardam/seed42/chkpt_best.pyth"},
        {"method": "mardam", "seed": 123, "ckpt": root / "output/multiseed_g1_prebest/mardam/seed123/chkpt_best.pyth"},
        {"method": "mardam", "seed": 456, "ckpt": root / "output/multiseed_g1_prebest/mardam/seed456/chkpt_best.pyth"},
        # AM
        {"method": "am", "seed": 42, "ckpt": root / "output/multiseed_g1_choice_b/am/seed42/chkpt_best.pyth"},
        {"method": "am", "seed": 123, "ckpt": root / "output/multiseed_g1_choice_b/am/seed123/chkpt_best.pyth"},
        {"method": "am", "seed": 456, "ckpt": root / "output/multiseed_g1_choice_b/am/seed456/chkpt_best.pyth"},
        # PolyNet
        {"method": "polynet", "seed": 42, "ckpt": root / "output/multiseed_g1_choice_b/polynet/seed42/chkpt_best.pyth"},
        {"method": "polynet", "seed": 123, "ckpt": root / "output/multiseed_g1_choice_b/polynet/seed123/chkpt_best.pyth"},
        {"method": "polynet", "seed": 456, "ckpt": root / "output/multiseed_g1_choice_b/polynet/seed456/chkpt_best.pyth"},
    ]

    results = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}

    print(f"=== Starting Multi-Seed Evaluation ===")
    print(f"Repo Root: {repo_root}")
    print(f"Test 500: {test_500} (exists={test_500.exists()})")
    print(f"Test 10240: {test_10240} (exists={test_10240.exists()})")

    for r in runs:
        key = f"{r['method']}_seed{r['seed']}"
        ckpt = r['ckpt']
        if not ckpt.exists():
            print(f"[{key}] Checkpoint not found: {ckpt} (skipping)")
            continue

        if key not in results:
            results[key] = {
                "method": r["method"],
                "seed": r["seed"],
                "checkpoint": str(ckpt)
            }

        # 1. Evaluate on test_500
        if "test_500" not in results[key] and test_500.exists():
            print(f"\n[{key}] Running on Test 500...")
            cmd = get_infer_command(repo_root, r["method"], ckpt, test_500, 500)
            ret, out, elapsed = run_cmd(cmd)
            if ret == 0:
                metrics = parse_infer_output(out)
                metrics["time_sec"] = round(elapsed, 2)
                results[key]["test_500"] = metrics
                print(f"[{key}] Test 500: mean={metrics.get('mean')}, std={metrics.get('std')}, skip/inst={metrics.get('skipped_per_inst')}, tw/inst={metrics.get('tw_viol_per_inst')} in {elapsed:.2f}s")
            else:
                print(f"[{key}] Test 500 FAILED:\n{out[-500:]}")

        # 2. Evaluate on test_10240
        if "test_10240" not in results[key] and test_10240.exists():
            print(f"\n[{key}] Running on Test 10,240...")
            cmd = get_infer_command(repo_root, r["method"], ckpt, test_10240, 10240)
            ret, out, elapsed = run_cmd(cmd)
            if ret == 0:
                metrics = parse_infer_output(out)
                metrics["time_sec"] = round(elapsed, 2)
                results[key]["test_10240"] = metrics
                print(f"[{key}] Test 10240: mean={metrics.get('mean')}, std={metrics.get('std')}, skip/inst={metrics.get('skipped_per_inst')}, tw/inst={metrics.get('tw_viol_per_inst')} in {elapsed:.2f}s")
            else:
                print(f"[{key}] Test 10240 FAILED:\n{out[-500:]}")

        # Save intermediate
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n=== Evaluation Finished! Saved to {out_file} ===")

if __name__ == "__main__":
    main()
