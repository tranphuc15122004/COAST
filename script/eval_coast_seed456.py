#!/usr/bin/env python3
"""Evaluate COAST seed 456 on Test 500 and Test 10240 and update multiseed_evaluation_full.json."""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from eval_multiseed_full import parse_infer_output, run_cmd, get_infer_command

def main():
    repo_root = Path(__file__).resolve().parent.parent
    ckpt_path = repo_root / "output/multiseed_g1_prebest/coast/seed456/chkpt_best.pyth"
    json_path = repo_root / "output/multiseed_g1_prebest/multiseed_evaluation_full.json"

    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} does not exist yet!")
        sys.exit(1)

    test_500 = repo_root / "data/test_sets/test_dvrptw_id_n50m3_500.pyth"
    test_10240 = repo_root / "datasets/dvrptw_n50m3_10240.pyth"

    print(f"=== Evaluating COAST Seed 456 ===")
    print(f"Checkpoint: {ckpt_path}")

    # Load existing results
    results = {}
    if json_path.exists():
        with open(json_path) as f:
            results = json.load(f)

    # 1. Test 500
    print("\n[coast_seed456] Running on Test 500...")
    cmd_500 = get_infer_command(repo_root, "coast", ckpt_path, test_500, 500)
    ret, out_500, elapsed_500 = run_cmd(cmd_500)
    parsed_500 = parse_infer_output(out_500)
    parsed_500["time_sec"] = round(elapsed_500, 2)
    print(f"[coast_seed456] Test 500: mean={parsed_500.get('mean')}, std={parsed_500.get('std')}, "
          f"skip/inst={parsed_500.get('skipped_per_inst')}, tw/inst={parsed_500.get('tw_viol_per_inst')} in {elapsed_500:.2f}s")

    # 2. Test 10240
    print("\n[coast_seed456] Running on Test 10,240...")
    cmd_10k = get_infer_command(repo_root, "coast", ckpt_path, test_10240, 10240)
    ret, out_10k, elapsed_10k = run_cmd(cmd_10k)
    parsed_10k = parse_infer_output(out_10k)
    parsed_10k["time_sec"] = round(elapsed_10k, 2)
    print(f"[coast_seed456] Test 10240: mean={parsed_10k.get('mean')}, std={parsed_10k.get('std')}, "
          f"skip/inst={parsed_10k.get('skipped_per_inst')}, tw/inst={parsed_10k.get('tw_viol_per_inst')} in {elapsed_10k:.2f}s")

    # Save into JSON
    results["coast_seed456"] = {
        "method": "coast",
        "seed": 456,
        "checkpoint": str(ckpt_path),
        "test_500": parsed_500,
        "test_10240": parsed_10k
    }

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved updated results to {json_path}")

    # Run summary
    summary_script = repo_root / "script/summarize_multiseed_results.py"
    if summary_script.exists():
        subprocess.run([sys.executable, str(summary_script), str(json_path)])

if __name__ == "__main__":
    main()
