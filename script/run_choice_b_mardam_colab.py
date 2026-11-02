#!/usr/bin/env python3
"""Run Choice B Warm-Start Fine-Tuning for MARDAM on Colab L4.

Seeds: 42, 123, 456.
Budget: 1 epoch (1000 iterations * batch_size 512 = 512,000 instances).
Starting from: /content/output/Mardam_DVRPTWn50m3_260315-1328/chkpt_best.pyth
"""

import os
import sys
import time
import subprocess
from pathlib import Path

REPO_ROOT = Path("/content")
OUTPUT_ROOT = REPO_ROOT / "output" / "multiseed_g1_prebest"
POOL_ROOT = REPO_ROOT / "output" / "multiseed_g1" / "_shared_pools"
LOG_DIR = OUTPUT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MARDAM_CKPT = REPO_ROOT / "output" / "Mardam_DVRPTWn50m3_260315-1328" / "chkpt_ep240.pyth"
SEEDS = [42, 123, 456]

def run_job(name, cmd, log_path):
    print(f"\n{'='*70}\n[START] {name}\nLog: {log_path}\nCommand: {' '.join(cmd)}\n{'='*70}", flush=True)
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT}/script"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    with open(log_path, "w", buffering=1) as log_f:
        p = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            log_f.write(line)
            if "ep=" in line or "BEST" in line or "val_mu=" in line or "Completed" in line or "Error" in line or "Traceback" in line:
                print(f"[{name}] {line.strip()}", flush=True)
        ret = p.wait()
    duration = time.time() - t0
    status = "SUCCESS" if ret == 0 else f"FAILED (code {ret})"
    print(f"[DONE] {name} -> {status} in {duration:.1f}s ({duration/60:.2f}m)", flush=True)
    if ret != 0:
        raise RuntimeError(f"Job {name} failed with exit code {ret}. See {log_path}")

def main():
    print(f"Choice B MARDAM Colab Runner starting on NVIDIA L4 at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    assert MARDAM_CKPT.is_file(), f"MARDAM ckpt not found: {MARDAM_CKPT}"
    
    for seed in SEEDS:
        train_file = POOL_ROOT / f"seed{seed}" / "train.pyth"
        val_file = POOL_ROOT / f"seed{seed}" / "validation.pyth"
        out_dir = OUTPUT_ROOT / "mardam" / f"seed{seed}"
        log_file = LOG_DIR / f"mardam_seed{seed}.log"
        
        cmd = [
            sys.executable, "script/train_mardam.py",
            "--rng-seed", str(seed),
            "--problem-type", "dvrptw",
            "--customers-count", "50",
            "--vehicles-count", "3",
            "--veh-capa", "200",
            "--veh-speed", "1",
            "--horizon", "480",
            "--min-cust-count", "50",
            "--loc-range", "0", "101",
            "--dem-range", "5", "41",
            "--dur-range", "10", "31",
            "--tw-ratio", "0.25", "0.5", "0.75", "1.0",
            "--tw-range", "30", "91",
            "--deg-of-dyna", "0.1", "0.25", "0.5", "0.75",
            "--appear-early-ratio", "0.0", "0.5", "0.75", "1.0",
            "--pending-cost", "2.0",
            "--pend-cost-growth", "0.0",
            "--late-cost", "1.0",
            "--late-cost-growth", "0.0",
            "--speed-var", "0.0",
            "--late-prob", "0.0",
            "--slow-down", "0.0",
            "--late-var", "0.0",
            "--model-size", "128",
            "--layer-count", "3",
            "--head-count", "8",
            "--ff-size", "512",
            "--tanh-xplor", "10",
            "--dropout", "0.1",
            "--epoch-count", "1",
            "--iter-count", "1000",
            "--batch-size", "512",
            "--test-batch-size", "10240",
            "--learning-rate", "0.0001",
            "--critic-rate", "0.001",
            "--weight-decay", "0.0",
            "--max-grad-norm", "2.0",
            "--entropy-coef", "0.01",
            "--baseline-type", "critic",
            "--critic-use-qval",
            "--adv-norm",
            "--amp",
            "--num-workers", "4",
            "--pin-memory",
            "--strict-run-budget",
            "--model-weight", str(MARDAM_CKPT),
            "--train-data-file", str(train_file),
            "--test-data-file", str(val_file),
            "--output-dir", str(out_dir),
        ]
        run_job(f"MARDAM-seed{seed}", cmd, log_file)

    print(f"\n{'='*70}\n[ALL COMPLETED] MARDAM 3 seeds finished successfully on Colab!\n{'='*70}", flush=True)

if __name__ == "__main__":
    main()
