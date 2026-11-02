#!/usr/bin/env python3
"""Run Choice B Warm-Start Fine-Tuning for PolyNet and AM on Colab L4.

Seeds: 42, 123, 456.
Budget:
- PolyNet: 50 iterations * 512 = 25,600 samples
- AM: 30 iterations * 256 = 7,680 samples
Starting from convergent best checkpoints:
- PolyNet: /content/third_party/baselines/RL4DVRPTW/data/_PolyNet/chkpt_best.pyth
- AM: /content/third_party/baselines/RL4DVRPTW/data/_AM/chkpt_best.pyth
"""

import os
import sys
import time
import subprocess
from pathlib import Path

REPO_ROOT = Path("/content")
BASELINE_ROOT = REPO_ROOT / "third_party" / "baselines" / "RL4DVRPTW"
OUTPUT_ROOT = REPO_ROOT / "output" / "multiseed_g1_choice_b"
SLICED_ROOT = OUTPUT_ROOT / "sliced_pools"
POOL_ROOT = REPO_ROOT / "output" / "multiseed_g1_b200" / "_shared_pools"
LOG_DIR = OUTPUT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

POLYNET_CKPT = BASELINE_ROOT / "data" / "_PolyNet" / "chkpt_best.pyth"
AM_CKPT = BASELINE_ROOT / "data" / "_AM" / "chkpt_best.pyth"

SEEDS = [42, 123, 456]

def run_job(name, cmd, cwd, log_path):
    print(f"\n{'='*70}\n[START] {name}\nLog: {log_path}\nCommand: {' '.join(cmd)}\n{'='*70}", flush=True)
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{BASELINE_ROOT}:{REPO_ROOT}/script"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    with open(log_path, "w", buffering=1) as log_f:
        p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            log_f.write(line)
            if "ep=" in line or "BEST" in line or "val_mu=" in line or "Completed" in line or "Error" in line or "Traceback" in line or "step " in line:
                print(f"[{name}] {line.strip()}", flush=True)
        ret = p.wait()
    duration = time.time() - t0
    status = "SUCCESS" if ret == 0 else f"FAILED (code {ret})"
    print(f"[DONE] {name} -> {status} in {duration:.1f}s ({duration/60:.2f}m)", flush=True)
    if ret != 0:
        raise RuntimeError(f"Job {name} failed with exit code {ret}. See {log_path}")

def main():
    print(f"Choice B Colab Runner starting on NVIDIA L4 at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    
    assert POLYNET_CKPT.is_file(), f"PolyNet ckpt not found: {POLYNET_CKPT}"
    assert AM_CKPT.is_file(), f"AM ckpt not found: {AM_CKPT}"
    
    # 1. PolyNet Seeds (50 iters * 512 = 25,600 samples)
    for seed in SEEDS:
        train_file = SLICED_ROOT / f"polynet_seed{seed}_train.pyth"
        val_file = POOL_ROOT / f"seed{seed}" / "validation.pyth"
        out_dir = OUTPUT_ROOT / "polynet" / f"seed{seed}"
        log_file = LOG_DIR / f"polynet_seed{seed}.log"
        
        cmd = [
            sys.executable, "-m", "polynet.train",
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
            "--layer-count", "5",
            "--head-count", "8",
            "--ff-size", "256",
            "--tanh-xplor", "10",
            "--dropout", "0.1",
            "--epoch-count", "1",
            "--iter-count", "50",
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
            "--resource-safe",
            "--model-weight", str(POLYNET_CKPT),
            "--train-data-file", str(train_file),
            "--test-data-file", str(val_file),
            "--output-dir", str(out_dir),
        ]
        run_job(f"PolyNet-seed{seed}", cmd, BASELINE_ROOT, log_file)

    # 2. AM Seeds (30 iters * 256 = 7,680 samples)
    for seed in SEEDS:
        train_file = SLICED_ROOT / f"am_seed{seed}_train.pyth"
        val_file = POOL_ROOT / f"seed{seed}" / "validation.pyth"
        out_dir = OUTPUT_ROOT / "am" / f"seed{seed}"
        log_file = LOG_DIR / f"am_seed{seed}.log"
        
        cmd = [
            sys.executable, "-m", "am.train",
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
            "--layer-count", "5",
            "--head-count", "8",
            "--ff-size", "256",
            "--tanh-xplor", "10",
            "--dropout", "0.1",
            "--epoch-count", "1",
            "--iter-count", "30",
            "--batch-size", "256",
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
            "--resource-safe",
            "--model-weight", str(AM_CKPT),
            "--train-data-file", str(train_file),
            "--test-data-file", str(val_file),
            "--output-dir", str(out_dir),
        ]
        run_job(f"AM-seed{seed}", cmd, BASELINE_ROOT, log_file)

    print(f"\n{'='*70}\n[ALL COMPLETED] PolyNet and AM 3 seeds finished successfully!\n{'='*70}", flush=True)

if __name__ == "__main__":
    main()
