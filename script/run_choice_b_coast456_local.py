#!/usr/bin/env python3
"""Run Choice B Warm-Start Fine-Tuning for COAST Seed 456 on Local Tesla T4.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_ROOT = REPO_ROOT / "output" / "multiseed_g1_choice_b"
POOL_ROOT = REPO_ROOT / "output" / "multiseed_g1" / "_shared_pools"
LOG_DIR = OUTPUT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

COAST_CKPT = REPO_ROOT / "output" / "Model_DVRPTWn50m3_260311-0727" / "chkpt_best.pyth"
seed = 456

def main():
    print(f"Resuming COAST Seed 456 on Tesla T4 at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    train_file = POOL_ROOT / f"seed{seed}" / "train.pyth"
    val_file = POOL_ROOT / f"seed{seed}" / "validation.pyth"
    out_dir = OUTPUT_ROOT / "coast" / f"seed{seed}"
    log_file = LOG_DIR / f"coast_seed{seed}.log"
    
    cmd = [
        sys.executable, "MODEL/train.py",
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
        "--layer-count", "2",
        "--head-count", "4",
        "--ff-size", "256",
        "--tanh-xplor", "10",
        "--edge-feat-size", "8",
        "--cust-k", "0",
        "--memory-size", "128",
        "--lookahead-hidden", "128",
        "--dropout", "0.1",
        "--ablation-profile", "vectra",
        "--epoch-count", "1",
        "--iter-count", "2000",
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
        "--model-weight", str(COAST_CKPT),
        "--train-data-file", str(train_file),
        "--test-data-file", str(val_file),
        "--output-dir", str(out_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    t0 = time.time()
    with open(log_file, "w", buffering=1) as log_f:
        p = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            log_f.write(line)
            if "ep=" in line or "BEST" in line or "val_mu=" in line or "Completed" in line or "Error" in line:
                print(f"[COAST-seed456] {line.strip()}", flush=True)
        ret = p.wait()
    dur = time.time() - t0
    status = "SUCCESS" if ret == 0 else f"FAILED (code {ret})"
    print(f"[COAST-seed456] Finished with {status} in {dur:.1f}s ({dur/60:.2f}m)", flush=True)

if __name__ == "__main__":
    main()
