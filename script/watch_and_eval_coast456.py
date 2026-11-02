#!/usr/bin/env python3
"""Watch for COAST seed 456 chkpt_best.pyth, evaluate immediately when ready."""

import time
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT = REPO_ROOT / "output/multiseed_g1_prebest/coast/seed456/chkpt_best.pyth"
RUNNER_LOG = REPO_ROOT / "output/multiseed_g1_prebest/coast_runner.log"

def main():
    print("Watcher started: waiting for COAST seed 456 chkpt_best.pyth...", flush=True)
    t0 = time.time()
    while True:
        if CKPT.exists() and CKPT.stat().st_size > 1000000:
            # Check if coast_runner has printed SUCCESS or validation finished
            if RUNNER_LOG.exists():
                txt = RUNNER_LOG.read_text()
                if "COAST-seed456" in txt and ("[DONE] COAST-seed456" in txt or "SUCCESS" in txt):
                    print("Found completed COAST seed 456 checkpoint!", flush=True)
                    break
        time.sleep(15)

    print(f"Waited {time.time()-t0:.1f}s. Launching eval_coast_seed456.py...", flush=True)
    res = subprocess.run([sys.executable, str(REPO_ROOT / "script/eval_coast_seed456.py")], text=True)
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()
