#!/usr/bin/env python3
"""Print or explicitly execute the paired COAST/COAST2 ablation matrix."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_commands(data_dir="data/coast_ablation", output_root="output", device="cuda", seeds=(42, 123, 456)):
    data_dir = Path(data_dir)
    output_root = Path(output_root)
    commands = []
    for seed in seeds:
        for variant in ("f11", "f10", "f01", "cm"):
            commands.append([
                sys.executable,
                "script/train_coast_ablation.py",
                "--variant", variant,
                "--seed", str(int(seed)),
                "--data-dir", str(data_dir),
                "--output-root", str(output_root / "coast_ablation"),
                "--device", device,
            ])
        commands.append([
            sys.executable,
            "script/train_coast2.py",
            "--seed", str(int(seed)),
            "--data-dir", str(data_dir),
            "--output-root", str(output_root / "coast2"),
            "--protocol", "full",
            "--device", device,
        ])
    return commands


def validate_full_inputs(data_dir, seeds):
    data_dir = Path(data_dir).expanduser().resolve()
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Full COAST2 matrix requires {}".format(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = manifest.get("protocol", {})
    if (protocol.get("train_count_per_seed") != 512000
            or protocol.get("validation_count") != 10240
            or protocol.get("primary_test_count") != 1000):
        raise ValueError("Data manifest counts do not match the registered full protocol")
    if not manifest.get("disjoint_instance_audit", {}).get("passed"):
        raise ValueError("Data manifest lacks a passed disjoint-instance audit")
    for seed in seeds:
        train_entry = manifest.get("train", {}).get(str(seed))
        if not train_entry or not Path(train_entry["path"]).is_file():
            raise FileNotFoundError("Missing shared train pool for seed {}".format(seed))
    for name in ("validation", "primary_test"):
        entry = manifest.get(name)
        if not entry or not Path(entry["path"]).is_file():
            raise FileNotFoundError("Missing shared {} pool".format(name))
    if not protocol.get("primary_test_not_used_for_checkpoint_selection"):
        raise ValueError("Manifest does not reserve primary test for final evaluation")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/coast_ablation")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 123, 456))
    parser.add_argument("--execute", action="store_true", help="Run all 15 jobs; default is dry-run")
    args = parser.parse_args(argv)

    commands = build_commands(args.data_dir, args.output_root, args.device, args.seeds)
    print(json.dumps({"count": len(commands), "commands": commands}, indent=2))
    if not args.execute:
        print("Dry-run only. Pass --execute to start training.")
        return
    if args.device != "cuda":
        raise ValueError("The registered full matrix requires CUDA")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no matrix jobs were started")
    validate_full_inputs(args.data_dir, args.seeds)
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(PROJECT_ROOT) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    for command in commands:
        print("Starting: {}".format(" ".join(command)), flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
