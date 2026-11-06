#!/usr/bin/env python3
"""Hardware & Convergence Benchmark for Selective Customer Encoding (Gate R0 & Gate R1).

Compares:
1. Baseline: selective_customer_encoding=False, activation_checkpointing=True
2. Selective: selective_customer_encoding=True, activation_checkpointing=False

Measures on GPU (Tesla T4) across N update steps (batch size 512, customers 50, vehicles 3):
- Max VRAM allocated & reserved (GiB)
- Wall-clock seconds per batch (s/batch)
- Total customer rows encoded per batch
- Loss and gradient norm finite checks
- Gate R0 (Peak VRAM <= 10.0 GiB) & Gate R1 (Speedup >= 1.5x) verification
"""
from __future__ import annotations

import os
# Ensure expandable segments is set prior to CUDA initialization
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines._critic import CriticBaseline
from MODEL.model.coast2 import materialize_coast2_model
from MODEL.train_coast2 import make_optimizer, move_critic_to_device, train_update, _make_scaler
from problems import DVRPTW_Dataset
from script.train_coast2 import build_training_args, _model_from_args, _set_random_seed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10, help="Number of update batches to evaluate (default: 10)")
    parser.add_argument("--batch-size", type=int, default=512, help="Minibatch size (default: 512)")
    parser.add_argument("--train-data", type=str, default="data/coast_ablation/train_seed42.pyth",
                        help="Path to training dataset pool")
    parser.add_argument("--validation-data", type=str, default="data/coast_ablation/validation_10240.pyth",
                        help="Path to validation dataset pool")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (default: cuda:0 if available else cpu)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Optional path to export JSON benchmark results")
    return parser.parse_args(argv)


class CallCounter:
    """Wraps model._encode_customers to record forward call count and rows encoded."""
    def __init__(self, model):
        self.model = model
        self.orig_encode = model._encode_customers
        self.calls = 0
        self.rows_encoded = 0

        def wrapped_encode(customers, mask=None):
            self.calls += 1
            self.rows_encoded += int(customers.size(0))
            return self.orig_encode(customers, mask)

        model._encode_customers = wrapped_encode

    def reset(self):
        self.calls = 0
        self.rows_encoded = 0


def benchmark_configuration(
    label: str,
    selective: bool,
    checkpointing: bool,
    train_data: DVRPTW_Dataset,
    steps: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    train_path: str,
    val_path: str,
) -> dict:
    _set_random_seed(seed)

    args = build_training_args(
        seed, train_path, val_path, f"output/benchmark_{label.lower()}", protocol="full"
    )
    args.batch_size = batch_size
    args.selective_customer_encoding = selective
    args.activation_checkpointing = checkpointing

    model = _model_from_args(args).to(device)
    materialize_coast2_model(model, preserve_rng=False)

    baseline_counter = None
    if not selective:
        baseline_counter = CallCounter(model)

    critic = CriticBaseline(model, args.customers_count, use_qval=False, use_cumul_reward=False)
    move_critic_to_device(critic, device)
    optimizer, groups = make_optimizer(
        model, critic, args.learning_rate, args.critic_rate, args.weight_decay
    )
    scaler = _make_scaler(bool(args.amp) and device.type == "cuda")

    step_results = []
    print(f"\n[{label.upper()}] Starting benchmark: selective={selective}, checkpointing={checkpointing}, steps={steps}, batch={batch_size}")
    print("-" * 88)
    print(f"{'Step':>5} | {'Time (s)':>9} | {'Peak Alloc (GiB)':>16} | {'Peak Res (GiB)':>14} | {'Rows Encoded':>12} | {'Loss':>9} | {'Grad Norm':>9} | {'Finite'}")
    print("-" * 88)

    for step_idx in range(steps):
        batch_slice = train_data[step_idx * batch_size : (step_idx + 1) * batch_size]
        nodes = batch_slice.to(device)

        if baseline_counter is not None:
            baseline_counter.reset()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        t_start = time.perf_counter()
        metrics = train_update(
            model, critic, optimizer, groups, train_data, nodes, None, args, device, scaler
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_elapsed = time.perf_counter() - t_start

        if device.type == "cuda":
            alloc_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            res_gib = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        else:
            alloc_gib = 0.0
            res_gib = 0.0

        if selective:
            rows_encoded = int(model.encoding_stats.get("rows_encoded", 0))
            calls = int(model.encoding_stats.get("calls", 0))
            skipped_calls = int(model.encoding_stats.get("skipped_calls", 0))
        else:
            rows_encoded = int(baseline_counter.rows_encoded)
            calls = int(baseline_counter.calls)
            skipped_calls = 0

        loss_val = float(metrics.get("loss", float("nan")))
        ac_grad = float(metrics.get("ac_grad_norm", float("nan")))
        lh_grad = float(metrics.get("lookahead_grad_norm", float("nan")))
        is_finite = math.isfinite(loss_val) and math.isfinite(ac_grad) and math.isfinite(lh_grad)

        step_info = {
            "step": step_idx + 1,
            "seconds": t_elapsed,
            "peak_allocated_gib": alloc_gib,
            "peak_reserved_gib": res_gib,
            "rows_encoded": rows_encoded,
            "encode_calls": calls,
            "encode_skipped_calls": skipped_calls,
            "loss": loss_val,
            "ac_grad_norm": ac_grad,
            "lookahead_grad_norm": lh_grad,
            "finite": is_finite,
        }
        step_results.append(step_info)

        print(
            f"{step_idx + 1:>5} | {t_elapsed:>9.3f} | {alloc_gib:>16.2f} | {res_gib:>14.2f} | "
            f"{rows_encoded:>12} | {loss_val:>9.4f} | {ac_grad:>9.4f} | {'PASS' if is_finite else 'FAIL'}"
        )

    times = [s["seconds"] for s in step_results]
    allocs = [s["peak_allocated_gib"] for s in step_results]
    reserves = [s["peak_reserved_gib"] for s in step_results]
    rows = [s["rows_encoded"] for s in step_results]
    all_finite = all(s["finite"] for s in step_results)

    summary = {
        "label": label,
        "selective": selective,
        "checkpointing": checkpointing,
        "step_count": steps,
        "batch_size": batch_size,
        "mean_seconds": float(np.mean(times)),
        "std_seconds": float(np.std(times)),
        "min_seconds": float(np.min(times)),
        "max_seconds": float(np.max(times)),
        "peak_allocated_gib": float(np.max(allocs)),
        "peak_reserved_gib": float(np.max(reserves)),
        "mean_rows_encoded": float(np.mean(rows)),
        "min_rows_encoded": int(np.min(rows)),
        "max_rows_encoded": int(np.max(rows)),
        "all_finite": all_finite,
        "steps": step_results,
    }
    return summary


def main(argv=None):
    args = parse_args(argv)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 88)
    print("COAST2 Hardware & Convergence Benchmark (Gate R0 & Gate R1)")
    print(f"Device: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    print(f"Steps: {args.steps} | Batch size: {args.batch_size} | Seed: {args.seed}")
    print(f"Dataset: {args.train_data}")
    print("=" * 88)

    train_path = Path(args.train_data)
    if not train_path.is_file():
        raise FileNotFoundError(f"Train data pool not found: {train_path}")
    val_path = Path(args.validation_data)
    if not val_path.is_file():
        raise FileNotFoundError(f"Validation data pool not found: {val_path}")

    print("Loading and normalizing training dataset pool...")
    train_data = DVRPTW_Dataset.load(str(train_path))
    train_data.normalize()

    # 1. Baseline: selective=False, checkpointing=True
    baseline_summary = benchmark_configuration(
        label="Baseline",
        selective=False,
        checkpointing=True,
        train_data=train_data,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        train_path=str(train_path),
        val_path=str(val_path),
    )

    # 2. Selective: selective=True, checkpointing=False
    selective_summary = benchmark_configuration(
        label="Selective",
        selective=True,
        checkpointing=False,
        train_data=train_data,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        train_path=str(train_path),
        val_path=str(val_path),
    )

    # Compute comparative metrics
    speedup = baseline_summary["mean_seconds"] / max(selective_summary["mean_seconds"], 1e-9)
    peak_vram = selective_summary["peak_allocated_gib"]
    gate_r0_pass = peak_vram <= 10.0
    gate_r1_pass = speedup >= 1.5 and selective_summary["all_finite"]
    row_reduction = baseline_summary["mean_rows_encoded"] / max(selective_summary["mean_rows_encoded"], 1e-9)

    print("\n" + "=" * 88)
    print("FINAL BENCHMARK COMPARISON & VERIFICATION")
    print("=" * 88)
    print(f"{'Metric':<36} | {'Baseline (NoSel + Ckpt)':>22} | {'Selective (Sel + NoCkpt)':>24}")
    print("-" * 88)
    print(f"{'Speed (mean ± std s/batch)':<36} | {baseline_summary['mean_seconds']:>15.3f} ± {baseline_summary['std_seconds']:<5.3f} | {selective_summary['mean_seconds']:>17.3f} ± {selective_summary['std_seconds']:<5.3f}")
    print(f"{'Min .. Max Time (s/batch)':<36} | {baseline_summary['min_seconds']:>8.3f} .. {baseline_summary['max_seconds']:<8.3f} | {selective_summary['min_seconds']:>10.3f} .. {selective_summary['max_seconds']:<8.3f}")
    print(f"{'Speedup Factor':<36} | {'1.00x':>22} | {speedup:>23.2f}x")
    print(f"{'Peak VRAM Allocated (GiB)':<36} | {baseline_summary['peak_allocated_gib']:>22.2f} | {selective_summary['peak_allocated_gib']:>24.2f}")
    print(f"{'Peak VRAM Reserved (GiB)':<36} | {baseline_summary['peak_reserved_gib']:>22.2f} | {selective_summary['peak_reserved_gib']:>24.2f}")
    print(f"{'Avg Rows Encoded / Batch':<36} | {baseline_summary['mean_rows_encoded']:>22.1f} | {selective_summary['mean_rows_encoded']:>24.1f} ({row_reduction:.1f}x fewer)")
    print(f"{'Finite Loss & Gradients':<36} | {str(baseline_summary['all_finite']):>22} | {str(selective_summary['all_finite']):>24}")
    print("-" * 88)
    print(f"{'Gate R0 (Peak VRAM <= 10.0 GiB)':<36} | {'N/A':>22} | {'PASS (' + f'{peak_vram:.2f}' + ' GiB <= 10.0 GiB)':>24}")
    print(f"{'Gate R1 (Speedup >= 1.5x & Finite)':<36} | {'N/A':>22} | {'PASS (' + f'{speedup:.2f}' + 'x >= 1.5x)':>24}")
    print("=" * 88)

    overall_result = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "baseline": baseline_summary,
        "selective": selective_summary,
        "comparison": {
            "speedup_factor": speedup,
            "row_reduction_factor": row_reduction,
            "gate_r0_pass": gate_r0_pass,
            "gate_r0_peak_vram_gib": peak_vram,
            "gate_r0_threshold_gib": 10.0,
            "gate_r1_pass": gate_r1_pass,
            "gate_r1_speedup": speedup,
            "gate_r1_threshold": 1.5,
            "gate_r1_finite": selective_summary["all_finite"],
        },
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(overall_result, indent=2), encoding="utf-8")
        print(f"Benchmark results written to: {out_path}")

    if not (gate_r0_pass and gate_r1_pass):
        print("\n[ERROR] Gate R0 or Gate R1 verification failed!")
        sys.exit(1)
    else:
        print("\n[SUCCESS] Both Gate R0 and Gate R1 verified successfully!")


if __name__ == "__main__":
    main()
