#!/usr/bin/env python3
"""Summarize Multi-Seed Evaluation Results for G1 Rebuttal."""

import os
import sys
import json
import numpy as np
from pathlib import Path

def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else "output/multiseed_g1_prebest/multiseed_evaluation_full.json"
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path) as f:
        data = json.load(f)

    methods = ["coast", "mardam", "am", "polynet"]
    seeds = [42, 123, 456]

    print("\n" + "="*80)
    print("DETAILED RESULTS PER SEED")
    print("="*80)
    print(f"{'Method':<10} | {'Seed':<5} | {'Test 500 (mean+-std)':<22} | {'Skip/inst':<10} | {'TW/inst':<10} | {'Test 10240 (mean+-std)':<24} | {'Skip/inst':<10} | {'TW/inst':<10}")
    print("-" * 120)

    stats = {m: {"test_500_means": [], "test_500_skips": [], "test_500_tws": [],
                 "test_10240_means": [], "test_10240_skips": [], "test_10240_tws": []} for m in methods}

    for m in methods:
        for s in seeds:
            k = f"{m}_seed{s}"
            if k in data:
                entry = data[k]
                t500 = entry.get("test_500", {})
                t10k = entry.get("test_10240", {})
                
                t500_str = f"{t500.get('mean', 'N/A')} +- {t500.get('std', 'N/A')}"
                t500_skip = t500.get("skipped_per_inst", "N/A")
                t500_tw = t500.get("tw_viol_per_inst", "N/A")

                t10k_str = f"{t10k.get('mean', 'N/A')} +- {t10k.get('std', 'N/A')}"
                t10k_skip = t10k.get("skipped_per_inst", "N/A")
                t10k_tw = t10k.get("tw_viol_per_inst", "N/A")

                print(f"{m:<10} | {s:<5} | {t500_str:<22} | {t500_skip:<10} | {t500_tw:<10} | {t10k_str:<24} | {t10k_skip:<10} | {t10k_tw:<10}")

                if "mean" in t500:
                    stats[m]["test_500_means"].append(t500["mean"])
                    stats[m]["test_500_skips"].append(t500.get("skipped_per_inst", 0))
                    stats[m]["test_500_tws"].append(t500.get("tw_viol_per_inst", 0))
                if "mean" in t10k:
                    stats[m]["test_10240_means"].append(t10k["mean"])
                    stats[m]["test_10240_skips"].append(t10k.get("skipped_per_inst", 0))
                    stats[m]["test_10240_tws"].append(t10k.get("tw_viol_per_inst", 0))

    print("\n" + "="*80)
    print("MULTI-SEED AGGREGATE SUMMARY (Across Seeds)")
    print("="*80)
    print(f"{'Method':<10} | {'Seeds':<5} | {'Test 500 (Seed Mean +- Std)':<28} | {'Test 10240 (Seed Mean +- Std)':<30} | {'Gap vs COAST (10k)':<20}")
    print("-" * 105)

    coast_10k_mean = np.mean(stats["coast"]["test_10240_means"]) if stats["coast"]["test_10240_means"] else None

    for m in methods:
        n_seeds_500 = len(stats[m]["test_500_means"])
        n_seeds_10k = len(stats[m]["test_10240_means"])

        if n_seeds_500 > 0:
            m_500 = np.mean(stats[m]["test_500_means"])
            s_500 = np.std(stats[m]["test_500_means"])
            res_500 = f"{m_500:.4f} +- {s_500:.4f}"
        else:
            res_500 = "N/A"

        if n_seeds_10k > 0:
            m_10k = np.mean(stats[m]["test_10240_means"])
            s_10k = np.std(stats[m]["test_10240_means"])
            res_10k = f"{m_10k:.4f} +- {s_10k:.4f}"
            if coast_10k_mean is not None:
                gap = m_10k - coast_10k_mean
                gap_pct = (gap / coast_10k_mean) * 100
                gap_str = f"+{gap:.4f} (+{gap_pct:.1f}%)" if gap > 0 else f"{gap:.4f}"
            else:
                gap_str = "N/A"
        else:
            res_10k = "N/A"
            gap_str = "N/A"

        print(f"{m:<10} | {n_seeds_500:<5} | {res_500:<28} | {res_10k:<30} | {gap_str:<20}")

    print("="*80 + "\n")

if __name__ == "__main__":
    main()
