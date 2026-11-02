#!/usr/bin/env python3
"""Compute Wilcoxon Signed-Rank Tests and Statistical Significance for G1 Multi-Seed Evaluation.

Evaluates all 12 checkpoints (COAST, MARDAM, AM, PolyNet across seeds 42, 123, 456)
on the standardized 500-instance benchmark (data/test_sets/test_dvrptw_id_n50m3_500.pyth).

Computes:
- Non-parametric Wilcoxon signed-rank test (W-statistic, p-value, z-statistic)
- Rank-biserial correlation (effect size r)
- Paired Student t-test (t-statistic, p-value)
- Paired Win / Tie / Loss counts and win rates
- Paired mean difference and 95% bootstrap confidence intervals (5,000 resamples)
"""

import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path("/home/tuantb/mardam")
DATA_500 = ROOT / "data/test_sets/test_dvrptw_id_n50m3_500.pyth"
OUT_DIR = ROOT / "output/multiseed_g1_prebest/wilcoxon"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    "coast_42": {
        "method": "coast", "seed": 42,
        "ckpt": ROOT / "output/multiseed_g1_prebest/coast/seed42/chkpt_best.pyth",
        "cmd": [sys.executable, str(ROOT / "MODEL/infer.py"), "--model-weight", str(ROOT / "output/multiseed_g1_prebest/coast/seed42/chkpt_best.pyth"), "--data-file", str(DATA_500), "--no-verify-routes"]
    },
    "coast_123": {
        "method": "coast", "seed": 123,
        "ckpt": ROOT / "output/multiseed_g1_prebest/coast/seed123/chkpt_best.pyth",
        "cmd": [sys.executable, str(ROOT / "MODEL/infer.py"), "--model-weight", str(ROOT / "output/multiseed_g1_prebest/coast/seed123/chkpt_best.pyth"), "--data-file", str(DATA_500), "--no-verify-routes"]
    },
    "coast_456": {
        "method": "coast", "seed": 456,
        "ckpt": ROOT / "output/multiseed_g1_prebest/coast/seed456/chkpt_best.pyth",
        "cmd": [sys.executable, str(ROOT / "MODEL/infer.py"), "--model-weight", str(ROOT / "output/multiseed_g1_prebest/coast/seed456/chkpt_best.pyth"), "--data-file", str(DATA_500), "--no-verify-routes"]
    },
    "mardam_42": {
        "method": "mardam", "seed": 42,
        "ckpt": ROOT / "output/multiseed_g1_prebest/mardam/seed42/chkpt_best.pyth",
        "cmd": [sys.executable, str(ROOT / "MODEL/infer_mardam.py"), "--model-weight", str(ROOT / "output/multiseed_g1_prebest/mardam/seed42/chkpt_best.pyth"), "--test-data-file", str(DATA_500), "--test-batch-size", "500", "--no-verify-routes"]
    },
    "mardam_123": {
        "method": "mardam", "seed": 123,
        "ckpt": ROOT / "output/multiseed_g1_prebest/mardam/seed123/chkpt_best.pyth",
        "cmd": [sys.executable, str(ROOT / "MODEL/infer_mardam.py"), "--model-weight", str(ROOT / "output/multiseed_g1_prebest/mardam/seed123/chkpt_best.pyth"), "--test-data-file", str(DATA_500), "--test-batch-size", "500", "--no-verify-routes"]
    },
    "mardam_456": {
        "method": "mardam", "seed": 456,
        "ckpt": ROOT / "output/multiseed_g1_prebest/mardam/seed456/chkpt_best.pyth",
        "cmd": [sys.executable, str(ROOT / "MODEL/infer_mardam.py"), "--model-weight", str(ROOT / "output/multiseed_g1_prebest/mardam/seed456/chkpt_best.pyth"), "--test-data-file", str(DATA_500), "--test-batch-size", "500", "--no-verify-routes"]
    },
    "am_42": {
        "method": "am", "seed": 42,
        "ckpt": ROOT / "output/multiseed_g1_choice_b/am/seed42/chkpt_best.pyth",
        "cmd": [sys.executable, str(ROOT / "third_party/baselines/RL4DVRPTW/am/infer.py"), "--problem-type", "dvrptw", "--customers-count", "50", "--vehicles-count", "3", "--veh-capa", "200", "--horizon", "480", "--pending-cost", "2.0", "--late-cost", "1.0", "--data-file", str(DATA_500), "--model-weight", str(ROOT / "output/multiseed_g1_choice_b/am/seed42/chkpt_best.pyth"), "--layer-count", "5", "--head-count", "8", "--ff-size", "256", "--model-size", "128", "--no-verify-routes"]
    },
    "am_123": {
        "method": "am", "seed": 123,
        "ckpt": ROOT / "output/multiseed_g1_choice_b/am/seed123/chkpt_best.pyth",
        "cmd": [sys.executable, str(ROOT / "third_party/baselines/RL4DVRPTW/am/infer.py"), "--problem-type", "dvrptw", "--customers-count", "50", "--vehicles-count", "3", "--veh-capa", "200", "--horizon", "480", "--pending-cost", "2.0", "--late-cost", "1.0", "--data-file", str(DATA_500), "--model-weight", str(ROOT / "output/multiseed_g1_choice_b/am/seed123/chkpt_best.pyth"), "--layer-count", "5", "--head-count", "8", "--ff-size", "256", "--model-size", "128", "--no-verify-routes"]
    },
    "am_456": {
        "method": "am", "seed": 456,
        "ckpt": ROOT / "output/multiseed_g1_choice_b/am/seed456/chkpt_best.pyth",
        "cmd": [sys.executable, str(ROOT / "third_party/baselines/RL4DVRPTW/am/infer.py"), "--problem-type", "dvrptw", "--customers-count", "50", "--vehicles-count", "3", "--veh-capa", "200", "--horizon", "480", "--pending-cost", "2.0", "--late-cost", "1.0", "--data-file", str(DATA_500), "--model-weight", str(ROOT / "output/multiseed_g1_choice_b/am/seed456/chkpt_best.pyth"), "--layer-count", "5", "--head-count", "8", "--ff-size", "256", "--model-size", "128", "--no-verify-routes"]
    },
    "polynet_42": {
        "method": "polynet", "seed": 42,
        "ckpt": ROOT / "output/multiseed_g1_choice_b/polynet/seed42/chkpt_best.pyth",
        "cmd": [sys.executable, str(ROOT / "third_party/baselines/RL4DVRPTW/polynet/infer.py"), "--problem-type", "dvrptw", "--customers-count", "50", "--vehicles-count", "3", "--veh-capa", "200", "--horizon", "480", "--pending-cost", "2.0", "--late-cost", "1.0", "--data-file", str(DATA_500), "--model-weight", str(ROOT / "output/multiseed_g1_choice_b/polynet/seed42/chkpt_best.pyth"), "--layer-count", "5", "--head-count", "8", "--ff-size", "256", "--model-size", "128", "--no-verify-routes"]
    },
    "polynet_123": {
        "method": "polynet", "seed": 123,
        "ckpt": ROOT / "output/multiseed_g1_choice_b/polynet/seed123/chkpt_best.pyth",
        "cmd": [sys.executable, str(ROOT / "third_party/baselines/RL4DVRPTW/polynet/infer.py"), "--problem-type", "dvrptw", "--customers-count", "50", "--vehicles-count", "3", "--veh-capa", "200", "--horizon", "480", "--pending-cost", "2.0", "--late-cost", "1.0", "--data-file", str(DATA_500), "--model-weight", str(ROOT / "output/multiseed_g1_choice_b/polynet/seed123/chkpt_best.pyth"), "--layer-count", "5", "--head-count", "8", "--ff-size", "256", "--model-size", "128", "--no-verify-routes"]
    },
    "polynet_456": {
        "method": "polynet", "seed": 456,
        "ckpt": ROOT / "output/multiseed_g1_choice_b/polynet/seed456/chkpt_best.pyth",
        "cmd": [sys.executable, str(ROOT / "third_party/baselines/RL4DVRPTW/polynet/infer.py"), "--problem-type", "dvrptw", "--customers-count", "50", "--vehicles-count", "3", "--veh-capa", "200", "--horizon", "480", "--pending-cost", "2.0", "--late-cost", "1.0", "--data-file", str(DATA_500), "--model-weight", str(ROOT / "output/multiseed_g1_choice_b/polynet/seed456/chkpt_best.pyth"), "--layer-count", "5", "--head-count", "8", "--ff-size", "256", "--model-size", "128", "--no-verify-routes"]
    },
}

def run_inferences():
    cost_dict = {}
    for key, cfg in CHECKPOINTS.items():
        json_path = OUT_DIR / f"{key}_results.json"
        if not json_path.exists():
            print(f"Running inference for {key}...")
            cmd = cfg["cmd"] + ["--save-json", str(json_path)]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if proc.returncode != 0:
                print(f"Error running {key}: {proc.stdout}")
                sys.exit(1)
        
        with open(json_path) as f:
            data = json.load(f)
            costs = np.array(data["costs"], dtype=np.float64)
            cost_dict[key] = costs
            print(f"[{key}] Loaded {len(costs)} costs: mean={costs.mean():.4f}, std={costs.std():.4f}")
    return cost_dict

def paired_bootstrap_ci(diffs, n_boot=5000, seed=2027):
    rng = np.random.RandomState(seed)
    boot_means = [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_boot)]
    return np.percentile(boot_means, [2.5, 97.5])

def analyze_comparisons(cost_dict):
    results = []
    
    # 1. Per-seed matched comparisons
    for seed in [42, 123, 456]:
        coast = cost_dict[f"coast_{seed}"]
        for comp in ["mardam", "am", "polynet"]:
            other = cost_dict[f"{comp}_{seed}"]
            diff = other - coast  # >0 means COAST is better (lower cost)
            
            # Wilcoxon signed-rank test
            w_res = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox")
            # Calculate rank-biserial correlation
            n = len(diff[diff != 0])
            total_rank_sum = n * (n + 1) / 2
            rank_biserial = 1.0 - (2.0 * (total_rank_sum - w_res.statistic) / total_rank_sum) if total_rank_sum > 0 else 0.0
            
            # Paired t-test
            t_res = stats.ttest_rel(other, coast, alternative="greater")
            
            # Win/Loss counts
            wins = int(np.sum(diff > 1e-5))
            losses = int(np.sum(diff < -1e-5))
            ties = len(diff) - wins - losses
            win_rate = wins / len(diff) * 100.0
            
            ci_low, ci_high = paired_bootstrap_ci(diff)
            
            res_item = {
                "comparison": f"COAST vs {comp.upper()} (Seed {seed})",
                "coast_mean": float(np.mean(coast)),
                "other_mean": float(np.mean(other)),
                "delta_mean": float(np.mean(diff)),
                "ci_95": [float(ci_low), float(ci_high)],
                "wilcoxon_stat": float(w_res.statistic),
                "wilcoxon_pvalue": float(w_res.pvalue),
                "rank_biserial": float(rank_biserial),
                "t_stat": float(t_res.statistic),
                "t_pvalue": float(t_res.pvalue),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "win_rate": float(win_rate)
            }
            results.append(res_item)
    
    # 2. Aggregated across all 3 seeds (pooled 3 x 500 = 1,500 paired episodes)
    for comp in ["mardam", "am", "polynet"]:
        coast_all = np.concatenate([cost_dict[f"coast_{s}"] for s in [42, 123, 456]])
        other_all = np.concatenate([cost_dict[f"{comp}_{s}"] for s in [42, 123, 456]])
        diff_all = other_all - coast_all
        
        w_res = stats.wilcoxon(diff_all, alternative="greater", zero_method="wilcox")
        n = len(diff_all[diff_all != 0])
        total_rank_sum = n * (n + 1) / 2
        rank_biserial = 1.0 - (2.0 * (total_rank_sum - w_res.statistic) / total_rank_sum) if total_rank_sum > 0 else 0.0
        
        t_res = stats.ttest_rel(other_all, coast_all, alternative="greater")
        wins = int(np.sum(diff_all > 1e-5))
        losses = int(np.sum(diff_all < -1e-5))
        ties = len(diff_all) - wins - losses
        win_rate = wins / len(diff_all) * 100.0
        
        ci_low, ci_high = paired_bootstrap_ci(diff_all)
        
        res_item = {
            "comparison": f"COAST vs {comp.upper()} (Pooled 3 Seeds, N=1500)",
            "coast_mean": float(np.mean(coast_all)),
            "other_mean": float(np.mean(other_all)),
            "delta_mean": float(np.mean(diff_all)),
            "ci_95": [float(ci_low), float(ci_high)],
            "wilcoxon_stat": float(w_res.statistic),
            "wilcoxon_pvalue": float(w_res.pvalue),
            "rank_biserial": float(rank_biserial),
            "t_stat": float(t_res.statistic),
            "t_pvalue": float(t_res.pvalue),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": float(win_rate)
        }
        results.append(res_item)
        
    return results

def main():
    print("=== Step 1: Running/Loading Inferences ===")
    costs = run_inferences()
    
    print("\n=== Step 2: Computing Wilcoxon and Statistical Tests ===")
    results = analyze_comparisons(costs)
    
    out_file = OUT_DIR / "wilcoxon_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_file}")
    
    # Print formatted markdown table
    print("\n### Bảng Kiểm Định Phi Tham Số Wilcoxon Signed-Rank Test:")
    print("| So sánh | Mean COAST | Mean Baseline | Paired $\\Delta$ | 95% Bootstrap CI | Tỷ lệ Thắng (Wins) | Wilcoxon $W$ | Wilcoxon $p$-value | Rank-Biserial $r$ |")
    print("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in results:
        p_str = f"< 10^{{-15}}" if r["wilcoxon_pvalue"] < 1e-15 else f"{r['wilcoxon_pvalue']:.4e}"
        print(f"| **{r['comparison']}** | {r['coast_mean']:.3f} | {r['other_mean']:.3f} | **+{r['delta_mean']:.3f}** | [{r['ci_95'][0]:.3f}, {r['ci_95'][1]:.3f}] | {r['wins']}/{r['wins']+r['losses']+r['ties']} ({r['win_rate']:.1f}%) | {r['wilcoxon_stat']:.0f} | **{p_str}** | +{r['rank_biserial']:.3f} |")

if __name__ == "__main__":
    main()
