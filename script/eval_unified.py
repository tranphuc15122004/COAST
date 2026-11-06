#!/usr/bin/env python3
"""
Unified evaluation script for COAST and all internal ablation baselines.
Usage:
    python script/eval_unified.py \
        --test-data data/test_sets/test_dvrptw_n50m3_id_1000.pyth \
        --models-dir output/ablation \
        --output output/eval_results/in_dist.json \
        --seeds 42,123,456,789,1024
"""
import torch, os, sys, json, argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from MODEL.model import VECTRA
from problems import DVRPTW_Environment


def load_vectra_from_checkpoint(ckpt_path, device):
    """Load a VECTRA model from checkpoint, auto-detecting config."""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Try to read args from the args.json sibling file
    args = {}
    args_path = os.path.join(os.path.dirname(ckpt_path), "..", "args.json")
    if os.path.exists(args_path):
        with open(args_path) as f:
            args = json.load(f)
    
    learner = VECTRA(
        cust_feat_size=7,
        veh_state_size=4,
        model_size=args.get("model_size", 128),
        layer_count=args.get("layer_count", 2),
        head_count=args.get("head_count", 4),
        ff_size=args.get("ff_size", 256),
        tanh_xplor=args.get("tanh_xplor", 10),
        greedy=True,
        edge_feat_size=args.get("edge_feat_size", 8),
        cust_k=args.get("cust_k"),
        memory_size=args.get("memory_size", 128),
        lookahead_hidden=args.get("lookahead_hidden", 128),
        dropout=args.get("dropout", 0.1),
        use_edge_features=args.get("use_edge_features", True),
        use_memory=args.get("use_memory", True),
        use_ownership=args.get("use_ownership", True),
        use_lookahead=args.get("use_lookahead", True),
        fusion_mode=args.get("fusion_mode", "mlp"),
        linear_fusion_weights=args.get("linear_fusion_weights", (1.0, 1.0, 1.0)),
    )
    learner.load_state_dict(checkpoint["model"], strict=False)
    learner.to(device)
    learner.eval()
    return learner


load_coast_from_checkpoint = load_vectra_from_checkpoint


def evaluate_model(learner, data, device, env_params=(2, 1)):
    """Run one evaluation pass, return costs tensor (N,)."""
    env = DVRPTW_Environment(data, None, None, *env_params)
    env.nodes = env.nodes.to(device)
    if env.init_cust_mask is not None:
        env.init_cust_mask = env.init_cust_mask.to(device)
    
    with torch.no_grad():
        _, _, rewards = learner(env)
    costs = -torch.stack(rewards).sum(dim=0).squeeze(-1)
    return costs.cpu()


def compute_stats(costs_list):
    """Compute mean, std from list of per-seed cost tensors."""
    stacked = torch.stack(costs_list)
    instance_means = stacked.mean(dim=0)
    mean = instance_means.mean().item()
    std = instance_means.std().item()
    per_seed_means = stacked.mean(dim=1).tolist()
    per_seed_stds = stacked.std(dim=1).tolist()
    return {
        "mean": mean,
        "std": std,
        "per_seed_means": per_seed_means,
        "per_seed_stds": per_seed_stds,
    }


def main():
    parser = argparse.ArgumentParser(description="Unified COAST/ablation evaluation")
    parser.add_argument("--test-data", required=True, help="Path to .pyth test dataset")
    parser.add_argument("--models-dir", default="output/ablation",
                        help="Root directory containing {profile}/seed{N}/chkpt_best.pyth")
    parser.add_argument("--output", required=True, help="Path for results JSON")
    parser.add_argument("--seeds", default="42,123,456,789,1024",
                        help="Comma-separated seed list")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile", default=None, help="Specific profile to evaluate (e.g. vectra, b0). If not provided, runs all.")
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    data = torch.load(args.test_data, map_location="cpu", weights_only=False)
    print(f"Test data: {data.nodes.size()}, vehicles={data.veh_count}")
    
    profiles = {
        "COAST": "vectra",
        "B0-None": "b0",
        "B1-Memory": "b1",
        "B3-Look": "b3",
        "B5-Linear": "b5",
        "EdgeOff": "edgeoff",
    }
    
    if args.profile:
        profiles = {k: v for k, v in profiles.items() if v == args.profile}
        if not profiles:
            profiles = {args.profile.capitalize(): args.profile}

    all_results = {}
    for model_name, profile in profiles.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name} (profile={profile})")
        
        seed_costs = []
        for seed in seeds:
            ckpt_path = os.path.join(args.models_dir, profile, f"seed{seed}", "chkpt_best.pyth")
            if not os.path.exists(ckpt_path):
                print(f"  [SKIP] seed={seed}: not found")
                continue
            
            learner = load_vectra_from_checkpoint(ckpt_path, device)
            costs = evaluate_model(learner, data, device)
            seed_costs.append(costs)
            print(f"  seed={seed}: mean={costs.mean():.4f} std={costs.std():.4f}")
        
        if seed_costs:
            stats = compute_stats(seed_costs)
            all_results[model_name] = stats
            print(f"  => OVERALL ({len(seed_costs)} seeds): {stats['mean']:.4f} ± {stats['std']:.4f}")
        else:
            print(f"  [FAIL] No checkpoints found")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {args.output}")
    
    # Quick summary
    if all_results:
        print("\nRanking (by mean cost):")
        for name, stats in sorted(all_results.items(), key=lambda x: x[1]["mean"]):
            print(f"  {name:15s}: {stats['mean']:.4f} ± {stats['std']:.4f}")


if __name__ == "__main__":
    main()
