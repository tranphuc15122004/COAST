#!/usr/bin/env python3
"""Common event-level G2 evaluation for COAST and online baselines.

Example:
    python script/eval_g2.py \
      --test-data data/test_sets/test_dvrptw_id_n50m3_500.pyth \
      --config-file output/Model_DVRPTWn50m3/args.json \
      --model-weight output/Model_DVRPTWn50m3/chkpt_best.pyth \
      --output-dir output/aaai2027/g2/id500

All methods use one DVRPTW_Environment instance per episode and the same
normalized trajectory reward. Per-instance rows are retained for paired
confidence intervals; decision-level latency is logged separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines._g2 import build_g2_policy
from baselines._g2_msa import ScenarioConsensusPolicy
from baselines._g2_neural import load_g2_neural_policy
from problems import DVRPTW_Dataset, DVRPTW_Environment


PER_INSTANCE_FIELDS = [
    "method", "budget_ms", "seed", "instance_id", "objective_cost",
    "distance", "late_time", "late_penalty", "pending_penalty",
    "decomposition_error", "active_customers", "served_customers",
    "dropped_customers", "service_rate", "total_demand", "served_demand",
    "decision_count", "latency_p50_s", "latency_p95_s",
    "policy_time_s", "timeout_count", "fallback_count", "invalid_actions",
]

SUMMARY_FIELDS = [
    "method", "budget_ms", "run_count", "instance_count", "mean_cost",
    "mean_cost_ci95_low", "mean_cost_ci95_high",
    "std_cost_across_instances", "std_seed_mean_cost",
    "mean_served_customers", "mean_service_rate",
    "latency_p50_s", "latency_p95_s", "mean_policy_time_s",
]

PAIRED_FIELDS = [
    "method", "budget_ms", "paired_instances", "mean_cost_difference_vs_coast",
    "ci95_low", "ci95_high", "relative_difference_percent",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-data", required=True, type=Path)
    parser.add_argument("--config-file", required=True, type=Path,
                        help="Explicit training args.json for checkpoint reconstruction.")
    parser.add_argument("--model-weight", required=True, action="append", type=Path,
                        help="COAST checkpoint; repeat once per training seed.")
    parser.add_argument("--am-model-weight", type=Path,
                        help="Optional reveal-aware AM checkpoint from third_party/baselines/RL4DVRPTW.")
    parser.add_argument("--am-config-file", type=Path,
                        help="Training args.json paired with --am-model-weight.")
    parser.add_argument("--polynet-model-weight", type=Path,
                        help="Optional reveal-aware PolyNet checkpoint from third_party/baselines/RL4DVRPTW.")
    parser.add_argument("--polynet-config-file", type=Path,
                        help="Training args.json paired with --polynet-model-weight.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/aaai2027/g2") / datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--methods",
        default="coast,nearest,random,edd,regret2,ort_rolling,msa_consensus",
        help="Comma-separated methods: coast, nearest, random, edd, regret2, ort_rolling, msa_consensus, am, polynet",
    )
    parser.add_argument("--time-budgets-ms", default="100,1000,5000")
    parser.add_argument("--random-seeds", default="42,123,456,789,1024")
    parser.add_argument("--msa-seeds", default="42,123,456")
    parser.add_argument("--msa-scenarios", type=int, default=5)
    parser.add_argument("--max-instances", type=int, default=0,
                        help="0 means every instance in the frozen test set.")
    parser.add_argument("--pending-cost", type=float, default=None)
    parser.add_argument("--late-cost", type=float, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True,
                        help="Warm up each selected policy on a synthetic training-distribution instance before timing.")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def _load_dataset(path: Path) -> DVRPTW_Dataset:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, DVRPTW_Dataset):
        return payload
    if isinstance(payload, dict) and "nodes" in payload:
        return DVRPTW_Dataset(**payload)
    raise ValueError(f"Unsupported DVRPTW test-data format: {path}")


def _clone_dataset(data: DVRPTW_Dataset) -> DVRPTW_Dataset:
    return DVRPTW_Dataset(
        data.veh_count,
        data.veh_capa,
        data.veh_speed,
        data.nodes.clone(),
        None if data.cust_mask is None else data.cust_mask.clone(),
    )


def _load_model(config_file: Path, model_weight: Path, device):
    from MODEL.infer import (
        _init_model,
        _load_model_weights_or_raise,
        parse_infer_args,
    )

    model_args = parse_infer_args([
        "--problem-type", "dvrptw",
        "--config-file", str(config_file),
        "--model-weight", str(model_weight),
        "--greedy",
    ])
    model_args.greedy = True
    learner = _init_model(
        model_args, DVRPTW_Dataset, DVRPTW_Environment, device
    )
    _load_model_weights_or_raise(str(model_weight), learner)
    learner.eval()
    return learner, model_args


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _quantile(values, probability):
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * float(probability)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - rank) + ordered[upper] * (rank - lower)


def _pre_step_cost(env, action_id):
    vehicle_id = int(env.cur_veh_idx[0, 0].item())
    vehicle = env.cur_veh
    destination = env.nodes.gather(
        1, action_id[:, :, None].expand(-1, -1, env.CUST_FEAT_SIZE)
    )
    distance = torch.pairwise_distance(
        vehicle[:, 0, :2], destination[:, 0, :2], keepdim=True
    )
    travel_time = distance / env.veh_speed
    arrival = torch.maximum(vehicle[:, :, 3] + travel_time, destination[:, :, 3])
    lateness = (arrival - destination[:, :, 4]).clamp_min_(0)
    return vehicle_id, float(distance.item()), float(lateness.item())


def _evaluate_episode(
    *,
    method: str,
    budget_ms: int,
    seed: int,
    instance_id: int,
    normalized_data: DVRPTW_Dataset,
    pending_cost: float,
    late_cost: float,
    device,
    learner=None,
    generation_config=None,
    normalization=None,
    msa_scenarios=5,
    decision_writer=None,
):
    instance_data = DVRPTW_Dataset(
        normalized_data.veh_count,
        normalized_data.veh_capa,
        normalized_data.veh_speed,
        normalized_data.nodes[instance_id:instance_id + 1].clone(),
        None if normalized_data.cust_mask is None else normalized_data.cust_mask[instance_id:instance_id + 1].clone(),
    )
    env = DVRPTW_Environment(instance_data, None, None, pending_cost, late_cost)
    env.nodes = env.nodes.to(device)
    if env.init_cust_mask is not None:
        env.init_cust_mask = env.init_cust_mask.to(device)
    env.reset()

    policy = None
    if method == "coast":
        learner._reset_memory(env)
        learner._reset_step_diagnostics()
    elif method in {"am", "polynet"}:
        pass
    elif method == "ort_rolling":
        from externals._ort_rolling import RollingHorizonPolicy
        policy = RollingHorizonPolicy(
            time_limit_ms=budget_ms,
            pending_cost=pending_cost,
            late_cost=late_cost,
        )
    elif method == "msa_consensus":
        policy = ScenarioConsensusPolicy(
            generation_config=generation_config,
            normalization=normalization,
            seed=seed,
            scenario_count=msa_scenarios,
            time_limit_ms=budget_ms,
            pending_cost=pending_cost,
            late_cost=late_cost,
        )
    else:
        policy = build_g2_policy(
            method, seed=seed, pending_cost=pending_cost, late_cost=late_cost
        )

    objective = 0.0
    distance_total = 0.0
    late_time_total = 0.0
    pending_penalty_total = 0.0
    latencies = []
    timeout_count = 0
    fallback_count = 0
    invalid_actions = 0
    step_index = 0
    neural_model_started = False

    with torch.no_grad():
        while not env.done:
            _sync(device)
            start = time.perf_counter()
            decision_info = {}
            if method == "coast":
                if env.new_customers:
                    learner._encode_customers(env.nodes, env.cust_mask)
                action, _ = learner.step(env)
            elif method in {"am", "polynet"}:
                if not neural_model_started:
                    learner.on_rollout_start(env)
                    neural_model_started = True
                action, _, _ = learner.step(env)
            else:
                deadline = (
                    start + budget_ms / 1000.0 if budget_ms > 0 else None
                )
                decision = policy.act(env, deadline)
                action = decision.action
                decision_info = decision.info
            _sync(device)
            latency = time.perf_counter() - start
            latencies.append(latency)

            vehicle_id, distance, late_time = _pre_step_cost(env, action)
            action_id = int(action[0, 0].item())
            action_invalid = bool(env.cur_veh_mask[0, 0, action_id].item())
            invalid_actions += int(action_invalid)

            reward = env.step(action)
            cost = -float(reward.item())
            distance_total += distance
            late_time_total += late_time
            pending_step = max(
                0.0, cost - distance - float(late_cost) * late_time
            )
            pending_penalty_total += pending_step
            objective += cost

            if decision_info.get("timed_out") or decision_info.get("status") == "no_incumbent":
                timeout_count += 1
            if decision_info.get("fallback"):
                fallback_count += 1

            if decision_writer is not None:
                decision_writer.write(json.dumps({
                    "method": method,
                    "budget_ms": int(budget_ms),
                    "seed": int(seed),
                    "instance_id": int(instance_id),
                    "step": int(step_index),
                    "vehicle_id": int(vehicle_id),
                    "action_id": action_id,
                    "action_invalid": action_invalid,
                    "latency_s": latency,
                    "info": decision_info,
                }, ensure_ascii=False, default=str) + "\n")
                decision_writer.flush()
            step_index += 1

    padding = (
        torch.zeros_like(env.served[0])
        if env.init_cust_mask is None
        else env.init_cust_mask[0]
    )
    active = ~padding
    active[0] = False
    served = env.served[0] & active
    active_count = int(active.sum().item())
    served_count = int(served.sum().item())
    total_demand = float((normalized_data.nodes[instance_id, :, 2] * normalization["demand_scale"]).sum().item())
    served_demand = float((normalized_data.nodes[instance_id, :, 2] * normalization["demand_scale"] * served.cpu()).sum().item())
    decomposition = distance_total + float(late_cost) * late_time_total + pending_penalty_total

    return {
        "method": method,
        "budget_ms": int(budget_ms),
        "seed": int(seed),
        "instance_id": int(instance_id),
        "objective_cost": objective,
        "distance": distance_total,
        "late_time": late_time_total,
        "late_penalty": float(late_cost) * late_time_total,
        "pending_penalty": pending_penalty_total,
        "decomposition_error": objective - decomposition,
        "active_customers": active_count,
        "served_customers": served_count,
        "dropped_customers": active_count - served_count,
        "service_rate": served_count / max(1, active_count),
        "total_demand": total_demand,
        "served_demand": served_demand,
        "decision_count": step_index,
        "latency_p50_s": _quantile(latencies, 0.50),
        "latency_p95_s": _quantile(latencies, 0.95),
        "policy_time_s": sum(latencies),
        "timeout_count": timeout_count,
        "fallback_count": fallback_count,
        "invalid_actions": invalid_actions,
        "decision_latencies": latencies,
    }


def _write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_mean_ci(values, replicates: int, seed: int):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = random.Random(seed)
    means = []
    count = len(values)
    for _ in range(max(100, int(replicates))):
        means.append(sum(values[rng.randrange(count)] for _ in range(count)) / count)
    return _quantile(means, 0.025), _quantile(means, 0.975)


def _aggregate(instance_rows, latency_rows, bootstrap_replicates):
    groups = defaultdict(list)
    for row in instance_rows:
        groups[(row["method"], int(row["budget_ms"]))].append(row)

    summary_rows = []
    for (method, budget_ms), rows in sorted(groups.items()):
        costs = [float(row["objective_cost"]) for row in rows]
        by_instance = defaultdict(list)
        by_seed = defaultdict(list)
        for row in rows:
            by_instance[int(row["instance_id"])].append(float(row["objective_cost"]))
            by_seed[int(row["seed"])].append(float(row["objective_cost"]))
        instance_means = [statistics.mean(values) for values in by_instance.values()]
        seed_means = [statistics.mean(values) for values in by_seed.values()]
        ci_low, ci_high = _bootstrap_mean_ci(
            instance_means,
            bootstrap_replicates,
            seed=3037 + budget_ms + sum(ord(char) for char in method),
        )
        served = [float(row["served_customers"]) for row in rows]
        service_rate = [float(row["service_rate"]) for row in rows]
        times = [float(row["policy_time_s"]) for row in rows]
        group_latencies = latency_rows[(method, budget_ms)]
        summary_rows.append({
            "method": method,
            "budget_ms": budget_ms,
            "run_count": len(rows),
            "instance_count": len({int(row["instance_id"]) for row in rows}),
            "mean_cost": statistics.mean(costs),
            "mean_cost_ci95_low": ci_low,
            "mean_cost_ci95_high": ci_high,
            "std_cost_across_instances": statistics.stdev(instance_means) if len(instance_means) > 1 else 0.0,
            "std_seed_mean_cost": statistics.stdev(seed_means) if len(seed_means) > 1 else 0.0,
            "mean_served_customers": statistics.mean(served),
            "mean_service_rate": statistics.mean(service_rate),
            "latency_p50_s": _quantile(group_latencies, 0.50),
            "latency_p95_s": _quantile(group_latencies, 0.95),
            "mean_policy_time_s": statistics.mean(times),
        })

    coast_by_instance = defaultdict(list)
    for row in groups.get(("coast", 0), []):
        coast_by_instance[int(row["instance_id"])].append(float(row["objective_cost"]))
    coast_means = {key: statistics.mean(values) for key, values in coast_by_instance.items()}

    paired_rows = []
    for (method, budget_ms), rows in sorted(groups.items()):
        if method == "coast":
            continue
        by_instance = defaultdict(list)
        for row in rows:
            by_instance[int(row["instance_id"])].append(float(row["objective_cost"]))
        common = sorted(set(by_instance) & set(coast_means))
        differences = [statistics.mean(by_instance[idx]) - coast_means[idx] for idx in common]
        if not differences:
            continue
        low, high = _bootstrap_mean_ci(
            differences,
            bootstrap_replicates,
            seed=2027 + budget_ms + sum(ord(char) for char in method),
        )
        base_mean = statistics.mean([coast_means[idx] for idx in common])
        mean_diff = statistics.mean(differences)
        paired_rows.append({
            "method": method,
            "budget_ms": budget_ms,
            "paired_instances": len(common),
            "mean_cost_difference_vs_coast": mean_diff,
            "ci95_low": low,
            "ci95_high": high,
            "relative_difference_percent": 100.0 * mean_diff / max(abs(base_mean), 1e-12),
        })

    return summary_rows, paired_rows


def _normalization_for_msa(data_before, data_after, loc_offset, loc_scale, time_scale, config):
    demand_scale = float(data_before.veh_capa)
    already_normalized = demand_scale <= 1.000001 and float(data_before.nodes[:, :, 2].max().item()) <= 1.000001
    if already_normalized:
        loc_range = config.get("loc_range", [0, 101])
        if isinstance(loc_range, list):
            loc_min, loc_max = float(loc_range[0]), float(loc_range[1])
        else:
            loc_min, loc_max = map(float, loc_range)
        loc_offset = loc_min
        loc_scale = max(loc_max - loc_min - 1.0, 1.0)
        time_scale = float(config.get("horizon", 480))
        demand_scale = float(config.get("veh_capa", 200))
    return {
        "loc_offset": float(loc_offset),
        "loc_scale": float(loc_scale),
        "time_scale": float(time_scale),
        "demand_scale": float(demand_scale),
    }


def _load_optional_neural_models(args, requested_methods, device):
    specs = {
        "am": (args.am_model_weight, args.am_config_file),
        "polynet": (args.polynet_model_weight, args.polynet_config_file),
    }
    learners = {}
    metadata = {}
    for method, (weight_path, config_path) in specs.items():
        if method not in requested_methods:
            continue
        if weight_path is None or config_path is None:
            raise ValueError(
                f"Method '{method}' requires both --{method}-model-weight and "
                f"--{method}-config-file."
            )
        if not weight_path.exists() or not config_path.exists():
            raise FileNotFoundError(
                f"Missing {method} checkpoint/config: {weight_path} / {config_path}"
            )
        learner, model_config = load_g2_neural_policy(
            method, config_path, weight_path, device
        )
        learners[method] = learner
        metadata[method] = {
            "checkpoint": str(weight_path.resolve()),
            "checkpoint_sha256": _sha256(weight_path),
            "config": str(config_path.resolve()),
            "config_sha256": _sha256(config_path),
            "adaptation": "reveal-aware step policy using the shared G2 simulator",
            "adapter_source": str(
                PROJECT_ROOT / "third_party" / "baselines" / "RL4DVRPTW"
                / method / "model.py"
            ),
            "adapter_source_sha256": _sha256(
                PROJECT_ROOT / "third_party" / "baselines" / "RL4DVRPTW"
                / method / "model.py"
            ),
            "config_model_size": model_config.get("model_size"),
            "config_layer_count": model_config.get("layer_count"),
            "config_head_count": model_config.get("head_count"),
            "checkpoint_match": learner._g2_checkpoint_match,
        }
    return learners, metadata


def _generate_warmup_data(config):
    def tuple_if_list(value):
        return tuple(value) if isinstance(value, list) else value

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2027)
        data = DVRPTW_Dataset.generate(
            batch_size=1,
            cust_count=int(config.get("customers_count", 50)),
            veh_count=int(config.get("vehicles_count", 3)),
            veh_capa=float(config.get("veh_capa", 200)),
            veh_speed=float(config.get("veh_speed", 1)),
            min_cust_count=config.get("min_cust_count"),
            cust_loc_range=tuple_if_list(config.get("loc_range", [0, 101])),
            cust_dem_range=tuple_if_list(config.get("dem_range", [5, 41])),
            horizon=int(config.get("horizon", 480)),
            cust_dur_range=tuple_if_list(config.get("dur_range", [10, 31])),
            tw_ratio=tuple_if_list(config.get("tw_ratio", [0.5])),
            cust_tw_range=tuple_if_list(config.get("tw_range", [30, 91])),
            dod=tuple_if_list(config.get("deg_of_dyna", [0.5])),
            d_early_ratio=tuple_if_list(config.get("appear_early_ratio", [0.5])),
        )
    raw_demand_scale = float(data.veh_capa)
    loc_offset = float(data.nodes[:, :, :2].min().item())
    normalized = _clone_dataset(data)
    loc_scale, time_scale = normalized.normalize()
    normalization = {
        "loc_offset": loc_offset,
        "loc_scale": float(loc_scale),
        "time_scale": float(time_scale),
        "demand_scale": raw_demand_scale,
    }
    return normalized, normalization


def _warmup_selected_methods(
    *, methods, budgets, coast_learners, neural_learners, generation_config,
    pending_cost, late_cost, device, msa_scenarios,
):
    warmup_data, warmup_normalization = _generate_warmup_data(generation_config)
    warmup_budget = min(budgets)
    for method in methods:
        instance = DVRPTW_Dataset(
            warmup_data.veh_count,
            warmup_data.veh_capa,
            warmup_data.veh_speed,
            warmup_data.nodes.clone(),
            None if warmup_data.cust_mask is None else warmup_data.cust_mask.clone(),
        )
        env = DVRPTW_Environment(instance, None, None, pending_cost, late_cost)
        env.nodes = env.nodes.to(device)
        if env.init_cust_mask is not None:
            env.init_cust_mask = env.init_cust_mask.to(device)
        env.reset()

        if method == "coast":
            learner = coast_learners[0][1]
            learner._reset_memory(env)
            learner._reset_step_diagnostics()
            if env.new_customers:
                learner._encode_customers(env.nodes, env.cust_mask)
            learner.step(env)
            continue
        if method in {"am", "polynet"}:
            learner = neural_learners[method]
            learner.on_rollout_start(env)
            learner.step(env)
            continue
        if method == "ort_rolling":
            from externals._ort_rolling import RollingHorizonPolicy

            policy = RollingHorizonPolicy(
                time_limit_ms=warmup_budget,
                pending_cost=pending_cost,
                late_cost=late_cost,
            )
        elif method == "msa_consensus":
            policy = ScenarioConsensusPolicy(
                generation_config=generation_config,
                normalization=warmup_normalization,
                seed=2027,
                scenario_count=msa_scenarios,
                time_limit_ms=warmup_budget,
                pending_cost=pending_cost,
                late_cost=late_cost,
            )
        else:
            policy = build_g2_policy(
                method, seed=2027, pending_cost=pending_cost, late_cost=late_cost
            )
        deadline = time.perf_counter() + warmup_budget / 1000.0
        policy.act(env, deadline)


def _parse_int_list(value):
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated integer list.")
    return values


def main(argv=None):
    args = parse_args(argv)
    if not args.test_data.exists():
        raise FileNotFoundError(args.test_data)
    if not args.config_file.exists():
        raise FileNotFoundError(args.config_file)
    for model_weight in args.model_weight:
        if not model_weight.exists():
            raise FileNotFoundError(model_weight)

    requested_methods = list(dict.fromkeys(
        part.strip().lower() for part in args.methods.split(",") if part.strip()
    ))
    allowed_methods = {
        "coast", "nearest", "random", "edd", "regret2", "ort_rolling",
        "msa_consensus", "am", "polynet",
    }
    unknown = sorted(set(requested_methods) - allowed_methods)
    if unknown:
        raise ValueError(f"Unknown G2 method(s): {', '.join(unknown)}")
    if "coast" not in requested_methods:
        raise ValueError("Include coast so paired comparisons can be computed.")
    budgets = _parse_int_list(args.time_budgets_ms)
    random_seeds = _parse_int_list(args.random_seeds)
    msa_seeds = _parse_int_list(args.msa_seeds)
    if any(budget <= 0 for budget in budgets):
        raise ValueError("All solver time budgets must be positive milliseconds.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    raw_data = _load_dataset(args.test_data)
    normalized_data = _clone_dataset(raw_data)
    loc_offset = float(normalized_data.nodes[:, :, :2].min().item())
    loc_scale, time_scale = normalized_data.normalize()
    with args.config_file.open(encoding="utf-8") as handle:
        generation_config = json.load(handle)
    normalization = _normalization_for_msa(
        raw_data, normalized_data, loc_offset, loc_scale, time_scale, generation_config
    )

    pending_cost = float(
        generation_config.get("pending_cost", 2.0)
        if args.pending_cost is None else args.pending_cost
    )
    late_cost = float(
        generation_config.get("late_cost", 1.0)
        if args.late_cost is None else args.late_cost
    )
    learners = []
    model_args = None
    for fallback_seed, model_weight in enumerate(args.model_weight):
        learner, model_args = _load_model(args.config_file, model_weight, device)
        match = re.search(r"seed[_-]?(\d+)", str(model_weight.parent), re.IGNORECASE)
        checkpoint_seed = int(match.group(1)) if match else fallback_seed
        learners.append((checkpoint_seed, learner, model_weight))
    neural_learners, neural_model_manifest = _load_optional_neural_models(
        args, requested_methods, device
    )

    instance_count = len(normalized_data)
    if args.max_instances > 0:
        instance_count = min(instance_count, args.max_instances)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    instance_rows = []
    latency_rows = defaultdict(list)
    per_instance_path = args.output_dir / "per_instance.csv"
    decision_path = args.output_dir / "decisions.jsonl"
    meta_path = args.output_dir / "manifest.json"

    manifest = {
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "method_set": requested_methods,
        "test_data": str(args.test_data.resolve()),
        "test_data_sha256": _sha256(args.test_data),
        "model_weights": [str(path.resolve()) for _, _, path in learners],
        "model_weight_sha256": [_sha256(path) for _, _, path in learners],
        "external_neural_models": neural_model_manifest,
        "config_file": str(args.config_file.resolve()),
        "config_sha256": _sha256(args.config_file),
        "simulator": "problems.DVRPTW_Environment",
        "simulator_sha256": _sha256(PROJECT_ROOT / "problems" / "_env_dtw.py"),
        "objective": {
            "pending_cost": pending_cost,
            "late_cost": late_cost,
            "primary_metric": "negative normalized DVRPTW_Environment trajectory return",
        },
        "normalization": normalization,
        "time_budgets_ms_per_event": budgets,
        "random_seeds": random_seeds,
        "msa_seeds": msa_seeds,
        "msa_scenarios": args.msa_scenarios,
        "bootstrap_replicates": args.bootstrap_replicates,
        "instance_count": instance_count,
        "batch_size": 1,
        "device": str(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "cpu": platform.processor() or platform.uname().processor or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "git_revision": _git_revision(),
        "model_profile": getattr(model_args, "ablation_profile", None),
        "model_greedy": True,
        "warmup": {
            "enabled": bool(args.warmup),
            "source": "synthetic instance generated with seed 2027 from COAST training config",
            "seed": 2027,
            "completed": False,
        },
    }
    try:
        from importlib.metadata import version
        manifest["ortools_version"] = version("ortools")
    except Exception:
        manifest["ortools_version"] = None
    if args.warmup:
        _warmup_selected_methods(
            methods=requested_methods,
            budgets=budgets,
            coast_learners=learners,
            neural_learners=neural_learners,
            generation_config=generation_config,
            pending_cost=pending_cost,
            late_cost=late_cost,
            device=device,
            msa_scenarios=args.msa_scenarios,
        )
        manifest["warmup"]["completed"] = True
    meta_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with decision_path.open("w", encoding="utf-8") as decision_file:
        with per_instance_path.open("w", newline="", encoding="utf-8") as instance_file:
            writer = csv.DictWriter(instance_file, fieldnames=PER_INSTANCE_FIELDS)
            writer.writeheader()
            for instance_id in range(instance_count):
                jobs = []
                if "coast" in requested_methods:
                    jobs.extend(("coast", 0, seed) for seed, _, _ in learners)
                if "nearest" in requested_methods:
                    jobs.append(("nearest", 0, 0))
                if "edd" in requested_methods:
                    jobs.append(("edd", 0, 0))
                if "random" in requested_methods:
                    jobs.extend(("random", 0, seed) for seed in random_seeds)
                budget_methods = [
                    method for method in ("regret2", "ort_rolling")
                    if method in requested_methods
                ]
                for method in budget_methods:
                    jobs.extend((method, budget, 0) for budget in budgets)
                if "msa_consensus" in requested_methods:
                    for seed in msa_seeds:
                        jobs.extend(("msa_consensus", budget, seed) for budget in budgets)
                for method in ("am", "polynet"):
                    if method in requested_methods:
                        jobs.append((method, 0, 0))

                for method, budget_ms, seed in jobs:
                    if method == "coast":
                        episode_learner = next(
                            (learner for checkpoint_seed, learner, _ in learners
                             if checkpoint_seed == seed),
                            None,
                        )
                    else:
                        episode_learner = neural_learners.get(method)
                    row = _evaluate_episode(
                        method=method,
                        budget_ms=budget_ms,
                        seed=seed,
                        instance_id=instance_id,
                        normalized_data=normalized_data,
                        pending_cost=pending_cost,
                        late_cost=late_cost,
                        device=device,
                        learner=episode_learner,
                        generation_config=generation_config,
                        normalization=normalization,
                        msa_scenarios=args.msa_scenarios,
                        decision_writer=decision_file,
                    )
                    latency_rows[(method, budget_ms)].extend(row.pop("decision_latencies"))
                    instance_rows.append(row)
                    writer.writerow(row)
                    instance_file.flush()
                    print(
                        f"[{instance_id + 1}/{instance_count}] {method} "
                        f"budget={budget_ms}ms seed={seed} "
                        f"cost={row['objective_cost']:.5f} "
                        f"p95={row['latency_p95_s']:.5f}s",
                        flush=True,
                    )

    summary_rows, paired_rows = _aggregate(
        instance_rows, latency_rows, args.bootstrap_replicates
    )
    _write_csv(args.output_dir / "summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(args.output_dir / "paired_vs_coast.csv", PAIRED_FIELDS, paired_rows)
    print(f"Saved G2 rows and manifest to {args.output_dir}")


if __name__ == "__main__":
    main()
