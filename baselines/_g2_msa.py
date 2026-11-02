"""Scenario-sampled rolling-horizon policy for the G2 comparison.

This is an explicit MSA-style adaptation: sample unrevealed requests from the
training generator, solve one reveal-constrained plan per sample, and choose
the active vehicle's first action by scenario consensus. The aggregation rule
is logged so the result is not confused with an exact implementation of a
published MSA variant.
"""

from __future__ import annotations

import time

import torch

from baselines._g2 import NearestPolicy, PolicyDecision, _visible_feasible_customer_ids
from externals._ort_rolling import solve_rolling_horizon
from problems import DVRPTW_Dataset


def _as_tuple(value, default):
    if value is None:
        value = default
    return tuple(value) if isinstance(value, list) else value


class ScenarioConsensusPolicy:
    name = "msa_consensus"

    def __init__(
        self,
        *,
        generation_config: dict,
        normalization: dict,
        seed: int,
        scenario_count: int,
        time_limit_ms: int,
        pending_cost: float,
        late_cost: float,
    ):
        self.config = dict(generation_config)
        self.normalization = dict(normalization)
        self.seed = int(seed)
        self.scenario_count = max(1, int(scenario_count))
        self.time_limit_ms = max(1, int(time_limit_ms))
        self.pending_cost = float(pending_cost)
        self.late_cost = float(late_cost)
        self._scenario_index = 0
    def _generator_kwargs(self, customer_count: int):
        config = self.config
        return {
            "batch_size": 1,
            "cust_count": int(customer_count),
            "veh_count": int(config.get("vehicles_count", config.get("veh_count", 3))),
            "veh_capa": float(config.get("veh_capa", 200)),
            "veh_speed": float(config.get("veh_speed", 1)),
            "cust_loc_range": _as_tuple(config.get("loc_range"), (0, 101)),
            "cust_dem_range": _as_tuple(config.get("dem_range"), (5, 41)),
            "horizon": int(config.get("horizon", 480)),
            "cust_dur_range": _as_tuple(config.get("dur_range"), (10, 31)),
            "tw_ratio": _as_tuple(config.get("tw_ratio"), 0.5),
            "cust_tw_range": _as_tuple(config.get("tw_range"), (30, 91)),
            "dod": _as_tuple(config.get("deg_of_dyna", config.get("dod", 0.5)), 0.5),
            "d_early_ratio": _as_tuple(
                config.get("appear_early_ratio"), 0.5
            ),
        }

    def _sample_future_nodes(self, env, scenario_idx: int):
        config = self.config
        horizon = float(config.get("horizon", 480))
        current_time = float(env.cur_veh[0, 0, 3].item()) * float(
            self.normalization["time_scale"]
        )
        early_end = max(1, int(horizon // 3))
        late_end = max(early_end + 1, int((2 * horizon) // 3))
        if current_time >= late_end:
            return env.nodes.new_empty((0, env.nodes.size(-1)))

        # The benchmark's request cardinality is part of the task definition.
        # Do not infer how many requests remain hidden from the test episode.
        customer_count = int(env.nodes.size(1) - 1)
        if env.init_cust_mask is not None:
            real_count = int((~env.init_cust_mask[0]).sum().item()) - 1
            customer_count = max(0, real_count)
        if customer_count <= 0:
            return env.nodes.new_empty((0, env.nodes.size(-1)))

        sample_seed = self.seed * 1_000_003 + scenario_idx * 97_409 + self._scenario_index
        self._scenario_index += 1
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(sample_seed)
            sampled = DVRPTW_Dataset.generate(
                **self._generator_kwargs(customer_count)
            ).nodes[0, 1:].clone()

        eligible = sampled[:, 6] > current_time
        future = sampled[eligible].clone()

        loc_scale = float(self.normalization["loc_scale"])
        loc_offset = float(self.normalization["loc_offset"])
        time_scale = float(self.normalization["time_scale"])
        demand_scale = float(self.normalization["demand_scale"])
        if loc_scale <= 0 or time_scale <= 0 or demand_scale <= 0:
            raise ValueError("MSA normalization scales must be positive.")

        future[:, :2] = (future[:, :2] - loc_offset) / loc_scale
        future[:, 2] = future[:, 2] / demand_scale
        future[:, 3:] = future[:, 3:] / time_scale
        return future.to(device=env.nodes.device, dtype=env.nodes.dtype)

    def act(self, env, deadline=None) -> PolicyDecision:
        if env.minibatch_size != 1:
            raise ValueError("MSA G2 policy requires batch_size=1.")

        visible = set(_visible_feasible_customer_ids(env))
        episode_deadline = (
            time.perf_counter() + self.time_limit_ms / 1000.0
            if deadline is None else deadline
        )
        budget_each_ms = max(1, self.time_limit_ms // self.scenario_count)
        votes = {}
        objectives = {}
        failures = 0
        solver_timeouts = 0
        scenario_runs = 0
        scenario_future_counts = []

        for scenario_idx in range(self.scenario_count):
            remaining_s = episode_deadline - time.perf_counter()
            if remaining_s <= 0:
                failures += self.scenario_count - scenario_idx
                break
            future_nodes = self._sample_future_nodes(env, scenario_idx)
            scenario_future_counts.append(int(future_nodes.size(0)))
            scenario_limit_ms = max(
                1, min(budget_each_ms, int(remaining_s * 1000))
            )
            plan = solve_rolling_horizon(
                env,
                time_limit_ms=scenario_limit_ms,
                pending_cost=self.pending_cost,
                late_cost=self.late_cost,
                extra_future_nodes=future_nodes,
                allowed_first_customer_ids=visible,
            )
            scenario_runs += 1
            solver_timeouts += int(bool(plan.info.get("timed_out")))
            if plan.info.get("status") == "no_requests":
                votes[0] = votes.get(0, 0) + 1
                objectives.setdefault(0, []).append(0.0)
                continue
            if plan.info.get("status") != "solution":
                failures += 1
                continue
            action = int(plan.action)
            votes[action] = votes.get(action, 0) + 1
            objectives.setdefault(action, []).append(float(plan.objective or 0.0))

        if not votes:
            fallback = NearestPolicy().act(env, deadline)
            fallback.info.update({
                "scenario_count": scenario_runs,
                "scenario_failures": failures,
                "scenario_timeouts": solver_timeouts,
                "scenario_future_counts": scenario_future_counts,
                "fallback": "nearest_no_scenario_solution",
                "timed_out": True,
            })
            return fallback

        selected = min(
            votes,
            key=lambda action: (
                -votes[action],
                sum(objectives.get(action, [0.0])) / max(1, len(objectives.get(action, []))),
                action,
            ),
        )
        return PolicyDecision(
            env.cur_veh_idx.new_tensor([[selected]]),
            {
                "scenario_count": scenario_runs,
                "scenario_failures": failures,
                "scenario_timeouts": solver_timeouts,
                "scenario_votes": votes,
                "aggregation": "majority_first_action_then_mean_plan_cost",
                "timed_out": solver_timeouts > 0 or time.perf_counter() > episode_deadline,
                "scenario_future_counts": scenario_future_counts,
            },
        )
