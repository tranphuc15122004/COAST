"""Transparent online policies used by the AAAI 2027 G2 comparison."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import torch


@dataclass
class PolicyDecision:
    action: torch.Tensor
    info: dict


def _action(env, customer_id: int) -> torch.Tensor:
    return env.cur_veh_idx.new_tensor([[int(customer_id)]])


def _require_single_instance(env):
    if env.minibatch_size != 1:
        raise ValueError(
            "G2 policies require batch_size=1 so asynchronous dynamic episodes "
            "cannot affect each other's event clocks."
        )


def _visible_feasible_customer_ids(env) -> list[int]:
    _require_single_instance(env)
    valid = ~env.cur_veh_mask[0, 0]
    valid[0] = False
    return valid.nonzero(as_tuple=False).flatten().detach().cpu().tolist()


class NearestPolicy:
    name = "nearest"

    def act(self, env, deadline=None) -> PolicyDecision:
        candidates = _visible_feasible_customer_ids(env)
        if not candidates:
            return PolicyDecision(_action(env, 0), {"fallback": "depot_no_feasible_customer"})
        vehicle_id = int(env.cur_veh_idx[0, 0].item())
        vehicle_xy = env.vehicles[0, vehicle_id, :2]
        candidate_ids = torch.as_tensor(candidates, device=env.nodes.device)
        candidate_xy = env.nodes[0, candidate_ids, :2]
        distances = torch.pairwise_distance(
            candidate_xy, vehicle_xy.unsqueeze(0).expand_as(candidate_xy)
        )
        selected = int(candidate_ids[distances.argmin()].item())
        return PolicyDecision(_action(env, selected), {})


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

    def act(self, env, deadline=None) -> PolicyDecision:
        candidates = _visible_feasible_customer_ids(env)
        if not candidates:
            return PolicyDecision(_action(env, 0), {"fallback": "depot_no_feasible_customer"})
        return PolicyDecision(_action(env, self.rng.choice(candidates)), {})


class EarliestDueDatePolicy:
    name = "edd"

    def act(self, env, deadline=None) -> PolicyDecision:
        candidates = _visible_feasible_customer_ids(env)
        if not candidates:
            return PolicyDecision(_action(env, 0), {"fallback": "depot_no_feasible_customer"})
        due_times = env.nodes[0, candidates, 4]
        selected_offset = min(
            range(len(candidates)),
            key=lambda idx: (float(due_times[idx].item()), candidates[idx]),
        )
        return PolicyDecision(_action(env, candidates[selected_offset]), {})


def _route_cost(env, vehicle_id: int, route: list[int], late_cost: float) -> float:
    """Incremental normalized distance + lateness for one route suffix."""
    if bool(env.veh_done[0, vehicle_id].item()):
        return 0.0

    vehicle = env.vehicles[0, vehicle_id]
    current_xy = vehicle[:2]
    current_time = float(vehicle[3].item())
    speed = float(env.veh_speed)
    depot = env.nodes[0, 0]
    total_distance = 0.0
    total_late = 0.0

    for customer_id in route:
        customer = env.nodes[0, customer_id]
        distance = float(
            torch.pairwise_distance(current_xy[None], customer[:2][None]).item()
        )
        arrival = max(current_time + distance / speed, float(customer[3].item()))
        total_distance += distance
        total_late += max(0.0, arrival - float(customer[4].item()))
        current_time = arrival + float(customer[5].item())
        current_xy = customer[:2]

    # Every unfinished vehicle eventually returns to the depot. Include that
    # common suffix so insertion deltas reflect the actual DVRPTW objective.
    distance = float(torch.pairwise_distance(current_xy[None], depot[:2][None]).item())
    depot_arrival = current_time + distance / speed
    total_distance += distance
    total_late += max(0.0, depot_arrival - float(depot[4].item()))
    return total_distance + float(late_cost) * total_late


class RegretInsertionPolicy:
    """Regret-2 insertion over known, unserved requests; replans each event."""

    name = "regret2"

    def __init__(self, pending_cost: float, late_cost: float):
        self.pending_cost = float(pending_cost)
        self.late_cost = float(late_cost)

    def _known_unserved(self, env) -> list[int]:
        padding = (
            torch.zeros_like(env.cust_mask[0])
            if env.init_cust_mask is None
            else env.init_cust_mask[0]
        )
        known = ~env.cust_mask[0] & ~env.served[0] & ~padding
        known[0] = False
        return known.nonzero(as_tuple=False).flatten().detach().cpu().tolist()

    def _insert_best(self, env, routes, customer_id, deadline):
        demand = float(env.nodes[0, customer_id, 2].item())
        choices = []
        for vehicle_id in range(env.veh_count):
            if bool(env.veh_done[0, vehicle_id].item()):
                continue
            remaining = float(env.vehicles[0, vehicle_id, 2].item())
            route_demand = sum(
                float(env.nodes[0, node_id, 2].item())
                for node_id in routes[vehicle_id]
            )
            if route_demand + demand > remaining + 1e-8:
                continue

            before = _route_cost(env, vehicle_id, routes[vehicle_id], self.late_cost)
            for position in range(len(routes[vehicle_id]) + 1):
                if deadline is not None and time.perf_counter() >= deadline:
                    return choices, True
                candidate = list(routes[vehicle_id])
                candidate.insert(position, customer_id)
                after = _route_cost(env, vehicle_id, candidate, self.late_cost)
                net_delta = after - before - self.pending_cost
                choices.append((net_delta, vehicle_id, position))
        return choices, False

    def act(self, env, deadline=None) -> PolicyDecision:
        _require_single_instance(env)
        remaining_ids = self._known_unserved(env)
        routes = [[] for _ in range(env.veh_count)]
        timed_out = False

        while remaining_ids:
            selected = None
            selected_regret = -math.inf
            for customer_id in remaining_ids:
                choices, hit_deadline = self._insert_best(
                    env, routes, customer_id, deadline
                )
                if hit_deadline:
                    timed_out = True
                    break
                choices.sort(key=lambda item: (item[0], item[1], item[2]))
                if not choices or choices[0][0] >= 0:
                    continue  # Dropping has zero incremental cost here.
                second_best = choices[1][0] if len(choices) > 1 else 0.0
                regret = max(0.0, second_best - choices[0][0])
                candidate = (regret, -choices[0][0], -customer_id, choices[0])
                if selected is None or candidate[:3] > selected[:3]:
                    selected = candidate
                    selected_regret = regret

            if timed_out or selected is None:
                break

            _, _, _, (net_delta, vehicle_id, position) = selected
            customer_id = -selected[2]
            routes[vehicle_id].insert(position, customer_id)
            remaining_ids.remove(customer_id)

        active_vehicle = int(env.cur_veh_idx[0, 0].item())
        route = routes[active_vehicle]
        if route:
            return PolicyDecision(
                _action(env, route[0]),
                {"timed_out": timed_out, "planned_customers": sum(map(len, routes)),
                 "selected_regret": float(selected_regret)},
            )

        if timed_out:
            nearest = NearestPolicy().act(env, deadline)
            nearest.info.update({"timed_out": True, "fallback": "nearest"})
            return nearest
        return PolicyDecision(
            _action(env, 0),
            {"timed_out": False, "planned_customers": sum(map(len, routes)),
             "fallback": "depot_no_profitable_insertion"},
        )


def build_g2_policy(name: str, *, seed=0, pending_cost=2.0, late_cost=1.0):
    name = name.strip().lower()
    if name == "nearest":
        return NearestPolicy()
    if name == "random":
        return RandomPolicy(seed)
    if name == "edd":
        return EarliestDueDatePolicy()
    if name in ("regret2", "regret-2", "regret_insertion"):
        return RegretInsertionPolicy(pending_cost, late_cost)
    raise ValueError(f"Unknown non-optimizer G2 policy: {name}")
