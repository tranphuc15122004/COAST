"""Online rolling-horizon OR-Tools planner for the G2 evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch


DIMENSION_SCALE = 1_000_000
OBJECTIVE_WEIGHT_SCALE = 1_000
OBJECTIVE_SCALE = DIMENSION_SCALE * OBJECTIVE_WEIGHT_SCALE


@dataclass
class RoutingPlan:
    action: int
    objective: float | None
    info: dict


def _single_instance(env):
    if env.minibatch_size != 1:
        raise ValueError("Rolling-horizon G2 planner requires batch_size=1.")


def _known_customer_ids(env):
    padding = (
        torch.zeros_like(env.cust_mask[0])
        if env.init_cust_mask is None
        else env.init_cust_mask[0]
    )
    known = ~env.cust_mask[0] & ~env.served[0] & ~padding
    known[0] = False
    return known.nonzero(as_tuple=False).flatten().detach().cpu().tolist()


def solve_rolling_horizon(
    env,
    *,
    time_limit_ms: int,
    pending_cost: float,
    late_cost: float,
    extra_future_nodes: torch.Tensor | None = None,
    allowed_first_customer_ids: set[int] | None = None,
) -> RoutingPlan:
    """Solve the revealed suffix routes and return the active vehicle's next action.

    All coordinates, times, service durations, demands, and vehicle states are
    in the same normalized units consumed by COAST. Arc, lateness, and drop
    costs are integerized together at a fixed scale, preserving their relative
    weights in the normalized trajectory objective.

    ``extra_future_nodes`` is used only by the sampled-scenario policy. These
    sampled requests are assigned reveal-constrained ready times and are never
    returned as the current visible action.
    """
    _single_instance(env)
    started = time.perf_counter()
    deadline = started + max(1, int(time_limit_ms)) / 1000.0

    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError as exc:
        raise RuntimeError(
            "OR-Tools is required for rolling-horizon G2 policies. "
            "Install the project's optional OR-Tools dependency."
        ) from exc

    known_ids = _known_customer_ids(env)
    future_nodes = (
        env.nodes.new_empty((0, env.nodes.size(-1)))
        if extra_future_nodes is None
        else extra_future_nodes.to(device=env.nodes.device, dtype=env.nodes.dtype)
    )
    if not known_ids and future_nodes.size(0) == 0:
        return RoutingPlan(0, 0.0, {"status": "no_requests", "fallback": False})

    vehicle_count = int(env.veh_count)
    known_nodes = env.nodes[0, known_ids] if known_ids else env.nodes.new_empty((0, env.nodes.size(-1)))
    customer_nodes = torch.cat((known_nodes, future_nodes), dim=0)
    customer_external_ids = list(known_ids) + [None] * int(future_nodes.size(0))
    customer_count = int(customer_nodes.size(0))

    if customer_count == 0:
        return RoutingPlan(0, 0.0, {"status": "no_requests", "fallback": False})

    # Routing nodes are [one current-position start per vehicle, customers,
    # one depot end]. OR-Tools uses distinct internal start/end indices.
    starts = list(range(vehicle_count))
    customer_offset = vehicle_count
    depot_index = vehicle_count + customer_count
    start_positions = env.vehicles[0, :, :2].detach().cpu()
    depot_xy = env.nodes[0, 0, :2].detach().cpu()
    places = torch.cat(
        (start_positions, customer_nodes[:, :2].detach().cpu(), depot_xy[None, :]),
        dim=0,
    )
    demands = [0.0] * vehicle_count + customer_nodes[:, 2].detach().cpu().tolist() + [0.0]
    service = [0.0] * vehicle_count + customer_nodes[:, 5].detach().cpu().tolist() + [0.0]
    ready = [0.0] * vehicle_count + customer_nodes[:, 3].detach().cpu().tolist() + [0.0]
    due = [
        float(env.nodes[0, 0, 4].item())
    ] * vehicle_count + customer_nodes[:, 4].detach().cpu().tolist() + [
        float(env.nodes[0, 0, 4].item())
    ]
    # A sampled future request cannot be served before its sampled reveal time.
    if future_nodes.size(0):
        future_start = len(known_ids)
        for idx in range(future_start, customer_count):
            ready[customer_offset + idx] = max(
                ready[customer_offset + idx],
                float(customer_nodes[idx, 6].item()),
            )

    ends = [depot_index] * vehicle_count
    manager = pywrapcp.RoutingIndexManager(depot_index + 1, vehicle_count, starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    def node_id(index):
        return manager.IndexToNode(index)

    def distance_units(from_index, to_index):
        from_node = node_id(from_index)
        to_node = node_id(to_index)
        distance = torch.pairwise_distance(
            places[from_node][None, :], places[to_node][None, :]
        )
        return int(round(float(distance.item()) * OBJECTIVE_SCALE))

    distance_callback = routing.RegisterTransitCallback(distance_units)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback)

    def demand_units(from_index):
        return int(round(demands[node_id(from_index)] * DIMENSION_SCALE))

    demand_callback = routing.RegisterUnaryTransitCallback(demand_units)
    remaining_capacity = env.vehicles[0, :, 2].detach().cpu().tolist()
    vehicle_capacities = [
        int(round(max(0.0, capacity) * DIMENSION_SCALE))
        if not bool(env.veh_done[0, vehicle_id].item())
        else 0
        for vehicle_id, capacity in enumerate(remaining_capacity)
    ]
    routing.AddDimensionWithVehicleCapacity(
        demand_callback, 0, vehicle_capacities, True, "Capacity"
    )

    speed = float(env.veh_speed)
    if speed <= 0:
        raise ValueError("Vehicle speed must be positive for OR-Tools planning.")

    def time_units(from_index, to_index):
        from_node = node_id(from_index)
        to_node = node_id(to_index)
        distance = torch.pairwise_distance(
            places[from_node][None, :], places[to_node][None, :]
        )
        travel = float(distance.item()) / speed
        return int(round((service[from_node] + travel) * DIMENSION_SCALE))

    time_callback = routing.RegisterTransitCallback(time_units)
    horizon = max(float(env.nodes[0, 0, 4].item()), 1.0)
    slack_units = int(round(horizon * DIMENSION_SCALE))
    max_time_units = int(round(max(2.0 * horizon, (customer_count + 1) * horizon) * DIMENSION_SCALE))
    routing.AddDimension(time_callback, slack_units, max_time_units, False, "Time")
    time_dimension = routing.GetDimensionOrDie("Time")

    for vehicle_id in range(vehicle_count):
        start_index = routing.Start(vehicle_id)
        start_time = float(env.vehicles[0, vehicle_id, 3].item())
        time_dimension.CumulVar(start_index).SetValue(
            int(round(start_time * DIMENSION_SCALE))
        )
        end_index = routing.End(vehicle_id)
        time_dimension.SetCumulVarSoftUpperBound(
            end_index,
            int(round(horizon * DIMENSION_SCALE)),
            int(round(float(late_cost) * OBJECTIVE_WEIGHT_SCALE)),
        )

    for customer_offset_idx in range(customer_count):
        route_node = customer_offset + customer_offset_idx
        routing_index = manager.NodeToIndex(route_node)
        time_dimension.CumulVar(routing_index).SetMin(
            int(round(ready[route_node] * DIMENSION_SCALE))
        )
        time_dimension.SetCumulVarSoftUpperBound(
            routing_index,
            int(round(due[route_node] * DIMENSION_SCALE)),
            int(round(float(late_cost) * OBJECTIVE_WEIGHT_SCALE)),
        )
        routing.AddDisjunction(
            [routing_index], int(round(float(pending_cost) * OBJECTIVE_SCALE))
        )

    active_vehicle = int(env.cur_veh_idx[0, 0].item())
    if allowed_first_customer_ids is not None:
        start_index = routing.Start(active_vehicle)
        first_var = routing.NextVar(start_index)
        allowed = set()
        for idx, external_id in enumerate(customer_external_ids):
            if external_id is not None and external_id in allowed_first_customer_ids:
                allowed.add(manager.NodeToIndex(customer_offset + idx))
        # The active vehicle may always end its current route at the depot.
        allowed.add(routing.End(active_vehicle))
        solver = routing.solver()
        for idx, external_id in enumerate(customer_external_ids):
            internal_index = manager.NodeToIndex(customer_offset + idx)
            if internal_index >= 0 and internal_index not in allowed:
                solver.Add(first_var != internal_index)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.log_search = False
    remaining_ms = max(1, int((deadline - time.perf_counter()) * 1000))
    params.time_limit.seconds = remaining_ms // 1000
    params.time_limit.nanos = (remaining_ms % 1000) * 1_000_000

    assignment = routing.SolveWithParameters(params)
    elapsed = time.perf_counter() - started
    if assignment is None:
        return RoutingPlan(
            0, None,
            {"status": "no_incumbent", "timed_out": True, "elapsed_s": elapsed},
        )

    first_index = assignment.Value(routing.NextVar(routing.Start(active_vehicle)))
    if routing.IsEnd(first_index):
        action = 0
    else:
        solved_node = manager.IndexToNode(first_index)
        solved_customer_offset = solved_node - customer_offset
        if solved_customer_offset < 0 or solved_customer_offset >= customer_count:
            action = 0
        else:
            external_id = customer_external_ids[solved_customer_offset]
            action = int(external_id) if external_id is not None else 0

    return RoutingPlan(
        action,
        float(assignment.ObjectiveValue()) / OBJECTIVE_SCALE,
        {
            "status": "solution",
            "elapsed_s": elapsed,
            "timed_out": elapsed >= max(0.0, float(time_limit_ms) / 1000.0 - 0.005),
            "solver_objective": float(assignment.ObjectiveValue()) / OBJECTIVE_SCALE,
            "known_customers": len(known_ids),
            "sampled_future_customers": int(future_nodes.size(0)),
        },
    )


class RollingHorizonPolicy:
    name = "ort_rolling"

    def __init__(self, *, time_limit_ms: int, pending_cost: float, late_cost: float):
        self.time_limit_ms = int(time_limit_ms)
        self.pending_cost = float(pending_cost)
        self.late_cost = float(late_cost)

    def act(self, env, deadline=None):
        from baselines._g2 import NearestPolicy, PolicyDecision

        time_limit_ms = self.time_limit_ms
        if deadline is not None:
            remaining_ms = int((deadline - time.perf_counter()) * 1000)
            time_limit_ms = max(1, min(time_limit_ms, remaining_ms))
        plan = solve_rolling_horizon(
            env,
            time_limit_ms=time_limit_ms,
            pending_cost=self.pending_cost,
            late_cost=self.late_cost,
        )
        if plan.info.get("status") == "no_incumbent":
            fallback = NearestPolicy().act(env, deadline)
            fallback.info.update(plan.info)
            fallback.info["fallback"] = "nearest_no_incumbent"
            return fallback
        return PolicyDecision(
            env.cur_veh_idx.new_tensor([[plan.action]]),
            plan.info,
        )
