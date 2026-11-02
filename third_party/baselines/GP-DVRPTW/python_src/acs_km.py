"""Python implementation of ACS-KM constructive heuristic.

Ports the core algorithms from the Java ACS-KM (Ant Colony System +
Kuhn-Munkres matching) to Python, using the Problem's own coordinate/speed/
service_time data so the constructed solutions are directly comparable with
GPHH evaluations inside soft_Simulation2.

Key algorithms:
  - Solomon I1 sequential insertion heuristic
  - KM (Hungarian) matching for route segment reassignment
  - Relocate + Exchange local search

Also provides ACS_DVRPTW_Simulation, an event-driven DVRPTW simulation that
replaces GP routing/sequencing policies with ACS-KM I1 insertion (online,
per-request-arrival) and route-order dispatch.
"""
from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

from python_src.sim.mod import Event, VehicleState
from python_src.sim.mod import soft_Simulation2
from python_src.sim.problem import Problem


DIST_CACHE: Dict[Tuple[int, int], float] = {}


def _dist(a, b) -> float:
    key = (id(a), id(b))
    cached = DIST_CACHE.get(key)
    if cached is not None:
        return cached
    d = math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
    DIST_CACHE[key] = d
    return d


class ACS_KM:
    """Construct vehicle routes using Solomon I1 insertion + KM matching.

    All geometry uses the Problem's own coordinates and parameters, so
    results can be fairly compared with GPHH-evaluated solutions.
    """

    def __init__(self, problem: Problem, params: Optional[dict] = None):
        self.problem = problem
        self.reqs = problem.requests
        self.depot = problem.depot
        self.req_map = {r.idx: r for r in self.reqs}
        self.capacity = problem.truck_capacity
        self.speed = problem.truck_speed

        p = params or {}
        self.mu = p.get("mu", 1.0)
        self.alfa1 = p.get("alfa1", 0.1)
        self.alfa2 = p.get("alfa2", 0.9)
        self.lambda_ = p.get("lambda", 2.0)
        self.max_vehicles = p.get("max_vehicles", problem.num_trucks)
        self.do_km = p.get("do_km", True)
        self.do_ls = p.get("do_ls", True)

    # ------------------------------------------------------------------
    # Route variable helpers
    # ------------------------------------------------------------------

    def _travel_time(self, a, b) -> float:
        return _dist(a, b) / self.speed

    def _compute_route_vars(
        self,
        routes: List[List[int]],
        begin_service: Dict[int, float],
        earliest_time: List[List[float]],
        latest_time: List[List[float]],
        start_tour: int = 0,
    ):
        for r in range(start_tour, len(routes)):
            tour = routes[r]
            et = earliest_time[r]
            lt = latest_time[r]
            n = len(tour)
            et.clear()
            lt.clear()
            et.extend([0.0] * n)
            lt.extend([0.0] * n)
            for i in range(n):
                city = tour[i]
                if i == 0:
                    et[i] = self.depot.open
                else:
                    prev = self._node(tour[i - 1])
                    cur = self._node(city)
                    v1 = cur.open
                    v2 = et[i - 1] + self._travel_time(prev, cur) + prev.service_time
                    et[i] = max(v1, v2)
            for i in range(n - 1, -1, -1):
                city = tour[i]
                cur = self._node(city)
                if i == n - 1:
                    v1 = cur.close
                    v2 = self.depot.close - self._travel_time(cur, self.depot) - cur.service_time
                    lt[i] = min(v1, v2)
                else:
                    nxt = self._node(tour[i + 1])
                    v1 = cur.close
                    v2 = lt[i + 1] - self._travel_time(cur, nxt) - cur.service_time
                    lt[i] = min(v1, v2)

    def _node(self, idx: int):
        if idx == 0:
            return self.depot
        return self.req_map[idx]

    # ------------------------------------------------------------------
    # Feasibility check
    # ------------------------------------------------------------------

    def _feasible_insertion(
        self,
        routes: List[List[int]],
        begin_service: Dict[int, float],
        earliest_time: List[List[float]],
        latest_time: List[List[float]],
        current_quantity: List[float],
        customer: int,
        route_idx: int,
        prev_pos: int,
        next_pos: int,
    ) -> bool:
        cust = self._node(customer)
        tour = routes[route_idx]
        prev = self._node(tour[prev_pos])
        nxt = self._node(tour[next_pos])

        new_qty = current_quantity[route_idx] + cust.demand
        if new_qty > self.capacity + 1e-9:
            return False

        travel_prev_cust = self._travel_time(prev, cust)
        arrival = begin_service.get(tour[prev_pos], 0.0) + prev.service_time + travel_prev_cust
        bs = max(arrival, cust.open)
        if bs > cust.close + 1e-9:
            return False

        v1 = cust.open
        v2 = earliest_time[route_idx][prev_pos] + travel_prev_cust + prev.service_time
        et = max(v1, v2)

        travel_cust_next = self._travel_time(cust, nxt)
        v3 = cust.close
        v4 = latest_time[route_idx][next_pos] - travel_cust_next - cust.service_time
        lt = min(v3, v4)

        if et > lt + 1e-9:
            return False

        return True

    # ------------------------------------------------------------------
    # Solomon I1 insertion heuristic
    # ------------------------------------------------------------------

    def _i1_sequential_insertion(
        self,
        routes: List[List[int]],
        begin_service: Dict[int, float],
        earliest_time: List[List[float]],
        latest_time: List[List[float]],
        current_quantity: List[float],
        current_time: List[float],
        unvisited: List[int],
        start_tour: int = 0,
        start_pos: Optional[List[int]] = None,
    ):
        if not unvisited:
            return
        self._compute_route_vars(routes, begin_service, earliest_time, latest_time, start_tour)

        remaining = set(unvisited)

        while remaining:
            best_insertions: List[Tuple[float, int, int, int]] = []

            for cust in remaining:
                best_c1 = float("inf")
                best_rt = -1
                best_pos = -1
                cust_obj = self._node(cust)

                for r in range(start_tour, len(routes)):
                    tour = routes[r]
                    if start_pos and r < len(start_pos):
                        lo = start_pos[r] + 1
                    else:
                        lo = 1
                    for p in range(lo, len(tour)):
                        if not self._feasible_insertion(
                            routes, begin_service, earliest_time, latest_time,
                            current_quantity, cust, r, p - 1, p,
                        ):
                            continue
                        prev = self._node(tour[p - 1])
                        nxt = self._node(tour[p])
                        c11 = _dist(prev, cust_obj) + _dist(cust_obj, nxt) - self.mu * _dist(prev, nxt)
                        travel_prev = self._travel_time(prev, cust_obj)
                        arrival = begin_service.get(tour[p - 1], 0.0) + prev.service_time + travel_prev
                        bs_cust = max(arrival, cust_obj.open)
                        travel_next = self._travel_time(cust_obj, nxt)
                        new_arrival = bs_cust + cust_obj.service_time + travel_next
                        new_bs_nxt = max(new_arrival, nxt.open)
                        old_bs_nxt = begin_service.get(tour[p], 0.0)
                        c12 = new_bs_nxt - old_bs_nxt
                        c1 = self.alfa1 * c11 + self.alfa2 * c12
                        if c1 < best_c1:
                            best_c1 = c1
                            best_rt = r
                            best_pos = p

                if best_c1 != float("inf"):
                    best_insertions.append((best_c1, cust, best_rt, best_pos))

            if not best_insertions:
                break

            cheapest = min(best_insertions, key=lambda x: x[0])
            _c1_val, best_cust, best_rt, best_pos = cheapest
            cust_obj = self._node(best_cust)

            # Insert customer at best position
            routes[best_rt].insert(best_pos, best_cust)
            new_qty = current_quantity[best_rt] + cust_obj.demand
            current_quantity[best_rt] = new_qty

            # Update route variables
            tour = routes[best_rt]
            prev = self._node(tour[best_pos - 1])
            nxt = self._node(tour[best_pos + 1])

            v1 = cust_obj.open
            v2 = earliest_time[best_rt][best_pos - 1] + self._travel_time(prev, cust_obj) + prev.service_time
            et = max(v1, v2)
            v3 = cust_obj.close
            v4 = latest_time[best_rt][best_pos] - self._travel_time(cust_obj, nxt) - cust_obj.service_time
            lt = min(v3, v4)
            earliest_time[best_rt].insert(best_pos, et)
            latest_time[best_rt].insert(best_pos, lt)

            # Update begin_service for this and subsequent positions
            for j in range(best_pos, len(tour)):
                if tour[j] == 0:
                    continue
                prev_n = self._node(tour[j - 1])
                cur_n = self._node(tour[j])
                arr = begin_service.get(tour[j - 1], 0.0) + prev_n.service_time + self._travel_time(prev_n, cur_n)
                bs = max(arr, cur_n.open)
                begin_service[tour[j]] = bs

            current_time[best_rt] = begin_service.get(tour[-1], 0.0)
            remaining.remove(best_cust)

    def _nearest_neighbor_fill(
        self,
        route: List[int],
        begin_service: Dict[int, float],
        current_quantity: float,
        unvisited: List[int],
    ) -> List[int]:
        if not unvisited:
            return unvisited
        pos = self._node(route[-1])
        t = begin_service.get(route[-1], 0.0)
        qty = current_quantity
        remaining = list(unvisited)
        while remaining:
            best = None
            best_d = float("inf")
            for c in remaining:
                d = _dist(pos, self._node(c))
                if d < best_d:
                    best_d = d
                    best = c
            if best is None:
                break
            cust = self._node(best)
            travel = self._travel_time(pos, cust)
            arrival = t + pos.service_time + travel
            bs = max(arrival, cust.open)
            if bs <= cust.close + 1e-9 and qty + cust.demand <= self.capacity + 1e-9:
                route.append(best)
                remaining.remove(best)
                begin_service[best] = bs
                pos = cust
                t = bs
                qty += cust.demand
            else:
                remaining.remove(best)
        return remaining

    def build_solution(self) -> Tuple[List[List[int]], float]:
        all_idxs = sorted([r.idx for r in self.reqs])
        unvisited = list(all_idxs)

        routes: List[List[int]] = []
        begin_service: Dict[int, float] = {0: 0.0}
        earliest_time: List[List[float]] = []
        latest_time: List[List[float]] = []
        current_quantity: List[float] = []
        current_time: List[float] = []

        def _add_route():
            routes.append([0, 0])
            earliest_time.append([])
            latest_time.append([])
            current_quantity.append(0.0)
            current_time.append(0.0)

        _add_route()

        self._i1_sequential_insertion(
            routes, begin_service, earliest_time, latest_time,
            current_quantity, current_time, unvisited,
        )

        unvisited_after = [c for c in all_idxs if c not in {x for r in routes for x in r}]
        while unvisited_after and len(routes) < self.max_vehicles:
            _add_route()
            added = len(unvisited_after)
            self._i1_sequential_insertion(
                routes, begin_service, earliest_time, latest_time,
                current_quantity, current_time, unvisited_after,
                start_tour=len(routes) - 1,
            )
            unvisited_after = [c for c in all_idxs if c not in {x for r in routes for x in r}]
            if len(unvisited_after) >= added:
                break

        if self.do_km and len(routes) >= 2:
            self._km_improve(routes, begin_service, current_quantity)

        if self.do_ls:
            self._relocate(routes, begin_service, current_quantity, current_time)
            self._exchange(routes, begin_service, current_quantity, current_time)

        result = []
        for r in routes:
            pruned = [c for c in r if c != 0]
            if pruned:
                result.append(pruned)

        total_dist = self._total_distance(routes)
        return result, total_dist

    def _total_distance(self, routes: List[List[int]]) -> float:
        total = 0.0
        for tour in routes:
            for i in range(len(tour) - 1):
                total += _dist(self._node(tour[i]), self._node(tour[i + 1]))
            if len(tour) >= 2:
                total += _dist(self._node(tour[-1]), self.depot)
        return total

    # ------------------------------------------------------------------
    # KM (Hungarian) matching improvement
    # ------------------------------------------------------------------

    def _km_improve(
        self,
        routes: List[List[int]],
        begin_service: Dict[int, float],
        current_quantity: List[float],
    ):
        from scipy.optimize import linear_sum_assignment
        n = len(routes)
        if n < 2:
            return

        # Use approximate split points (midpoint of each route)
        split_pos = []
        for r in range(n):
            tour = routes[r]
            mid = max(1, len(tour) // 2)
            split_pos.append(mid)

        cost = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                left = routes[i][:split_pos[i]]
                right = routes[j][split_pos[j]:]
                merged = left + right
                d = 0.0
                for k in range(len(merged) - 1):
                    d += _dist(self._node(merged[k]), self._node(merged[k + 1]))
                cost[i][j] = d

        row_ind, col_ind = linear_sum_assignment(cost)

        new_routes: List[List[int]] = []
        new_quantity: List[float] = []
        for i, j in zip(row_ind, col_ind):
            left = routes[i][:split_pos[i]]
            right = routes[j][split_pos[j]:]
            merged = left + right
            if merged[-1] != 0:
                merged.append(0)
            new_routes.append(merged)
            tour_requests = [c for c in merged if c != 0]
            qty = sum(self._node(c).demand for c in tour_requests)
            new_quantity.append(qty)

        routes.clear()
        routes.extend(new_routes)
        current_quantity.clear()
        current_quantity.extend(new_quantity)

        self._rebuild_begin_service(routes, begin_service)

    def _rebuild_begin_service(
        self,
        routes: List[List[int]],
        begin_service: Dict[int, float],
    ):
        begin_service.clear()
        begin_service[0] = 0.0
        for tour in routes:
            t = 0.0
            for i, city in enumerate(tour):
                if i == 0:
                    begin_service[city] = t
                    continue
                if city == 0:
                    continue
                prev = self._node(tour[i - 1])
                cur = self._node(city)
                travel = self._travel_time(prev, cur)
                arrival = t + prev.service_time + travel
                bs = max(arrival, cur.open)
                begin_service[city] = bs
                t = bs

    # ------------------------------------------------------------------
    # Relocate local search
    # ------------------------------------------------------------------

    def _relocate(
        self,
        routes: List[List[int]],
        begin_service: Dict[int, float],
        current_quantity: List[float],
        current_time: List[float],
    ):
        improved = True
        while improved:
            improved = False
            for r in range(len(routes)):
                tour = routes[r]
                p = 1
                while p < len(tour) - 1:
                    cust = tour[p]
                    cust_obj = self._node(cust)
                    if cust == 0:
                        p += 1
                        continue
                    before = self._total_distance(routes)
                    qty_r = current_quantity[r]
                    tour.pop(p)
                    current_quantity[r] -= cust_obj.demand

                    best_gain = 0.0
                    best_t = -1
                    best_pos = -1
                    for t in range(len(routes)):
                        if t == r and p == len(tour):
                            continue
                        for pos in range(1, len(routes[t])):
                            candidate_routes = [list(x) for x in routes]
                            candidate_routes[t].insert(pos, cust)
                            d = self._total_distance(candidate_routes)
                            gain = before - d
                            if gain > best_gain + 1e-9:
                                best_gain = gain
                                best_t = t
                                best_pos = pos

                    if best_gain > 1e-9:
                        routes[best_t].insert(best_pos, cust)
                        current_quantity[best_t] += cust_obj.demand
                        improved = True
                        # tour shortened; don't increment p
                    else:
                        tour.insert(p, cust)
                        current_quantity[r] = qty_r
                        p += 1

        self._rebuild_begin_service(routes, begin_service)

    # ------------------------------------------------------------------
    # Exchange local search
    # ------------------------------------------------------------------

    def _exchange(
        self,
        routes: List[List[int]],
        begin_service: Dict[int, float],
        current_quantity: List[float],
        current_time: List[float],
    ):
        improved = True
        while improved:
            improved = False
            for r1 in range(len(routes)):
                t1 = routes[r1]
                p1 = 1
                while p1 < len(t1) - 1:
                    c1 = t1[p1]
                    if c1 == 0:
                        p1 += 1
                        continue
                    for r2 in range(r1 + 1, len(routes)):
                        t2 = routes[r2]
                        p2 = 1
                        while p2 < len(t2) - 1:
                            c2 = t2[p2]
                            if c2 == 0 or c1 == c2:
                                p2 += 1
                                continue

                            before = self._total_distance(routes)
                            t1[p1], t2[p2] = t2[p2], t1[p1]
                            after = self._total_distance(routes)

                            if after < before - 1e-9:
                                improved = True
                                p2 += 1  # p2 still valid (length unchanged)
                            else:
                                t1[p1], t2[p2] = t2[p2], t1[p1]
                                p2 += 1
                    p1 += 1
        self._rebuild_begin_service(routes, begin_service)


# ──────────────────────────────────────────────────────────────────────────
# ACS-KM DVRPTW Simulation (event-driven, soft_Simulation2-based)
# ──────────────────────────────────────────────────────────────────────────

class ACS_DVRPTW_Simulation(soft_Simulation2):
    """Event-driven DVRPTW simulation using ACS-KM I1 insertion.

    Replaces GPHH's GP routing/sequencing policies with Solomon I1
    cheapest-insertion (assigns each arriving request to the best route
    position) and route-order dispatch.

    Uses the same event loop, physics, and metrics as soft_Simulation2
    for fair comparison with GPHH solutions.

    Inherits simulate_until and route_vehicle_to from soft_Simulation2.

    Limitation: I1 inserts each arriving request into the uncommitted
    future (positions >= next_pos) of each vehicle's route. For scenarios
    where many requests arrive after vehicles have already departed
    (e.g., h100r101 where 73/100 arrive after time 0), the available
    insertion slots may be insufficient, leaving requests pending.
    The static or near-static case (all requests at time 0, e.g.,
    h100c101) achieves 100% service.
    """

    def __init__(self, problem: Problem, do_km: bool = False, do_ls: bool = False):
        self.problem = problem
        self.time: float = 0.0
        self.vehicles: List[VehicleState] = [VehicleState(problem) for _ in range(problem.num_trucks)]
        self.has_departed_once: List[bool] = [False for _ in range(problem.num_trucks)]
        self.completed_single_tour: List[bool] = [False for _ in range(problem.num_trucks)]
        self._events: List[Tuple[float, int, Event]] = []
        self._counter = 0

        self.do_km = do_km
        self.do_ls = do_ls

        self.routes: List[List[int]] = [[0, 0] for _ in range(problem.num_trucks)]
        self.begin_service: Dict[int, float] = {0: 0.0}
        self.earliest_time: List[List[float]] = [[] for _ in range(problem.num_trucks)]
        self.latest_time: List[List[float]] = [[] for _ in range(problem.num_trucks)]
        self.current_quantity: List[float] = [0.0 for _ in range(problem.num_trucks)]
        self.next_pos: List[int] = [1 for _ in range(problem.num_trucks)]

        self.req_map = {r.idx: r for r in problem.requests}
        self.depot = problem.depot
        self.capacity = problem.truck_capacity

        for i in range(problem.num_trucks):
            self._recompute_et_lt(i)

    # -- helpers -----------------------------------------------------------

    def _node(self, idx: int):
        return self.depot if idx == 0 else self.req_map[idx]

    @staticmethod
    def _dist(a, b) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def _travel_time(self, a, b) -> float:
        return self._dist(a, b) / self.problem.truck_speed

    def _push_event(self, event: Event) -> None:
        heapq.heappush(self._events, (event.time(), self._counter, event))
        self._counter += 1

    def _pop_event(self) -> Optional[Event]:
        if not self._events:
            return None
        _, _, ev = heapq.heappop(self._events)
        return ev

    # -- route variable maintenance ---------------------------------------

    def _recompute_et_lt(self, route_idx: int) -> None:
        tour = self.routes[route_idx]
        et = self.earliest_time[route_idx]
        lt = self.latest_time[route_idx]
        n = len(tour)
        et.clear()
        lt.clear()
        et.extend([0.0] * n)
        lt.extend([0.0] * n)

        et[0] = self.depot.open
        for i in range(1, n):
            prev = self._node(tour[i - 1])
            cur = self._node(tour[i])
            et[i] = max(cur.open, et[i - 1] + prev.service_time + self._travel_time(prev, cur))

        lt[n - 1] = self.depot.close
        for i in range(n - 2, -1, -1):
            cur = self._node(tour[i])
            nxt = self._node(tour[i + 1])
            lt[i] = min(cur.close, lt[i + 1] - self._travel_time(cur, nxt) - cur.service_time)

    def _rebuild_begin_service(self, route_idx: int) -> None:
        tour = self.routes[route_idx]
        t = 0.0
        for i in range(1, len(tour)):
            city = tour[i]
            if city == 0:
                continue
            prev = self._node(tour[i - 1])
            cur = self._node(city)
            travel = self._travel_time(prev, cur)
            arrival = t + prev.service_time + travel
            bs = max(arrival, cur.open)
            self.begin_service[city] = bs
            t = bs

    def _feasible_insertion(self, route_idx: int, position: int, customer: int) -> bool:
        tour = self.routes[route_idx]
        cust = self._node(customer)
        prev = self._node(tour[position - 1])
        nxt = self._node(tour[position])

        if self.current_quantity[route_idx] + cust.demand > self.capacity + 1e-9:
            return False

        travel_prev_cust = self._travel_time(prev, cust)
        arrival = self.begin_service.get(tour[position - 1], 0.0) + prev.service_time + travel_prev_cust
        bs = max(arrival, cust.open)
        if bs > cust.close + 1e-9:
            return False

        v1 = cust.open
        v2 = self.earliest_time[route_idx][position - 1] + travel_prev_cust + prev.service_time
        et = max(v1, v2)

        travel_cust_next = self._travel_time(cust, nxt)
        v3 = cust.close
        v4 = self.latest_time[route_idx][position] - travel_cust_next - cust.service_time
        lt = min(v3, v4)

        return et <= lt + 1e-9

    # -- I1 cheapest insertion ---------------------------------------------

    def _cheapest_insertion(self, customer: int) -> Tuple[Optional[int], int]:
        best_vehicle: Optional[int] = None
        best_position = -1
        best_cost: Optional[float] = None

        for vi in range(self.problem.num_trucks):
            if self.completed_single_tour[vi]:
                continue
            tour = self.routes[vi]
            # Only consider the uncommitted future of the route
            # (positions < next_pos have been dispatched and can't be reordered)
            for pos in range(self.next_pos[vi], len(tour)):
                if not self._feasible_insertion(vi, pos, customer):
                    continue
                prev = self._node(tour[pos - 1])
                nxt = self._node(tour[pos])
                cust = self._node(customer)
                cost = self._dist(prev, cust) + self._dist(cust, nxt) - self._dist(prev, nxt)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_vehicle = vi
                    best_position = pos

        return best_vehicle, best_position

    # -- simulation decision points (overrides soft_Simulation2) -----------

    def handle_request(self, request, _cb=None, total_delay_container=None):
        vehicle, pos = self._cheapest_insertion(request.idx)
        if vehicle is None:
            return

        tour = self.routes[vehicle]
        tour.insert(pos, request.idx)

        self.current_quantity[vehicle] += request.demand

        self._recompute_et_lt(vehicle)
        self._rebuild_begin_service(vehicle)

        self.vehicles[vehicle].enqueue(request, self.time)

    def handle_vehicle_finish(self, vehicle: int, request) -> None:
        pass

    def update_vehicle_queue(
        self,
        vehicle: int,
        _cb,
        total_distance_container: List[float],
        total_delay_container: List[float],
    ) -> None:
        if self.completed_single_tour[vehicle]:
            self.vehicles[vehicle].queue.clear()
            return

        state = self.vehicles[vehicle]
        if self.time < state.busy_until:
            return

        route = self.routes[vehicle]
        np = self.next_pos[vehicle]

        # Skip trailing depot markers and stale entries
        while np < len(route) - 1 and route[np] == 0:
            np += 1

        # No more customers to serve
        if np >= len(route) - 1:
            # Already at depot — nothing to do
            if getattr(state.cur_request, "idx", 0) == 0:
                return
            self.route_vehicle_to(vehicle, self.problem.depot, _cb, total_distance_container)
            self.next_pos[vehicle] = np
            return

        target_idx = route[np]
        found = None
        for qi, (req, _) in enumerate(state.queue):
            if req.idx == target_idx:
                found = qi
                break

        if found is None:
            self.next_pos[vehicle] = np + 1
            return

        request = state.queue[found][0]
        if request.demand > state.total_demand:
            self.route_vehicle_to(vehicle, self.problem.depot, _cb, total_distance_container)
            return

        state.queue.pop(found)
        self.next_pos[vehicle] = np + 1

        start_time = self.time + state.time_cost(self.problem, request, self.time)
        if start_time > request.close:
            total_delay_container[0] += start_time - request.close

        self.route_vehicle_to(vehicle, request, _cb, total_distance_container)


__all__ = ["ACS_KM", "ACS_DVRPTW_Simulation"]
