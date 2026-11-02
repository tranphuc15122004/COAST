"""Simulation runtime (Python port of src/sim/mod.rs).

Responsibilities:
- `Event`: simple timestamped event used by the simulation priority queue.
- `VehicleState`: track per-vehicle route/queue/state and provide cost
    helpers used by Program contexts.
- `Simulation`: event-driven simulation that batches requests into time
    slots, routes vehicles using `Program.calc(ctx)` for routing and
    sequencing decisions, and returns `(total_distance, failed_count)`.

Adapters:
- `routing_rule_route_request(...)` and `sequencing_rule_sequence_request(...)`
    adapt `Program.calc` (with `RoutingContext`/`SequencingContext`) into the
    simulation decision points.
"""
from __future__ import annotations

import heapq
from typing import List, Dict, Tuple, Optional

from python_src.gp.GPtree import Program
from python_src.sim.problem import Problem
from python_src.sim.ctx import RoutingContext, SequencingContext


class Event:
    def __init__(self, kind: str, payload, time: float):
        self.kind = kind
        self.payload = payload
        self._time = float(time)

    def time(self) -> float:
        return self._time

    def __repr__(self) -> str:
        return f"Event({self.kind}, time={self._time})"


class VehicleState:
    def __init__(self, problem):
        self.cur_request = problem.depot
        self.queue: List[Tuple[object, float]] = []
        self.total_demand: float = problem.truck_capacity
        self.busy_until: float = 0.0
        self.route: Dict[int, int] = {}
        self.dropped: Dict[int, int] = {}

    def time_cost(self, problem, req, time: float) -> float:
        return max(self.distance_to(req) / problem.truck_speed, req.open - time)

    def raw_time_cost(self, problem, req, _: float) -> float:
        return self.distance_to(req) / problem.truck_speed

    def time_until_open(self, req, time: float) -> float:
        return time - req.time

    def distance_to(self, request) -> float:
        return self.dist(self.cur_request.x - request.x, self.cur_request.y - request.y)

    @staticmethod
    def dist(x: float, y: float) -> float:
        return (x * x + y * y) ** 0.5

    def enqueue(self, request, time: float) -> None:
        self.queue.append((request, time))

    @staticmethod
    def median(values) -> float:
        xs = sorted(values)
        n = len(xs)
        if n == 0:
            return 0.0
        if n % 2 == 1:
            return xs[n // 2]
        return 0.5 * (xs[n // 2] + xs[n // 2 - 1])

    def median_queue_pos(self) -> Tuple[float, float]:
        x = (r[0].x for r in self.queue)
        y = (r[0].y for r in self.queue)
        return (self.median(list(x)), self.median(list(y)))


class Simulation:
    def __init__(self, problem : Problem, routing_rule: Program, sequencing_rule: Program):
        self.problem = problem
        self.routing_rule = routing_rule
        self.sequencing_rule = sequencing_rule
        self.time: float = 0.0
        self.vehicles: List[VehicleState] = [VehicleState(problem) for _ in range(problem.num_trucks)]
        self._events: List[Tuple[float, int, Event]] = []
        self._counter = 0

    def _push_event(self, event: Event) -> None:
        heapq.heappush(self._events, (event.time(), self._counter, event))
        self._counter += 1

    def _pop_event(self) -> Optional[Event]:
        if not self._events:
            return None
        _, _, ev = heapq.heappop(self._events)
        return ev

    def simulate_until(self, time_slot: float, time_max: float) -> Tuple[float, int]:
        batched_requests: Dict[int, List[object]] = {}
        for request in self.problem.requests:
            timeslot_idx = int((request.time / time_slot).__ceil__()) if time_slot != 0 else 0
            batched_requests.setdefault(timeslot_idx, []).append(request)

        for idx, requests in batched_requests.items():
            ev = Event("requests", requests, idx * time_slot)
            self._push_event(ev)

        total_distance = 0.0
        total_failed = 0
        while self._events:
            # peek next event
            time, _, ev = self._events[0]
            if time > time_max:
                break

            ev = self._pop_event()
            assert ev is not None
            self.time = ev.time()
            if ev.kind == "requests":
                for request in ev.payload:
                    self.handle_request(request, lambda: None, total_failed_container := [total_failed])
                    total_failed = total_failed_container[0]
            elif ev.kind == "vehicle_finish":
                vehicle, request = ev.payload
                self.handle_vehicle_finish(vehicle, request)

            for vehicle_idx in range(self.problem.num_trucks):
                self.update_vehicle_queue(vehicle_idx, lambda: None, total_failed_container := [total_failed], total_distance_container := [total_distance])
                total_failed = total_failed_container[0]
                total_distance = total_distance_container[0]

        # route remaining vehicles to depot
        for vehicle in range(self.problem.num_trucks):
            self.route_vehicle_to(vehicle, self.problem.depot, lambda d: None, total_distance_container := [total_distance])
            total_distance = total_distance_container[0]

        # log routes (kept minimal here)
        return total_distance, total_failed

    def handle_request(self, request, _cb=None, total_failed_container=None):
        vehicle = self.routing_rule_route_request(self.problem, self.time, self.vehicles, request)
        if vehicle is not None:
            self.vehicles[vehicle].enqueue(request, self.time)
        else:
            if total_failed_container is not None:
                total_failed_container[0] += 1

    def handle_vehicle_finish(self, vehicle: int, request) -> None:
        # placeholder for logging
        return

    def update_vehicle_queue(self, vehicle: int, _cb, total_failed_container: List[int], total_distance_container: List[float]) -> None:
        state = self.vehicles[vehicle]
        if self.time < state.busy_until:
            return

        cache: Dict[int, float] = {}
        while True:
            idx = self.sequencing_rule_sequence_request(self.problem, self.time, state, cache)
            if idx is None:
                break
            queue = state.queue
            request = queue[idx][0]
            if request.demand > state.total_demand:
                # cannot serve this request yet - need to return to depot first to refill capacity
                # DO NOT pop the request from queue, leave it there for next iteration
                self.route_vehicle_to(vehicle, self.problem.depot, _cb, total_distance_container)
                return

            # swap_remove equivalent: pop idx (only pop when we're sure we'll serve it)
            queue.pop(idx)
            start_time = self.time + state.time_cost(self.problem, request, self.time)
            if start_time > request.close:
                # treat as failed (re-route attempt)
                self.handle_request(request, _cb, total_failed_container)
                continue

            self.route_vehicle_to(vehicle, request, _cb, total_distance_container)
            return

    def route_vehicle_to(self, vehicle: int, request, _cb, total_distance_container: List[float]) -> None:
        state = self.vehicles[vehicle]
        distance = state.distance_to(request)
        total_distance_container[0] += distance
        time = max(self.time + distance / self.problem.truck_speed, request.open) + getattr(request, "service_time", 0.0)
        if getattr(request, "idx", 0) == 0:
            state.total_demand = self.problem.truck_capacity
        else:
            state.total_demand -= getattr(request, "demand", 0.0)
        # schedule finish
        ev = Event("vehicle_finish", (vehicle, request), time)
        self._push_event(ev)
        # Use float time as key to avoid collisions when multiple requests served at similar times
        state.route[time - getattr(request, "service_time", 0.0)] = getattr(request, "idx", 0)
        state.cur_request = request
        state.busy_until = time

    # Helper adapters to use Program.calc as routing/sequencing rule
    def routing_rule_route_request(self, problem, time: float, vehicles: List[VehicleState], request) -> Optional[int]:
        candidates = []
        for i in range(len(vehicles)):
            cost = vehicles[i].raw_time_cost(problem, request, time)
            if time + cost <= request.close:
                # evaluate program
                ctx = RoutingContext(vehicle_state=vehicles[i], problem=problem, time=time, request=request)
                value = float(self.routing_rule.calc(ctx))
                candidates.append((value, i))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def sequencing_rule_sequence_request(self, problem, time: float, vehicle_state: VehicleState, cache: Dict[int, float]) -> Optional[int]:
        if not vehicle_state.queue:
            return None
        best_idx = None
        best_val = None
        for i in range(len(vehicle_state.queue)):
            request_idx = vehicle_state.queue[i][0].idx
            if request_idx not in cache:
                ctx = SequencingContext(vehicle_state=vehicle_state, problem=problem, time=time, request=vehicle_state.queue[i][0], ready_time=vehicle_state.queue[i][1])
                cache[request_idx] = float(self.sequencing_rule.calc(ctx))
            val = cache[request_idx]
            if best_val is None or val < best_val:
                best_val = val
                best_idx = i
        return best_idx



class soft_Simulation:
    def __init__(self, problem : Problem, routing_rule: Program, sequencing_rule: Program):
        self.problem = problem
        self.routing_rule = routing_rule
        self.sequencing_rule = sequencing_rule
        self.time: float = 0.0
        self.vehicles: List[VehicleState] = [VehicleState(problem) for _ in range(problem.num_trucks)]
        self._events: List[Tuple[float, int, Event]] = []
        self._counter = 0

    def _push_event(self, event: Event) -> None:
        heapq.heappush(self._events, (event.time(), self._counter, event))
        self._counter += 1

    def _pop_event(self) -> Optional[Event]:
        if not self._events:
            return None
        _, _, ev = heapq.heappop(self._events)
        return ev

    def simulate_until(self, time_slot: float, time_max: float) -> Tuple[float, int]:
        batched_requests: Dict[int, List[object]] = {}
        for request in self.problem.requests:
            timeslot_idx = int((request.time / time_slot).__ceil__()) if time_slot != 0 else 0
            batched_requests.setdefault(timeslot_idx, []).append(request)

        for idx, requests in batched_requests.items():
            ev = Event("requests", requests, idx * time_slot)
            self._push_event(ev)

        total_distance = 0.0
        total_failed = 0
        while self._events:
            # peek next event
            time, _, ev = self._events[0]
            if time > time_max:
                break

            ev = self._pop_event()
            assert ev is not None
            self.time = ev.time()
            if ev.kind == "requests":
                for request in ev.payload:
                    self.handle_request(request, lambda: None, total_failed_container := [total_failed])
                    total_failed = total_failed_container[0]
            elif ev.kind == "vehicle_finish":
                vehicle, request = ev.payload
                self.handle_vehicle_finish(vehicle, request)

            for vehicle_idx in range(self.problem.num_trucks):
                self.update_vehicle_queue(vehicle_idx, lambda: None, total_failed_container := [total_failed], total_distance_container := [total_distance])
                total_failed = total_failed_container[0]
                total_distance = total_distance_container[0]

        # route remaining vehicles to depot
        for vehicle in range(self.problem.num_trucks):
            self.route_vehicle_to(vehicle, self.problem.depot, lambda d: None, total_distance_container := [total_distance])
            total_distance = total_distance_container[0]

        # log routes (kept minimal here)
        return total_distance, total_failed

    def handle_request(self, request, _cb=None, total_failed_container=None):
        vehicle = self.routing_rule_route_request(self.problem, self.time, self.vehicles, request)
        if vehicle is not None:
            self.vehicles[vehicle].enqueue(request, self.time)
        else:
            if total_failed_container is not None:
                total_failed_container[0] += 1

    def handle_vehicle_finish(self, vehicle: int, request) -> None:
        # placeholder for logging
        return

    def update_vehicle_queue(self, vehicle: int, _cb, total_failed_container: List[int], total_distance_container: List[float]) -> None:
        state = self.vehicles[vehicle]
        if self.time < state.busy_until:
            return

        cache: Dict[int, float] = {}
        while True:
            idx = self.sequencing_rule_sequence_request(self.problem, self.time, state, cache)
            if idx is None:
                break
            queue = state.queue
            request = queue[idx][0]
            if request.demand > state.total_demand:
                # cannot serve this request yet - need to return to depot first to refill capacity
                # DO NOT pop the request from queue, leave it there for next iteration
                self.route_vehicle_to(vehicle, self.problem.depot, _cb, total_distance_container)
                return

            # swap_remove equivalent: pop idx (only pop when we're sure we'll serve it)
            queue.pop(idx)
            start_time = self.time + state.time_cost(self.problem, request, self.time)
            if start_time > request.close:
                # treat as failed (re-route attempt)
                self.handle_request(request, _cb, total_failed_container)
                continue

            self.route_vehicle_to(vehicle, request, _cb, total_distance_container)
            return

    def route_vehicle_to(self, vehicle: int, request, _cb, total_distance_container: List[float]) -> None:
        state = self.vehicles[vehicle]
        distance = state.distance_to(request)
        total_distance_container[0] += distance
        time = max(self.time + distance / self.problem.truck_speed, request.open) + getattr(request, "service_time", 0.0)
        if getattr(request, "idx", 0) == 0:
            state.total_demand = self.problem.truck_capacity
        else:
            state.total_demand -= getattr(request, "demand", 0.0)
        # schedule finish
        ev = Event("vehicle_finish", (vehicle, request), time)
        self._push_event(ev)
        # Use float time as key to avoid collisions when multiple requests served at similar times
        state.route[time - getattr(request, "service_time", 0.0)] = getattr(request, "idx", 0)
        state.cur_request = request
        state.busy_until = time

    # Helper adapters to use Program.calc as routing/sequencing rule
    def routing_rule_route_request(self, problem, time: float, vehicles: List[VehicleState], request) -> Optional[int]:
        candidates = []
        for i in range(len(vehicles)):
            cost = vehicles[i].raw_time_cost(problem, request, time)
            if time + cost <= request.close:
                # evaluate program
                ctx = RoutingContext(vehicle_state=vehicles[i], problem=problem, time=time, request=request)
                value = float(self.routing_rule.calc(ctx))
                candidates.append((value, i))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def sequencing_rule_sequence_request(self, problem, time: float, vehicle_state: VehicleState, cache: Dict[int, float]) -> Optional[int]:
        if not vehicle_state.queue:
            return None
        best_idx = None
        best_val = None
        for i in range(len(vehicle_state.queue)):
            request_idx = vehicle_state.queue[i][0].idx
            if request_idx not in cache:
                ctx = SequencingContext(vehicle_state=vehicle_state, problem=problem, time=time, request=vehicle_state.queue[i][0], ready_time=vehicle_state.queue[i][1])
                cache[request_idx] = float(self.sequencing_rule.calc(ctx))
            val = cache[request_idx]
            if best_val is None or val < best_val:
                best_val = val
                best_idx = i
        return best_idx


# cần định nghĩa cho bài toán DVRPTW với răng buộc mềm cho TW
class soft_Simulation2:
    def __init__(self, problem : Problem, routing_rule: Program, sequencing_rule: Program):
        self.problem = problem
        self.routing_rule = routing_rule
        self.sequencing_rule = sequencing_rule
        self.time: float = 0.0
        self.vehicles: List[VehicleState] = [VehicleState(problem) for _ in range(problem.num_trucks)]
        # One-tour constraint state per vehicle:
        # - has_departed_once: vehicle has left depot to serve at least one customer.
        # - completed_single_tour: vehicle has returned to depot after departure and is locked.
        self.has_departed_once: List[bool] = [False for _ in range(problem.num_trucks)]
        self.completed_single_tour: List[bool] = [False for _ in range(problem.num_trucks)]
        self._events: List[Tuple[float, int, Event]] = []
        self._counter = 0
        
    def _push_event(self, event: Event) -> None:
        heapq.heappush(self._events, (event.time(), self._counter, event))
        self._counter += 1

    def _pop_event(self) -> Optional[Event]:
        if not self._events:
            return None
        _, _, ev = heapq.heappop(self._events)
        return ev
    
    def simulate_until(self, time_slot: float, time_max: float) -> Tuple[float, int, float]:
        batched_requests: Dict[int, List[object]] = {}
        for request in self.problem.requests:
            timeslot_idx = int((request.time / time_slot).__ceil__()) if time_slot != 0 else 0
            batched_requests.setdefault(timeslot_idx, []).append(request)

        total_batched = sum(len(v) for v in batched_requests.values())
        if total_batched < len(self.problem.requests):
            print(f"WARNING: Only {total_batched}/{len(self.problem.requests)} requests were batched!")

        for idx, requests in batched_requests.items():
            ev = Event("requests", requests, idx * time_slot)
            self._push_event(ev)

        total_distance = 0.0
        total_delay = 0.0
        while self._events:
            # process all events (no time_max cutoff to ensure all requests are handled)
            ev = self._pop_event()
            assert ev is not None
            self.time = ev.time()
            if ev.kind == "requests":
                for request in ev.payload:
                    total_delay_container = [total_delay]
                    self.handle_request(request, lambda: None, total_delay_container)
                    total_delay = total_delay_container[0]
            elif ev.kind == "vehicle_finish":
                vehicle, request = ev.payload
                self.handle_vehicle_finish(vehicle, request)

            for vehicle_idx in range(self.problem.num_trucks):
                self.update_vehicle_queue(vehicle_idx, lambda: None, total_distance_container := [total_distance], total_delay_container := [total_delay])
                total_distance = total_distance_container[0]
                total_delay = total_delay_container[0]

        # After main event loop, process any remaining queued requests
        total_queued_after_main_loop = sum(len(v.queue) for v in self.vehicles)
        if total_queued_after_main_loop > 0:
            for i, v in enumerate(self.vehicles):
                if v.queue:
                    print(f"WARNING: Vehicle {i} has {len(v.queue)} queued requests after main loop")

        # force process all remaining queued requests before routing to depot
        max_iterations = 10000  # safety limit
        iterations = 0
        while iterations < max_iterations:
            iterations += 1
            any_processed = False
            total_queued = sum(len(v.queue) for v in self.vehicles)
            if total_queued > 0 and iterations == 1:
                print(f"DEBUG: Force processing {total_queued} queued requests...")
            
            # Advance time to the earliest vehicle that becomes available
            min_busy_until = min((v.busy_until for v in self.vehicles if v.queue), default=self.time)
            if min_busy_until > self.time:
                print(f"DEBUG: Advancing time from {self.time} to {min_busy_until}")
                self.time = min_busy_until
            
            for vehicle_idx in range(self.problem.num_trucks):
                if self.vehicles[vehicle_idx].queue:
                    self.update_vehicle_queue(vehicle_idx, lambda: None, total_distance_container := [total_distance], total_delay_container := [total_delay])
                    total_distance = total_distance_container[0]
                    total_delay = total_delay_container[0]
                    any_processed = True
            if not any_processed:
                break
        if iterations >= max_iterations:
            print(f"WARNING: Force processing stopped after {max_iterations} iterations")

        # route remaining vehicles to depot
        for vehicle in range(self.problem.num_trucks):
            self.route_vehicle_to(vehicle, self.problem.depot, lambda d: None, total_distance_container := [total_distance])
            total_distance = total_distance_container[0]

        served_ids = set()
        for vehicle in self.vehicles:
            for request_id in vehicle.route.values():
                if request_id != 0:
                    served_ids.add(request_id)

        pending = max(0, len(self.problem.requests) - len(served_ids))
        return total_distance, pending, total_delay

    def handle_request(self, request, _cb=None, total_delay_container=None):
        vehicle = self.routing_rule_route_request(self.problem, self.time, self.vehicles, request)
        if vehicle is not None:
            self.vehicles[vehicle].enqueue(request, self.time)
        else:
            # No available vehicle under one-tour constraint: keep as unserved.
            print(f"WARNING: no available vehicle for request {request.idx} at time {self.time}; request stays pending")

    def handle_vehicle_finish(self, vehicle: int, request) -> None:
        # placeholder for logging
        return

    def update_vehicle_queue(self, vehicle: int, _cb, total_distance_container: List[float], total_delay_container: List[float]) -> None:
        if self.completed_single_tour[vehicle]:
            state = self.vehicles[vehicle]
            if state.queue:
                state.queue.clear()
            return

        state = self.vehicles[vehicle]
        if self.time < state.busy_until:
            return

        cache: Dict[int, float] = {}
        while True:
            idx = self.sequencing_rule_sequence_request(self.problem, self.time, state, cache)
            if idx is None:
                break
            queue = state.queue
            request = queue[idx][0]
            if request.demand > state.total_demand:
                # cannot serve this request yet - need to return to depot first to refill capacity
                # DO NOT pop the request from queue, leave it there for next iteration
                self.route_vehicle_to(vehicle, self.problem.depot, _cb, total_distance_container)
                return

            # swap_remove equivalent: pop idx (only pop when we're sure we'll serve it)
            queue.pop(idx)
            start_time = self.time + state.time_cost(self.problem, request, self.time)
            # Soft TW: serve even if late and accumulate delay
            if start_time > request.close:
                delay = start_time - request.close
                total_delay_container[0] += delay
            # Always serve the request (no re-routing for lateness)
            self.route_vehicle_to(vehicle, request, _cb, total_distance_container)
            return

    def route_vehicle_to(self, vehicle: int, request, _cb, total_distance_container: List[float]) -> None:
        state = self.vehicles[vehicle]
        if self.completed_single_tour[vehicle] and getattr(request, "idx", 0) != 0:
            return

        from_depot = getattr(state.cur_request, "idx", 0) == 0
        to_depot = getattr(request, "idx", 0) == 0
        if from_depot and not to_depot:
            if self.completed_single_tour[vehicle]:
                return
            self.has_departed_once[vehicle] = True

        distance = state.distance_to(request)
        total_distance_container[0] += distance
        time = max(self.time + distance / self.problem.truck_speed, request.open) + getattr(request, "service_time", 0.0)
        if getattr(request, "idx", 0) == 0:
            state.total_demand = self.problem.truck_capacity
        else:
            state.total_demand -= getattr(request, "demand", 0.0)
        # schedule finish
        ev = Event("vehicle_finish", (vehicle, request), time)
        self._push_event(ev)
        # Use float time as key to avoid collisions when multiple requests served at similar times
        state.route[time - getattr(request, "service_time", 0.0)] = getattr(request, "idx", 0)
        state.cur_request = request
        state.busy_until = time

        # Lock the vehicle after it completes exactly one full tour (serve then return to depot).
        if to_depot and self.has_departed_once[vehicle]:
            self.completed_single_tour[vehicle] = True
            if state.queue:
                state.queue.clear()

    # Helper adapters to use Program.calc as routing/sequencing rule
    def routing_rule_route_request(self, problem, time: float, vehicles: List[VehicleState], request) -> Optional[int]:
        candidates = []
        fallback_candidates = []
        for i in range(len(vehicles)):
            if self.completed_single_tour[i]:
                continue
            cost = vehicles[i].raw_time_cost(problem, request, time)
            ctx = RoutingContext(vehicle_state=vehicles[i], problem=problem, time=time, request=request)
            value = float(self.routing_rule.calc(ctx))
            if time + cost <= request.close:
                # feasible within time window
                candidates.append((value, i))
            else:
                # not feasible, but track as fallback
                fallback_candidates.append((value, i))
        
        # prefer feasible candidates, but if none exist, use fallback (serve late)
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        elif fallback_candidates:
            fallback_candidates.sort(key=lambda x: x[0])
            return fallback_candidates[0][1]
        else:
            return None

    def sequencing_rule_sequence_request(self, problem, time: float, vehicle_state: VehicleState, cache: Dict[int, float]) -> Optional[int]:
        if not vehicle_state.queue:
            return None
        best_idx = None
        best_val = None
        for i in range(len(vehicle_state.queue)):
            request_idx = vehicle_state.queue[i][0].idx
            if request_idx not in cache:
                ctx = SequencingContext(vehicle_state=vehicle_state, problem=problem, time=time, request=vehicle_state.queue[i][0], ready_time=vehicle_state.queue[i][1])
                cache[request_idx] = float(self.sequencing_rule.calc(ctx))
            val = cache[request_idx]
            if best_val is None or val < best_val:
                best_val = val
                best_idx = i
        return best_idx


class RouteReplayer:
    def __init__(self, problem: Problem):
        self.problem = problem

    def evaluate(self, routes: List[List[int]]) -> Tuple[float, int, float]:
        """
        Replay pre-computed routes through the simulation environment for fair
        comparison with GP-evaluated solutions.

        Args:
            routes: List of routes, one per vehicle. Each route is a list of
                    request.idx values (from Problem.requests) in visit order.
                    Depot nodes (idx=0) should be excluded.

        Returns:
            (total_distance, pending, total_delay) using the same physics
            (Euclidean distance, truck_speed, service_time, soft TW) as
            soft_Simulation2.
        """
        total_distance = 0.0
        total_delay = 0.0
        served: set[int] = set()

        request_map = {r.idx: r for r in self.problem.requests}

        for route in routes:
            if not route:
                continue
            pos = self.problem.depot
            time = 0.0
            capacity = self.problem.truck_capacity

            for req_idx in route:
                if req_idx == 0:
                    continue
                request = request_map.get(req_idx)
                if request is None:
                    continue

                dx = pos.x - request.x
                dy = pos.y - request.y
                travel_dist = (dx * dx + dy * dy) ** 0.5
                total_distance += travel_dist
                travel_time = travel_dist / self.problem.truck_speed

                arrival = time + travel_time
                if arrival > request.close:
                    total_delay += arrival - request.close

                start_time = max(arrival, request.open)
                time = start_time + request.service_time
                capacity -= request.demand
                pos = request
                served.add(request.idx)

            dx = pos.x - self.problem.depot.x
            dy = pos.y - self.problem.depot.y
            total_distance += (dx * dx + dy * dy) ** 0.5

        pending = len(self.problem.requests) - len(served)
        return total_distance, pending, total_delay


__all__ = ["Event", "VehicleState", "Simulation", "soft_Simulation", "RouteReplayer"]
