"""Program contexts for simulation (Python port of src/sim/ctx.rs).

This module implements two context classes consumed by GP `Program`:

- `RoutingContext`: exposes routing-specific terminals and internals used
    when deciding which vehicle should handle a request.
- `SequencingContext`: exposes sequencing-specific terminals and internals
    used when deciding which queued request a vehicle should serve next.

Both contexts implement the minimal ProgramContext-like API expected by
`gp.program.Program.calc` (methods such as `num_terminals`,
`num_internals`, `internal_num_children`, `terminal`, and `internal`).

Helpers:
- `common_internal`, `common_format_terminal` and related helpers mirror
    the Rust implementations for consistent operator semantics and formatting.
"""


from __future__ import annotations

from typing import List
from dataclasses import dataclass

# Try to import the ProgramContext base so Routing/Sequencing contexts
# can explicitly subclass it for clearer relationship to the GP API.
try:
    from gp.GPtree import ProgramContext
except Exception:
    try:
        from gp.program import ProgramContext
    except Exception:
        # fallback: define a minimal sentinel so subclassing still works
        class ProgramContext:  # type: ignore
            pass

def safe_div(x: float, y: float) -> float:
    if abs(y) < 1e-4:
        return 1.0
    return x / y


def common_num_internal() -> int:
    return 6


def common_internal_num_children(_: int) -> int:
    return 2


def common_format_terminal(index: int) -> str:
    return {
        0: "sum",
        1: "sub",
        2: "mul",
        3: "div",
        4: "min",
        5: "max",
    }[index]


def common_internal(idx: int, child_values: List[float]) -> float:
    x = child_values[0]
    y = child_values[1]
    if idx == 0:
        return x + y
    if idx == 1:
        return x - y
    if idx == 2:
        return x * y
    if idx == 3:
        return safe_div(x, y)
    if idx == 4:
        return min(x, y)
    if idx == 5:
        return max(x, y)
    raise RuntimeError("invalid internal index")


class RoutingContext(ProgramContext):
    def __init__(self, vehicle_state: object, problem: object, time: float, request: object):
        self.vehicle_state = vehicle_state
        self.problem = problem
        self.time = time
        self.request = request

    # ProgramContext-compatible API ------------------------------------------------
    def internal(self, idx: int, child_values: List[float]) -> float:
        return common_internal(idx, child_values)

    @staticmethod
    def internal_num_children(index: int) -> int:
        return common_internal_num_children(index)

    @staticmethod
    def num_internals() -> int:
        return common_num_internal()

    @staticmethod
    def format_internal(index: int) -> str:
        return common_format_terminal(index)


    # Địnhh nghĩa các node TERMINAL trong ROUTING TREE
    def terminal(self, idx: int) -> float:
        # Mirrors Rust implementation of RoutingContext::terminal
        if idx == 0:
            return float(len(self.vehicle_state.queue)) / float(len(self.problem.requests))
        if idx == 1:
            queued = sum(r[0].demand for r in self.vehicle_state.queue)
            return (self.problem.truck_capacity - queued) / float(self.problem.total_demand())
        if idx == 2:
            x, y = self.vehicle_state.median_queue_pos()
            rx, ry = (self.request.x, self.request.y)
            return (((x - rx) ** 2 + (y - ry) ** 2) ** 0.5) / self.problem.truck_speed / self.problem.depot.close
        if idx == 3:
            return self.vehicle_state.raw_time_cost(self.problem, self.request, self.time) / self.problem.depot.close
        if idx == 4:
            return self.request.demand / float(self.problem.total_demand())
        raise RuntimeError("invalid terminal index for RoutingContext")

    @staticmethod
    def num_terminals() -> int:
        return 5


class SequencingContext(ProgramContext):
    def __init__(self, vehicle_state: object, problem: object, time: float, request: object, ready_time: float):
        self.vehicle_state = vehicle_state
        self.problem = problem
        self.time = time
        self.request = request
        self.ready_time = ready_time

    # ProgramContext-compatible API ------------------------------------------------
    def internal(self, idx: int, child_values: List[float]) -> float:
        return common_internal(idx, child_values)

    @staticmethod
    def internal_num_children(index: int) -> int:
        return common_internal_num_children(index)

    @staticmethod
    def num_internals() -> int:
        return common_num_internal()

    @staticmethod
    def format_internal(index: int) -> str:
        return common_format_terminal(index)

    # định nghĩa các node TERMINAL trong cây SEQUENCING
    def terminal(self, idx: int) -> float:
        raw_time_cost = self.vehicle_state.raw_time_cost(self.problem, self.request, self.time)
        time_until_close = self.request.close - self.vehicle_state.busy_until
        wait_time = self.time - self.request.open
        if idx == 0:
            return raw_time_cost / self.problem.depot.close
        if idx == 1:
            return (self.time - self.ready_time) / self.problem.depot.close
        if idx == 2:
            return safe_div(time_until_close - raw_time_cost, time_until_close)
        if idx == 3:
            return self.request.demand / float(self.problem.total_demand())
        if idx == 4:
            return wait_time / self.problem.depot.close
        if idx == 5:
            return self.request.time / self.problem.depot.close
        raise RuntimeError("invalid terminal index for SequencingContext")

    @staticmethod
    def num_terminals() -> int:
        return 6


__all__ = ["RoutingContext", "SequencingContext"]

# Provide convenient aliases matching the Rust API: callers sometimes try
# `sim_ctx.RoutingProgram` / `sim_ctx.SequencingProgram`. In Python these
# can simply point to the GP `Program` implementation.
try:
    from gp.GPtree import Program as _Program
except Exception:    
    _Program = None

if _Program is not None:
    RoutingProgram = _Program
    SequencingProgram = _Program
    __all__.extend(["RoutingProgram", "SequencingProgram"])
