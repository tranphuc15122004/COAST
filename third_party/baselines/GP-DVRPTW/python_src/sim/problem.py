"""
Problem representation and CSV loader (Python port of src/sim/problem.rs).
Responsibilities:
- `Request`: simple data holder for a single request (position, demand,
    time windows and service properties).
- `Problem`: holds `depot`, `requests`, and vehicle parameters and provides
    CSV loading via `Problem.load(...)` and training-time cloning via
    `clone_training(...)` used to build smaller/stressed training problems.

Notes:
- `Problem.load` expects the dataset CSV format used by the project and
    extracts fields into `Request` instances; the first row is treated as
    the depot.
"""
from __future__ import annotations

from typing import List, Optional
import csv
import torch
from problems._data_dtw import DVRPTW_Dataset 
                

# mỗi request là một dòng trong file CSV
class Request:
    def __init__(self, idx: int, x: float, y: float, demand: float, open: float, close: float, service_time: float, time: float):
        self.idx = (idx)
        self.x = (x)
        self.y = (y)
        self.demand = (demand)
        self.open = (open)
        self.close = (close)
        self.service_time = (service_time)
        self.time = (time)


class Problem:
    def __init__(self, depot: Request, requests: List[Request], truck_speed: float, truck_capacity: float, num_trucks: int):
        self.depot = depot
        self.requests = requests
        self.truck_speed = (truck_speed)
        self.truck_capacity = (truck_capacity)
        self.num_trucks = (num_trucks)

    @staticmethod
    def _normalize_like_dtw(
        depot: Request,
        requests: List[Request],
        truck_speed: float,
        truck_capacity: float,
    ) -> tuple[float, float]:
        loc_values: List[float] = [depot.x, depot.y]
        for req in requests:
            loc_values.extend([req.x, req.y])

        loc_max = max(loc_values)
        loc_min = min(loc_values)
        loc_scl = loc_max - loc_min
        t_scl = depot.close

        if loc_scl > 0.0:
            depot.x = (depot.x - loc_min) / loc_scl
            depot.y = (depot.y - loc_min) / loc_scl
            for req in requests:
                req.x = (req.x - loc_min) / loc_scl
                req.y = (req.y - loc_min) / loc_scl

        if truck_capacity > 0.0:
            for req in requests:
                req.demand /= truck_capacity
            truck_capacity = 1.0

        if t_scl > 0.0:
            depot.open /= t_scl
            depot.close /= t_scl
            depot.service_time /= t_scl
            depot.time /= t_scl

            for req in requests:
                req.open /= t_scl
                req.close /= t_scl
                req.service_time /= t_scl
                req.time /= t_scl

        if loc_scl > 0.0 and t_scl > 0.0:
            truck_speed = truck_speed * t_scl / loc_scl

        return truck_speed, truck_capacity

    @staticmethod
    def load(
        csv_path: str,
        truck_speed: float,
        truck_capacity: float,
        num_trucks: int,
        normalize_inputs: bool = True,
    ) -> "Problem":
        requests: List[Request] = []
        with open(csv_path, newline='') as fh:
            reader = csv.reader(fh)
            # skip header
            try:
                next(reader)
            except StopIteration:
                raise ValueError("CSV is empty")
            for idx, row in enumerate(reader):
                if not row or all(not c.strip() for c in row):
                    continue
                # parse floats, allow extra columns
                vals = [float(tok) for tok in row]
                req = Request(
                    idx=idx,
                    x=float(vals[0]),
                    y=float(vals[1]),
                    demand=float(vals[2]),
                    open=float(vals[3]),
                    close=float(vals[4]),
                    service_time= float(vals[5]),
                    time=float(vals[7]),
                )
                requests.append(req)

        if not requests:
            raise ValueError("no requests found in CSV")
        depot = requests.pop(0)
        if normalize_inputs:
            truck_speed, truck_capacity = Problem._normalize_like_dtw(
                depot,
                requests,
                truck_speed,
                truck_capacity,
            )
        return Problem(depot, requests, truck_speed, truck_capacity, num_trucks)
    
    @staticmethod
    def tensor_load(
        tensor_load: str,
        instance_num: int = 0,
        truck_speed: Optional[float] = None,
        truck_capacity: float = 1.0,
        num_trucks: int = 1,
        normalize_inputs: bool = True,
    ) -> "Problem":
        # Load a dataset saved by the tools/data pipeline (torch file)
        torch.serialization.add_safe_globals([DVRPTW_Dataset])
        ds = torch.load(tensor_load, weights_only=False)

        # Extract nodes tensor (support both dict and object with attribute)
        if isinstance(ds, dict) and 'nodes' in ds:
            nodes = ds['nodes']
        elif hasattr(ds, 'nodes'):
            nodes = ds.nodes
        else:
            raise ValueError("dataset does not contain 'nodes' tensor")

        # Prefer dataset speed if caller did not provide one.
        if truck_speed is None:
            if isinstance(ds, dict) and 'veh_speed' in ds:
                truck_speed = float(ds['veh_speed'])
            elif hasattr(ds, 'veh_speed'):
                truck_speed = float(ds.veh_speed)
            else:
                truck_speed = 1.0

        # If nodes has a batch dimension, select the requested scenario
        if nodes.dim() == 3:
            batch_size = nodes.size(0)
            if not (0 <= instance_num < batch_size):
                raise IndexError(f"instance_num {instance_num} out of range (batch_size={batch_size})")
            inst = nodes[instance_num]
        else:
            # single-instance tensor, ignore instance_num
            inst = nodes

        n_nodes, feat = inst.size()

        # Read depot (first row)
        depot_row = inst[0]
        depot_x = float(depot_row[0].item()) if feat >= 1 else 0.0
        depot_y = float(depot_row[1].item()) if feat >= 2 else 0.0
        depot_close = float(depot_row[4].item()) if feat >= 5 else 0.0
        depot = Request(idx=0, x=depot_x, y=depot_y, demand=0.0, open=0.0, close=depot_close, service_time=0.0, time=0.0)

        requests: List[Request] = []
        for j in range(1, n_nodes):
            row = inst[j]
            x = float(row[0].item()) if feat >= 1 else 0.0
            y = float(row[1].item()) if feat >= 2 else 0.0
            demand = float(row[2].item()) if feat >= 3 else 0.0
            open_t = float(row[3].item()) if feat >= 4 else 0.0
            close_t = float(row[4].item()) if feat >= 5 else open_t
            service_time = float(row[5].item()) if feat >= 6 else 0.0
            time = float(row[6].item()) if feat >= 7 else 0.0

            req = Request(idx=j, x=x, y=y, demand=demand, open=open_t, close=close_t, service_time=service_time, time=time)
            requests.append(req)

        if normalize_inputs:
            truck_speed, truck_capacity = Problem._normalize_like_dtw(
                depot,
                requests,
                truck_speed,
                truck_capacity,
            )

        #requests.sort(key=lambda x: x.time)
        return Problem(depot, requests, truck_speed, truck_capacity, num_trucks)

    def clone_training(self, time_limit: float, stress_factor: float) -> "Problem":
        requests: List[Request] = []
        current_index = 0
        turn = 0.0
        original_requests = list(self.requests)
        n = len(original_requests)
        for req in original_requests:
            nr = Request(
                idx=req.idx,
                x=req.x * stress_factor,
                y=req.y * stress_factor,
                demand=req.demand,
                open=req.open * stress_factor,
                close=req.close * stress_factor,
                service_time=req.service_time * stress_factor,
                time=req.time,
            )
            if nr.time > time_limit:
                # select a time reference request
                time_req = original_requests[current_index % n]
                nr.time = time_limit * turn + (time_req.time + time_req.open * 1.5) / 2.5
                nr.open = time_limit * turn + time_req.open
                nr.close = time_limit * turn + time_req.close

                current_index += 1
                if original_requests[current_index % n].time > time_limit:
                    current_index = 0
                    turn += 1.0

            requests.append(nr)

        return Problem(self.depot, requests, self.truck_speed, self.truck_capacity, self.num_trucks)

    def total_demand(self) -> float:
        return sum(r.demand for r in self.requests)


__all__ = ["Request", "Problem"]

