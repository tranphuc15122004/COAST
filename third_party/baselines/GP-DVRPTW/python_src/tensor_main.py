#!/usr/bin/env python3
"""Python port of src/main.rs (approximate).

Responsibilities:
- Parse CLI (expect a problem CSV path).
- Load environment from `.env`.
- Build a `Problem` from CSV and run:
    - heuristics: evaluate a few hand-coded routing+sequencing pairs.
    - GP: run genetic programming loop (population init, evaluate, select,
        crossover/mutate) and log results.

Key functions/classes:
- `heuristics(problem)`: runs simple routing+sequencing programs and logs
    heuristic results.
- `gp(problem)`: runs the GP loop using `GPContext` from `gp.mod`.
- `Individual`: small wrapper holding routing/sequencing `Program`s and
    evaluation helpers used by the GP loop.

Notes:
- Module expects the `gp`, `sim`, and `log` packages to be importable from
    `python_src` (package imports are used). Environment variables (or `.env`)
    control logging and GP parameters.
"""
import os
import sys
import argparse
import random
import importlib.util
from contextlib import suppress
import numpy as np
from functools import partial
from typing import List
from dotenv import load_dotenv, find_dotenv

HERE = os.path.dirname(__file__)

# Repo root (parent of `python_src/`). Needed for `import python_src.*`.
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))


dotenv_path = find_dotenv()
if not dotenv_path:
    alt = os.path.normpath(os.path.join(HERE, '..', '.env'))
    if os.path.exists(alt):
        dotenv_path = alt
if dotenv_path:
    load_dotenv(dotenv_path)


# Ensure imports work whether executed as:
#   - `python -m python_src.tensor_main ...` (from repo root)
#   - `python python_src/tensor_main.py ...`
#   - `python ./tensor_main.py ...` (from within python_src/)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Some modules historically use `import sim.*` (without the `python_src.` prefix).
# Keeping `python_src/` itself on the path preserves compatibility.
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# load ports via normal package imports
from python_src.gp import mod as gp_mod
from python_src.gp import GPtree as gp_program
from python_src.log import logger as log_mod
from python_src.sim import mod as sim_mod
from python_src.sim import ctx as sim_ctx
from python_src.sim import problem as problem_mod

# Loggers
MAIN = log_mod.Logger("MAIN")
HEU = log_mod.Logger("HEU")
SIM = log_mod.Logger("SIM")
GP = log_mod.Logger("GP")
LASTPOP = log_mod.Logger("LASTPOP")
LASTROUTE = log_mod.Logger("LASTROUTE")
ROUTE = log_mod.Logger("ROUTE")
ROUTEEVAL = log_mod.Logger("ROUTEEVAL")
DEBUG = log_mod.Logger("DEBUG")

# Configs (match Rust defaults)
CONST_RATE = float(os.environ.get("CONST_RATE", "0.1"))
WEIGHT = float(os.environ.get("WEIGHT", "0.1"))
LATE_WEIGHT = float(os.environ.get("LATE_WEIGHT", "1"))
PENDING_WEIGHT = float(os.environ.get("PENDING_WEIGHT", "2"))
SEED = float(os.environ.get("SEED", "15122004"))

NUM_TIME_SLOT = float(os.environ.get("NUM_TIME_SLOT", "50.0"))
NUM_GEN = int(os.environ.get("NUM_GEN", "200"))
POP_SIZE = int(os.environ.get("POP_SIZE", "100"))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "6"))
CROSSOVER_RATE = float(os.environ.get("CROSSOVER_RATE", "0.8"))
MUTATION_RATE = float(os.environ.get("MUTATION_RATE", "0.1"))
TRAIN_FACTOR = float(os.environ.get("TRAIN_FACTOR", "0.2"))
STRESS_FACTOR = float(os.environ.get("STRESS_FACTOR", "1.0"))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def fitness(problem : problem_mod.Problem, result):
    distance, num_fail, total_delay = result

    # distance normalization (avoid div0)
    tot_dist = problem.truck_speed * problem.depot.close * float(problem.num_trucks)
    dist_norm = distance / tot_dist if tot_dist > 0 else 0.0

    # failures normalization
    nreq = len(problem.requests)
    fail_norm = num_fail / nreq

    max_total_delay = sum(max(0.0, problem.depot.close - r.close) for r in problem.requests)
    max_total_delay = max(1.0, max_total_delay)
    delay_norm = total_delay / max_total_delay

    return 0.1 * dist_norm + 0.2 * (delay_norm) + 0.7* fail_norm

def heuristics(problem : problem_mod.Problem):
    # Build simple programs similar to Rust's example.
    try:
        CR = sim_ctx.RoutingProgram.terminal(3)
    except Exception:
        CR = gp_program.Program.terminal(3)
    try:
        CS = sim_ctx.SequencingProgram.from_vec([gp_program.Node.Internal(5).into(), gp_program.Node.Terminal(0).into(), gp_program.Node.Terminal(4).into()])
    except Exception:
        CS = gp_program.Program.from_vec([gp_program.Node.Internal(5), gp_program.Node.Terminal(0), gp_program.Node.Terminal(4)])
    try:
        W = sim_ctx.SequencingProgram.terminal(3)
    except Exception:
        W = gp_program.Program.terminal(3)
    try:
        WIQ = sim_ctx.RoutingProgram.terminal(1)
    except Exception:
        WIQ = gp_program.Program.terminal(1)

    for name, r, s in [("C+C", CR, CS), ("C+W", CR, W), ("WIQ+C", WIQ, CS)]:
        simulation = sim_mod.soft_Simulation2(problem, r, s)
        result = simulation.simulate_until(problem.depot.close / NUM_TIME_SLOT, float("inf"))
        log_mod.log(HEU, "heuristic_result", name=name, result=result, fitness=fitness(problem, result))


# mỗi cá thể gồm 2 cây
class Individual:
    def __init__(self, routing, sequencing):
        self.routing = routing
        self.sequencing = sequencing
        self.result = None

    #khởi tạo quần thể
    @staticmethod
    def ramp_half_and_half(gpc  : gp_mod.GPContext):
        # generate routing programs using RoutingContext and sequencing programs
        # using SequencingContext so the generator can query terminal/internal counts
        try:
            r_ctx = sim_ctx.RoutingContext
            s_ctx = sim_ctx.SequencingContext
        except Exception:
            r_ctx = None
            s_ctx = None
        r_pop = gpc.ramp_half_and_half(context=r_ctx)
        s_pop = gpc.ramp_half_and_half(context=s_ctx)
        out = []
        out.extend(Individual(r, s) for r, s in zip(r_pop, s_pop))
        return out

    def crossover_with(self, gpc, other):
        r1, r2 = gpc.crossover(self.routing, other.routing)
        s1, s2 = gpc.crossover(self.sequencing, other.sequencing)
        return Individual(r1, s1), Individual(r2, s2)

    def mutate(self, gpc):
        return Individual(gpc.mutation(self.routing), gpc.mutation(self.sequencing))


    # Cần xem xét sửa hàm này
    def evaluate(self, cache, problem, time_slot):
        if self.result is not None:
            return self.result[3]
        cache_key = f"{self.routing}:{self.sequencing}"
        if cache_key in cache:
            dist, nb_fail ,total_delay, fit = cache[cache_key]
        else:
            sim = sim_mod.soft_Simulation2(problem, self.routing, self.sequencing)
            
            dist, nb_fail , total_delay = sim.simulate_until(time_slot, float("inf"))
            fit = fitness(problem, (dist, nb_fail , total_delay))
            cache[cache_key] = (dist, nb_fail , total_delay, fit)
        self.result = (dist, nb_fail , total_delay, fit)
        return fit


def select_parent(gpc, pop):
    # sample 8 and pick best by fitness
    idxs = random.sample(range(len(pop)), k=min(8, len(pop)))
    # lower fitness is better
    return min(idxs, key=lambda i: pop[i].result[3])


def gp(problem : problem_mod.Problem):
    time_slot = problem.depot.close / NUM_TIME_SLOT
    train_time_slot = time_slot / STRESS_FACTOR
    training_problem = problem.clone_training(time_slot * TRAIN_FACTOR, STRESS_FACTOR)

    # Construct GPContext -- adapt to Python port's constructor
    # allow deterministic runs by setting SEED env var
    if seed_env := (os.environ.get("SEED") or str(SEED)):
        try:
            seed_val = int(seed_env)
        except Exception:
            seed_val = None
    else:
        seed_val = None

    try:
        rng = random.Random(seed_val) if seed_val is not None else random.Random()
        gpc = gp_mod.GPContext(rng=rng, num_population=POP_SIZE, max_depth=MAX_DEPTH)
    except Exception:
        # fallback to attribute style
        gpc = gp_mod.GPContext()
        gpc.rng = random.Random(seed_val) if seed_val is not None else random.Random()
        gpc.num_population = POP_SIZE
        gpc.max_depth = MAX_DEPTH

    cache = {}
    pop : List[Individual] = Individual.ramp_half_and_half(gpc)

    for gen in range(1, NUM_GEN + 1):
        # Tính toán fittness cho tường cá thể
        for ind in pop:
            ind.evaluate(cache, training_problem, train_time_slot)

        # keep best individuals by fitness (lower is better)
        pop.sort(key=lambda i: i.result[3])
        pop = pop[:gpc.num_population]
        result = pop[0].result

        log_mod.log(
            GP,
            "new_gen",
            gen=gen,
            result=(result[0], result[1], result[2]),
            fitness=result[3],
            routing=str(pop[0].routing),
            sequencing=str(pop[0].sequencing),
        )
        
        # tính toán fittness trên toàn bộ dữ liệu cho cá thể tốt nhất
        sim = sim_mod.soft_Simulation2(problem, pop[0].routing, pop[0].sequencing)
        
        full_result = sim.simulate_until(time_slot, float("inf"))
        log_mod.log(GP, "full_result", result=full_result, fitness=fitness(problem, full_result))

        with suppress(Exception):
            log_mod.log(GP, "base64", routing=pop[0].routing.base64(), sequencing=pop[0].sequencing.base64())

        if gen == NUM_GEN:
            for vehicle in range(problem.num_trucks):
                v = sim.vehicles[vehicle]
                log_mod.log(LASTROUTE, "route_log", vehicle=vehicle, route=getattr(v, "route", None), dropped=getattr(v, "dropped", None))
            for i in pop:
                log_mod.log(LASTPOP, "lastpop", routing=str(i.routing), sequencing=str(i.sequencing))

        new_pop = list(pop)
        half = gpc.num_population // 2
        for _ in range(half):
            p1 = select_parent(gpc, pop)
            p2 = select_parent(gpc, pop)
            x = random.random()
            if x <= CROSSOVER_RATE:
                c1, c2 = pop[p1].crossover_with(gpc, pop[p2])
                new_pop.extend([c1, c2])
            elif x <= CROSSOVER_RATE + MUTATION_RATE:
                m1 = pop[p1].mutate(gpc)
                m2 = pop[p2].mutate(gpc)
                new_pop.extend([m1, m2])
            else:
                new_pop.append(pop[p1])
                new_pop.append(pop[p2])
        pop = new_pop


def main():
    parser = argparse.ArgumentParser(description="Run GP-DVRPTW on a tensor scenario file")
    parser.add_argument("scenario_path", help="Path to tensor dataset file")
    parser.add_argument("--instance-num", type=int, default=int(os.environ.get("PROBLEM_INSTANCE_NUM", "4")))
    parser.add_argument("--truck-speed", type=float, default=float(os.environ.get("PROBLEM_TRUCK_SPEED", "2")))
    parser.add_argument("--truck-capacity", type=float, default=float(os.environ.get("PROBLEM_TRUCK_CAPACITY", "200.0")))
    parser.add_argument("--num-trucks", type=int, default=int(os.environ.get("PROBLEM_NUM_TRUCKS", "5")))
    parser.add_argument(
        "--no-normalize-inputs",
        dest="normalize_inputs",
        action="store_false",
        help="Disable input normalization",
    )
    parser.set_defaults(normalize_inputs=_env_bool("PROBLEM_NORMALIZE_INPUTS", True))
    args = parser.parse_args()

    path = args.scenario_path
    if not os.path.exists(path):
        alt = os.path.join(PROJECT_ROOT, path)
        if os.path.exists(alt):
            path = alt

    problem = problem_mod.Problem.tensor_load(
        path,
        args.instance_num,
        args.truck_speed,
        args.truck_capacity,
        args.num_trucks,
        normalize_inputs=args.normalize_inputs,
    )
    #problem = problem_mod.Problem.load(path, 1.0, 1300.0, 10)
    
    log_mod.log(MAIN, "start")

    if not HEU.enabled() and not GP.enabled():
        print(
            "Nothing to do: both HEU and GP are disabled. "
            "Set LOG_HEU=stdout and/or LOG_GP=stdout (in .env or env vars) to run."
        )
        return

    if HEU.enabled():
        log_mod.log(MAIN, "heu_start")
        heuristics(problem)
    if GP.enabled():
        log_mod.log(MAIN, "gp_start")
        gp(problem)


if __name__ == "__main__":
    main()
