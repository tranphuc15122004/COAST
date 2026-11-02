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
import re
import subprocess
import argparse
import random
import importlib.util
from functools import partial
from typing import List, Optional, Tuple
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
#   - `python -m python_src.main ...` (from repo root)
#   - `python python_src/main.py ...`
#   - `python ./main.py ...` (from within python_src/)
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
ACS = log_mod.Logger("ACS")

# Configs (match Rust defaults)
CONST_RATE = float(os.environ.get("CONST_RATE", "0.1"))
WEIGHT = float(os.environ.get("WEIGHT", "0.1"))
LATE_WEIGHT = float(os.environ.get("LATE_WEIGHT", "1"))
PENDING_WEIGHT = float(os.environ.get("PENDING_WEIGHT", "2"))


NUM_TIME_SLOT = float(os.environ.get("NUM_TIME_SLOT", "50.0"))
NUM_GEN = int(os.environ.get("NUM_GEN", "100"))
POP_SIZE = int(os.environ.get("POP_SIZE", "100"))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "6"))
CROSSOVER_RATE = float(os.environ.get("CROSSOVER_RATE", "0.8"))
MUTATION_RATE = float(os.environ.get("MUTATION_RATE", "0.1"))
TRAIN_FACTOR = float(os.environ.get("TRAIN_FACTOR", "0.2"))
STRESS_FACTOR = float(os.environ.get("STRESS_FACTOR", "1.0"))

# ACS-KM configuration
ACS_KM_JAR = os.environ.get("ACS_KM_JAR", os.path.join(PROJECT_ROOT, "acs-km", "build", "libs", "acs-km-1.0-SNAPSHOT.jar"))
ACS_KM_TIME_LIMIT_NS = os.environ.get("ACS_KM_TIME_LIMIT_NS", "60000000000")  # 60 seconds in ns
ACS_KM_WEIGHT = os.environ.get("ACS_KM_WEIGHT", "0.5")
ACS_KM_FLAGS = os.environ.get("ACS_KM_FLAGS", "-l")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def fitness(problem, result):
    distance, pending, total_late = result
    return distance + LATE_WEIGHT * total_late + PENDING_WEIGHT * float(pending)


def heuristics(problem):
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
        for r, s in zip(r_pop, s_pop):
            out.append(Individual(r, s))
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
            return self.result[2]
        cache_key = f"{self.routing}:{self.sequencing}"
        if cache_key in cache:
            dist, pending, total_late, fit = cache[cache_key]
        else:
            sim = sim_mod.soft_Simulation2(problem, self.routing, self.sequencing)
            
            dist, pending, total_late = sim.simulate_until(time_slot, float("inf"))
            fit = fitness(problem, (dist, pending, total_late))
            cache[cache_key] = (dist, pending, total_late, fit)
        self.result = (dist, pending, total_late, fit)
        return fit


def select_parent(gpc, pop):
    # sample 8 and pick best by fitness
    idxs = random.sample(range(len(pop)), k=min(8, len(pop)))
    best = min(idxs, key=lambda i: pop[i].result[3])
    return best


def gp(problem : problem_mod.Problem):
    time_slot = problem.depot.close / NUM_TIME_SLOT
    train_time_slot = time_slot / STRESS_FACTOR
    training_problem = problem.clone_training(time_slot * TRAIN_FACTOR, STRESS_FACTOR)

    # Construct GPContext -- adapt to Python port's constructor
    # allow deterministic runs by setting SEED env var
    seed_env = os.environ.get("SEED")
    if seed_env is not None and seed_env != "":
        try:
            seed_val = int(seed_env)
        except Exception:
            seed_val = None
    else:
        seed_val = None

    try:
        if seed_val is not None:
            rng = random.Random(seed_val)
        else:
            rng = random.Random()
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

        pop.sort(key=lambda i: i.result[3])
        pop = pop[:gpc.num_population]
        result = pop[0].result

        log_mod.log(GP, "new_gen", gen=gen, result=(result[0], result[1], result[2]), fitness=result[3], routing=str(pop[0].routing), sequencing=str(pop[0].sequencing))
        
        # tính toán fittness trên toàn bộ dữ liệu cho cá thể tốt nhất
        sim = sim_mod.soft_Simulation2(problem, pop[0].routing, pop[0].sequencing)
        
        full_result = sim.simulate_until(time_slot, float("inf"))
        log_mod.log(GP, "full_result", result=full_result, fitness=fitness(problem, full_result))

        try:
            log_mod.log(GP, "base64", routing=pop[0].routing.base64(), sequencing=pop[0].sequencing.base64())
        except Exception:
            pass

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


def parse_acs_km_output(stdout: str) -> Optional[dict]:
    lines = stdout.splitlines()
    result: dict = {}
    parsing_routes = False
    routes: List[List[int]] = []
    for line in lines:
        m = re.search(r'Final best solution >> No\. of used vehicles=(\d+) total tours length=([\d.]+) \(scalled value = ([\d.]+)\)', line)
        if m:
            result["used_vehicles"] = int(m.group(1))
            result["total_tour_length"] = float(m.group(2))
            result["scaled_value"] = float(m.group(3))
            parsing_routes = True
            continue
        if parsing_routes:
            if re.search(r'Total number of evaluations', line):
                parsing_routes = False
            else:
                parts = line.strip().split()
                if parts and all(p.replace('.', '', 1).lstrip('-').isdigit() for p in parts):
                    ints = [int(x) for x in parts]
                    if ints:
                        routes.append(ints)
        m = re.search(r'Total number of evaluations: (\d+)', line)
        if m:
            result["evaluations"] = int(m.group(1))
        m = re.search(r'Total number of feasible solutions: (\d+)', line)
        if m:
            result["feasible_solutions"] = int(m.group(1))
        if re.search(r'The final solution is valid \(feasible\)', line):
            result["feasible"] = True
        elif re.search(r'The final solution is not valid \(feasible\)', line):
            result["feasible"] = False
    if routes:
        result["routes"] = routes
    return result if result else None


def run_acs_km(csv_path: str) -> Optional[dict]:
    if not os.path.exists(ACS_KM_JAR):
        log_mod.log(ACS, "acs_km_error", error=f"JAR not found: {ACS_KM_JAR}")
        return None
    if not os.path.exists(csv_path):
        log_mod.log(ACS, "acs_km_error", error=f"CSV not found: {csv_path}")
        return None

    cmd = ["java", "-jar", ACS_KM_JAR, csv_path, ACS_KM_TIME_LIMIT_NS, ACS_KM_WEIGHT]
    if ACS_KM_FLAGS:
        cmd.extend(ACS_KM_FLAGS.split())

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        log_mod.log(ACS, "acs_km_error", error="ACS-KM timed out after 300s")
        return None

    if proc.returncode != 0:
        log_mod.log(ACS, "acs_km_error", returncode=proc.returncode, stderr=proc.stderr[:500])
        return None

    result = parse_acs_km_output(proc.stdout)
    return result


def main():
    parser = argparse.ArgumentParser(description="Run GP-DVRPTW on a CSV scenario")
    parser.add_argument("scenario_path", help="Path to scenario CSV")
    parser.add_argument("--num-trucks", type=int, default=int(os.environ.get("PROBLEM_NUM_TRUCKS", "10")))
    parser.add_argument("--truck-capacity", type=float, default=float(os.environ.get("PROBLEM_TRUCK_CAPACITY", "1300.0")))
    parser.add_argument("--truck-speed", type=float, default=float(os.environ.get("PROBLEM_TRUCK_SPEED", "1.0")))
    parser.add_argument("--instance-num", type=int, default=int(os.environ.get("PROBLEM_INSTANCE_NUM", "0")))
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
        # Common invocation: run from within `python_src/` and pass a path like
        # `datasets/...` which actually lives at the repo root.
        alt = os.path.join(PROJECT_ROOT, path)
        if os.path.exists(alt):
            path = alt
    problem = problem_mod.Problem.load(
        path,
        args.truck_speed,
        args.truck_capacity,
        args.num_trucks,
        normalize_inputs=args.normalize_inputs,
    )

    log_mod.log(MAIN, "start")

    if not HEU.enabled() and not GP.enabled() and not ACS.enabled():
        print(
            "Nothing to do: HEU, GP, and ACS are all disabled. "
            "Set LOG_HEU=stdout, LOG_GP=stdout, and/or LOG_ACS=stdout (in .env or env vars) to run."
        )
        return

    if HEU.enabled():
        log_mod.log(MAIN, "heu_start")
        heuristics(problem)
    if GP.enabled():
        log_mod.log(MAIN, "gp_start")
        gp(problem)
    if ACS.enabled():
        log_mod.log(MAIN, "acs_start")
        acs_result = run_acs_km(args.scenario_path)
        if acs_result:
            log_mod.log(ACS, "acs_km_result", **{k: v for k, v in acs_result.items() if k != "routes"})
            raw_routes = acs_result.get("routes")
            if raw_routes:
                python_routes: List[List[int]] = []
                for raw_route in raw_routes:
                    py_route = [c for c in raw_route if c > 0]
                    python_routes.append(py_route)
                replayer = sim_mod.RouteReplayer(problem)
                rd, rp, rtd = replayer.evaluate(python_routes)
                log_mod.log(ACS, "acs_km_replayed", total_distance=rd, pending=rp, total_delay=rtd)
        else:
            log_mod.log(ACS, "acs_km_result", error="failed to run ACS-KM")


if __name__ == "__main__":
    main()
