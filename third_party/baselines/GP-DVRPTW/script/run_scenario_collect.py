#!/usr/bin/env python3
import argparse
import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_env_overrides(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --env value '{item}'. Expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --env key in '{item}'")
        out[key] = value
    return out


def parse_json_events(raw_stdout: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for line in raw_stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def collect_summary(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    heuristic_events: List[Dict[str, Any]] = []
    gp_new_gen: List[Dict[str, Any]] = []
    gp_full_result: List[Dict[str, Any]] = []
    acs_result: Optional[Dict[str, Any]] = None
    acs_replayed: Optional[Dict[str, Any]] = None
    current_gen: Optional[int] = None

    for event in events:
        logger_name = event.get("__")
        message = event.get("_")

        if logger_name == "HEU" and message == "heuristic_result":
            heuristic_events.append(event)
            continue

        if logger_name == "GP" and message == "new_gen":
            gen_val = event.get("gen")
            try:
                current_gen = int(gen_val)
            except (TypeError, ValueError):
                current_gen = None
            gp_new_gen.append(event)
            continue

        if logger_name == "GP" and message == "full_result":
            full_event = dict(event)
            full_event["gen"] = current_gen
            gp_full_result.append(full_event)

        if logger_name == "ACS" and message == "acs_km_result":
            acs_result = event

        if logger_name == "ACS" and message == "acs_km_replayed":
            acs_replayed = event

    first_three_heuristics = heuristic_events[:3]

    best_result: Optional[Dict[str, Any]] = None
    if gp_full_result:
        candidate = [e for e in gp_full_result if isinstance(e.get("fitness"), (int, float))]
        if candidate:
            best_result = min(candidate, key=lambda e: float(e["fitness"]))
        else:
            best_result = gp_full_result[0]
    elif gp_new_gen:
        candidate = [e for e in gp_new_gen if isinstance(e.get("fitness"), (int, float))]
        if candidate:
            best_result = min(candidate, key=lambda e: float(e["fitness"]))
        else:
            best_result = gp_new_gen[0]

    final_generation: Optional[Dict[str, Any]] = None
    if gp_new_gen:
        with_gen: List[Tuple[int, Dict[str, Any]]] = []
        for e in gp_new_gen:
            try:
                with_gen.append((int(e.get("gen")), e))
            except (TypeError, ValueError):
                pass

        if with_gen:
            _, last_gen_event = max(with_gen, key=lambda x: x[0])
            final_generation = {
                "new_gen": last_gen_event,
                "full_result": next(
                    (x for x in gp_full_result if x.get("gen") == last_gen_event.get("gen")),
                    None,
                ),
            }
        else:
            final_generation = {"new_gen": gp_new_gen[-1], "full_result": None}

    return {
        "best_result": best_result,
        "final_generation": final_generation,
        "first_three_heuristics": first_three_heuristics,
        "acs_result": acs_result,
        "acs_replayed": acs_replayed,
        "counts": {
            "events": len(events),
            "heuristics": len(heuristic_events),
            "gp_new_gen": len(gp_new_gen),
            "gp_full_result": len(gp_full_result),
            "acs": 1 if acs_result else 0,
        },
    }


def pin_process_to_single_core(proc: subprocess.Popen, core_index: int) -> None:
    cpu_total = os.cpu_count() or 1
    if core_index < 0 or core_index >= cpu_total:
        raise ValueError(f"Invalid core index {core_index}. This machine has {cpu_total} CPU cores.")

    if os.name == "nt":
        mask = 1 << core_index
        handle = proc._handle
        result = ctypes.windll.kernel32.SetProcessAffinityMask(handle, ctypes.c_size_t(mask))
        if result == 0:
            err_code = ctypes.windll.kernel32.GetLastError()
            raise OSError(f"SetProcessAffinityMask failed with error code {err_code}")
        return

    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(proc.pid, {core_index})
        return

    raise OSError("CPU affinity is not supported on this platform")


def resolve_core_index(core_arg: str, child_pid: int) -> Tuple[int, str]:
    cpu_total = os.cpu_count() or 1
    raw = (core_arg or "auto").strip().lower()
    if raw == "auto":
        core = child_pid % cpu_total
        return core, "auto(pid_mod_cpu_count)"

    core = int(raw)
    if core < 0 or core >= cpu_total:
        raise ValueError(f"Invalid core index {core}. This machine has {cpu_total} CPU cores.")
    return core, "manual"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one scenario with an input path and collect: best result, "
            "final generation result, and first 3 heuristic results."
        )
    )
    parser.add_argument("scenario_path", help="Path to the scenario CSV")
    parser.add_argument(
        "--entry",
        default="python_src/main.py",
        help="Entrypoint script relative to repo root (default: python_src/main.py)",
    )
    parser.add_argument("--out", help="Optional output JSON file path (default: output/<scenario_name>.json)")
    parser.add_argument(
        "--silent-child",
        action="store_true",
        help="Do not stream child process output in real time",
    )
    parser.add_argument(
        "--no-print-summary",
        action="store_true",
        help="Do not print final JSON summary to stdout (still writes output file)",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra env override in KEY=VALUE format (can be used multiple times)",
    )
    parser.add_argument(
        "--core",
        default="auto",
        help="Pin child process to one CPU core index, or 'auto' (default: auto)",
    )
    parser.add_argument("--num-trucks", type=int, help="Override number of vehicles" , default=10)
    parser.add_argument("--truck-capacity", type=float, help="Override vehicle capacity" , default= 1300.0)
    parser.add_argument("--truck-speed", type=float, help="Override vehicle speed" , default=1.0)
    parser.add_argument("--instance-num", type=int, help="Instance index for tensor datasets" , default= 0)
    parser.add_argument(
        "--no-normalize-inputs",
        action="store_true",
        help="Disable input normalization in problem loader",
    )

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    scenario_path = Path(args.scenario_path)
    if not scenario_path.is_absolute():
        scenario_path = (repo_root / scenario_path).resolve()

    if not scenario_path.exists():
        print(f"Scenario path not found: {scenario_path}", file=sys.stderr)
        return 2

    entry_path = Path(args.entry)
    if not entry_path.is_absolute():
        entry_path = (repo_root / entry_path).resolve()

    if not entry_path.exists():
        print(f"Entry script not found: {entry_path}", file=sys.stderr)
        return 2

    try:
        env_overrides = parse_env_overrides(args.env)
    except ValueError as err:
        print(str(err), file=sys.stderr)
        return 2

    env = os.environ.copy()
    env.update(
        {
            "LOG_HEU": "stdout",
            "LOG_GP": "stdout",
            "LOG_MAIN": "stdout",
            "LOG_ACS": "stdout",
        }
    )
    env.update(env_overrides)

    cmd = [sys.executable, str(entry_path), str(scenario_path)]
    if args.num_trucks is not None:
        cmd.extend(["--num-trucks", str(args.num_trucks)])
    if args.truck_capacity is not None:
        cmd.extend(["--truck-capacity", str(args.truck_capacity)])
    if args.truck_speed is not None:
        cmd.extend(["--truck-speed", str(args.truck_speed)])
    if args.instance_num is not None:
        cmd.extend(["--instance-num", str(args.instance_num)])
    if args.no_normalize_inputs:
        cmd.append("--no-normalize-inputs")
    print(f"[runner] start: {' '.join(cmd)}", flush=True)

    merged_lines: List[str] = []
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        selected_core, core_mode = resolve_core_index(args.core, proc.pid)
        pin_process_to_single_core(proc, selected_core)
        print(f"[runner] child pinned to core={selected_core} ({core_mode})", flush=True)
    except Exception as err:
        proc.kill()
        proc.wait()
        print(f"[runner] failed to set CPU affinity: {err}", file=sys.stderr, flush=True)
        return 2

    assert proc.stdout is not None
    for line in proc.stdout:
        merged_lines.append(line)
        if not args.silent_child:
            print(line, end="")

    return_code = proc.wait()
    merged_output = "".join(merged_lines)
    print(f"[runner] done: return_code={return_code}", flush=True)

    events = parse_json_events(merged_output)
    summary = collect_summary(events)
    summary["meta"] = {
        "command": cmd,
        "return_code": return_code,
        "core": selected_core,
        "core_mode": core_mode,
        "scenario_path": str(scenario_path),
        "entry": str(entry_path),
    }

    if return_code != 0:
        summary["child_output_tail"] = "\n".join(merged_output.splitlines()[-50:])

    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = (repo_root / out_path).resolve()
    else:
        out_path = (repo_root / "output" / f"{scenario_path.stem}.json").resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload, encoding="utf-8")
    print(f"[runner] summary saved: {out_path}", flush=True)

    if not args.no_print_summary:
        print(payload)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
