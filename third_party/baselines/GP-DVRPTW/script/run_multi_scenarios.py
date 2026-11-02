#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set


def parse_env_overrides(items: List[str]) -> List[str]:
    normalized: List[str] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --env value '{item}'. Expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --env key in '{item}'")
        normalized.append(f"{key}={value}")
    return normalized


def parse_cores(cores_arg: Optional[str], core_count: Optional[int] = None) -> List[int]:
    cpu_total = os.cpu_count() or 1
    if core_count is not None:
        if core_count <= 0:
            raise ValueError("--core-count must be > 0")
        if core_count > cpu_total:
            raise ValueError(
                f"--core-count {core_count} exceeds available cores ({cpu_total})."
            )
        return list(range(core_count))

    if not cores_arg:
        return list(range(cpu_total))

    out: List[int] = []
    for raw in cores_arg.split(","):
        raw = raw.strip()
        if not raw:
            continue
        core = int(raw)
        if core < 0 or core >= cpu_total:
            raise ValueError(f"Core index {core} out of range. This machine has {cpu_total} cores.")
        out.append(core)

    if not out:
        raise ValueError("No valid cores parsed from --cores")
    return out


def expand_scenario_inputs(repo_root: Path, raw_inputs: List[str], from_file: Optional[str]) -> List[Path]:
    collected: List[Path] = []
    skipped: List[str] = []

    def _add_path(path: Path) -> None:
        if path.exists() and path.is_file() and path.suffix.lower() == ".csv":
            collected.append(path.resolve())

    def _expand_one(raw: str) -> None:
        raw = raw.strip().lstrip("\ufeff")
        if not raw:
            return
        path = Path(raw)
        has_glob = any(ch in raw for ch in ["*", "?", "["])
        if has_glob:
            base = repo_root if not path.is_absolute() else Path(".")
            for p in base.glob(raw):
                if p.is_file() and p.suffix.lower() == ".csv":
                    collected.append(p.resolve())
            if not any(base.glob(raw)):
                skipped.append(raw)
            return

        if not path.is_absolute():
            path = (repo_root / path).resolve()

        if path.is_dir():
            for p in sorted(path.rglob("*.csv")):
                collected.append(p.resolve())
            if not any(path.rglob("*.csv")):
                skipped.append(raw)
            return

        if path.exists() and path.is_file() and path.suffix.lower() == ".csv":
            collected.append(path.resolve())
        else:
            skipped.append(raw)

    for item in raw_inputs:
        _expand_one(item)

    if from_file:
        list_path = Path(from_file)
        if not list_path.is_absolute():
            list_path = (repo_root / list_path).resolve()
        if list_path.exists():
            for line in list_path.read_text(encoding="utf-8").splitlines():
                candidate = line.strip()
                if not candidate or candidate.startswith("#"):
                    continue
                _expand_one(candidate)

    if not collected:
        default_dir = repo_root / "datasets"
        if default_dir.exists():
            for p in sorted(default_dir.rglob("*.csv")):
                collected.append(p.resolve())

    unique: List[Path] = []
    seen: Set[str] = set()
    for path in sorted(collected):
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)

    for item in skipped:
        print(f"[multi] skip unresolved input: {item}")
    return unique


def stream_output(proc: subprocess.Popen, prefix: str, lock: threading.Lock, sink: List[str]) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        sink.append(line)
        with lock:
            print(f"[{prefix}] {line}", end="")


def make_output_path(repo_root: Path, scenario_path: Path, output_dir: Path) -> Path:
    try:
        rel = scenario_path.resolve().relative_to(repo_root.resolve())
        stem = str(rel.with_suffix("")).replace("\\", "__").replace("/", "__")
    except Exception:
        stem = scenario_path.stem
    return output_dir / f"{stem}.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Turnkey parallel runner for multiple scenarios, one process per CPU core."
    )
    parser.add_argument("scenarios", nargs="*", help="Scenario inputs: csv files, directories, or glob patterns")
    parser.add_argument(
        "--runner",
        default="script/run_scenario_collect.py",
        help="Runner script path (default: script/run_scenario_collect.py)",
    )
    parser.add_argument(
        "--from-file",
        help="Text file containing scenario paths/patterns (one per line)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to save per-scenario summary JSON files (default: output)",
    )
    parser.add_argument(
        "--cores",
        help="Comma-separated core list, e.g. 0,1,2 (default: auto-detect from machine)",
    )
    parser.add_argument(
        "--core-count",
        type=int,
        help="Use first N cores (0..N-1), e.g. 30 means cores 0-29",
    )
    parser.add_argument(
        "--core-mode",
        choices=["balanced", "self"],
        default="balanced",
        help="balanced: manager auto-assigns unique cores; self: child self-selects core",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=0,
        help="Max concurrent processes (default: auto, equals selected cores and capped by scenarios)",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Forward env override to each scenario runner in KEY=VALUE format",
    )
    parser.add_argument(
        "--silent-child",
        action="store_true",
        help="Pass --silent-child to each scenario runner",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Retry count for failed scenarios (default: 1)",
    )
    parser.add_argument(
        "--show-child-summary",
        action="store_true",
        help="Allow each child process to print its full JSON summary",
    )
    parser.add_argument("--num-trucks", type=int, help="Override number of vehicles" , default= 10)
    parser.add_argument("--truck-capacity", type=float, help="Override vehicle capacity" , default= 1300.0)
    parser.add_argument("--truck-speed", type=float, help="Override vehicle speed" , default= 1.0)
    parser.add_argument("--instance-num", type=int, help="Instance index for tensor datasets" , default= 0)
    parser.add_argument(
        "--no-normalize-inputs",
        action="store_true",
        help="Disable input normalization in problem loader",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    runner_path = Path(args.runner)
    if not runner_path.is_absolute():
        runner_path = (repo_root / runner_path).resolve()
    if not runner_path.exists():
        print(f"Runner not found: {runner_path}", file=sys.stderr)
        return 2

    try:
        env_overrides = parse_env_overrides(args.env)
        if args.cores and args.core_count is not None:
            raise ValueError("Use either --cores or --core-count, not both.")
        cores = parse_cores(args.cores, args.core_count)
    except Exception as err:
        print(str(err), file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = expand_scenario_inputs(repo_root, args.scenarios, args.from_file)

    if not scenarios:
        print("No valid scenario paths.", file=sys.stderr)
        return 2

    auto_core_selection = not args.cores and args.core_count is None

    if args.max_parallel > 0:
        max_parallel = args.max_parallel
    else:
        max_parallel = len(cores)
        if auto_core_selection:
            max_parallel = min(max_parallel, len(scenarios))

    if args.core_mode == "balanced":
        max_parallel = min(max_parallel, len(cores))
    if max_parallel <= 0:
        print("max_parallel must be > 0", file=sys.stderr)
        return 2

    if args.retries < 0:
        print("--retries must be >= 0", file=sys.stderr)
        return 2

    print(
        f"[multi] scenarios={len(scenarios)}, core_mode={args.core_mode}, cores={cores}, "
        f"max_parallel={max_parallel}, retries={args.retries}, auto_core_selection={auto_core_selection}",
        flush=True,
    )

    available_cores: List[int] = list(cores)
    pending: List[Dict[str, object]] = [{"scenario": s, "attempt": 1} for s in scenarios]
    running: Dict[int, Dict[str, object]] = {}
    completed: Dict[str, Dict[str, object]] = {}
    print_lock = threading.Lock()

    try:
        while pending or running:
            while pending and len(running) < max_parallel:
                if args.core_mode == "balanced" and not available_cores:
                    break

                job = pending.pop(0)
                scenario = job["scenario"]
                attempt = int(job["attempt"])
                core: Optional[int] = None
                if args.core_mode == "balanced":
                    core = available_cores.pop(0)
                out_path = make_output_path(repo_root, scenario, output_dir)

                cmd = [
                    sys.executable,
                    str(runner_path),
                    str(scenario),
                    "--out",
                    str(out_path),
                ]
                if args.core_mode == "balanced":
                    cmd.extend(["--core", str(core)])
                else:
                    cmd.extend(["--core", "auto"])
                if args.silent_child:
                    cmd.append("--silent-child")
                if not args.show_child_summary:
                    cmd.append("--no-print-summary")
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
                for item in env_overrides:
                    cmd.extend(["--env", item])

                proc = subprocess.Popen(
                    cmd,
                    cwd=str(repo_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                sink: List[str] = []
                core_label = f"core{core}" if core is not None else "core-auto"
                prefix = f"{scenario.name}|{core_label}|try{attempt}"
                t = threading.Thread(target=stream_output, args=(proc, prefix, print_lock, sink), daemon=True)
                t.start()
                running[proc.pid] = {
                    "proc": proc,
                    "thread": t,
                    "core": core,
                    "scenario": scenario,
                    "output": out_path,
                    "sink": sink,
                    "started": time.time(),
                    "attempt": attempt,
                }
                print(
                    f"[multi] started: scenario={scenario.name}, core={core if core is not None else 'auto'}, "
                    f"attempt={attempt}",
                    flush=True,
                )

            finished_pids: List[int] = []
            for pid, info in running.items():
                proc = info["proc"]
                code = proc.poll()
                if code is None:
                    continue

                thread = info["thread"]
                thread.join(timeout=1.0)
                core = info["core"]
                elapsed = time.time() - float(info["started"])
                scenario = info["scenario"]
                output_path = info["output"]
                attempt = int(info["attempt"])

                if core is not None:
                    available_cores.append(int(core))
                    available_cores.sort()

                ok = code == 0 and output_path.exists()
                status = "OK" if ok else "FAIL"
                print(
                    f"[multi] finished: scenario={scenario.name}, core={core}, status={status}, "
                    f"code={code}, elapsed={elapsed:.1f}s, output={output_path}",
                    flush=True,
                )

                key = str(scenario)
                if ok:
                    completed[key] = {
                        "scenario": key,
                        "status": "OK",
                        "attempts": attempt,
                        "core": core,
                        "output": str(output_path),
                        "return_code": int(code),
                        "elapsed_sec": round(elapsed, 3),
                    }
                else:
                    if attempt <= args.retries:
                        pending.append({"scenario": scenario, "attempt": attempt + 1})
                        print(
                            f"[multi] retry queued: scenario={scenario.name}, next_attempt={attempt + 1}",
                            flush=True,
                        )
                    else:
                        completed[key] = {
                            "scenario": key,
                            "status": "FAIL",
                            "attempts": attempt,
                            "core": core,
                            "output": str(output_path),
                            "return_code": int(code),
                            "elapsed_sec": round(elapsed, 3),
                        }
                finished_pids.append(pid)

            for pid in finished_pids:
                del running[pid]

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("[multi] interrupted, terminating running processes...", flush=True)
        for info in running.values():
            proc = info["proc"]
            try:
                proc.terminate()
            except Exception:
                pass
        time.sleep(0.5)
        for info in running.values():
            proc = info["proc"]
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
        return 130

    total = len(scenarios)
    ok_count = sum(1 for item in completed.values() if item.get("status") == "OK")
    fail_count = total - ok_count

    report = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "total": total,
            "ok": ok_count,
            "failed": fail_count,
            "core_mode": args.core_mode,
            "cores": cores,
            "max_parallel": max_parallel,
            "retries": args.retries,
        },
        "results": sorted(completed.values(), key=lambda x: x["scenario"]),
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"run_summary_{stamp}.json"
    latest_path = output_dir / "run_summary_latest.json"
    report_text = json.dumps(report, ensure_ascii=False, indent=2)
    report_path.write_text(report_text, encoding="utf-8")
    latest_path.write_text(report_text, encoding="utf-8")

    print(f"[multi] run summary saved: {report_path}", flush=True)
    print(f"[multi] latest summary: {latest_path}", flush=True)

    if fail_count > 0:
        print(f"[multi] done with warnings: ok={ok_count}, failed={fail_count}", flush=True)
        return 1

    print(f"[multi] done: all scenarios completed (ok={ok_count})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
