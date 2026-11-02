#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


OUTPUT_DIRS_DEFAULT = [
    "output/100",
    "output/200",
    "output/400",
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_num(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def scenario_key_from_path(raw_path: str) -> str:
    p = Path(raw_path)
    return p.stem.lower()


def parse_cmd_params(cmd: List[str]) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[int]]:
    num_trucks: Optional[int] = None
    truck_capacity: Optional[float] = None
    truck_speed: Optional[float] = None
    instance_num: Optional[int] = None

    for i, token in enumerate(cmd):
        if token == "--num-trucks" and i + 1 < len(cmd):
            try:
                num_trucks = int(cmd[i + 1])
            except ValueError:
                pass
        elif token == "--truck-capacity" and i + 1 < len(cmd):
            try:
                truck_capacity = float(cmd[i + 1])
            except ValueError:
                pass
        elif token == "--truck-speed" and i + 1 < len(cmd):
            try:
                truck_speed = float(cmd[i + 1])
            except ValueError:
                pass
        elif token == "--instance-num" and i + 1 < len(cmd):
            try:
                instance_num = int(cmd[i + 1])
            except ValueError:
                pass

    return num_trucks, truck_capacity, truck_speed, instance_num


def build_run_map(summary_path: Path) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    payload = load_json(summary_path)
    run_timestamp = payload.get("meta", {}).get("timestamp", "")
    run_map: Dict[str, Dict[str, Any]] = {}

    for item in payload.get("results", []):
        scenario = str(item.get("scenario", ""))
        if not scenario:
            continue
        run_map[scenario_key_from_path(scenario)] = {
            "run_status": item.get("status", ""),
            "run_attempts": item.get("attempts", ""),
            "run_core": item.get("core", ""),
            "run_return_code": item.get("return_code", ""),
            "run_elapsed_sec": item.get("elapsed_sec", ""),
        }

    return run_timestamp, run_map


def get_result_triplet(event: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not event:
        return None, None, None
    result = event.get("result")
    if not isinstance(result, list) or len(result) < 3:
        return None, None, None
    return safe_num(result[0]), safe_num(result[1]), safe_num(result[2])


def build_rows_for_folder(folder: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    summary_path = folder / "run_summary_latest.json"
    run_timestamp = ""
    run_map: Dict[str, Dict[str, Any]] = {}
    if summary_path.exists():
        run_timestamp, run_map = build_run_map(summary_path)

    for json_path in sorted(folder.glob("datasets__*.json")):
        payload = load_json(json_path)

        meta = payload.get("meta", {})
        scenario_path = str(meta.get("scenario_path", ""))
        scenario_name = Path(scenario_path).stem if scenario_path else json_path.stem
        scenario_key = scenario_name.lower()

        best = payload.get("best_result") or {}
        final_gen = (payload.get("final_generation") or {}).get("full_result") or {}
        new_gen = (payload.get("final_generation") or {}).get("new_gen") or {}
        heuristics = payload.get("first_three_heuristics") or []
        acs = payload.get("acs_result") or {}
        acs_rp = payload.get("acs_replayed") or {}

        best_r1, best_r2, best_r3 = get_result_triplet(best)
        final_r1, final_r2, final_r3 = get_result_triplet(final_gen)

        h = []
        for i in range(3):
            if i < len(heuristics) and isinstance(heuristics[i], dict):
                h.append(heuristics[i])
            else:
                h.append({})

        h1_r1, h1_r2, h1_r3 = get_result_triplet(h[0])
        h2_r1, h2_r2, h2_r3 = get_result_triplet(h[1])
        h3_r1, h3_r2, h3_r3 = get_result_triplet(h[2])

        cmd = meta.get("command")
        if not isinstance(cmd, list):
            cmd = []
        num_trucks, truck_capacity, truck_speed, instance_num = parse_cmd_params(cmd)

        run_item = run_map.get(scenario_key, {})

        best_fitness = safe_num(best.get("fitness"))
        final_fitness = safe_num(final_gen.get("fitness"))
        gap_final_vs_best = None
        if best_fitness is not None and final_fitness is not None:
            gap_final_vs_best = final_fitness - best_fitness

        rows.append(
            {
                "group_folder": folder.name,
                "run_timestamp": run_timestamp,
                "scenario_name": scenario_name,
                "scenario_path": scenario_path,
                "data_file": json_path.name,
                "run_status": run_item.get("run_status", ""),
                "run_attempts": run_item.get("run_attempts", ""),
                "run_core": run_item.get("run_core", ""),
                "run_return_code": run_item.get("run_return_code", ""),
                "run_elapsed_sec": run_item.get("run_elapsed_sec", ""),
                "events": (payload.get("counts") or {}).get("events", ""),
                "gp_new_gen_count": (payload.get("counts") or {}).get("gp_new_gen", ""),
                "gp_full_result_count": (payload.get("counts") or {}).get("gp_full_result", ""),
                "best_gen": best.get("gen", ""),
                "best_fitness": best_fitness,
                "best_result_1": best_r1,
                "best_result_2": best_r2,
                "best_result_3": best_r3,
                "final_gen": final_gen.get("gen", new_gen.get("gen", "")),
                "final_fitness": final_fitness,
                "final_result_1": final_r1,
                "final_result_2": final_r2,
                "final_result_3": final_r3,
                "gap_final_vs_best": gap_final_vs_best,
                "final_routing_expr": new_gen.get("routing", ""),
                "final_sequencing_expr": new_gen.get("sequencing", ""),
                "heu1_name": h[0].get("name", ""),
                "heu1_fitness": safe_num(h[0].get("fitness")),
                "heu1_result_1": h1_r1,
                "heu1_result_2": h1_r2,
                "heu1_result_3": h1_r3,
                "heu2_name": h[1].get("name", ""),
                "heu2_fitness": safe_num(h[1].get("fitness")),
                "heu2_result_1": h2_r1,
                "heu2_result_2": h2_r2,
                "heu2_result_3": h2_r3,
                "heu3_name": h[2].get("name", ""),
                "heu3_fitness": safe_num(h[2].get("fitness")),
                "heu3_result_1": h3_r1,
                "heu3_result_2": h3_r2,
                "heu3_result_3": h3_r3,
                "num_trucks": num_trucks,
                "truck_capacity": truck_capacity,
                "truck_speed": truck_speed,
                "instance_num": instance_num,
                "acs_used_vehicles": acs.get("used_vehicles", ""),
                "acs_total_tour_length": safe_num(acs.get("total_tour_length")),
                "acs_scaled_value": safe_num(acs.get("scaled_value")),
                "acs_evaluations": acs.get("evaluations", ""),
                "acs_feasible_solutions": acs.get("feasible_solutions", ""),
                "acs_feasible": acs.get("feasible", ""),
                "acs_replayed_distance": safe_num(acs_rp.get("total_distance")),
                "acs_replayed_pending": acs_rp.get("pending", ""),
                "acs_replayed_delay": safe_num(acs_rp.get("total_delay")),
            }
        )

    return rows


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    columns = [
        "group_folder",
        "run_timestamp",
        "scenario_name",
        "scenario_path",
        "data_file",
        "run_status",
        "run_attempts",
        "run_core",
        "run_return_code",
        "run_elapsed_sec",
        "events",
        "gp_new_gen_count",
        "gp_full_result_count",
        "best_gen",
        "best_fitness",
        "best_result_1",
        "best_result_2",
        "best_result_3",
        "final_gen",
        "final_fitness",
        "final_result_1",
        "final_result_2",
        "final_result_3",
        "gap_final_vs_best",
        "heu1_name",
        "heu1_fitness",
        "heu1_result_1",
        "heu1_result_2",
        "heu1_result_3",
        "heu2_name",
        "heu2_fitness",
        "heu2_result_1",
        "heu2_result_2",
        "heu2_result_3",
        "heu3_name",
        "heu3_fitness",
        "heu3_result_1",
        "heu3_result_2",
        "heu3_result_3",
        "num_trucks",
        "truck_capacity",
        "truck_speed",
        "instance_num",
        "final_routing_expr",
        "final_sequencing_expr",
        "acs_used_vehicles",
        "acs_total_tour_length",
        "acs_scaled_value",
        "acs_evaluations",
        "acs_feasible_solutions",
        "acs_feasible",
        "acs_replayed_distance",
        "acs_replayed_pending",
        "acs_replayed_delay",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a combined CSV summary for selected output folders.")
    parser.add_argument(
        "--folders",
        nargs="+",
        default=OUTPUT_DIRS_DEFAULT,
        help="Output folders to aggregate (default: 100/200/400 folders)",
    )
    parser.add_argument(
        "--out",
        default="output/summary_100_200_400.csv",
        help="Destination CSV path",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    folders: List[Path] = []
    for raw in args.folders:
        p = Path(raw)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Folder not found: {p}")
        folders.append(p)

    all_rows: List[Dict[str, Any]] = []
    for folder in folders:
        all_rows.extend(build_rows_for_folder(folder))

    all_rows.sort(key=lambda x: (str(x.get("group_folder", "")), str(x.get("scenario_name", ""))))

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (repo_root / out_path).resolve()

    write_csv(all_rows, out_path)
    print(f"[aggregate] rows={len(all_rows)}")
    print(f"[aggregate] csv={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
