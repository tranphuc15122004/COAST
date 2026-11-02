#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import set_random_seed


CSV_FIELDS = ["x", "y", "demand", "open", "close", "servicetime", "time"]


def _import_model_infer(model_name: str):
    if model_name == "am":
        return importlib.import_module("am.infer")
    elif model_name == "polynet":
        return importlib.import_module("polynet.infer")
    raise ValueError(f"Unknown model: {model_name}")


def _format_cli_number(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def find_csv_files(root: Path, pattern: str) -> list[Path]:
    p = Path(root)
    if not p.exists():
        return []
    return sorted(p for p in p.glob(pattern) if p.is_file())


def load_manifest(datasets_root: Path) -> dict[str, Any]:
    manifest_path = datasets_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _safe_get(seq: Any, idx: int, default: Any = None) -> Any:
    if isinstance(seq, list) and len(seq) > idx:
        return seq[idx]
    return default


def _safe_first(seq: Any, default: Any = None) -> Any:
    return _safe_get(seq, 0, default)


def parse_infer_json(payload: dict[str, Any]) -> dict[str, Any]:
    route_diag = _safe_first(payload.get("route_diagnostics", []), {}) or {}
    raw_comp = _safe_first(payload.get("raw_cost_components", []), {}) or {}
    norm_comp = _safe_first(payload.get("normalized_cost_components", []), {}) or {}

    return {
        "normalized_cost": _safe_first(payload.get("normalized_costs", [])),
        "raw_replay_cost": _safe_first(payload.get("raw_replay_costs", [])),
        "total_skipped_customers": payload.get("total_skipped_customers"),
        "total_tw_violations": payload.get("total_tw_violations"),
        "total_appearance_violations": payload.get("total_appearance_violations"),
        "active_customers": route_diag.get("active_customers"),
        "visited_customers": route_diag.get("visited_customers"),
        "visit_steps": route_diag.get("visit_steps"),
        "missing_count": route_diag.get("missing_count"),
        "duplicate_count": route_diag.get("duplicate_count"),
        "extra_count": route_diag.get("extra_count"),
        "raw_total_cost": raw_comp.get("total_cost"),
        "raw_distance": raw_comp.get("distance"),
        "raw_late_time": raw_comp.get("late_time"),
        "raw_late_penalty": raw_comp.get("late_penalty"),
        "raw_skipped_orders": raw_comp.get("skipped_orders"),
        "raw_skipped_penalty": raw_comp.get("skipped_penalty"),
        "normalized_total_cost": norm_comp.get("total_cost"),
        "normalized_distance": norm_comp.get("distance"),
        "normalized_late_time": norm_comp.get("late_time"),
        "normalized_late_penalty": norm_comp.get("late_penalty"),
        "normalized_skipped_orders": norm_comp.get("skipped_orders"),
        "normalized_skipped_penalty": norm_comp.get("skipped_penalty"),
    }


def run_cell(
    model_name: str,
    infer_args: argparse.Namespace,
    learner: torch.nn.Module,
    model_infer: Any,
    dataset_cls: type,
    env_cls: type,
    device: torch.device,
    cell_dir: Path,
    datasets_root: Path,
    run_dir: Path,
) -> dict[str, Any]:
    csv_files = find_csv_files(cell_dir, "*.csv")
    if not csv_files:
        return {"status": "failed", "error_message": "no CSV files in cell"}

    instance_rows: list[dict[str, Any]] = []
    for idx, csv_path in enumerate(csv_files):
        result_json = run_dir / f"instance_{idx:05d}.infer.json"
        ensure_dir(result_json.parent)

        start = time.perf_counter()
        try:
            infer_args.data_csv = str(csv_path)
            data = model_infer._build_dataset(infer_args, dataset_cls)
            raw_data = model_infer._clone_dataset(data)
            if not infer_args.no_normalize:
                data.normalize()

            env_params = model_infer._build_env_params(infer_args)
            env = env_cls(data, None, None, *env_params)
            env.nodes = env.nodes.to(device)
            if env.init_cust_mask is not None:
                env.init_cust_mask = env.init_cust_mask.to(device)

            routes, costs = model_infer._run_inference(infer_args, env, learner)

            raw_cost_components = model_infer._compute_cost_components(
                raw_data, routes, infer_args.pending_cost, infer_args.late_cost
            )
            normalized_cost_components = model_infer._compute_cost_components(
                data, routes, infer_args.pending_cost, infer_args.late_cost
            )
            raw_replay_costs = raw_data.nodes.new_tensor(
                [c["total_cost"] for c in raw_cost_components]
            )
            route_diagnostics = [
                model_infer._route_diag_for_instance(data, routes, inst_idx)
                for inst_idx in range(len(routes))
            ]
            constraint_diagnostics = model_infer._check_route_constraints(raw_data, routes)

            model_infer._save_json(
                result_json, routes, costs,
                raw_replay_costs=raw_replay_costs,
                route_diagnostics=route_diagnostics,
                constraint_diagnostics=constraint_diagnostics,
                raw_cost_components=raw_cost_components,
                normalized_cost_components=normalized_cost_components,
            )

            duration_sec = time.perf_counter() - start
            payload = json.loads(result_json.read_text(encoding="utf-8"))

            row: dict[str, Any] = {
                "dataset_relpath": str(csv_path.relative_to(datasets_root)),
                "dataset_abspath": str(csv_path.resolve()),
                "status": "ok",
                "duration_sec": round(duration_sec, 6),
                "error_message": "",
            }
            row.update(parse_infer_json(payload))
        except Exception as exc:
            duration_sec = time.perf_counter() - start
            row = {
                "dataset_relpath": str(csv_path.relative_to(datasets_root)),
                "dataset_abspath": str(csv_path.resolve()),
                "status": "failed",
                "duration_sec": round(duration_sec, 6),
                "error_message": str(exc),
            }

        instance_rows.append(row)

    cell_duration = sum(float(r.get("duration_sec", 0)) for r in instance_rows)
    ok_rows = [r for r in instance_rows if r.get("status") == "ok"]

    if not ok_rows:
        return {"status": "failed", "error_message": "all instances failed", "rows": instance_rows}

    return {
        "status": "ok",
        "instance_rows": len(instance_rows),
        "row_count": len(ok_rows),
        "duration_sec": round(cell_duration, 6),
        "normalized_cost_mean": fmean(float(r["normalized_cost"]) for r in ok_rows),
        "normalized_cost_std": pstdev(float(r["normalized_cost"]) for r in ok_rows) if len(ok_rows) > 1 else 0.0,
        "total_skipped_customers_mean": fmean(float(r["total_skipped_customers"]) for r in ok_rows),
        "total_tw_violations_mean": fmean(float(r["total_tw_violations"]) for r in ok_rows),
        "total_appearance_violations_mean": fmean(float(r["total_appearance_violations"]) for r in ok_rows),
        "visited_customers_mean": fmean(float(r["visited_customers"]) for r in ok_rows),
        "error_message": "",
        "rows": instance_rows,
    }


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return fmean(values), pstdev(values)


def build_master_tables(
    run_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    master_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []

    for record in run_records:
        if record.get("status") != "ok":
            continue
        rows = record.get("rows", [])
        for row in rows:
            master_rows.append(dict(row))

        cell_rows.append({k: record.get(k) for k in [
            "algorithm", "cell_id", "dod", "tw_ratio", "seed",
            "status", "duration_sec", "instance_rows", "row_count",
            "normalized_cost_mean", "normalized_cost_std",
            "total_skipped_customers_mean",
            "total_tw_violations_mean", "total_appearance_violations_mean",
            "visited_customers_mean",
        ] if k in record})

    return master_rows, cell_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_model_for_cell(
    model_name: str,
    config_file: Path,
    model_weight: Path,
    problem_type: str,
    vehicles_count: int,
    veh_capa: int,
    veh_speed: float,
    greedy: bool,
    sample: bool,
    verify_rollouts: int,
    no_verify_routes: bool,
    first_csv: Path,
) -> tuple[Any, Any, Any, Any, Any]:
    model_infer = _import_model_infer(model_name)

    infer_argv = [
        "--config-file", str(config_file.resolve()),
        "--model-weight", str(model_weight.resolve()),
        "--problem-type", problem_type,
        "--vehicles-count", str(vehicles_count),
        "--veh-capa", str(veh_capa),
        "--veh-speed", _format_cli_number(veh_speed),
        "--verify-rollouts", str(verify_rollouts),
        "--data-csv", str(first_csv),
    ]
    if sample:
        infer_argv.append("--sample")
    else:
        infer_argv.append("--greedy")
    if no_verify_routes:
        infer_argv.append("--no-verify-routes")

    infer_args = model_infer.parse_infer_args(infer_argv)

    device = torch.device("cuda" if torch.cuda.is_available() and not infer_args.no_cuda else "cpu")
    set_random_seed(infer_args.rng_seed, deterministic=True)

    dataset_cls = model_infer._dataset_cls(infer_args.problem_type)
    env_cls = model_infer._environment_cls(infer_args.problem_type)
    if dataset_cls is None or env_cls is None:
        raise ValueError(f"Unsupported problem type '{infer_args.problem_type}'")

    learner = model_infer._init_model(infer_args, env_cls, device)
    learner.eval()
    model_infer._load_model_weights_or_raise(infer_args.model_weight, learner)
    learner.eval()

    return model_infer, infer_args, learner, dataset_cls, env_cls, device


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run DVRPTW dynamic benchmark for AM/Polynet models."
    )
    parser.add_argument("--datasets-root", type=Path,
                        default=Path("../mardam/datasets/dvrptw_dynamic_grid"))
    parser.add_argument("--run-root", type=Path, default=Path("output/dynamic_benchmark"))
    parser.add_argument("--problem-type", type=str, default="dvrptw")
    parser.add_argument("--vehicles-count", type=int, default=3)
    parser.add_argument("--veh-capa", type=int, default=200)
    parser.add_argument("--veh-speed", type=float, default=1.0)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--greedy", action="store_true", default=True)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--verify-rollouts", type=int, default=1)
    parser.add_argument("--no-verify-routes", action="store_true")

    parser.add_argument("--am-config-file", type=Path, default=None)
    parser.add_argument("--am-model-weight", type=Path, default=None)
    parser.add_argument("--polynet-config-file", type=Path, default=None)
    parser.add_argument("--polynet-model-weight", type=Path, default=None)

    args = parser.parse_args()

    datasets_root = args.datasets_root.resolve()
    if not datasets_root.exists():
        raise FileNotFoundError(f"Dataset not found: {datasets_root}")

    manifest = load_manifest(datasets_root)
    cells = manifest.get("cells", [])
    if args.max_cells is not None:
        cells = cells[: args.max_cells]

    model_specs: list[tuple[str, str, Path, Path]] = []
    if args.am_config_file and args.am_model_weight:
        model_specs.append(("am", "AM", args.am_config_file, args.am_model_weight))
    if args.polynet_config_file and args.polynet_model_weight:
        model_specs.append(("polynet", "PolyNet", args.polynet_config_file, args.polynet_model_weight))

    if not model_specs:
        print("No models specified. Provide --am-* and/or --polynet-* flags.")
        return 1

    run_root = args.run_root.resolve()
    ensure_dir(run_root)

    run_records: list[dict[str, Any]] = []

    for model_key, model_label, config_file, model_weight in model_specs:
        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_file}")
        if not model_weight.exists():
            raise FileNotFoundError(f"Weight not found: {model_weight}")

        print(f"\n{'=' * 60}")
        print(f"  Model: {model_label} ({model_key})")
        print(f"{'=' * 60}")

        last_model_infer = None
        last_learner = None
        last_infer_args = None
        last_dataset_cls = None
        last_env_cls = None
        last_device = None

        for cell_idx, cell_meta in enumerate(cells):
            cell_dir = datasets_root / str(cell_meta["cell_relpath"])
            run_dir = run_root / model_key / str(cell_meta["cell_relpath"])

            csvs = find_csv_files(cell_dir, "*.csv")
            first_csv = csvs[0] if csvs else None
            if first_csv is None:
                print(f"  [{cell_idx + 1}/{len(cells)}] {cell_meta['cell_id']} -> SKIP (no CSVs)")
                continue

            if last_learner is None:
                (model_infer, infer_args, learner, ds_cls,
                 env_cls, device) = load_model_for_cell(
                    model_key, config_file, model_weight,
                    args.problem_type, args.vehicles_count,
                    args.veh_capa, args.veh_speed,
                    args.greedy, args.sample,
                    args.verify_rollouts, args.no_verify_routes,
                    first_csv,
                )
                last_model_infer = model_infer
                last_infer_args = infer_args
                last_dataset_cls = ds_cls
                last_env_cls = env_cls
                last_device = device
                last_learner = learner

            print(f"  [{cell_idx + 1}/{len(cells)}] {cell_meta['cell_id']} -> ", end="", flush=True)
            record = run_cell(
                model_key, last_infer_args, last_learner,
                last_model_infer, last_dataset_cls, last_env_cls,
                last_device, cell_dir, datasets_root, run_dir,
            )
            record.update({
                "algorithm": model_label,
                "cell_id": cell_meta["cell_id"],
                "cell_relpath": cell_meta["cell_relpath"],
                "dod": cell_meta["dod"],
                "tw_ratio": cell_meta["tw_ratio"],
                "seed": cell_meta["seed"],
                "cell_dir": str(cell_dir),
                "run_dir": str(run_dir),
            })
            run_records.append(record)
            status = record.get("status", "?")
            print(f"{status} ({record.get('duration_sec', 0):.1f}s)")

        if last_learner is not None:
            del last_learner
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    master_rows, cell_rows = build_master_tables(run_records)
    master_csv = run_root / "master_summary.csv"
    cell_csv = run_root / "cell_summary.csv"
    master_json = run_root / "master_summary.json"
    overview_json = run_root / "overview.json"

    write_csv(master_csv, master_rows)
    write_csv(cell_csv, cell_rows)
    write_json(master_json, {"manifest": manifest, "rows": master_rows})

    overview: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "datasets_root": str(datasets_root),
        "run_root": str(run_root),
        "run_records": run_records,
        "master_row_count": len(master_rows),
        "cell_row_count": len(cell_rows),
    }
    write_json(overview_json, overview)

    print(f"\n{'=' * 60}")
    print(f"Benchmark root : {datasets_root}")
    print(f"Run root       : {run_root}")
    print(f"Master CSV     : {master_csv}")
    print(f"Cell CSV       : {cell_csv}")
    print(f"Master rows    : {len(master_rows)}")
    print(f"Cell rows      : {len(cell_rows)}")

    return 0 if all(r.get("status") == "ok" for r in run_records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
