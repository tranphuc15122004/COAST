#!/usr/bin/env python3
"""Launch the four G1 neural methods on matched shared pools and seeds.

Configuration lives in ``script/train_multiseed_g1.json``. For example:

    python script/train_multiseed.py --dry-run
    python script/train_multiseed.py

Set ``gpu_ids``, ``parallel_per_gpu``, ``job_gpu_map`` and ``output_root`` in
that JSON file, or override them on the command line. ``job_gpu_map`` accepts
method keys (``"coast"``) and exact run keys (``"coast:42"``); exact run keys
take precedence. Each child process receives one physical GPU through
``CUDA_VISIBLE_DEVICES`` and sees that device as ``cuda:0``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "script" / "train_multiseed_g1.json"
BASELINE_ROOT = REPO_ROOT / "third_party" / "baselines" / "RL4DVRPTW"
B200_BASELINE_REQUIREMENTS = REPO_ROOT / "requirements.b200-baselines.txt"

DEFAULT_METHODS = ("coast", "mardam", "am", "polynet")
DEFAULT_SEEDS = (42, 123, 456, 789, 1024)

METHODS: dict[str, dict[str, Any]] = {
    "coast": {
        "cwd": REPO_ROOT,
        "entry": ("MODEL/train.py",),
        "layers": 2,
        "heads": 4,
        "ff_size": 256,
        "iterations": 1000,
        "batch_size": 512,
        "extra": ("--ablation-profile", "vectra", "--memory-size", "128"),
    },
    "mardam": {
        # This is the entrypoint whose model state dict matches the selected
        # paper checkpoint (AttentionLearner, 3/8/512, 728,582 parameters).
        "cwd": REPO_ROOT,
        "entry": ("script/train_mardam.py",),
        "layers": 3,
        "heads": 8,
        "ff_size": 512,
        "iterations": 1000,
        "batch_size": 512,
        "extra": (),
    },
    "am": {
        "cwd": BASELINE_ROOT,
        "entry": ("-m", "am.train"),
        "layers": 5,
        "heads": 8,
        "ff_size": 256,
        "iterations": 2000,
        "batch_size": 256,
        "extra": ("--resource-safe",),
    },
    "polynet": {
        "cwd": BASELINE_ROOT,
        "entry": ("-m", "polynet.train"),
        "layers": 5,
        "heads": 8,
        "ff_size": 256,
        "iterations": 1000,
        "batch_size": 512,
        "extra": ("--resource-safe",),
    },
}

# These files define generated instances, dynamics, objective simulation and
# the policy-gradient loss. Refuse to train if the two package copies diverge.
PARITY_FILES = (
    "problems/_data.py",
    "problems/_data_tw.py",
    "problems/_data_dtw.py",
    "problems/_env.py",
    "problems/_env_tw.py",
    "problems/_env_dtw.py",
    "layers/_loss.py",
)

GENERATION_ARGS: tuple[tuple[str, tuple[Any, ...]], ...] = (
    ("--problem-type", ("dvrptw",)),
    ("--customers-count", (50,)),
    ("--vehicles-count", (3,)),
    ("--veh-capa", (200,)),
    ("--veh-speed", (1,)),
    ("--horizon", (480,)),
    ("--loc-range", (0, 101)),
    ("--dem-range", (5, 41)),
    ("--dur-range", (10, 31)),
    ("--tw-ratio", (0.25, 0.5, 0.75, 1.0)),
    ("--tw-range", (30, 91)),
    ("--deg-of-dyna", (0.1, 0.25, 0.5, 0.75)),
    ("--appear-early-ratio", (0.0, 0.5, 0.75, 1.0)),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def parse_gpu_ids(value: str) -> list[str]:
    ids = [part.strip() for part in value.split(",") if part.strip()]
    if not ids:
        raise argparse.ArgumentTypeError("provide at least one GPU id, e.g. 0,1")
    return ids


def parse_capacity(value: str) -> int | dict[str, int]:
    """Parse a uniform limit (``2``) or per-device limits (``0=2,1=4``)."""
    if "=" not in value:
        try:
            capacity = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("expected an integer or GPU=COUNT pairs") from exc
        if capacity < 1:
            raise argparse.ArgumentTypeError("parallel run count must be positive")
        return capacity

    result: dict[str, int] = {}
    try:
        for item in value.split(","):
            gpu, count = item.split("=", 1)
            result[gpu.strip()] = int(count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected GPU=COUNT pairs such as 0=2,1=4") from exc
    if not result or any(count < 1 for count in result.values()):
        raise argparse.ArgumentTypeError("parallel run counts must be positive")
    return result


def parse_job_gpu(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected METHOD[:SEED]=GPU, e.g. coast:42=1")
    key, gpu = value.split("=", 1)
    key, gpu = key.strip().lower(), gpu.strip()
    if not key or not gpu:
        raise argparse.ArgumentTypeError("expected METHOD[:SEED]=GPU")
    return key, gpu


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)

    if args.methods:
        config["methods"] = [value.lower() for value in args.methods]
    if args.seeds:
        config["seeds"] = args.seeds
    if args.output_root:
        config["output_root"] = args.output_root
    if args.python:
        config["python"] = args.python
    if args.gpu_ids:
        config["gpu_ids"] = args.gpu_ids
    if args.parallel_per_gpu is not None:
        config["parallel_per_gpu"] = args.parallel_per_gpu
    if args.job_gpu:
        mapping = dict(config.get("job_gpu_map", {}))
        mapping.update(args.job_gpu)
        config["job_gpu_map"] = mapping
    if args.epoch_count is not None:
        config["epoch_count"] = args.epoch_count
    if args.expected_gpu_name is not None:
        config["expected_gpu_name"] = args.expected_gpu_name
    if args.allow_any_gpu:
        config["expected_gpu_name"] = None
    config["_config_path"] = str(config_path)
    config["_dry_run"] = bool(args.dry_run)
    config["_preflight_only"] = bool(args.preflight_only)
    config["_generate_shared_pools_only"] = bool(args.generate_shared_pools_only)
    return config


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    methods = [str(item).lower() for item in config.get("methods", DEFAULT_METHODS)]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError("unknown methods: {} (choose from {})".format(", ".join(unknown), ", ".join(METHODS)))
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be a nonempty list without duplicates")

    seeds = [int(item) for item in config.get("seeds", DEFAULT_SEEDS)]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list without duplicates")

    gpu_ids = [str(item) for item in config.get("gpu_ids", ["0"])]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("gpu_ids must be a nonempty list without duplicates")

    capacities_raw = config.get("parallel_per_gpu", 1)
    if isinstance(capacities_raw, int):
        capacities = {gpu: capacities_raw for gpu in gpu_ids}
    elif isinstance(capacities_raw, dict):
        capacities = {gpu: int(capacities_raw.get(gpu, 1)) for gpu in gpu_ids}
    else:
        raise ValueError("parallel_per_gpu must be an integer or a GPU-to-count object")
    if any(value < 1 for value in capacities.values()):
        raise ValueError("each GPU concurrency limit must be positive")

    mapping = {str(key).lower(): str(value) for key, value in config.get("job_gpu_map", {}).items()}
    for key, gpu in mapping.items():
        if gpu not in gpu_ids:
            raise ValueError("job_gpu_map entry {}={} refers to a GPU outside gpu_ids".format(key, gpu))
        if ":" in key:
            method, seed_text = key.split(":", 1)
            if method not in methods:
                raise ValueError("job_gpu_map entry {} refers to a method not selected".format(key))
            try:
                if int(seed_text) not in seeds:
                    raise ValueError("job_gpu_map entry {} refers to a seed not selected".format(key))
            except ValueError as exc:
                raise ValueError("invalid job_gpu_map key {}; expected method:integer-seed".format(key)) from exc
        elif key not in methods:
            raise ValueError("job_gpu_map entry {} must be a selected method or method:seed".format(key))

    output_root = Path(str(config.get("output_root", "output/multiseed_g1"))).expanduser()
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()

    expected_gpu_name = config.get("expected_gpu_name", "B200")
    if expected_gpu_name is not None:
        expected_gpu_name = str(expected_gpu_name).strip()
        if not expected_gpu_name:
            raise ValueError("expected_gpu_name must be a nonempty string or null")

    config["methods"] = methods
    config["seeds"] = seeds
    config["gpu_ids"] = gpu_ids
    config["parallel_per_gpu"] = capacities
    config["job_gpu_map"] = mapping
    config["output_root"] = output_root
    config["python"] = str(config.get("python") or sys.executable)
    config["expected_gpu_name"] = expected_gpu_name
    epoch_count = int(config.get("epoch_count", 500))
    if epoch_count < 1:
        raise ValueError("epoch_count must be positive")
    config["epoch_count"] = epoch_count
    return config


def compare_shared_sources() -> dict[str, str]:
    hashes: dict[str, str] = {}
    mismatches = []
    for relative in PARITY_FILES:
        coast_path = REPO_ROOT / relative
        baseline_path = BASELINE_ROOT / relative
        if not coast_path.is_file() or not baseline_path.is_file():
            mismatches.append("missing {} or {}".format(coast_path, baseline_path))
            continue
        coast_hash = sha256_file(coast_path)
        baseline_hash = sha256_file(baseline_path)
        hashes[relative] = coast_hash
        if coast_hash != baseline_hash:
            mismatches.append("{} differs (COAST={}, RL4DVRPTW={})".format(relative, coast_hash, baseline_hash))
    if mismatches:
        raise RuntimeError("shared simulator/generator/loss preflight failed:\n  " + "\n  ".join(mismatches))
    return hashes


def assign_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    mapping = config["job_gpu_map"]
    counts = {gpu: 0 for gpu in config["gpu_ids"]}
    jobs = []
    for seed in config["seeds"]:
        for method in config["methods"]:
            exact_key = "{}:{}".format(method, seed)
            gpu = mapping.get(exact_key, mapping.get(method))
            if gpu is None:
                # Assign jobs toward the least-loaded configured GPU, taking
                # each device's concurrency limit into account.
                gpu = min(
                    config["gpu_ids"],
                    key=lambda item: (counts[item] / config["parallel_per_gpu"][item], counts[item]),
                )
            counts[gpu] += 1
            output_dir = config["output_root"] / method / "seed{}".format(seed)
            pool_dir = config["output_root"] / "_shared_pools" / "seed{}".format(seed)
            jobs.append({
                "method": method,
                "seed": seed,
                "gpu": gpu,
                "output_dir": output_dir,
                "train_data_file": pool_dir / "train.pyth",
                "test_data_file": pool_dir / "validation.pyth",
                "status": "queued",
                "exit_code": None,
                "started_at": None,
                "finished_at": None,
            })
    return jobs


def build_command(config: dict[str, Any], job: dict[str, Any]) -> list[str]:
    method = METHODS[job["method"]]
    command = [config["python"], *method["entry"]]
    command.extend(("--rng-seed", str(job["seed"])))
    for flag, values in GENERATION_ARGS:
        command.append(flag)
        command.extend(str(value) for value in values)

    values: tuple[tuple[str, Any], ...] = (
        ("--pending-cost", 2),
        ("--late-cost", 1),
        ("--epoch-count", config.get("epoch_count", 500)),
        ("--iter-count", method["iterations"]),
        ("--batch-size", method["batch_size"]),
        ("--test-batch-size", 10240),
        ("--model-size", 128),
        ("--layer-count", method["layers"]),
        ("--head-count", method["heads"]),
        ("--ff-size", method["ff_size"]),
        ("--tanh-xplor", 10),
        ("--dropout", 0.1),
        ("--baseline-type", "critic"),
        ("--learning-rate", 0.0001),
        ("--critic-rate", 0.001),
        ("--weight-decay", 0.0),
        ("--max-grad-norm", 2.0),
        ("--entropy-coef", 0.01),
        ("--num-workers", 4),
        ("--output-dir", str(job["output_dir"])),
        ("--train-data-file", str(job["train_data_file"])),
        ("--test-data-file", str(job["test_data_file"])),
    )
    for flag, value in values:
        command.extend((flag, str(value)))
    command.extend(("--adv-norm", "--amp", "--pin-memory", "--strict-run-budget"))
    command.extend(METHODS[job["method"]]["extra"])
    return command


def make_generation_parameters() -> tuple[list[Any], dict[str, Any]]:
    # Keep these values in lockstep with GENERATION_ARGS and the archived G1
    # args.json files. The order is the Dataset.generate signature order.
    gen_params = [50, 3, 200, 1, None, (0, 101), (5, 41), 480, (10, 31),
                  (0.25, 0.5, 0.75, 1.0), (30, 91),
                  (0.1, 0.25, 0.5, 0.75), (0.0, 0.5, 0.75, 1.0)]
    metadata = {
        "problem_type": "dvrptw",
        "customers_count": 50,
        "vehicles_count": 3,
        "veh_capa": 200,
        "veh_speed": 1,
        "min_cust_count": None,
        "loc_range": [0, 101],
        "dem_range": [5, 41],
        "horizon": 480,
        "dur_range": [10, 31],
        "tw_ratio": [0.25, 0.5, 0.75, 1.0],
        "tw_range": [30, 91],
        "deg_of_dyna": [0.1, 0.25, 0.5, 0.75],
        "appear_early_ratio": [0.0, 0.5, 0.75, 1.0],
    }
    return gen_params, metadata


def generate_shared_pools(config: dict[str, Any], source_hashes: dict[str, str]) -> dict[str, Any]:
    import torch

    sys.path.insert(0, str(REPO_ROOT))
    from problems import DVRPTW_Dataset

    output_root: Path = config["output_root"]
    final_pool_root = output_root / "_shared_pools"
    if final_pool_root.exists():
        raise FileExistsError("shared-pool output already exists: {}".format(final_pool_root))

    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".shared-pools-", dir=str(output_root)))
    pool_records: dict[str, Any] = {}
    gen_params, generation_metadata = make_generation_parameters()
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for seed in config["seeds"]:
            random.seed(seed)
            torch.manual_seed(seed)

            seed_dir = staging / "seed{}".format(seed)
            seed_dir.mkdir(parents=True)

            # Generate train first, then validation, matching the trainers'
            # historical call order. Both pools remain raw on disk; each
            # trainer normalizes the loaded copy in its own process.
            train_data = DVRPTW_Dataset.generate(512000, *gen_params)
            train_path = seed_dir / "train.pyth"
            train_data.save(str(train_path))
            del train_data

            validation_data = DVRPTW_Dataset.generate(10240, *gen_params)
            validation_path = seed_dir / "validation.pyth"
            validation_data.save(str(validation_path))
            del validation_data

            pool_records[str(seed)] = {
                "train": {
                    "path": str((final_pool_root / seed_dir.name / train_path.name).relative_to(output_root)),
                    "samples": 512000,
                    "sha256": sha256_file(train_path),
                },
                "validation": {
                    "path": str((final_pool_root / seed_dir.name / validation_path.name).relative_to(output_root)),
                    "samples": 10240,
                    "sha256": sha256_file(validation_path),
                },
                "generation_order": ["train", "validation"],
            }
            print("[pool] seed={} train={} validation={}".format(
                seed,
                pool_records[str(seed)]["train"]["sha256"],
                pool_records[str(seed)]["validation"]["sha256"],
                flush=True,
            ))

        pool_manifest = {
            "created_at": utc_now(),
            "generator_source_sha256": source_hashes["problems/_data_dtw.py"],
            "generation": generation_metadata,
            "pools": pool_records,
        }
        write_json_atomic(staging / "manifest.json", pool_manifest)
        os.replace(staging, final_pool_root)
        return pool_manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        torch.set_num_threads(original_threads)


def validate_gpu_device_info(
    gpu: str,
    device_info: dict[str, Any],
    *,
    expected_gpu_name: str | None,
) -> None:
    if not device_info.get("available") or not device_info.get("cuda_context_ok", False):
        raise RuntimeError(
            "GPU id {} is not available to the selected Python runtime; "
            "the launcher refuses silent CPU fallback".format(gpu)
        )

    name = str(device_info.get("name") or "")
    if expected_gpu_name and expected_gpu_name.lower() not in name.lower():
        raise RuntimeError(
            "GPU id {} is {!r}; expected a device name containing {!r}".format(
                gpu, name, expected_gpu_name
            )
        )

    if expected_gpu_name and "b200" in expected_gpu_name.lower():
        capability = device_info.get("compute_capability") or []
        if len(capability) < 2 or int(capability[0]) < 10:
            raise RuntimeError(
                "GPU id {} reports compute capability {}; expected Blackwell "
                "compute capability >= 10.0 for B200".format(gpu, capability)
            )


def build_gpu_probe_script() -> str:
    """Return a standalone probe that tests the selected Python/CUDA runtime."""
    return """\
import json
import platform
import torch

available = torch.cuda.is_available() and torch.cuda.device_count() == 1
info = {
    "available": bool(available),
    "name": None,
    "compute_capability": None,
    "total_memory_gb": None,
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_context_ok": False,
}

if available:
    try:
        torch.cuda.init()
        props = torch.cuda.get_device_properties(0)
        # Exercise a CUDA kernel and synchronize, catching runtimes that can
        # enumerate a B200 but cannot execute PyTorch operations on it.
        result = torch.ones((32, 32), device="cuda") @ torch.ones((32, 32), device="cuda")
        torch.cuda.synchronize()
        if not torch.all(result == 32):
            raise RuntimeError("CUDA matrix-multiply sanity check returned an incorrect result")
        info.update({
            "name": props.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
            "cuda_context_ok": True,
        })
    except Exception as exc:
        info["cuda_context_error"] = "{}: {}".format(type(exc).__name__, exc)

print(json.dumps(info))
"""


def preflight_gpus(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    probe = build_gpu_probe_script()
    info: dict[str, dict[str, Any]] = {}
    for gpu in config["gpu_ids"]:
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        result = subprocess.run(
            [config["python"], "-c", probe],
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError("GPU {} preflight failed: {}".format(gpu, result.stderr.strip()))
        try:
            device_info = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError("GPU {} preflight returned invalid output".format(gpu)) from exc
        try:
            validate_gpu_device_info(
                gpu,
                device_info,
                expected_gpu_name=config["expected_gpu_name"],
            )
        except RuntimeError as exc:
            runtime = "Python {} / PyTorch {} / CUDA {}".format(
                device_info.get("python"),
                device_info.get("torch"),
                device_info.get("torch_cuda"),
            )
            detail = device_info.get("cuda_context_error")
            suffix = "; CUDA probe: {}".format(detail) if detail else ""
            raise RuntimeError("{} ({}{})".format(exc, runtime, suffix)) from exc
        info[gpu] = device_info
        print(
            "[gpu] id={} name={} cc={}.{} memory={}GiB torch={} cuda={} python={}".format(
                gpu,
                device_info["name"],
                *device_info["compute_capability"],
                device_info["total_memory_gb"],
                device_info["torch"],
                device_info["torch_cuda"],
                device_info["python"],
            ),
            flush=True,
        )
    return info


def build_child_environment(method_name: str, gpu: str, seed: int | None = None) -> dict[str, str]:
    method = METHODS[method_name]
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if seed is not None:
        env["PYTHONHASHSEED"] = str(seed)
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    inherited_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(method["cwd"]) + (
        os.pathsep + inherited_pythonpath if inherited_pythonpath else ""
    )
    return env


def preflight_trainer_clis(
    config: dict[str, Any], jobs: list[dict[str, Any]]
) -> None:
    """Import every selected trainer and confirm it accepts the planned flags."""
    for method_name in config["methods"]:
        job = next(item for item in jobs if item["method"] == method_name)
        command = [config["python"], *METHODS[method_name]["entry"], "--help"]
        result = subprocess.run(
            command,
            cwd=str(METHODS[method_name]["cwd"]),
            env=build_child_environment(method_name, job["gpu"]),
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            missing_baseline_runtime = any(
                "No module named '{}'".format(module) in detail
                for module in ("torchrl", "lightning")
            )
            install_hint = ""
            if method_name in ("am", "polynet") and missing_baseline_runtime:
                install_hint = (
                    "\nInstall the B200 baseline dependencies into the selected interpreter "
                    "without applying the baseline's older PyTorch pins:\n  {} -m pip install -r {}"
                ).format(config["python"], B200_BASELINE_REQUIREMENTS)
            raise RuntimeError(
                "{} trainer import/CLI preflight failed: {}{}".format(
                    method_name, detail, install_hint
                )
            )
        help_text = result.stdout + result.stderr
        required_options = {
            value
            for value in build_command(config, job)
            if value.startswith("--")
        }
        missing_options = sorted(
            option for option in required_options if option not in help_text
        )
        if missing_options:
            raise RuntimeError(
                "{} trainer does not advertise planned option(s): {}".format(
                    method_name, ", ".join(missing_options)
                )
            )
        print("[trainer] method={} imports/CLI OK".format(method_name), flush=True)


def check_existing_pools_valid(output_root: Path, seeds: list[int]) -> bool:
    manifest_path = output_root / "_shared_pools" / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        pools = data.get("pools", {})
        for seed in seeds:
            record = pools.get(str(seed))
            if not record:
                return False
            for split in ("train", "validation"):
                if split not in record:
                    return False
                path = output_root / record[split]["path"]
                if not path.is_file():
                    return False
        return True
    except Exception:
        return False


def check_no_overwrite(config: dict[str, Any], jobs: list[dict[str, Any]]) -> None:
    output_root: Path = config["output_root"]
    conflicts = []
    if (output_root / "manifest.json").exists():
        try:
            m = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
            existing_runs = {(r.get("method"), r.get("seed")) for r in m.get("runs", [])}
            planned_runs = {(j["method"], j["seed"]) for j in jobs}
            overlap = existing_runs & planned_runs
            if overlap:
                conflicts.append(str(output_root / "manifest.json") + " (overlapping runs: {})".format(overlap))
        except Exception:
            conflicts.append(str(output_root / "manifest.json"))
    if (output_root / "_shared_pools").exists() and not check_existing_pools_valid(output_root, config["seeds"]):
        conflicts.append(str(output_root / "_shared_pools") + " (invalid or missing requested seeds)")
    for job in jobs:
        if job["output_dir"].exists():
            conflicts.append(str(job["output_dir"]))
    if conflicts:
        raise FileExistsError(
            "refusing to overwrite existing run artifacts:\n  " + "\n  ".join(conflicts)
            + "\nChoose a new output_root to start another matrix."
        )


def print_plan(config: dict[str, Any], jobs: list[dict[str, Any]], source_hashes: dict[str, str]) -> None:
    print("Methods: {}".format(", ".join(config["methods"])))
    print("Seeds: {}".format(", ".join(str(seed) for seed in config["seeds"])))
    print("Python: {}".format(config["python"]))
    print("Expected GPU name: {}".format(config["expected_gpu_name"] or "any CUDA GPU"))
    print("Planned jobs: {}".format(len(jobs)))
    print("Output root: {}".format(config["output_root"]))
    print("Per-GPU concurrency: {}".format(json.dumps(config["parallel_per_gpu"], sort_keys=True)))
    print("Shared simulator/generator/loss SHA-256:")
    for relative, digest in source_hashes.items():
        print("  {}  {}".format(relative, digest))
    for job in jobs:
        print("\n[{} seed={} GPU={}] {}".format(job["method"], job["seed"], job["gpu"], job["output_dir"]))
        print("  cwd: {}".format(METHODS[job["method"]]["cwd"]))
        print("  command: {}".format(subprocess.list2cmdline(build_command(config, job))))


def run_jobs(config: dict[str, Any], jobs: list[dict[str, Any]], source_hashes: dict[str, str],
             pool_manifest: dict[str, Any], gpu_info: dict[str, dict[str, Any]]) -> int:
    output_root: Path = config["output_root"]
    manifest_path = output_root / "manifest.json"
    manifest = {
        "created_at": utc_now(),
        "config_path": config["_config_path"],
        "output_root": str(output_root),
        "methods": config["methods"],
        "seeds": config["seeds"],
        "gpu_ids": config["gpu_ids"],
        "parallel_per_gpu": config["parallel_per_gpu"],
        "python": config["python"],
        "expected_gpu_name": config["expected_gpu_name"],
        "gpu_info": gpu_info,
        "shared_source_sha256": source_hashes,
        "shared_pools": pool_manifest,
        "run_count": len(jobs),
        "runs": [],
    }
    if manifest_path.exists():
        try:
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            planned_keys = {(j["method"], j["seed"]) for j in jobs}
            old_runs = [r for r in old_manifest.get("runs", []) if (r.get("method"), r.get("seed")) not in planned_keys]
            manifest["runs"] = old_runs
            manifest["run_count"] = len(old_runs) + len(jobs)
        except Exception:
            pass
    base_idx = len(manifest["runs"])
    for job in jobs:
        entry = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in job.items()
        }
        entry["command"] = build_command(config, job)
        entry["working_directory"] = str(METHODS[job["method"]]["cwd"])
        entry["log_path"] = str(job["output_dir"] / "launcher.log")
        entry["train_pool_sha256"] = pool_manifest["pools"][str(job["seed"])]["train"]["sha256"]
        entry["validation_pool_sha256"] = pool_manifest["pools"][str(job["seed"])]["validation"]["sha256"]
        entry["gpu_name"] = gpu_info[job["gpu"]]["name"]
        manifest["runs"].append(entry)
    write_json_atomic(manifest_path, manifest)

    pending = list(range(len(jobs)))
    active: dict[int, dict[str, Any]] = {}
    active_by_gpu = {gpu: 0 for gpu in config["gpu_ids"]}
    failed = False

    def update_job(index: int, updates: dict[str, Any]) -> None:
        manifest["runs"][base_idx + index].update(updates)
        write_json_atomic(manifest_path, manifest)

    try:
        while pending or active:
            launched = False
            if not failed:
                for index in list(pending):
                    job = jobs[index]
                    gpu = job["gpu"]
                    if active_by_gpu[gpu] >= config["parallel_per_gpu"][gpu]:
                        continue

                    job["output_dir"].mkdir(parents=True, exist_ok=False)
                    command = build_command(config, job)
                    env = build_child_environment(job["method"], gpu, job["seed"])
                    log_stream = (job["output_dir"] / "launcher.log").open("w", encoding="utf-8")
                    write_json_atomic(job["output_dir"] / "job.json", {
                        "method": job["method"],
                        "seed": job["seed"],
                        "gpu_id": gpu,
                        "gpu_name": gpu_info[gpu]["name"],
                        "command": command,
                        "working_directory": str(METHODS[job["method"]]["cwd"]),
                        "train_pool_sha256": pool_manifest["pools"][str(job["seed"])]["train"]["sha256"],
                        "validation_pool_sha256": pool_manifest["pools"][str(job["seed"])]["validation"]["sha256"],
                        "shared_source_sha256": source_hashes,
                        "started_at": utc_now(),
                    })
                    try:
                        process = subprocess.Popen(
                            command,
                            cwd=str(METHODS[job["method"]]["cwd"]),
                            env=env,
                            stdout=log_stream,
                            stderr=subprocess.STDOUT,
                        )
                    except Exception:
                        log_stream.close()
                        update_job(index, {"status": "failed_to_start", "finished_at": utc_now()})
                        failed = True
                        pending.remove(index)
                        break

                    active[index] = {"process": process, "log_stream": log_stream}
                    active_by_gpu[gpu] += 1
                    pending.remove(index)
                    job["status"] = "running"
                    update_job(index, {"status": "running", "started_at": utc_now(), "pid": process.pid})
                    print("[start] {} seed={} GPU={} pid={} -> {}".format(
                        job["method"], job["seed"], gpu, process.pid, job["output_dir"]), flush=True)
                    launched = True

            completed = []
            for index, running in active.items():
                return_code = running["process"].poll()
                if return_code is not None:
                    completed.append((index, return_code))
            for index, return_code in completed:
                job = jobs[index]
                active[index]["log_stream"].close()
                del active[index]
                active_by_gpu[job["gpu"]] -= 1
                status = "completed" if return_code == 0 else "failed"
                job["status"] = status
                job["exit_code"] = return_code
                job["finished_at"] = utc_now()
                update_job(index, {"status": status, "exit_code": return_code, "finished_at": job["finished_at"]})
                print("[{}] {} seed={} GPU={} exit={}".format(status, job["method"], job["seed"], job["gpu"], return_code), flush=True)
                if return_code != 0:
                    failed = True

            if failed and pending:
                for index in pending:
                    jobs[index]["status"] = "not_started"
                    update_job(index, {"status": "not_started", "reason": "a previous training job failed"})
                pending.clear()

            if active and not launched and not completed:
                time.sleep(0.25)
    except KeyboardInterrupt:
        print("[interrupt] sending SIGINT to active training processes", flush=True)
        for running in active.values():
            try:
                running["process"].send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
        for index, running in list(active.items()):
            return_code = running["process"].wait()
            running["log_stream"].close()
            job = jobs[index]
            update_job(index, {
                "status": "interrupted",
                "exit_code": return_code,
                "finished_at": utc_now(),
            })
        for index in pending:
            update_job(index, {"status": "not_started", "reason": "launcher interrupted"})
        failed = True

    manifest["finished_at"] = utc_now()
    manifest["status"] = "failed" if failed else "completed"
    write_json_atomic(manifest_path, manifest)
    print("[{}] manifest: {}".format(manifest["status"], manifest_path), flush=True)
    return 1 if failed else 0


def _generate_shared_pools_with_selected_python(config: dict[str, Any]) -> dict[str, Any]:
    """Generate datasets with the same Python/Torch runtime used by trainers."""
    manifest_path = config["output_root"] / "_shared_pools" / "manifest.json"
    if check_existing_pools_valid(config["output_root"], config["seeds"]):
        print("[pool] reusing verified existing shared pools in {}".format(manifest_path.parent), flush=True)
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    internal_config = {
        "methods": config["methods"],
        "seeds": config["seeds"],
        "python": config["python"],
        "gpu_ids": config["gpu_ids"],
        "parallel_per_gpu": config["parallel_per_gpu"],
        "job_gpu_map": config["job_gpu_map"],
        "output_root": str(config["output_root"]),
        "expected_gpu_name": config["expected_gpu_name"],
    }
    with tempfile.TemporaryDirectory(prefix="multiseed-pool-config-") as temp_dir:
        config_path = Path(temp_dir) / "config.json"
        config_path.write_text(json.dumps(internal_config, indent=2) + "\n", encoding="utf-8")
        command = [
            config["python"],
            str(Path(__file__).resolve()),
            "--config",
            str(config_path),
            "--generate-shared-pools-only",
        ]
        print("[pool] generator runtime: {}".format(config["python"]), flush=True)
        result = subprocess.run(command, cwd=str(REPO_ROOT), text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "shared-pool generation failed with exit code {}".format(result.returncode)
            )

    manifest_path = config["output_root"] / "_shared_pools" / "manifest.json"
    try:
        pool_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("shared-pool generator did not write a valid manifest") from exc

    expected_seeds = {str(seed) for seed in config["seeds"]}
    if set(pool_manifest.get("pools", {})) != expected_seeds:
        raise RuntimeError("shared-pool manifest does not contain exactly the requested seeds")
    for seed in config["seeds"]:
        record = pool_manifest["pools"][str(seed)]
        for split in ("train", "validation"):
            item = record[split]
            path = config["output_root"] / item["path"]
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raise RuntimeError(
                    "shared {} pool for seed {} is missing or has a hash mismatch".format(
                        split, seed
                    )
                )
    return pool_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="JSON launcher config")
    parser.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS, help="override configured methods")
    parser.add_argument("--seeds", nargs="+", type=int, help="override configured training seeds")
    parser.add_argument("--epoch-count", type=int, help="override configured epoch count (default: 500)")
    parser.add_argument("--output-root", help="override configured output root")
    parser.add_argument("--python", help="Python executable used for every trainer")
    parser.add_argument("--gpu-ids", type=parse_gpu_ids, help="physical GPU ids, e.g. 0,1")
    parser.add_argument(
        "--parallel-per-gpu",
        type=parse_capacity,
        help="concurrent jobs per GPU (integer or GPU=COUNT pairs such as 0=2,1=4)",
    )
    parser.add_argument(
        "--job-gpu",
        action="append",
        type=parse_job_gpu,
        help="route a method or exact run, e.g. --job-gpu coast:42=1",
    )
    parser.add_argument("--dry-run", action="store_true", help="print commands and paths; do not generate pools or train")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="check the selected CUDA runtime, B200 devices, and all trainer CLIs without generating data",
    )
    parser.add_argument(
        "--expected-gpu-name",
        help="override expected GPU-name substring from config (default: B200)",
    )
    parser.add_argument(
        "--allow-any-gpu",
        action="store_true",
        help="disable the expected GPU-name check; CUDA must still be available",
    )
    parser.add_argument(
        "--generate-shared-pools-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = validate_config(load_config(args))
        if args.generate_shared_pools_only:
            source_hashes = compare_shared_sources()
            generate_shared_pools(config, source_hashes)
            return 0

        jobs = assign_jobs(config)
        source_hashes = compare_shared_sources()
        print_plan(config, jobs, source_hashes)
        if config["_dry_run"]:
            check_no_overwrite(config, jobs)
            print("\n[dry-run] No pools or training outputs were created.")
            return 0

        if config["_preflight_only"]:
            preflight_trainer_clis(config, jobs)
            gpu_info = preflight_gpus(config)
            print("[preflight] B200 runtime and trainer checks passed; no data or training outputs were created.")
            return 0

        check_no_overwrite(config, jobs)
        preflight_trainer_clis(config, jobs)
        gpu_info = preflight_gpus(config)
        pool_manifest = _generate_shared_pools_with_selected_python(config)
        return run_jobs(config, jobs, source_hashes, pool_manifest, gpu_info)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print("[error] {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
