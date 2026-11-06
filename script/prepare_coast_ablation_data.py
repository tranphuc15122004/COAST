#!/usr/bin/env python3
"""Create fixed, disjoint-by-seed DVRPTW pools for the COAST ablation runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from problems import DVRPTW_Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_SOURCE_FILES = (
    "script/prepare_coast_ablation_data.py",
    "problems/_data_dtw.py",
    "problems/_data.py",
    "problems/_env.py",
    "problems/_env_tw.py",
    "problems/_env_dtw.py",
)


DEFAULT_DATASET_KWARGS = {
    "cust_count": 50,
    "veh_count": 3,
    "veh_capa": 200,
    "veh_speed": 1,
    "min_cust_count": None,
    "cust_loc_range": (0, 101),
    "cust_dem_range": (5, 41),
    "horizon": 480,
    "cust_dur_range": (10, 31),
    "cust_tw_range": (30, 91),
    "tw_ratio": (0.25, 0.5, 0.75, 1.0),
    "dod": (0.1, 0.25, 0.5, 0.75),
    "d_early_ratio": (0.0, 0.5, 0.75, 1.0),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_source_hashes():
    return {name: _sha256(PROJECT_ROOT / name) for name in DATA_SOURCE_FILES}


def _generate_and_save(path: Path, count: int, seed: int, dataset_kwargs: dict, seen_instances):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        data = DVRPTW_Dataset.generate(batch_size=count, **dataset_kwargs)
    raw_rows = data.nodes.contiguous().numpy().view("uint8").reshape(len(data), -1)
    for row_index, row in enumerate(raw_rows):
        signature = hashlib.blake2b(memoryview(row), digest_size=16).digest()
        if signature in seen_instances:
            raise ValueError(
                "Raw DVRPTW instance overlap detected in {} at row {}".format(path.name, row_index)
            )
        seen_instances.add(signature)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.save(path)
    return {
        "path": str(path.resolve()),
        "count": int(len(data)),
        "seed": int(seed),
        "sha256": _sha256(path),
    }


def generate_data_pools(
    output_dir,
    seeds=(42, 123, 456),
    train_count=512_000,
    validation_count=10_240,
    test_count=1_000,
    validation_seed=202701,
    test_seed=202702,
    dataset_kwargs=None,
):
    seeds = [int(seed) for seed in seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Training seeds must be unique")
    stream_seeds = seeds + [int(validation_seed), int(test_seed)]
    if len(set(stream_seeds)) != len(stream_seeds):
        raise ValueError("Train, validation, and primary-test RNG seeds must be distinct")
    if min(train_count, validation_count, test_count) <= 0:
        raise ValueError("All pool counts must be positive")

    params = dict(DEFAULT_DATASET_KWARGS)
    if dataset_kwargs:
        unknown = set(dataset_kwargs) - set(params)
        if unknown:
            raise ValueError("Unknown DVRPTW generator options: {}".format(sorted(unknown)))
        params.update(dataset_kwargs)

    root = Path(output_dir).expanduser().resolve()
    seen_instances = set()
    train = {}
    for seed in seeds:
        train["{}".format(seed)] = _generate_and_save(
            root / "train_seed{}.pyth".format(seed), train_count, seed, params, seen_instances
        )
    validation = _generate_and_save(
        root / "validation_{}.pyth".format(validation_count),
        validation_count,
        validation_seed,
        params,
        seen_instances,
    )
    primary_test = _generate_and_save(
        root / "primary_test_{}.pyth".format(test_count),
        test_count,
        test_seed,
        params,
        seen_instances,
    )

    manifest = {
        "schema_version": 1,
        "code_sha256": data_source_hashes(),
        "disjoint_instance_audit": {
            "algorithm": "blake2b-128 over raw node-feature bytes",
            "instances_checked": len(seen_instances),
            "passed": True,
        },
        "protocol": {
            "dataset": "DVRPTW n50m3 unless overridden for a smoke test",
            "generator": "DVRPTW_Dataset.generate",
            "train_count_per_seed": int(train_count),
            "validation_count": int(validation_count),
            "primary_test_count": int(test_count),
            "shared_train_pool_across_variants": True,
            "shared_validation_across_variants": True,
            "primary_test_not_used_for_checkpoint_selection": True,
            "generator_kwargs": params,
        },
        "train": train,
        "validation": validation,
        "primary_test": primary_test,
    }
    manifest = json.loads(json.dumps(manifest))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 123, 456))
    parser.add_argument("--train-count", type=int, default=512_000)
    parser.add_argument("--validation-count", type=int, default=10_240)
    parser.add_argument("--test-count", type=int, default=1_000)
    parser.add_argument("--validation-seed", type=int, default=202701)
    parser.add_argument("--test-seed", type=int, default=202702)
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    manifest = generate_data_pools(
        args.output_dir,
        args.seeds,
        args.train_count,
        args.validation_count,
        args.test_count,
        args.validation_seed,
        args.test_seed,
    )
    print("Wrote manifest: {}".format(Path(args.output_dir).resolve() / "manifest.json"))
    for seed, item in manifest["train"].items():
        print("train seed {}: {} instances sha256={}".format(seed, item["count"], item["sha256"]))
    print("validation: {} instances sha256={}".format(
        manifest["validation"]["count"], manifest["validation"]["sha256"]
    ))
    print("primary test: {} instances sha256={}".format(
        manifest["primary_test"]["count"], manifest["primary_test"]["sha256"]
    ))


if __name__ == "__main__":
    main()
