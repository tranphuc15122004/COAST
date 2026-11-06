#!/usr/bin/env python3
"""Strict checkpoint loading and evaluation for the four locked COAST variants."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from MODEL.model.ablation_study import build_ablation_model, variant_spec
from problems import DVRPTW_Dataset, DVRPTW_Environment
from script.prepare_coast_ablation_data import data_source_hashes
from script.train_coast_ablation import training_code_hashes


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_primary_test_manifest(test_path, manifest, expected_count=None):
    test_path = Path(test_path).expanduser().resolve()
    if manifest.get("code_sha256") != data_source_hashes():
        raise ValueError("Data-generation/simulator code differs from the pool manifest")
    primary = manifest.get("primary_test")
    validation = manifest.get("validation")
    train = manifest.get("train", {})
    if not primary or not validation or not train:
        raise ValueError("Manifest is missing a primary-test, validation, or train pool")
    audit = manifest.get("disjoint_instance_audit", {})
    audit_count = sum(entry.get("count", 0) for entry in train.values())
    audit_count += validation.get("count", 0) + primary.get("count", 0)
    if (
        audit.get("passed") is not True
        or audit.get("algorithm") != "blake2b-128 over raw node-feature bytes"
        or audit.get("instances_checked") != audit_count
    ):
        raise ValueError("Manifest does not certify a fresh primary test with no raw-instance overlap")
    if Path(primary["path"]).expanduser().resolve() != test_path:
        raise ValueError("Evaluator input is not the manifest's fresh primary test")
    all_entries = list(train.values()) + [validation, primary]
    paths = [Path(entry["path"]).expanduser().resolve() for entry in all_entries]
    hashes = [entry.get("sha256") for entry in all_entries]
    if len(set(paths)) != len(paths) or len(set(hashes)) != len(hashes):
        raise ValueError("Train, validation, and fresh primary test pools must be separate")
    if len({entry.get("seed") for entry in all_entries}) != len(all_entries):
        raise ValueError("Train, validation, and fresh primary test must use distinct RNG seeds")
    if expected_count is not None and primary.get("count") != expected_count:
        raise ValueError("Fresh primary test has {} instances; expected {}".format(
            primary.get("count"), expected_count
        ))
    if not test_path.is_file() or _sha256(test_path) != primary.get("sha256"):
        raise ValueError("Fresh primary-test file is missing or its hash does not match the manifest")
    return True


def load_model_from_study_checkpoint(checkpoint_path, device="cpu"):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    config_path = checkpoint_path.parent / "args.json"
    if not config_path.is_file():
        raise FileNotFoundError("Study checkpoint is missing sibling args.json: {}".format(config_path))
    args = json.loads(config_path.read_text(encoding="utf-8"))
    if "study_variant" not in args:
        raise ValueError("args.json does not identify a study_variant: {}".format(config_path))
    required_config = (
        "model_size", "layer_count", "head_count", "ff_size", "edge_feat_size",
        "memory_size", "lookahead_hidden", "tanh_xplor", "dropout", "cust_k",
        "adaptive_depth", "adaptive_min_layers", "adaptive_easy_ratio",
        "latent_bottleneck", "latent_tokens", "latent_min_nodes",
        "fusion_mode", "linear_fusion_weights",
        "use_edge_features", "use_memory", "use_ownership", "use_lookahead",
        "compatibility_uses_edges", "ownership_scorer", "study_label",
        "rng_seed", "train_data_sha256", "validation_data_sha256", "code_sha256",
    )
    missing_config = [key for key in required_config if key not in args]
    if missing_config:
        raise ValueError(
            "Study args.json is missing required model fields: {}".format(
                ", ".join(missing_config)
            )
        )
    spec = variant_spec(args["study_variant"])
    expected_flags = {
        "use_edge_features": spec.use_edge_features,
        "use_memory": spec.use_memory,
        "use_ownership": spec.use_ownership,
        "use_lookahead": spec.use_lookahead,
        "compatibility_uses_edges": spec.compatibility_uses_edges,
        "ownership_scorer": spec.ownership_scorer,
        "study_label": spec.label,
    }
    mismatched_flags = [
        key for key, expected in expected_flags.items()
        if key in args and args[key] != expected
    ]
    if mismatched_flags:
        raise ValueError(
            "Sibling args.json is inconsistent with variant {}: {}".format(
                spec.name, ", ".join(mismatched_flags)
            )
        )
    current_code = training_code_hashes()
    saved_code = args["code_sha256"]
    model_files = (
        "MODEL/model/vectra.py",
        "MODEL/model/ablation_study.py",
        "layers/Mymodel_layers.py",
    )
    mismatched_code = [
        path for path in model_files
        if saved_code.get(path) != current_code.get(path)
    ]
    if mismatched_code:
        raise ValueError(
            "Checkpoint model-code hashes differ from the current implementation: {}".format(
                ", ".join(mismatched_code)
            )
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_variant = checkpoint.get("study_variant") if isinstance(checkpoint, dict) else None
    if checkpoint_variant is not None and checkpoint_variant != spec.name:
        raise ValueError(
            "Checkpoint variant {} disagrees with sibling args {}".format(
                checkpoint_variant, spec.name
            )
        )
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("Expected a training checkpoint containing a model state_dict")

    model_kwargs = {
        key: args[key]
        for key in (
            "model_size", "layer_count", "head_count", "ff_size", "tanh_xplor",
            "edge_feat_size", "cust_k", "memory_size", "lookahead_hidden", "dropout",
            "adaptive_depth", "adaptive_min_layers", "adaptive_easy_ratio",
            "latent_bottleneck", "latent_tokens", "latent_min_nodes", "fusion_mode",
            "linear_fusion_weights", "selective_customer_encoding",
        )
        if key in args
    }
    model = build_ablation_model(spec.name, DVRPTW_Dataset.CUST_FEAT_SIZE, 4, greedy=True, **model_kwargs)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    model.validation_data_sha256 = args.get("validation_data_sha256")
    return model


def evaluate_dataset(model, dataset, device, batch_size=512, env_params=(2.0, 1.0)):
    """Evaluate a normalized dataset in bounded batches and return per-instance costs."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.eval()
    costs = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            data = SimpleNamespace(
                veh_count=dataset.veh_count,
                veh_capa=dataset.veh_capa,
                veh_speed=dataset.veh_speed,
                nodes=dataset.nodes[start:stop],
                cust_mask=None if dataset.cust_mask is None else dataset.cust_mask[start:stop],
            )
            env = DVRPTW_Environment(data, None, None, *env_params)
            env.nodes = env.nodes.to(device)
            if env.init_cust_mask is not None:
                env.init_cust_mask = env.init_cust_mask.to(device)
            _, _, rewards = model(env)
            costs.append(-torch.stack(rewards).sum(dim=0).squeeze(-1).cpu())
    return torch.cat(costs)


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-data", required=True, help="Fresh normalized-at-load .pyth data pool")
    parser.add_argument("--models-dir", required=True, help="Root containing {variant}/seed{seed}/")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--variants", nargs="+", choices=("f11", "f10", "f01", "cm"), default=("f11", "f10", "f01", "cm"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 123, 456))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    test_path = Path(args.test_data).expanduser().resolve()
    manifest_path = test_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Primary-test evaluation requires sibling manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_primary_test_manifest(test_path, manifest, expected_count=1000)
    dataset = DVRPTW_Dataset.load(args.test_data)
    dataset.normalize()
    results = {}
    models_root = Path(args.models_dir)
    for variant in args.variants:
        spec = variant_spec(variant)
        results[variant] = {"label": spec.label, "seeds": {}}
        for seed in args.seeds:
            run_dir = models_root / variant / "seed{}".format(seed)
            checkpoint_path = run_dir / "chkpt_best.pyth"
            if not checkpoint_path.is_file():
                raise FileNotFoundError("Missing best checkpoint: {}".format(checkpoint_path))
            model = load_model_from_study_checkpoint(checkpoint_path, device)
            if model.validation_data_sha256 != manifest["validation"]["sha256"]:
                raise ValueError("Checkpoint was selected with a different validation pool")
            if model.validation_data_sha256 == manifest["primary_test"]["sha256"]:
                raise ValueError("Primary test was reused for checkpoint selection")
            values = evaluate_dataset(model, dataset, device, args.batch_size)
            results[variant]["seeds"][str(seed)] = {
                "mean_cost": values.mean().item(),
                "std_cost": values.std(unbiased=False).item(),
                "num_instances": int(values.numel()),
                "checkpoint": str(checkpoint_path),
            }
            print("{} seed {}: {:.4f} ± {:.4f}".format(
                variant, seed, values.mean().item(), values.std(unbiased=False).item()
            ))
        seed_means = [row["mean_cost"] for row in results[variant]["seeds"].values()]
        results[variant]["mean_across_seeds"] = sum(seed_means) / len(seed_means)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved {}".format(output))


if __name__ == "__main__":
    main()
