#!/usr/bin/env python3
"""Train one of the four pre-registered COAST two-branch ablation variants."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from functools import partial
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MODEL.model.ablation_study import build_ablation_model, variant_spec
from problems import DVRPTW_Dataset
from script.prepare_coast_ablation_data import data_source_hashes
from utils import parse_args


TRAIN_SOURCE_FILES = (
    "script/train_coast_ablation.py",
    "MODEL/train.py",
    "MODEL/model/vectra.py",
    "MODEL/model/ablation_study.py",
    "layers/Mymodel_layers.py",
    "baselines/_critic.py",
    "utils/_args.py",
    "utils/_chkpt.py",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_code_hashes():
    return {name: _sha256(PROJECT_ROOT / name) for name in TRAIN_SOURCE_FILES}


def build_training_args(
    variant,
    seed,
    train_data_file,
    validation_data_file,
    output_dir,
    *,
    protocol="full",
    device="cuda",
    resume_state=None,
    selective_customer_encoding=True,
):
    """Resolve a fixed full recipe, with a tiny internal-only smoke recipe."""
    spec = variant_spec(variant)
    if protocol not in ("full", "smoke"):
        raise ValueError("protocol must be 'full' or 'smoke'")
    if device not in ("cuda", "cpu"):
        raise ValueError("device must be 'cuda' or 'cpu'")
    if protocol == "full" and device != "cuda":
        raise ValueError("The locked full protocol requires CUDA and AMP; CPU is supported only for smoke runs")

    if protocol == "full":
        problem = ("50", "3")
        schedule = ("500", "1000", "512", "10240", "512")
        model = ("128", "2", "4", "256", "128", "128", "0.1")
        workers, amp, checkpoint_period = "4", True, "5"
    else:
        problem = ("5", "2")
        schedule = ("1", "1", "2", "4", "2")
        model = ("16", "1", "4", "32", "16", "8", "0.0")
        workers, amp, checkpoint_period = "0", False, "1"

    argv = [
        "--problem-type", "dvrptw",
        "--customers-count", problem[0], "--vehicles-count", problem[1],
        "--veh-capa", "200", "--veh-speed", "1", "--horizon", "480",
        "--loc-range", "0", "101", "--dem-range", "5", "41",
        "--dur-range", "10", "31", "--tw-ratio", "0.25", "0.5", "0.75", "1.0",
        "--tw-range", "30", "91", "--deg-of-dyna", "0.1", "0.25", "0.5", "0.75",
        "--appear-early-ratio", "0.0", "0.5", "0.75", "1.0",
        "--pending-cost", "2", "--late-cost", "1",
        "--model-size", model[0], "--layer-count", model[1], "--head-count", model[2],
        "--ff-size", model[3], "--edge-feat-size", "8", "--memory-size", model[4],
        "--lookahead-hidden", model[5], "--dropout", model[6],
        "--epoch-count", schedule[0], "--iter-count", schedule[1], "--batch-size", schedule[2],
        "--test-batch-size", schedule[3], "--validation-batch-size", schedule[4],
        "--learning-rate", "0.0001", "--critic-rate", "0.001", "--weight-decay", "0",
        "--max-grad-norm", "2", "--entropy-coef", "0.01", "--baseline-type", "critic",
        "--ablation-profile", "none", "--rng-seed", str(int(seed)),
        "--num-workers", workers, "--checkpoint-period", checkpoint_period,
        "--train-data-file", str(train_data_file), "--test-data-file", str(validation_data_file),
        "--output-dir", str(output_dir),
    ]
    if amp:
        argv.append("--amp")
    if device == "cpu":
        argv.append("--no-cuda")
    if resume_state is not None:
        argv.extend(("--resume-state", str(resume_state)))

    args = parse_args(argv)
    args.study_variant = spec.name
    args.study_label = spec.label
    args.ownership_scorer = spec.ownership_scorer
    args.compatibility_uses_edges = spec.compatibility_uses_edges
    args.run_protocol = protocol
    args.validation_batch_size = int(schedule[4])
    args.use_edge_features = spec.use_edge_features
    args.use_memory = spec.use_memory
    args.use_ownership = spec.use_ownership
    args.use_lookahead = spec.use_lookahead
    args.selective_customer_encoding = bool(selective_customer_encoding)
    args.data_manifest = None
    args.train_data_sha256 = _sha256(train_data_file) if Path(train_data_file).is_file() else None
    args.validation_data_sha256 = _sha256(validation_data_file) if Path(validation_data_file).is_file() else None
    args.code_sha256 = training_code_hashes()
    return args


def _validate_pool(path, count, customers, vehicles, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("{} data pool does not exist: {}".format(label, path))
    data = DVRPTW_Dataset.load(path)
    try:
        if len(data) != count:
            raise ValueError("{} pool has {} instances; expected {}".format(label, len(data), count))
        if data.nodes.size(1) != customers + 1 or data.veh_count != vehicles:
            raise ValueError(
                "{} pool shape/vehicle count is ({}, {}), expected ({}, {})".format(
                    label, data.nodes.size(1) - 1, data.veh_count, customers, vehicles
                )
            )
        if data.veh_capa != 200 or data.veh_speed != 1:
            raise ValueError("{} pool capacity/speed differs from the locked n50m3 protocol".format(label))
    finally:
        del data


def validate_data_manifest(manifest, args):
    """Reject pools whose metadata no longer matches the resolved run recipe."""
    protocol = manifest.get("protocol", {})
    if manifest.get("code_sha256") != data_source_hashes():
        raise ValueError("Data-generation/simulator code differs from the pool manifest")
    audit = manifest.get("disjoint_instance_audit", {})
    expected_audit_count = (
        protocol.get("train_count_per_seed", 0) * len(manifest.get("train", {}))
        + protocol.get("validation_count", 0)
        + protocol.get("primary_test_count", 0)
    )
    if (
        audit.get("passed") is not True
        or audit.get("algorithm") != "blake2b-128 over raw node-feature bytes"
        or audit.get("instances_checked") != expected_audit_count
    ):
        raise ValueError("Manifest does not certify raw-instance disjointness across all pools")
    if not protocol.get("shared_train_pool_across_variants"):
        raise ValueError("Manifest does not declare shared train pools across variants")
    if not protocol.get("shared_validation_across_variants"):
        raise ValueError("Manifest does not declare a shared validation pool")
    if not protocol.get("primary_test_not_used_for_checkpoint_selection"):
        raise ValueError("Manifest does not reserve the primary test set")
    if protocol.get("train_count_per_seed") != args.iter_count * args.batch_size:
        raise ValueError("Manifest train-pool size differs from the requested training schedule")
    if protocol.get("validation_count") != args.test_batch_size:
        raise ValueError("Manifest validation-pool size differs from the requested validation size")
    if args.run_protocol == "full" and protocol.get("primary_test_count") != 1000:
        raise ValueError("Full-study manifest must reserve the fresh 1,000-instance primary test")

    expected_generator = {
        "cust_count": args.customers_count,
        "veh_count": args.vehicles_count,
        "veh_capa": args.veh_capa,
        "veh_speed": args.veh_speed,
        "min_cust_count": args.min_cust_count,
        "cust_loc_range": args.loc_range,
        "cust_dem_range": args.dem_range,
        "horizon": args.horizon,
        "cust_dur_range": args.dur_range,
        "cust_tw_range": args.tw_range,
        "tw_ratio": args.tw_ratio,
        "dod": args.deg_of_dyna,
        "d_early_ratio": args.appear_early_ratio,
    }
    json_expected = json.loads(json.dumps(expected_generator))
    if protocol.get("generator_kwargs") != json_expected:
        raise ValueError("Manifest generator parameters differ from the resolved training recipe")

    train_entry = manifest.get("train", {}).get(str(args.rng_seed))
    validation_entry = manifest.get("validation")
    test_entry = manifest.get("primary_test")
    if not train_entry or not validation_entry or not test_entry:
        raise ValueError("Manifest is missing a train, validation, or primary-test pool")
    entries = (train_entry, validation_entry, test_entry)
    paths = [Path(entry["path"]).expanduser().resolve() for entry in entries]
    hashes = [entry.get("sha256") for entry in entries]
    if len(set(paths)) != 3 or len(set(hashes)) != 3:
        raise ValueError("Train, validation, and primary test must be fresh and separate pools")
    if len({entry.get("seed") for entry in entries}) != 3:
        raise ValueError("Train, validation, and primary test must use distinct RNG seeds")
    if paths[0] != Path(args.train_data_file).expanduser().resolve():
        raise ValueError("Selected seed's train pool path differs from the run input")
    if paths[1] != Path(args.test_data_file).expanduser().resolve():
        raise ValueError("Shared validation pool path differs from the run input")
    expected_counts = (args.iter_count * args.batch_size, args.test_batch_size, protocol["primary_test_count"])
    for entry, path, count in zip(entries, paths, expected_counts):
        if entry.get("count") != count or not path.is_file():
            raise ValueError("Manifest pool is missing or has an incorrect instance count: {}".format(path))
        if _sha256(path) != entry.get("sha256"):
            raise ValueError("Manifest pool hash mismatch: {}".format(path))
    return True


def _validate_resume(args):
    resume_path = Path(args.resume_state).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if resume_path.parent != output_dir:
        raise ValueError("Resume checkpoint must be in the configured run output directory")
    config_path = resume_path.parent / "args.json"
    if not config_path.is_file():
        raise FileNotFoundError("Cannot validate resume without sibling args.json")
    import json

    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    fields = (
        "study_variant", "run_protocol", "rng_seed", "problem_type", "customers_count",
        "vehicles_count", "veh_capa", "veh_speed", "horizon", "tw_ratio", "deg_of_dyna",
        "appear_early_ratio", "pending_cost", "late_cost", "model_size", "layer_count",
        "head_count", "ff_size", "edge_feat_size", "memory_size", "cust_k", "dropout",
        "epoch_count", "iter_count", "batch_size", "test_batch_size", "validation_batch_size",
        "learning_rate", "critic_rate", "weight_decay", "max_grad_norm", "entropy_coef",
        "baseline_type", "use_edge_features", "use_memory", "use_ownership", "use_lookahead",
        "compatibility_uses_edges", "ownership_scorer", "selective_customer_encoding", "train_data_sha256",
        "validation_data_sha256", "data_manifest_sha256", "code_sha256",
    )
    mismatched = [
        key for key in fields
        if (saved_config.get(key, False) if key == "selective_customer_encoding" else saved_config.get(key)) != getattr(args, key, None)
    ]
    if mismatched:
        raise ValueError("Resume config differs in locked fields: {}".format(", ".join(mismatched)))
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    if checkpoint.get("study_variant") != args.study_variant:
        raise ValueError("Resume checkpoint is missing or has a mismatched study_variant")
    state = checkpoint.get("training_state", {})
    if not state.get("epoch_complete", False):
        raise ValueError("Study runs resume only from a completed epoch checkpoint")
    if state.get("scaler") is None:
        raise ValueError("Resume checkpoint lacks AMP scaler state; exact resume is unavailable")
    checkpoint_epoch = int(checkpoint["ep"]) + 1
    latest_epochs = []
    for path in resume_path.parent.glob("chkpt_ep*.pyth"):
        try:
            latest_epochs.append(int(path.stem[len("chkpt_ep"):]))
        except ValueError:
            continue
    if not latest_epochs or checkpoint_epoch != max(latest_epochs):
        raise ValueError("Resume only from the latest completed epoch checkpoint")


def run_training(args):
    if args.run_protocol == "full" and not args.no_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable in this environment; use a CUDA-enabled runtime for full runs"
        )
    if args.no_cuda:
        args.amp = False
    expected_train_count = args.iter_count * args.batch_size
    _validate_pool(
        args.train_data_file, expected_train_count, args.customers_count, args.vehicles_count, "train"
    )
    _validate_pool(
        args.test_data_file, args.test_batch_size, args.customers_count, args.vehicles_count, "validation"
    )
    args.train_data_sha256 = _sha256(args.train_data_file)
    args.validation_data_sha256 = _sha256(args.test_data_file)

    train_path = Path(args.train_data_file).resolve()
    validation_path = Path(args.test_data_file).resolve()
    manifest_path = train_path.parent / "manifest.json"
    if args.run_protocol == "full" and (
        not manifest_path.is_file() or validation_path.parent != train_path.parent
    ):
        raise FileNotFoundError(
            "Full-study runs require train, validation, and primary-test pools from one manifest.json"
        )
    if manifest_path.is_file() and validation_path.parent == train_path.parent:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_data_manifest(manifest, args)
        args.data_manifest = str(manifest_path.resolve())
        args.data_manifest_sha256 = _sha256(manifest_path)

    output_dir = Path(args.output_dir).expanduser().resolve()
    args.output_dir = str(output_dir)
    if args.resume_state is None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                "Run directory is non-empty; select its matching epoch checkpoint with --resume-state"
            )
    else:
        _validate_resume(args)

    import MODEL.train as trainer

    previous_model_factory = trainer.VECTRA
    trainer.VECTRA = partial(
        build_ablation_model,
        args.study_variant,
        selective_customer_encoding=bool(getattr(args, "selective_customer_encoding", True)),
    )
    try:
        trainer.main(args)
    finally:
        trainer.VECTRA = previous_model_factory


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("f11", "f10", "f01", "cm"))
    parser.add_argument("--seed", required=True, type=int, choices=(42, 123, 456))
    parser.add_argument("--data-dir", default="data/coast_ablation")
    parser.add_argument("--train-data", default=None)
    parser.add_argument("--validation-data", default=None)
    parser.add_argument("--output-root", default="output/coast_ablation")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume-state", default=None)
    parser.add_argument(
        "--selective-customer-encoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable selective customer encoding (default: True)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved full protocol without training")
    return parser


def main(argv=None):
    cli = _build_parser().parse_args(argv)
    data_dir = Path(cli.data_dir)
    train_file = Path(cli.train_data) if cli.train_data else data_dir / "train_seed{}.pyth".format(cli.seed)
    validation_file = (
        Path(cli.validation_data) if cli.validation_data else data_dir / "validation_10240.pyth"
    )
    output_dir = Path(cli.output_dir) if cli.output_dir else Path(cli.output_root) / cli.variant / "seed{}".format(cli.seed)
    args = build_training_args(
        cli.variant,
        cli.seed,
        train_file,
        validation_file,
        output_dir,
        device=cli.device,
        resume_state=cli.resume_state,
        selective_customer_encoding=cli.selective_customer_encoding,
    )
    if cli.dry_run:
        import json
        print(json.dumps(vars(args), indent=2))
        return
    run_training(args)


if __name__ == "__main__":
    main()
