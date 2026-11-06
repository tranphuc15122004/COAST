#!/usr/bin/env python3
"""Train the isolated COAST2 post-decision lookahead variant."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines._critic import CriticBaseline
from MODEL.coast2_checkpoint import read_coast2_checkpoint, save_coast2_checkpoint
from MODEL.model.coast2 import COAST2, materialize_coast2_model
from MODEL.train_coast2 import (
    _make_scaler,
    capture_coast2_training_state,
    evaluate_costs,
    make_optimizer,
    move_critic_to_device,
    restore_coast2_training_state,
    train_update,
)
from problems import DVRPTW_Dataset, DVRPTW_Environment


COAST2_SOURCE_FILES = (
    "MODEL/model/vectra.py",
    "MODEL/model/coast2.py",
    "MODEL/coast2_checkpoint.py",
    "MODEL/train_coast2.py",
    "script/train_coast2.py",
    "script/eval_coast2.py",
    "script/prepare_coast_ablation_data.py",
    "script/run_coast2_matrix.py",
    "script/coast2_matrix.json",
    "layers/Mymodel_layers.py",
    "layers/_mha.py",
    "layers/_loss.py",
    "baselines/_critic.py",
    "baselines/_base.py",
    "problems/_env.py",
    "problems/_env_tw.py",
    "problems/_env_dtw.py",
    "problems/_data.py",
    "problems/_data_dtw.py",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes():
    return {name: _sha256(PROJECT_ROOT / name) for name in COAST2_SOURCE_FILES if (PROJECT_ROOT / name).is_file()}


def _set_random_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _validate_pool(path, count, customers, vehicles, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("{} data pool does not exist: {}".format(label, path))
    data = DVRPTW_Dataset.load(path)
    if len(data) != int(count):
        raise ValueError("{} pool has {} instances; expected {}".format(label, len(data), count))
    if data.nodes.size(1) != int(customers) + 1 or data.veh_count != int(vehicles):
        raise ValueError("{} pool shape or fleet size differs from the COAST2 protocol".format(label))
    if data.veh_capa != 200 or data.veh_speed != 1:
        raise ValueError("{} pool capacity/speed differs from the locked n50m3 protocol".format(label))


def _validate_data_manifest(manifest, args):
    from script.prepare_coast_ablation_data import data_source_hashes

    protocol = manifest.get("protocol", {})
    if manifest.get("code_sha256") != data_source_hashes():
        raise ValueError("Data-generation/simulator code differs from the shared pool manifest")
    audit = manifest.get("disjoint_instance_audit", {})
    expected_count = (
        protocol.get("train_count_per_seed", 0) * len(manifest.get("train", {}))
        + protocol.get("validation_count", 0)
        + protocol.get("primary_test_count", 0)
    )
    if (audit.get("passed") is not True
            or audit.get("algorithm") != "blake2b-128 over raw node-feature bytes"
            or audit.get("instances_checked") != expected_count):
        raise ValueError("Shared data manifest does not certify raw-instance disjointness")
    for flag in (
        "shared_train_pool_across_variants",
        "shared_validation_across_variants",
        "primary_test_not_used_for_checkpoint_selection",
    ):
        if protocol.get(flag) is not True:
            raise ValueError("Shared data manifest does not declare {}".format(flag))
    if protocol.get("train_count_per_seed") != args.iter_count * args.batch_size:
        raise ValueError("Manifest train-pool size differs from the COAST2 update schedule")
    if protocol.get("validation_count") != args.test_batch_size:
        raise ValueError("Manifest validation size differs from the COAST2 validation size")
    if args.run_protocol == "full" and protocol.get("primary_test_count") != 1000:
        raise ValueError("Full COAST2 runs require a separate 1,000-instance primary test set")

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
    if protocol.get("generator_kwargs") != json.loads(json.dumps(expected_generator)):
        raise ValueError("Shared data generator parameters differ from the COAST2 training recipe")

    train_entry = manifest.get("train", {}).get(str(args.rng_seed))
    validation_entry = manifest.get("validation")
    test_entry = manifest.get("primary_test")
    if not train_entry or not validation_entry or not test_entry:
        raise ValueError("Shared data manifest is missing train, validation, or primary-test pools")
    entries = (train_entry, validation_entry, test_entry)
    paths = [Path(entry["path"]).expanduser().resolve() for entry in entries]
    if len(set(paths)) != 3 or len({entry.get("sha256") for entry in entries}) != 3:
        raise ValueError("Shared train, validation, and test pools must be separate")
    if len({entry.get("seed") for entry in entries}) != 3:
        raise ValueError("Shared train, validation, and test pools must have different seeds")
    if paths[:2] != [Path(args.train_data_file).resolve(), Path(args.test_data_file).resolve()]:
        raise ValueError("Selected COAST2 data paths differ from the shared manifest")
    expected_counts = (args.iter_count * args.batch_size, args.test_batch_size, protocol["primary_test_count"])
    for entry, path, count in zip(entries, paths, expected_counts):
        if entry.get("count") != count or not path.is_file() or _sha256(path) != entry.get("sha256"):
            raise ValueError("Shared data pool is missing, wrong-sized, or has a hash mismatch: {}".format(path))


def build_training_args(seed, train_data_file, validation_data_file, output_dir,
                        *, protocol="full", device="cuda", resume_state=None,
                        activation_offload_cpu=False, activation_checkpointing=False,
                        selective_customer_encoding=None,
                        allow_runtime_optimization_resume=False,
                        epochs=None, batch_size=None, iter_count=None):
    """Resolve COAST2's isolated config against the locked F11 AC recipe."""
    if protocol not in ("full", "smoke"):
        raise ValueError("protocol must be 'full' or 'smoke'")
    if device not in ("cuda", "cpu"):
        raise ValueError("device must be 'cuda' or 'cpu'")
    if protocol == "full" and device != "cuda":
        raise ValueError("The locked full protocol requires CUDA and AMP; CPU is supported only for smoke runs")
    if activation_offload_cpu and activation_checkpointing:
        raise ValueError("Choose activation checkpointing or CPU activation offload, not both")
    if selective_customer_encoding is None:
        selective_customer_encoding = not (activation_checkpointing or activation_offload_cpu)
    elif selective_customer_encoding and activation_checkpointing:
        raise ValueError("Cannot combine activation_checkpointing with selective_customer_encoding")
    full = protocol == "full"
    args = argparse.Namespace(
        problem_type="dvrptw",
        customers_count=50 if full else 5,
        vehicles_count=3 if full else 2,
        veh_capa=200,
        veh_speed=1,
        pending_cost=2.0,
        late_cost=1.0,
        min_cust_count=None,
        loc_range=(0, 101),
        dem_range=(5, 41),
        horizon=480,
        dur_range=(10, 31),
        tw_ratio=(0.25, 0.5, 0.75, 1.0),
        tw_range=(30, 91),
        deg_of_dyna=(0.1, 0.25, 0.5, 0.75),
        appear_early_ratio=(0.0, 0.5, 0.75, 1.0),
        model_size=128 if full else 16,
        layer_count=2 if full else 1,
        head_count=4,
        ff_size=256 if full else 32,
        tanh_xplor=10,
        edge_feat_size=8,
        cust_k=None,
        memory_size=128 if full else 16,
        lookahead_hidden=128 if full else 8,
        dropout=0.1 if full else 0.0,
        adaptive_depth=False,
        adaptive_min_layers=1,
        adaptive_easy_ratio=0.6,
        latent_bottleneck=False,
        latent_tokens=32,
        latent_min_nodes=64,
        fusion_mode="mlp",
        linear_fusion_weights=(1.0, 1.0, 1.0),
        epoch_count=(int(epochs) if epochs is not None else (500 if full else 1)),
        iter_count=(int(iter_count) if iter_count is not None else (1000 if full else 1)),
        batch_size=(int(batch_size) if batch_size is not None else (512 if full else 2)),
        test_batch_size=10240 if full else 4,
        validation_batch_size=512 if full else 2,
        learning_rate=1e-4,
        critic_rate=1e-3,
        weight_decay=0.0,
        max_grad_norm=2.0,
        entropy_coef=0.01,
        adv_norm=True,
        baseline_type="critic",
        critic_use_qval=False,
        loss_use_cumul=False,
        rate_decay=None,
        critic_decay=None,
        rng_seed=int(seed),
        num_workers=4 if full else 0,
        pin_memory=bool(full),
        amp=bool(full),
        activation_offload_cpu=bool(activation_offload_cpu),
        activation_checkpointing=bool(activation_checkpointing),
        selective_customer_encoding=bool(selective_customer_encoding),
        allow_runtime_optimization_resume=bool(allow_runtime_optimization_resume),
        no_cuda=device == "cpu",
        verbose=True,
        train_data_file=str(Path(train_data_file).expanduser().resolve()),
        test_data_file=str(Path(validation_data_file).expanduser().resolve()),
        output_dir=str(Path(output_dir).expanduser().resolve()),
        resume_state=None if resume_state is None else str(Path(resume_state).expanduser().resolve()),
        run_protocol=protocol,
        study_variant="coast2",
        study_label="C + O + post-decision Lookahead",
        coast2_lookahead=True,
        lookahead_loss_coef=1.0,
        lookahead_huber_beta=1.0,
        lookahead_cost_scale=10.0,
        lookahead_horizon_scale=1.0,
        lookahead_rank=32,
        lookahead_memory_rank=16,
        lookahead_hidden_size=64,
        lookahead_gate_init=0.05,
        data_manifest=None,
        data_manifest_sha256=None,
        keep_checkpoints=5,
    )
    args.code_sha256 = _source_hashes()
    args.train_data_sha256 = _sha256(args.train_data_file) if Path(args.train_data_file).is_file() else None
    args.validation_data_sha256 = _sha256(args.test_data_file) if Path(args.test_data_file).is_file() else None
    return args


def _model_from_args(args):
    model = COAST2(
        DVRPTW_Dataset.CUST_FEAT_SIZE,
        DVRPTW_Environment.VEH_STATE_SIZE,
        model_size=args.model_size,
        layer_count=args.layer_count,
        head_count=args.head_count,
        ff_size=args.ff_size,
        tanh_xplor=args.tanh_xplor,
        edge_feat_size=args.edge_feat_size,
        cust_k=args.cust_k,
        memory_size=args.memory_size,
        lookahead_hidden=args.lookahead_hidden,
        dropout=args.dropout,
        adaptive_depth=args.adaptive_depth,
        adaptive_min_layers=args.adaptive_min_layers,
        adaptive_easy_ratio=args.adaptive_easy_ratio,
        latent_bottleneck=args.latent_bottleneck,
        latent_tokens=args.latent_tokens,
        latent_min_nodes=args.latent_min_nodes,
        use_edge_features=True,
        use_memory=True,
        use_ownership=True,
        fusion_mode=args.fusion_mode,
        linear_fusion_weights=tuple(args.linear_fusion_weights),
        cost_scale=args.lookahead_cost_scale,
        horizon_scale=float(getattr(args, "lookahead_horizon_scale", 480.0)),
        lookahead_rank=args.lookahead_rank,
        lookahead_memory_rank=args.lookahead_memory_rank,
        lookahead_hidden_size=args.lookahead_hidden_size,
        gate_init=args.lookahead_gate_init,
        selective_customer_encoding=bool(getattr(args, "selective_customer_encoding", True)),
        activation_checkpointing=bool(getattr(args, "activation_checkpointing", False)),
    )
    model.selective_customer_encoding = bool(getattr(args, "selective_customer_encoding", True))
    model.activation_checkpointing = bool(getattr(args, "activation_checkpointing", False))
    return model


def _jsonable_args(args):
    return {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }


def _training_signature(args):
    config = {
        "protocol": args.run_protocol,
        "seed": int(args.rng_seed),
        "schedule": [int(args.epoch_count), int(args.iter_count), int(args.batch_size)],
        "validation": [int(args.test_batch_size), int(args.validation_batch_size)],
        "model": {
            key: getattr(args, key) for key in (
                "model_size", "layer_count", "head_count", "ff_size", "tanh_xplor",
                "edge_feat_size", "cust_k", "memory_size", "lookahead_hidden", "dropout",
                "adaptive_depth", "adaptive_min_layers", "adaptive_easy_ratio",
                "latent_bottleneck", "latent_tokens", "latent_min_nodes", "fusion_mode",
                "linear_fusion_weights",
            )
        },
        "optimizer": [float(args.learning_rate), float(args.critic_rate), float(args.weight_decay), float(args.max_grad_norm)],
        "ac": [bool(args.adv_norm), float(args.entropy_coef), float(args.pending_cost),
               float(args.late_cost), args.baseline_type, bool(args.critic_use_qval),
               bool(args.loss_use_cumul)],
        "lookahead": [float(args.lookahead_loss_coef), float(args.lookahead_huber_beta),
                      float(args.lookahead_cost_scale), int(args.lookahead_rank),
                      int(args.lookahead_memory_rank), int(args.lookahead_hidden_size),
                      float(args.lookahead_gate_init)],
        "runtime": [bool(args.amp), bool(args.no_cuda), int(args.num_workers),
                    bool(args.activation_offload_cpu),
                    bool(args.activation_checkpointing),
                    bool(getattr(args, "selective_customer_encoding", True))],
        "data_sha256": [args.train_data_sha256, args.validation_data_sha256],
        "data_manifest_sha256": args.data_manifest_sha256,
        "code_sha256": args.code_sha256,
    }
    # Preserve the old signature schema when inspecting saved legacy args.
    if hasattr(args, "lookahead_horizon_scale"):
        config["lookahead"].append(float(args.lookahead_horizon_scale))
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), config


def _validate_runtime_optimization_resume(previous_args, args):
    """Allow only the measured CPU-offload -> GPU-checkpointing transition."""
    previous_values = dict(previous_args)
    previous_values.setdefault("activation_checkpointing", False)
    previous_values.setdefault("selective_customer_encoding", False)
    previous = argparse.Namespace(**previous_values)
    _, old_config = _training_signature(previous)
    _, new_config = _training_signature(args)

    old_runtime = list(old_config["runtime"])
    if len(old_runtime) == 4:  # Checkpoints created before checkpointing was added.
        old_runtime.append(False)
    if len(old_runtime) == 5:
        old_runtime.append(bool(previous_values.get("selective_customer_encoding", False)))
    new_runtime = list(new_config["runtime"])
    if (
        old_runtime[:3] != new_runtime[:3]
        or old_runtime[3:5] != [True, False]
        or new_runtime[3:5] != [False, True]
        or old_runtime[5:] != new_runtime[5:]
    ):
        raise ValueError("COAST2 runtime optimization resume requires CPU-offload -> GPU-checkpointing only")

    old_hashes, new_hashes = old_config["code_sha256"], new_config["code_sha256"]
    if set(old_hashes) != set(new_hashes):
        raise ValueError("COAST2 runtime optimization resume has a different source file set")
    changed_sources = sorted(path for path in old_hashes if old_hashes[path] != new_hashes[path])
    allowed_sources = {"MODEL/model/coast2.py", "script/train_coast2.py"}
    if not changed_sources or not set(changed_sources).issubset(allowed_sources):
        raise ValueError("COAST2 runtime optimization resume has unapproved source changes")

    old_base, new_base = json.loads(json.dumps(old_config)), json.loads(json.dumps(new_config))
    for config in (old_base, new_base):
        config.pop("runtime")
        config.pop("code_sha256")
    if old_base != new_base:
        raise ValueError("COAST2 resume training arguments differ beyond the runtime optimization")
    return {
        "transition": "activation_offload_cpu_to_activation_checkpointing",
        "previous_runtime": old_runtime,
        "current_runtime": new_runtime,
        "changed_sources": changed_sources,
    }


def _make_training_loader(data, args, epoch):
    generator = torch.Generator(device="cpu")
    generator.manual_seed((int(args.rng_seed) + 1_000_003 * int(epoch)) % (2**63 - 1))
    return DataLoader(
        data,
        int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory and torch.cuda.is_available() and not args.no_cuda),
        persistent_workers=int(args.num_workers) > 0,
        generator=generator,
    )


def _prune_epoch_checkpoints(output_dir, keep=5):
    paths = []
    for path in Path(output_dir).glob("chkpt_ep*.pyth"):
        try:
            epoch = int(path.name[len("chkpt_ep"):-len(".pyth")])
        except ValueError:
            continue
        paths.append((epoch, path))
    for _, path in sorted(paths, reverse=True)[max(1, int(keep)):]:
        path.unlink(missing_ok=True)


def _train_epoch(model, critic, optimizer, groups, scaler, train_data, args, device, epoch, update_log):
    model.train()
    critic.project.train()
    loader = _make_training_loader(train_data, args, epoch)
    if len(loader) != int(args.iter_count):
        raise ValueError("COAST2 loader has {} batches; protocol requires {}".format(len(loader), args.iter_count))
    totals = {}
    successful = skipped = 0
    for batch_index, batch in enumerate(loader):
        if train_data.cust_mask is None:
            nodes, mask = batch.to(device), None
        else:
            nodes, mask = batch[0].to(device), batch[1].to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        metrics = train_update(
            model, critic, optimizer, groups, train_data, nodes, mask,
            args, device, scaler,
        )
        elapsed = time.perf_counter() - start
        metrics.update({
            "epoch": int(epoch + 1),
            "batch": int(batch_index + 1),
            "elapsed_seconds": elapsed,
            "instances_per_second": int(nodes.size(0)) / max(elapsed, 1e-9),
            "learning_rate_ac": float(optimizer.param_groups[0]["lr"]),
            "learning_rate_lookahead": float(optimizer.param_groups[1]["lr"]),
            "learning_rate_critic": float(optimizer.param_groups[2]["lr"]),
            "peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "peak_reserved_memory_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
            "mfu": None,
        })
        with Path(update_log).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, sort_keys=True) + "\n")
        if not metrics["successful"]:
            skipped += 1
            continue
        successful += 1
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key not in ("epoch", "batch", "successful"):
                totals[key] = totals.get(key, 0.0) + float(value)
    if skipped or successful != int(args.iter_count):
        raise RuntimeError(
            "COAST2 requires {} successful updates and zero skipped; got {} / {}".format(
                args.iter_count, successful, skipped
            )
        )
    return {key: value / successful for key, value in totals.items()}, successful, skipped


def _load_resume(args, signature, model, critic, optimizer, scaler, device):
    checkpoint = read_coast2_checkpoint(args.resume_state, map_location="cpu")
    if checkpoint.get("training_signature") != signature:
        if not args.allow_runtime_optimization_resume:
            raise ValueError("COAST2 resume checkpoint protocol/data/code signature does not match")
        previous_args = checkpoint.get("args")
        if not isinstance(previous_args, dict):
            raise ValueError("COAST2 checkpoint has no saved arguments for runtime transition validation")
        old_signature, _ = _training_signature(argparse.Namespace(
            **{
                **previous_args,
                "activation_checkpointing": previous_args.get("activation_checkpointing", False),
                "selective_customer_encoding": previous_args.get("selective_customer_encoding", False),
            }
        ))
        if checkpoint.get("training_signature") != old_signature:
            raise ValueError("COAST2 checkpoint signature disagrees with its saved arguments")
        args.runtime_optimization_resume = _validate_runtime_optimization_resume(previous_args, args)
    else:
        args.runtime_optimization_resume = None
    model.load_state_dict(checkpoint["model"], strict=True)
    critic.project.load_state_dict(checkpoint["critic"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch, global_step = restore_coast2_training_state(checkpoint["training_state"], scaler)
    metadata = checkpoint.get("metadata", {})
    if metadata.get("epoch") != start_epoch or metadata.get("global_step") != global_step:
        raise ValueError("COAST2 checkpoint metadata disagrees with its training state")
    if global_step != start_epoch * int(args.iter_count):
        raise ValueError("COAST2 resume global step differs from the epoch-boundary schedule")
    return start_epoch, global_step, float(checkpoint["best_validation_cost"])


def run_training(args):
    if args.run_protocol == "full" and (args.no_cuda or not torch.cuda.is_available()):
        raise RuntimeError("The locked COAST2 full protocol requires CUDA and AMP")
    if args.run_protocol == "full" and not args.amp:
        raise RuntimeError("The locked COAST2 full protocol requires AMP")
    if args.no_cuda:
        args.amp = False

    train_path = Path(args.train_data_file).expanduser().resolve()
    validation_path = Path(args.test_data_file).expanduser().resolve()
    if train_path.is_file():
        _validate_pool(train_path, args.iter_count * args.batch_size, args.customers_count, args.vehicles_count, "train")
        args.train_data_sha256 = _sha256(train_path)
    else:
        args.train_data_sha256 = None
    _validate_pool(validation_path, args.test_batch_size, args.customers_count, args.vehicles_count, "validation")
    args.validation_data_sha256 = _sha256(validation_path)
    args.code_sha256.update(_source_hashes())

    manifest_path = train_path.parent / "manifest.json"
    if manifest_path.is_file() and validation_path.parent == train_path.parent:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_data_manifest(manifest, args)
        args.data_manifest = str(manifest_path.resolve())
        args.data_manifest_sha256 = _sha256(manifest_path)
    else:
        args.data_manifest = None
        args.data_manifest_sha256 = None

    output_dir = Path(args.output_dir).expanduser().resolve()
    args.output_dir = str(output_dir)
    if args.resume_state is None and output_dir.exists() and any(output_dir.iterdir()):
        print("[train_coast2] Warning: output directory is non-empty: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    signature, signature_config = _training_signature(args)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    _set_random_seed(args.rng_seed)

    if train_path.is_file():
        train_data = DVRPTW_Dataset.load(train_path)
    else:
        print("[train_coast2] Training file not found at '{}', generating training data...".format(train_path))
        gen_count = int(args.iter_count) * int(args.batch_size)
        train_data = DVRPTW_Dataset.generate(
            gen_count,
            cust_count=args.customers_count,
            veh_count=args.vehicles_count,
            veh_capa=args.veh_capa,
            veh_speed=args.veh_speed,
            horizon=args.horizon,
            cust_dur_range=args.dur_range,
            cust_tw_range=args.tw_range,
            tw_ratio=args.tw_ratio,
            dod=args.deg_of_dyna,
            d_early_ratio=args.appear_early_ratio,
        )
    validation_data = DVRPTW_Dataset.load(validation_path)
    train_data.normalize()
    validation_data.normalize()

    model = _model_from_args(args).to(device)
    critic = CriticBaseline(model, args.customers_count, use_qval=False, use_cumul_reward=False)
    move_critic_to_device(critic, device)
    materialize_coast2_model(model, preserve_rng=False)
    optimizer, groups = make_optimizer(
        model, critic, args.learning_rate, args.critic_rate, args.weight_decay
    )
    scaler = _make_scaler(bool(args.amp) and device.type == "cuda")
    start_epoch, global_step, best_validation_cost = 0, 0, float("inf")
    if args.resume_state is not None:
        start_epoch, global_step, best_validation_cost = _load_resume(
            args, signature, model, critic, optimizer, scaler, device
        )

    (output_dir / "args.json").write_text(
        json.dumps(_jsonable_args(args), indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps({
            "training_signature": signature,
            "config": signature_config,
            "resume_transition": getattr(args, "runtime_optimization_resume", None),
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    update_log = output_dir / "updates.jsonl"
    epoch_log = output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, int(args.epoch_count)):
        train_metrics, successful, skipped = _train_epoch(
            model, critic, optimizer, groups, scaler,
            train_data, args, device, epoch, update_log,
        )
        global_step += successful
        validation_costs = evaluate_costs(
            model, validation_data, device,
            batch_size=args.validation_batch_size,
            pending_cost=args.pending_cost,
            late_cost=args.late_cost,
        )
        validation_mean = float(validation_costs.mean().item())
        validation_std = float(validation_costs.std(unbiased=False).item())
        if not math.isfinite(validation_mean):
            raise FloatingPointError("COAST2 validation returned a non-finite mean")
        completed_epoch = epoch + 1
        is_best = validation_mean < best_validation_cost
        if is_best:
            best_validation_cost = validation_mean
        state = capture_coast2_training_state(scaler, completed_epoch, global_step)
        checkpoint_fields = {
            "critic": critic.state_dict(),
            "optimizer": optimizer.state_dict(),
            "training_state": state,
            "training_signature": signature,
            "best_validation_cost": best_validation_cost,
            "args": _jsonable_args(args),
        }
        metadata = {
            "epoch": completed_epoch,
            "global_step": global_step,
            "best_validation_cost": best_validation_cost,
            "validation_mean_cost": validation_mean,
            "validation_std_cost": validation_std,
            "successful_updates": successful,
            "skipped_updates": skipped,
        }
        save_coast2_checkpoint(
            model, output_dir / "chkpt_ep{}.pyth".format(completed_epoch),
            metadata=metadata, **checkpoint_fields,
        )
        if is_best:
            save_coast2_checkpoint(
                model, output_dir / "chkpt_best.pyth",
                metadata=metadata, **checkpoint_fields,
            )
        _prune_epoch_checkpoints(output_dir, args.keep_checkpoints)
        row = {
            "epoch": completed_epoch,
            "global_step": global_step,
            "successful_updates": successful,
            "skipped_updates": skipped,
            "train": train_metrics,
            "validation_mean_cost": validation_mean,
            "validation_std_cost": validation_std,
            "best_validation_cost": best_validation_cost,
        }
        with epoch_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            "COAST2 epoch {}/{} updates={} train_loss={:.5g} continuation={:.5g} "
            "val_cost={:.5g} best={:.5g}".format(
                completed_epoch, args.epoch_count, global_step,
                train_metrics["loss"], train_metrics["continuation_loss"],
                validation_mean, best_validation_cost,
            ),
            flush=True,
        )

    final_state = capture_coast2_training_state(scaler, int(args.epoch_count), global_step)
    save_coast2_checkpoint(
        model, output_dir / "chkpt_final.pyth",
        metadata={"epoch": int(args.epoch_count), "global_step": global_step},
        critic=critic.state_dict(), optimizer=optimizer.state_dict(),
        training_state=final_state, training_signature=signature,
        best_validation_cost=best_validation_cost, args=_jsonable_args(args),
    )
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=(42, 123, 456))
    parser.add_argument("--data-dir", default="data/coast_ablation")
    parser.add_argument("--train-data", default=None)
    parser.add_argument("--validation-data", default=None)
    parser.add_argument("--output-root", default="output/coast2")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--protocol", choices=("full", "smoke"), default="full")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume-state", default=None)
    parser.add_argument("--activation-offload-cpu", action="store_true",
                        help="Offload tensors saved for backward to CPU to reduce CUDA memory")
    parser.add_argument("--activation-checkpointing", action="store_true",
                        help="Recompute COAST2 customer and scoring activations during backward to reduce GPU memory")
    parser.add_argument("--selective-customer-encoding", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable selective customer encoding (default: True, incompatible with --activation-checkpointing)")
    parser.add_argument("--epochs", "--epoch-count", dest="epochs", type=int, default=None,
                        help="Override epoch count (default: 500 for full, 1 for smoke)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (default: 512 for full, 2 for smoke)")
    parser.add_argument("--iter-count", "--updates-per-epoch", dest="iter_count", type=int, default=None,
                        help="Override updates per epoch (default: 1000 for full, 1 for smoke)")
    parser.add_argument("--allow-runtime-optimization-resume", action="store_true",
                        help="Allow only the validated CPU-offload to GPU-checkpointing resume transition")
    parser.add_argument("--dry-run", action="store_true")
    cli = parser.parse_args(argv)

    data_dir = Path(cli.data_dir)
    train_file = Path(cli.train_data) if cli.train_data else data_dir / "train_seed{}.pyth".format(cli.seed)
    validation_file = Path(cli.validation_data) if cli.validation_data else data_dir / "validation_10240.pyth"
    run_dir = Path(cli.output_dir) if cli.output_dir else Path(cli.output_root) / "seed{}".format(cli.seed)
    args = build_training_args(
        cli.seed, train_file, validation_file, run_dir,
        protocol=cli.protocol, device=cli.device, resume_state=cli.resume_state,
        activation_offload_cpu=cli.activation_offload_cpu,
        activation_checkpointing=cli.activation_checkpointing,
        selective_customer_encoding=cli.selective_customer_encoding,
        allow_runtime_optimization_resume=cli.allow_runtime_optimization_resume,
        epochs=cli.epochs,
        batch_size=cli.batch_size,
        iter_count=cli.iter_count,
    )
    if cli.dry_run:
        print(json.dumps(_jsonable_args(args), indent=2, sort_keys=True))
        return
    run_training(args)


if __name__ == "__main__":
    main()
