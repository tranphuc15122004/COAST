import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from MODEL.train import _make_training_loader, val_epoch
from problems import DVRPTW_Dataset, DVRPTW_Environment
from script.prepare_coast_ablation_data import generate_data_pools
from script.train_coast_ablation import build_training_args, run_training, validate_data_manifest


@pytest.mark.parametrize("variant", ["f11", "f10", "f01", "cm"])
def test_full_run_config_freezes_protocol_and_variant_contract(tmp_path, variant):
    args = build_training_args(
        variant,
        123,
        tmp_path / "train_seed123.pyth",
        tmp_path / "validation.pyth",
        tmp_path / variant / "seed123",
    )
    assert (args.epoch_count, args.iter_count, args.batch_size) == (500, 1000, 512)
    assert args.test_batch_size == 10240
    assert args.validation_batch_size == 512
    assert (args.customers_count, args.vehicles_count, args.problem_type) == (50, 3, "dvrptw")
    assert (args.model_size, args.layer_count, args.head_count, args.ff_size) == (128, 2, 4, 256)
    assert (args.learning_rate, args.critic_rate, args.weight_decay) == (1e-4, 1e-3, 0.0)
    assert args.baseline_type == "critic" and args.adv_norm and args.entropy_coef == 0.01
    assert args.use_lookahead is False and args.rng_seed == 123
    assert args.study_variant == variant
    assert args.ablation_profile == "none"
    assert args.compatibility_uses_edges is (variant != "f01")
    assert args.selective_customer_encoding is True


def test_ablation_selective_customer_encoding_flag(tmp_path):
    args_default = build_training_args(
        "f11", 42, tmp_path / "train.pyth", tmp_path / "valid.pyth", tmp_path / "out1",
        protocol="smoke", device="cpu",
    )
    assert args_default.selective_customer_encoding is True

    args_disabled = build_training_args(
        "f11", 42, tmp_path / "train.pyth", tmp_path / "valid.pyth", tmp_path / "out2",
        protocol="smoke", device="cpu", selective_customer_encoding=False,
    )
    assert args_disabled.selective_customer_encoding is False


def test_full_protocol_cannot_silently_switch_to_cpu_or_disable_amp(tmp_path):
    with pytest.raises(ValueError, match="CUDA"):
        build_training_args(
            "f11", 42, tmp_path / "train.pyth", tmp_path / "validation.pyth",
            tmp_path / "run", device="cpu",
        )


def test_study_loader_uses_same_shuffle_order_for_paired_variants():
    dataset = TensorDataset(torch.arange(32))
    base = dict(batch_size=4, num_workers=0, pin_memory=False, rng_seed=42)
    orders = []
    for variant in ("f11", "f01"):
        args = type("Args", (), {**base, "study_variant": variant})()
        loader = _make_training_loader(dataset, args, epoch=7)
        orders.append(torch.cat([batch[0] for batch in loader]))
    assert torch.equal(orders[0], orders[1])


def test_validation_chunking_preserves_full_pool_greedy_costs():
    data = DVRPTW_Dataset.generate(batch_size=7, cust_count=5, veh_count=2)
    data.normalize()
    env = DVRPTW_Environment(data, None, None, 2, 1)
    from MODEL.model.ablation_study import build_ablation_model

    model = build_ablation_model(
        "f11", 7, 4, model_size=16, layer_count=1, head_count=4,
        ff_size=32, memory_size=16, lookahead_hidden=8, dropout=0.0,
    )
    model.eval()
    whole = val_epoch(type("Args", (), {"problem_type": "dvrptw", "validation_batch_size": 7})(), env, model)
    chunked = val_epoch(type("Args", (), {"problem_type": "dvrptw", "validation_batch_size": 2})(), env, model)
    assert whole[0] == pytest.approx(chunked[0], abs=1e-6)
    assert whole[1] == pytest.approx(chunked[1], abs=1e-6)


def test_data_preparation_reuses_seeded_pools_and_writes_hash_manifest(tmp_path):
    root = tmp_path / "pools"
    manifest = generate_data_pools(
        root,
        seeds=[42],
        train_count=8,
        validation_count=4,
        test_count=3,
        validation_seed=202701,
        test_seed=202702,
        dataset_kwargs={"cust_count": 5, "veh_count": 2},
    )
    assert manifest["train"]["42"]["count"] == 8
    assert manifest["validation"]["count"] == 4
    assert manifest["primary_test"]["count"] == 3
    assert "problems/_data_dtw.py" in manifest["code_sha256"]
    assert "problems/_env_dtw.py" in manifest["code_sha256"]
    assert manifest["disjoint_instance_audit"] == {
        "algorithm": "blake2b-128 over raw node-feature bytes",
        "instances_checked": 15,
        "passed": True,
    }
    for entry in (
        manifest["train"]["42"],
        manifest["validation"],
        manifest["primary_test"],
    ):
        assert len(entry["sha256"]) == 64
        assert Path(entry["path"]).is_file()

    train = DVRPTW_Dataset.load(manifest["train"]["42"]["path"])
    validation = DVRPTW_Dataset.load(manifest["validation"]["path"])
    primary_test = DVRPTW_Dataset.load(manifest["primary_test"]["path"])
    assert len(train) == 8 and len(validation) == 4 and len(primary_test) == 3
    assert not torch.equal(train.nodes[0], validation.nodes[0])
    assert not torch.equal(validation.nodes[0], primary_test.nodes[0])
    saved_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["protocol"] == manifest["protocol"]

    repeat = generate_data_pools(
        tmp_path / "repeat",
        seeds=[42],
        train_count=8,
        validation_count=4,
        test_count=3,
        validation_seed=202701,
        test_seed=202702,
        dataset_kwargs={"cust_count": 5, "veh_count": 2},
    )
    repeated_train = DVRPTW_Dataset.load(repeat["train"]["42"]["path"])
    assert torch.equal(train.nodes, repeated_train.nodes)


def test_training_rejects_a_pool_manifest_with_wrong_generation_contract(tmp_path):
    manifest = generate_data_pools(
        tmp_path / "pools",
        seeds=[42],
        train_count=2,
        validation_count=4,
        test_count=3,
        validation_seed=202701,
        test_seed=202702,
        dataset_kwargs={"cust_count": 5, "veh_count": 2},
    )
    args = build_training_args(
        "f11", 42,
        manifest["train"]["42"]["path"],
        manifest["validation"]["path"],
        tmp_path / "run",
        protocol="smoke",
    )
    validate_data_manifest(manifest, args)

    invalid = json.loads(json.dumps(manifest))
    invalid["primary_test"]["path"] = invalid["validation"]["path"]
    with pytest.raises(ValueError, match="fresh and separate"):
        validate_data_manifest(invalid, args)


@pytest.mark.parametrize("variant", ["f11", "f10", "f01", "cm"])
def test_one_update_critic_training_saves_strictly_loadable_variant(tmp_path, variant):
    train = DVRPTW_Dataset.generate(
        batch_size=2, cust_count=5, veh_count=2, dod=1.0, d_early_ratio=1.0
    )
    valid = DVRPTW_Dataset.generate(
        batch_size=4, cust_count=5, veh_count=2, dod=1.0, d_early_ratio=1.0
    )
    assert (train.nodes[:, 1:, 6] > 0).all()
    train_path = tmp_path / "train.pyth"
    valid_path = tmp_path / "validation.pyth"
    train.save(train_path)
    valid.save(valid_path)
    output_dir = tmp_path / variant / "seed42"
    args = build_training_args(
        variant,
        42,
        train_path,
        valid_path,
        output_dir,
        protocol="smoke",
    )
    run_training(args)

    checkpoint = output_dir / "chkpt_best.pyth"
    assert checkpoint.is_file()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["study_variant"] == variant
    epoch_state = torch.load(output_dir / "chkpt_ep1.pyth", map_location="cpu", weights_only=False)
    assert epoch_state["training_state"]["epoch_complete"] is True
    assert epoch_state["training_state"]["completed_epoch"] == 1
    assert epoch_state["training_state"]["global_step"] == 1
    update_rows = [
        json.loads(line)
        for line in (output_dir / "update_counts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert update_rows == [{
        "epoch": 1,
        "scheduled_updates": 1,
        "successful_updates": 1,
        "skipped_updates": 0,
        "amp_fallbacks": 0,
    }]
    from script.eval_coast_ablation import load_model_from_study_checkpoint

    model = load_model_from_study_checkpoint(checkpoint, "cpu")
    assert model.variant_name == variant
    assert model.use_lookahead is False

    if variant == "f11":
        resume_args = build_training_args(
            variant,
            42,
            train_path,
            valid_path,
            output_dir,
            protocol="smoke",
            resume_state=output_dir / "chkpt_ep1.pyth",
        )
        run_training(resume_args)
