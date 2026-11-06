import json
import copy
from argparse import Namespace

import pytest
import torch

from MODEL.train_coast2 import (
    _saved_activation_context, clip_gradient_groups, make_optimizer,
    move_critic_to_device, train_update,
)
from MODEL.model.coast2 import COAST2, materialize_coast2_model
from problems import DVRPTW_Dataset
from script.prepare_coast_ablation_data import generate_data_pools
from script.train_coast2 import (
    _prune_epoch_checkpoints, _validate_runtime_optimization_resume,
    build_training_args, run_training,
)


def _tiny_model_and_critic():
    model = COAST2(
        7, 4, model_size=16, layer_count=1, head_count=4, ff_size=32,
        memory_size=16, dropout=0.0,
    )
    from baselines._critic import CriticBaseline
    critic = CriticBaseline(model, cust_count=5, use_qval=False, use_cumul_reward=False)
    return model, critic


def test_optimizer_parameter_groups_are_disjoint_and_learning_rates_match_ac():
    model, critic = _tiny_model_and_critic()
    optimizer, groups = make_optimizer(model, critic, learner_lr=1e-4, critic_lr=1e-3)

    assert groups["ac"] and groups["lookahead"]
    assert len(optimizer.param_groups) == 3
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 1e-4, 1e-3]


def test_critic_project_moves_with_coast2_device_without_baseline_wrapper():
    model, critic = _tiny_model_and_critic()

    move_critic_to_device(critic, torch.device("meta"))

    assert next(critic.parameters()).device.type == "meta"


def test_cuda_amp_overflow_retries_update_in_full_precision():
    if not torch.cuda.is_available():
        pytest.skip("CUDA AMP fallback requires a visible CUDA device")
    from baselines._critic import CriticBaseline
    from MODEL.model.coast2 import materialize_coast2_model
    from MODEL.train_coast2 import move_critic_to_device
    from script.train_coast2 import _model_from_args

    args = build_training_args(
        42, "/tmp/coast2_amp_train.pyth", "/tmp/coast2_amp_val.pyth",
        "/tmp/coast2_amp_run", protocol="smoke", device="cuda",
    )
    args.customers_count, args.vehicles_count = 50, 3
    args.model_size, args.layer_count, args.ff_size, args.memory_size = 128, 2, 256, 128
    args.lookahead_hidden, args.dropout, args.amp, args.no_cuda = 128, 0.1, True, False
    torch.manual_seed(42)
    data = DVRPTW_Dataset.generate(
        batch_size=8, cust_count=50, veh_count=3, dod=args.deg_of_dyna,
        tw_ratio=args.tw_ratio, d_early_ratio=args.appear_early_ratio,
    )
    data.normalize()
    device = torch.device("cuda")
    model = _model_from_args(args).to(device)
    critic = CriticBaseline(model, 50, use_qval=False, use_cumul_reward=False)
    move_critic_to_device(critic, device)
    materialize_coast2_model(model, preserve_rng=False)
    optimizer, groups = make_optimizer(model, critic)
    scaler = torch.amp.GradScaler("cuda", init_scale=65536.0)

    metrics = train_update(
        model, critic, optimizer, groups, data, data.nodes.to(device), None,
        args, device, scaler,
    )

    assert metrics["successful"]
    assert metrics["amp_fallback"]


def test_cuda_saved_activation_offload_preserves_gradients():
    if not torch.cuda.is_available():
        pytest.skip("CUDA activation offload requires a visible CUDA device")
    device = torch.device("cuda")
    value = torch.tensor([2.0], device=device, requires_grad=True)
    weight = torch.tensor([3.0], device=device, requires_grad=True)

    with _saved_activation_context(device, enabled=True):
        loss = (value * weight).square().sum()
    loss.backward()

    assert value.grad.item() == pytest.approx(36.0)
    assert weight.grad.item() == pytest.approx(24.0)


def test_resume_allows_only_the_validated_activation_runtime_transition(tmp_path):
    old_args = build_training_args(
        42, tmp_path / "train.pyth", tmp_path / "valid.pyth", tmp_path / "old",
        protocol="full", device="cuda", activation_offload_cpu=True,
    )
    old = dict(vars(old_args))
    old.pop("activation_checkpointing")
    old.pop("allow_runtime_optimization_resume")
    old["code_sha256"] = dict(old["code_sha256"])
    old["code_sha256"]["MODEL/model/coast2.py"] = "old-model-hash"
    old["code_sha256"]["script/train_coast2.py"] = "old-script-hash"

    new_args = build_training_args(
        42, tmp_path / "train.pyth", tmp_path / "valid.pyth", tmp_path / "new",
        protocol="full", device="cuda", resume_state=tmp_path / "old.pyth",
        activation_checkpointing=True, allow_runtime_optimization_resume=True,
    )
    _validate_runtime_optimization_resume(old, new_args)

    changed_recipe = copy.deepcopy(new_args)
    changed_recipe.batch_size = 256
    with pytest.raises(ValueError, match="training arguments differ"):
        _validate_runtime_optimization_resume(old, changed_recipe)

    previous_time_units = copy.deepcopy(old)
    previous_time_units["lookahead_horizon_scale"] = 480.0
    with pytest.raises(ValueError, match="training arguments differ"):
        _validate_runtime_optimization_resume(previous_time_units, new_args)


def test_full_recipe_and_epoch_shuffle_match_existing_f11_protocol(tmp_path):
    coast2_args = build_training_args(
        123, tmp_path / "train.pyth", tmp_path / "validation.pyth",
        tmp_path / "coast2", protocol="full", device="cuda",
    )
    from script.train_coast_ablation import build_training_args as build_f11_args
    f11_args = build_f11_args(
        "f11", 123, tmp_path / "train.pyth", tmp_path / "validation.pyth",
        tmp_path / "f11", protocol="full", device="cuda",
    )
    for key in (
        "customers_count", "vehicles_count", "epoch_count", "iter_count", "batch_size",
        "test_batch_size", "validation_batch_size", "model_size", "layer_count",
        "head_count", "ff_size", "memory_size", "dropout", "learning_rate",
        "critic_rate", "weight_decay", "max_grad_norm", "entropy_coef", "adv_norm",
        "baseline_type", "critic_use_qval", "loss_use_cumul", "amp", "rng_seed",
    ):
        assert getattr(coast2_args, key) == getattr(f11_args, key), key

    from MODEL.train import _make_training_loader as f11_loader
    from script.train_coast2 import _make_training_loader as coast2_loader
    dataset = torch.utils.data.TensorDataset(torch.arange(32))
    shared = dict(rng_seed=123, batch_size=4, num_workers=0, pin_memory=False)
    old_args = Namespace(**shared, study_variant="f11")
    new_args = Namespace(**shared, no_cuda=True)
    old_order = torch.cat([batch[0] for batch in f11_loader(dataset, old_args, epoch=9)])
    new_order = torch.cat([batch[0] for batch in coast2_loader(dataset, new_args, epoch=9)])
    assert torch.equal(old_order, new_order)


def test_separate_gradient_clipping_is_independent_of_predictor_gradient_magnitude():
    ac_param = torch.nn.Parameter(torch.tensor([0.0]))
    lookahead_param = torch.nn.Parameter(torch.tensor([0.0]))
    ac_param.grad = torch.tensor([3.0])
    lookahead_param.grad = torch.tensor([100.0])

    ac_norm, lookahead_norm = clip_gradient_groups([ac_param], [lookahead_param], 2.0)

    assert ac_norm == 3.0
    assert lookahead_norm == 100.0
    assert ac_param.grad.item() == pytest.approx(2.0)
    assert lookahead_param.grad.item() == pytest.approx(2.0)


def test_epoch_checkpoint_retention_keeps_latest_resume_points_and_best(tmp_path):
    for epoch in range(1, 8):
        (tmp_path / "chkpt_ep{}.pyth".format(epoch)).write_bytes(b"epoch")
    best = tmp_path / "chkpt_best.pyth"
    best.write_bytes(b"best")

    _prune_epoch_checkpoints(tmp_path, keep=3)

    assert sorted(path.name for path in tmp_path.glob("chkpt_ep*.pyth")) == [
        "chkpt_ep5.pyth", "chkpt_ep6.pyth", "chkpt_ep7.pyth"
    ]
    assert best.read_bytes() == b"best"


def test_shadow_auxiliary_stress_preserves_the_ac_parameter_update():
    torch.manual_seed(13)
    data = DVRPTW_Dataset.generate(batch_size=2, cust_count=3, veh_count=1, dod=0.0)
    data.normalize()
    template = COAST2(
        7, 4, model_size=16, layer_count=1, head_count=4, ff_size=32,
        memory_size=16, dropout=0.0,
    )
    from baselines._critic import CriticBaseline
    template_critic = CriticBaseline(template, cust_count=3, use_qval=False, use_cumul_reward=False)
    template_critic.to("cpu")
    materialize_coast2_model(template, preserve_rng=False)
    template.gate_logit.data.fill_(-100.0)
    template.gate_logit.requires_grad_(False)
    model_state = copy.deepcopy(template.state_dict())
    critic_state = copy.deepcopy(template_critic.state_dict())

    ac_updates = []
    grad_norms = []
    predictor_updates = []
    for beta in (0.0, 1.0, 100.0):
        model = COAST2(
            7, 4, model_size=16, layer_count=1, head_count=4, ff_size=32,
            memory_size=16, dropout=0.0,
        )
        critic = CriticBaseline(model, cust_count=3, use_qval=False, use_cumul_reward=False)
        critic.to("cpu")
        materialize_coast2_model(model, preserve_rng=False)
        model.load_state_dict(model_state, strict=True)
        critic.load_state_dict(critic_state)
        optimizer, groups = make_optimizer(model, critic)
        args = Namespace(
            pending_cost=2.0, late_cost=1.0, adv_norm=True, entropy_coef=0.01,
            lookahead_loss_coef=beta, lookahead_huber_beta=1.0,
            max_grad_norm=2.0, amp=False, no_cuda=True,
        )
        torch.manual_seed(2026)
        metrics = train_update(
            model, critic, optimizer, groups, data, data.nodes, data.cust_mask,
            args, torch.device("cpu"),
        )
        assert metrics["successful"]
        ac_updates.append([
            p.detach().clone() for p in model.actor_parameters()
        ] + [p.detach().clone() for p in critic.parameters()])
        predictor_updates.append([p.detach().clone() for p in model.lookahead_parameters()])
        grad_norms.append(metrics["ac_grad_norm"])

    for candidate in ac_updates[1:]:
        for reference, value in zip(ac_updates[0], candidate):
            assert torch.equal(reference, value)
    assert grad_norms[0] == pytest.approx(grad_norms[1], abs=1e-7)
    assert grad_norms[1] == pytest.approx(grad_norms[2], abs=1e-7)
    assert any(not torch.equal(a, b) for a, b in zip(predictor_updates[0], predictor_updates[1]))


def test_one_cpu_smoke_update_saves_strict_checkpoint_and_runs_validation(tmp_path):
    pools = generate_data_pools(
        tmp_path / "pools", seeds=[42], train_count=2, validation_count=4,
        test_count=3, validation_seed=202701, test_seed=202702,
        dataset_kwargs={"cust_count": 5, "veh_count": 2},
    )
    train_path = pools["train"]["42"]["path"]
    valid_path = pools["validation"]["path"]
    output_dir = tmp_path / "run"
    args = build_training_args(
        42, train_path, valid_path, output_dir, protocol="smoke", device="cpu"
    )

    run_training(args)

    checkpoint_path = output_dir / "chkpt_best.pyth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    history = [json.loads(line) for line in (output_dir / "metrics.jsonl").read_text().splitlines()]
    assert checkpoint["format"] == "coast2_postdecision_v1"
    assert checkpoint["training_state"]["global_step"] == 1
    assert checkpoint["model_config"]["horizon_scale"] == 1.0
    assert history[-1]["successful_updates"] == 1
    assert history[-1]["skipped_updates"] == 0

    from MODEL.coast2_checkpoint import load_coast2_model
    from script.eval_coast2 import evaluate_dataset
    restored = load_coast2_model(checkpoint_path, "cpu")
    assert restored.horizon_scale == 1.0
    costs_from_checkpoint = evaluate_dataset(
        restored, DVRPTW_Dataset.load(valid_path), "cpu", batch_size=2
    )
    costs_from_path = evaluate_dataset(
        checkpoint_path, DVRPTW_Dataset.load(valid_path), "cpu", batch_size=2
    )
    assert costs_from_checkpoint.shape == (4,)
    assert torch.isfinite(costs_from_checkpoint).all()
    assert torch.equal(costs_from_checkpoint, costs_from_path)
    assert costs_from_path.mean().item() == pytest.approx(history[-1]["validation_mean_cost"], abs=1e-6)

    resume_args = build_training_args(
        42, train_path, valid_path, output_dir, protocol="smoke", device="cpu",
        resume_state=output_dir / "chkpt_ep1.pyth",
    )
    run_training(resume_args)
    final_checkpoint = torch.load(
        output_dir / "chkpt_final.pyth", map_location="cpu", weights_only=False
    )
    assert final_checkpoint["training_state"]["global_step"] == 1

    manifest_before = (output_dir / "run_manifest.json").read_bytes()
    bad_resume_path = output_dir / "bad_resume.pyth"
    bad_checkpoint = torch.load(
        output_dir / "chkpt_ep1.pyth", map_location="cpu", weights_only=False
    )
    bad_checkpoint["training_signature"] = "invalid"
    torch.save(bad_checkpoint, bad_resume_path)
    bad_resume_args = build_training_args(
        42, train_path, valid_path, output_dir, protocol="smoke", device="cpu",
        resume_state=bad_resume_path,
    )
    with pytest.raises(ValueError, match="signature"):
        run_training(bad_resume_args)
    assert (output_dir / "run_manifest.json").read_bytes() == manifest_before
