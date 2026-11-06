import torch
import pytest

from MODEL.model.coast2 import COAST2, select_candidate_context
from problems import DVRPTW_Dataset, DVRPTW_Environment


def test_actor_and_auxiliary_losses_have_disjoint_gradient_routes():
    data = DVRPTW_Dataset.generate(batch_size=2, cust_count=5, veh_count=2, dod=1.0)
    data.normalize()
    env = DVRPTW_Environment(data, None, None, 2, 1)
    env.reset()
    model = COAST2(
        7, 4, model_size=16, layer_count=1, head_count=4, ff_size=32,
        memory_size=16, dropout=0.0,
    )
    model.veh_speed = env.veh_speed
    model._reset_memory(env)
    model._encode_customers(env.nodes, env.cust_mask)

    logp, cache = model.score_step(env)
    actor_loss = -logp[:, 1].mean()
    actor_loss.backward()

    assert model.gate_logit.grad is not None
    assert model.score_fusion[0].weight.grad is not None
    assert all(p.grad is None for p in model.postdecision_lookahead.parameters())

    model.zero_grad(set_to_none=True)
    selected = select_candidate_context(cache["context"], torch.ones(2, 1, dtype=torch.long))
    prediction = model.postdecision_lookahead(selected)
    auxiliary_loss = prediction.square().mean()
    auxiliary_loss.backward()

    assert all(p.grad is not None for p in model.postdecision_lookahead.parameters())
    assert model.gate_logit.grad is None
    assert model.score_fusion[0].weight.grad is None


def test_autocast_actor_lookahead_does_not_cache_predictor_without_gradients():
    if not torch.cuda.is_available():
        pytest.skip("CUDA autocast weight-cache behavior requires a visible CUDA device")
    device = torch.device("cuda")
    data = DVRPTW_Dataset.generate(batch_size=1, cust_count=3, veh_count=1, dod=0.0)
    data.normalize()
    env = DVRPTW_Environment(data, data.nodes.to(device), None, 2, 1)
    env.reset()
    model = COAST2(
        7, 4, model_size=16, layer_count=1, head_count=4, ff_size=32,
        memory_size=16, dropout=0.0,
    ).to(device)
    model.veh_speed = env.veh_speed
    model._reset_memory(env)
    model._encode_customers(env.nodes, env.cust_mask)

    with torch.autocast("cuda", dtype=torch.float16):
        logp, cache = model.score_step(env)
        _, _, selected = model.select_action(env, logp, cache, greedy=True)
        prediction = model.postdecision_lookahead(selected)
        assert prediction.requires_grad
        prediction.square().mean().backward()

    assert all(parameter.grad is not None for parameter in model.postdecision_lookahead.parameters())


def test_activation_checkpointing_preserves_the_cpu_update(tmp_path):
    from baselines._critic import CriticBaseline
    from MODEL.model.coast2 import materialize_coast2_model
    from MODEL.train_coast2 import make_optimizer, move_critic_to_device, train_update
    from script.train_coast2 import _model_from_args, build_training_args

    plain_args = build_training_args(
        42, tmp_path / "train.pyth", tmp_path / "valid.pyth", tmp_path / "plain",
        protocol="smoke", device="cpu",
    )
    checkpoint_args = build_training_args(
        42, tmp_path / "train.pyth", tmp_path / "valid.pyth", tmp_path / "checkpoint",
        protocol="smoke", device="cpu", activation_checkpointing=True,
    )
    assert checkpoint_args.activation_checkpointing

    torch.manual_seed(31)
    data = DVRPTW_Dataset.generate(batch_size=2, cust_count=5, veh_count=2, dod=0.5)
    data.normalize()

    plain_model = _model_from_args(plain_args)
    materialize_coast2_model(plain_model, preserve_rng=False)
    plain_critic = CriticBaseline(plain_model, 5, use_qval=False, use_cumul_reward=False)
    move_critic_to_device(plain_critic, torch.device("cpu"))

    checkpoint_model = _model_from_args(checkpoint_args)
    materialize_coast2_model(checkpoint_model, preserve_rng=False)
    checkpoint_model.load_state_dict(plain_model.state_dict())
    checkpoint_critic = CriticBaseline(
        checkpoint_model, 5, use_qval=False, use_cumul_reward=False
    )
    move_critic_to_device(checkpoint_critic, torch.device("cpu"))
    checkpoint_critic.load_state_dict(plain_critic.state_dict())

    plain_optimizer, plain_groups = make_optimizer(plain_model, plain_critic)
    checkpoint_optimizer, checkpoint_groups = make_optimizer(
        checkpoint_model, checkpoint_critic
    )
    torch.manual_seed(2026)
    plain_metrics = train_update(
        plain_model, plain_critic, plain_optimizer, plain_groups,
        data, data.nodes, data.cust_mask, plain_args, torch.device("cpu"),
    )
    torch.manual_seed(2026)
    checkpoint_metrics = train_update(
        checkpoint_model, checkpoint_critic, checkpoint_optimizer, checkpoint_groups,
        data, data.nodes, data.cust_mask, checkpoint_args, torch.device("cpu"),
    )

    assert plain_metrics["successful"] and checkpoint_metrics["successful"]
    for key in ("loss", "ac_loss", "continuation_loss", "return"):
        assert checkpoint_metrics[key] == pytest.approx(plain_metrics[key], abs=1e-6)
    for plain, checkpointed in zip(plain_model.parameters(), checkpoint_model.parameters()):
        torch.testing.assert_close(checkpointed, plain, atol=1e-6, rtol=1e-5)
    for plain, checkpointed in zip(plain_critic.parameters(), checkpoint_critic.parameters()):
        torch.testing.assert_close(checkpointed, plain, atol=1e-6, rtol=1e-5)


def test_activation_checkpointing_recomputes_customer_encoding_on_backward(monkeypatch):
    from MODEL.model import coast2 as coast2_module

    model = COAST2(
        7, 4, model_size=16, layer_count=1, head_count=4, ff_size=32,
        memory_size=16, dropout=0.0,
    )
    model.train()
    model.activation_checkpointing = True
    data = DVRPTW_Dataset.generate(batch_size=2, cust_count=5, veh_count=2)
    data.normalize()
    calls = []
    original_checkpoint = coast2_module.checkpoint

    def track_recompute(function, *inputs, **kwargs):
        def tracked(*inner_inputs):
            calls.append(True)
            return function(*inner_inputs)
        return original_checkpoint(tracked, *inputs, **kwargs)

    monkeypatch.setattr(coast2_module, "checkpoint", track_recompute)
    model._encode_customers(data.nodes, data.cust_mask)
    model.cust_repr.square().mean().backward()

    assert len(calls) >= 2
    assert any(parameter.grad is not None for parameter in model.cust_encoder.parameters())
