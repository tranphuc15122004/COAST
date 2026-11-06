"""Time-unit contract for normalized COAST2 training and saved-model inference."""
import copy

import pytest
import torch

from MODEL.coast2_checkpoint import load_coast2_model, save_coast2_checkpoint
from MODEL.model.coast2 import COAST2, materialize_coast2_model
from problems import DVRPTW_Dataset, DVRPTW_Environment
from script.train_coast2 import _model_from_args, _training_signature, build_training_args


def _normalized_data():
    torch.manual_seed(42)
    data = DVRPTW_Dataset.generate(batch_size=2, cust_count=5, veh_count=2, dod=0.0)
    data.normalize()
    data.nodes[:, 1:, 3] = 0.4
    data.nodes[:, 1:, 4] = 0.8
    data.nodes[:, 1:, 5] = 0.05
    return data


def _score(model, data):
    env = DVRPTW_Environment(data, None, None, 2, 1)
    env.reset()
    env.vehicles[:, 0, 3] = 0.2
    env.vehicles[:, 1, 3] = 0.3
    env._update_cur_veh()
    model.veh_speed = env.veh_speed
    model._reset_memory(env)
    model._encode_customers(env.nodes, env.cust_mask)
    logp, cache = model.score_step(env)
    return env, logp, cache


def test_new_training_context_uses_normalized_times_without_dividing_again(tmp_path):
    args = build_training_args(42, tmp_path / "train", tmp_path / "val", tmp_path / "out",
                               protocol="smoke", device="cpu")
    model = _model_from_args(args).eval()
    assert args.lookahead_horizon_scale == 1.0
    assert model.horizon_scale == 1.0
    with torch.no_grad():
        env, _, cache = _score(model, _normalized_data())
    context = cache["context"]
    torch.testing.assert_close(context["scalars"][:, 1, 2:5],
                               torch.tensor([[0.4, 0.8, 0.05]]).expand(2, -1))
    torch.testing.assert_close(context["scalars"][:, 1, 7], torch.full((2,), 0.2))
    torch.testing.assert_close(context["peer_state"][:, 0, 3], torch.full((2,), 0.3))
    distance = torch.pairwise_distance(env.cur_veh[:, 0, :2], env.nodes[:, 1, :2])
    available = torch.maximum(0.2 + distance / env.veh_speed, env.nodes[:, 1, 3]) + 0.05
    torch.testing.assert_close(context["active_state"][:, 1, 3], available)
    _, signature = _training_signature(args)
    assert signature["lookahead"][-1] == 1.0


def test_default_model_expects_normalized_context_times():
    model = COAST2(7, 4, model_size=16, layer_count=1, head_count=4, ff_size=32,
                   memory_size=16, dropout=0.0)
    assert model.horizon_scale == 1.0


@pytest.mark.parametrize("time_scale,omit_saved_scale", [(1.0, False), (480.0, False), (480.0, True)])
def test_inference_restores_the_saved_time_convention(tmp_path, time_scale, omit_saved_scale):
    torch.manual_seed(43)
    model = COAST2(7, 4, model_size=16, layer_count=1, head_count=4, ff_size=32,
                   memory_size=16, dropout=0.0, horizon_scale=time_scale).eval()
    materialize_coast2_model(model)
    data = _normalized_data()
    with torch.no_grad():
        _, reference, context = _score(model, data)
    path = tmp_path / "model.pyth"
    payload = save_coast2_checkpoint(model, path)
    if omit_saved_scale:
        payload["model_config"].pop("horizon_scale")
        torch.save(payload, path)
    restored = load_coast2_model(path).eval()
    assert restored.horizon_scale == time_scale
    with torch.no_grad():
        _, actual, actual_context = _score(restored, copy.deepcopy(data))
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    for field in context["context"]:
        torch.testing.assert_close(actual_context["context"][field], context["context"][field], atol=0, rtol=0)


def test_context_time_scale_changes_the_resume_signature(tmp_path):
    args = build_training_args(42, tmp_path / "train", tmp_path / "val", tmp_path / "out",
                               protocol="smoke", device="cpu")
    previous = copy.deepcopy(args)
    previous.lookahead_horizon_scale = 480.0
    assert _training_signature(args)[0] != _training_signature(previous)[0]
