"""Observation isolation and trajectory/context alignment audit for COAST2."""
import copy

import pytest
import torch

from MODEL.model.coast2 import COAST2, materialize_coast2_model
from MODEL.train_coast2 import _pack_selected_context, _rollout_training_batch, make_optimizer, train_update
from baselines._critic import CriticBaseline
from problems import DVRPTW_Dataset, DVRPTW_Environment
from script.train_coast2 import build_training_args


def _fixture(batch_size=2):
    torch.manual_seed(312)
    data = DVRPTW_Dataset.generate(batch_size=batch_size, cust_count=5, veh_count=2, dod=0.0)
    data.normalize()
    data.nodes[:, :, 6] = 0
    data.nodes[:, 3:, 6] = 0.7
    data.nodes[:, :3, :2] = torch.tensor([[0.2, 0.2], [0.3, 0.2], [0.2, 0.4]])
    data.nodes[:, 3:, :2] = torch.tensor([[0.4, 0.3], [0.35, 0.25], [0.3, 0.3]])
    model = COAST2(7, 4, model_size=16, layer_count=1, head_count=4,
                   ff_size=32, memory_size=16, dropout=0.0).eval()
    materialize_coast2_model(model)
    return data, model


def _observe(data, model):
    env = DVRPTW_Environment(data, None, None, 2, 1)
    env.reset()
    model.veh_speed = env.veh_speed
    model._reset_memory(env)
    model._encode_customers(env.nodes, env.cust_mask)
    logp, cache = model.score_step(env)
    return env, logp, cache


@pytest.mark.parametrize("attribute", [
    pytest.param("coordinates", marks=pytest.mark.xfail(strict=True,
        reason="Known leakage: GraphEncoder normalizes distances using hidden coordinates")),
    "demand_times_and_reveal",
])
def test_legal_policy_is_invariant_to_unrevealed_attributes(attribute):
    data, model = _fixture()
    altered = copy.deepcopy(data)
    if attribute == "coordinates":
        altered.nodes[:, 3:, :2] = torch.tensor([[0.95, 0.95], [0.9, 0.95], [0.95, 0.9]])
    else:
        altered.nodes[:, 3:, 2:7] += 0.1
    with torch.no_grad():
        env, logp, cache = _observe(data, model)
        _, changed_logp, changed = _observe(altered, model)
    legal = ~env.cur_veh_mask.squeeze(1)
    torch.testing.assert_close(logp[legal], changed_logp[legal], atol=1e-7, rtol=1e-7)
    for field in ("candidate_repr", "remaining_repr", "scalars", "active_state"):
        value, other = cache["context"][field], changed["context"][field]
        torch.testing.assert_close(value[legal], other[legal], atol=1e-7, rtol=1e-7)


def test_observable_geometry_prototype_removes_hidden_coordinate_dependence():
    data, model = _fixture()
    altered = copy.deepcopy(data)
    altered.nodes[:, 3:, :2] = 0.95

    def observable_geometry(module, inputs, kwargs):
        hidden = inputs[1]
        coords = kwargs["coords"]
        kwargs["coords"] = torch.where(hidden.unsqueeze(-1), coords[:, :1, :], coords)
        return inputs, kwargs

    hook = model.cust_encoder.register_forward_pre_hook(observable_geometry, with_kwargs=True)
    try:
        with torch.no_grad():
            env, original, _ = _observe(data, model)
            _, changed, _ = _observe(altered, model)
        legal = ~env.cur_veh_mask.squeeze(1)
        torch.testing.assert_close(original[legal], changed[legal], atol=1e-7, rtol=1e-7)
    finally:
        hook.remove()


@pytest.mark.xfail(strict=True, reason="Known preprocessing dependence: location bounds include unrevealed nodes")
def test_dataset_normalization_preserves_observed_inputs_when_hidden_coordinates_change():
    torch.manual_seed(314)
    data = DVRPTW_Dataset.generate(batch_size=1, cust_count=5, veh_count=2, dod=0.0)
    data.nodes[:, :, :2] = torch.tensor([[20., 20.], [30., 20.], [20., 40.],
                                        [40., 30.], [35., 25.], [30., 30.]])
    data.nodes[:, :, 6] = 0
    data.nodes[:, 3:, 6] = 300
    altered = copy.deepcopy(data)
    altered.nodes[:, 3:, :2] = 95
    data.normalize()
    altered.normalize()
    torch.testing.assert_close(data.nodes[:, :3], altered.nodes[:, :3], atol=0, rtol=0)


def test_remaining_pool_has_the_manual_visible_customer_statistics():
    data, model = _fixture()
    with torch.no_grad():
        env, _, cache = _observe(data, model)
    context = cache["context"]
    # Only customers 1 and 2 are known. Serving 1 leaves exactly customer 2.
    torch.testing.assert_close(context["remaining_repr"][:, 1], model.cust_repr[:, 2])
    torch.testing.assert_close(context["scalars"][:, 1, 0], torch.full((2,), 1 / 50))
    torch.testing.assert_close(context["scalars"][:, 1, 1], env.nodes[:, 2, 2] / env.veh_capa)
    torch.testing.assert_close(context["scalars"][:, 1, 2:5], env.nodes[:, 2, 3:6] / model.horizon_scale)
    torch.testing.assert_close(context["scalars"][:, 1, 5:7], env.nodes[:, 2, :2])
    torch.testing.assert_close(context["remaining_repr"][:, 0], model.cust_repr[:, 1:3].mean(1))


@pytest.mark.parametrize("candidate", [0, 1])
def test_active_postdecision_state_matches_actual_environment_step(candidate):
    data, model = _fixture()
    data.nodes[:, :, 6] = 0
    with torch.no_grad():
        env, _, cache = _observe(data, model)
        active_idx = env.cur_veh_idx.clone()
        expected = cache["context"]["active_state"][:, candidate].clone()
        env.step(torch.full_like(active_idx, candidate))
        actual = env.vehicles.gather(1, active_idx[:, :, None].expand(-1, -1, 4)).squeeze(1)
    actual[:, 3] /= model.horizon_scale
    torch.testing.assert_close(expected[:, :4], actual)
    torch.testing.assert_close(expected[:, 4], env.veh_done.gather(1, active_idx).squeeze(1).float())


def test_dynamic_rollout_context_packing_matches_candidate_predictions_and_inference(tmp_path):
    data, model = _fixture(batch_size=3)
    inference = copy.deepcopy(model)
    critic = CriticBaseline(model, 5, use_qval=False, use_cumul_reward=False)
    args = build_training_args(42, tmp_path / "train", tmp_path / "val", tmp_path / "out",
                               protocol="smoke", device="cpu")
    chosen_predictions, snapshots, actions, frozen = [], [], [], []
    original_select = model.select_action

    def select_greedy(env, logp, cache, greedy=None):
        action, selected_logp, context = original_select(env, logp, cache, greedy=True)
        chosen_predictions.append(cache["predicted_cost_scaled"].squeeze(1).gather(1, action).clone())
        snapshots.append(context)
        frozen.append({key: value.clone() for key, value in context.items()})
        actions.append(action.clone())
        return action, selected_logp, context

    model.select_action = select_greedy
    with torch.no_grad():
        logps, rewards, _, alive, contexts = _rollout_training_batch(
            model, critic, data, data.nodes, data.cust_mask, args
        )
        packed_prediction = model.postdecision_lookahead(_pack_selected_context(contexts))
        packed_prediction = packed_prediction.reshape(len(contexts), 3, 1)
        torch.testing.assert_close(packed_prediction, torch.stack(chosen_predictions), atol=1e-6, rtol=1e-6)
        for snapshot, original in zip(snapshots, frozen):
            for key in snapshot:
                torch.testing.assert_close(snapshot[key], original[key], atol=0, rtol=0)
        inference.greedy = True
        inference_actions, inference_logps, inference_rewards = inference(
            DVRPTW_Environment(data, None, None, 2, 1)
        )
    assert len(inference_actions) == len(actions)
    for index, (_, action) in enumerate(inference_actions):
        torch.testing.assert_close(action, actions[index], atol=0, rtol=0)
        torch.testing.assert_close(inference_logps[index], logps[index])
        torch.testing.assert_close(inference_rewards[index], rewards[index])
    assert torch.stack(alive).shape == packed_prediction.shape


def test_retaining_scoring_activations_matches_the_checkpointed_update(tmp_path):
    data, checkpointed = _fixture()
    checkpointed.activation_checkpointing = True
    retained = copy.deepcopy(checkpointed)
    original_score = retained.score_step

    def score_with_retained_activations(env):
        retained.activation_checkpointing = False
        try:
            return original_score(env)
        finally:
            retained.activation_checkpointing = True

    retained.score_step = score_with_retained_activations
    critics = [CriticBaseline(model, 5, use_qval=False, use_cumul_reward=False)
               for model in (checkpointed, retained)]
    critics[1].load_state_dict(critics[0].state_dict())
    args = build_training_args(42, tmp_path / "train", tmp_path / "val", tmp_path / "out",
                               protocol="smoke", device="cpu", activation_checkpointing=True)
    metrics = []
    for model, critic in zip((checkpointed, retained), critics):
        optimizer, groups = make_optimizer(model, critic)
        torch.manual_seed(2026)
        metrics.append(train_update(model, critic, optimizer, groups, data, data.nodes,
                                    data.cust_mask, args, torch.device("cpu")))
    for key in ("loss", "return", "ac_loss", "continuation_loss"):
        assert metrics[0][key] == pytest.approx(metrics[1][key], abs=1e-6)
    for original, changed in zip(checkpointed.parameters(), retained.parameters()):
        torch.testing.assert_close(original, changed, atol=1e-6, rtol=1e-5)


def test_inference_can_omit_selected_context_copy_without_changing_the_trajectory():
    data, original = _fixture()
    optimized = copy.deepcopy(original)
    original.greedy = optimized.greedy = True

    def select_without_context_copy(env, logp, cache, greedy=None):
        action = logp.argmax(dim=1, keepdim=True)
        optimized._update_memory(env.cur_veh_idx, action, cache["veh_repr"], cache["edge_emb"])
        return action, logp.gather(1, action), None

    optimized.select_action = select_without_context_copy
    with torch.no_grad():
        reference = original(DVRPTW_Environment(data, None, None, 2, 1))
        changed = optimized(DVRPTW_Environment(data, None, None, 2, 1))
    assert len(reference[0]) == len(changed[0])
    for (ref_vehicle, ref_action), (vehicle, action) in zip(reference[0], changed[0]):
        torch.testing.assert_close(ref_vehicle, vehicle)
        torch.testing.assert_close(ref_action, action)
    for index in (1, 2):
        torch.testing.assert_close(torch.stack(reference[index]), torch.stack(changed[index]))
