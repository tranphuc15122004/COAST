import torch

from MODEL.model.coast2 import COAST2
from problems import DVRPTW_Dataset, DVRPTW_Environment


def _small_model():
    return COAST2(
        cust_feat_size=7,
        veh_state_size=4,
        model_size=16,
        layer_count=1,
        head_count=4,
        ff_size=32,
        memory_size=16,
        dropout=0.0,
    )


def test_coast2_scoring_context_is_candidate_shaped_finite_and_non_mutating():
    data = DVRPTW_Dataset.generate(
        batch_size=2, cust_count=5, veh_count=2, dod=0.5, d_early_ratio=0.5
    )
    data.normalize()
    env = DVRPTW_Environment(data, None, None, 2, 1)
    env.reset()
    model = _small_model()
    model.veh_speed = env.veh_speed
    model._reset_memory(env)
    model._encode_customers(env.nodes, env.cust_mask)
    before_vehicles = env.vehicles.clone()
    before_served = env.served.clone()
    before_masks = env.cust_mask.clone()

    logp, cache = model.score_step(env)

    assert logp.shape == (2, 6)
    assert torch.isfinite(logp[torch.isfinite(logp)]).all()
    assert cache["scores"].shape == (2, 1, 6)
    assert cache["context"]["candidate_repr"].shape == (2, 6, 16)
    assert cache["context"]["active_state"].shape == (2, 6, 5)
    assert cache["context"]["scalars"].shape == (2, 6, 8)
    assert torch.equal(env.vehicles, before_vehicles)
    assert torch.equal(env.served, before_served)
    assert torch.equal(env.cust_mask, before_masks)


def test_coast2_uses_compact_head_and_does_not_allocate_legacy_lookahead():
    model = COAST2(
        7, 4, model_size=128, layer_count=1, head_count=4, ff_size=32,
        memory_size=128, dropout=0.0,
    )

    assert not any(name.startswith("lookahead_head.") for name, _ in model.named_parameters())
    assert sum(p.numel() for p in model.postdecision_lookahead.parameters()) == 29617
    assert model.gate_logit.numel() == 1
    assert model.postdecision_lookahead.input_size == 232

    from layers.Mymodel_layers import LookaheadHead
    legacy = LookaheadHead(128, hidden_size=128)
    legacy(torch.zeros(1, 1, 128), torch.zeros(1, 1, 128), torch.zeros(1, 1, 1, 128))
    assert sum(p.numel() for p in legacy.parameters()) == 49409
    assert 29617 + 1 < sum(p.numel() for p in legacy.parameters())


def test_postdecision_state_matches_the_environment_action_transition():
    data = DVRPTW_Dataset.generate(batch_size=1, cust_count=5, veh_count=2, dod=0.0)
    data.normalize()
    env = DVRPTW_Environment(data, None, None, 2, 1)
    env.reset()
    model = _small_model().eval()
    model.veh_speed = env.veh_speed
    model._reset_memory(env)
    model._encode_customers(env.nodes, env.cust_mask)
    vehicle = model._repr_vehicle(env.vehicles, env.cur_veh_idx, env.cur_veh_mask)
    edge = model._compute_edge_embedding(env.vehicles, env.nodes, env.cur_veh_idx, env.cur_veh_mask)

    context = model._postdecision_context(env, vehicle, edge)
    candidate = 1
    distance = torch.pairwise_distance(
        env.cur_veh[:, 0, :2], env.nodes[:, candidate, :2]
    )
    arrival = torch.maximum(
        env.cur_veh[:, 0, 3] + distance / env.veh_speed,
        env.nodes[:, candidate, 3],
    )
    expected_available = arrival + env.nodes[:, candidate, 5]
    expected_capacity = env.cur_veh[:, 0, 2] - env.nodes[:, candidate, 2]

    assert torch.allclose(context["active_state"][:, candidate, 0:2], env.nodes[:, candidate, 0:2])
    assert torch.allclose(context["active_state"][:, candidate, 2], expected_capacity)
    assert torch.allclose(
        context["active_state"][:, candidate, 3], expected_available / model.horizon_scale
    )
    assert context["active_state"][:, candidate, 4].item() == 0.0


def test_selected_prediction_matches_the_same_candidate_from_all_action_scoring():
    from MODEL.model.coast2 import select_candidate_context

    data = DVRPTW_Dataset.generate(batch_size=2, cust_count=5, veh_count=2, dod=0.0)
    data.normalize()
    env = DVRPTW_Environment(data, None, None, 2, 1)
    env.reset()
    model = _small_model().eval()
    model.veh_speed = env.veh_speed
    model._reset_memory(env)
    model._encode_customers(env.nodes, env.cust_mask)
    _, cache = model.score_step(env)
    chosen = torch.tensor([[2], [3]], dtype=torch.long)

    selected_context = select_candidate_context(cache["context"], chosen)
    selected_prediction = model.postdecision_lookahead(selected_context)
    all_prediction = cache["predicted_cost_scaled"].squeeze(1).gather(1, chosen)

    assert torch.allclose(selected_prediction, all_prediction, atol=1e-6, rtol=1e-6)


def test_depot_keeps_all_known_customers_in_remaining_pool_and_empty_pool_is_finite():
    data = DVRPTW_Dataset.generate(batch_size=1, cust_count=5, veh_count=1, dod=0.0)
    data.normalize()
    env = DVRPTW_Environment(data, None, None, 2, 1)
    env.reset()
    model = _small_model().eval()
    model.veh_speed = env.veh_speed
    model._reset_memory(env)
    model._encode_customers(env.nodes, env.cust_mask)
    vehicle = model._repr_vehicle(env.vehicles, env.cur_veh_idx, env.cur_veh_mask)
    edge = model._compute_edge_embedding(env.vehicles, env.nodes, env.cur_veh_idx, env.cur_veh_mask)
    context = model._postdecision_context(env, vehicle, edge)

    assert abs(context["scalars"][0, 0, 0].item() - 5 / 50) < 1e-7
    assert abs(context["scalars"][0, 1, 0].item() - 4 / 50) < 1e-7
    assert torch.count_nonzero(context["peer_state"]) == 0
    assert torch.count_nonzero(context["peer_memory"]) == 0

    env.cust_mask[:, 1:] = True
    env._refresh_mask()
    env._update_cur_veh()
    model._encode_customers(env.nodes, env.cust_mask)
    vehicle = model._repr_vehicle(env.vehicles, env.cur_veh_idx, env.cur_veh_mask)
    edge = model._compute_edge_embedding(env.vehicles, env.nodes, env.cur_veh_idx, env.cur_veh_mask)
    empty = model._postdecision_context(env, vehicle, edge)
    assert torch.count_nonzero(empty["remaining_repr"]) == 0
    assert torch.count_nonzero(empty["scalars"][:, :, :7]) == 0
    assert torch.isfinite(model.postdecision_lookahead(empty)).all()


def test_remaining_pool_excludes_served_and_padding_customers():
    data = DVRPTW_Dataset.generate(batch_size=1, cust_count=5, veh_count=2, dod=0.0)
    data.normalize()
    padding = torch.zeros(1, 6, dtype=torch.bool)
    padding[:, 5] = True
    env = DVRPTW_Environment(data, None, padding, 2, 1)
    env.reset()
    env.served[:, 1] = True
    model = _small_model().eval()
    model.veh_speed = env.veh_speed
    model._reset_memory(env)
    model._encode_customers(env.nodes, env.cust_mask)
    vehicle = model._repr_vehicle(env.vehicles, env.cur_veh_idx, env.cur_veh_mask)
    edge = model._compute_edge_embedding(env.vehicles, env.nodes, env.cur_veh_idx, env.cur_veh_mask)

    context = model._postdecision_context(env, vehicle, edge)

    assert abs(context["scalars"][0, 0, 0].item() - 3 / 50) < 1e-7


def test_peer_pool_is_invariant_to_non_active_vehicle_permutation():
    data = DVRPTW_Dataset.generate(batch_size=1, cust_count=5, veh_count=3, dod=0.0)
    data.normalize()
    env = DVRPTW_Environment(data, None, None, 2, 1)
    env.reset()
    model = _small_model().eval()
    model.veh_speed = env.veh_speed
    model._reset_memory(env)
    model._veh_memory.normal_(0.0, 1.0)
    model._encode_customers(env.nodes, env.cust_mask)
    vehicle = model._repr_vehicle(env.vehicles, env.cur_veh_idx, env.cur_veh_mask)
    edge = model._compute_edge_embedding(env.vehicles, env.nodes, env.cur_veh_idx, env.cur_veh_mask)
    before = model._postdecision_context(env, vehicle, edge)

    env.vehicles[:, 1:] = env.vehicles[:, [2, 1]]
    model._veh_memory[:, 1:] = model._veh_memory[:, [2, 1]]
    after = model._postdecision_context(env, vehicle, edge)

    assert torch.allclose(before["peer_state"], after["peer_state"], atol=1e-7, rtol=1e-7)
    assert torch.allclose(before["peer_memory"], after["peer_memory"], atol=1e-7, rtol=1e-7)


def test_valid_candidate_context_does_not_use_hidden_request_attributes():
    data = DVRPTW_Dataset.generate(
        batch_size=1, cust_count=5, veh_count=2, dod=1.0, d_early_ratio=0.0
    )
    data.normalize()
    env = DVRPTW_Environment(data, None, None, 2, 1)
    env.reset()
    env.cust_mask[:, 1] = False
    env._refresh_mask()
    env._update_cur_veh()
    model = _small_model().eval()
    model.veh_speed = env.veh_speed
    model._reset_memory(env)
    model._encode_customers(env.nodes, env.cust_mask)
    vehicle = model._repr_vehicle(env.vehicles, env.cur_veh_idx, env.cur_veh_mask)
    edge = model._compute_edge_embedding(env.vehicles, env.nodes, env.cur_veh_idx, env.cur_veh_mask)
    before = model._postdecision_context(env, vehicle, edge)

    env.nodes[:, 2, 0:6] += 17.0
    after = model._postdecision_context(env, vehicle, edge)

    for field in ("remaining_repr", "active_state", "scalars"):
        assert torch.equal(before[field][:, 1], after[field][:, 1])


def test_zero_gate_matches_the_existing_c_plus_o_scorer(tmp_path):
    from MODEL.model.ablation_study import build_ablation_model

    data = DVRPTW_Dataset.generate(batch_size=1, cust_count=5, veh_count=2, dod=0.0)
    data.normalize()
    base = build_ablation_model(
        "f11", 7, 4, model_size=16, layer_count=1, head_count=4,
        ff_size=32, memory_size=16, lookahead_hidden=8, dropout=0.0,
    ).eval()
    coast2 = COAST2(
        7, 4, model_size=16, layer_count=1, head_count=4, ff_size=32,
        memory_size=16, lookahead_hidden=8, dropout=0.0,
    ).eval()
    dyna_base = DVRPTW_Environment(data, None, None, 2, 1)
    dyna_base.reset()
    base._reset_memory(dyna_base)
    base.veh_speed = dyna_base.veh_speed
    base._encode_customers(dyna_base.nodes, dyna_base.cust_mask)
    vehicle = base._repr_vehicle(dyna_base.vehicles, dyna_base.cur_veh_idx, dyna_base.cur_veh_mask)
    edge = base._compute_edge_embedding(dyna_base.vehicles, dyna_base.nodes, dyna_base.cur_veh_idx, dyna_base.cur_veh_mask)
    owner = base._compute_owner_bias(dyna_base.cur_veh_idx)
    base._update_memory(dyna_base.cur_veh_idx, torch.zeros(1, 1, dtype=torch.long), vehicle, edge)

    shared = base.state_dict()
    coast2.load_state_dict(shared, strict=False)
    coast2.gate_logit.data.fill_(-100.0)
    dyna_coast2 = DVRPTW_Environment(data, None, None, 2, 1)
    dyna_coast2.reset()
    coast2._reset_memory(dyna_coast2)
    coast2.veh_speed = dyna_coast2.veh_speed
    coast2._encode_customers(dyna_coast2.nodes, dyna_coast2.cust_mask)

    base_lookahead = base._compute_lookahead(vehicle, base.cust_repr, edge)
    expected = base._score_customers(
        vehicle, base.cust_repr, edge, owner, base_lookahead, dyna_base.cur_veh_mask
    )
    _, actual_cache = coast2.score_step(dyna_coast2)

    assert torch.allclose(actual_cache["scores"], expected, atol=1e-6, rtol=1e-6)
