import json
import torch
import pytest

from MODEL.model.ablation_study import (
    ROLE_AWARE_GENERIC_OWNERSHIP,
    build_ablation_model,
    variant_spec,
)
from problems import DVRPTW_Dataset, DVRPTW_Environment


SMALL_MODEL = dict(
    model_size=16,
    layer_count=1,
    head_count=4,
    ff_size=32,
    tanh_xplor=10,
    edge_feat_size=8,
    memory_size=16,
    lookahead_hidden=8,
    dropout=0.0,
)


def test_four_variants_encode_the_locked_ablation_contract():
    expected = {
        "f11": (True, True, True, True),
        "f10": (True, False, False, True),
        "f01": (True, True, True, False),
        "cm": (True, True, True, True),
    }
    for name, contract in expected.items():
        spec = variant_spec(name)
        assert (
            spec.use_edge_features,
            spec.use_memory,
            spec.use_ownership,
            spec.compatibility_uses_edges,
        ) == contract
        assert spec.use_lookahead is False

    assert variant_spec("cm").ownership_scorer == ROLE_AWARE_GENERIC_OWNERSHIP
    assert variant_spec("f11").ownership_scorer == "ownership"


def test_f01_removes_edges_from_compatibility_but_keeps_memory_contract():
    torch.manual_seed(3)
    reference = build_ablation_model("f11", 7, 4, **SMALL_MODEL)
    no_compat_edges = build_ablation_model("f01", 7, 4, **SMALL_MODEL)
    for model in (reference, no_compat_edges):
        model.coord_memory.input_proj(torch.zeros(1, 48))
        model.owner_head(torch.zeros(1, 2, 16), torch.zeros(1, 3, 16))
    no_compat_edges.load_state_dict(reference.state_dict(), strict=True)

    with torch.no_grad():
        reference.cross_fusion.edge_bias.weight.fill_(1.0)
        no_compat_edges.cross_fusion.edge_bias.weight.copy_(reference.cross_fusion.edge_bias.weight)
        for layer in reference.score_fusion:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
        reference.score_fusion[0].weight[0, 0] = 1.0
        reference.score_fusion[2].weight[0, 0] = 1.0
        no_compat_edges.score_fusion.load_state_dict(reference.score_fusion.state_dict())

    vehicle = torch.ones(1, 1, 16)
    customers = torch.zeros(1, 3, 16)
    edges = torch.stack((torch.zeros(16), torch.ones(16), 2 * torch.ones(16))).view(1, 1, 3, 16)
    owner = torch.zeros(1, 1, 3)
    lookahead = torch.zeros_like(owner)
    mask = torch.zeros_like(owner, dtype=torch.bool)

    reference_with_edges = reference._score_customers(vehicle, customers, edges, owner, lookahead, mask)
    reference_without_edges = reference._score_customers(vehicle, customers, torch.zeros_like(edges), owner, lookahead, mask)
    ablated_with_edges = no_compat_edges._score_customers(vehicle, customers, edges, owner, lookahead, mask)
    ablated_without_edges = no_compat_edges._score_customers(vehicle, customers, torch.zeros_like(edges), owner, lookahead, mask)

    assert not torch.allclose(reference_with_edges, reference_without_edges)
    assert torch.allclose(ablated_with_edges, ablated_without_edges)

    reference._veh_memory = torch.randn(1, 2, 16)
    no_compat_edges._veh_memory = reference._veh_memory.clone()
    reference.cust_repr = customers
    no_compat_edges.cust_repr = customers
    vehicle_index = torch.tensor([[0]])
    assert torch.allclose(
        reference._compute_owner_bias(vehicle_index),
        no_compat_edges._compute_owner_bias(vehicle_index),
    )

    selected = torch.tensor([[1]])
    reference._update_memory(vehicle_index, selected, vehicle, edges)
    no_compat_edges._update_memory(vehicle_index, selected, vehicle, edges)
    assert torch.allclose(reference._veh_memory, no_compat_edges._veh_memory)


def test_cm_generic_score_uses_active_and_peer_memory_and_backpropagates():
    model = build_ablation_model("cm", 7, 4, **SMALL_MODEL)
    head = model.generic_memory_scorer
    with torch.no_grad():
        head.memory_projection.weight.copy_(torch.eye(16))
        head.context_projection.weight.zero_()
        head.context_projection.bias.zero_()
        head.context_projection.weight[0, 32] = 1.0  # active-memory feature 0
        head.output_projection.weight.zero_()
        head.output_projection.bias.zero_()
        head.output_projection.weight[0, 0] = 1.0

    memory = torch.zeros(1, 2, 16, requires_grad=True)
    memory.data[0, 0, 0] = 1.0
    memory.data[0, 1, 0] = 2.0
    customers = torch.ones(1, 3, 16, requires_grad=True)
    active_zero = head(memory, customers, torch.tensor([[0]]))
    active_one = head(memory, customers, torch.tensor([[1]]))

    assert active_zero.shape == (1, 1, 3)
    assert not torch.allclose(active_zero, active_one)
    active_zero.sum().backward()
    assert memory.grad is not None and torch.isfinite(memory.grad).all()
    assert memory.grad.abs().sum() > 0


def test_cm_scorer_capacity_matches_ownership_head_within_five_percent():
    generic = build_ablation_model("cm", 7, 4, model_size=128, memory_size=128)
    reference = build_ablation_model("f11", 7, 4, model_size=128, memory_size=128)
    memory = torch.zeros(1, 3, 128)
    customers = torch.zeros(1, 5, 128)
    reference.owner_head(memory, customers)  # materialize its LazyLinear

    generic_count = sum(p.numel() for p in generic.generic_memory_scorer.parameters())
    ownership_count = sum(p.numel() for p in reference.owner_head.parameters())
    assert abs(generic_count - ownership_count) / ownership_count <= 0.05


def test_role_aware_memory_scorer_is_peer_permutation_invariant_and_finite_for_one_vehicle():
    model = build_ablation_model("cm", 7, 4, **SMALL_MODEL)
    head = model.generic_memory_scorer.eval()
    memory = torch.randn(2, 4, 16)
    customers = torch.randn(2, 5, 16)
    active = torch.tensor([[1], [3]])
    original = head(memory, customers, active)

    permutation = torch.tensor([2, 0, 3, 1])
    permuted_memory = memory[:, permutation]
    permuted_active = torch.tensor([[3], [2]])
    permuted = head(permuted_memory, customers, permuted_active)
    assert torch.allclose(original, permuted, atol=1e-6)

    one_vehicle = head(memory[:, :1], customers, torch.zeros(2, 1, dtype=torch.long))
    assert one_vehicle.shape == (2, 1, 5)
    assert torch.isfinite(one_vehicle).all()


def _small_eval_model(variant="cm"):
    data = DVRPTW_Dataset.generate(batch_size=2, cust_count=5, veh_count=2)
    data.normalize()
    env = DVRPTW_Environment(data, None, None, 2, 1)
    model = build_ablation_model(variant, 7, 4, greedy=True, **SMALL_MODEL)
    model.eval()
    with torch.no_grad():
        model(env)
    return model, data


def test_saved_study_checkpoint_loads_strictly_from_its_sibling_config(tmp_path):
    from script.eval_coast_ablation import load_model_from_study_checkpoint
    from script.train_coast_ablation import training_code_hashes

    model, _ = _small_eval_model()
    ckpt_dir = tmp_path / "cm" / "seed42"
    ckpt_dir.mkdir(parents=True)
    args = {
        "study_variant": "cm",
        "model_size": 16,
        "layer_count": 1,
        "head_count": 4,
        "ff_size": 32,
        "tanh_xplor": 10,
        "edge_feat_size": 8,
        "memory_size": 16,
        "lookahead_hidden": 8,
        "dropout": 0.0,
        "cust_k": None,
        "tanh_xplor": 10,
        "adaptive_depth": False,
        "adaptive_min_layers": 1,
        "adaptive_easy_ratio": 0.7,
        "latent_bottleneck": False,
        "latent_tokens": 32,
        "latent_min_nodes": 64,
        "use_edge_features": True,
        "use_memory": True,
        "use_ownership": True,
        "use_lookahead": False,
        "compatibility_uses_edges": True,
        "ownership_scorer": ROLE_AWARE_GENERIC_OWNERSHIP,
        "study_label": "C + G",
        "rng_seed": 42,
        "train_data_sha256": "a" * 64,
        "validation_data_sha256": "b" * 64,
        "code_sha256": training_code_hashes(),
        "fusion_mode": "mlp",
        "linear_fusion_weights": [1.0, 1.0, 1.0],
    }
    (ckpt_dir / "args.json").write_text(json.dumps(args), encoding="utf-8")
    torch.save({"model": model.state_dict(), "ep": 0}, ckpt_dir / "chkpt_best.pyth")

    loaded = load_model_from_study_checkpoint(ckpt_dir / "chkpt_best.pyth", "cpu")
    assert loaded.training is False
    assert loaded.use_lookahead is False
    assert loaded.variant_name == "cm"
    assert set(loaded.state_dict()) == set(model.state_dict())

    saved_code_hash = args["code_sha256"]["MODEL/model/vectra.py"]
    args["code_sha256"]["MODEL/model/vectra.py"] = "0" * 64
    (ckpt_dir / "args.json").write_text(json.dumps(args), encoding="utf-8")
    with pytest.raises(ValueError, match="model-code hashes"):
        load_model_from_study_checkpoint(ckpt_dir / "chkpt_best.pyth", "cpu")
    args["code_sha256"]["MODEL/model/vectra.py"] = saved_code_hash

    del args["head_count"]
    (ckpt_dir / "args.json").write_text(json.dumps(args), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required model fields"):
        load_model_from_study_checkpoint(ckpt_dir / "chkpt_best.pyth", "cpu")
    args["head_count"] = 4

    args["use_memory"] = False
    (ckpt_dir / "args.json").write_text(json.dumps(args), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent"):
        load_model_from_study_checkpoint(ckpt_dir / "chkpt_best.pyth", "cpu")


def test_primary_test_evaluator_rejects_validation_pool_reuse(tmp_path):
    import json

    import pytest

    from script.eval_coast_ablation import validate_primary_test_manifest
    from script.prepare_coast_ablation_data import generate_data_pools

    manifest = generate_data_pools(
        tmp_path,
        seeds=[42],
        train_count=2,
        validation_count=3,
        test_count=3,
        validation_seed=202701,
        test_seed=202702,
        dataset_kwargs={"cust_count": 5, "veh_count": 2},
    )
    assert validate_primary_test_manifest(
        manifest["primary_test"]["path"], manifest, expected_count=3
    ) is True

    invalid = json.loads(json.dumps(manifest))
    invalid["primary_test"] = invalid["validation"]
    with pytest.raises(ValueError, match="fresh primary test"):
        validate_primary_test_manifest(
            invalid["validation"]["path"], invalid, expected_count=3
        )


def test_f10_has_no_active_memory_or_ownership_signal():
    model = build_ablation_model("f10", 7, 4, **SMALL_MODEL)
    data = DVRPTW_Dataset.generate(batch_size=2, cust_count=5, veh_count=2)
    env = DVRPTW_Environment(data, None, None, 2, 1)
    model._reset_memory(env)
    model.cust_repr = torch.randn(2, 6, 16)
    assert model._veh_memory is None
    assert model._compute_owner_bias(torch.tensor([[0], [1]])).count_nonzero() == 0


def test_study_does_not_register_dormant_memory_ownership_or_lookahead_parameters():
    f10 = build_ablation_model("f10", 7, 4, **SMALL_MODEL)
    f11 = build_ablation_model("f11", 7, 4, **SMALL_MODEL)
    assert not hasattr(f10, "coord_memory")
    assert not hasattr(f10, "owner_head")
    assert not hasattr(f10, "lookahead_head")
    assert not hasattr(f11, "lookahead_head")
    assert any(name.startswith("coord_memory.") for name in f11.state_dict())
