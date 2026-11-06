import torch
import pytest
from MODEL.model.vectra import VECTRA
from problems import DVRPTW_Dataset, DVRPTW_Environment


def test_selective_encoding_eval_math_equivalence():
    torch.manual_seed(42)
    model_full = VECTRA(7, 4, model_size=32, layer_count=2, head_count=4, ff_size=64, dropout=0.0)
    model_sel = VECTRA(7, 4, model_size=32, layer_count=2, head_count=4, ff_size=64, dropout=0.0)
    model_sel.load_state_dict(model_full.state_dict())

    model_full.eval()
    model_sel.eval()
    model_sel.selective_customer_encoding = True

    batch_size = 16
    nodes = torch.rand(batch_size, 10, 7)
    mask_0 = torch.zeros(batch_size, 10, dtype=torch.bool)
    mask_0[:, 5:] = True  # dynamic nodes hidden

    # Step 0: Initial encode
    model_full._reset_customer_encoding()
    model_sel._reset_customer_encoding()
    model_full._encode_customers(nodes, mask_0)
    model_sel._encode_customers(nodes, mask_0)

    assert torch.allclose(model_full.cust_repr, model_sel.cust_repr, atol=1e-5)

    # Step 1: Reveal node 5 for only 3 rows (rows 2, 7, 11)
    mask_1 = mask_0.clone()
    mask_1[[2, 7, 11], 5] = False

    model_full._encode_customers(nodes, mask_1)
    model_sel._encode_customers(nodes, mask_1)

    assert torch.allclose(model_full.cust_repr, model_sel.cust_repr, atol=1e-5)


def test_selective_encoding_autograd_gradient_equivalence():
    torch.manual_seed(123)
    model_full = VECTRA(7, 4, model_size=32, layer_count=2, head_count=4, ff_size=64, dropout=0.0)
    model_sel = VECTRA(7, 4, model_size=32, layer_count=2, head_count=4, ff_size=64, dropout=0.0)
    model_sel.load_state_dict(model_full.state_dict())

    model_full.train()
    model_sel.train()
    model_sel.selective_customer_encoding = True

    batch_size = 16
    nodes = torch.rand(batch_size, 10, 7)
    mask_0 = torch.zeros(batch_size, 10, dtype=torch.bool)
    mask_0[:, 5:] = True

    # Step 0
    model_full._reset_customer_encoding()
    model_sel._reset_customer_encoding()
    model_full._encode_customers(nodes, mask_0)
    model_sel._encode_customers(nodes, mask_0)

    # Fake loss on step 0 cust_repr
    loss_0_full = (model_full.cust_repr ** 2).sum()
    loss_0_sel = (model_sel.cust_repr ** 2).sum()

    # Step 1: Reveal node 5 for rows 2, 7
    mask_1 = mask_0.clone()
    mask_1[[2, 7], 5] = False
    model_full._encode_customers(nodes, mask_1)
    model_sel._encode_customers(nodes, mask_1)

    loss_1_full = (model_full.cust_repr * 1.5).sum()
    loss_1_sel = (model_sel.cust_repr * 1.5).sum()

    total_full = loss_0_full + loss_1_full
    total_sel = loss_0_sel + loss_1_sel

    total_full.backward()
    total_sel.backward()

    for (name_f, p_f), (name_s, p_s) in zip(model_full.named_parameters(), model_sel.named_parameters()):
        if p_f.grad is not None:
            assert torch.allclose(p_f.grad, p_s.grad, atol=1e-4), f"Grad mismatch on {name_f}"


def test_selective_encoding_incompatible_adaptive_depth():
    # Incompatible in __init__
    with pytest.raises(ValueError, match="selective_customer_encoding is incompatible with adaptive_depth"):
        VECTRA(7, 4, model_size=32, adaptive_depth=True, selective_customer_encoding=True)

    # Incompatible when setting attribute dynamically
    model = VECTRA(7, 4, model_size=32, adaptive_depth=True, selective_customer_encoding=False)
    model.selective_customer_encoding = True
    nodes = torch.rand(4, 8, 7)
    mask = torch.zeros(4, 8, dtype=torch.bool)
    with pytest.raises(ValueError, match="selective_customer_encoding is incompatible with adaptive_depth"):
        model._encode_customers(nodes, mask)


def test_selective_encoding_state_dict_keys_unchanged():
    model_base = VECTRA(7, 4, model_size=32, selective_customer_encoding=False)
    model_sel = VECTRA(7, 4, model_size=32, selective_customer_encoding=True)

    base_sd = model_base.state_dict()
    sel_sd = model_sel.state_dict()
    assert set(base_sd.keys()) == set(sel_sd.keys())

    # Check shapes of all initialized parameters/buffers
    for k, v_base in base_sd.items():
        v_sel = sel_sd[k]
        if isinstance(v_base, torch.nn.UninitializedParameter):
            assert isinstance(v_sel, torch.nn.UninitializedParameter)
        else:
            assert v_base.shape == v_sel.shape


def test_selective_encoding_stats_and_skipping():
    torch.manual_seed(42)
    model = VECTRA(7, 4, model_size=32, selective_customer_encoding=True)
    model.eval()

    batch_size = 8
    nodes = torch.rand(batch_size, 10, 7)
    mask_0 = torch.zeros(batch_size, 10, dtype=torch.bool)
    mask_0[:, 5:] = True

    model._reset_customer_encoding()
    assert model.encoding_stats == {"calls": 0, "rows_encoded": 0, "skipped_calls": 0}

    # Cold start: all 8 rows encoded
    model._encode_customers(nodes, mask_0)
    assert model.encoding_stats["calls"] == 1
    assert model.encoding_stats["rows_encoded"] == 8
    assert model.encoding_stats["skipped_calls"] == 0

    # Same mask: skipped call!
    model._encode_customers(nodes, mask_0)
    assert model.encoding_stats["calls"] == 1
    assert model.encoding_stats["rows_encoded"] == 8
    assert model.encoding_stats["skipped_calls"] == 1

    # Reveal in 2 rows: rows 1 and 4
    mask_1 = mask_0.clone()
    mask_1[[1, 4], 5] = False
    model._encode_customers(nodes, mask_1)
    assert model.encoding_stats["calls"] == 2
    assert model.encoding_stats["rows_encoded"] == 8 + 2
    assert model.encoding_stats["skipped_calls"] == 1


def test_selective_encoding_default_false():
    model = VECTRA(7, 4, model_size=32)
    assert model.selective_customer_encoding is False


def test_selective_encoding_all_dirty_rows():
    torch.manual_seed(42)
    model = VECTRA(7, 4, model_size=32, selective_customer_encoding=True)
    model.eval()

    batch_size = 8
    nodes = torch.rand(batch_size, 10, 7)
    mask_0 = torch.zeros(batch_size, 10, dtype=torch.bool)
    mask_0[:, 5:] = True

    model._reset_customer_encoding()
    model._encode_customers(nodes, mask_0)
    assert model.encoding_stats["calls"] == 1
    assert model.encoding_stats["rows_encoded"] == 8

    # All rows reveal node 5
    mask_1 = mask_0.clone()
    mask_1[:, 5] = False
    model._encode_customers(nodes, mask_1)
    assert model.encoding_stats["calls"] == 2
    assert model.encoding_stats["rows_encoded"] == 16


def test_selective_encoding_direct_interface():
    torch.manual_seed(42)
    model = VECTRA(7, 4, model_size=32, selective_customer_encoding=True)
    model.eval()

    batch_size = 4
    nodes = torch.rand(batch_size, 8, 7)
    mask = torch.zeros(batch_size, 8, dtype=torch.bool)
    model._reset_customer_encoding()
    model._encode_customers(nodes, mask)

    # Empty dirty slice guard test
    dirty_empty = torch.zeros(batch_size, dtype=torch.bool)
    model._encode_customers_selective(nodes, mask, dirty_empty)
    assert model.encoding_stats["calls"] == 1
    assert model.encoding_stats["rows_encoded"] == 4

    # Non-empty dirty slice
    dirty = torch.tensor([False, True, False, True])
    model._encode_customers_selective(nodes, mask, dirty)
    assert model.encoding_stats["calls"] == 2
    assert model.encoding_stats["rows_encoded"] == 4 + 2


def test_selective_encoding_cold_start_on_new_nodes_or_mutation():
    torch.manual_seed(42)
    model = VECTRA(7, 4, model_size=32, selective_customer_encoding=True)
    model.eval()

    batch_size = 4
    nodes_a = torch.rand(batch_size, 8, 7)
    mask_0 = torch.zeros(batch_size, 8, dtype=torch.bool)
    mask_0[:, 4:] = True

    model._reset_customer_encoding()
    model._encode_customers(nodes_a, mask_0)
    assert model.encoding_stats["calls"] == 1
    assert model.encoding_stats["rows_encoded"] == 4

    # Different tensor reference triggers cold start
    nodes_b = torch.rand(batch_size, 8, 7)
    model._encode_customers(nodes_b, mask_0)
    assert model.encoding_stats["calls"] == 2
    assert model.encoding_stats["rows_encoded"] == 8

    # Regression: a node that was revealed becomes hidden again (triggers cold start)
    mask_1 = mask_0.clone()
    mask_1[:, 4] = False
    model._encode_customers(nodes_b, mask_1)
    assert model.encoding_stats["calls"] == 3
    assert model.encoding_stats["rows_encoded"] == 12

    mask_revert = mask_0.clone()  # node 4 is hidden again
    model._encode_customers(nodes_b, mask_revert)
    # Cold start triggers full re-encode
    assert model.encoding_stats["calls"] == 4
    assert model.encoding_stats["rows_encoded"] == 16


def test_selective_encoding_multistep_rollout_eval_math_equivalence():
    torch.manual_seed(42)
    model_full = VECTRA(7, 4, model_size=32, layer_count=2, head_count=4, ff_size=64, dropout=0.0)
    model_sel = VECTRA(7, 4, model_size=32, layer_count=2, head_count=4, ff_size=64, dropout=0.0)
    model_sel.load_state_dict(model_full.state_dict())
    model_full.eval()
    model_sel.eval()
    model_sel.selective_customer_encoding = True

    batch_size = 8
    num_nodes = 12
    nodes = torch.rand(batch_size, num_nodes, 7)

    # Start with nodes 4..11 hidden
    mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool)
    mask[:, 4:] = True

    model_full._reset_customer_encoding()
    model_sel._reset_customer_encoding()

    # Initial encode
    model_full._encode_customers(nodes, mask)
    model_sel._encode_customers(nodes, mask)
    assert torch.allclose(model_full.cust_repr, model_sel.cust_repr, atol=1e-5)

    # Progressive reveal across 4 steps
    for step in range(4):
        reveal_node = 4 + step
        dirty_rows = torch.rand(batch_size) > 0.4
        if not dirty_rows.any():
            dirty_rows[0] = True
        mask[dirty_rows, reveal_node] = False

        model_full._encode_customers(nodes, mask)
        model_sel._encode_customers(nodes, mask)
        assert torch.allclose(model_full.cust_repr, model_sel.cust_repr, atol=1e-5)


