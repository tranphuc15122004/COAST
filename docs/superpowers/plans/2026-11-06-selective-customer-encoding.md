# Selective Dynamic Customer Encoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement row-selective dynamic customer re-encoding (`selective_customer_encoding`) for `COAST2` and `VECTRA` to only re-encode instances where new customer requests have actually been revealed, cutting `GraphEncoder` compute and activation memory by ~70–85% while preserving 100% mathematical and gradient correctness.

**Architecture:** 
1. Core mechanism inside `VECTRA._encode_customers` and `COAST2._encode_customers` tracks `_encoded_cust_mask` per episode. When `selective_customer_encoding=True`, it detects `dirty_rows` via `mask.ne(self._encoded_cust_mask).any(dim=1)`.
2. When `0 < K < B`, it extracts the `K` active rows (`index_select`), runs only those `K` instances through `GraphEncoder`, and updates `cust_enc` and `cust_repr` out-of-place via `index_put` to ensure full PyTorch Autograd and AMP safety.
3. When `K == 0`, it returns immediately (O(1) no-op), eliminating redundant re-encodings.
4. Preserves 100% integrity of `problems/` so data hashes in `data/coast_ablation/manifest.json` remain untouched.

**Tech Stack:** PyTorch 2.x, CUDA (Tesla T4), AMP (autocast/GradScaler), pytest.

**Spec:** Brainstorming discussion on 2026-11-06 covering technical autograd safety, variational dropout dynamics, and ablation matrix protocol.

## Global Constraints

- Never mutate `self.cust_repr` in-place (`cust_repr[dirty] = ...` is forbidden; must use out-of-place `index_put` or tensor reconstruction).
- Do not modify any file in `problems/` (`problems/_env_dtw.py`, `problems/_data_dtw.py`) to keep manifest SHA-256 hashes valid.
- Default `selective_customer_encoding = False` on baseline `VECTRA` so legacy inference and historical checkpoints behave 100% identically.
- Reject incompatible configurations: raise `ValueError` if `selective_customer_encoding=True` and `adaptive_depth=True`.
- Ensure `state_dict()` keys and tensor shapes are completely unchanged (no new learnable parameters).

---

### Task 1: Core Selective Encoding Mechanism in `VECTRA`

**Files:**
- Modify: `MODEL/model/vectra.py`
- Test: `test/test_selective_customer_encoding.py`

**Interfaces:**
- Produces: `VECTRA.selective_customer_encoding: bool`, `VECTRA._reset_customer_encoding()`, `VECTRA._encode_customers_selective(customers, mask, dirty_rows)`
- Consumes: `GraphEncoder`, `depot_embedding`, `cust_embedding`, `cust_project`, `dropout`

- [ ] **Step 1: Write unit test asserting selective encoding mathematically equals full encoding in eval mode (FP32)**

```python
# test/test_selective_customer_encoding.py
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
```

- [ ] **Step 2: Run test to verify it fails before implementation**

Run: `pytest test/test_selective_customer_encoding.py -v`
Expected: FAIL (`AttributeError: 'VECTRA' object has no attribute 'selective_customer_encoding'`)

- [ ] **Step 3: Implement selective customer encoding in `MODEL/model/vectra.py`**

In `VECTRA.__init__`:
- Add `self.selective_customer_encoding = False`
- Add `self._encoded_cust_mask = None`
- Add `self._encoded_nodes_ref = None`
- Add `self._encoded_nodes_version = None`
- Add `self.encoding_stats = {"calls": 0, "rows_encoded": 0, "skipped_calls": 0}`

Add `_reset_customer_encoding()`:
```python
def _reset_customer_encoding(self):
    self._encoded_cust_mask = None
    self._encoded_nodes_ref = None
    self._encoded_nodes_version = None
    self.encoding_stats = {"calls": 0, "rows_encoded": 0, "skipped_calls": 0}
```

Implement `_encode_customers`:
- If `not self.selective_customer_encoding`: execute original code.
- If `self.selective_customer_encoding`:
  - Validate: `if self.adaptive_depth: raise ValueError(...)`
  - Determine if cold start:
    `cold = (self._encoded_cust_mask is None or mask is None or self._encoded_cust_mask.shape != mask.shape or customers is not self._encoded_nodes_ref or customers._version != self._encoded_nodes_version or (mask & ~self._encoded_cust_mask).any())`
  - If `cold`:
    run full batch encode, record `self._encoded_cust_mask = mask.detach().clone()`, `self._encoded_nodes_ref = customers`, `self._encoded_nodes_version = customers._version`, update stats.
  - Else:
    `dirty_rows = mask.ne(self._encoded_cust_mask).any(dim=1)`
    - If `not dirty_rows.any()`: `self.encoding_stats["skipped_calls"] += 1; return`
    - If `dirty_rows.all()`: run full batch, update `_encoded_cust_mask`, return.
    - Else:
      `row_idx = dirty_rows.nonzero(as_tuple=False).squeeze(-1)`
      sub-encode `nodes_sub = customers.index_select(0, row_idx)`, `mask_sub = mask.index_select(0, row_idx)`.
      `sub_enc, sub_repr = self._run_encode_pipeline(nodes_sub, mask_sub)`
      `self.cust_enc = self.cust_enc.index_put((row_idx,), sub_enc.to(self.cust_enc.dtype))`
      `self.cust_repr = self.cust_repr.index_put((row_idx,), sub_repr.to(self.cust_repr.dtype))`
      `self._encoded_cust_mask = mask.detach().clone()`
      `self.encoding_stats["calls"] += 1`
      `self.encoding_stats["rows_encoded"] += row_idx.numel()`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/test_selective_customer_encoding.py::test_selective_encoding_eval_math_equivalence -v`
Expected: PASS

- [ ] **Step 5: Write unit test for autograd gradient equivalence (train mode, FP32, dropout=0)**

```python
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
```

- [ ] **Step 6: Run autograd gradient test**

Run: `pytest test/test_selective_customer_encoding.py::test_selective_encoding_autograd_gradient_equivalence -v`
Expected: PASS with 0 in-place modification errors.

---

### Task 2: Integration with `COAST2` and Training Rollout

**Files:**
- Modify: `MODEL/model/coast2.py`
- Modify: `MODEL/train_coast2.py`
- Test: `test/test_selective_customer_encoding.py`

**Interfaces:**
- Produces: `COAST2.selective_customer_encoding: bool`, compatible `_encode_customers` override
- Consumes: `VECTRA._reset_customer_encoding`, `VECTRA._encode_customers`

- [ ] **Step 1: Write test for COAST2 rollout with selective customer encoding**

```python
def test_coast2_rollout_with_selective_encoding():
    from MODEL.model.coast2 import COAST2, materialize_coast2_model
    from MODEL.train_coast2 import _rollout_training_batch
    from baselines._critic import CriticBaseline
    from argparse import Namespace
    
    model = COAST2(7, 4, model_size=32, layer_count=1, head_count=4, ff_size=64,
                   memory_size=32, lookahead_hidden=32, dropout=0.1)
    materialize_coast2_model(model)
    model.selective_customer_encoding = True
    model.activation_checkpointing = False
    
    critic = CriticBaseline(model, 10, use_qval=False, hidden_size=32)
    
    data = Namespace(veh_count=2, veh_capa=200, veh_speed=1.0)
    nodes = torch.rand(8, 11, 7)
    nodes[:, 0, :2] = 0.5  # depot
    nodes[:, :, 6] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]).unsqueeze(0)
    cust_mask = None
    args = Namespace(pending_cost=2.0, late_cost=1.0)
    
    logps, rewards, bl_vals, alive, contexts = _rollout_training_batch(
        model, critic, data, nodes, cust_mask, args
    )
    assert len(logps) > 0
    assert model.encoding_stats["skipped_calls"] > 0 or model.encoding_stats["rows_encoded"] < len(logps) * 8
```

- [ ] **Step 2: Run test to verify behavior and inspect requirements**

Run: `pytest test/test_selective_customer_encoding.py::test_coast2_rollout_with_selective_encoding -v`

- [ ] **Step 3: Update `COAST2` and `train_coast2.py`**

In `MODEL/model/coast2.py`:
- In `_encode_customers`:
  Ensure if `self.selective_customer_encoding`, it cleanly invokes `super()._encode_customers(customers, mask)`.
  If both `self.activation_checkpointing` and `self.selective_customer_encoding` are set to True, raise `ValueError("Cannot combine activation_checkpointing with selective_customer_encoding")`.
- In `MODEL/train_coast2.py`:
  - In `_rollout_training_batch`:
    Call `model._reset_customer_encoding()` right after `env.reset()`.
    Remove redundant `if env.new_customers: model._encode_customers(env.nodes, env.cust_mask)` before the `while` loop (or let the no-op handle it).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/test_selective_customer_encoding.py -v`
Expected: PASS all tests.

---

### Task 3: Support in Ablation Models & Runners

**Files:**
- Modify: `MODEL/model/ablation_study.py`
- Modify: `script/train_coast2.py`
- Modify: `script/train_coast_ablation.py`
- Test: `test/test_coast_ablation_training.py`

**Interfaces:**
- Produces: CLI argument `--selective-customer-encoding` (default: True for modern runs, False when reproducing legacy exact-baseline runs)

- [ ] **Step 1: Write test for ablation models with selective customer encoding enabled**

```python
def test_ablation_variants_support_selective_encoding():
    from MODEL.model.ablation_study import build_ablation_model
    for variant in ("f11", "f10", "f01", "cm"):
        model = build_ablation_model(variant, 7, 4, model_size=16, layer_count=1, head_count=4, ff_size=32, memory_size=16)
        model.selective_customer_encoding = True
        model._reset_customer_encoding()
        nodes = torch.rand(4, 6, 7)
        mask = torch.zeros(4, 6, dtype=torch.bool)
        model._encode_customers(nodes, mask)
        assert model.cust_repr is not None
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest test/test_selective_customer_encoding.py::test_ablation_variants_support_selective_encoding -v`

- [ ] **Step 3: Add CLI flag and config wiring in `script/train_coast2.py` and `script/train_coast_ablation.py`**

In `script/train_coast2.py`:
- Add argument `--selective-customer-encoding / --no-selective-customer-encoding` (default True).
- When `selective_customer_encoding=True`: set `activation_checkpointing=False` by default (unless user explicitly requests otherwise, which triggers validation error).
- Record `selective_customer_encoding` in `args` and `args.json`.
- In `_model_from_args(args)`: set `model.selective_customer_encoding = getattr(args, "selective_customer_encoding", False)`.

In `script/train_coast_ablation.py`:
- Add argument `--selective-customer-encoding / --no-selective-customer-encoding` (default True).
- Pass to `build_training_args` and set on model during `run_training`.

- [ ] **Step 4: Verify preflight smoke test still passes**

Run: `python script/preflight_coast_ablation.py`
Expected: PASS for all variants `f11`, `f10`, `f01`, `cm`.

---

### Task 4: Hardware & Convergence Verification (Gate R0 & Gate R1)

**Files:**
- Create: `script/benchmark_selective_encoding.py`

- [ ] **Step 1: Create benchmark script to measure peak VRAM, speed (s/batch), and rows encoded**

Benchmark script executes 10 update steps of batch size 512 on GPU:
1. `selective_customer_encoding = False` with `activation_checkpointing = True` (baseline)
2. `selective_customer_encoding = True` with `activation_checkpointing = False` (selective)

Metrics to output:
- Max VRAM allocated & reserved (GiB)
- Wall-clock seconds per batch (s/batch)
- Total rows encoded per batch
- Loss & gradient norm check (finite check)

- [ ] **Step 2: Run benchmark script on Tesla T4**

Run: `python script/benchmark_selective_encoding.py`
Expected:
- Peak VRAM <= 10.0 GiB (safe distance from 14.5 GiB)
- Speed improvement >= 1.5x (e.g. ~1.2-1.6s/batch vs 2.63s/batch)
- Finite loss and gradients on all steps

---

### Task 5: Documentation & Git Commit

**Files:**
- Create: `docs/superpowers/specs/2026-11-06-selective-customer-encoding-design.md`
- Commit: Code and tests

- [ ] **Step 1: Check git status and run full test suite**

Run: `pytest test/test_selective_customer_encoding.py test/test_coast*`
Expected: All tests PASS.

- [ ] **Step 2: Git commit clean changes**

Commit with clear descriptive commit message.
