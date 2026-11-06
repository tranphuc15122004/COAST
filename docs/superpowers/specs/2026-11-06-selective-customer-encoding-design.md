# Selective Dynamic Customer Encoding for COAST2 & Ablation Models

**Date:** 2026-11-06  
**Status:** Implemented & Verified on Hardware (Gates R0 & R1 Passed)  
**Authors:** Antigravity Agent & User  
**Branch / Target:** `main`  

---

## 1. Executive Summary

In Dynamic Vehicle Routing with Time Windows (DVRPTW), customer requests are revealed over time as the fleet progresses along the temporal horizon.
- **Legacy 1-time initial encode (prior to commit 39613e4):** Encoded all nodes at $t=0$ with dynamic nodes masked out. In PyTorch, masked nodes were initialized to zero vectors ($\vec{0}$) and never refreshed during the episode. This crippled the `OwnershipHead` and `CoordinationMemory` for newly arrived requests (uniform $1/M$ logits, 0 gradient) and collapsed the post-decision lookahead representations in `COAST2`.
- **Full-batch dynamic re-encode (commits 39613e4 .. 2026-11-05):** Re-encoded the entire batch of 512 instances every time `env.new_customers` became true. Because 512 instances have asynchronous fleet clocks, an event in just 1 of 512 instances triggered full re-encoding across all 512 rows, causing 20–37 full-batch graph transformer passes per rollout (~13,600 forward rows/batch + 13,600 backward recomputed rows under activation checkpointing). Training slowed to ~43.8 min/epoch on Tesla T4.
- **Selective Dynamic Encoding (this work):** Solves both issues by identifying `dirty_rows` (the specific instances in the batch where new customer reveals occurred), running the `GraphEncoder` only on those $K$ instances via out-of-place `index_put`, and returning immediately ($O(1)$ no-op) when no rows have changed.

### Hardware Verification (Tesla T4, Batch 512, $N=50, M=3$):
- **Speedup:** **2.33x faster** (**1.54 s/batch** vs **3.59 s/batch**, bringing epoch time from ~44 min down to **~17–20 min**).
- **VRAM Headroom (Gate R0):** **8.49 GiB allocated / 8.58 GiB reserved** without activation checkpointing, comfortably below the 10.0 GiB target and 14.5 GiB physical limit.
- **Total Compute Savings:** Forward rows encoded dropped from **13,619 to 2,205 rows/batch** (6.2x reduction), and backward recomputation dropped to **0** (eliminated), resulting in a **12.35x reduction** in customer encoding work.

---

## 2. Technical Architecture

### 2.1. In-Place Autograd Safety
In PyTorch Autograd, mutating intermediate tensors in-place increments `tensor._version` and invalidates saved graph variables.
Selective encoding strictly uses out-of-place functional operations:
```python
sub_enc, sub_repr = self._run_encode_pipeline(nodes_sub, mask_sub)
self.cust_enc = self.cust_enc.index_put((row_idx,), sub_enc.to(self.cust_enc.dtype))
self.cust_repr = self.cust_repr.index_put((row_idx,), sub_repr.to(self.cust_repr.dtype))
```

### 2.2. Robust Cache Invalidation & Cold-Start Logic
Inside `VECTRA._encode_customers`:
```python
cold = (
    self._encoded_cust_mask is None
    or self.cust_enc is None
    or self.cust_repr is None
    or mask is None
    or self._encoded_cust_mask.shape != mask.shape
    or self._encoded_cust_mask.device != mask.device
    or customers is not self._encoded_nodes_ref
    or getattr(customers, "_version", None) != self._encoded_nodes_version
    or (mask.bool() & ~self._encoded_cust_mask.bool()).any()
)
```
- A cold start performs a full batch encode and saves the snapshot.
- A warm step computes `dirty_rows = mask.ne(self._encoded_cust_mask).any(dim=1)`.
- If `not dirty_rows.any()`: increments `skipped_calls` and returns immediately without touching GPU execution graphs.
- If `dirty_rows.all()`: calls `_encode_customers_full`.
- If partial ($0 < K < B$): delegates to `_encode_customers_selective`.

### 2.3. Mutual Exclusivity Constraints
- **Adaptive Depth:** Raising `ValueError` if `selective_customer_encoding=True` and `adaptive_depth=True` because dynamic layer counts based on batch-subset means would distort layer depth.
- **Activation Checkpointing:** Mutual exclusivity enforced in both `COAST2.__init__` and `COAST2._encode_customers`. Because VRAM with selective encoding remains $\le 8.5$ GiB, activation checkpointing is unneeded and disabling it removes all backward recomputation passes.

### 2.4. Zero-Impact on Simulator / Manifest Hash
No files in `problems/` were modified. The manifest `SHA-256` hashes covering all 1,547,240 instances in `data/coast_ablation/manifest.json` remain 100% valid.

---

## 3. Integration Across Codebase

| Component | File | Role |
|---|---|---|
| **Base Model** | `MODEL/model/vectra.py` | Core selective encoding, `_run_encode_pipeline`, `_reset_customer_encoding`. Default `selective_customer_encoding=False`. |
| **COAST2 Architecture** | `MODEL/model/coast2.py` | Inherits selective encoding, delegates clean override, validates mutual exclusivity with checkpointing. |
| **COAST2 Rollout** | `MODEL/train_coast2.py` | Resets cache in `_rollout_training_batch`, removes redundant pre-loop call. |
| **Ablation Models** | `MODEL/model/ablation_study.py` | `build_ablation_model` forwards `selective_customer_encoding` to `F11`, `F10`, `F01`, and `CM`. |
| **Baselines** | `baselines/_base.py`, `baselines/_critic.py` | Resets customer encoding cache on rollout initialization. |
| **Runners** | `script/train_coast2.py`, `script/train_coast_ablation.py` | Adds CLI flag `--selective-customer-encoding / --no-selective-customer-encoding` (default True). |
| **Evaluators** | `script/eval_coast_ablation.py`, `script/preflight_coast_ablation.py` | Supports strict checkpoint loading with selective encoding flag. |
| **Benchmark** | `script/benchmark_selective_encoding.py` | End-to-end GPU benchmark tracking peak memory, latency, and rows encoded. |

---

## 4. Verification & Test Evidence

- **Math Equivalence (Eval mode, FP32):** `test_selective_encoding_eval_math_equivalence` proves `max |cust_repr_full - cust_repr_sel| < 1e-5` across arbitrary dynamic reveals.
- **Autograd Equivalence (Train mode, FP32, dropout=0):** `test_selective_encoding_autograd_gradient_equivalence` proves parameter gradients match across all layers within floating-point tolerance with zero autograd in-place modification errors.
- **Preflight Smoke:** `script/preflight_coast_ablation.py` passes for all 4 variants (`f11`, `f10`, `f01`, `cm`). `script/preflight_coast2.py` passes.
- **Test Suite:** 89 passed tests across `test_selective_customer_encoding.py`, `test_coast_ablation*`, and `test_coast2*`.
