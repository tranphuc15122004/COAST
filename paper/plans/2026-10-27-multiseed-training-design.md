# Multi-seed training runner design

- **Date:** 2026-10-27
- **Status:** Implemented as a configurable launcher for four primary neural methods × five seeds (20 runs); training has not been launched.
- **Purpose:** Prepare a reproducible launcher for the selected G1 neural methods. This task prepares code only; it does not launch training.

## 1. Research question and hypothesis

The current paper reports a single selected training run for most neural methods. The experiment asks whether the primary comparison among COAST, MARDAM, AM, and PolyNet remains consistent across independent training seeds.

**Hypothesis:** COAST's held-out objective remains competitive across five seeds against the three primary neural comparators. The experiment estimates seed-to-seed variation in the main comparison; it does not change methods, tune hyperparameters, or guarantee that every reported effect will remain significant.

### Variables

- **Independent variable:** training seed, with the same five seed values paired across methods.
- **Dependent variables:** validation objective used for checkpoint selection; after training, held-out mean objective per checkpoint and its variation across seeds. Lower cost is better.
- **Controlled variables:** simulator/generator source, instance-generation parameters, objective weights, validation-set size, model-specific architecture, optimizer/loss settings, epoch count, and each method's recorded update/batch schedule.

## 2. Method matrix and scope

Run five seeds `42, 123, 456, 789, 1024` for these four primary methods (20 runs total):

| Method | Trainer/profile | Epochs | Iterations × batch | Samples per epoch | Architecture |
|---|---|---:|---:|---:|---|
| COAST | `MODEL/train.py`, `vectra` | 500 | 1,000 × 512 | 512,000 | 128 dim; 2 layers, 4 heads, FF 256 |
| MARDAM | `script/train_mardam.py` (`AttentionLearner`) | 500 | 1,000 × 512 | 512,000 | 128 dim; 3 layers, 8 heads, FF 512 |
| AM | `third_party/baselines/RL4DVRPTW/am/train.py` | 500 | 2,000 × 256 | 512,000 | 128 dim; 5 layers, 8 heads, FF 256 |
| PolyNet | `third_party/baselines/RL4DVRPTW/polynet/train.py` | 500 | 1,000 × 512 | 512,000 | 128 dim; 5 layers, 8 heads, FF 256 |

The schedules above preserve the full stored configurations for the four primary methods. Architectural ablations (B0, B1, B3, B5, EdgeOff, NoOwnership, and NoLookahead) remain the existing one-seed component analyses and are outside this multi-seed run matrix. Do not use the four-method multi-seed result to claim seed robustness for the ablation effects. GP-DVRPTW is excluded because its current protocol evolves rules per benchmark instance rather than using these neural training entry points.

## 3. Shared task and seed/pool handling

All methods use the G1 DVRPTW n50m3 task: 50 customers, 3 vehicles, capacity 200, speed 1, horizon 480, pending penalty 2, lateness penalty 1, and the problem-generation ranges/distributions recorded in the G1 artifact inventory. The launcher checks that shared generator, simulator and loss source files match across COAST and the RL4DVRPTW package before generating pools. The two `_env_dtw.py` copies differed in the repository snapshot, so the RL4DVRPTW copy has been aligned to the current COAST implementation.

For each seed, the launcher generates one raw 512,000-instance training pool followed by one raw 10,240-instance validation pool, then passes those exact files to all four trainers. Each trainer loads the pool through its local dataset class and applies its existing normalization. The launcher records file SHA-256 values in the pool manifest, global run manifest and each run's `job.json`; trainers check loaded sample counts against the configured update/batch schedule. Pool identity is independent of downstream trainer RNG call order.

Seed initialization will cover Python, NumPy, PyTorch CPU, and CUDA RNGs. The existing `set_random_seed` helper will be used with deterministic-algorithm mode disabled so this does not silently change each trainer's existing cuDNN/runtime profile. The subprocess also receives `PYTHONHASHSEED` and `--rng-seed` explicitly.

Common optimization settings are critic baseline, Adam, policy learning rate `1e-4`, critic learning rate `1e-3`, `adv_norm=true`, entropy coefficient `0.01`, max gradient norm `2.0`, and AMP enabled, matching the saved G1 configurations. The bundled RL4DVRPTW `mdam/train.py` is not used: its model initialization is an unimplemented `learner = None` placeholder. The selected MARDAM checkpoint's state dict matches the repository's `AttentionLearner` (728,582 parameters), so the launcher uses `script/train_mardam.py` and passes the shared loss options explicitly.

## 4. Runner behavior and artifacts

The root-level Python runner, `script/train_multiseed.py`, provides:

- defaults for the four methods and five seeds, plus CLI selection of methods, seeds, Python executable, and visible GPU;
- configurable per-GPU concurrency (sequential by default), round-robin GPU assignment, and per-method or per-run GPU routing;
- one shared raw training and validation pool per seed;
- a `--dry-run` mode that prints all resolved commands and output paths without starting training;
- separate fresh output paths under `output/multiseed_g1/<method>/seed<seed>`;
- preflight protection against overwriting an existing run or silently reusing old checkpoints;
- a run manifest with method, seed, command, output path, GPU id, timestamps, exit status, source hashes and pool fingerprints;
- fail-fast behavior on nonzero trainer exit or inconsistent pool fingerprints.

Each trainer continues to own checkpoint selection and its existing training log format. The launcher does not automatically evaluate on held-out test data or calculate paper tables. After training, evaluate each selected checkpoint on the same fixed held-out test sets and report the mean and standard deviation across five seed-level results. Keep per-instance paired comparisons separate from seed-level uncertainty.

## 5. Alternatives considered

1. **Extend `script/train_all_seeds.sh`.** This is the smallest change, but it currently covers only six internal profiles and does not unify the external RL4DVRPTW trainers or their method-specific configurations.
2. **Keep separate launch scripts per repository.** This follows existing repository boundaries, but duplicates seed/output bookkeeping and makes it easier for command-line settings to drift.
3. **Use one Python runner with explicit method adapters (chosen).** It preserves the existing trainers, expresses method-specific schedules in one place, and can check outputs and pool fingerprints consistently. It is a modest amount of orchestration code but avoids a second training implementation.

## 6. Validation scope

This task prepares launch code; no training, inference, or model-quality claim is part of the implementation. Static validation covers syntax, argument parsing, source-parity preflight, and dry-run output. Runtime training remains the user's experiment and is not started by the launcher implementation.

## 7. Review criteria

```yaml
review_criteria:
  metrics:
    - name: planned_method_seed_jobs
      direction: "=="
      threshold: 20
    - name: training_pool_hashes_per_seed
      direction: "=="
      threshold: 1 unique fingerprint across all four methods
    - name: validation_pool_hashes_per_seed
      direction: "=="
      threshold: 1 unique fingerprint across all four methods
  performance: []
  observability:
    - Each job reports its method, seed, command, output path, and final status.
    - Each run config records the train and validation pool fingerprints.
  stability:
    - Existing output is never overwritten implicitly.
    - A failed trainer or pool mismatch is surfaced as a failed run.
  custom:
    - No training starts from the preparation/validation commands.
    - Results are aggregated at the training-seed level, not by treating seed × instance as independent replicates.
```

## 8. Implementation boundary

The launcher and pool-loading hooks are implemented. It does not retrain prior checkpoints, modify B1/B3 results, change GP's per-instance evolution, evaluate held-out test sets, or start training as part of code preparation.
