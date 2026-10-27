# Multi-seed training runner design

- **Date:** 2026-10-27
- **Status:** High-level scope approved by the author; awaiting review of this written design.
- **Purpose:** Prepare a reproducible launcher for the selected G1 neural methods. This task prepares code only; it does not launch training.

## 1. Research question and hypothesis

The current paper reports a single selected training run for most neural methods. The experiment asks whether COAST's main comparison and its most direct architectural controls remain consistent across independent training seeds.

**Hypothesis:** COAST's held-out objective remains competitive across five seeds, and the direct removal/fusion controls continue to show the component effects discussed in the paper. The experiment estimates seed-to-seed variation; it does not change the method, tune hyperparameters, or guarantee that every reported effect will remain significant.

### Variables

- **Independent variable:** training seed, with the same five seed values paired across methods.
- **Dependent variables:** validation objective used for checkpoint selection; after training, held-out mean objective per checkpoint and its variation across seeds. Lower cost is better.
- **Controlled variables:** simulator/generator source, instance-generation parameters, objective weights, validation-set size, model-specific architecture, optimizer/loss settings, epoch count, and each method's recorded update/batch schedule.

## 2. Method matrix and scope

Run five seeds `42, 123, 456, 789, 1024` for these nine methods (45 runs total):

| Method | Trainer/profile | Epochs | Iterations × batch | Samples per epoch | Architecture |
|---|---|---:|---:|---:|---|
| COAST | `MODEL/train.py`, `vectra` | 500 | 1,000 × 512 | 512,000 | 128 dim; 2 layers, 4 heads, FF 256 |
| MARDAM | `third_party/baselines/RL4DVRPTW/mdam/train.py` | 500 | 1,000 × 512 | 512,000 | 128 dim; 3 layers, 8 heads, FF 512 |
| AM | `third_party/baselines/RL4DVRPTW/am/train.py` | 500 | 2,000 × 256 | 512,000 | 128 dim; 5 layers, 8 heads, FF 256 |
| PolyNet | `third_party/baselines/RL4DVRPTW/polynet/train.py` | 500 | 1,000 × 512 | 512,000 | 128 dim; 5 layers, 8 heads, FF 256 |
| B0 | `MODEL/train.py`, `b0` | 500 | 1,000 × 512 | 512,000 | 128 dim; 2 layers, 4 heads, FF 256 |
| NoOwnership | `MODEL/train.py`, `no_ownership` | 500 | 1,000 × 512 | 512,000 | 128 dim; 2 layers, 4 heads, FF 256 |
| NoLookahead | `MODEL/train.py`, `no_lookahead` | 500 | 1,000 × 512 | 512,000 | 128 dim; 2 layers, 4 heads, FF 256 |
| EdgeOff | `MODEL/train.py`, `edgeoff` | 500 | 1,000 × 512 | 512,000 | 128 dim; 2 layers, 4 heads, FF 256 |
| B5 | `MODEL/train.py`, `b5` | 500 | 2,000 × 256 | 512,000 | 128 dim; 2 layers, 4 heads, FF 256 |

The schedules above preserve the full stored configurations where available. The currently stored NoOwnership and NoLookahead configurations describe resumed phases (`20 × 2,048`) rather than a complete from-scratch schedule. New runs for these two profiles therefore use the standard internal schedule (`1,000 × 512`) so every run sees 512,000 training instances per epoch.

B1 and B3 remain one-seed structural controls for this experiment. The manuscript describes them as changing multiple connections at once; they are less direct tests of individual components than NoOwnership, NoLookahead, EdgeOff, and B5. The multi-seed result must not be used to claim seed robustness for B1/B3. GP-DVRPTW is excluded because its current protocol evolves rules per benchmark instance rather than using these neural training entry points.

## 3. Shared task and seed/pool handling

All methods use the G1 DVRPTW n50m3 task: 50 customers, 3 vehicles, capacity 200, speed 1, horizon 480, pending penalty 2, lateness penalty 1, and the problem-generation ranges/distributions recorded in the G1 artifact inventory. The RL baseline copies use generator/simulator files byte-identical to the current COAST files.

Each run receives its seed before data generation. The trainers generate the training pool first and the 10,240-instance validation pool second. For a given seed, the same generator source, arguments, and call order should produce identical pool tensors across the nine methods. The implementation will make this check explicit by computing SHA-256 fingerprints of normalized training and validation tensors and recording them in each run's configuration/manifest. The runner will flag a seed/method pool mismatch instead of silently treating it as a matched run.

Seed initialization will cover Python, NumPy, PyTorch CPU, and CUDA RNGs. The existing `set_random_seed` helper will be used with deterministic-algorithm mode disabled so this does not silently change each trainer's existing cuDNN/runtime profile. The subprocess also receives `PYTHONHASHSEED` and `--rng-seed` explicitly.

Common optimization settings are critic baseline, Adam, policy learning rate `1e-4`, critic learning rate `1e-3`, `adv_norm=true`, entropy coefficient `0.01`, max gradient norm `2.0`, and AMP enabled, matching the saved G1 configurations. The MARDAM run uses the bundled RL4DVRPTW trainer, whose call site passes the loss options explicitly; the older root-level MARDAM entry point is not used.

## 4. Runner behavior and artifacts

Implement one root-level Python runner, `script/train_multiseed.py`, with:

- defaults for the nine methods and five seeds, plus CLI selection of methods, seeds, Python executable, and visible GPU;
- sequential execution by default to avoid simultaneous GPU-memory contention;
- a `--dry-run` mode that prints all resolved commands and output paths without starting training;
- separate fresh output paths under `output/multiseed_g1/<method>/seed<seed>`;
- preflight protection against overwriting an existing run or silently reusing old checkpoints;
- a run manifest with method, seed, command, output path, timestamps, exit status, and pool fingerprints;
- fail-fast behavior on nonzero trainer exit or inconsistent pool fingerprints.

Each trainer continues to own checkpoint selection and its existing training log format. The launcher does not automatically evaluate on held-out test data or calculate paper tables. After training, evaluate each selected checkpoint on the same fixed held-out test sets and report the mean and standard deviation across five seed-level results. Keep per-instance paired comparisons separate from seed-level uncertainty.

## 5. Alternatives considered

1. **Extend `script/train_all_seeds.sh`.** This is the smallest change, but it currently covers only six internal profiles and does not unify the external RL4DVRPTW trainers or their method-specific configurations.
2. **Keep separate launch scripts per repository.** This follows existing repository boundaries, but duplicates seed/output bookkeeping and makes it easier for command-line settings to drift.
3. **Use one Python runner with explicit method adapters (chosen).** It preserves the existing trainers, expresses method-specific schedules in one place, and can check outputs and pool fingerprints consistently. It is a modest amount of orchestration code but avoids a second training implementation.

## 6. Validation scope

This task prepares launch code; no training, inference, or model-quality claim is part of the implementation. Apply static validation to the runner and its command matrix, including argument parsing, path preflight, and dry-run output. Skip runtime training validation because a full run is the experiment itself and is not authorized by this task. The user will launch the selected jobs after reviewing the code.

## 7. Review criteria

```yaml
review_criteria:
  metrics:
    - name: planned_method_seed_jobs
      direction: "=="
      threshold: 45
    - name: training_pool_hashes_per_seed
      direction: "=="
      threshold: 1 unique fingerprint across all nine methods
    - name: validation_pool_hashes_per_seed
      direction: "=="
      threshold: 1 unique fingerprint across all nine methods
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

After this document is reviewed, implement the runner and minimal seed/fingerprint hooks in the existing trainers/utilities. Preserve the trainer algorithms and the approved method schedules. Do not retrain prior checkpoints, modify B1/B3 results, change GP's per-instance evolution, or start any training job as part of the code-preparation task.
