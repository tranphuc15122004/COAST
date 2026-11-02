# COAST: Coordinated Online Assignment with Structured Trajectory Signals

> Anonymous AAAI submission — code supplement

This repository implements **COAST**, a reinforcement-learning policy for the
Dynamic Vehicle Routing Problem with Time Windows (DVRPTW).  The implementation
class is named `VECTRA` for historical reasons and is defined in
`MODEL/model/vectra.py`.

COAST makes one event-driven dispatch decision at a time.  It combines three
candidate-level signals: edge-aware vehicle--customer compatibility, a
fleet-relative ownership signal derived from per-vehicle memory, and a learned
candidate-conditioned lookahead signal.  All three are optimized end-to-end
with policy-gradient feedback; neither ownership nor lookahead has an auxiliary
supervised objective.

## Repository contents

| Path | Purpose |
|---|---|
| `MODEL/` | COAST/VECTRA training and inference entry points |
| `layers/` | Attention, encoder, feature, fusion, and loss layers |
| `problems/` | DVRPTW data generators and event-driven environments |
| `baselines/` | Critic, rollout, and heuristic baselines |
| `script/` | Training, batch evaluation, data generation, and analysis commands |
| `data/test_sets/` | Fixed in-distribution and OOD DVRPTW test sets |
| `datasets/` | Solomon-style, scale, and dynamic-grid benchmark instances |
| `output/` | Local experiment artefacts, including reference checkpoints in this workspace |
| `paper_results/` | Paper-facing CSV exports, tables, and figure-generation code |

## Environment

The core implementation requires Python, PyTorch, SciPy, Matplotlib, and
TQDM.  Batch reporting additionally uses NumPy, Pandas, and OpenPyXL.
OR-Tools is required only for the OR-Tools baseline; LKH is required only for
the LKH baseline.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use a PyTorch build compatible with the available CPU or CUDA runtime.  CPU
execution is supported by adding `--no-cuda` to Python entry points; it is much
slower than GPU inference and training.  The pinned PyTorch version was
validated with CUDA 12.4; replace it with the matching build for another CUDA
runtime or for CPU-only execution.

## Quick verification

Run a small forward/backward smoke test from the repository root:

```bash
PYTHONPATH=. python test/model_test.py
```

The test constructs synthetic VRP and VRPTW instances, executes the VECTRA
policy, and checks that a policy-gradient backward pass completes.

## Reproducing reference COAST inference

The following command evaluates the reference COAST checkpoint on the fixed
500-instance in-distribution test set currently included in this workspace.
The `args.json` file must always be paired with its checkpoint: it records the
architecture and ablation settings used to train that model.

```bash
PYTHONPATH=. python MODEL/infer.py \
  --problem-type dvrptw \
  --config-file output/Model_DVRPTWn50m3_260311-0727/args.json \
  --model-weight output/Model_DVRPTWn50m3_260311-0727/chkpt_best.pyth \
  --data-file data/test_sets/test_dvrptw_id_n50m3_500.pyth \
  --greedy \
  --max-print-instances 1 \
  --save-json reproduced/coast_id_n50m3.json
```

Add `--no-cuda` when CUDA is unavailable.  The JSON output contains the
selected routes, costs, and route-replay verification results.

## Reproducing the internal OOD comparison

The repository contains fixed ID/OOD instances and checkpoints for COAST,
MARDAM, and the internal ablations.  The command below deliberately excludes
external models that are not redistributed with this repository.

```bash
PYTHONPATH=. python script/run_ood_experiments.py \
  --models vectra,mardam,b0,b1,b3,b5,edgeoff,no_ownership,no_lookahead \
  --datasets-dir data/test_sets \
  --output-dir reproduced/ood_eval \
  --max-print-instances 1
```

The resulting `reproduced/ood_eval/ood_summary.csv` can be compared with the
paper-facing CSV files in `paper_results/csv_exports/`.

## Training

Train the full COAST configuration with the default n50m3 DVRPTW schedule:

```bash
bash script/train_vectra_main.sh
```

To train the paper's internal variants across the configured random seeds:

```bash
bash script/train_all_seeds.sh
```

For the current COAST-only run on the B200, use
[`script/train_coast_b200.json`](script/train_coast_b200.json). It schedules
COAST for the existing five seeds, one job at a time on GPU 0. Set `PYTHON` to
the interpreter from `fast_infer_text_sum` under the server's
`/workspace/storage-shared/` mount. The launcher inherits the active
interpreter because the config has `python: null`. COAST does not need the
AM/PolyNet dependency overlay below. From the MARDAM checkout, activate that
environment first:

```bash
source /workspace/storage-shared/<fast_infer_text_sum-path>/.venv/bin/activate
export PYTHON="$(command -v python)"
"$PYTHON" -c 'import torch; print(torch.__version__, torch.version.cuda)'
"$PYTHON" script/train_multiseed.py --config script/train_coast_b200.json --dry-run
"$PYTHON" script/train_multiseed.py --config script/train_coast_b200.json --preflight-only
"$PYTHON" script/train_multiseed.py --config script/train_coast_b200.json
```

Run the final command in `tmux` or `screen`. The launcher checks the COAST CLI,
the B200 device, and a CUDA matrix multiplication before it creates the shared
train/validation pools. The five runs write under
`output/coast_multiseed_b200/coast/seed*/` and refuse to overwrite existing
outputs.

The full four-model matrix remains available through
[`script/train_multiseed_g1.json`](script/train_multiseed_g1.json). Its AM and
PolyNet trainers need `torchrl` and `lightning`, which are not in the B200 base
environment. Install this repository's small overlay only when running those
methods; it does not pin PyTorch or CUDA:

```bash
"$PYTHON" -m pip install -r requirements.b200-baselines.txt
```

Do not install `third_party/baselines/RL4DVRPTW/requirements.txt` into the
B200 environment: it pins PyTorch `2.9.1` and CUDA 12.8 packages, which conflict
with the server's PyTorch 2.13/CUDA 13.0 stack. The root requirements for the
original COAST environment also pin a different PyTorch version (`2.6.0`).

The launcher creates one raw training and validation pool per seed, shares
those files across the four methods, and records their hashes with each run.
For multiple allocated GPUs, edit `gpu_ids`, `parallel_per_gpu`, and optionally
`job_gpu_map` in the config; keep concurrency at one per GPU unless the server
allocation and available host memory have been checked. A run-specific mapping
such as `"coast:42": "1"` takes precedence over a method-wide mapping such as
`"mardam": "0"`. Ensure `output_root` has enough storage for the shared pools
and checkpoints. The launcher stops if the shared simulator/generator/loss
sources differ, the configured GPU is not visible, or an output path already
exists. Multi-seed jobs fail on out of memory rather than silently changing the
configured batch or validation result. Choose a new `output_root` for a new
matrix.

The shell commands generate timestamped output directories; the multi-seed
runner uses the configured output root and refuses to overwrite existing runs.
Training is stochastic; the supplied `chkpt_best.pyth` files are the reference
artefacts for result reproduction, while retraining is intended to reproduce
the protocol and performance trend rather than byte-identical weights.

## Benchmarks and reference artefacts

| Evaluation setting | Included input assets |
|---|---|
| In-distribution and OOD DVRPTW | `data/test_sets/` |
| Solomon-style benchmark | `datasets/100/`, `datasets/h200/`, `datasets/h400/` |
| Scale benchmark | `datasets/dvrptw_n{20,50,100,200,400}m*_10240.pyth` |
| Dynamic sensitivity grid | `datasets/dvrptw_dynamic_grid/` |
| Tables and figures | `paper_results/` |

The reference checkpoints in the current workspace are stored under `output/`:

| Model | Checkpoint directory |
|---|---|
| COAST | `output/Model_DVRPTWn50m3_260311-0727/` |
| MARDAM | `output/Mardam_DVRPTWn50m3_260315-1328/` |
| Ablations B0, B1, B3, B5, EdgeOff | `output/ablation/<profile>/seed42/` |
| No-ownership / no-lookahead | `output/ablation/no_ownership/`, `output/ablation/no_lookahead/` |

For each directory, use only the matched `args.json` and `chkpt_best.pyth`
pair.  Other files in `output/` are intermediate training logs or checkpoints
and are not needed for inference.

## External baselines

Reviewer-facing source snapshots for RL4DVRPTW (including AM and PolyNet)
and GP-DVRPTW are bundled under [`third_party/baselines/`](third_party/baselines/README.md).
Their datasets, trained weights/checkpoints, and generated outputs are not
included; see the snapshot READMEs and `PROVENANCE.json` manifests.  The
reported summary values are retained in `paper_results/csv_exports/` for
transparency.  LKH is supported through `externals/_lkh.py` when a compatible
LKH executable is available.  OR-Tools runs require the optional `ortools`
package.

## Reproducibility notes

- Run commands from the repository root with `PYTHONPATH=.`.
- Preserve the checkpoint/configuration pairing; loading a checkpoint with CLI
  defaults can silently instantiate a different architecture or ablation.
- Fixed evaluation sets should be used for numerical comparisons.  They can be
  regenerated by `script/generate_ood_sets.py`, but the included files are the
  reference inputs.
- `paper_results/csv_exports/` records the paper-facing aggregates.  Fresh
  runs should be written to a new directory such as `reproduced/`, never over
  the reference artefacts.

## Anonymous release note

This repository is prepared as an anonymous supplementary code submission.
It intentionally contains no author names, affiliations, URLs, or contact
details.  Citation metadata and a software license will be added only in the
camera-ready release, as permitted by the review process.
