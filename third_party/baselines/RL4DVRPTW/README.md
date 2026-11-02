# RL4DVRPTW reviewer source snapshot

This directory contains the source modules and launch/evaluation scripts for the RL baselines used alongside COAST. The snapshot includes AM, PolyNet, MDAM, MVMoe, their shared simulator/utilities, and the `rl4co` source imported by these implementations. It also preserves the AM and PolyNet `args.json` configurations.

## Scope

- Source, shell scripts, package metadata, and configuration files are included.
- Model checkpoints, datasets, generated inference outputs, caches, `.git` metadata, and machine-specific editor files are not included.
- The included launcher scripts refer to those omitted checkpoints/data by their expected relative paths. Supply matching assets separately to reproduce runs.
- The files are copied from the source working tree, including source changes present at snapshot time. The manifest lists included files that differed from the source commit. `PROVENANCE.json` records the source commit and SHA-256 for every included payload file (the manifest itself is excluded).

## Use

Run commands from this directory so the original relative paths resolve. Install the pinned dependencies from `requirements.txt`; some entries may require a compatible CUDA/PyTorch environment. For example, inspect `run_paper_eval.sh` for the paper comparison entry point and override its documented paths/variables for locally supplied assets.

No `LICENSE` file was present in the source tree at snapshot time; this snapshot does not add or imply a license.
