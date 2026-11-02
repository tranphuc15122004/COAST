# GP-DVRPTW reviewer source snapshot

This directory contains the GP-DVRPTW implementation used for the paper's GP baseline, including the Python experiment runner and simulator, the local `problems` package, scenario scripts, tests, and supplementary Rust implementation.

## Scope

- The paper's Python path is `python_src/main.py`; `python_src/tensor_main.py` is also included.
- Source, documentation, dependency/build metadata, and scripts are included.
- Datasets, generated result folders/files, `acs-km` build/runtime artifacts, caches, `.git` metadata, and the original `.env` are omitted.
- `.env.example` contains sanitized run settings reconstructed from the upstream README and code defaults; it is not a copy of the original `.env`.
- `UPSTREAM_README.md` preserves the original project README.
- `PROVENANCE.json` records the source commit and SHA-256 for every included payload file (the manifest itself is excluded). It also identifies included files that differed from the source commit.

## Use

The original Python entry point expects to be run from this directory. Install PyTorch and `python-dotenv` in the environment, place the required dataset at the expected path, and create a local `.env` from `.env.example` before running. The Rust crate is retained as supplementary implementation code; the Python runner is the path associated with the paper GP runs.

No `LICENSE` file was present in the source tree at snapshot time; this snapshot does not add or imply a license.
