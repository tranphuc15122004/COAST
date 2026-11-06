#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python3}
DATA_DIR=${DATA_DIR:-"$ROOT_DIR/data/coast_ablation"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$ROOT_DIR/output/coast_ablation"}

cd "$ROOT_DIR"
for variant in f11 f10 f01 cm; do
  for seed in 42 123 456; do
    PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" script/train_coast_ablation.py \
      --variant "$variant" \
      --seed "$seed" \
      --data-dir "$DATA_DIR" \
      --output-root "$OUTPUT_ROOT" \
      --device cuda
  done
done
