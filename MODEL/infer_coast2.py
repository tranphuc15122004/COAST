"""Small inference API for COAST2 checkpoints, independent of legacy infer.py."""
from __future__ import annotations

from pathlib import Path

from MODEL.coast2_checkpoint import load_coast2_model
from MODEL.train_coast2 import evaluate_costs
from problems import DVRPTW_Dataset


def infer_dataset(checkpoint, data_file, device="cpu", batch_size=512,
                  pending_cost=2.0, late_cost=1.0):
    checkpoint = Path(checkpoint)
    data_file = Path(data_file)
    if not checkpoint.is_file():
        raise FileNotFoundError("COAST2 checkpoint does not exist: {}".format(checkpoint))
    if not data_file.is_file():
        raise FileNotFoundError("COAST2 evaluation data does not exist: {}".format(data_file))
    model = load_coast2_model(checkpoint, device)
    dataset = DVRPTW_Dataset.load(data_file)
    if len(dataset) <= 0:
        raise ValueError("COAST2 inference dataset is empty")
    dataset.normalize()
    return evaluate_costs(
        model, dataset, device=device, batch_size=batch_size,
        pending_cost=pending_cost, late_cost=late_cost,
    )
