"""Versioned, strict checkpoint I/O for the isolated COAST2 model."""
from __future__ import annotations

import os
from pathlib import Path

import torch

from MODEL.model.coast2 import COAST2, materialize_coast2_model


COAST2_CHECKPOINT_FORMAT = "coast2_postdecision_v1"


def save_coast2_checkpoint(model: COAST2, path, *, metadata=None, **state):
    if not isinstance(model, COAST2):
        raise TypeError("save_coast2_checkpoint expects a COAST2 model")
    materialize_coast2_model(model, preserve_rng=True)
    reserved = {"format", "model_config", "model", "metadata"}
    overlap = reserved.intersection(state)
    if overlap:
        raise ValueError("reserved checkpoint fields cannot be overridden: {}".format(sorted(overlap)))
    payload = {
        "format": COAST2_CHECKPOINT_FORMAT,
        "model_config": dict(model.coast2_config),
        "model": model.state_dict(),
        "metadata": dict(metadata or {}),
        **state,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return payload


def read_coast2_checkpoint(path, map_location="cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("COAST2 checkpoint does not exist: {}".format(path))
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != COAST2_CHECKPOINT_FORMAT:
        raise ValueError("Checkpoint is not a supported COAST2 version")
    for key in ("model_config", "model"):
        if key not in payload:
            raise ValueError("COAST2 checkpoint is missing {!r}".format(key))
    return payload


def load_coast2_model(path, device="cpu") -> COAST2:
    payload = read_coast2_checkpoint(path, map_location="cpu")
    saved_config = payload["model_config"]
    if not isinstance(saved_config, dict):
        raise ValueError("COAST2 model_config must be a mapping")
    config = dict(saved_config)
    # Early COAST2 used 480 by default; do not reinterpret older checkpoints.
    config.setdefault("horizon_scale", 480.0)
    model = COAST2(**config)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(torch.device(device))
