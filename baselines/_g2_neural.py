"""Load the repository's reveal-aware AM/PolyNet adapters for G2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


def load_g2_neural_policy(method: str, config_path: Path, weight_path: Path, device):
    """Load an AM or PolyNet step policy from the local reviewer snapshot.

    The adapter receives the shared DVRPTW environment at every event and
    refreshes its customer encoding only when requests are revealed. Its
    checkpoint and training configuration must match the selected method.
    """
    method = method.strip().lower()
    if method not in {"am", "polynet"}:
        raise ValueError(f"Unsupported G2 neural comparator: {method}")

    external_root = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "baselines"
        / "RL4DVRPTW"
    )
    if not external_root.is_dir():
        raise FileNotFoundError(f"Expected reviewer baseline source at {external_root}")
    external_root_string = str(external_root)
    if external_root_string not in sys.path:
        sys.path.insert(0, external_root_string)

    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    shared = {
        "cust_feat_size": 7,
        "veh_state_size": 4,
        "model_size": int(config.get("model_size", 128)),
        "layer_count": int(config.get("layer_count", 3)),
        "head_count": int(config.get("head_count", 8)),
        "ff_size": int(config.get("ff_size", 512)),
        "tanh_xplor": float(config.get("tanh_xplor", 10)),
        "greedy": True,
    }
    if method == "am":
        from am.model import AM_DVRPTW

        learner = AM_DVRPTW(**shared)
    else:
        from polynet.model import PolyNet_DVRPTW

        poly_k = config.get("cust_k")
        if poly_k is None:
            poly_k = int(config.get("customers_count", 50)) + 1
        learner = PolyNet_DVRPTW(
            **shared,
            k=max(2, int(poly_k)),
            encoder_type=str(config.get("encoder_type", "AM")),
        )

    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format: {weight_path}")

    try:
        result = learner.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{method} checkpoint does not match its configured architecture: {weight_path}"
        ) from exc
    model_state = learner.state_dict()
    matched = [
        key for key, value in state_dict.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    ]
    if not matched:
        raise RuntimeError(
            f"No checkpoint tensors matched the {method} model at {weight_path}."
        )
    learner.to(device)
    learner.eval()
    learner._g2_checkpoint_match = {
        "matched_tensors": len(matched),
        "provided_tensors": len(state_dict),
        "missing_tensors": len(result.missing_keys),
        "unexpected_tensors": len(result.unexpected_keys),
    }
    return learner, config
