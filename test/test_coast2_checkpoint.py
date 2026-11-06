import pytest
import torch

from MODEL.coast2_checkpoint import COAST2_CHECKPOINT_FORMAT, load_coast2_model, save_coast2_checkpoint
from MODEL.model.coast2 import COAST2


def _model():
    return COAST2(
        7, 4, model_size=16, layer_count=1, head_count=4, ff_size=32,
        memory_size=16, dropout=0.0,
    )


def test_coast2_checkpoint_round_trip_is_strict_and_versioned(tmp_path):
    model = _model()
    path = tmp_path / "coast2.pyth"
    save_coast2_checkpoint(model, path, metadata={"global_step": 3})

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    restored = load_coast2_model(path, "cpu")

    assert checkpoint["format"] == COAST2_CHECKPOINT_FORMAT
    assert checkpoint["metadata"]["global_step"] == 3
    assert restored.coast2_config == model.coast2_config
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def test_coast2_loader_rejects_legacy_or_unversioned_checkpoints(tmp_path):
    path = tmp_path / "legacy.pyth"
    torch.save({"model": _model().state_dict()}, path)

    with pytest.raises(ValueError, match="COAST2"):
        load_coast2_model(path, "cpu")


def test_coast2_loader_rejects_architecture_mismatch_without_partial_load(tmp_path):
    model = _model()
    path = tmp_path / "wrong_config.pyth"
    save_coast2_checkpoint(model, path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["model_config"]["model_size"] = 32
    torch.save(payload, path)

    with pytest.raises(RuntimeError):
        load_coast2_model(path, "cpu")
