import random
from argparse import Namespace

import numpy as np
import pytest
import torch

from utils._chkpt import capture_training_state, load_checkpoint, restore_training_state, save_checkpoint


def test_resume_restores_random_streams_and_starts_after_completed_epoch(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    args = Namespace(
        output_dir=str(tmp_path),
        resume_state=str(tmp_path / "chkpt_ep1.pyth"),
        rate_decay=None,
        baseline_type="none",
        study_variant="f11",
        iter_count=1,
    )

    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)
    state = capture_training_state(scaler, completed_epoch=1, global_step=1)
    expected = (random.random(), float(np.random.rand()), torch.rand(3))
    save_checkpoint(args, 0, model, optimizer, training_state=state)

    fresh_model = torch.nn.Linear(3, 2)
    fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    start_epoch = load_checkpoint(args, fresh_model, fresh_optimizer, scaler=scaler)

    observed = (random.random(), float(np.random.rand()), torch.rand(3))
    assert start_epoch == 1
    assert torch.equal(fresh_model.weight, model.weight)
    assert observed[0] == expected[0]
    assert observed[1] == expected[1]
    assert torch.equal(observed[2], expected[2])


def test_resume_refuses_a_checkpoint_from_a_different_study_variant(tmp_path):
    source_model = torch.nn.Linear(3, 2)
    source_optimizer = torch.optim.Adam(source_model.parameters(), lr=1e-3)
    source_args = Namespace(
        output_dir=str(tmp_path),
        rate_decay=None,
        baseline_type="none",
        study_variant="f11",
        iter_count=1,
    )
    state = capture_training_state(
        torch.amp.GradScaler("cuda", enabled=False), completed_epoch=1, global_step=1
    )
    save_checkpoint(source_args, 0, source_model, source_optimizer, training_state=state)

    target_args = Namespace(
        output_dir=str(tmp_path),
        resume_state=str(tmp_path / "chkpt_ep1.pyth"),
        rate_decay=None,
        baseline_type="none",
        study_variant="f10",
    )
    target_model = torch.nn.Linear(3, 2)
    target_optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="variant"):
        load_checkpoint(target_args, target_model, target_optimizer)


def test_restore_refuses_incomplete_epoch_state():
    with pytest.raises(ValueError, match="completed epoch"):
        restore_training_state({"epoch_complete": False})


def test_restore_refuses_state_without_global_step():
    with pytest.raises(ValueError, match="global_step"):
        restore_training_state({
            "epoch_complete": True,
            "completed_epoch": 1,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_cpu_rng": torch.get_rng_state(),
            "torch_cuda_rng": None,
            "scaler": {},
        })


def test_epoch_checkpoints_keep_only_the_five_latest_resume_points(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    args = Namespace(
        output_dir=str(tmp_path), rate_decay=None, baseline_type="none", study_variant="f11"
    )
    for epoch in range(7):
        state = capture_training_state(scaler, completed_epoch=epoch + 1, global_step=epoch + 1)
        save_checkpoint(args, epoch, model, optimizer, training_state=state)
    torch.save({"model": model.state_dict()}, tmp_path / "chkpt_best.pyth")

    saved = sorted(path.name for path in tmp_path.glob("chkpt_ep*.pyth"))
    assert saved == ["chkpt_ep3.pyth", "chkpt_ep4.pyth", "chkpt_ep5.pyth", "chkpt_ep6.pyth", "chkpt_ep7.pyth"]
    assert (tmp_path / "chkpt_best.pyth").is_file()


def test_failed_checkpoint_write_preserves_previous_completed_checkpoint(tmp_path, monkeypatch):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    args = Namespace(output_dir=str(tmp_path), rate_decay=None, baseline_type="none", study_variant="f11")
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    state = capture_training_state(scaler, completed_epoch=1, global_step=1)
    save_checkpoint(args, 0, model, optimizer, training_state=state)
    checkpoint_path = tmp_path / "chkpt_ep1.pyth"
    original = checkpoint_path.read_bytes()

    def fail_save(_obj, path, **_kwargs):
        with open(path, "wb") as stream:
            stream.write(b"partial checkpoint")
        raise OSError("simulated interrupted checkpoint write")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="interrupted"):
        save_checkpoint(args, 0, model, optimizer, training_state=state)

    assert checkpoint_path.read_bytes() == original
    assert not (tmp_path / "chkpt_ep1.pyth.tmp").exists()
