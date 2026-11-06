import torch
import os
import os.path
from pathlib import Path
import random
import numpy as np


def checkpoint_args_json_path(model_weight):
    """Return the sibling training config path for a checkpoint, if present."""
    if not model_weight:
        return None
    candidate = Path(model_weight).expanduser().resolve().parent / "args.json"
    return candidate if candidate.is_file() else None


def atomic_torch_save(payload, path):
    """Write a complete checkpoint before replacing the previous file."""
    output_path = Path(path)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, output_path)
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise

def capture_training_state(scaler=None, completed_epoch=None, global_step=None):
    """Capture state needed to resume exactly at the next epoch boundary."""
    return {
        "epoch_complete": completed_epoch is not None,
        "completed_epoch": None if completed_epoch is None else int(completed_epoch),
        "global_step": None if global_step is None else int(global_step),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        "scaler": None if scaler is None else scaler.state_dict(),
    }


def restore_training_state(state, scaler=None):
    if not state or not state.get("epoch_complete", False):
        raise ValueError("Study checkpoints can resume only from a completed epoch")
    if state.get("completed_epoch") is None:
        raise ValueError("Completed epoch checkpoint is missing its epoch index")
    required = ("python_rng", "numpy_rng", "torch_cpu_rng", "scaler", "global_step")
    missing = [key for key in required if key not in state or state[key] is None]
    if missing:
        raise ValueError("Training state is missing exact-resume fields: {}".format(", ".join(missing)))

    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_cpu_rng"])
    cuda_rng = state.get("torch_cuda_rng")
    if cuda_rng is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_rng) != torch.cuda.device_count():
            raise RuntimeError("CUDA device count differs from the exact-resume checkpoint")
        torch.cuda.set_rng_state_all(cuda_rng)
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    return int(state["completed_epoch"])


def save_checkpoint(args, ep, learner, optim, baseline = None, lr_sched = None, training_state = None):
    checkpoint = {
            "ep": ep,
            "model": learner.state_dict(),
            "optim": optim.state_dict()
            }
    if getattr(args, "study_variant", None) is not None:
        checkpoint["study_variant"] = args.study_variant
    if training_state is not None:
        checkpoint["training_state"] = training_state
    if args.rate_decay is not None:
        checkpoint["lr_sched"] = lr_sched.state_dict()
    if args.baseline_type == "critic":
        checkpoint["critic"] = baseline.state_dict()
    atomic_torch_save(checkpoint, os.path.join(args.output_dir, "chkpt_ep{}.pyth".format(ep+1)))
    if getattr(args, "study_variant", None) is not None:
        _prune_epoch_checkpoints(args.output_dir, getattr(args, "keep_checkpoints", KEEP_LAST))


def _prune_epoch_checkpoints(output_dir, keep):
    if keep is None:
        return
    keep = max(1, int(keep))

    def epoch_number(name):
        try:
            return int(name[len("chkpt_ep"):-len(".pyth")])
        except (TypeError, ValueError):
            return -1

    files = [
        name for name in os.listdir(output_dir)
        if name.startswith("chkpt_ep") and name.endswith(".pyth")
    ]
    for old in sorted(files, key=epoch_number, reverse=True)[keep:]:
        try:
            os.remove(os.path.join(output_dir, old))
        except OSError:
            pass

KEEP_LAST = 5

def save_checkpoint_in_train(args, ep, learner, optim, baseline = None, lr_sched = None, training_state = None):
    checkpoint = {
            "ep": ep,
            "model": learner.state_dict(),
            "optim": optim.state_dict()
            }
    if getattr(args, "study_variant", None) is not None:
        checkpoint["study_variant"] = args.study_variant
    if training_state is not None:
        checkpoint["training_state"] = training_state
    if args.rate_decay is not None:
        checkpoint["lr_sched"] = lr_sched.state_dict()
    if args.baseline_type == "critic":
        checkpoint["critic"] = baseline.state_dict()
    out_name = "chkpt_ep{}.pyth".format(ep+1)
    out_path = os.path.join(args.output_dir, out_name)
    atomic_torch_save(checkpoint, out_path)

    # keep only the latest KEEP_LAST checkpoints (or args.keep_checkpoints if provided)
    try:
        keep = getattr(args, 'keep_checkpoints', KEEP_LAST)
        if keep is None:
            keep = KEEP_LAST
        files = [f for f in os.listdir(args.output_dir) if f.startswith("chkpt_ep") and f.endswith('.pyth')]

        def epoch_from_name(name):
            try:
                s = name[len("chkpt_ep"):-len(".pyth")]
                return int(s)
            except Exception:
                return -1

        files_sorted = sorted(files, key=epoch_from_name, reverse=True)
        for old in files_sorted[int(keep):]:
            p = os.path.join(args.output_dir, old)
            try:
                os.remove(p)
            except Exception:
                pass
    except Exception:
        # do not fail saving checkpoint because of cleanup errors
        pass

def load_checkpoint(args, learner, optim, baseline = None, lr_sched = None, scaler = None):
    checkpoint = torch.load(args.resume_state, map_location='cpu', weights_only=False)

    study_variant = getattr(args, "study_variant", None)
    if study_variant is not None and checkpoint.get("study_variant") != study_variant:
        raise ValueError("Resume checkpoint study variant does not match the requested variant")

    if study_variant is not None:
        learner.load_state_dict(checkpoint["model"], strict=True)
    else:
        # Historical non-study runs retain their permissive checkpoint behavior.
        try:
            learner.load_state_dict(checkpoint["model"], strict=True)
        except RuntimeError as e:
            print(f"[load_checkpoint] strict load failed ({e}); retrying with strict=False")
            result = learner.load_state_dict(checkpoint["model"], strict=False)
            if result.missing_keys:
                print(f"[load_checkpoint] Missing keys (will use random init): {result.missing_keys}")
            if result.unexpected_keys:
                print(f"[load_checkpoint] Unexpected keys (ignored): {result.unexpected_keys}")

    optim.load_state_dict(checkpoint["optim"])
    if args.rate_decay is not None and "lr_sched" in checkpoint:
        lr_sched.load_state_dict(checkpoint["lr_sched"])
    if args.baseline_type == "critic" and baseline is not None and "critic" in checkpoint:
        baseline.load_state_dict(checkpoint["critic"])
    if study_variant is not None:
        state = checkpoint.get("training_state")
        completed_epoch = restore_training_state(state, scaler)
        if completed_epoch != int(checkpoint["ep"]) + 1:
            raise ValueError("Checkpoint epoch and captured training state disagree")
        if int(state["global_step"]) != completed_epoch * int(args.iter_count):
            raise ValueError("Checkpoint global step does not match the locked update schedule")
        return completed_epoch
    return int(checkpoint["ep"]) + 1

import torch

def load_model_weights(args, learner, strict=False):
    """
    Load only model weights (policy network), in-place.

    Used for:
    - inference
    - fine-tune
    - eval
    - warm-start training (optimizer reset)

    Args:
        args.model_weight: path to checkpoint or state_dict
        learner: nn.Module (updated in-place)
        strict: whether to require exact key match (default False — safe for
            cross-version loads with minor key differences)
    """
    path = getattr(args, "model_weight", None)
    if not path:
        print("[load_model_weights] No --model-weight provided; skip loading.")
        return

    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"[load_model_weights] ERROR loading '{path}': {e}")
        return

    # Determine checkpoint format
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
        source = "checkpoint['model']"
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        source = "state_dict"
    else:
        print(f"[load_model_weights] Unexpected checkpoint format: {type(checkpoint)}")
        return

    # Load weights (in-place)
    try:
        res = learner.load_state_dict(state_dict, strict=strict)
    except Exception as e:
        print(f"[load_model_weights] ERROR applying state_dict (strict={strict}): {e}")
        return

    # Normalize missing / unexpected keys across PyTorch versions
    if isinstance(res, (tuple, list)):
        missing, unexpected = res
    else:
        missing = res.missing_keys
        unexpected = res.unexpected_keys

    provided = len(state_dict)
    model_keys = len(learner.state_dict())
    matched = sum(1 for k in state_dict if k in learner.state_dict())

    print(
        f"[load_model_weights] Loaded from '{path}' ({source}) | "
        f"provided={provided}, matched={matched}, model_keys={model_keys}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}, strict={strict}"
    )

    if missing:
        print(f"[load_model_weights] Sample missing keys ({len(missing)}): {missing[:10]}")
    if unexpected:
        print(f"[load_model_weights] Sample unexpected keys ({len(unexpected)}): {unexpected[:10]}")
