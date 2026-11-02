from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from script import train_multiseed


def test_default_matrix_contains_four_methods_and_five_seeds(tmp_path: Path) -> None:
    config = train_multiseed.validate_config(
        {
            "methods": list(train_multiseed.DEFAULT_METHODS),
            "seeds": list(train_multiseed.DEFAULT_SEEDS),
            "gpu_ids": ["0"],
            "parallel_per_gpu": 1,
            "job_gpu_map": {},
            "output_root": str(tmp_path / "matrix"),
            "python": sys.executable,
        }
    )

    jobs = train_multiseed.assign_jobs(config)

    assert len(jobs) == 20
    assert config["expected_gpu_name"] == "B200"
    assert {(job["method"], job["seed"]) for job in jobs} == {
        (method, seed)
        for method in train_multiseed.DEFAULT_METHODS
        for seed in train_multiseed.DEFAULT_SEEDS
    }


def test_b200_coast_config_selects_only_coast_for_the_five_seeds() -> None:
    config_path = train_multiseed.REPO_ROOT / "script" / "train_coast_b200.json"
    config = train_multiseed.validate_config(json.loads(config_path.read_text(encoding="utf-8")))

    jobs = train_multiseed.assign_jobs(config)

    assert config["methods"] == ["coast"]
    assert config["seeds"] == list(train_multiseed.DEFAULT_SEEDS)
    assert len(jobs) == 5
    assert all(job["method"] == "coast" for job in jobs)
    assert config["output_root"].name == "coast_multiseed_b200"


def test_coast_run_explicitly_selects_full_vectra_profile(tmp_path: Path) -> None:
    config = train_multiseed.validate_config(
        {
            "methods": ["coast"],
            "seeds": [42],
            "gpu_ids": ["0"],
            "parallel_per_gpu": 1,
            "job_gpu_map": {},
            "output_root": str(tmp_path / "matrix"),
            "python": sys.executable,
        }
    )
    job = train_multiseed.assign_jobs(config)[0]

    command = train_multiseed.build_command(config, job)

    assert command[command.index("--ablation-profile") + 1] == "vectra"


def test_b200_preflight_has_a_safe_preflight_only_mode() -> None:
    args = train_multiseed.build_parser().parse_args(["--preflight-only"])

    assert args.preflight_only is True


def test_b200_baseline_overlay_preserves_the_preinstalled_torch_stack() -> None:
    requirements_path = train_multiseed.REPO_ROOT / "requirements.b200-baselines.txt"
    requirements = {
        line.strip().lower()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "torchrl==0.14.1" in requirements
    assert "lightning==2.6.6" in requirements
    assert not any(
        line.startswith(("torch==", "torchvision==", "tensordict==", "nvidia-", "triton=="))
        for line in requirements
    )


def test_baseline_import_preflight_reports_b200_dependency_overlay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = train_multiseed.validate_config(
        {
            "methods": ["am"],
            "seeds": [42],
            "gpu_ids": ["0"],
            "parallel_per_gpu": 1,
            "job_gpu_map": {},
            "output_root": str(tmp_path / "matrix"),
            "python": sys.executable,
        }
    )
    jobs = train_multiseed.assign_jobs(config)

    def missing_torchrl(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="",
            stderr="ModuleNotFoundError: No module named 'torchrl'",
        )

    monkeypatch.setattr(train_multiseed.subprocess, "run", missing_torchrl)

    with pytest.raises(RuntimeError, match="requirements.b200-baselines.txt"):
        train_multiseed.preflight_trainer_clis(config, jobs)


def test_gpu_runtime_probe_is_valid_python() -> None:
    source = train_multiseed.build_gpu_probe_script()

    compile(source, "<gpu-runtime-probe>", "exec")


def test_b200_preflight_rejects_a_different_gpu_model() -> None:
    with pytest.raises(RuntimeError, match="B200"):
        train_multiseed.validate_gpu_device_info(
            "0",
            {
                "available": True,
                "name": "NVIDIA H100",
                "compute_capability": [9, 0],
                "cuda_context_ok": True,
            },
            expected_gpu_name="B200",
        )


def test_b200_preflight_rejects_non_blackwell_compute_capability() -> None:
    with pytest.raises(RuntimeError, match="compute capability"):
        train_multiseed.validate_gpu_device_info(
            "0",
            {
                "available": True,
                "name": "NVIDIA B200",
                "compute_capability": [9, 0],
                "cuda_context_ok": True,
            },
            expected_gpu_name="B200",
        )


def test_b200_preflight_accepts_blackwell_device() -> None:
    train_multiseed.validate_gpu_device_info(
        "0",
        {
            "available": True,
            "name": "NVIDIA B200",
            "compute_capability": [10, 0],
            "total_memory_gb": 180.0,
            "cuda_context_ok": True,
        },
        expected_gpu_name="B200",
    )


@pytest.mark.parametrize("entrypoint", ["MODEL/train.py", "script/train_mardam.py"])
def test_training_cli_does_not_require_optional_ortools(entrypoint: str) -> None:
    code = """
import importlib.abc
import runpy
import sys

class BlockOrtools(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "ortools" or fullname.startswith("ortools."):
            raise ModuleNotFoundError("blocked optional ortools import")
        return None

sys.meta_path.insert(0, BlockOrtools())
sys.argv = [sys.argv[1], "--help"]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(train_multiseed.REPO_ROOT / entrypoint)],
        cwd=train_multiseed.REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


@pytest.mark.parametrize("module_name", ["MODEL.infer", "MODEL.infer_mardam"])
def test_inference_reconstructs_architecture_from_checkpoint_sibling_args(
    module_name: str, tmp_path: Path
) -> None:
    config_path = tmp_path / "args.json"
    config_path.write_text(
        json.dumps({
            "model_size": 128,
            "layer_count": 2,
            "head_count": 4,
            "ff_size": 256,
            "edge_feat_size": 8,
            "ablation_profile": "vectra",
        }),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "chkpt_best.pyth"
    checkpoint_path.touch()
    infer_module = importlib.import_module(module_name)

    args = infer_module.parse_infer_args([
        "--model-weight", str(checkpoint_path), "--no-cuda"
    ])

    assert args.head_count == 4
    assert args.ff_size == 256
    assert args.layer_count == 2
    assert args.config_file == str(config_path)
