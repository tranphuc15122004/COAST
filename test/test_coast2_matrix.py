import json
from pathlib import Path

from script.run_coast2_matrix import build_commands


def test_registered_matrix_pairs_coast2_with_four_existing_ablation_controls():
    commands = build_commands(
        data_dir="data/shared", output_root="output/study", device="cuda", seeds=(42, 123)
    )

    assert len(commands) == 10
    assert [command[command.index("--variant") + 1] for command in commands if "--variant" in command] == [
        "f11", "f10", "f01", "cm", "f11", "f10", "f01", "cm"
    ]
    coast2 = [command for command in commands if "script/train_coast2.py" in command]
    assert [command[command.index("--seed") + 1] for command in coast2] == ["42", "123"]
    assert all(command[command.index("--protocol") + 1] == "full" for command in coast2)


def test_matrix_manifest_locks_shared_pools_seeds_and_primary_contrast():
    path = Path("script/coast2_matrix.json")
    spec = json.loads(path.read_text(encoding="utf-8"))

    assert spec["seeds"] == [42, 123, 456]
    assert spec["shared_data"]["same_train_and_validation_pools_across_variants"] is True
    assert spec["primary_contrast"] == "COAST2 vs f11, paired by seed and shared validation/test instances"
    assert spec["full_runs"] == 15
