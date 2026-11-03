import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "dgx"))
from make_r4_tasks import periods_of  # noqa: E402


def test_periods_follow_the_task_length():
    assert periods_of(60, 2) == [1, 60, 120]
    assert periods_of(500, 2) == [1, 500, 1000]


def test_task_list_covers_each_design_cell_on_its_reported_seeds():
    design = yaml.safe_load(open(ROOT / "benchmark" / "r4_design.yaml"))
    bench = yaml.safe_load(open(ROOT / "benchmark" / "tasks.yaml"))["datasets"]
    recipes = yaml.safe_load(open(ROOT / "benchmark" / "recipes.yaml"))["recipes"]
    lines = subprocess.run([sys.executable, "scripts/dgx/make_r4_tasks.py", "--kaggle"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.splitlines()
    expected = sum(len(keys) * (len(bench[n]["seeds"]) - 1) for n, keys in design["cells"].items())
    assert len(lines) == expected == len(set(lines))
    assert all("--method naive" in ln and "--seed 0 " not in ln for ln in lines)
    s = recipes["earvn"]["modern_cnn"]["steps_per_task"]
    assert any(f"--bench earvn " in ln and f"--body-cms-periods 1,{s},{2 * s} " in ln for ln in lines)
