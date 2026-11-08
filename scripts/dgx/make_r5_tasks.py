"""Write the R5 task list (one python run per line), in benchmark/r5_design.yaml's design.
Most informative lines first, so a session cut short still answers the main question.

    python scripts/dgx/make_r5_tasks.py --kaggle > tasks_r5.txt
"""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle", action="store_true", help="configs/<name>_kaggle.yaml, runs_kaggle/ output")
    ap.add_argument("--only", default="", help="comma-separated datasets (default: all in the design)")
    args = ap.parse_args()
    bench = yaml.safe_load(open(ROOT / "benchmark" / "tasks.yaml"))["datasets"]
    models = yaml.safe_load(open(ROOT / "benchmark" / "models.yaml"))["models"]
    recipes = yaml.safe_load(open(ROOT / "benchmark" / "recipes.yaml"))["recipes"]
    design = yaml.safe_load(open(ROOT / "benchmark" / "r5_design.yaml"))
    only = set(args.only.split(",")) if args.only else None
    root = "runs_kaggle" if args.kaggle else "runs_dgx"

    def config(name):
        return f"configs/{name}_kaggle.yaml" if args.kaggle else bench[name]["dgx_config"]

    def train_line(name, key, seed, extra):
        r = recipes[name][key]
        return (f"scripts/train_continual.py --bench {name} --data {config(name)} --train configs/r1_base.yaml "
                f"--method naive --arch {models[key]} --lr-body {r['lr_body']:g} "
                f"--steps-per-task {r['steps_per_task']} --seed {seed} {extra} --no-save --out {root}/r5 "
                f"--tag {key}_lr{r['lr_body']:g}_s{r['steps_per_task']}")

    def diag_line(name, key, arms):
        return (f"scripts/run_cms_diagnostics.py --dataset {name} --model {key} --seed {design['diag_seed']} "
                f"--arms {','.join(arms)} --data {config(name)} --out {root}/diag --delete-weights")

    lines = []
    if only is None or "earvn" in only:
        seeds = [s for s in bench["earvn"]["seeds"] if s != 0]
        # experiment 1: full-size slow updates (seed 1 first in every cell)
        for seed in seeds:
            for key in design["cells_exp1"]:
                for v in design["variants_exp1"].values():
                    lines.append(train_line("earvn", key, seed,
                                            f"--body-cms-periods 1,16,256 --body-cms-lr-pow {v['lr_pow']:g} "
                                            f"--body-cms-slow {v['slow']}"))
        # experiment 2: diagnostics of the new arms and R4b
        d = design["diag_exp2"]
        lines.append(diag_line(d["dataset"], d["model"], d["arms"]))
        # experiment 3: perfect-memory upper bound
        for seed in seeds:
            lines.append(train_line("earvn", design["cell_exp3"], seed,
                                    f"--snapshot-after-task {design['snapshot_after_task']}"))
    # experiment 5: diagnostics on more cells
    for d in design["diag_exp5"]:
        if only is None or d["dataset"] in only:
            lines.append(diag_line(d["dataset"], d["model"], d["arms"]))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
