"""Write the R1 task lists (one python run per line, for scripts/dgx/enqueue.sh).

    python scripts/dgx/make_r1_tasks.py tune > tasks_r1_tune.txt     # the grid, seed 0
    python scripts/dgx/make_r1_tasks.py final > tasks_r1_final.txt   # chosen recipes, reported seeds
"""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def line(name, data_cfg, model_key, arch, lr, steps, seed, out):
    return (f"scripts/train_continual.py --bench {name} --data {data_cfg} --train configs/r1_base.yaml "
            f"--method joint --arch {arch} --lr-body {lr:g} --steps-per-task {steps} --seed {seed} "
            f"--no-save --out {out} --tag {model_key}_lr{lr:g}_s{steps}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["tune", "final"])
    ap.add_argument("--kaggle", action="store_true", help="configs/<name>_kaggle.yaml, runs_kaggle/ output")
    ap.add_argument("--only", default="", help="comma-separated datasets (default: all)")
    args = ap.parse_args()
    which = args.which
    bench = yaml.safe_load(open(ROOT / "benchmark" / "tasks.yaml"))["datasets"]
    models = yaml.safe_load(open(ROOT / "benchmark" / "models.yaml"))["models"]
    if args.only:
        bench = {n: bench[n] for n in args.only.split(",")}
    out = "runs_kaggle" if args.kaggle else "runs_dgx"

    def config(name, b):
        return f"configs/{name}_kaggle.yaml" if args.kaggle else b["dgx_config"]

    if which == "tune":
        grid = yaml.safe_load(open(ROOT / "benchmark" / "r1_grid.yaml"))
        for name, b in bench.items():
            for key, arch in models.items():
                for lr in grid["lr_body"]:
                    for steps in grid["steps_per_task"][name]:
                        print(line(name, config(name, b), key, arch, lr, steps, 0, f"{out}/r1_tune"))
    else:
        recipes = yaml.safe_load(open(ROOT / "benchmark" / "recipes.yaml"))["recipes"]
        for name, b in bench.items():
            for key, arch in models.items():
                r = recipes[name][key]
                for seed in b["seeds"]:
                    if seed != 0:
                        print(line(name, config(name, b), key, arch, r["lr_body"], r["steps_per_task"],
                                   seed, f"{out}/r1_final"))


if __name__ == "__main__":
    main()
