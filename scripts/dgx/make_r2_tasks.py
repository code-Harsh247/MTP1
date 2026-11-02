"""Write the R2 task lists (one python run per line), in benchmark/r2_grid.yaml's design.

    python scripts/dgx/make_r2_tasks.py first > tasks_r2_first.txt    # naive, EWC/LwF tuning, replay
    python scripts/dgx/make_r2_tasks.py second > tasks_r2_second.txt  # EWC/LwF at the chosen strength

Naive comes first, because the forgetting verdict depends on it alone.
"""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["first", "second"])
    ap.add_argument("--kaggle", action="store_true", help="configs/<name>_kaggle.yaml, runs_kaggle/ output")
    ap.add_argument("--only", default="", help="comma-separated datasets (default: all)")
    args = ap.parse_args()
    bench = yaml.safe_load(open(ROOT / "benchmark" / "tasks.yaml"))["datasets"]
    models = yaml.safe_load(open(ROOT / "benchmark" / "models.yaml"))["models"]
    grid = yaml.safe_load(open(ROOT / "benchmark" / "r2_grid.yaml"))
    recipes = yaml.safe_load(open(ROOT / "benchmark" / "recipes.yaml"))["recipes"]
    if args.only:
        bench = {n: bench[n] for n in args.only.split(",")}
    out = "runs_kaggle/r2" if args.kaggle else "runs_dgx/r2"

    def line(name, b, key, method, seed, extra=""):
        r = recipes[name][key]
        config = f"configs/{name}_kaggle.yaml" if args.kaggle else b["dgx_config"]
        return (f"scripts/train_continual.py --bench {name} --data {config} --train configs/r1_base.yaml "
                f"--method {method} --arch {models[key]} --lr-body {r['lr_body']:g} "
                f"--steps-per-task {r['steps_per_task']} --seed {seed}{extra} --no-save --out {out} "
                f"--tag {key}_lr{r['lr_body']:g}_s{r['steps_per_task']}")

    def cells():
        for name, b in bench.items():
            for key in models:
                yield name, b, key, [s for s in b["seeds"] if s != 0]

    if args.which == "first":
        for name, b, key, seeds in cells():
            for seed in seeds:
                print(line(name, b, key, "naive", seed))
        for name, b, key, _ in cells():
            for method, values in grid["strengths"].items():
                for lam in values:
                    print(line(name, b, key, method, 0, f" --lam {lam:g}"))
        for name, b, key, seeds in cells():
            for seed in seeds:
                print(line(name, b, key, "replay", seed, f" --buffer-per-id {grid['replay_k']}"))
    else:
        chosen = yaml.safe_load(open(ROOT / "benchmark" / "r2_strengths.yaml"))["strengths"]
        for name, b, key, seeds in cells():
            for method in grid["strengths"]:
                for seed in seeds:
                    print(line(name, b, key, method, seed, f" --lam {chosen[name][key][method]:g}"))


if __name__ == "__main__":
    main()
