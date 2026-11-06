"""Write the R4b task list (one python run per line), in benchmark/r4b_design.yaml's design.

    python scripts/dgx/make_r4b_tasks.py > tasks_r4b.txt
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
    design = yaml.safe_load(open(ROOT / "benchmark" / "r4b_design.yaml"))
    recipes = yaml.safe_load(open(ROOT / "benchmark" / "recipes.yaml"))["recipes"]
    names = args.only.split(",") if args.only else list(design["cells"])
    out = "runs_kaggle/r4b" if args.kaggle else "runs_dgx/r4b"
    every = ",".join(str(e) for e in design["every"])
    for name in names:
        b = bench[name]
        config = f"configs/{name}_kaggle.yaml" if args.kaggle else b["dgx_config"]
        for key in design["cells"].get(name, []):
            r = recipes[name][key]
            for seed in b["seeds"]:
                if seed != 0:
                    print(f"scripts/train_continual.py --bench {name} --data {config} --train configs/r1_base.yaml "
                          f"--method naive --arch {models[key]} --lr-body {r['lr_body']:g} "
                          f"--steps-per-task {r['steps_per_task']} --seed {seed} --consolidate-every {every} "
                          f"--consolidate-beta {design['beta']:g} --no-save --out {out} "
                          f"--tag {key}_lr{r['lr_body']:g}_s{r['steps_per_task']}")


if __name__ == "__main__":
    main()
