"""Write the R3 task lists (one python run per line), in benchmark/r3_rule.yaml's design.

    python scripts/dgx/make_r3_tasks.py tune      # seed 0, each slow-lr value
    python scripts/dgx/make_r3_tasks.py cms       # naive + CMS, reported seeds (primary)
    python scripts/dgx/make_r3_tasks.py replay    # replay + CMS, reported seeds (secondary)
"""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["tune", "cms", "replay"])
    ap.add_argument("--kaggle", action="store_true", help="configs/<name>_kaggle.yaml, runs_kaggle/ output")
    ap.add_argument("--only", default="", help="comma-separated datasets (default: all)")
    args = ap.parse_args()
    bench = yaml.safe_load(open(ROOT / "benchmark" / "tasks.yaml"))["datasets"]
    models = yaml.safe_load(open(ROOT / "benchmark" / "models.yaml"))["models"]
    rule = yaml.safe_load(open(ROOT / "benchmark" / "r3_rule.yaml"))
    recipes = yaml.safe_load(open(ROOT / "benchmark" / "recipes.yaml"))["recipes"]
    if args.only:
        bench = {n: bench[n] for n in args.only.split(",")}
    out = "runs_kaggle/r3" if args.kaggle else "runs_dgx/r3"
    periods = ",".join(str(p) for p in rule["periods"])

    def line(name, b, key, method, seed, pow_):
        r = recipes[name][key]
        config = f"configs/{name}_kaggle.yaml" if args.kaggle else b["dgx_config"]
        extra = f" --buffer-per-id {rule['replay_k']}" if method == "replay" else ""
        return (f"scripts/train_continual.py --bench {name} --data {config} --train configs/r1_base.yaml "
                f"--method {method} --arch {models[key]} --lr-body {r['lr_body']:g} "
                f"--steps-per-task {r['steps_per_task']} --seed {seed}{extra} --body-cms-periods {periods} "
                f"--body-cms-lr-pow {pow_:g} --no-save --out {out} "
                f"--tag {key}_lr{r['lr_body']:g}_s{r['steps_per_task']}")

    if args.which == "tune":
        for name, b in bench.items():
            for key in models:
                for pow_ in rule["lr_pows"]:
                    print(line(name, b, key, "naive", 0, pow_))
    else:
        chosen = yaml.safe_load(open(ROOT / "benchmark" / "r3_lr_pow.yaml"))["lr_pow"]
        method = "naive" if args.which == "cms" else "replay"
        for name, b in bench.items():
            for key in models:
                for seed in b["seeds"]:
                    if seed != 0:
                        print(line(name, b, key, method, seed, chosen[name][key]))


if __name__ == "__main__":
    main()
