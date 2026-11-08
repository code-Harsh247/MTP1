"""Write the R5b task list (one python run per line), in benchmark/r5b_design.yaml's design.
Most informative lines first, so a session cut short still answers the main questions.

    python scripts/dgx/make_r5b_tasks.py --kaggle > tasks_r5b.txt
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
    design = yaml.safe_load(open(ROOT / "benchmark" / "r5b_design.yaml"))
    only = set(args.only.split(",")) if args.only else None
    root = "runs_kaggle" if args.kaggle else "runs_dgx"
    k = design["snapshot_after_task"]

    def wanted(name):
        return only is None or name in only

    def seeds(name):
        return [s for s in bench[name]["seeds"] if s != 0]

    def config(name):
        return f"configs/{name}_kaggle.yaml" if args.kaggle else bench[name]["dgx_config"]

    def train_line(name, key, seed, extra):
        r = recipes[name][key]
        return (f"scripts/train_continual.py --bench {name} --data {config(name)} --train configs/r1_base.yaml "
                f"--method naive --arch {models[key]} --lr-body {r['lr_body']:g} "
                f"--steps-per-task {r['steps_per_task']} --seed {seed} {extra} --no-save --out {root}/r5b "
                f"--tag {key}_lr{r['lr_body']:g}_s{r['steps_per_task']}")

    def diag_line(name, key, seed, arms):
        return (f"scripts/run_cms_diagnostics.py --dataset {name} --model {key} --seed {seed} "
                f"--arms {','.join(arms)} --data {config(name)} --out {root}/diag --delete-weights")

    def snap(share=None):
        return f"--snapshot-after-task {k}" + (f" --snapshot-mix {share:g}" if share else "")

    def mix_lines(seed_pick):
        return [train_line(c["dataset"], c["model"], s, snap(share))
                for c in design["cells_exp2"] if wanted(c["dataset"])
                for s in seed_pick(seeds(c["dataset"])) for share in reversed(design["mix_shares"])]

    def snap_lines(seed_pick, datasets):
        return [train_line(c["dataset"], c["model"], s, snap())
                for c in design["cells_exp1"] if c["dataset"] in datasets and wanted(c["dataset"])
                for s in seed_pick(seeds(c["dataset"]))]

    first, rest = (lambda ss: ss[:1]), (lambda ss: ss[1:])
    lines = mix_lines(first)                                     # exp. 2, seed 1
    lines += snap_lines(first, {"earvn"})                        # exp. 1, EarVN seed 1
    lines += [diag_line(c["dataset"], c["model"], design["diag_seed"], ["r5snap"])
              for c in design["diag_exp1"] if wanted(c["dataset"])]
    d = design["diag_exp3"]
    if wanted(d["dataset"]):
        lines.append(diag_line(d["dataset"], d["model"], d["seed"], d["arms"]))
    lines += mix_lines(rest)                                     # exp. 2, other seeds
    lines += snap_lines(rest, {"earvn"})                         # exp. 1, EarVN other seeds
    lines += snap_lines(lambda ss: ss, {"awe", "ami"})           # exp. 1, small datasets
    print("\n".join(lines))


if __name__ == "__main__":
    main()
