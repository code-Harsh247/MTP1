"""Train the diagnostic runs for one model x dataset and analyse each (scripts/diagnose_cms.py).
Every run uses --save-stages, the cell's recipe and R3's slow-lr choice. Finished runs are skipped,
so it can be restarted.

Arms:
    naive     plain sequential fine-tuning
    r3        CMS, periods 1,16,256 (R3)
    r4        CMS, periods 1,S,2S (R4)
    r4b       copies consolidated at task ends, every 1,2 tasks, beta 0.5 (R4b)
    r5pow1    R3's periods with the slow lr scaled by the period (lr_pow 1)       (R5 exp. 1a)
    r5step    R3's periods, slow levels in per-step mode                           (R5 exp. 1b)
    r5snap    copies 2-3 frozen at copy 1's weights after task 1                   (R5 exp. 3)

    python scripts/run_cms_diagnostics.py                       # EarVN ConvNeXt-T, seed 1: naive, r3, r4
    python scripts/run_cms_diagnostics.py --arms r3 --data configs/earvn_kaggle.yaml \
        --out runs_kaggle/diag --delete-weights                 # one arm, as a Kaggle task line
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def arm_table(S, pow_, slow_tasks, r4b, r5):
    """arm -> (label, run-folder prefix, extra trainer flags)."""
    r3p = "1,16,256"
    every = ",".join(str(e) for e in r4b["every"])
    return {
        "naive": ("naive", "naive", []),
        "r3": ("R3 CMS", f"naive_bodycms1-16-256_p{pow_:g}",
               ["--body-cms-periods", r3p, "--body-cms-lr-pow", f"{pow_:g}"]),
        "r4": ("R4 CMS", f"naive_bodycms1-{S}-{slow_tasks * S}_p{pow_:g}",
               ["--body-cms-periods", f"1,{S},{slow_tasks * S}", "--body-cms-lr-pow", f"{pow_:g}"]),
        "r4b": ("R4b consolidation", f"naive_consol{every.replace(',', '-')}_b{r4b['beta']:g}",
                ["--consolidate-every", every, "--consolidate-beta", f"{r4b['beta']:g}"]),
        "r5pow1": ("R5 lr_pow 1", "naive_bodycms1-16-256_p1",
                   ["--body-cms-periods", r3p, "--body-cms-lr-pow", "1"]),
        "r5step": ("R5 per-step", "naive_bodycms1-16-256_p0_perstep",
                   ["--body-cms-periods", r3p, "--body-cms-lr-pow", "0", "--body-cms-slow", "per-step"]),
        "r5snap": ("R5 snapshot", f"naive_snapshot{r5['snapshot_after_task']}",
                   ["--snapshot-after-task", str(r5["snapshot_after_task"])]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="earvn")
    ap.add_argument("--model", default="modern_cnn")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data", default="configs/earvn.yaml", help="this machine's data config")
    ap.add_argument("--out", default="runs/diag")
    ap.add_argument("--arms", default="naive,r3,r4", help="comma-separated; see the list above")
    ap.add_argument("--delete-weights", action="store_true",
                    help="remove the saved weights after the analysis (keeps diag.md, diag.json, metrics.json)")
    ap.add_argument("--offline", action="store_true", help="use only cached pretrained weights")
    ap.add_argument("--steps-per-task", type=int, help="override the recipe (only for a quick test)")
    args = ap.parse_args()
    bm = ROOT / "benchmark"
    recipe = yaml.safe_load(open(bm / "recipes.yaml"))["recipes"][args.dataset][args.model]
    arch = yaml.safe_load(open(bm / "models.yaml"))["models"][args.model]
    pow_ = yaml.safe_load(open(bm / "r3_lr_pow.yaml"))["lr_pow"][args.dataset][args.model]
    slow_tasks = yaml.safe_load(open(bm / "r4_design.yaml"))["slow_tasks"]
    r4b = yaml.safe_load(open(bm / "r4b_design.yaml"))
    r5 = yaml.safe_load(open(bm / "r5_design.yaml"))
    n_tasks = yaml.safe_load(open(bm / "tasks.yaml"))["datasets"][args.dataset]["n_tasks"]
    S = args.steps_per_task or recipe["steps_per_task"]
    tag = f"{args.model}_lr{recipe['lr_body']:g}_s{recipe['steps_per_task']}"
    arms = arm_table(S, pow_, slow_tasks, r4b, r5)
    env = {**os.environ, **({"HF_HUB_OFFLINE": "1"} if args.offline else {})}
    dirs = []
    for arm in args.arms.split(","):
        label, folder, extra = arms[arm]
        run_dir = ROOT / args.out / args.dataset / "continual" / f"{folder}_{tag}" / f"seed{args.seed}"
        dirs.append(run_dir)
        if (run_dir / "diag.md").exists() or ((run_dir / f"stage{n_tasks - 1}.pt").exists()
                                             and (run_dir / "metrics.json").exists()):
            print(f"{label}: already trained, skipping", flush=True)
            continue
        cmd = [sys.executable, "scripts/train_continual.py", "--bench", args.dataset, "--data", args.data,
               "--train", "configs/r1_base.yaml", "--method", "naive", "--arch", arch,
               "--lr-body", f"{recipe['lr_body']:g}", "--steps-per-task", str(S), "--seed", str(args.seed),
               *extra, "--no-save", "--save-stages", "--out", args.out, "--tag", tag]
        print(f"training {label} ...", flush=True)
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    todo = [d for d in dirs if not (d / "diag.md").exists()]
    if todo:
        subprocess.run([sys.executable, "scripts/diagnose_cms.py", *map(str, todo)], cwd=ROOT, env=env, check=True)
    if args.delete_weights:
        for d in dirs:
            for f in d.glob("*.pt"):
                f.unlink()
    print("Done. Each run folder now has diag.md:", *dirs, sep="\n  ")


if __name__ == "__main__":
    main()
