"""R0: write the frozen task lists of benchmark/tasks.yaml and pin every file by its hash.

For each dataset and seed: read data/<name>/splits_seed{s}.csv, deal the training people into
tasks and hold out their seen-person test images (nlear.continual.make_task_splits), write
data/<name>/tasks_seed{s}.csv, and record the md5 of both files in benchmark/hashes_<name>.txt.
A per-task summary goes to benchmark/summary_<name>.txt.

    .venv/Scripts/python scripts/make_task_lists.py                  # all local datasets
    .venv/Scripts/python scripts/make_task_lists.py --only uerc      # on the DGX
    .venv/Scripts/python scripts/make_task_lists.py --check          # verify against the hashes
"""
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nlear.continual import make_task_splits  # noqa: E402

KEEP = ["path", "identity", "split", "role", "known", "task", "cl_role", "cl_gallery"]


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def task_path(splits_template, seed):
    p = Path(splits_template.format(seed=seed))
    return p.with_name(p.name.replace("splits_", "tasks_", 1))


def summarize(df, n_tasks):
    tr = df[df["split"] == "train"]
    rows = []
    for k in range(n_tasks):
        t = tr[tr["task"] == k]
        rows.append({"task": k, "people": t["identity"].nunique(),
                     "train_imgs": int((t["cl_role"] == "train").sum()),
                     "gallery_imgs": int(t["cl_gallery"].sum()),
                     "held_probes": int((t["cl_role"] == "held_probe").sum())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(ROOT / "benchmark" / "tasks.yaml"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip", nargs="*", default=["uerc"], help="datasets not on this machine")
    ap.add_argument("--check", action="store_true", help="recompute and compare with the hash files")
    args = ap.parse_args()
    bench = yaml.safe_load(open(args.bench))["datasets"]
    names = args.only or [n for n in bench if n not in (args.skip or [])]
    ok = True
    for name in names:
        b = bench[name]
        dcfg = yaml.safe_load(open(ROOT / b["config"]))
        lines, summary = [], []
        for seed in b["seeds"]:
            split_file = Path(dcfg["data"]["splits"].format(seed=seed))
            df = pd.read_csv(split_file, dtype={"identity": str, "path": str})
            tasks = make_task_splits(df, b["n_tasks"], b["holdout_frac"], n_gallery=b["held_gallery"], seed=seed)
            out = task_path(dcfg["data"]["splits"], seed)
            if not args.check:
                tasks[KEEP].to_csv(out, index=False)
            lines += [f"{md5(split_file)}  {split_file.as_posix()}", f"{md5(out)}  {out.as_posix()}"]
            s = summarize(tasks, b["n_tasks"])
            summary.append(f"seed {seed}: {s['people'].sum()} train people, per task "
                           f"{s['people'].min()}-{s['people'].max()}; train imgs {s['train_imgs'].sum()}, "
                           f"gallery imgs {s['gallery_imgs'].sum()}, held-out probes {s['held_probes'].sum()}")
        hash_file = ROOT / "benchmark" / f"hashes_{name}.txt"
        if args.check:
            same = hash_file.exists() and hash_file.read_text().splitlines() == lines
            ok &= same
            print(f"[{name}] {'matches' if same else 'DIFFERS FROM'} {hash_file.name}")
        else:
            hash_file.write_text("\n".join(lines) + "\n")
            (ROOT / "benchmark" / f"summary_{name}.txt").write_text(
                f"{name}: {b['n_tasks']} tasks, holdout {b['holdout_frac']}, held gallery {b['held_gallery']}\n"
                + "\n".join(summary) + "\n")
            print(f"[{name}] wrote {len(b['seeds'])} task files; " + summary[0])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
