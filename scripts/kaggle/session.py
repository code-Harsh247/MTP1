"""One Kaggle session: find the attached code and data, set up, run the task list.

Called from the notebook cell in notes/kaggle_runbook.md. Attach the nlear-code and nlear-data
datasets and, from the second session on, the previous session's output (it holds ledger.tsv).
Results land in the output folder: results.tar.gz and ledger.tsv.
"""
import argparse
import importlib.metadata
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import Runner  # noqa: E402

TIMM_VERSION = "1.0.30"      # same as the DGX and the laptop
DATASETS = ("ami", "awe", "iitd", "earvn", "uerc")


def walk_files(root):
    for folder, _, files in os.walk(root, followlinks=True):
        for f in files:
            yield Path(folder) / f


def find_gpus():
    """GPUs from nvidia-smi. Stops the session rather than training on the CPU or a P100."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        sys.exit("no GPU in this session: set Accelerator to GPU T4 x2")
    gpus = []
    for row in out.strip().splitlines():
        index, name = row.split(",", 1)
        gpus.append({"index": int(index), "name": name.strip()})
    if any("P100" in g["name"] for g in gpus):
        sys.exit("this session has a P100, which recent PyTorch may not support: set Accelerator to GPU T4 x2")
    return gpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", choices=["tune", "final"], help="task list from make_r1_tasks.py")
    ap.add_argument("--r2", choices=["first", "second"], help="task list from make_r2_tasks.py")
    ap.add_argument("--r3", choices=["cms", "replay"],
                    help="cms: tune the slow lr on seed 0, select by the committed rule, then run naive + CMS; "
                         "replay: replay + CMS, with benchmark/r3_lr_pow.yaml already committed")
    ap.add_argument("--tasks", help="or a task file, relative to the code folder")
    ap.add_argument("--only", default="ami,awe,iitd,earvn")
    ap.add_argument("--input-root", default="/kaggle/input")
    ap.add_argument("--work", default="/tmp/nlear")
    ap.add_argument("--output", default="/kaggle/working")
    ap.add_argument("--hours", type=float, default=11.0)
    ap.add_argument("--min-left", type=float, default=1.0)
    ap.add_argument("--no-pip", action="store_true")
    args = ap.parse_args()
    if sum(map(bool, (args.r1, args.r2, args.r3, args.tasks))) != 1:
        sys.exit("give exactly one of --r1, --r2, --r3 and --tasks")

    files = list(walk_files(args.input_root))
    code = next((f.parents[1] for f in files if f.as_posix().endswith("scripts/train_continual.py")), None)
    if code is None:
        sys.exit(f"nlear-code is not attached: no scripts/train_continual.py under {args.input_root}")
    work = Path(args.work)
    if not work.exists():
        shutil.copytree(code, work)
    data = work / "kaggle_data"
    only = args.only.split(",")
    for name in only:
        if name not in DATASETS:
            sys.exit(f"unknown dataset {name}")
        found = next((f.parent for f in files if f.as_posix().endswith(f"{name}/manifest.csv")), None)
        if found is None:
            sys.exit(f"nlear-data does not have {name}: no {name}/manifest.csv under {args.input_root}")
        if not (data / name).exists():
            print(f"copying {name} to local disk")
            shutil.copytree(found, data / name)     # local disk reads faster than /kaggle/input

    if not args.no_pip:
        try:
            have = importlib.metadata.version("timm")
        except importlib.metadata.PackageNotFoundError:
            have = None
        if have != TIMM_VERSION:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"timm=={TIMM_VERSION}"], check=True)

    def task_lines(script, which):
        return subprocess.run([sys.executable, f"scripts/dgx/{script}", which, "--kaggle", "--only", args.only],
                              cwd=work, capture_output=True, text=True, check=True).stdout

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    previous = [f for f in files if f.name == "ledger.tsv"]
    gpus = find_gpus()
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    keep = ["runs_kaggle", "benchmark/r3_lr_pow.yaml"]
    t0 = time.time()

    if args.r3 == "cms":
        attached = next((f for f in files if f.name == "r3_lr_pow.yaml"), None)
        if attached:     # a resumed session: the earlier one already selected
            shutil.copy(attached, work / "benchmark" / "r3_lr_pow.yaml")
        else:
            lines = [ln for ln in task_lines("make_r3_tasks.py", "tune").splitlines() if ln.strip()]
            print(f"{len(lines)} tuning lines, GPUs: {gpus}", flush=True)
            Runner(lines, gpus, args.hours, args.min_left, work, output, keep, previous, "tune").run()
            sel = subprocess.run([sys.executable, "scripts/r3_select.py", "runs_kaggle/r3", "--allow-missing"],
                                 cwd=work, capture_output=True, text=True)
            print(sel.stdout, sel.stderr, flush=True)
            if sel.returncode != 0:
                sys.exit("selection failed: no naive + CMS runs started")
            shutil.copy(work / "benchmark" / "r3_lr_pow.yaml", output / "r3_lr_pow.yaml")
            previous = [*previous, output / "ledger.tsv"]

    if args.r1 or args.r2 or args.r3:
        script, which = (("make_r1_tasks.py", args.r1) if args.r1 else
                         ("make_r2_tasks.py", args.r2) if args.r2 else ("make_r3_tasks.py", args.r3))
        tasks = task_lines(script, which)
    else:
        tasks = (work / args.tasks).read_text()
    lines = [ln.strip() for ln in tasks.splitlines() if ln.strip()]
    print(f"{len(lines)} task lines, {len(previous)} earlier ledgers, GPUs: {gpus}", flush=True)
    hours_left = args.hours - (time.time() - t0) / 3600
    runner = Runner(lines, gpus, hours_left, args.min_left, work, output, keep, previous)
    counts, left = runner.run()
    print("all task lines finished" if counts["failed"] + counts["killed"] + left == 0
          else "not finished: run another session with this output attached")


if __name__ == "__main__":
    main()
