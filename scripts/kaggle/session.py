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
    ap.add_argument("--tasks", help="or a task file, relative to the code folder")
    ap.add_argument("--only", default="ami,awe,iitd,earvn")
    ap.add_argument("--input-root", default="/kaggle/input")
    ap.add_argument("--work", default="/tmp/nlear")
    ap.add_argument("--output", default="/kaggle/working")
    ap.add_argument("--hours", type=float, default=11.0)
    ap.add_argument("--min-left", type=float, default=1.0)
    ap.add_argument("--no-pip", action="store_true")
    args = ap.parse_args()
    if bool(args.r1) == bool(args.tasks):
        sys.exit("give exactly one of --r1 and --tasks")

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

    if args.r1:
        tasks = subprocess.run([sys.executable, "scripts/dgx/make_r1_tasks.py", args.r1, "--kaggle",
                                "--only", args.only], cwd=work, capture_output=True, text=True, check=True).stdout
    else:
        tasks = (work / args.tasks).read_text()
    lines = [ln.strip() for ln in tasks.splitlines() if ln.strip()]

    output = Path(args.output).resolve()
    previous = [f for f in files if f.name == "ledger.tsv"]
    gpus = find_gpus()
    print(f"{len(lines)} task lines, {len(previous)} earlier ledgers, GPUs: {gpus}", flush=True)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    runner = Runner(lines, gpus, args.hours, args.min_left, work, output, ["runs_kaggle"], previous)
    counts, left = runner.run()
    print("all task lines finished" if counts["failed"] + counts["killed"] + left == 0
          else "not finished: run another session with this output attached")


if __name__ == "__main__":
    main()
