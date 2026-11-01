"""Run task lines on a Kaggle session, one per GPU at a time (the worker.sbatch idea, in Python).

Kaggle kills a session at its time limit and then keeps nothing, so this runner never overruns:
it starts no task with less than --min-left hours to go, and at --hours it stops whatever is
still running. Each session keeps a cumulative ledger, so the next session skips finished lines.
A task line is the python arguments (script + args), as in scripts/dgx/enqueue.sh.
"""
import argparse
import os
import shlex
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

MAX_FAILS_IN_A_ROW = 5   # e.g. no internet for the weights: don't burn through the whole list


def read_done(ledgers):
    """Rows of earlier ledgers whose status is done, keyed by task line."""
    done = {}
    for f in ledgers:
        for row in Path(f).read_text().splitlines():
            parts = row.split("\t")
            if len(parts) == 6 and parts[1] == "done":
                done[parts[5]] = row
    return done


class Runner:
    def __init__(self, lines, gpus, hours, min_left, cwd, output, keep, previous=()):
        self.cwd, self.output, self.gpus = Path(cwd), Path(output), gpus
        self.keep = keep                     # folders under cwd that go into results.tar.gz
        self.output.mkdir(parents=True, exist_ok=True)
        self.logs = self.cwd / "logs_kaggle"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.ledger = self.output / "ledger.tsv"
        done = read_done(previous)
        self.ledger.write_text("".join(r + "\n" for r in done.values()))
        self.todo = [ln for ln in lines if ln not in done]
        self.n_skipped = len(lines) - len(self.todo)
        self.total = len(self.todo)
        self.lock = threading.Lock()
        self.fails_in_a_row = 0
        self.counts = {"done": 0, "failed": 0, "killed": 0}
        start = time.time()
        self.deadline = start + hours * 3600
        self.stop_taking = self.deadline - min_left * 3600

    def next_task(self):
        with self.lock:
            if self.todo and time.time() < self.stop_taking and self.fails_in_a_row < MAX_FAILS_IN_A_ROW:
                n = self.total - len(self.todo)
                return n, self.todo.pop(0)
        return None, None

    def finish(self, line, status, sec, gpu):
        with self.lock:
            self.counts[status] += 1
            self.fails_in_a_row = 0 if status == "done" else self.fails_in_a_row + 1
            with open(self.ledger, "a") as f:
                f.write(f"{time.strftime('%F_%T')}\t{status}\t{sec}\t{gpu['index']}\t{gpu['name']}\t{line}\n")
            self.pack()
            print(f"[gpu {gpu['index']}] {status} in {sec}s: {line}", flush=True)

    def pack(self):
        """Rewrite results.tar.gz, so a session that dies early still has what finished."""
        tmp = self.output / "results.tar.gz.tmp"
        with tarfile.open(tmp, "w:gz") as tar:
            for name in [*self.keep, self.logs.name]:
                if (self.cwd / name).exists():
                    tar.add(self.cwd / name, arcname=name)
        os.replace(tmp, self.output / "results.tar.gz")

    def slot(self, gpu):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu["index"])}
        while True:
            n, line = self.next_task()
            if line is None:
                return
            print(f"[gpu {gpu['index']}] start {n + 1}/{self.total}: {line}", flush=True)
            t0 = time.time()
            with open(self.logs / f"task_{self.n_skipped + n:04d}.out", "w") as log:
                p = subprocess.Popen([sys.executable, *shlex.split(line)], cwd=self.cwd, env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
                while p.poll() is None and time.time() < self.deadline:
                    time.sleep(min(5, max(0.05, self.deadline - time.time())))
                if p.poll() is None:
                    p.kill()
                    p.wait()
                    status = "killed"
                else:
                    status = "done" if p.returncode == 0 else "failed"
            self.finish(line, status, round(time.time() - t0), gpu)

    def run(self):
        threads = [threading.Thread(target=self.slot, args=(g,)) for g in self.gpus]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.pack()
        left = len(self.todo)
        print(f"session over: {self.counts}, {left} not started, {self.n_skipped} already done earlier")
        if self.fails_in_a_row >= MAX_FAILS_IN_A_ROW:
            print(f"STOPPED EARLY: {MAX_FAILS_IN_A_ROW} tasks failed in a row; see logs_kaggle in results.tar.gz")
        return self.counts, left


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks")
    ap.add_argument("--cwd", default=".")
    ap.add_argument("--output", default="kaggle_out")
    ap.add_argument("--hours", type=float, default=11.0)
    ap.add_argument("--min-left", type=float, default=1.0)
    ap.add_argument("--gpus", default="0", help="comma-separated indices")
    ap.add_argument("--keep", default="runs_kaggle")
    ap.add_argument("--previous", nargs="*", default=[], help="ledger.tsv files of earlier sessions")
    args = ap.parse_args()
    lines = [ln.strip() for ln in Path(args.tasks).read_text().splitlines() if ln.strip()]
    gpus = [{"index": int(g), "name": "unknown"} for g in args.gpus.split(",")]
    Runner(lines, gpus, args.hours, args.min_left, args.cwd, args.output, args.keep.split(","),
           args.previous).run()


if __name__ == "__main__":
    main()
