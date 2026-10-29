#!/bin/bash
# Show the worker job(s) and the queue.
#   bash scripts/dgx/worker_status.sh          # summary + last 10 finished tasks
#   bash scripts/dgx/worker_status.sh 40       # last 40 finished tasks
BASE=${NLEAR_BASE:-/dgx1/dsamantaai/santhosh/MTP_Harsh/nlear}
Q=$BASE/queue
N=${1:-10}
echo "== worker jobs"
if command -v squeue > /dev/null; then
  squeue -n harsh_worker -o "%.10i %.8T %.10M %.10L %.6D %b %R" 2>/dev/null || true
else
  echo "(squeue not on PATH: use an interactive ssh session)"
fi
echo "== queue"
for s in pending running done failed; do
  printf '%-8s %5s\n' "$s" "$(ls -1 "$Q/$s" 2>/dev/null | wc -l)"
done
[ -e "$Q/STOP" ] && echo "STOP file present: workers exit after their current task"
if [ -n "$(ls -A "$Q/running" 2>/dev/null)" ]; then
  echo "== running"; ls -1 "$Q/running"
fi
if [ -n "$(ls -A "$Q/failed" 2>/dev/null)" ]; then
  echo "== failed (see logs/worker/<name>.out)"; ls -1 "$Q/failed" | tail -n "$N"
fi
if [ -s "$Q/ledger.tsv" ]; then
  echo "== last $N finished (time, task, result, seconds, gpu, job)"
  tail -n "$N" "$Q/ledger.tsv"
fi
