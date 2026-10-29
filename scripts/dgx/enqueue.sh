#!/bin/bash
# Send a task list to the worker queue: one task file per non-empty line, in order.
#   bash scripts/dgx/enqueue.sh tasks_r1_tune.txt            # tag = file name
#   bash scripts/dgx/enqueue.sh tasks_r1_tune.txt r1tune     # explicit tag
# Files are written under a dot-name and renamed, so a worker never reads a half-written task.
set -euo pipefail
BASE=${NLEAR_BASE:-/dgx1/dsamantaai/santhosh/MTP_Harsh/nlear}
Q=$BASE/queue
mkdir -p "$Q/pending"
src=${1:?usage: enqueue.sh TASKS_FILE [TAG]}
tag=${2:-$(basename "$src" .txt)}
stamp=$(date +%Y%m%d-%H%M%S)
i=0
while IFS= read -r line || [ -n "$line" ]; do
  line=${line%$'\r'}
  [ -z "${line// }" ] && continue
  name=$(printf '%s_%s_%04d.task' "$stamp" "$tag" "$i")
  printf '%s\n' "$line" > "$Q/pending/.$name"
  mv "$Q/pending/.$name" "$Q/pending/$name"
  i=$((i + 1))
done < "$src"
echo "queued $i tasks as ${stamp}_${tag}_*.task"
