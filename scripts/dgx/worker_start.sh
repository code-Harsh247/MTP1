#!/bin/bash
# Submit a worker (pilot job). Run from an INTERACTIVE ssh session (sbatch is not on the PATH of
# one-shot `ssh host "cmd"` calls).
#   bash scripts/dgx/worker_start.sh            # 24 h, 2 GPUs
#   bash scripts/dgx/worker_start.sh 72 4       # 72 h (the gpu2 QOS maximum), 4 GPUs
# A shorter time limit usually starts sooner (Slurm backfills short jobs into gaps).
set -euo pipefail
BASE=${NLEAR_BASE:-/dgx1/dsamantaai/santhosh/MTP_Harsh/nlear}
H=${1:-24}
G=${2:-2}
if [ "$H" -gt 72 ]; then echo "the gpu2 QOS allows at most 72 h"; exit 1; fi
cd "$BASE"
mkdir -p logs
rm -f queue/STOP
sbatch --time="$H:00:00" --gres="gpu:$G" --cpus-per-task=$((8 * G)) --mem=$((64 * G))G \
       --export=ALL,WALL_H="$H" scripts/dgx/worker.sbatch
