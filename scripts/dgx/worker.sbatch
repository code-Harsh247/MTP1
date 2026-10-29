#!/bin/bash
# Pilot ("worker") job: wait in the Slurm queue ONCE, then keep the GPUs and run whatever task
# lines are dropped into queue/pending/, one task per GPU at a time, until there is no work for
# IDLE_MIN minutes, a queue/STOP file appears, or too little wall time is left to start a task.
#
# Start it with scripts/dgx/worker_start.sh (sets --time, GPUs, CPUs, memory consistently).
# Send work with scripts/dgx/enqueue.sh, check with scripts/dgx/worker_status.sh.
#
# A task file holds ONE line: the python arguments (script + args), as in py_array.sbatch, e.g.
#   scripts/train_baseline.py --data configs/ami_dgx.yaml --seed 0 --val-only --no-save
# Queue layout:   pending/ -> running/ -> done/ | failed/   (moves are atomic renames)
# Each task's output: logs/worker/<task>.out; one row per finished task in queue/ledger.tsv.
#SBATCH --job-name=harsh_worker
#SBATCH --partition=dgx_all
#SBATCH --account=dsamantaai
#SBATCH --qos=gpu2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
BASE=${NLEAR_BASE:-/dgx1/dsamantaai/santhosh/MTP_Harsh/nlear}
cd "$BASE" || exit 1
PY=${PY:-/home/dsamantaai/miniconda3/envs/harsh_nlear/bin/python}   # full path: conda activate is unreliable
export HF_HUB_OFFLINE=1          # weights are pre-cached on the login node (scripts/dgx/prepare.py --weights)
Q=$BASE/queue
IDLE_MIN=${IDLE_MIN:-30}         # give the GPUs back after this long without work (shared account)
WALL_H=${WALL_H:-24}             # must match --time; worker_start.sh passes both
MIN_LEFT_H=${MIN_LEFT_H:-3}      # never START a task with less than this much wall time left
POLL_SEC=${POLL_SEC:-30}
mkdir -p "$Q"/pending "$Q"/running "$Q"/done "$Q"/failed logs/worker
START=$(date +%s)
STOP_TAKING=$(( START + (WALL_H - MIN_LEFT_H) * 3600 ))
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
echo "worker ${SLURM_JOB_ID:-local} on $(hostname): GPUs ${GPUS[*]}, wall ${WALL_H}h, idle exit ${IDLE_MIN} min, $(date)"

claim() {   # print the name of one pending task this slot now owns, or nothing
  local f
  for f in $(ls -1 "$Q/pending" 2>/dev/null | sort); do
    if mv "$Q/pending/$f" "$Q/running/$f" 2>/dev/null; then echo "$f"; return; fi
  done
}

slot() {
  local gpu=$1 idle_since task line log t0 rc dt dest
  idle_since=$(date +%s)
  while true; do
    if [ -e "$Q/STOP" ]; then echo "[gpu $gpu] STOP file found: slot exits"; return; fi
    if [ "$(date +%s)" -ge "$STOP_TAKING" ]; then echo "[gpu $gpu] < ${MIN_LEFT_H}h left: slot exits"; return; fi
    task=$(claim)
    if [ -z "$task" ]; then
      if [ $(( $(date +%s) - idle_since )) -ge $(( IDLE_MIN * 60 )) ]; then
        echo "[gpu $gpu] idle for ${IDLE_MIN} min: slot exits"; return
      fi
      sleep "$POLL_SEC"; continue
    fi
    line=$(tr -d '\r' < "$Q/running/$task")
    log=logs/worker/${task%.task}.out
    echo "[gpu $gpu] start $task: $line ($(date +%H:%M:%S))"
    t0=$(date +%s)
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu "$PY" $line > "$log" 2>&1
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    if [ "$rc" -eq 0 ]; then dest=done; else dest=failed; fi
    mv "$Q/running/$task" "$Q/$dest/$task"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%F_%T)" "$task" "$dest" "$dt" "$gpu" "${SLURM_JOB_ID:-local}" >> "$Q/ledger.tsv"
    echo "[gpu $gpu] $dest $task (exit $rc, ${dt}s)"
    idle_since=$(date +%s)
  done
}

for g in "${GPUS[@]}"; do
  slot "$g" &
  sleep 1          # stagger the slots so they don't race for the same first task
done
wait
echo "worker done $(date)"
