#!/bin/bash
# Continual-learning runs as a Slurm job array. The gpu2 QOS caps how many jobs one user may
# have SUBMITTED (QOSMaxSubmitJobPerUserLimit; every array element counts, and the dsamantaai
# account is shared), so each array element runs several task lines in sequence:
# element i runs lines i, i+N, i+2N, ... of $TASKS, where N = number of array elements.
# Line format: <method> <lam> <buffer_per_id> <seed>        e.g.  "ewc 1e4 0 3"
#   sbatch --array=0-7 --export=ALL,TASKS=$PWD/tasks_c2.txt scripts/dgx/cl_array.sbatch
#SBATCH --job-name=harsh_cl
#SBATCH --partition=dgx_all
#SBATCH --account=dsamantaai
#SBATCH --qos=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -uo pipefail
cd /dgx1/dsamantaai/santhosh/MTP_Harsh/nlear
# Call the env's python by full path: `conda activate` does not reliably take effect in
# non-interactive shells on this cluster (pip resolved to the system python, 2026-10-20).
PY=/home/dsamantaai/miniconda3/envs/harsh_nlear/bin/python

N=${SLURM_ARRAY_TASK_COUNT:?not an array job}
NLINES=$(grep -c . "$TASKS")
echo "array element $SLURM_ARRAY_TASK_ID of $N on $(hostname); $NLINES task lines"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for (( i = SLURM_ARRAY_TASK_ID; i < NLINES; i += N )); do
  LINE=$(sed -n "$((i + 1))p" "$TASKS" | tr -d '\r')
  read -r METHOD LAM K SEED <<< "$LINE"
  echo "task $i: method=$METHOD lam=$LAM k=$K seed=$SEED  ($(date +%H:%M:%S))"
  # One failed run must not stop the remaining runs of this element.
  "$PY" scripts/train_continual.py --data configs/earvn_dgx.yaml --method "$METHOD" \
      --lam "$LAM" --buffer-per-id "$K" --seed "$SEED" --out runs_dgx \
    || echo "task $i FAILED (exit $?)"
done
echo "array element $SLURM_ARRAY_TASK_ID done ($(date +%H:%M:%S))"
