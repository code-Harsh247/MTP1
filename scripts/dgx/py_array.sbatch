#!/bin/bash
# Generic job array: each line of $TASKS is the argument list of one python run (script + args),
#   e.g. "scripts/train_baseline.py --data configs/uerc.yaml --seed 0 --val-only --set epochs=2"
# Element i runs lines i, i+N, i+2N, ... (N = array size): the gpu2 QOS limits how many jobs the
# shared account may have submitted, and every array element counts.
#   sbatch --array=0-4 --export=ALL,TASKS=$PWD/tasks_u1_tune.txt scripts/dgx/py_array.sbatch
#SBATCH --job-name=harsh_py
#SBATCH --partition=dgx_all
#SBATCH --account=dsamantaai
#SBATCH --qos=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -uo pipefail
cd /dgx1/dsamantaai/santhosh/MTP_Harsh/nlear
PY=/home/dsamantaai/miniconda3/envs/harsh_nlear/bin/python   # full path: conda activate is unreliable here

N=${SLURM_ARRAY_TASK_COUNT:?not an array job}
NLINES=$(grep -c . "$TASKS")
echo "array element $SLURM_ARRAY_TASK_ID of $N on $(hostname); $NLINES task lines"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for (( i = SLURM_ARRAY_TASK_ID; i < NLINES; i += N )); do
  LINE=$(sed -n "$((i + 1))p" "$TASKS" | tr -d '\r')
  echo "task $i: $LINE  ($(date +%H:%M:%S))"
  # shellcheck disable=SC2086
  "$PY" $LINE || echo "task $i FAILED (exit $?)"
done
echo "array element $SLURM_ARRAY_TASK_ID done ($(date +%H:%M:%S))"
