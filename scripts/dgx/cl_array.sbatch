#!/bin/bash
# Continual-learning runs as a Slurm job array: one line of $TASKS per array index.
# Line format: <method> <lam> <buffer_per_id> <seed>        e.g.  "ewc 1e4 0 3"
#   sbatch --array=0-$(( $(wc -l < tasks.txt) - 1 ))%8 --export=ALL,TASKS=$PWD/tasks.txt scripts/dgx/cl_array.sbatch
#SBATCH --job-name=harsh_cl
#SBATCH --partition=dgx_all
#SBATCH --account=dsamantaai
#SBATCH --qos=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd /dgx1/dsamantaai/santhosh/MTP_Harsh/nlear
# Call the env's python by full path: `conda activate` does not reliably take effect in
# non-interactive shells on this cluster (pip resolved to the system python, 2026-10-20).
PY=/home/dsamantaai/miniconda3/envs/harsh_nlear/bin/python

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS")
read -r METHOD LAM K SEED <<< "$LINE"
echo "task $SLURM_ARRAY_TASK_ID: method=$METHOD lam=$LAM k=$K seed=$SEED on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PY" scripts/train_continual.py --data configs/earvn_dgx.yaml --method "$METHOD" \
    --lam "$LAM" --buffer-per-id "$K" --seed "$SEED" --out runs_dgx
