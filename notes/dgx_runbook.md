# Running the continual-learning experiments on the DGX

The DGX runs many seeds and methods **in parallel** (a Slurm job array, at most 8 at a time), while the laptop runs them one after another.

- **Login:** `dsamantaai@gpucluster.iitkgp.ac.in` (VPN may be needed off campus).
- **The account is shared:** everything goes under `/dgx1/dsamantaai/santhosh/MTP_Harsh/`, jobs are named `harsh_*`, and **never `scancel -u`**.
- **Same splits everywhere:** the split and manifest CSVs are *copied from the laptop*. That keeps the people, tasks and held-out images identical to the laptop runs.

Run every command below from `C:\MTP1` in **Git Bash** on the laptop, unless it says "on the DGX". Replace `DGX` with `dsamantaai@gpucluster.iitkgp.ac.in`.

## A. Ship code and split files (repeat whenever the code changes)

`git archive` packs the **last commit**, so commit first.

```bash
git archive -o nlear_code.tar.gz HEAD
```

```bash
ssh DGX "mkdir -p /dgx1/dsamantaai/santhosh/MTP_Harsh/nlear /dgx1/dsamantaai/santhosh/MTP_Harsh/nlear_data/earvn"
```

```bash
scp nlear_code.tar.gz DGX:/dgx1/dsamantaai/santhosh/MTP_Harsh/
```

```bash
scp data/earvn/manifest.csv data/earvn/splits_seed*.csv data/earvn/splits_g5_seed*.csv DGX:/dgx1/dsamantaai/santhosh/MTP_Harsh/nlear_data/earvn/
```

```bash
ssh DGX "cd /dgx1/dsamantaai/santhosh/MTP_Harsh && tar -xzf nlear_code.tar.gz -C nlear"
```

## B. One-time environment setup (on the DGX)

```bash
ssh DGX
cd /dgx1/dsamantaai/santhosh/MTP_Harsh/nlear
conda create -y -n harsh_nlear python=3.12
conda activate harsh_nlear
```

Check the CUDA version the driver supports with `nvidia-smi` (on a GPU node, or ask the admins). Then install PyTorch from the matching index. `cu121` works with drivers ≥ 525; use `cu124` or `cu128` if the driver is newer.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m pytest -q          # should report all tests passing
```

Build the image cache, download the pretrained weights and check the splits. This takes about 2 minutes; if the cluster discourages work on login nodes, run it inside `srun --partition=dgx_all --account=dsamantaai --qos=gpu2 --gres=gpu:1 --pty bash` instead.

```bash
python scripts/dgx/prepare.py --data configs/earvn_dgx.yaml
```

## C0. (Not needed: tuning finished on the laptop, 2026-10-20) λ tuning on seed 0

This is the same 6-run sweep as the laptop's (`runs/earvn/c2_tune.log`), run in parallel in about 5 minutes. Use it if the laptop sweep is still running when the DGX is ready.

```bash
cd /dgx1/dsamantaai/santhosh/MTP_Harsh/nlear
mkdir -p logs
printf '%s\n' "ewc 1e2 0 0" "ewc 1e4 0 0" "ewc 1e6 0 0" "lwf 1 0 0" "lwf 10 0 0" "lwf 100 0 0" > tasks_tune.txt
sbatch --array=0-5 --export=ALL,TASKS=$PWD/tasks_tune.txt scripts/dgx/cl_array.sbatch
```

When all 6 jobs are done, print their summaries and paste them to Claude:

```bash
grep -h -E "^task [0-9]+:|^SUMMARY" logs/harsh_cl_*.out
```

## C. Submit the C2 runs (on the DGX)

The λ values (both per method) were fixed from the seed-0 tuning on the laptop (`runs/earvn/c2_tune.log`); the task list has 40 lines. The gpu2 QOS limits how many jobs a user may have *submitted* (every array element counts, and the account is shared), so the 40 lines run as **8 array elements of 5 runs each** (element i runs lines i, i+8, …; with 8 elements that is one method across all seeds). If fewer submit slots are free, use fewer elements, e.g. `--array=0-3`; each then runs 10 lines.

```bash
cd /dgx1/dsamantaai/santhosh/MTP_Harsh/nlear
mkdir -p logs
bash scripts/dgx/make_c2_tasks.sh > tasks_c2.txt
squeue -u dsamantaai          # count the jobs already queued on the shared account
sbatch --array=0-7 --export=ALL,TASKS=$PWD/tasks_c2.txt scripts/dgx/cl_array.sbatch
```

Watch only our jobs:

```bash
squeue -u dsamantaai -n harsh_cl
```

To cancel only our jobs, use `scancel -n harsh_cl` (never `scancel -u`). Logs are written to `logs/harsh_cl_<jobid>_<index>.out`.

## D. Bring results back (laptop, Git Bash)

This copies metrics only; the model files (about 100 MB each) stay on the DGX.

```bash
ssh DGX "cd /dgx1/dsamantaai/santhosh/MTP_Harsh/nlear && tar czf c2_results.tar.gz --exclude='*.pt' runs_dgx"
```

```bash
scp DGX:/dgx1/dsamantaai/santhosh/MTP_Harsh/nlear/c2_results.tar.gz .
```

```bash
tar -xzf c2_results.tar.gz && .venv/Scripts/python scripts/summarize_continual.py runs_dgx/earvn/continual --matrix
```
