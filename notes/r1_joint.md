# R1: joint baselines and tuned recipes

*Design committed 2026-10-31, before any tuning run. Results are appended below when they exist.*

## What R1 does

For each of the 25 model × dataset cells, R1:

1. picks one recipe on seed-0 data;
2. trains the joint model (all tasks at once) with that recipe on the reported seeds.

Joint is the upper bound R2 compares against. Every later arm of that cell reuses its recipe.

## Fixed recipe (`configs/r1_base.yaml`)

The same for all five families: AdamW, weight decay 0.05, head lr 1e-3, batch 64, dropout 0.3, v1's augmentation, ArcFace (s = 30, m = 0.3), fp32.

**Changed from v1:** ResNet-50 used weight decay 5e-3 in v1. One value for all families keeps the comparison simple, but this may not suit ResNet-50 equally well, and it isn't tuned here.

## Grid (`benchmark/r1_grid.yaml`)

6 configs per cell, so **150 joint runs** on seed 0.

| | Values |
|---|---|
| `lr_body` | 3e-5, 1e-4, 3e-4 |
| Steps per task, AMI / AWE / IITD | 60, 160 |
| Steps per task, EarVN | 250, 500 |
| Steps per task, UERC | 870, 1,740 |

Joint trains for n_tasks × steps per task in total.

## Selection rule (`scripts/r1_select.py`, unit-tested)

Per cell, on seed 0 only:

1. **Main score:** final rank-1 on the held-out images of the people trained on. This is the study's primary metric.
2. **Near-ties:** among configs within 0.005 of the best, take the highest AUROC on unseen validation people.
3. **Exact ties:** fewer steps, then lower learning rate.

Failed or non-finite runs are not eligible.

**Why this differs from v1:** v1 selected on unseen-people DIR@5%. Study v2's question is forgetting of learned people, so selection uses their score. AMI and AWE also have only 5 known and 5 unknown validation people, which makes DIR@5% too noisy to choose on. **The cost:** this could favour recipes that are slightly worse on unseen people. R1 still reports those metrics.

**Known weakness:** on AMI and AWE, seed 0 has only 70 held-out probes, so one probe moves rank-1 by about 0.014. Their selections are partly noise.

## Trainer changes

- `--bench <name>` loads the frozen task file and refuses it if its md5, or its split file's md5, differs from `benchmark/hashes_<name>.txt`. It also refuses seeds outside the benchmark. This closes R0's first known gap.
- `--arch`, `--lr-body`, `--no-save`.
- `--body-cms-periods` works for any family (the old `--vit-cms-*` names still work).

**Disk:** `--no-save` skips `model.pt` (about 100 MB per run) and the per-run task file.

## Smoke test (laptop, 2026-10-31)

These confirm the code runs. **They are not results:** they are single short runs, partly with arbitrary settings.

| Run | Checks | Outcome |
|---|---|---|
| AMI, joint, ResNet-50 | frozen tasks, joint path, selection script | ran; rank-1 0.957 |
| IITD, naive, MobileNetV3 | enrolling with training images | ran; rank-1 0.910, forgetting 0.048 |
| AWE, naive, Swin, body CMS 1,2,4 for 20 steps per task | body CMS through the trainer | ran |
| AMI, seed 9 | seed outside the benchmark | refused, as intended |

## Things to watch in R2 and R3

1. **Forgetting may be mild on some cells.** Every evaluation re-enrolls each person with the *current* model, so it measures drift in the features, not lost class weights. The IITD smoke run's 0.048 is below R2's 0.10 threshold. That was one short run, so it proves nothing, but cells may fail the rule. If they do, that gets reported as it is; the rule is not relaxed afterwards.
2. **CMS periods and short runs.** With 60 steps per task (300 total), a copy updating every 256 steps changes only once. R3's decision rule must deal with this before any R3 run.

## How to run

Enqueue `make_r1_tasks.py tune` (runbook section W), run `r1_select.py` on the results, commit `benchmark/recipes.yaml`, then enqueue `make_r1_tasks.py final`.
