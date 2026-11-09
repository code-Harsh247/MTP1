# R5b results (exploratory): does the dilution result hold, and is the blend the bottleneck?

Design: `benchmark/r5b_design.yaml`, committed 370b001 before any run. Kaggle session: all 48 task
lines finished with no tracebacks (`logs_kaggle_r5b/`). Raw files: `runs_kaggle/r5b/` (44 runs) and
7 new `diag.json` files in `runs_kaggle/diag/`. Tables: `notes/r5b_tables.md` (`scripts/r5b_report.py`).
Cross-check: forgetting and final rank-1 recomputed from every raw A matrix, 0 mismatches. A separate
script recomputed the inputs to every verdict (drop, seeds better, final, joint), and all verdicts agree.

## Experiment 1: the snapshot oracle on all 9 shown cells

Copies 2–3 are set to copy 1 at the end of task 1 and then frozen; the mixing weights stay learned.
The EarVN ConvNeXt-T row is R5's.

| Cell | Task-1 forg., naive → snapshot | Seeds better (task 1) | Forgetting, naive → snapshot | Final, naive → snapshot | R3 test |
|---|---|---|---|---|---|
| EarVN ConvNeXt-T | 0.445 → 0.272 | 3 of 3 | 0.407 → 0.317 | 0.597 → 0.636 | REDUCED |
| EarVN ResNet-50 | 0.432 → 0.268 | 3 of 3 | 0.411 → 0.338 | 0.553 → 0.578 | REDUCED |
| EarVN MobileNetV3 | 0.287 → 0.264 | 2 of 3 | 0.329 → 0.306 | 0.536 → 0.525 | not reduced |
| EarVN DeiT-S | 0.525 → 0.438 | 3 of 3 | 0.476 → 0.444 | 0.524 → 0.538 | not reduced |
| EarVN Swin-T | 0.502 → 0.436 | 3 of 3 | 0.463 → 0.430 | 0.546 → 0.563 | not reduced |
| AWE ConvNeXt-T | 0.329 → 0.243 | 4 of 5 | 0.232 → 0.221 | 0.529 → 0.520 | not reduced |
| AWE DeiT-S | 0.386 → 0.314 | 3 of 5 | 0.271 → 0.243 | 0.340 → 0.369 | not reduced |
| AWE Swin-T | 0.443 → 0.271 | 4 of 5 | 0.286 → 0.246 | 0.426 → 0.417 | not reduced |
| AMI DeiT-S | 0.157 → 0.086 | 4 of 5 | 0.161 → 0.079 | 0.843 → 0.903 | FIXED |

- **A perfect memory of task 1 helps task 1 in every cell, but only partly.**
  - Task-1 forgetting falls in all 9 cells. The size varies:
    - about 0.16–0.17 on the EarVN ConvNeXt-T, EarVN ResNet-50 and AWE Swin-T;
    - 0.07–0.09 on the other transformers and AWE ConvNeXt-T;
    - 0.02 on MobileNetV3.
  - It never comes close to zero on EarVN or AWE: the best case still loses 0.27 of task-1 rank-1.
- **Against R3's test, the oracle reduces overall forgetting in 2 cells (the EarVN CNNs) and fixes 1.**
  - The fixed cell is AMI DeiT-S: 0.161 → 0.079, all 5 seeds better, final 0.903 against joint 1.000.
  - This oracle uses the task boundary and freezes after task 1 only, so it is an upper bound, not
    a method. AMI is also the near-ceiling dataset with the mildest forgetting.

### Diagnostics (seed 1): is the kept memory diluted?

Prestated reading: dilution generalises if, in each diagnosed cell, a frozen copy used alone beats
the whole model on task 1 by at least 0.10.

| Cell | Task 1 at end: whole model | Frozen copy alone | Gap | Part that recovers most of the lost task-1 accuracy |
|---|---|---|---|---|
| EarVN ConvNeXt-T (R5) | 0.577 | 0.787 | +0.21 | copy 1 (94%) |
| EarVN ResNet-50 | 0.441 | 0.573 | +0.13 | rest of backbone (64%) |
| EarVN DeiT-S | 0.365 | 0.469 | +0.10 | rest of backbone (104%) |
| AWE Swin-T | 0.429 | 0.429 | 0.00 | rest of backbone (125%) |

- **Dilution holds in 3 of the 4 diagnosed cells, but not on AWE Swin-T.** So by the prestated
  reading it does not generalise to every cell. It holds on the EarVN CNNs and, only just, on EarVN
  DeiT-S.
  - On AWE Swin-T the frozen copy alone does no better than the whole model.
  - There the pointwise copies do not matter: restoring any copy, or the mixing weights, recovers 0%.
  - All the forgetting is in the rest of the backbone. AWE's task-1 test has 14 probes, so one probe
    is 0.07.
- On every cell except the EarVN ConvNeXt-T, the part that recovers most is the rest of the backbone,
  not the pointwise copies. Dilution is real where it occurs, but outside the ConvNeXt the forgetting
  outside CMS is the larger effect. On the transformers this matches R5's finding.

## Experiment 2: fixing the frozen copies' share of the output after task 1

Mean of 3 seeds, EarVN. "learned" is the snapshot with learned mixing; its frozen share ends near 0.6.

| Model | Arm | Task-1 forg. | Forgetting | Last-task rank-1 | Final rank-1 |
|---|---|---|---|---|---|
| ConvNeXt-T | naive | 0.445 | 0.407 | 0.945 | 0.597 |
| ConvNeXt-T | learned (≈0.6) | 0.272 | 0.317 | 0.929 | 0.636 |
| ConvNeXt-T | 0.8 | 0.228 | 0.289 | 0.913 | 0.639 |
| ConvNeXt-T | 0.95 | 0.171 | 0.258 | 0.872 | 0.632 |
| DeiT-S | naive | 0.525 | 0.476 | 0.928 | 0.524 |
| DeiT-S | learned (≈0.64) | 0.438 | 0.444 | 0.928 | 0.538 |
| DeiT-S | 0.8 | 0.445 | 0.443 | 0.919 | 0.532 |
| DeiT-S | 0.95 | 0.434 | 0.445 | 0.905 | 0.520 |

- **ConvNeXt-T: giving the old memory more of the output steadily cuts forgetting, at a cost.**
  - Task-1 forgetting falls from 0.272 to 0.228 to 0.171. At 0.95 it is lower than learned mixing
    on every seed (0.148, 0.224 and 0.142).
  - Last-task rank-1 falls from 0.929 to 0.872, so final average rank-1 stays about the same (0.632–0.639).
  - The prestated threshold was not met: at 0.95, task-1 forgetting is 0.171, more than half of 0.272.
    The blend is part of the bottleneck, but not all of it.
- **DeiT-S: no effect**, as predicted (0.438, 0.445 and 0.434). Its forgetting lies outside the
  pointwise layers.
- **Where the rest of the forgetting goes** (ConvNeXt-T, share 0.95, seed 2):
  - The whole model scores 0.576 on task 1, against 0.500 for the learned-mixing snapshot.
  - Restoring the neck alone recovers 93%, and the rest of the backbone 91%.
  - With the blend fixed, the neck and the rest of the backbone take over the forgetting.
  - The frozen copy alone scores 0.694, lower than with learned mixing (0.825), because the neck
    and the backbone have adapted to the new tasks.

## Experiment 3: a second diagnostic seed (EarVN ConvNeXt-T, seed 2)

| Arm | Task 1 after task 1 → end | Main location (seed 2) | Seed 1 |
|---|---|---|---|
| naive | 0.900 → 0.378 | pointwise layers 86% | 93% |
| R3 CMS | 0.878 → 0.389 | copy 1 47% | 47% |
| snapshot | 0.802 → 0.500 | copy 1 108%; frozen copy alone 0.825 (gap +0.33) | 94%; gap +0.21 |
| snapshot, share 0.95 | 0.802 → 0.576 | neck 93%, rest of backbone 91%; gap +0.12 | — |

- R5's seed-1 diagnostics replicate on seed 2: the pointwise layers hold the forgetting in the plain
  ConvNeXt-T, copy 1 holds it with R3, and the snapshot shows a large dilution gap.

## What R5b says

1. **On the CNN where CMS sits at the right place (EarVN ConvNeXt-T), the R5 story holds on a
   second seed.**
   - The old memory is kept but diluted.
   - Giving it a larger share of the output cuts task-1 forgetting further (0.27 → 0.17), but costs
     accuracy on new people.
   - The remaining forgetting moves to the neck and the rest of the backbone.
2. **Across all 9 cells, even a perfect frozen memory of task 1 in the pointwise layers removes only
   part of the forgetting.**
   - It reduces overall forgetting by R3's test in 2 cells and fixes 1 (AMI DeiT-S).
   - Elsewhere, especially on transformers, the forgetting is mostly outside the pointwise layers,
     so a CMS memory there cannot prevent it, however it is updated or combined.
3. **Dilution is one cause, not the whole explanation.** It holds on 3 of the 4 diagnosed cells. The
   larger, more general cause is that the parts CMS does not cover (the neck and the rest of the
   backbone) forget on their own.

**Caveats:**

- Exploratory.
- The snapshot is an oracle that knows the task boundary and only protects task 1.
- Diagnostics come from one or two seeds.
- AWE and AMI task tests are tiny.
