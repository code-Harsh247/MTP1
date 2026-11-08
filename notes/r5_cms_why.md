# R5 results (exploratory): why CMS did not reduce forgetting

Design: `benchmark/r5_design.yaml` (committed 31373ed before any run). Kaggle session, 18 of 18 tasks
finished (`logs_kaggle_r5/`, only a timm warning). Raw files: `runs_kaggle/r5/` (15 runs) and the
new folders in `runs_kaggle/diag/`. Tables: `notes/r5_tables.md` (`scripts/r5_report.py`).
Cross-check: forgetting and final rank-1 recomputed from every raw A matrix, 0 mismatches.

The two explanations under test:

- **A. The slow update is too small.** One AdamW step per period moves a slow copy about as far as one
  ordinary step.
- **B. The memory is diluted.** The layer outputs a weighted sum of the copies, and the neck and the
  rest of the backbone forget outside CMS.

## Experiment 1: full-size slow updates (tests A). Verdict: A is not the cause

EarVN, seeds 1–3, against naive with R3's rule.

| Model | Variant | Naive forg. | R3 forg. | R5 forg. | Seeds better | Naive final | R5 final | Verdict |
|---|---|---|---|---|---|---|---|---|
| ResNet-50 | lr_pow 1 | 0.411 | 0.409 | 0.420 | 1 of 3 | 0.553 | 0.501 | not reduced |
| ResNet-50 | per-step | 0.411 | 0.409 | 0.445 | 0 of 3 | 0.553 | 0.535 | not reduced |
| ConvNeXt-T | lr_pow 1 | 0.407 | 0.371 | 0.233 | 3 of 3 | 0.597 | 0.130 | not reduced (diverged) |
| ConvNeXt-T | per-step | 0.407 | 0.371 | 0.435 | 0 of 3 | 0.597 | 0.582 | not reduced |

- **Per-step works as intended, but forgetting gets worse.**
  - The slow copies now move further than copy 1. On ConvNeXt seed 1 the movement from pretrained is
    0.154 for copy 2 and 0.148 for copy 3, against 0.091 for copy 1 (R3: 0.041 and 0.010).
  - Forgetting nonetheless rises above naive in both cells, and no seed is better.
  - The slow copies learn the newest people instead of keeping the old ones. Alone, copy 2 scores
    0.851 on the last task but only 0.358 on task 1. Copy 1 alone scores 0.537 on task 1.
  - A slow period by itself does not make a copy remember: given full-size updates, a slow copy
    follows the current task like the fast one does.
- **lr_pow 1 is not a valid result on ConvNeXt.**
  - Training diverged on all three seeds: the loss went from about 4.4 to 11.6, tasks 2–5 stayed
    near 10% rank-1, and copy 3 moved 2.57 from pretrained.
  - Its low forgetting (0.233) comes from never learning the later tasks, and the final-accuracy
    condition of the rule correctly rejects it.
  - On ResNet-50 it trained, but did not reduce forgetting.

## Experiment 3: a perfect slow memory (tests B). Verdict: helps, but most of task 1 is still lost

EarVN ConvNeXt-T. Copies 2–3 are set to copy 1 at the end of task 1 and then frozen.
This is an oracle that uses the task boundary, not a method.

| | Naive | Snapshot |
|---|---|---|
| Task-1 forgetting (mean of 3 seeds) | 0.445 | 0.272 (better on 3 of 3) |
| Forgetting, all tasks | 0.407 | 0.317 (better on 3 of 3) |
| Final average rank-1 | 0.597 | 0.636 |

- By the numbers this would pass R3's REDUCED test: a drop of 0.09, all seeds better, final not lower.
  It is far from FIXED, because forgetting is 0.317, not under 0.10.
- **The memory is intact but does not reach the output** (seed 1 diagnostics):
  - Alone, a frozen copy scores 0.787 on task 1. The whole model scores 0.577.
  - Task 1 drops from 0.801 to 0.532 right after task 2, while two of three copies hold the task-1
    weights exactly.
  - Putting copy 1 alone back to its after-task-1 weights recovers 94% of the lost task-1 loss.
    The frozen copies hold the memory, but the moving copy (mixing weight 0.41), the neck and the
    rest of the backbone override it in the summed output.
  - The neck alone recovers 35% and the rest of the backbone 47%. These shares overlap and do not
    add up to 100%.
- **Cost of the oracle:**
  - Task 1 is learned less well (0.801 against naive's 0.885), because only copy 1 trains on it while
    copies 2–3 still hold pretrained weights.
  - Part of the smaller task-1 drop therefore comes from starting lower.

## Experiment 2: diagnostics of the new arms (EarVN ConvNeXt-T, seed 1)

| Arm | Task 1: after task 1 → end | Movement of copies 1/2/3 | Each copy alone, task 1 |
|---|---|---|---|
| R3 (earlier) | 0.877 → 0.460 | 0.097 / 0.041 / 0.010 | 0.42 / 0.28 / 0.39 |
| R5 per-step | 0.891 → 0.490 | 0.091 / 0.154 / 0.148 | 0.54 / 0.36 / 0.33 |
| R4b consolidation | 0.879 → 0.503 | 0.099 / 0.084 / 0.036 | 0.37 / **0.52 / 0.64** |
| R5 snapshot | 0.801 → 0.577 | 0.100 / 0.054 / 0.054 | 0.34 / **0.79 / 0.79** |
| R5 lr_pow 1 | 0.837 → 0.115 | 0.096 / 0.330 / 2.565 | diverged |

- In R4b too, a single slow copy remembers task 1 better than the whole model: copy 3 alone gets
  0.64 against 0.503. This is the same dilution seen in the snapshot run.

## Experiment 5: other cells (naive and R3, seed 1)

| Cell | Arm | Task 1: after task 1 → end | Part holding most of the lost task-1 loss |
|---|---|---|---|
| EarVN DeiT-S | naive | 0.874 → 0.368 | rest of backbone (62%) |
| EarVN DeiT-S | R3 | 0.853 → 0.341 | rest of backbone (79%) |
| AWE Swin-T | naive | 0.786 → 0.500 | pointwise layers (50%) |
| AWE Swin-T | R3 | 0.857 → 0.429 | rest of backbone (over 100%, shares overlap) |

- The pattern differs from the CNN. On the EarVN ConvNeXt, the pointwise layers held 93% of naive's
  lost loss.
- On the transformers, most of the forgetting sits in the parts CMS does not cover: attention,
  normalisation and embeddings.
- In R3 the slow copies again barely move: 0.065 and 0.019 against 0.162 on DeiT-S, and 0.041 and 0.011
  against 0.078 on Swin.

## What this says

1. **A is refuted as a sufficient cause.**
   - Making the slow updates full-size made the slow copies move, but they then learned the new tasks
     and forgetting rose.
   - The update rule alone does not protect a copy. A slow period only delays when a copy follows the
     current task.
2. **B is supported.**
   - Even a perfect frozen memory of task 1 cuts task-1 forgetting by only about 40% (0.445 to
     0.272). The weighted sum lets the moving copy, the neck and the rest of the backbone override it.
   - The same dilution appears in R4b.
3. **CMS covers the wrong parts on transformers.** Most of their forgetting is outside the pointwise
   layers.

**Caveats:**

- Exploratory and designed after R3–R4b.
- The diagnostics use a single seed (seed 1).
- Experiment 3 is an oracle.
- Experiment 1 covers two CNN cells on one dataset.
