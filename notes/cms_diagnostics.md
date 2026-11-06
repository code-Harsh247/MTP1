# Why CMS did not reduce forgetting: diagnostics

*Run 2026-11-06 on Kaggle (2 × Tesla T4), EarVN ConvNeXt-T, seed 1: naive, R3 CMS (periods 1, 16, 256) and R4 CMS (1, 500, 1000), each saving its weights after every task. Scripts: `scripts/run_cms_diagnostics.py`, `scripts/diagnose_cms.py`. Raw tables: `runs_kaggle/diag/earvn/continual/*/seed1/diag.md`. In every run the reloaded weights reproduced the logged accuracies exactly.*

**One model, one dataset, one seed.** These results explain this cell; they are not a measurement across the study.

## 1. In the plain model, forgetting sits in the pointwise layers

Task 1 rank-1 falls from 0.885 to 0.493. Putting one part back to its after-task-1 weights:

| Part put back | Task 1 rank-1 | Share of the loss recovered |
|---|---|---|
| Pointwise layers (the MLPs CMS replaces) | 0.856 | **93%** |
| Neck | 0.608 | 29% |
| Rest of the backbone | 0.564 | 18% |

So CMS was placed where the forgetting happens. Putting the MLPs back also drops the last task from 0.93 to 0.43, so they hold the new people too.

## 2. With CMS, the forgetting moves into the fast copy

| Part put back | R3 share recovered | R4 share recovered |
|---|---|---|
| Copy 1 (every step) | 47% | **69%** |
| Copy 2 | 28% | 5% |
| Copy 3 | 6% | 5% |
| Mixing weights | 1% | 1% |
| Neck | 25% | 24% |
| Rest of the backbone | 17% | 24% |

## 3. The slow copies barely learn anything

Relative change from the pretrained weights after all 5 tasks:

| | Copy 1 | Copy 2 | Copy 3 |
|---|---|---|---|
| R3 | 0.097 | 0.041 | **0.010** |
| R4 | 0.100 | **0.007** | **0.004** |

After task 1, R3's copy 3 and R4's copy 2 had each moved only 0.003. Used alone, the slow copies score 0.29–0.33 on the **last** task, against 0.85–0.89 for the fast copy. They learned little of the newest people, so they had little to keep of the old ones either.

**Likely cause: the optimizer.** NL's rule (Eq. 71) applies the *sum* of a period's gradients, which with plain gradient descent is about as large as C ordinary steps. We step each slow level with AdamW, which normalizes the update, so one slow step moves about as far as one ordinary step: a period-C copy learns roughly 1/C as much. The slow-lr grid (`lr_pow` 0 or 0.5, so up to √C) does not make up for that. Scaling by C (`lr_pow` 1), or plain gradient descent for the slow levels, would match the paper's rule, but has not been tested.

## 4. The model leans more on the fast copy over time

Mean mixing weight of copy 1 grows from 0.333 to 0.388 (R3) and 0.398 (R4), while the slow copies fall to about 0.30. The drift is small, and putting the mixing weights back changes task 1 by only 1%. So the model is not escaping the slow memory; the slow memory is simply too weak.

## What this means

| Hypothesis (from the R3 discussion) | Verdict on this cell |
|---|---|
| Slow copies never learned the old people | **Supported:** they moved 0.4–1% against 10% for the fast copy |
| No consolidation from fast to slow | **Consistent:** nothing carries copy 1's learning into the slow copies |
| The model routes around slow memory | **Mostly ruled out:** the mixing weights barely moved |
| Forgetting happens outside CMS | **Ruled out:** 93% of it is in the pointwise layers |

**The failure is in how the slow copies are updated, not in where CMS sits.** The next experiment is slow levels updated with the summed gradient at full size (`lr_pow` 1, or SGD), or with an explicit fast-to-slow consolidation at each task boundary. It would need its own pre-registration.
