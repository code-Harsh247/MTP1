# C1: Continual-learning harness and the forgetting gap (EarVN, P1)

*Date: 2026-10-19. Code: `nlear/continual.py`, `nlear/engine.py`, `scripts/train_continual.py`, `scripts/summarize_continual.py`. Runs: `runs/earvn/continual/{joint,naive}/seed0..4`. Log: `runs/earvn/continual_c1.log`.*

## Protocol P1 (identity-incremental), as implemented

- **Tasks.** The 115 EarVN training identities of each open-set split (seeds 0–4, the same splits as Phases 2–3) are dealt into **5 tasks of 23 people**.
- **Held-out images.** 20% of every training identity's images are held out (2 as gallery, the rest as probes) to measure forgetting of *seen* people. That is about 3,960 held-out images; each task keeps about 3,100 training images.
- **Training.** The Phase 2 config (ResNet-50, ArcFace, regularization). Each task gets **250 steps** (~5 epochs of its images, the length the baseline was tuned to). Joint training gets the same total budget (1,250 steps on all tasks at once).
- **Class-incremental head.** At task t, only the people seen so far take part in the ArcFace softmax (tested: unseen classes get no gradient).
- **Measurements after every task:**
  - **A[t][j]**: open-set identification of task j's people from their held-out images (cosine_proto; a fixed set of 17 never-trained unknown test people, about 2,850 probes).
  - **U[t]**: the Phase 2 open-set protocol on the unseen val/test people.
- **Summary** (`nlear/continual.py`, hand-checked in `tests/test_continual.py`):
  - final average over tasks
  - forgetting: the largest drop since the task was learned
  - backward transfer (BWT)

## Results (5 seeds, mean ± std)

| Method | Seen DIR@5% | Seen rank-1 | Forgetting DIR@5% | Forgetting rank-1 | BWT DIR@5% | Unseen rank-1 | Unseen AUROC | Unseen DIR@5% |
|---|---|---|---|---|---|---|---|---|
| **Joint** (upper bound) | **0.491 ± 0.037** | **0.802 ± 0.010** | — | — | — | **0.527 ± 0.068** | **0.685 ± 0.028** | **0.162 ± 0.034** |
| **Naive** fine-tuning (lower bound) | 0.181 ± 0.035 | 0.531 ± 0.011 | 0.170 ± 0.015 | 0.262 ± 0.019 | −0.170 ± 0.015 | 0.412 ± 0.058 | 0.605 ± 0.042 | 0.090 ± 0.023 |

Mean accuracy matrix for naive training, DIR@5%. Rows show the stage after task t was learned; columns are tasks j. The diagonal is "just learned".

| | task 0 | task 1 | task 2 | task 3 | task 4 |
|---|---|---|---|---|---|
| after task 0 | **0.185** | 0.032 | 0.031 | 0.042 | 0.028 |
| after task 1 | 0.058 | **0.271** | 0.043 | 0.044 | 0.041 |
| after task 2 | 0.078 | 0.114 | **0.343** | 0.066 | 0.056 |
| after task 3 | 0.093 | 0.110 | 0.149 | **0.362** | 0.067 |
| after task 4 | 0.101 | 0.097 | 0.134 | 0.151 | **0.421** |

## Reading

1. **The forgetting is large and consistent (C1 "done" criterion met).** Naive sequential training ends at a seen-people DIR@5% of 0.18 against 0.49 for joint training. Each task loses about 0.17 DIR@5% and 0.26 rank-1 after it has been learned. The std across seeds is small (0.015), so the effect is clear, not noise.
2. **The pattern is classic catastrophic forgetting.** Each new task is learned well (the diagonal rises from 0.19 to 0.42 as the backbone keeps improving), and the previous task immediately drops by a factor of 2–5. Example: task 1 goes 0.27 → 0.11 one stage later.
3. **Forgetting also hurts unseen people.** Unknown rejection on never-seen people drops from AUROC 0.685 (joint) to 0.605 (naive), and DIR@5% from 0.162 to 0.090. Each narrow task over-specializes the embedding, which is the same effect as the "peak at ~5 epochs" in Phase 2. This matters for the supervisor's open-set requirement.
4. **There is plenty of room for a continual-learning method.** On seen people the gap between naive and joint is 0.31 DIR@5% (and 0.27 rank-1); on unseen people it is 0.07 DIR@5%.

*Side note:* unseen DIR@5% for joint training (0.162) is lower than the Phase 2 baseline (0.209), because 20% of the training images are now held out and the budget is 1,250 steps instead of about 1,550. This is fine, since every C-phase method uses the same protocol.

## Next: C2 (baselines)

Add **EWC**, **LwF / feature distillation** and **experience replay** (2 and 5 images per old person) to `train_continual.py` through the engine's `extra_loss` and `batch_fn` hooks. Tune their strength on the **validation** people only.
