# U3: Continual learning at UERC scale (replication of the C3b slow-backbone effect)

*This file, `scripts/u3_report.py` and `scripts/dgx/make_u3_tasks.sh` were committed **before** any U3 run (2026-10-20). Results are appended below.*

## Hypothesis (from EarVN, now replicated at scale)

On EarVN, a **slow-updating backbone** improved **final seen-people rank-1** over the *identical* architecture at the *same* precision, both with replay k=5. The slow backbone updates every 16 steps on the summed gradient, with its lr × 16^0.5; this is the minimal two-level NL frequency split ([NL] Eq. 71). The result:

- C3, seeds 1–5: D+R vs ctrl+R, +0.041 rank-1, better on 5/5 seeds;
- C3b, fresh seeds 6–10: slowbb+R vs ctrl+R, +0.038 rank-1, better on 5/5 seeds.

That was a supporting result, not a pre-registered claim. U3 tests it as a pre-registered claim on UERC.

## Protocol

- **Data:** UERC P1 (identity-incremental), 917 training identities dealt into **10 tasks of about 92**. 20% of each training identity's images are held out (at least 3) to measure seen-person performance. Unseen people are the val/test identities, as in U1.
- **Training:** the U1 recipe (ResNet-50 + ArcFace, regularization, 128×96). **870 steps per task** (about 4 epochs of each task's images, matching U1's tuned 4 epochs); joint training gets 8,700 steps on all data.
- **Precision:** **all arms in fp32** (`plan_uerc.md` rule; C3b found that precision alone shifts forgetting).
- **Seeds:** **1–3**. **No hyperparameter is re-tuned:** the slow-backbone setting is exactly the one from C3/C3b. That is the point of a replication, so seed 0 is not needed.

## Arms (`scripts/dgx/make_u3_tasks.sh`)

| Arm | Flags | Role |
|---|---|---|
| **ctrl+R** | replay k=5, `--cms-periods 1,1,1` | **primary comparison** (same architecture, all levels every step) |
| **slowbb+R** | replay k=5, `--cms-periods 1,1,1 --cms-body-period 16 --cms-lr-pow 0.5` | **primary comparison** (slow backbone) |
| D+R | replay k=5, `--cms-periods 1,16,256 --cms-body-period 16 --cms-lr-pow 0.5` | secondary (adds the CMS block chain) |
| replay | replay k=5, plain neck, `--fp32` | reference |
| naive | `--fp32` | lower reference |
| joint | `--fp32` | upper reference |

**Deviation from `plan_uerc.md`:** EWC and LwF are not run. C2 showed that replay dominates them, and neither bears on this claim.

## Decision rule

- **Primary:** **final seen rank-1** (the mean over the 10 tasks of rank-1 after the last task), **slowbb+R vs ctrl+R**, paired per seed.
- **CONFIRMED** if slowbb+R is higher on **3 of 3** seeds: the slow-backbone retention effect replicates at scale. Otherwise **NOT CONFIRMED**.

**Reported, not decision criteria:**

- forgetting (rank-1 and DIR@5%), final seen DIR@5%;
- unseen-people AUROC, DIR@5% and rank-1;
- D+R vs slowbb+R (does the CMS chain add anything at scale?);
- slowbb+R vs plain replay, and the naive/joint gap.

## Results (2026-10-20; DGX array 34954, 18 runs, 0 failures; `runs_dgx/u3_report.txt`)

The rule, report script, task list and training code were unchanged since the pre-registration commit `15fdf08` (checked with `git diff`).

**Test, UERC P1 (10 tasks), seeds 1–3, all arms fp32:**

| Arm | Seen rank-1 | Forget rank-1 ↓ | Seen DIR@5% | Forget DIR@5% ↓ | Unseen rank-1 | Unseen AUROC | Unseen DIR@5% |
|---|---|---|---|---|---|---|---|
| joint | **0.899 ± 0.002** | — | **0.764 ± 0.001** | — | **0.639** | **0.774** | **0.352** |
| naive | 0.595 ± 0.011 | 0.304 | 0.283 | 0.393 | 0.444 | 0.652 | 0.158 |
| replay | 0.720 ± 0.009 | 0.124 | 0.423 | 0.184 | 0.476 | 0.682 | 0.181 |
| **ctrl+R** | **0.728 ± 0.005** | **0.116** | 0.421 | **0.179** | 0.488 | 0.685 | 0.187 |
| **slowbb+R** | 0.713 ± 0.011 | 0.138 | 0.404 | 0.202 | 0.469 | 0.669 | 0.176 |
| D+R | 0.717 ± 0.003 | 0.130 | 0.409 | 0.186 | 0.464 | 0.680 | 0.173 |

**Decision** (primary: final seen rank-1, slowbb+R vs ctrl+R, paired): better on **0/3** seeds, mean **−0.015 ± 0.007** (per seed −0.009, −0.011, −0.025).

### VERDICT: **NOT CONFIRMED**

The effect **reverses** at scale. Against the identical same-precision control, the slow backbone is worse on **every** reported metric in 0/3 seeds:

| Metric | Difference vs ctrl+R |
|---|---|
| seen rank-1 | −0.015 |
| forgetting (rank-1) | 0.022 worse |
| seen DIR@5% | −0.017 |
| unseen AUROC | −0.016 |
| unseen DIR@5% | −0.011 |

The EarVN finding (+0.04 rank-1, 5/5 seeds in two rounds) does not generalize.

### Reported observations

- **The CMS block chain on top of the slow backbone** (D+R vs slowbb+R) recovers part of the loss: forgetting is better in 3/3 seeds, as is unseen AUROC. D+R still stays below ctrl+R.
- **Replay is very effective at scale:** seen rank-1 0.720 vs 0.595 for naive, and forgetting 0.124 vs 0.304. The plain same-architecture fp32 control (ctrl+R) is slightly better still (+0.008 rank-1, 3/3). That is a small architecture/precision effect, not an NL one.
- **Joint training remains far ahead** (seen rank-1 0.90, DIR@5% 0.76 vs 0.72 and 0.42), so the continual-learning gap stays large. It also hurts unseen people: unseen AUROC is 0.77 for joint vs 0.68 for replay.

### Interpretation

- The EarVN setting had 5 tasks of 23 people and 250 steps per task, so the slow backbone updated only about 15 times per task. That likely acted as a strong regularizer, which helped a small, overfitting-prone backbone retain old people.
- At UERC scale (10 tasks of about 92 people, 870 steps per task) the backbone *needs* to keep learning. Slowing it costs more plasticity than it saves in stability.
- **With this, none of the NL mechanisms tested in this project gives a robust improvement over standard methods:** surprise memory, meta-learned memory, the CMS block chain, and the slow-backbone frequency split. The positive EarVN effect was real on that data, but setting-dependent.
