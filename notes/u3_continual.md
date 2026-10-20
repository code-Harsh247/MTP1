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

## Results

*(appended after the runs)*
