# C3: CMS neck (Continuum Memory System) vs the C2 bars

*This decision procedure was written and committed **before** any C3 result existed (2026-10-20). Results are appended below it. It is applied mechanically by `scripts/c3_report.py`.*

## Method

`nlear/models.py` `CMSNeck` + `nlear/cms.py` `CMSScheduler` implement [NL] Eq. 70–71 (sequential variant, Eq. 73):

- **Structure.** The ResNet-50 features go through a projection, then a chain of residual MLP blocks (levels), then BN.
- **Update periods.** Block ℓ updates only every C⁽ℓ⁾ steps, using the gradient summed over that window. Its own AdamW persists across tasks, because slow levels are the long-term memory.
- **Initialization.** New blocks start as the identity, so CMS starts out equal to the plain neck.
- **Training.** fp32; the control runs in fp32 too.
- **Unit tests** (`tests/test_cms.py`) check:
  - a slow level changes only on its period;
  - its update equals one AdamW step on the summed gradient;
  - all-period-1 means no slow optimizer.

**Variants** (`scripts/dgx/make_c3_tasks.sh`), each run with naive training and with replay k=5:

| ID | Block periods | Backbone period | Slow-level lr | Role |
|---|---|---|---|---|
| ctrl | 1, 1, 1 | 1 | — | **Control:** the same architecture and precision, no multi-frequency |
| A | 1, 16, 256 | 1 | = base | CMS |
| B | 1, 16, 256 | 1 | base × period^0.5 | CMS |
| C | 1, 16, 256 | 16 | = base | CMS, backbone also slow |
| D | 1, 16, 256 | 16 | base × period^0.5 | CMS, backbone also slow |
| E | 1, 4, 16, 64 | 1 | base × period^0.5 | CMS, 4 levels, faster spectrum |

All 12 configurations run on seeds 0–5 (72 runs, one DGX submission).

## Selection (seed 0 only)

- **Without replay:** among A–E, pick the variant with the highest **final seen DIR@5%** on seed 0. Call it CMS\*.
- **With replay:** the same rule among A–E + replay. Call it CMS\*+R.
- The control is never selectable.
- Only CMS\* and CMS\*+R's seeds 1–5 count as test results. The other variants' seeds 1–5 are listed for transparency but carry no claim.

## Decision rule (from plan_continual.md §5, with the C2 bars)

All comparisons are **paired per seed, on seeds 1–5**. "Wins" means better on at least 4 of 5 seeds.

- **(a) No stored data.** CMS\* wins against **naive** *and* against **EWC λ=10²** (the C2 rule pick) on **both**:
  - lower forgetting (DIR@5%), and
  - higher final seen DIR@5%.
- **(b) With replay.** CMS\*+R wins against **replay k=5** on final seen DIR@5%.
- **Attribution to NL.** Whichever of (a) or (b) holds, the CMS arm must also win against its **same-architecture control** (ctrl, or ctrl+R) on final seen DIR@5%. Otherwise the gain comes from the extra neck capacity, not from the multi-frequency updates.

**Outcome:**

- **GO:** (a) or (b) holds, *with* attribution.
- **Partial:** (a) or (b) holds *without* attribution. The improvement is real but not due to NL.
- **NO-GO:** neither holds. Per the plan, the ViT + CMS variant then gets one try before reporting a negative result.

The unseen-people open-set metrics are reported in every case.

## Results (2026-10-20; DGX Slurm array 34930, 72 runs; `runs_dgx/c3_report.txt`)

The decision procedure ran exactly as committed (`scripts/c3_report.py`, unchanged since `7e535dd`).

**Seed-0 selection** (final seen DIR@5%):

- Without replay: A 0.156, B 0.145, C 0.117, D 0.155, E 0.125, so **CMS\* = A**.
- With replay k=5: A 0.155, B 0.151, C 0.111, **D 0.186**, E 0.147, so **CMS\*+R = D**.

**Test results, seeds 1–5:**

| Method | Seen DIR@5% | Forgetting DIR@5% | Seen rank-1 | Unseen DIR@5% | Unseen AUROC |
|---|---|---|---|---|---|
| naive | 0.183 ± 0.037 | 0.180 ± 0.022 | 0.536 ± 0.021 | 0.089 ± 0.036 | 0.597 ± 0.050 |
| EWC λ=10² | 0.185 ± 0.038 | 0.171 ± 0.019 | 0.549 ± 0.013 | 0.077 ± 0.031 | 0.585 ± 0.055 |
| ctrl (same architecture) | 0.178 ± 0.025 | 0.179 ± 0.017 | 0.531 ± 0.017 | 0.081 ± 0.019 | 0.613 ± 0.039 |
| **CMS\* = A** | 0.180 ± 0.024 | 0.192 ± 0.020 | 0.530 ± 0.018 | 0.088 ± 0.027 | 0.610 ± 0.039 |
| replay k=5 | **0.221 ± 0.010** | 0.071 ± 0.016 | 0.583 ± 0.014 | 0.083 ± 0.033 | 0.600 ± 0.015 |
| ctrl + replay | 0.211 ± 0.014 | 0.051 ± 0.003 | 0.584 ± 0.021 | 0.089 ± 0.025 | 0.610 ± 0.028 |
| **CMS\*+R = D + replay** | 0.215 ± 0.016 | **0.029 ± 0.014** | **0.625 ± 0.023** | **0.111 ± 0.021** | **0.646 ± 0.015** |
| joint (upper bound) | 0.501 ± 0.033 | — | 0.809 ± 0.007 | 0.181 ± 0.052 | 0.708 ± 0.030 |

**Decision** (paired, seeds 1–5):

- (a) CMS\* vs naive: forgetting lower in 1/5, seen DIR@5% higher in 1/5. CMS\* vs EWC: 0/5 and 2/5. **(a) fails.**
- (b) CMS\*+R vs replay k=5: seen DIR@5% higher in 1/5 (mean −0.006). **(b) fails.**
- **Verdict: NO-GO** on the pre-registered criterion. CMS does not improve the final open-set identification of seen people (DIR@5%) over naive, EWC, or replay.

## Exploratory findings (NOT pre-registered; hypotheses, not claims)

Paired over seeds 1–5 (`runs_dgx/c3_exploratory.txt`), **D + replay vs replay k=5**:

| Metric | Seeds better | Mean improvement |
|---|---|---|
| Forgetting (DIR@5%) | **5/5** | 0.042 ± 0.021 lower |
| Seen rank-1 | **5/5** | +0.043 ± 0.018 |
| Forgetting (rank-1) | **5/5** | 0.054 ± 0.009 lower |
| Unseen AUROC | **5/5** | +0.046 ± 0.022 |
| Unseen DIR@5% | 4/5 | +0.028 ± 0.034 |
| Seen DIR@5% (the primary metric) | 1/5 | −0.006 ± 0.010 |

D + replay is also better than its same-architecture control (ctrl + replay) in 5/5 seeds on forgetting, seen rank-1, unseen AUROC and unseen DIR@5%.

**Reading.**

- CMS with a *slow backbone* plus replay retains seen people better (lower forgetting, higher rank-1). It is also the **first method in C2–C3 that improves unseen-people open-set metrics**: unseen AUROC 0.646 vs 0.600, where every C2 method stayed at naive's level. Its seen-people *open-set* score at FAR 5% (DIR@5%), however, does not move.
- One explanation: the improvement is in how well the embedding ranks the right person first (rank-1, AUROC), but not enough to change acceptance at the strict 5% false-accept threshold with only 2 gallery images.

**Why this is not a claim yet:**

1. These metrics were read after seeing the data (7 metrics × 2 references), so chance findings are possible.
2. **Confound:** variant D changes two things, the CMS levels *and* a slow backbone (period 16, lr × 16^0.5). Variant C (slow backbone, no lr scaling) is much worse, so the interaction matters. A "slow backbone only, no CMS levels" control is missing.
3. Only 5 seeds.

## Proposed confirmatory test (C3b), to be pre-registered before running

- **Fresh seeds 6–10.** New identity splits; seeds 0–5 stay untouched.
- **Primary metrics fixed in advance:** forgetting (DIR@5%), seen rank-1 and unseen AUROC.
- **Arms:**
  - replay k=5
  - D + replay
  - **slow-backbone-only + replay** (`--cms-periods 1,1,1 --cms-body-period 16 --cms-lr-pow 0.5`), the missing control
  - ctrl + replay
- **Claim** if D + replay beats replay **and** slow-backbone-only + replay on all three metrics in at least 4/5 seeds.
- **Cost:** one DGX round of 20 runs, about 15 minutes.
