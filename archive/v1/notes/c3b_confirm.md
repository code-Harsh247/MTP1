# C3b: Confirmatory test of the C3 exploratory finding

*This file, `scripts/c3b_report.py` and `scripts/dgx/make_c3b_tasks.sh` were committed **before** any C3b run existed (2026-10-20). Results are appended below.*

## The hypothesis being tested

In C3, read *after* seeing the data, **variant D + replay k=5** beat replay k=5 on 5/5 seeds for forgetting, seen rank-1 and unseen AUROC. Variant D is the CMS neck with block periods 1/16/256, a **backbone with update period 16**, and slow lr × period^0.5. Two worries remain:

1. The finding could be chance from reading many metrics.
2. Variant D changes two things at once: it adds the CMS block chain *and* slows the backbone.

## Design

- **Fresh data:** **seeds 6–10**, new identity-disjoint splits made for C3b only. Seeds 0–5 are byte-identical to before and are not used here.
- **Everything else is unchanged:** P1 protocol, 5 tasks × 250 steps, the Phase 2 config, and replay with 5 images per learned person in every arm.

| Arm | Flags | What it is |
|---|---|---|
| **R** | `replay`, buffer 5 | plain replay (AMP, as in C2) |
| **ctrl+R** | `--cms-periods 1,1,1` | same CMS architecture, everything updates every step (fp32) |
| **slowbb+R** | `--cms-periods 1,1,1 --cms-body-period 16 --cms-lr-pow 0.5` | **minimal two-level frequency split**: backbone slow (every 16 steps, lr × 4), everything else fast. No CMS block chain |
| **D+R** | `--cms-periods 1,16,256 --cms-body-period 16 --cms-lr-pow 0.5` | the C3 finding: slow backbone **plus** the 3-level CMS block chain |

20 runs (4 arms × 5 seeds), one DGX submission.

## Primary metrics (fixed now)

1. **Forgetting** (DIR@5%; lower is better)
2. **Seen rank-1** (higher is better)
3. **Unseen AUROC**, the never-seen-people protocol (higher is better)

Seen DIR@5% is reported but is *not* primary here: C3 already established that CMS does not move it.

"X beats Y" means X is better than Y on **all three** primary metrics, each in **at least 4 of 5** seeds, paired per seed.

## Outcomes (all three registered in advance)

- **CONFIRMED (CMS chain):** D+R beats R **and** D+R beats slowbb+R. The CMS block chain adds value beyond a slow backbone. This would be the thesis's positive NL result.
- **CONFIRMED (frequency split only):** D+R beats R, and slowbb+R also beats R, but D+R does **not** beat slowbb+R. The gain comes from separating update speeds (a slow backbone level and fast upper levels, which is NL's core principle, Eq. 71), not from the extra CMS blocks. Still an NL-derived positive result, with a smaller claim.
- **NOT CONFIRMED:** D+R does not beat R. The C3 observation was chance or seed-specific, and C3 stays a clean negative.

ctrl+R is reported as the architecture/precision reference (fp32, same parameters as D+R), not as a decision arm.

## Results (2026-10-20; 20 runs on the DGX; `runs_dgx/c3b_report.txt`, `runs_dgx/c3b_supporting.txt`)

The rule, report script and task list were unchanged since the pre-registration commit `42a50f0` (checked with `git diff`).

**Fresh seeds 6–10, mean ± std:**

| Arm | Forgetting ↓ | Seen rank-1 ↑ | Unseen AUROC ↑ | Seen DIR@5% | Unseen DIR@5% |
|---|---|---|---|---|---|
| R (replay k=5, AMP) | 0.080 ± 0.008 | 0.570 ± 0.013 | 0.619 ± 0.029 | **0.210 ± 0.008** | 0.105 ± 0.010 |
| ctrl+R (CMS arch, fp32) | 0.055 ± 0.011 | 0.580 ± 0.023 | 0.622 ± 0.043 | 0.190 ± 0.010 | 0.096 ± 0.009 |
| slowbb+R | 0.046 ± 0.014 | 0.618 ± 0.023 | **0.630 ± 0.026** | 0.199 ± 0.019 | 0.112 ± 0.020 |
| D+R | **0.034 ± 0.016** | **0.626 ± 0.019** | 0.628 ± 0.043 | 0.203 ± 0.018 | **0.113 ± 0.020** |

**Pre-registered decision** (paired; "beats" means all 3 primary metrics, each at least 4/5 seeds):

| Comparison | Forgetting | Seen rank-1 | Unseen AUROC | Result |
|---|---|---|---|---|
| D+R vs R | 5/5 (+0.046) | 5/5 (+0.056) | **2/5** (+0.008) | does not beat |
| slowbb+R vs R | 5/5 (+0.035) | 5/5 (+0.048) | 4/5 (+0.011) | beats |
| D+R vs slowbb+R | 4/5 (+0.011) | 3/5 (+0.008) | 2/5 (−0.003) | does not beat |

### VERDICT: **NOT CONFIRMED**

The C3 hypothesis (D + replay beats replay on all three metrics) fails, because **the unseen-people AUROC gain from C3 (+0.046) did not replicate** (+0.008, 2/5). The C3 observation that CMS was "the first method to help unseen people" is **withdrawn**.

### What did replicate, and what it is due to

**Supporting comparisons** (reference only, not decision arms; `runs_dgx/c3b_supporting.txt`):

| Paired, seeds 6–10 | Forgetting | Seen rank-1 | Unseen AUROC | Seen DIR@5% | Unseen DIR@5% |
|---|---|---|---|---|---|
| ctrl+R vs R: **precision/architecture only** | 5/5 (+0.025) | 4/5 (+0.009) | 4/5 (+0.003) | **0/5 (−0.020)** | 0/5 (−0.009) |
| slowbb+R vs ctrl+R: **frequency split only** | 3/5 (+0.009) | **5/5 (+0.038)** | 3/5 (+0.008) | 3/5 (+0.009) | 4/5 (+0.016) |

1. **Most of the forgetting reduction is not an NL effect.** Changing the neck and training in fp32 (ctrl+R) already lowers forgetting in 5/5 seeds, while lowering seen DIR@5% in 0/5 seeds (it gets worse). This **precision/architecture confound** also affects the C2 → C3 comparisons, where the C2 baselines used AMP and the CMS arms used fp32.
2. **The one effect that holds up and is attributable to NL's frequency principle** is better **seen-people rank-1 from a slow-updating backbone**:
   - +0.038 against the identical same-precision control, 5/5 fresh seeds;
   - +0.048 against replay, 5/5;
   - consistent with C3 on seeds 1–5 (D+R vs ctrl+R: +0.041, 5/5).

   This comparison was a pre-registered *arm* but not a pre-registered *decision* comparison, so it is **strong supporting evidence, not a confirmed claim**.
3. **The CMS block chain adds nothing measurable** beyond the slow backbone (D+R vs slowbb+R: 4/5, 3/5, 2/5, all within noise).
4. **No NL variant improves open-set acceptance** (DIR@5%, seen or unseen) **or unseen-people rejection** robustly.

### Implications for the thesis

- Across Phase 3, C3 and C3b, NL's specific mechanisms tested here do **not** solve the open-set problems:
  - surprise-based memories;
  - CMS block chains;
  - meta-learned write rules.
- NL's core design principle, **separating parameters by update frequency** (a slow backbone level and fast upper levels, Eq. 71), gives a small but consistent **retention** benefit for already-enrolled people (rank-1 +0.04 in continual learning with replay).
- Two methodological findings are themselves worth reporting:
  - **Pre-registered confirmation caught a false positive.** The C3 unseen-AUROC effect didn't replicate.
  - **Numerical precision is a hidden confound** in continual-learning comparisons.
