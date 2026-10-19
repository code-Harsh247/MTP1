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

## Results

*(appended after the runs)*
