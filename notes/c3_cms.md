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

## Results

*(appended after the runs)*
