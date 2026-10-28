# C2: Continual-learning baselines (EarVN, P1)

*Date: 2026-10-20.*

- **Code:** `nlear/cl_methods.py`, `scripts/train_continual.py`.
- **Tuning (laptop, seed 0):** `runs/earvn/c2_tune.log`.
- **Final runs (DGX, Slurm array 34923, seeds 1–5):** `runs_dgx/earvn/continual/<method>/seed1..5` (unpacked from `c2_results.tar.gz`; model files left on the DGX).

## Setup

- **Protocol:** P1 as in C1. Each seed has 115 training identities split into 5 tasks of 23, with 250 steps per task, the Phase 2 backbone config, cosine_proto scoring, and 20% of every training identity's images held out.
- **Methods:**
  - **EWC (online):** a diagonal Fisher from 30 batches after each task. The penalty is λ/2 Σ F (θ − θ*)² on the embedding network.
  - **LwF:** embedding distillation from a frozen copy of the previous model, λ‖norm(e_new) − norm(e_old)‖².
  - **Replay:** k training images kept per learned identity; half of every batch comes from this buffer.
- **Tuning discipline:** λ was chosen on **seed 0 only**, and the results below are on **seeds 1–5**, which tuning never touched. Seed 5's split was added for this; seeds 0–4 are byte-identical to before.
- **Cross-machine check:** joint training on seed 1 gives a final seen DIR@5% of 0.565 on the DGX and 0.560 on the laptop (C1). Results are consistent across machines.

### λ tuning (seed 0)

| Setting | Final seen DIR@5% | Forgetting |
|---|---|---|
| naive | 0.143 | 0.192 |
| EWC 10² / **10⁴** / 10⁶ | 0.135 / **0.128** / 0.052 | 0.207 / **0.015** / 0.004 |
| LwF 1 / **10** / 100 | 0.150 / **0.125** / 0.049 | 0.166 / **0.035** / 0.019 |

The selection rule was **best final seen DIR@5%**, which picks **EWC λ = 10²** and **LwF λ = 1**. Both are close to naive, and single-seed differences of this size are noise. So the **stability settings** (EWC 10⁴, LwF 10), which nearly remove forgetting, were also run on seeds 1–5 and are reported transparently as such. They were not selected by the rule.

## Results (test, seeds 1–5, mean ± std)

| Method | Seen DIR@5% | Seen rank-1 | Forgetting DIR@5% | Forgetting rank-1 | Unseen rank-1 | Unseen AUROC | Unseen DIR@5% |
|---|---|---|---|---|---|---|---|
| **Joint** (upper bound) | **0.501 ± 0.033** | **0.809 ± 0.007** | — | — | **0.530 ± 0.071** | **0.708 ± 0.030** | **0.181 ± 0.052** |
| Naive (lower bound) | 0.183 ± 0.037 | 0.536 ± 0.021 | 0.180 ± 0.022 | 0.251 ± 0.022 | 0.409 ± 0.055 | 0.597 ± 0.050 | 0.089 ± 0.036 |
| EWC λ=10² (rule pick) | 0.185 ± 0.038 | 0.549 ± 0.013 | 0.171 ± 0.019 | 0.235 ± 0.018 | 0.415 ± 0.065 | 0.585 ± 0.055 | 0.077 ± 0.031 |
| EWC λ=10⁴ (stability) | 0.148 ± 0.038 | 0.523 ± 0.029 | **0.032 ± 0.027** | **0.025 ± 0.014** | 0.361 ± 0.067 | 0.576 ± 0.047 | 0.059 ± 0.018 |
| LwF λ=1 (rule pick) | 0.198 ± 0.040 | 0.575 ± 0.017 | 0.154 ± 0.025 | 0.201 ± 0.016 | 0.432 ± 0.061 | 0.600 ± 0.052 | 0.081 ± 0.036 |
| LwF λ=10 (stability) | 0.158 ± 0.023 | 0.549 ± 0.016 | 0.037 ± 0.009 | 0.057 ± 0.013 | 0.411 ± 0.070 | 0.630 ± 0.046 | 0.075 ± 0.032 |
| Replay k=2 | 0.181 ± 0.019 | 0.543 ± 0.012 | 0.111 ± 0.021 | 0.155 ± 0.020 | 0.407 ± 0.057 | 0.622 ± 0.035 | 0.095 ± 0.031 |
| **Replay k=5** | **0.221 ± 0.010** | **0.583 ± 0.014** | 0.071 ± 0.016 | 0.096 ± 0.021 | 0.410 ± 0.066 | 0.600 ± 0.015 | 0.083 ± 0.033 |

**Paired differences against naive** (the same seed and the same task order; the counts are seeds out of 5):

| Method | Δ seen DIR@5% | Δ forgetting | Δ unseen DIR@5% |
|---|---|---|---|
| EWC 10² | +0.002 ± 0.006 (3/5 better) | −0.010 ± 0.009 (4/5 lower) | −0.012 ± 0.010 (1/5 better) |
| EWC 10⁴ | −0.035 ± 0.007 (0/5) | **−0.148 ± 0.017 (5/5)** | −0.030 ± 0.021 (0/5) |
| LwF 1 | +0.015 ± 0.012 (4/5) | −0.026 ± 0.018 (4/5) | −0.008 ± 0.008 (1/5) |
| LwF 10 | −0.025 ± 0.020 (1/5) | **−0.143 ± 0.020 (5/5)** | −0.014 ± 0.014 (0/5) |
| Replay k=2 | −0.002 ± 0.022 (3/5) | −0.070 ± 0.006 (5/5) | +0.006 ± 0.015 (3/5) |
| **Replay k=5** | **+0.038 ± 0.032 (4/5)** | −0.109 ± 0.012 (5/5) | −0.006 ± 0.015 (3/5) |

## Findings

1. **Replay with 5 images per person is the strongest baseline.** It has the best final seen DIR@5% (0.221, +0.038 over naive in 4/5 seeds) and cuts forgetting from 0.180 to 0.071. Its seed-to-seed std is also the smallest (0.010). This is the bar for CMS + replay (decision rule (b)).
2. **Without stored data, nothing beats naive by much.** At the rule-picked strengths, EWC (+0.002) and LwF (+0.015) barely move the final score.
3. **The stability–plasticity trade-off is sharp.** EWC 10⁴ and LwF 10 almost **eliminate forgetting** (0.03–0.04, 5/5 seeds), but end *lower* than naive on final seen DIR@5% (0.148 and 0.158). They protect the old people by freezing a model that then learns each new group poorly.
4. **Forgetting is only half of the gap.** Even with near-zero forgetting, sequential methods end far below joint training (0.15–0.22 vs 0.50). Learning 23 people at a time gives a weaker embedding than learning all 115 together, so **positive transfer between tasks is missing**, not only retention. A method that reaches joint-level performance would need both. This shapes what CMS has to show: its slow levels are supposed to keep consolidating across tasks, not just freeze.
5. **No baseline helps unseen people.** Unknown rejection on never-seen people stays at naive's level (unseen DIR@5% 0.06–0.095, against 0.181 for joint) for every method. The stability settings make it slightly worse. So continual training damages open-set generalization, and none of the standard fixes repair it. This is room for CMS to stand out on the supervisor's requirement.

## What C3 (CMS) must show

The decision rule fixed in advance (`plan_continual.md` §5), with the C2 bars filled in:

- **(a) Without stored data:** beat naive **and** EWC (rule pick, λ=10²) on **both** forgetting (< 0.171) **and** final seen DIR@5% (> 0.185), in at least 4 of 5 seeds. For transparency, also compare against the stability settings, which already reach forgetting ≈ 0.03 but only 0.15–0.16 final.
- **(b) With replay:** CMS + replay k=5 beats replay k=5 alone (final seen DIR@5% 0.221 ± 0.010; forgetting 0.071).
- **Reported regardless:** unseen-people open-set metrics, where every baseline is stuck at naive's level.
