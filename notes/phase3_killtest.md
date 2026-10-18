# Phase 3: Kill test (EarVN1.0)

*Date: 2026-10-19. Code: `nlear/memory.py`, `scripts/meta_train.py`, `scripts/rescore.py`, `scripts/kill_test_earvn.sh`. Log: `runs/earvn/kill_test.log`.*

**Question:** does an NL associative memory's *surprise* reject unknown people better than cosine similarity, on the same embeddings?

## The memory (Titans-style, one per enrolled identity)

Equations from `notes/nl_equations.md` §3:

- Keys and values: k = norm(z W_K), v = z W_V.
- Loss: ℓ = ‖M(k) − v‖².
- Write rule: S ← η S − θ ∇ℓ (momentum = "past surprise"), then M ← (1 − α) M + S (α = forget gate).
- Writing: each identity's memory starts at a shared M₀ and takes `steps` full-batch steps on its gallery embeddings.
- Score: score_i(probe) = −ℓ(M_i; probe), the negative surprise.
- Architecture: "mlp" is M(k) = W₁ silu(W₂ k), Titans' deep memory without the residual (with k = v, the residual would be the identity trap).
- **Verified:** a linear memory written by gradient descent from M₀ = 0 equals the subspace baseline to within 1e-3 (`tests/test_memory.py`).

## Protocol: split backbone

**Why.** Meta-training the memory needs embeddings of people the backbone has *not* seen. Open-set episodes built from the backbone's own training identities are almost solved (EarVN, seeds 0–2: DIR@5% ≈ 0.92, AUROC ≈ 0.98), while the same episodes on unseen validation identities give DIR@5% ≈ 0.27–0.34. A memory meta-trained on the backbone's training identities would learn from a task that doesn't resemble the real one.

**How (per seed).**

1. 58 of the 115 training identities are held out as `split="meta"`.
2. The backbone (same Phase 2 config) is trained on the other 57.
3. Everything is embedded with that backbone.
4. Meta-training uses only the 58 meta identities.
5. **All arms are scored on these same embeddings.**

Absolute numbers are lower than Phase 2 (half the backbone training identities), but the comparison between arms is fair. Runs are in `runs/earvn/splitbb_resnet50/seed0..4`.

**Episodes** mimic the test: 8 enrolled identities (2 gallery images each) and 8 never-enrolled identities, 10 probes each. The loss is an open-set softmax over [a·score₁ … a·score₈, b], with b a learned "unknown" logit. Model selection uses the best **validation** DIR@5% over checkpoints every 100 episodes, and the untrained start (episode 0) competes too.

## Arms (fixed before the 5-seed run)

| Arm | What | Config source |
|---|---|---|
| 1 | cosine_proto | Phase 2 baseline |
| 2 | linear memory (= subspace) | exact equivalence |
| 3 | untrained deep memory: 512 hidden, 10 steps, lr 0.5, momentum 0.9, forget 0.3 | val sweep (full-backbone EarVN, 5 seeds; `runs/earvn/mem_untrained_tune.log`) |
| 4 | meta-trained deep memory: W_K, W_V, M₀, lr, momentum, forget | starts exactly at arm 3 (identity projections) |
| 4′ | meta-trained write rule only (lr, momentum, forget) | M₀ and projections frozen |
| 4b | **control:** same episodes and loss, learned projection + cosine, no memory | starts exactly at arm 1 |
| ref | EVM (gallery / +train negatives) | Phase 2 |

## Development findings (before the 5-seed run)

**Untrained memory sweep** (arm 3; full backbone, val, 5 seeds):

| Scorer | val DIR@5% | val AUROC |
|---|---|---|
| cosine_proto | **0.320** | **0.779** |
| linear memory = subspace | 0.283 | 0.739 |
| deep memory, no forget gate | 0.264 | 0.721 |
| deep memory, forget 0.1 | 0.298 | 0.743 |
| deep memory, **forget 0.3** | 0.301 | 0.752 |
| deep memory, lr 2 + 10 steps | diverged / collapsed | |

The deep memory beats the linear one, and the **forget gate matters most**, in agreement with Titans' own ablation (weight decay is its largest single contribution). But untrained, it stays below cosine.

**Meta-training diagnostics** (split backbone, seed 0, val DIR@5%):

| Variant | Start | After 100 episodes | After 400 episodes |
|---|---|---|---|
| everything learned, meta lr 3e-4 | 0.207 | 0.114 | 0.083 |
| projections frozen | 0.207 | 0.146 | 0.147 |
| everything learned, meta lr 3e-5 | 0.207 | 0.162 | 0.164 |
| M₀ frozen, meta lr 3e-5 | 0.207 | 0.204 | 0.190 |
| write rule only | 0.207 | 0.207 | 0.190 |
| control (cosine + projection), lr 3e-4 | 0.184 | 0.187 | 0.182 |
| control, lr 3e-5 | 0.184 | 0.190 | 0.189 |

Three things happened along the way:

1. **Stability fix.** Unrolled 10-step writes with momentum diverged to NaN once the meta-gradient pushed the write lr up. The learned lr is now bounded (sigmoid, at most 1.0), and non-finite episodes are skipped.
2. **Meta-learning M₀ hurts the most.** With projections frozen and the rates nearly unchanged, validation still drops from 0.207 to 0.146. A learned M₀ becomes a shared "average ear" prior inside every identity's memory, so unknown probes become less surprising. This is the **identity trap** of `nl_equations.md` §8.3, arriving through meta-learning instead of architecture.
3. **Overfitting in every variant.** The episode loss keeps falling (1.66 → 1.52) while validation DIR falls. Meanwhile the control (no memory) gains nothing. With 58 meta-training identities there is too little signal to meta-learn anything that transfers.

## 5-seed kill test results

Test, EarVN split-backbone embeddings, 5 seeds. "Δ vs cos" is the per-seed paired difference in DIR@5% against cosine_proto, which is valid because every arm uses identical embeddings. "Wins" counts the seeds where the arm beats cosine.

**2 gallery images per identity (the main setting):**

| Arm | Rank-1 | AUROC | DIR@1% | DIR@5% | DIR@10% | Δ DIR@5% vs cos | Wins |
|---|---|---|---|---|---|---|---|
| 1 cosine_proto | **0.441 ± 0.064** | **0.644 ± 0.015** | 0.050 ± 0.020 | **0.122 ± 0.035** | **0.172 ± 0.037** | — | — |
| 2 linear memory (= subspace) | 0.382 ± 0.062 | 0.622 ± 0.020 | 0.050 ± 0.023 | 0.116 ± 0.029 | 0.162 ± 0.035 | −0.006 ± 0.014 | 3/5 |
| 3 deep memory, untrained | 0.381 ± 0.069 | 0.628 ± 0.023 | 0.051 ± 0.026 | 0.116 ± 0.041 | 0.162 ± 0.043 | −0.005 ± 0.009 | 1/5 |
| 4 deep memory, meta-trained (full) | 0.381 ± 0.069 | 0.628 ± 0.023 | 0.051 ± 0.026 | 0.116 ± 0.041 | 0.162 ± 0.043 | −0.005 ± 0.009 | 1/5 |
| 4′ deep memory, meta-trained rule | 0.388 ± 0.072 | 0.631 ± 0.022 | 0.054 ± 0.026 | 0.121 ± 0.038 | 0.164 ± 0.039 | −0.001 ± 0.009 | 2/5 |
| 4b control (projection + cosine) | 0.444 ± 0.065 | 0.635 ± 0.018 | 0.047 ± 0.018 | 0.117 ± 0.036 | 0.168 ± 0.041 | −0.005 ± 0.004 | 0/5 |
| EVM | 0.374 ± 0.057 | 0.611 ± 0.026 | 0.039 ± 0.017 | 0.096 ± 0.031 | 0.139 ± 0.033 | −0.026 ± 0.029 | 1/5 |
| EVM + train negatives | 0.287 ± 0.111 | 0.585 ± 0.072 | 0.050 ± 0.016 | 0.096 ± 0.027 | 0.128 ± 0.039 | −0.026 ± 0.028 | 1/5 |

**5 gallery images per identity:**

| Arm | Rank-1 | AUROC | DIR@5% | Δ DIR@5% vs cos | Wins |
|---|---|---|---|---|---|
| 1 cosine_proto | **0.554 ± 0.034** | **0.677 ± 0.017** | **0.165 ± 0.026** | — | — |
| 2 linear memory | 0.457 ± 0.032 | 0.641 ± 0.025 | 0.147 ± 0.025 | −0.018 ± 0.019 | 0/5 |
| 3 / 4 deep memory | 0.351 ± 0.062 | 0.621 ± 0.017 | 0.093 ± 0.030 | −0.072 ± 0.017 | 0/5 |
| 4′ meta-trained rule | 0.363 ± 0.066 | 0.624 ± 0.014 | 0.098 ± 0.031 | −0.067 ± 0.018 | 0/5 |
| 4b control | 0.545 ± 0.029 | 0.664 ± 0.019 | 0.153 ± 0.020 | −0.012 ± 0.007 | 0/5 |

**Checkpoint selected on validation (episode 0 = untrained start):**

- full meta-training: [0, 0, 0, 0, 0]. **Meta-training never improved on its start in any seed.**
- write rule only: [0, 100, 800, 0, 700]
- control: [600, 0, 200, 300, 100]

## Verdict: **NO-GO** for the surprise-rejection claim (as measured on EarVN)

By the decision rule fixed in advance (plan.md §6: no gain over max-cosine at matched FAR means the claim is dropped):

1. **No memory arm beats cosine.** With 2 gallery images, all memory variants are within noise of cosine for rejection (Δ DIR@5% between −0.006 and −0.001; wins at most 3/5 seeds), and all have clearly lower rank-1 (0.38 vs 0.44).
2. **With 5 gallery images, the deep memory is clearly worse** (−0.07 DIR@5%, 0/5 seeds). It was tuned for 2 images; with more images it fits each gallery so well that it fails to generalize to the same person's other photos.
3. **Meta-training did not help.** The fully meta-trained memory never beat its untrained start on validation in any seed. The control (same training without a memory) also did not beat cosine. With 58 meta-training identities there is no transferable signal for either.
4. **Positive findings worth keeping for the thesis:**
   - The linear memory is mathematically and empirically the subspace method.
   - The deep memory beats the linear one only when it has a forget gate, matching Titans' ablation.
   - Meta-learning M₀ reproduces the identity trap.

   These are honest negative and diagnostic results.

## Caveat: scale

Meta-training had only 58 identities. The NL claim might still hold with much more meta-training data. UERC's split backbone would leave about 450–650 meta identities, roughly 10× more. Whether to spend one more test on that (on the DGX) or switch to the fallback directions now is a decision for the supervisor (see below).

## Options

- **A. One scaled retest on UERC (DGX).** The same kill test, protocol and arms, but with about 10× the meta-training identities. If it is still no-go, switch to B or C. It costs DGX time, plus moving the pipeline to the cluster, which is needed anyway.
- **B. Fallback 1, from plan.md §6: NL memory for continual enrollment.** Enroll people over time without retraining. Measure forgetting and the accuracy of earlier enrollees as the gallery grows, with CMS-style multi-frequency updates. NL was designed for exactly this, and the code here (`AssocMemory`, the write rule) carries over.
- **C. Fallback 2, from plan.md §6: CMS multi-frequency backbone** versus the standard backbone.
