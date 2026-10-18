# Plan: Nested Learning for Open-Set Ear Recognition

*Created 2026-10-18. Replaces the earlier MAML plan.*

## 1. Goal

Build an ear recognition system that:

1. recognizes people **not seen during training** (they are only enrolled from a few gallery images), and
2. **rejects unknown people** who were never enrolled.

**Thesis claim (hypothesis):** an associative memory built on Nested Learning (NL) principles gives a principled rejection signal, its own *surprise* (prediction error). This should beat the standard "reject if max similarity < threshold" rule at matched false-accept rates.

## 2. Core idea in one picture

```
ear image ──► backbone (SLOW level, trained offline) ──► embedding z
                                                          │
                       enrollment writes into ───►  identity memory (MEDIUM level)
                                                          │
probe z ──► query memory ──► surprise s_i for each identity i
                              │
              min_i s_i < τ ? ──► yes: identity = argmin_i s_i
                              └─► no : "unknown"
```

Optional FAST level: a small per-probe memory adapted at test time to the probe's pose and occlusion (Phase 5).

## 3. Method design

Exact equations and design traps are in [notes/nl_equations.md](notes/nl_equations.md) §8.

**Backbone (slow).** ImageNet-pretrained ResNet-50 or ViT-S. Fine-tune on training identities with ArcFace/CosFace and L2-normalize the embedding.

**Identity memory (medium).** We will try two variants and keep the better one:

- **Variant A: per-identity memory.** Each enrolled person *i* gets a small memory M_i, a linear map or a 2-layer MLP. It is written at enrollment by a few Titans-style gradient steps on an associative loss, `ℓ_i(z) = ‖M_i(W_k z) − W_v z‖²`, using that person's gallery embeddings.
  - Surprise of a probe against person *i* is `s_i = ℓ_i(z_probe)`, or the gradient norm, as in Titans.
  - Identity is `argmin s_i`. If `min s_i > τ`, the probe is unknown.
- **Variant B: one shared memory.** A single memory M maps keys to identity codes. Surprise is the distance between the retrieved value and the nearest stored code.

The projections W_k and W_v and the memory's learning rate, momentum and forget gate are **meta-trained** on training identities. Training is episodic: sample some identities, write their memories, query with known and held-out "unknown" identities, and train so that known probes are unsurprising and unknown probes are surprising.

**Rejection threshold τ.** Calibrate it on **validation identities only**, never on test.

## 4. Data and protocol

**Staged datasets: small first, UERC last** (agreed 2026-10-18). Sizes come from the literature and must be verified on disk in Phase 0. All sets live under `/dgx1/dsamantaai/santhosh/Ear_Datasets/` on the DGX.

| Stage | Dataset | Subjects / images | Role | Where it trains |
|---|---|---|---|---|
| S1 | **AMI** | 100 / ~700, controlled | Pipeline sanity: does everything run, and do numbers look sane? | Laptop |
| S2 | **EarVN1.0** | 164 / ~28,400, in the wild | **Main development set**: baselines, kill test, memory design | Laptop (DGX for sweeps) |
| S3 | **AWE** | 100 / 1,000, in the wild, hard | Harder images; comparable to published results | Laptop |
| S4 | **UERC 2023 `public/`** | 1,310 / large | Final scale and the thesis results | DGX |
| — | IITD | 125–221 / ~500–800, controlled | Cross-dataset test only | — |

UERC notes:

- Path: `.../Ear_Datasets/UERC 2023 Competition Package with sequestered anonymized data/data/public/`.
- The sequestered test set is unlabeled, so all splits come from `public/`.
- Don't use the truncated `uerc23_dataset.zip`.

**Splits are identity-disjoint:**

- train identities: about 70%
- validation identities: about 10%, split into known and unknown
- test identities: about 20%, with some enrolled (known) and the rest never enrolled (unknown)

**Small datasets need repeated splits.** AWE and AMI leave only about 20 test identities (about 10 known and 10 unknown), so one split is too noisy. On S1–S3, report the **mean ± std over 5 random identity splits** (seeds 0–4). On UERC, one split plus 3 training seeds.

**Backbones are always ImageNet-pretrained.** Training from scratch on S1–S3 would only memorize the training set.

**Cross-dataset checks:**

- Train on EarVN1.0 → test on AWE and IITD.
- Later, train on UERC → test on EarVN1.0, AWE and IITD.
- VGGFace-Ear likely overlaps UERC, so exclude it.

**Caveat.** With only about 70–115 training identities, the memory's episodic meta-training can overfit (the identity trap). Positive results on S1–S3 are a **go signal, not the thesis result**; the claim must hold on UERC.
- **Metrics:**
  - closed-set: Rank-1
  - verification: TAR@FAR and EER
  - open-set: DIR@FAR (1%, 0.1%) and AUROC for known-versus-unknown

## 5. Phases

| Phase | Work | Where | Done when |
|---|---|---|---|
| **0. Setup** ✅ | Python env and repo layout. AMI, EarVN1.0, AWE and IITD copied to `data/`, layouts verified, exact duplicates removed, 5 open-set splits each. Findings are in [notes/phase0_data.md](notes/phase0_data.md#phase-0-results-2026-10-18). | Laptop | Splits and the data loader run on all four sets |
| **1. Read and pin down NL** ✅ | Read the Nested Learning and Titans papers. Write down the exact memory update, surprise, momentum and forget-gate equations. Check whether official code exists (none does). | — | [notes/nl_equations.md](notes/nl_equations.md) is written |
| **2. Baseline** ✅ (incl. EVM) — results in [notes/phase2_baseline.md](notes/phase2_baseline.md) | Pretrained ResNet with ArcFace. Open-set scores: max cosine, a **subspace-residual** classifier (the linear-memory equivalent, see notes §8.3), and EVM or score normalization. Run in order: AMI (sanity) → EarVN1.0 → AWE. | Laptop | Baseline table on S1–S3 (mean ± std over 5 splits) |
| **3. Kill test** | On EarVN1.0, freeze the baseline backbone and compare four rejection scores on saved embeddings: (1) max cosine, (2) linear-memory surprise, (3) deep-memory surprise without meta-training, (4) deep-memory surprise with episodic meta-training. Confirm on AWE. | Laptop | Go or no-go decision (see §6) |
| **4. Full NL-Ear** | Meta-train the memory parameters, try Variant B, and run ablations: memory depth, number of write steps, momentum on/off, forget gate on/off. Develop on EarVN1.0, then **scale to UERC** (S4) for the main results. | Laptop → DGX | Main results table on UERC |
| **5. Extras (if time)** | (a) Fast test-time memory for pose and occlusion. (b) Continual enrollment: add identities in batches and measure forgetting. (c) Cross-dataset tests (see §4). | DGX | Extra tables |
| **6. Write-up** | Thesis chapters, figures, and score-distribution plots | — | Draft to the supervisor |

## 6. Decision rule (fixed in advance)

After Phase 3: if surprise-based rejection shows **no gain** over max-cosine at matched FAR, do not push the "surprise" claim further. Fall back to:

- the NL memory as an **enrollment and continual-learning** mechanism (Phase 5b) as the main contribution, or
- a CMS multi-frequency backbone compared against the standard backbone.

Report the negative result honestly either way.

## 7. Compute split

The laptop has an RTX 3050 Laptop GPU (4 GB VRAM), 16 GB RAM and a Ryzen 7 5800H.

- **Laptop:** everything on AMI, EarVN1.0 and AWE. That means training a pretrained ResNet-18/50 at 128 px with mixed precision and moderate batch sizes, memory experiments, evaluation and plots. A run should take minutes up to about an hour.
- **DGX (Slurm, `dgx_all` / account `dsamantaai` / QOS `gpu2`, `harsh_` job prefix; login `dsamantaai@gpucluster.iitkgp.ac.in`):** hyperparameter sweeps, all UERC training, and final runs.

## 8. Risks

- NL is validated on language and sequence modeling, not vision. A gain is not guaranteed, which is why strong baselines and the Phase 3 kill test come first.
- Official HOPE/Titans code may be incomplete, so the memory module may need a reimplementation. It is small (about 100–200 lines).
- Surprise may only measure "how different from the gallery" and add nothing beyond cosine distance. Phase 3 is designed to catch this early.
- **Identity trap:** a memory that learns something close to the identity map gives low surprise for *everyone*, unknowns included. Mitigations: W_K ≠ W_V, low-capacity memories, the forget gate, and meta-training with unknowns in every episode.
- A linear memory reduces to a classic subspace classifier. The NL contribution must come from deep memory and a meta-learned write rule.
- Few gallery images per identity may make per-identity memories overfit. Keep the memories tiny and regularized.

## 9. Open questions for the supervisor

1. Is surprise-based open-set rejection acceptable as the main thesis claim, with the fallback in §6?
2. Which backbone is expected: CNN or ViT?
3. What is the deadline for first results?
