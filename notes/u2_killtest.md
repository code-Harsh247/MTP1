# U2: Phase 3 memory kill test at UERC scale

*This file, `scripts/u2_tune.py`, `scripts/u2_seed.py` and `scripts/u2_report.py` were committed **before** any U2 run (2026-10-20). Results are appended below.*

## Why rerun it

On EarVN (Phase 3) the NL memory never beat cosine, and **meta-training never improved on its untrained start**. Meta-training had only **58** identities, and even the no-memory control learned nothing from them. So the failure might simply be data starvation. UERC's split backbone leaves about **459** meta-training identities per seed, 8× more. This is the fair test of the original thesis idea: **surprise-based unknown rejection with a meta-learned NL memory.**

## Protocol (same as Phase 3, scaled up)

- **Split backbone (per seed):**
  - 459 of the 917 training identities are held out as `split="meta"`.
  - The backbone (the U1 recipe: ResNet-50 + ArcFace, 4 epochs, batch 64, AMP) trains on the other 458.
  - Everything is embedded with it, and **all arms are scored on the same embeddings**.
- **Episodes:** 32 enrolled identities (2 gallery images) + 32 strangers, 10 probes each, drawn from the meta pool. This is closer to the 131 + 131 test structure than Phase 3's 8 + 8.
- **Meta-training:** 3,000 episodes. The checkpoint is selected every 250 episodes on **validation** DIR@5%, and the untrained start (episode 0) competes.
- **Seeds:** **0** chooses arm 3's configuration on validation only (`u2_tune.py`). **1–3** are the test seeds (`u2_seed.py`).

## Arms

| Arm | What it is | Key in metrics.json |
|---|---|---|
| 1 | cosine_proto (U1 baseline scorer) | `cosine_proto` |
| 2 | linear memory (= subspace) | `subspace` |
| 3 | untrained deep memory: MLP 512 hidden, lr 0.5, momentum 0.9, steps ∈ {3, 10}, forget ∈ {0, 0.1, 0.3}, chosen on seed-0 val (best DIR@5% to 3 d.p., tie → AUROC) | `mem:mlp:…` (from `u2_arm3_choice.json`) |
| **4** | **meta-trained memory**: W_K, W_V, M₀ and the write rule, starting exactly at arm 3 | `meta_memory_full` |
| 4′ | meta-trained write rule only (lr, momentum, forget) | `meta_memory_rule` |
| 4b | **control**: same episodes and loss, learned projection + cosine, no memory (meta lr 3e-5) | `meta_proto_ctrl` |
| ref | EVM (gallery negatives) | `evm` |

## Decision rule

- **Primary metric:** test **DIR@5%** (2 gallery images, all sources), paired per seed, on seeds 1–3.
- **GO:** arm **4** beats arm **1** (cosine) **and** arm **4b** (control) on the primary metric in **3 of 3** seeds. That would mean a meta-learned NL memory rejects strangers better than cosine, and the gain comes from the memory, not from extra training on more people.
- **PARTIAL:** arm 4 beats arm 1 in 3/3 seeds but not arm 4b. The gain is from extra training (the control matches it), not the memory.
- **NO-GO:** arm 4 does not beat arm 1 in 3/3 seeds. Combined with Phase 3, the surprise-memory idea is then negative at both scales.

**Reported for every arm** (not decision criteria):

- arm 4′ and arms 2–3 vs arm 1;
- AUROC, rank-1, DIR@1%/10%;
- the 5-gallery variant;
- per source (VGGFace2 / others);
- **which checkpoint each meta-trained arm selected** (did meta-training ever beat its start?).

## Results (2026-10-20; DGX; `runs_dgx/u2_report.txt`)

The rule and scripts were unchanged since the pre-registration commit `dfe77ac` (checked with `git diff`). All 3 test seeds completed, with 0 failed steps.

**Arm 3 choice** (seed-0 val): `mem:mlp:512:10:0.5:0.9:0.3`, the **same configuration as in Phase 3**. Its val DIR@5% was 0.294, against 0.305 for cosine.

**Test, seeds 1–3, all sources, 2 gallery images:**

| Arm | Rank-1 | AUROC | DIR@1% | **DIR@5%** | DIR@10% |
|---|---|---|---|---|---|
| 1 cosine | **0.542 ± 0.017** | **0.715 ± 0.005** | 0.131 | 0.241 ± 0.007 | 0.307 |
| 2 linear memory | 0.510 | 0.699 | 0.125 | 0.232 | 0.292 |
| 3 deep memory | 0.523 | 0.709 | 0.126 | 0.236 | 0.300 |
| **4 meta-trained memory** | 0.522 | 0.709 | 0.126 | **0.236 ± 0.012** | 0.300 |
| 4′ meta-trained rule | 0.537 | **0.715** | **0.133** | **0.246 ± 0.007** | **0.310** |
| 4b control | **0.542** | 0.712 | 0.126 | 0.235 ± 0.014 | 0.302 |
| EVM | 0.422 | 0.662 | 0.108 | 0.201 | 0.254 |

**5 gallery images (DIR@5%):** cosine **0.366**, control 0.356, rule 0.290, deep and meta memory **0.188–0.190**, linear 0.333.

**Per source (2 gallery, DIR@5%):**

- VGGFace2: cosine 0.267, meta memory 0.261, rule 0.272, control 0.260.
- UERC19 + new: cosine 0.320, meta memory 0.324, rule 0.322, control 0.316, EVM 0.331.

All differences are within about 1 std.

**Decision** (primary: test DIR@5%, paired over seeds 1–3):

- arm 4 vs cosine: better on **0/3** (mean −0.005)
- arm 4 vs control: better on 1/3 (mean +0.0005)

### VERDICT: **NO-GO**

Even with **8× more meta-training identities** (459 vs 58), the meta-learned NL memory does not reject unknown people better than cosine. **Meta-training of the full memory again never improved on its untrained start**: the episode-0 checkpoint was selected on all 3 seeds, exactly as on EarVN. So the Phase 3 negative result was **not** caused by data starvation. It holds at both scales.

### Reported observations (not decision criteria)

1. **Write-rule-only meta-learning (4′) now learns.** It selected late checkpoints (episodes 3000, 1500, 3000; in Phase 3 mostly early or 0) and beat cosine on 3/3 seeds at 2 gallery images. The gain is **tiny** (+0.005 DIR@5%, below one seed-std), and with 5 gallery images it is clearly worse than cosine (0.290 vs 0.366). It is the first sign that meta-learning in this framework picks up *some* transferable signal with enough data, but it is not a useful effect and not a claim.
2. **Per-identity deep memories degrade with more gallery images** (5 gallery: 0.19 vs 0.37 for cosine), as on EarVN. Writing a small MLP on 5 images fits those images and generalizes worse to the same person's other photos.
3. **Backbone data dominates.** The same recipe on half the training identities (458, split backbone) gives DIR@5% 0.241, against **0.370** with all 917 (U1). Doubling the training people is worth +0.13; no NL arm changes the result by more than ±0.01.

### Conclusion across Phase 3 and U2

Surprise-based unknown rejection with a Titans-style NL memory (untrained, deep, or meta-learned) **does not beat a cosine-to-prototype baseline** for open-set ear recognition, at either 115 or 917 training identities. The decisive factor for rejecting unknown people is the quality of the embedding, which is driven by the number of training identities, not the scoring memory placed on top of it.
