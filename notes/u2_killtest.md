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

## Results

*(appended after the runs)*
