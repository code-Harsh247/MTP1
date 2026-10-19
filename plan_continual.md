# Plan v3 (draft): Continual ear recognition with Nested Learning

*Drafted 2026-10-19 after the Phase 3 kill test came out no-go (`notes/phase3_killtest.md`).*

**Status 2026-10-19:** adopted. The user waived supervisor sign-off (C0).

Defaults for the §8 questions:

1. The new question is accepted.
2. P1 (identity-incremental) comes first.
3. Replay is run as a baseline, and CMS is judged under both §5 conditions: (a) without stored data, (b) on top of replay.
4. Phase 3 is reported as its own chapter.
5. Deadline unknown: C4/C5 scope is decided after C3.

## 1. Why this direction

- **What failed.** The Phase 3 kill test asked whether an NL memory's *surprise* rejects unknown people better than cosine similarity. The answer was no, in every variant.
- **What NL is built for.** The Nested Learning paper's own evidence is about **continual learning**: CMS (the Continuum Memory System) beat EWC and other methods on class-incremental tasks ([NL] §9.1). Our new direction plays to that strength instead of the one we tested.

> **New thesis question:** can an ear recognizer built with NL's multi-frequency memory (CMS) **keep learning new people and new capture conditions over time without forgetting the old ones**, while still recognizing unseen people and rejecting unknowns?

The supervisor's two requirements stay in place. After every learning stage we still evaluate **unseen people** (identity-disjoint open-set protocol) and **unknown rejection** (DIR@FAR, AUROC).

## 2. Important framing decision

Simply *adding people to the gallery* is not a continual-learning problem. Storing prototype embeddings is perfect memory, so nothing is forgotten and NL cannot help. The problem NL addresses is when the **model itself keeps training** on new data, for example:

- a deployed system that is periodically fine-tuned on newly collected people, or
- a system that moves to a new camera or setting (a new dataset).

Training on the new data overwrites what the network knew. That is **catastrophic forgetting**: embeddings of earlier people and domains get worse.

## 3. The two protocols

**P1: Identity-incremental (within EarVN1.0).** This is the main protocol.

- **Tasks.** The 115 EarVN training identities are split into **5 tasks of 23 people**. The model trains on task 1, then task 2, and so on. It never sees earlier tasks' data again, except through the replay baseline's small buffer.
- **Held-out images for seen people.** 20% of every training identity's images are held back. EarVN has 107–300 images per person, so this is cheap. They let us measure forgetting of *seen* people.
- **Measurements after every task t:**
  - **A[t, j]**: open-set identification on task j's people (j ≤ t), using their held-out images (2 gallery images, the rest as probes, other tasks' people as unknowns).
  - **U[t]**: the Phase 2 open-set protocol on the **unseen** val/test identities, the same splits as before.
- **5 seeds** (task orders and identity splits), reported as mean ± std, as before.

**P2: Domain-incremental (across datasets).** This is the realistic follow-up.

- Stages: AMI → EarVN1.0 → AWE (→ UERC on the DGX).
- After each stage, run every dataset's Phase 2 open-set test protocol.
- Forgetting here means losing performance on earlier capture conditions. Example: lab-quality AMI ears after training on web-collected AWE.

**Metrics** (standard in continual learning, plus ours):

- **Final average performance**: the mean over tasks of A[T, j], using DIR@5% and rank-1.
- **Forgetting**: the mean over tasks of the largest drop, max over t < T of (A[t, j] − A[T, j]).
- **Backward transfer**: whether learning later tasks helps or hurts earlier ones.
- **Unseen-people open-set performance U[t] over time**: the supervisor's requirement.

## 4. Methods compared

| Method | What it is | Role |
|---|---|---|
| **Joint training** | All tasks' data at once | Upper bound |
| **Naive fine-tuning** | Train on each task in turn, nothing special | Lower bound (shows how bad forgetting is) |
| **EWC** | Penalize changes to weights that mattered for old tasks | Classic regularization baseline (the NL paper compares to it too) |
| **LwF / feature distillation** | Keep new embeddings close to the old model's on new data | Distillation baseline, no stored data |
| **Experience replay** | Keep a small buffer (e.g. 5 images per old person) and mix it in | Strongest simple baseline; must be beaten or matched |
| **NL: CMS neck** | Embedding head = chain of MLP blocks updated at different frequencies ([NL] Eq. 70–71) | Main NL method, cheap |
| **NL: CMS + replay** | Both combined | Does CMS add value on top of replay? |
| **NL: M3 optimizer** (optional) | Multi-scale momentum ([NL] §7.2), the paper's only vision result | NL applied to the optimizer instead of the architecture |
| **NL: ViT + CMS** (stage 2) | Pretrained ViT-S with its MLP blocks turned into CMS, initialized from pretrained weights ([NL] §7.3) | Closest to the paper's Hope-Attention; only if the CMS neck shows promise |

**How CMS works here** (from `notes/nl_equations.md` §5):

- The embedding head is a chain of MLP blocks MLP⁽¹⁾ … MLP⁽ᵏ⁾.
- Block ℓ only updates every C⁽ℓ⁾ steps, using its *accumulated* gradient over that window, e.g. every 1 / 16 / 256 steps.
- Fast blocks adapt to the current task. Slow blocks change rarely, so old knowledge survives in them and can flow back.
- Variants to test: sequential vs. independent (aggregated) blocks ([NL] Eq. 73–74); 2–4 levels; frequency spacing.

**Reused from Phases 0–3:**

- data pipeline, splits, RAM image cache
- the ArcFace training loop (it becomes the per-task trainer)
- the open-set evaluation (cosine_proto stays the scorer)
- the seed-and-validation discipline
- the `AssocMemory` write-rule code, if a memory-based variant is wanted later

## 5. Decision rule (fixed in advance, like Phase 3)

On P1 (EarVN, 5 seeds), CMS **succeeds** if **either** of these holds:

- (a) **CMS beats naive fine-tuning and EWC** on forgetting **and** on final DIR@5%, with no extra stored data, in at least 4 of 5 seeds; or
- (b) **CMS + replay beats replay alone** at the same buffer size.

Otherwise the CMS neck is a no-go. We then try the ViT + CMS variant once (it is closer to the paper's architecture) before reporting a negative result.

Both outcomes are reportable. Phases 1–3 plus a clean continual-learning comparison make a complete thesis either way.

## 6. Phases

| Phase | Work | Where | Done when |
|---|---|---|---|
| **C0. Sign-off** (waived) | Supervisor agrees on the question, protocols and decision rule | — | Waived by the user, 2026-10-19 |
| **C1. Continual harness** ✅ — [notes/c1_continual_harness.md](notes/c1_continual_harness.md): naive forgets 0.17 DIR@5%; joint 0.49 vs naive 0.18 seen DIR@5% | Task splits and held-out seen-identity images; sequential trainer; accuracy matrix A[t, j], forgetting and U[t]; joint and naive baselines | Laptop (EarVN) | Joint vs. naive shows clear forgetting on 5 seeds |
| **C2. Baselines** ✅ — [notes/c2_baselines.md](notes/c2_baselines.md): replay k=5 best (0.221 seen DIR@5% vs naive 0.183, joint 0.501); EWC/LwF trade forgetting for plasticity; none help unseen people | EWC, LwF, replay (buffer sizes 2 and 5 images per id); tuned on validation tasks only | Laptop | Baseline table on P1 |
| **C3. CMS neck** | CMS module (Eq. 70–71, levels and frequencies), unit tests; CMS and CMS + replay; optional M3; **decision rule §5** | Laptop | Go / no-go on P1 |
| **C4. Scale-up** | If go: ViT-S + CMS; P2 domain-incremental AMI → EarVN → AWE | Laptop → DGX | P2 table |
| **C5. UERC** | P1 on UERC (1,310 ids → e.g. 10 tasks of ~90); side check: rerun the Phase 3 surprise kill test with ~10× more meta-training ids | DGX | Final tables |
| **C6. Write-up** | Thesis: Phase 1–3 (surprise rejection, negative but informative) + continual learning | — | Draft to supervisor |

## 7. Risks

- **Replay is hard to beat.** A few stored images per person often removes most forgetting. That's why success condition (b) exists, and why the CMS-only comparison is against methods without stored data.
- **A CMS neck on a ResNet might be too small to matter**, since most of the network is in the CNN body. If C3 is borderline, C4 (ViT, where every block has an MLP) is the real test.
- **Privacy angle (useful framing).** Biometric systems often *cannot* store old people's images. Methods that don't need replay (CMS, EWC, LwF) are then the only legal option, which makes condition (a) meaningful by itself.
- **Compute.** P1 on EarVN: 5 tasks × several methods × 5 seeds, at ~1.5 min per task on the laptop, is manageable. ViT and UERC need the DGX.

## 8. Questions for the supervisor

1. Is changing the main question from "better unknown rejection" to "continual learning without forgetting (with open-set evaluation kept)" acceptable?
2. Which scenario matters more for the thesis story: identity-incremental (P1) or domain-incremental (P2)?
3. Is a replay buffer allowed in the target scenario, or should we assume old images cannot be stored (privacy)?
4. How should the Phase 3 negative result be reported: as its own chapter or as motivation?
5. What is the deadline? That decides whether C4 (ViT) and C5 (UERC) are in scope.
