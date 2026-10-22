# Plan: UERC 2023 (final-scale experiments)

*Drafted 2026-10-20 after C3b. Context: `plan.md` (Phases 0–3), `plan_continual.md` (C1–C3b).*

## Why UERC

1. **More training people.** About 900 training identities instead of 115 on EarVN, so the backbone should generalize far better to unseen people. EarVN showed a large gap: open-set DIR@5% about 0.92 on training people vs about 0.3 on new people.
2. **A fair test of the data-hungry NL ideas.** Meta-trained memory failed with 58 meta-training identities, and even its no-memory control learned nothing. UERC gives about 450.
3. **More trustworthy numbers.** The test split has about 260 people instead of 33, and UERC is the community's benchmark.

**Every NL result on UERC is compared with baselines trained on UERC**: same splits, same budget, same precision.

## Fixed design decisions

- **Data:** UERC 2023 `public/` only, because the sequestered set is unlabeled. Don't use the truncated `uerc23_dataset.zip`. Exclude VGGFace-Ear everywhere, since it likely overlaps UERC.
- **Splits:** identity-disjoint 70/10/20 as before, with val/test identities half known and half unknown, and 2 gallery images (plus a g5 variant). Splits are stratified by **source dataset**; UERC mixes 3 sources, about 94% VGGFace2-derived.
- **Seeds:** **seed 0 is used for all tuning (val split only); seeds 1–3 are the reported test runs.** With about 260 test people per split, 3 seeds are enough. Add 4–5 if a comparison is borderline.
- **Precision:** **every arm of every comparison runs in the same precision.** C3b found that fp32 vs mixed precision alone changes forgetting. Continual experiments therefore run everything in fp32, including plain replay.
- **Pre-registration:** as in C3/C3b. The decision rule, report script and task list are committed before the runs they judge.

## Phases

| Phase | Work | Output |
|---|---|---|
| **U0. Inspect and prepare** | Inspect the UERC layout on the DGX: image count, identities, how the source dataset is encoded, image sizes. Then build the manifest (with duplicate removal), splits for seeds 0–3 and the RAM cache, all on the DGX. | `notes/uerc_data.md`; splits on the DGX |
| **U1. Static baselines** ✅ ([notes/u1_baseline.md](notes/u1_baseline.md): cosine_proto rank-1 0.662, AUROC 0.784, DIR@5% 0.370; g5 0.809/0.860/0.536) | ResNet-50 + ArcFace (the Phase 2 recipe). Re-tune training length and regularization on seed-0 **val**, since the EarVN "5 epochs" won't transfer to 8× more data. Scorers: cosine_proto, subspace, EVM. Reported metrics: rank-1, AUROC, DIR@FAR, EER/TAR (for literature comparison). | UERC baseline table |
| **U2. Phase 3 rerun at scale** ✅ **NO-GO** ([notes/u2_killtest.md](notes/u2_killtest.md): meta-trained memory 0/3 vs cosine; meta-training never beat its start; not data starvation) | Split backbone (half the training identities for the backbone, about 450 held out for meta-training). Arms as in Phase 3: cosine, linear memory, untrained deep memory, meta-trained memory, write-rule-only, **projection + cosine control**. Pre-registered rule: does surprise-based rejection beat cosine *and* the control? | Go/no-go for the original NL idea at scale |
| **U3. Continual learning (P1) at scale** ✅ **NOT CONFIRMED** ([notes/u3_continual.md](notes/u3_continual.md): slow backbone worse than its control on 0/3 seeds, rank-1 −0.015; the EarVN effect reverses at scale) | About 10 tasks of about 90 people. Baselines (re-tuned on seed 0): joint, naive, replay k=5, EWC, LwF. NL arms: slow backbone + replay, CMS (D) + replay, same-architecture control + replay. **Pre-registered claim from C3b:** a slow backbone improves seen rank-1 over the same-architecture control (and forgetting/unseen metrics are reported). | Confirms or refutes the one surviving NL effect at scale |
| **U4 (optional). M3 optimizer** ✅ **NO-GO** ([notes/u4_m3.md](notes/u4_m3.md): M3 below AdamW on 0/3 seeds, DIR@5% −0.069; its NL slow memory makes it worse than its own control) | NL's multi-scale-momentum optimizer ([NL] §7.2, its only vision result) vs AdamW for the static recognizer. First on EarVN (cheap), then on UERC if promising. | Does NL's optimizer give a better ear model? |
| **B. CMS inside a pretrained ViT** ✅ ([notes/b_vit_cms.md](notes/b_vit_cms.md)) | DeiT-S with 3 multi-frequency copies of every MLP (NL §7.3). B1 static: **NO-GO** (DIR@5% −0.025, 0/3). B2 continual + replay: **GO** on seen rank-1 (+0.012, 3/3) but via more plasticity, with more forgetting and worse unseen-people metrics. Plain ViT is the best baseline (DIR@5% 0.433 vs 0.370) | B1/B2 tables |

## Practical points

- **Size:** UERC is much larger than EarVN. U0 measures it. At 128×96 the RAM cache is about 37 KB per image, so 250k images would be about 9 GB, which needs `--mem=64G` and a one-time cache build of roughly 30 minutes (parallelized if needed).
- **Compute:** a full UERC baseline is about 10× an EarVN run, so minutes to tens of minutes per run on a DGX GPU. The gpu2 QOS submit limit stays the main constraint: arrays of about 6 elements with several runs each.
- **Everything stays on the DGX.** The images are never copied to the laptop. Results (metrics only) come back as before.
- **Ordering:** U1 is needed by everything else. U2 and U3 can run in parallel after U1. U4 is independent.
