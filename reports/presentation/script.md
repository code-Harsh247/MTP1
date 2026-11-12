# Presentation script: Catastrophic Forgetting in Continual Ear Recognition

About 15 minutes at a relaxed pace (≈140 words a minute). Times in brackets are targets per slide; the running clock is on the right of each heading.
**[next]** marks where to change slide. Text in *italics* is a stage direction, not to be read.

If you are running long, cut in this order: the metrics sentence on slide 7, the recipe sentence on slide 6, the training table on slide 13. Each saves about 15 seconds.

---

## Opening (0:00 – 1:45)

### Slide 1 · Title (15 s) · 0:00

Good morning. My project is about a simple problem with a hard answer: when a recognition system keeps learning new people, it starts forgetting the ones it already knows. I tested whether a new idea called Nested Learning can stop that, for ear recognition.

**[next]**

### Slide 2 · Recognition systems forget the people they already know (45 s) · 0:15

Picture a deployed ear recognition system. People are enrolled group after group, and retraining from scratch every time is not practical. Ears are a good biometric for this: they are stable with age, captured without contact, and complement the face.

The problem is catastrophic forgetting. *Point at the chart.* This is a real run from my experiments. A ConvNeXt learns the first group of people to 0.88 rank-1. Then it learns a second group, reaches 0.94 on them, and its accuracy on the first group drops to 0.49. Nothing was deleted. Learning the new people simply overwrote the old ones.

**[next]**

### Slide 3 · Two gaps, three research questions (40 s) · 1:00

Two things were missing. First, nobody had measured this carefully for ears, across several datasets and model families. Second, Nested Learning, published at NeurIPS 2025, proposes a memory system that should resist forgetting, but it had only been shown on text.

So I ask three questions. Is forgetting real here? Does Nested Learning's memory reduce it? And if not, why not, and what would it take? The talk follows these in order. The third one is where most of the interesting work is.

**[next]**

---

## RQ1 · Is forgetting real? (1:45 – 4:50)

### Slide 4 · Transition (8 s) · 1:45

First question: is forgetting actually a problem in ear recognition?

**[next]**

### Slide 5 · A frozen, pre-registered benchmark (50 s) · 1:55

To answer it, I built a benchmark. On the left are the five model families I use, all ImageNet-pretrained: a classic CNN, ResNet-50; a modern CNN, ConvNeXt; a lightweight one, MobileNet; and two transformers, DeiT and Swin. They run on four public ear datasets.

In each dataset, the training people are split into five equal groups of new people, and the model learns them one group at a time. Those are the five tasks.

Forgetting is measured per old task: its best accuracy at any point, minus its accuracy at the end. If task 1 peaked at 0.88 and ended at 0.46, it forgot 0.42. Every decision rule was written down and committed before the runs it judges, and there are 1017 training runs in total.

**[next]**

### Slide 6 · Training: one shared recipe (40 s) · 2:45

Every model is trained the same way. The ear image goes through the pretrained backbone, a small neck turns it into a 512-dimensional ear embedding, and an ArcFace head gives the training loss, pulling images of the same person together. The head is only for training.

The tasks are learned in turn: each task trains only on its own new people, with a fixed number of steps. The upper bound is one joint model trained on everyone at once. The recipe on the right is identical for all models; only the learning rate and the step budget were tuned, once, and then frozen.

**[next]**

### Slide 7 · Evaluation: re-enroll everyone after every task (40 s) · 3:25

After every task, I test the model as a deployed system would be used. Everyone learned so far is enrolled again with the current model: their gallery images become one average embedding. Held-out images, never trained on, are matched to the nearest person. I also mix in people the model has never seen, to test rejection.

The rank-1 accuracy on each task fills one cell of a five-by-five accuracy matrix. *Optional:* Rank-1 is the main metric; open-set detection rates are reported too.

The key point is at the bottom right: because everyone is re-enrolled with the current model, forgetting here means the embedding itself has drifted.

**[next]**

### Slide 8 · Yes: every model family forgets on large data (45 s) · 4:05

The answer to the first question is yes. My rule was strict: forgetting counts only if it shows on every seed. It shows in 9 of the 20 model–dataset combinations.

On EarVN, the largest dataset, all five families forget heavily, between 0.33 and 0.48. AWE shows it for three families, AMI for one. IITD shows none, because accuracy there is near the ceiling and hides it. And CNN or transformer makes no difference.

*Point at the matrix.* Read down the first column: task 1 starts at 0.88 and slides to 0.46 as the later tasks arrive.

**[next]**

---

## RQ2 · Does Nested Learning help? (4:50 – 8:23)

### Slide 9 · Transition (8 s) · 4:50

So forgetting is real. Second question: can Nested Learning reduce it?

**[next]**

### Slide 10 · What is Nested Learning? (45 s) · 4:58

Nested Learning gives a new way to look at a network. Instead of one stack of layers trained by one optimiser, it sees the model as many memories, nested inside each other. Every part, the layers, the optimiser's state, attention, is a memory that compresses its own stream of data.

What separates these memories is how often they are updated. *Point at the dots.* A part updated every step is short-term memory. A part updated rarely is long-term memory.

And that is why it matters for us: what a fast part overwrites, a slow part can still hold. In principle, that is a way to learn continually without forgetting.

**[next]**

### Slide 11 · The Continuum Memory System (40 s) · 5:43

The Continuum Memory System, CMS, turns this idea into a layer. It takes one MLP block and replaces it with several copies that differ only in how often they are updated: in my setup, every step, every 16 steps, and every 256 steps.

Each copy is updated once per period, with the sum of the gradients it collected. The layer's output is a learned weighted sum of the copies. All copies start from the pretrained weights, so it can be dropped into a pretrained model. In the paper, this reduced forgetting compared with EWC on text.

**[next]**

### Slide 12 · The CMS model (45 s) · 6:23

This is the model I tested. On top, it is the same pipeline as before. The only change is inside the backbone. *Point at the dashed panel.* In every block, the rest of the block, attention or convolutions, stays as it was. But the pointwise layer, the one that plays the role of an MLP, becomes a CMS layer: three copies, updated every step, every 16 steps and every 256 steps, blended together.

The blend weights start equal, a third each, and all copies start as the pretrained layer. So at the start, the CMS model is exactly the original model. Depending on the family, this replaces between 12 and 32 layers.

**[next]**

### Slide 13 · How it trains, and how it is judged (35 s) · 7:08

The fast copy updates every step. The medium and slow copies collect their gradients and take one step every 16 and every 256 steps. Everything else updates every step, with the same recipe and budget as plain training, so the comparison is fair.

Before running anything, I fixed what success means. *Point at the right card.* Reduced: forgetting drops by at least 0.05, on every seed, without hurting accuracy. Fixed is stricter still. It is judged on the 9 cells where forgetting was shown.

**[next]**

### Slide 14 · 0 of 9 (40 s) · 7:43

*Let the number land for a second before speaking.*

Zero of nine. CMS did not reduce forgetting in a single cell. The largest drop anywhere was 0.036, below the 0.05 the rule asks for. I also added CMS on top of replay, and it helped in zero of twenty comparisons.

*Pause.*

This method was designed precisely to stop forgetting, and here it meets a problem with clear forgetting. So "it did not work" is not a good enough answer. The rest of the talk is about why.

**[next]**

---

## RQ3 · Why did it fail? (8:23 – 13:26)

### Slide 15 · Transition (8 s) · 8:23

Third question: why did a method built for this problem not work, and what would it take?

**[next]**

### Slide 16 · Clue: right place, idle memory (50 s) · 8:31

I treated this like an investigation. I saved the model after every task and took it apart, on EarVN ConvNeXt.

First clue: where does the forgetting happen? In the plain model, if I put back only the pointwise layers to how they were after task 1, 93% of the lost accuracy comes back. So CMS is sitting exactly where the forgetting is.

Second clue: *point at the chart.* After five tasks, the fast copy has moved about 10% from its starting weights. The slow copy has moved about 1%. Used alone, a slow copy barely recognises even the newest people. It never really learned anything, so it has nothing to remember.

**[next]**

### Slide 17 · Suspect A: the slow updates are too small (45 s) · 9:21

That gives the first suspect. The optimiser, AdamW, scales every step to a similar size. So a copy updated every 256 steps takes one normal-sized step, where Nested Learning's rule implies the effect of 256. It learns about 1/256 as much as intended.

My first two attempts went in this direction. R4 tied the slow updates to the tasks: 2 of 9 cells improved, a weak signal. R4b moved knowledge into the slow copies at the end of each task instead: 0 of 9. Neither settles it, so I tested the suspect directly.

**[next]**

### Slide 18 · Suspect A ruled out (50 s) · 10:06

In R5, the slow copies get full-size updates: an optimiser step on every gradient, applied to the copy once per period. Mechanically it works. The slow copies now move 15%, more than the fast copy.

But forgetting gets worse, not better. *Point at the table.* 0.445 against 0.411 for ResNet, 0.435 against 0.407 for ConvNeXt, and no seed improved.

The lesson is in the dark box: a slow copy that learns at full strength learns the newest people too, and forgets like everything else. A slower clock, by itself, does not protect memory. Suspect A is ruled out.

**[next]**

### Slide 19 · Suspect B: kept but drowned out (50 s) · 10:56

Second suspect: maybe the memory is there, but drowned out. To test it, I used an oracle, a deliberate cheat. At the end of task 1, I freeze two of the three copies at exactly the task-1 weights. A perfect memory of task 1.

It helps: task-1 forgetting drops from 0.445 to 0.272. But most of the loss remains. Here is the giveaway: a frozen copy used on its own still scores 0.79 on task 1. The whole model scores 0.58. The memory is intact. But the output is a blend, and the moving copy, along with the rest of the network, overrides it.

**[next]**

### Slide 20 · More weight on the old memory helps, at a price (45 s) · 11:46

If the memory is drowned out, giving it a bigger share should help. So I fixed the frozen copies' share of the output: 0.8, then 0.95, instead of the learned value near 0.6.

On ConvNeXt, it helps: task-1 forgetting falls from 0.27 to 0.23 to 0.17. But there is a price: accuracy on the newest people drops, because the model has less freedom to learn. On DeiT, nothing changes at all.

And when I take the 0.95 model apart, the remaining forgetting has simply moved, into the neck and the rest of the backbone. That points to the real culprit.

**[next]**

### Slide 21 · The real culprit (55 s) · 12:31

So I ran the perfect-memory oracle on all nine cells. *Point at the chart.* Task-1 forgetting falls everywhere, the orange bars are always lower, but large forgetting remains. Overall, it reduces forgetting in only 2 of 9 cells.

Then I took four of these models apart. *Point at the table.* Dilution shows up in three of them: a frozen copy alone beats the whole model. But look at the last column: except for the ConvNeXt, most of the lost accuracy sits in the rest of the backbone, attention, normalisation, embeddings, not in the CMS layers at all.

So the real culprit: most of the forgetting happens outside the layers CMS covers. Even a perfect memory there cannot stop it.

**[next]**

---

## The answer (13:26 – 15:11)

### Slide 22 · Why CMS failed, in one chain (50 s) · 13:26

Putting it together, as one chain. One: on CNNs, CMS is in the right place. Two: but its slow copies barely move, so they never learn the people. Three: when I make them learn properly, they forget like everything else. Four: even a perfect frozen memory is diluted in the blend, 0.79 alone, 0.58 together. Five: and on most models, most of the forgetting happens elsewhere anyway.

So, the answers. RQ1: yes, forgetting is real, and it does not depend on the architecture. RQ2: no, CMS reduced it in zero of nine cells. RQ3: not yet. A working version must cover more of the network, and let its memory actually reach the output.

**[next]**

### Slide 23 · Contributions, limits and next steps (45 s) · 14:16

To close. The contributions: a frozen benchmark for continual ear recognition; evidence that forgetting is real across model families; the first test of Nested Learning's memory on vision and biometrics; and diagnostics that show why it failed.

The limits: the largest dataset, UERC, is not included; EarVN has three seeds; the follow-up stages were designed after the main result, so they are exploratory; and the dilution tests use an oracle.

And the next steps follow from the findings: put CMS beyond the pointwise layers, choose between copies instead of blending them, and combine it with Learning without Forgetting, so new people can be enrolled without storing anyone's images.

**[next]**

### Slide 24 · Thank you (10 s) · 15:01

Thank you. In one line: forgetting in ear recognition is real, and CMS did not fix it, because its memory is diluted where it sits and most of the forgetting happens elsewhere. I'm happy to take questions.

---

## Likely questions, with short answers

**Why only rank-1 as the main metric?**
It is the standard identification metric and the one fixed before the runs as primary. Open-set detection rates and AUROC were also computed and are in the report; the verdicts are made on rank-1.

**Isn't 3 seeds on EarVN too few?**
It is a limitation, and I say so. But the rule required every seed to agree, which is strict. And EarVN has 3,732 held-out probes, about 750 per task, so per-run noise is small there.

**Did you tune CMS enough? Maybe other periods work.**
R3 tuned the slow learning-rate exponent per cell. R4 tried task-aligned periods, R4b consolidation, and R5 full-size updates. None reduced forgetting reliably. The oracle in R5b shows the ceiling: even a perfect memory in those layers reduces forgetting in only 2 of 9 cells (Backups B2, B3).

**Why put CMS only in the pointwise layers?**
That is where Nested Learning puts it: it replaces the MLP blocks. And on ConvNeXt the diagnostics show 93% of the forgetting is there. The finding that transformers forget mostly elsewhere is one of the results, and the obvious next step.

**Is the oracle fair?**
It is not a method; it uses the task boundary on purpose. It answers "what is the best a slow memory in these layers could possibly do?", and the answer is: not enough.

**What about LwF and replay?**
They work: on EarVN they cut forgetting by 0.17 to 0.29. CMS on top of replay added nothing in 0 of 20 comparisons. Combining CMS with LwF is in the next steps.

**Why is IITD showing no forgetting?**
Accuracy there is near the ceiling, and with few images per person the training images double as the gallery. So forgetting cannot show on rank-1.

**What was pre-registered and what was not?**
The benchmark, R2 and R3 were committed before their runs. R4 to R5b were designed after R3 was seen, so I label them exploratory throughout, though each was also committed before its own runs.
