# Nested Learning & Titans: Equations We Need

*Phase 1 notes. Written 2026-10-18.*

Sources:

- **[NL]** Behrouz, Razaviyayn, Zhong, Mirrokni. *Nested Learning: The Illusion of Deep Learning Architectures.* NeurIPS 2025, arXiv:2512.24695. File: `papers/nested_learning_2512.24695.pdf`.
- **[T]** Behrouz, Zhong, Mirrokni. *Titans: Learning to Memorize at Test Time.* arXiv:2501.00663. File: `papers/titans_2501.00663.pdf`.

Equation numbers below are the papers' own, so every formula can be traced back. **No official code exists for either paper** (checked 2026-10-18). Our implementation follows these equations directly.

---

## 1. Associative memory: the building block

**[NL] Def. 1 (Eq. 6).** An associative memory is an operator M(·) that maps keys K to values V. It is learned by minimizing an objective L̃:

```
M* = argmin_M  L̃( M(K) ; V )
```

The paper's own definitions (NL §3.1):

- **Memory** is a neural update caused by an input.
- **Learning** is the process of acquiring an effective and useful memory.

In NL, *everything* is an associative memory: layers trained by backprop, momentum, Adam, attention, and MLP blocks. Each one compresses its own **context flow**, meaning the data it sees.

## 2. Levels and update frequency

**[NL] Def. 2.** The **update frequency** f_A of a component A is its number of updates per unit of time, where one unit is one update step on one data point.

**[NL] Def. 3 and 4 (Eq. 19–20).** A **nested system** has K ordered levels:

- Components in the same level share an update frequency.
- A higher level has a lower frequency.
- Each level has optimization problems (L, C, Θ): an objective, a context, and a parameter set. Each problem has **its own gradient flow**.
- In a Nested System of Associative Memories (NSAM), every context C is a set of key–value pairs, and each memory is updated by gradient descent on its own objective.

**Knowledge transfer between levels ([NL] §3.3).** The ways one level can pass information to another:

| Type | Form | Example |
|---|---|---|
| Direct, parametric (Eq. 24–25) | M⁽⁰⁾(· ; θ⁽¹⁾) or M⁽⁰⁾(· ; M⁽¹⁾(·)) | Fast weights / linear attention |
| Direct, non-parametric (Eq. 27) | M⁽⁰⁾(· ; C⁽¹⁾) | Softmax attention |
| Via backprop | Both levels share one gradient flow, at different frequencies | CMS |
| Via initialization (Eq. 28) | The slow level meta-learns the initial state of the fast level | MAML, meta-learned M₀ |
| Via generation | One level generates the weights or context of another | Hypernetworks; architecture → gradients → optimizer |

> **Design recipe ([NL] §3.3):** (1) choose the optimization problems and their frequencies, then (2) choose how knowledge transfers between levels.

## 3. Surprise ([T] §3.1, [NL] §3.1)

**Plain surprise ([T] Eq. 8).** The surprise is the gradient of the memory's loss on the new input:

```
M_t = M_{t-1} − θ_t ∇ℓ(M_{t-1}; x_t)          ← ∇ℓ is "surprise"
```

**Associative loss ([T] Eq. 11–12):**

```
k_t = x_t W_K ,   v_t = x_t W_V
ℓ(M_{t-1}; x_t) = ‖ M_{t-1}(k_t) − v_t ‖²₂
```

- W_K and W_V are **outer-loop** parameters, treated as hyperparameters of the inner loop.
- M's weights are **inner-loop** parameters, learned at test time.

**Momentum and forgetting ([T] Eq. 13–14):**

```
S_t = η_t S_{t-1} − θ_t ∇ℓ(M_{t-1}; x_t)      ← past surprise + momentary surprise
M_t = (1 − α_t) M_{t-1} + S_t                  ← α_t ∈ [0,1] is the forget gate (weight decay)
```

- η_t (surprise decay), θ_t (learning rate) and α_t (forget gate) are **data-dependent**: functions of x_t, learned in the outer loop.

**Retrieval ([T] Eq. 15).** Retrieval is a forward pass with no update:

```
q_t = x_t W_Q ,   y_t = M*(q_t)
```

**Local Surprise Signal ([NL] Eq. 8–9).** In NL's terms, backprop itself is an associative memory. It maps each input to its **Local Surprise Signal (LSS)**, ∇_y L: the error of that input's prediction, which is zero when the loss is minimized.

**Implementation details ([T] §4.4):**

- residual connections in all blocks
- SiLU activation on q/k/v
- ℓ2-normalize q and k
- a 1-D depthwise conv after q/k/v (it helps, but this is sequence-specific)
- output gating and normalization

**Ablations ([T] Table 5).** All components help. The biggest losses come from removing:

1. weight decay (forget gate)
2. momentum
3. convolution
4. persistent memory

Deep memory (L_M ≥ 2) beats linear memory.

## 4. Delta-rule updates ([NL] §4.4–4.5, §8.1)

**Plain GD (dot-product objective).** The update is Hebbian and independent of the memory's current state:

```
M_t = M_{t-1} + v_t k_tᵀ
```

**Delta Gradient Descent (DGD) with weight decay ([NL] Eq. 88).** This is what Hope uses:

```
M_t = M_{t-1} (α_t I − η_t k_t k_tᵀ) − η_t ∇L(M_{t-1}; k_t, v̂_t)
```

**Linear memory with ℓ2 loss ([NL] Eq. 93):**

```
M_t = M_{t-1}(α_t I − η_t k_t k_tᵀ) − η_t (M_{t-1} k_t − v̂_t) k_tᵀ
```

The k kᵀ term lets the memory **overwrite** an old association along the current key direction instead of only adding to it. That is better capacity management.

**Momentum variants ([NL] Eq. 46–52):**

- Delta Momentum (Eq. 48–49)
- Deep Momentum GD, "DMGD", which replaces the momentum matrix with an MLP (Eq. 50)
- nonlinear output on the momentum: with Newton–Schulz, this recovers **Muon** (Eq. 52)

## 5. Continuum Memory System (CMS) ([NL] §7)

**Chain of MLPs at different frequencies ([NL] Eq. 70–71).** CMS is a chain of MLP blocks MLP^(f₁) … MLP^(f_k). Block ℓ has chunk size C⁽ℓ⁾ = max_i C⁽ⁱ⁾ / f_ℓ:

```
y_t = MLP^(f_k)( MLP^(f_{k-1})( … MLP^(f_1)(x_t) ) )

θ^(fℓ)_{i+1} = θ^(fℓ)_i − Σ_{t=i−C⁽ℓ⁾}^{i} η_t⁽ℓ⁾ f(θ_t^(fℓ); x_t)    if i ≡ 0 (mod C⁽ℓ⁾)
             = θ^(fℓ)_i                                              otherwise
```

- f(·) is the optimizer's error term, e.g. ∇L(θ; x_t) with the **task objective**.
- A standard Transformer MLP is the special case k = 1 with frequency 0.

**Variants:**

- **Nested** (Eq. 72): level ℓ meta-learns the initial state of level ℓ+1, and that level is re-initialized after each context.
- **Sequential** (Eq. 73): blocks run in series, and all initial states are learned by backprop at the lowest frequency.
- **Independent / head-wise** (Eq. 74): `y_t = Agg(MLP^(f_k)(x_t), …, MLP^(f_1)(x_t))`, e.g. a learnable weighted sum.

**Why it resists forgetting.** Knowledge overwritten in a fast block survives in slower blocks, and backprop can carry it back.

**Pretrained initialization ([NL] §7.3).** CMS blocks can be **initialized from a pretrained model's MLPs**: MLP₀⁽ℓ⁾ = MLP_pretrained. A small inner learning rate keeps a block close to the pretrained weights. ← Directly usable with a pretrained ViT.

## 6. Hope = self-modifying Titans + CMS ([NL] §8)

**Self-referential Titans ([NL] Eq. 86–88).** Every projection is itself a memory and adapts in context. Only q is fixed:

```
y_t = M_memory,t-1(q_t),  k_t = M_k,t-1(x_t),  v_t = M_v,t-1(x_t),
η_t = M_η,t-1(x_t),       α_t = M_α,t-1(x_t),          q_t = x_t W_q  (only fixed projection)
v̂_□,t = M_□,t-1(v_t)                                     (each memory generates its own target)
M_□,t = M_□,t-1(α_t I − η_t k_t k_tᵀ) − η_t ∇L(M_□,t-1; k_t, v̂_□,t),   □ ∈ {k, v, q, η, α, memory}
```

- **Memory architecture (Eq. 89):** a residual 2-layer MLP, `M_□(x) = x + W_□,1 σ(W_□,2 x)`.
- The **initial states M_□,0 of all memories are meta-learned** in the outer loop.
- **Chunked training (Eq. 90):** gradients are taken with respect to the previous chunk's final state, so the work within a chunk is parallel.

**Hope (Eq. 94–97).** Hope is self-modifying Titans followed by a CMS chain. It ℓ2-normalizes q and k and uses a local conv with window 4. **Hope-Attention** swaps the Titans part for softmax attention.

**Hope ablations ([NL] Table 6).** Each of these hurts:

- DGD → plain GD
- removing momentum
- removing weight decay
- removing CMS
- freezing the k/v/q projections at the slowest level

## 7. What the papers actually tested (relevant evidence)

- **Class-incremental learning** ([NL] §9.1, Fig. 6) on CLINC, Banking and DBpedia. These are text-intent datasets, and **CLINC includes out-of-scope detection**. Llama-3 MLPs converted to CMS beat ICL, EWC and InCA.
- **More levels → better continual and in-context learning.** A higher lowest-frequency gives a more adaptive but weaker long-term memory ([NL] Fig. 7, Table 8).
- **The only vision result: the M3 optimizer** (a CMS applied to momentum) trains ViT on ImageNet-21K (24M and 86M parameters). It gets better train/test loss than AdamW and Muon ([NL] §9.7, Fig. 11).
- No experiment covers biometrics, open-set recognition, or using surprise as an out-of-distribution score. **That gap is our contribution space.**

---

## 8. Mapping to ear recognition: decisions and caveats

### 8.1 What "time" and "context" mean for us

NL is defined over a stream of data points. For ears, there are three possible context flows:

| Context flow | Level | Update frequency |
|---|---|---|
| Training set (all training identities) | Backbone, W_K / W_V / W_Q, η / α / M₀ | Lowest (pre-training) |
| **Enrollment stream:** gallery images of each person, then a stream of people over time | Identity memory | Medium (per enrollment) |
| Patch tokens inside one image | Optional fast level (Hope over patches) | Highest (per image) |

The **enrollment stream** is the natural place for NL in our problem. It is a real stream, it grows over time, and forgetting matters there.

### 8.2 Surprise as the open-set score: precise definition

For a probe embedding z and each enrolled identity i with memory M_i:

```
k = norm(SiLU(z W_K)),   v = SiLU(z W_V)
s_i(z) = ‖ M_i(k) − v ‖²₂                    (loss form, Titans Eq. 12)
   or    ‖ ∇_{M_i} ℓ(M_i; z) ‖               (gradient form, Titans Eq. 8)
decision: î = argmin_i s_i ;  "unknown" if s_î > τ   (τ calibrated on validation identities)
```

**Linear memory with ℓ2-normalized keys.** Here ∇ℓ = 2(Mk − v)kᵀ, so ‖∇ℓ‖ = 2‖Mk − v‖·‖k‖ = 2‖Mk − v‖. The loss form and gradient form give the same ranking. They differ only for deep (MLP) memories, so that comparison is worth an ablation.

### 8.3 Two traps found while reading (important)

**1. The identity trap.** If W_K = W_V, or the memory can learn something close to the identity map, then M(k) ≈ v for *every* input. Surprise is then low for unknowns too, and rejection fails.

Fixes:
- Keep W_K ≠ W_V.
- Keep memories low-capacity.
- Use the forget gate.
- **Meta-train** with unknown identities in each episode, with a loss that pushes s(unknown) up and s(known) down. This is the key outer-loop objective.

**2. The linear memory is just a subspace method.** A linear M_i written by the delta rule from M₀ = 0 converges to a projection onto the span of that person's gallery keys. The surprise is then the residual distance of v from that subspace, which is essentially the classic **subspace classifier** (CLAFIC/OSM).

Therefore:
- (a) Add a **subspace-residual baseline** to Phase 2, because the claim must beat it.
- (b) The NL-specific gain has to come from what a subspace method cannot do: deep memory, a meta-learned M₀, learned η/α, momentum, and possibly self-generated values (Hope Eq. 87).

### 8.4 Which NL components we use

| Component | Use in NL-Ear | Phase |
|---|---|---|
| Titans memory, Eq. 11–15 + delta rule, Eq. 93 | Per-identity memory M_i, written at enrollment | 3–4 |
| Surprise, Eq. 8 / 12 | Open-set rejection score | 3–4 |
| Momentum + forget gate, Eq. 13–14 | Memory write rule; ablate both | 4 |
| Meta-learned M₀, η, α, W_K, W_V (Eq. 28, Hope) | Outer-loop training on training identities | 4 |
| Deep memory, Eq. 89 (residual 2-layer MLP) | Compare with linear memory (vs. the subspace baseline) | 4 |
| CMS, Eq. 70–74, pretrained init §7.3 | Backbone MLP blocks; continual enrollment experiment | 5 |
| M3 optimizer | Optional backbone-training ablation (only vision result in [NL]) | 5 |
| Self-referential Hope over patches | Optional per-image fast level | 5 (stretch) |

### 8.5 Updates to plan.md that follow from this

- Phase 2: add the **subspace-residual baseline** next to max-cosine and EVM.
- Phase 3 kill test: compare four scores: (1) max-cosine, (2) linear-memory surprise (≈ subspace), (3) deep-memory surprise *without* meta-training, (4) deep-memory surprise *with* the episodic meta-training described above. A gain only for (4) still counts as an NL result, since meta-learning the memory's write rule is the NL contribution.
