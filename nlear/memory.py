"""Per-identity associative memories written Titans-style; surprise = open-set score.

Equations (notes/nl_equations.md):
    k = norm(z W_K), v = z W_V                                   [T] Eq. 11 (q = k here)
    l(M; k, v) = || M(k) - v ||^2                                [T] Eq. 12
    S_t = eta * S_{t-1} - theta * grad l(M_{t-1})                [T] Eq. 14 (momentum = past surprise)
    M_t = (1 - alpha) * M_{t-1} + S_t                            [T] Eq. 13 (alpha = forget gate)
    surprise_i(z) = l(M_i; k(z), v(z))                           (the open-set score, negated)

Each enrolled identity i gets its own memory M_i, starting from a shared initial state M_0
(meta-learned in arm 4, fixed otherwise) and written with `steps` full-batch gradient steps
on that identity's gallery embeddings. Memories are stored batched ([n_ids, ...]) so all
identities are written in parallel; because they are independent, the gradient of the
summed loss w.r.t. the batched weights is exactly each identity's own gradient.

Architectures: "linear" (M(k) = W k; with M_0 = 0, k = v and plain GD this converges to the
projection onto the gallery span = the subspace baseline) or "mlp" (M(k) = W1 silu(W2 k),
Titans' deep memory without the residual, which would be the identity trap when k = v).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_sigmoid(p):
    return math.log(p / (1 - p))


def _inv_softplus(x):
    return math.log(math.expm1(x))


class AssocMemory(nn.Module):
    def __init__(self, d_in, arch="mlp", hidden=256, d_key=None, d_val=None, steps=5,
                 lr=1.0, momentum=0.9, forget=0.0, learn_proj=False, learn_init=False,
                 learn_rates=False, init_scale=0.1, seed=0, proj_init="orthogonal", proj_act=True):
        super().__init__()
        self.arch, self.steps = arch, steps
        self.learn_proj, self.proj_act = learn_proj, proj_act
        d_key = d_key or d_in
        d_val = d_val or d_in
        if learn_proj:
            self.W_K = nn.Linear(d_in, d_key, bias=False)
            self.W_V = nn.Linear(d_in, d_val, bias=False)
            if proj_init == "identity":
                # Start exactly at the untrained memory (k = v = z): meta-training moves from there.
                if d_key != d_in or d_val != d_in:
                    raise ValueError("identity projection init needs d_key = d_val = d_in")
                nn.init.eye_(self.W_K.weight)
                nn.init.eye_(self.W_V.weight)
            else:
                nn.init.orthogonal_(self.W_K.weight)
                nn.init.orthogonal_(self.W_V.weight)  # different random init: W_K != W_V
        elif d_key != d_in or d_val != d_in:
            raise ValueError("fixed projections are the identity, so d_key = d_val = d_in")
        self.d_key, self.d_val = d_key, d_val

        g = torch.Generator().manual_seed(seed)
        if arch == "linear":
            init = [torch.zeros(d_val, d_key)]
        elif arch == "mlp":
            # W1 = 0 so M_0 outputs 0 (like the linear memory); W2 small random so gradients flow
            init = [torch.randn(hidden, d_key, generator=g) * init_scale / math.sqrt(d_key),
                    torch.zeros(d_val, hidden)]
        else:
            raise ValueError(arch)
        self._n_init = len(init)
        for i, w in enumerate(init):
            if learn_init:
                setattr(self, f"init{i}", nn.Parameter(w))
            else:
                self.register_buffer(f"init{i}", w)

        self.lr_max = max(1.0, 2 * lr)
        raw_lr = _inv_sigmoid(lr / self.lr_max) if learn_rates else _inv_softplus(lr)
        raw = {"lr": raw_lr, "momentum": _inv_sigmoid(min(max(momentum, 1e-4), 1 - 1e-4)),
               "forget": _inv_sigmoid(min(max(forget, 1e-4), 1 - 1e-4))}
        for k, v in raw.items():
            t = torch.tensor(float(v))
            if learn_rates:
                setattr(self, f"raw_{k}", nn.Parameter(t))
            else:
                self.register_buffer(f"raw_{k}", t)
        self._fixed = {"momentum": momentum, "forget": forget}

    @property
    def init(self):
        """Initial memory state M_0 (read live, so .to(device) is respected)."""
        return [getattr(self, f"init{i}") for i in range(self._n_init)]

    # rates (theta = lr, eta = momentum, alpha = forget); exact 0 kept when not learned.
    # A learned write lr is bounded to (0, lr_max): unrolled meta-gradients otherwise push it
    # into the range where the 10-step write diverges (seen in Phase 3: NaN after ~100 episodes).
    def rates(self):
        if isinstance(self.raw_lr, nn.Parameter):
            lr = self.lr_max * torch.sigmoid(self.raw_lr)
        else:
            lr = F.softplus(self.raw_lr)
        mom = torch.sigmoid(self.raw_momentum) if self.raw_momentum.requires_grad or self._fixed["momentum"] > 0 \
            else torch.zeros(())
        fgt = torch.sigmoid(self.raw_forget) if self.raw_forget.requires_grad or self._fixed["forget"] > 0 \
            else torch.zeros(())
        return lr, mom, fgt

    def keys_values(self, z):
        z = F.normalize(z, dim=-1)
        if self.learn_proj:
            act = F.silu if self.proj_act else (lambda x: x)
            return F.normalize(act(self.W_K(z)), dim=-1), act(self.W_V(z))
        return z, z

    def apply(self, params, k):
        """k [m, dk] (shared probes) or [n, g, dk] (per-identity) -> outputs from every memory."""
        if self.arch == "linear":
            (W,) = params
            return torch.einsum("nvk,mk->mnv", W, k) if k.dim() == 2 else torch.einsum("nvk,ngk->ngv", W, k)
        W2, W1 = params
        if k.dim() == 2:
            return torch.einsum("nvh,mnh->mnv", W1, F.silu(torch.einsum("nhk,mk->mnh", W2, k)))
        return torch.einsum("nvh,ngh->ngv", W1, F.silu(torch.einsum("nhk,ngk->ngh", W2, k)))

    def write(self, Zg, create_graph=False):
        """Zg [n_ids, g, d_in] gallery embeddings -> per-identity memory weights."""
        K, V = self.keys_values(Zg)
        n = Zg.shape[0]
        params = [w.unsqueeze(0).expand(n, *w.shape).clone() for w in self.init]
        # The inner loop differentiates w.r.t. the memory weights, so they must require grad
        # even when M_0 itself is fixed (a buffer or a frozen parameter).
        params = [p if (create_graph and p.requires_grad) else p.detach().requires_grad_(True)
                  for p in params]
        lr, mom, fgt = self.rates()
        S = [torch.zeros_like(p) for p in params]
        with torch.enable_grad():
            for _ in range(self.steps):
                loss = (self.apply(params, K) - V).pow(2).sum(-1).mean(-1).sum()  # per-id mean, summed
                grads = torch.autograd.grad(loss, params, create_graph=create_graph)
                S = [mom * s - lr * gr for s, gr in zip(S, grads)]
                params = [(1 - fgt) * p + s for p, s in zip(params, S)]
                if not create_graph:
                    params = [p.detach().requires_grad_(True) for p in params]
                    S = [s.detach() for s in S]
        return params if create_graph else [p.detach() for p in params]

    def surprise(self, params, Zp, chunk=1024):
        """Zp [m, d_in] probes -> [m, n_ids] associative loss of every probe in every memory."""
        out = []
        for i in range(0, Zp.shape[0], chunk):
            k, v = self.keys_values(Zp[i:i + chunk])
            out.append((self.apply(params, k) - v.unsqueeze(1)).pow(2).sum(-1))
        return torch.cat(out)


def group_gallery(G, gl):
    """Gallery rows grouped per identity -> (classes, Zg [n_ids, g, d]); needs equal g per id."""
    classes = sorted(set(gl))
    groups = [G[[j for j, c in enumerate(gl) if c == cls]] for cls in classes]
    sizes = {len(x) for x in groups}
    if len(sizes) != 1:
        raise ValueError(f"unequal gallery sizes per identity: {sorted(sizes)}")
    return classes, torch.stack(groups)


def make_memory_scorer(memory, device="cpu"):
    """Scorer compatible with nlear.eval.protocol: S = -surprise (higher = more familiar)."""
    def fn(G, gl, P, **_):
        classes, Zg = group_gallery(G, gl)
        mem = memory.to(device)
        with torch.no_grad():
            params = mem.write(Zg.to(device).float())
            S = -mem.surprise(params, P.to(device).float())
        return S.cpu(), classes
    return fn
