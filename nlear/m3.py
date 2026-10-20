"""Multi-scale Momentum Muon (M3), Nested Learning paper Algorithm 1 ([NL] Sec. 7.2).

For every matrix parameter (conv kernels are viewed as [out, in*kh*kw]):

    M1 <- b1*M1 + g                         fast momentum, every step           (Alg. 1 line 7)
    V  <- b2*V + (1-b2)*g^2                 second moment, every step           (line 8, Adam-style)
    M2 <- b3*M2 + sum(g over last f steps)  slow momentum, every f steps (CMS)  (line 3)
    O1 = NewtonSchulz(M1), O2 = NewtonSchulz(M2)                                (lines 4, 9)
    W  <- W - lr * (O1 + alpha*O2) / (sqrt(V_hat) + eps)                        (line 10)
  (+ decoupled weight decay W <- W*(1 - lr*wd), as in AdamW)

Interpretation notes (the paper gives the algorithm but no hyper-parameter values):
* Alg. 1 writes M1 <- M1 + b1*g, M2 <- M2 + b3*sum g, V <- V + b2*g^2 while calling the b's
  "momentum factors < 1"; we read them as the usual decaying momenta (Muon / Adam form),
  with Adam's bias correction on V.
* alpha = 0 removes the slow (CMS) memory: the "no multi-scale" control.
* Newton-Schulz: Muon's quintic iteration (a, b, c) = (3.4445, -4.7750, 2.0315), 5 steps.
"""
import torch


def newton_schulz(G, steps=5, eps=1e-7):
    """Approximately orthogonalize a 2-D matrix (Muon's quintic Newton-Schulz)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if transposed else X


class M3(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.95, 0.999, 0.99), alpha=1.0, freq=16,
                 weight_decay=0.0, eps=1e-8, ns_steps=5):
        defaults = dict(lr=lr, betas=betas, alpha=alpha, freq=freq, weight_decay=weight_decay,
                        eps=eps, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2, b3 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim < 2:
                    raise ValueError("M3 is for matrix parameters; put biases/norms in AdamW")
                g = p.grad.reshape(p.shape[0], -1).float()
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["M1"] = torch.zeros_like(g)
                    st["M2"] = torch.zeros_like(g)
                    st["V"] = torch.zeros_like(g)
                    st["acc"] = torch.zeros_like(g)
                    st["O2"] = torch.zeros_like(g)
                st["step"] += 1
                t = st["step"]
                st["M1"].mul_(b1).add_(g)
                st["V"].mul_(b2).addcmul_(g, g, value=1 - b2)
                st["acc"].add_(g)
                if group["alpha"] > 0 and t % group["freq"] == 0:      # slow memory (CMS level)
                    st["M2"].mul_(b3).add_(st["acc"])
                    st["acc"].zero_()
                    st["O2"] = newton_schulz(st["M2"], group["ns_steps"])
                elif group["alpha"] == 0:
                    st["acc"].zero_()
                o = newton_schulz(st["M1"], group["ns_steps"])
                if group["alpha"] > 0:
                    o = o + group["alpha"] * st["O2"]
                v_hat = st["V"] / (1 - b2 ** t)
                update = o / (v_hat.sqrt() + group["eps"])
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.view_as(p).to(p.dtype), alpha=-group["lr"])
        return loss


def split_params(model, extra_adamw=()):
    """(matrix params for M3, other params for AdamW) in the Muon convention."""
    mats, rest = [], list(extra_adamw)
    for p in model.parameters():
        (mats if p.ndim >= 2 else rest).append(p)
    return mats, rest
