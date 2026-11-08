"""Multi-frequency parameter updates for the Continuum Memory System ([NL] Eq. 71).

Parameters are assigned to levels with an update period C (in optimizer steps):

    theta^(l) <- theta^(l) - opt( sum of gradients over the last C^(l) steps )   every C^(l) steps
    theta^(l) unchanged                                                          otherwise

Period-1 parameters ("fast") go into the ordinary per-task optimizer that nlear.engine
steps every iteration. Every slower level gets its own persistent AdamW: its gradients are
simply NOT zeroed between steps, so .grad accumulates the window sum, and the scheduler's
hook steps and clears that level every C steps. The step counter runs across task
boundaries (one continuous data stream), and slow optimizers keep their state across tasks
because the slow levels are the long-term memory.

Mixed-precision loss scaling would mix different scales inside an accumulated window, so
CMS runs train in fp32 (and so does the same-architecture control).

Level assignment (cfg):
    body_period    backbone body                      (default 1)
    proj_period    neck projection + BN               (default 1)
    periods        one per CMS block, e.g. [1, 16, 256]
    lr_pow         slow-level lr = base_lr * period ** lr_pow   (default 0: same lr)
The ArcFace head is always fast (it is task-specific bookkeeping, not memory).
"""
import torch


class TaskConsolidation:
    """Study v2 R4b: knowledge transfer between CMS levels at task boundaries ([NL] Sec. 3.3, step 2:
    "how knowledge transfers between levels"), in place of slow gradient updates.

    Copy 1 of every CMSParallel module trains every step. The slower copies get no gradient. At the
    end of task k (0-based), copy l moves part of the way toward copy l-1 if (k+1) % every[l-1] == 0:

        theta_l <- theta_l + beta * (theta_{l-1} - theta_l)

    Slower copies are updated first, so each takes the previous state of the copy above it. A
    chain fast -> medium -> slow thus holds a running average of the fast copy's end-of-task
    weights at two time scales. At initialisation all copies are equal, so the model computes
    exactly the plain model until the first task ends.
    """

    def __init__(self, model, every, beta):
        from .models import CMSParallel
        self.mods = [m for m in model.body.modules() if isinstance(m, CMSParallel)]
        if not self.mods or any(len(m.copies) != len(every) + 1 for m in self.mods):
            raise ValueError("need CMSParallel modules with len(every) + 1 copies")
        self.every, self.beta = list(every), beta
        for m in self.mods:
            for c in m.copies[1:]:
                c.requires_grad_(False)
        self.n_updates = {l + 1: 0 for l in range(len(every))}

    @torch.no_grad()
    def end_task(self, k):
        for lvl in range(len(self.every), 0, -1):          # slowest first
            if (k + 1) % self.every[lvl - 1]:
                continue
            for m in self.mods:
                for slow, fast in zip(m.copies[lvl].parameters(), m.copies[lvl - 1].parameters()):
                    slow.lerp_(fast, self.beta)
            self.n_updates[lvl] += 1


class TaskSnapshot:
    """Study v2 R5, experiment 3: an upper bound on what a perfect slow memory can save.

    Copies 2.. of every CMSParallel module get no gradient. At the end of task `after` (1-based) every
    slow copy is set to copy 1's weights and then never changes again, so 2 of the 3 copies hold the
    model's pointwise layers exactly as they were after that task. Before that they stay at the
    pretrained weights, so the model computes the plain model until copy 1 starts to move.
    """

    def __init__(self, model, after):
        from .models import CMSParallel
        self.mods = [m for m in model.body.modules() if isinstance(m, CMSParallel)]
        if not self.mods or any(len(m.copies) < 2 for m in self.mods):
            raise ValueError("need CMSParallel modules with at least 2 copies")
        self.after = after
        for m in self.mods:
            for c in m.copies[1:]:
                c.requires_grad_(False)
        # keyed like TaskConsolidation: key l counts updates of copy l + 1
        self.n_updates = {l: 0 for l in range(1, len(self.mods[0].copies))}

    @torch.no_grad()
    def end_task(self, k):
        if k + 1 != self.after:
            return
        for m in self.mods:
            for slow in m.copies[1:]:
                for s, f in zip(slow.parameters(), m.copies[0].parameters()):
                    s.copy_(f)
        for l in self.n_updates:
            self.n_updates[l] += 1


class _PerStepSlow:
    """One slow level in "per-step" mode (R5, experiment 1b): a shadow copy of the level's weights
    takes an ordinary AdamW step on every step's gradient, which is computed at the live (unchanged)
    weights; every `period` steps the live weights are set to the shadow. So a period-C copy applies
    the sum of C ordinary optimizer updates at once, as in [NL] Eq. 71 with f = the optimizer's update,
    instead of one normalised step on the summed gradient."""

    def __init__(self, groups, weight_decay):
        self.live = [p for ps, _ in groups for p in ps]
        self.shadow = [p.detach().clone().requires_grad_(True) for p in self.live]
        it = iter(self.shadow)
        self.opt = torch.optim.AdamW([{"params": [next(it) for _ in ps], "lr": lr} for ps, lr in groups],
                                     weight_decay=weight_decay)

    def accumulate(self):
        for p, s in zip(self.live, self.shadow):
            s.grad = None if p.grad is None else p.grad.detach().clone()
            p.grad = None
        self.opt.step()

    @torch.no_grad()
    def apply(self):
        for p, s in zip(self.live, self.shadow):
            p.copy_(s)


class CMSScheduler:
    def __init__(self, model, head, t, periods, body_period=1, proj_period=1, lr_pow=0.0):
        if len(periods) != len(model.neck.blocks):
            raise ValueError(f"{len(periods)} periods for {len(model.neck.blocks)} CMS blocks")
        self.t = t
        groups = []  # (period, params, base_lr)
        groups.append((body_period, list(model.body.parameters()), t["lr_body"]))
        neck = model.neck
        groups.append((proj_period, list(neck.proj.parameters()) + list(neck.bn.parameters()), t["lr_head"]))
        for block, p in zip(neck.blocks, periods):
            groups.append((p, list(block.parameters()), t["lr_head"]))
        groups.append((1, list(head.parameters()), t["lr_head"]))
        self._setup(groups, t, lr_pow)

    @classmethod
    def for_body(cls, model, head, t, periods, lr_pow=0.0, slow_mode="sum"):
        """CMS inside the backbone (v1 Option B, v2 R3): copy l of every CMSParallel module
        (nlear.models.cms_targets) updates every periods[l] steps; everything else in the body
        (attention, depthwise/3x3 convs, embeddings, norms, aggregation weights), the neck and
        the head are fast."""
        from .models import CMSParallel
        self = cls.__new__(cls)
        mods = [m for m in model.body.modules() if isinstance(m, CMSParallel)]
        if not mods or any(len(m.copies) != len(periods) for m in mods):
            raise ValueError("model has no CMSParallel modules with that many levels")
        levels = [[p for m in mods for p in m.copies[i].parameters()] for i in range(len(periods))]
        in_levels = {id(p) for lvl in levels for p in lvl}
        rest = [p for p in model.body.parameters() if id(p) not in in_levels]
        groups = [(per, lvl, t["lr_body"]) for per, lvl in zip(periods, levels)]
        groups.append((1, rest, t["lr_body"]))
        groups.append((1, list(model.neck.parameters()) + list(head.parameters()), t["lr_head"]))
        self._setup(groups, t, lr_pow, slow_mode)
        return self

    for_vit = for_body   # v1 name

    def _setup(self, groups, t, lr_pow, slow_mode="sum"):
        """slow_mode "sum": one AdamW step on the gradients summed over the period (R3, R4).
        "per-step": an AdamW step on every gradient, applied to the model every period (R5)."""
        if slow_mode not in ("sum", "per-step"):
            raise ValueError(f"unknown slow_mode {slow_mode}")
        self.slow_mode = slow_mode
        self.t = t
        self.fast = []                       # [(params, lr)] stepped by the per-task optimizer
        by_period = {}
        for period, params, lr in groups:
            if period == 1:
                self.fast.append((params, lr))
            else:
                by_period.setdefault(period, []).append((params, lr * period ** lr_pow))
        if slow_mode == "per-step":
            self.slow = {period: _PerStepSlow(g, t["weight_decay"]) for period, g in sorted(by_period.items())}
        else:
            self.slow = {period: torch.optim.AdamW([{"params": ps, "lr": lr} for ps, lr in g],
                                                   weight_decay=t["weight_decay"])
                         for period, g in sorted(by_period.items())}
        self.global_step = 0
        self.n_updates = {p: 0 for p in self.slow}

    def fast_optimizer(self):
        """Fresh optimizer for the period-1 parameters (created per task, like the baselines)."""
        return torch.optim.AdamW([{"params": ps, "lr": lr} for ps, lr in self.fast],
                                 weight_decay=self.t["weight_decay"])

    def hook(self, step, scaler):
        """Called by nlear.engine.train_steps after backward, before the fast optimizer steps."""
        if scaler.is_enabled():
            raise RuntimeError("CMS needs fp32 training (set train.amp: false)")
        self.global_step += 1
        for period, opt in self.slow.items():
            if self.slow_mode == "per-step":
                opt.accumulate()
                if self.global_step % period == 0:
                    opt.apply()
                    self.n_updates[period] += 1
                continue
            if self.global_step % period == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                self.n_updates[period] += 1
