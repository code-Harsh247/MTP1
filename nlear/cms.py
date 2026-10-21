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
    def for_vit(cls, model, head, t, periods, lr_pow=0.0):
        """CMS inside a ViT (Option B): copy l of every block's CMSParallelMLP updates every
        periods[l] steps; attention, embeddings, norms, aggregation weights, neck and head are fast."""
        self = cls.__new__(cls)
        blocks = [b for b in model.body.blocks if hasattr(b.mlp, "copies")]
        if not blocks or len(blocks[0].mlp.copies) != len(periods):
            raise ValueError("model has no CMSParallelMLP blocks with that many levels")
        levels = [[p for b in blocks for p in b.mlp.copies[i].parameters()] for i in range(len(periods))]
        in_levels = {id(p) for lvl in levels for p in lvl}
        rest = [p for p in model.body.parameters() if id(p) not in in_levels]
        groups = [(per, lvl, t["lr_body"]) for per, lvl in zip(periods, levels)]
        groups.append((1, rest, t["lr_body"]))
        groups.append((1, list(model.neck.parameters()) + list(head.parameters()), t["lr_head"]))
        self._setup(groups, t, lr_pow)
        return self

    def _setup(self, groups, t, lr_pow):
        self.t = t
        self.fast = []                       # [(params, lr)] stepped by the per-task optimizer
        by_period = {}
        for period, params, lr in groups:
            if period == 1:
                self.fast.append((params, lr))
            else:
                by_period.setdefault(period, []).append((params, lr * period ** lr_pow))
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
            if self.global_step % period == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                self.n_updates[period] += 1
