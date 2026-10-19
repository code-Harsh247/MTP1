"""Continual-learning baselines for plan_continual.md (C2): EWC, LwF, replay.

Each method plugs into nlear.engine.train_steps through:
    extra_loss(model, x, emb, y)   added to the ArcFace loss during a task
    batch_fn(step)                 what each batch contains (replay mixes in old images)
    end_task(...)                  bookkeeping after a task (Fisher, snapshot, buffer)
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F

from .data.cache import to_input


class EWC:
    """Online Elastic Weight Consolidation (Kirkpatrick et al. 2017; Schwarz et al. 2018).

    penalty = lam/2 * sum_i F_i (theta_i - theta*_i)^2, where F is a running (decay `gamma`)
    diagonal Fisher estimated from squared gradients of the ArcFace loss on each finished
    task, and theta* are the weights after the latest task. Applied to the embedding model
    (backbone + neck), which is what recognition uses.
    """

    def __init__(self, lam, gamma=1.0, fisher_batches=30):
        self.lam, self.gamma, self.fisher_batches = lam, gamma, fisher_batches
        self.fisher, self.anchor = None, None

    def extra_loss(self, model, x, emb, y):
        if self.fisher is None:
            return emb.new_zeros(())
        pen = sum((f * (p - a).pow(2)).sum()
                  for p, f, a in zip(model.parameters(), self.fisher, self.anchor))
        return 0.5 * self.lam * pen

    def end_task(self, model, head, images, rows, labels, device, active, batch_size, gen):
        model.eval()
        fisher = [torch.zeros_like(p) for p in model.parameters()]
        for _ in range(self.fisher_batches):
            b = torch.randint(len(rows), (batch_size,), generator=gen)
            x = to_input(images[rows[b]], device)
            loss, _ = head(model(x), labels[b].to(device), active)
            grads = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
            for f, g in zip(fisher, grads):
                if g is not None:
                    f += g.detach().pow(2) / self.fisher_batches
        if self.fisher is None:
            self.fisher = fisher
        else:
            self.fisher = [self.gamma * old + new for old, new in zip(self.fisher, fisher)]
        self.anchor = [p.detach().clone() for p in model.parameters()]


class LwF:
    """Learning without Forgetting, as embedding distillation (standard for metric learning):
    keep the model from the end of the previous task frozen and penalize
    lam * mean ||norm(emb_new) - norm(emb_old)||^2 on the current batch. Stores no old data.
    """

    def __init__(self, lam):
        self.lam, self.old = lam, None

    def extra_loss(self, model, x, emb, y):
        if self.old is None:
            return emb.new_zeros(())
        with torch.no_grad():
            target = F.normalize(self.old(x).float(), dim=1)
        return self.lam * (F.normalize(emb.float(), dim=1) - target).pow(2).sum(1).mean()

    def end_task(self, model, *_, **__):
        self.old = copy.deepcopy(model).eval()
        for p in self.old.parameters():
            p.requires_grad_(False)


class Replay:
    """Experience replay: keep `per_id` training images of every learned identity; each batch
    is half current-task images, half buffer images (once the buffer is non-empty)."""

    def __init__(self, per_id, seed=0):
        self.per_id = per_id
        self.rng = np.random.default_rng(seed)
        self.rows = torch.empty(0, dtype=torch.long)
        self.labels = torch.empty(0, dtype=torch.long)

    def batch_fn(self, rows, labels, batch_size, gen):
        def fn(step):
            if len(self.rows) == 0:
                b = torch.randint(len(rows), (batch_size,), generator=gen)
                return rows[b], labels[b]
            n_cur = batch_size // 2
            b = torch.randint(len(rows), (n_cur,), generator=gen)
            r = torch.randint(len(self.rows), (batch_size - n_cur,), generator=gen)
            return torch.cat([rows[b], self.rows[r]]), torch.cat([labels[b], self.labels[r]])
        return fn

    def end_task(self, model, head, images, rows, labels, *_, **__):
        keep_r, keep_l = [], []
        for lab in labels.unique():
            idx = torch.where(labels == lab)[0].numpy()
            pick = self.rng.choice(idx, min(self.per_id, len(idx)), replace=False)
            keep_r.append(rows[pick])
            keep_l.append(labels[pick])
        self.rows = torch.cat([self.rows, *keep_r])
        self.labels = torch.cat([self.labels, *keep_l])
