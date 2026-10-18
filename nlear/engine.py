"""Shared training/embedding loop used by the continual-learning trainer (plan_continual.md).

train_steps() runs `steps` ArcFace updates on the given image rows with a warmup + cosine
schedule. Continual-learning methods plug in through two hooks:
    batch_fn(step) -> (rows, labels)   what each batch contains (e.g. replay mixes old rows in)
    extra_loss(model, x, emb, y)       added to the ArcFace loss (e.g. EWC penalty, distillation)
"""
import math

import torch

from .data.cache import to_input


def lr_factor(step, total, warm):
    if step < warm:
        return (step + 1) / max(1, warm)
    return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


@torch.no_grad()
def embed(model, images, rows, device, bs=256):
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        x = to_input(images[rows[i:i + bs]], device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(x).float())
    return torch.cat(out) if out else torch.empty(0)


def train_steps(model, head, opt, images, batch_fn, steps, device, augment, scaler,
                warm=0, active=None, extra_loss=None, step_hook=None):
    """Run `steps` optimizer steps. Returns mean loss and train accuracy over the run."""
    base_lrs = [g.get("initial_lr", g["lr"]) for g in opt.param_groups]
    for g, lr in zip(opt.param_groups, base_lrs):
        g["initial_lr"] = lr
    model.train()
    head.train()
    loss_sum = acc_sum = n_sum = 0.0
    for step in range(steps):
        f = lr_factor(step, steps, warm)
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * f
        rows, y = batch_fn(step)
        x = to_input(images[rows], device, augment)
        y = y.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
            emb = model(x)
            loss, cos = head(emb, y, active)
            if extra_loss is not None:
                loss = loss + extra_loss(model, x, emb, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if step_hook is not None:      # e.g. CMS: decide which levels may update this step
            step_hook(step, scaler)
        scaler.step(opt)
        scaler.update()
        loss_sum += loss.item() * len(y)
        acc_sum += (cos.argmax(1) == y).float().sum().item()
        n_sum += len(y)
    return {"loss": loss_sum / max(1, n_sum), "train_acc": acc_sum / max(1, n_sum)}
