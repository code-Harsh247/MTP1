"""Phase 3 arm 4 (meta-trained NL memory) and arm 4b (control: same training, no memory).

Meta-trains on the embeddings of the held-out "meta" identities of a split-backbone run
(people the backbone never saw). Each episode mimics the open-set test: n_known enrolled
identities (n_gallery images each) + n_unknown never-enrolled identities, q probes each.

  --model memory : Titans-style per-identity memory. Meta-learned: W_K, W_V, initial state
                   M_0, write-rule lr / momentum / forget gate. Score = -surprise.
  --model proto  : control. Same episodes, same loss, but only a learned linear projection
                   scored by cosine to the identity prototype. Any gain of `memory` over
                   `proto` is due to the memory, not to extra training on more people.

Loss: open-set softmax over [a*score_1 .. a*score_N, b] (b = learned "unknown" logit).
Model selection: best val DIR@FAR=5% (cosine-style protocol on the real val split);
test is evaluated once with the selected checkpoint.

    python scripts/meta_train.py --run runs/earvn/splitbb_resnet50/seed0 --model memory
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nlear.eval.protocol import run_protocol, source_blobs, variant_blobs  # noqa: E402
from nlear.memory import AssocMemory, group_gallery, make_memory_scorer  # noqa: E402


class ProtoHead(nn.Module):
    """Control model: learned projection + cosine to the mean gallery embedding."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W = nn.Linear(d_in, d_out, bias=False)
        nn.init.orthogonal_(self.W.weight)

    def scores(self, Zg, P):
        protos = F.normalize(F.normalize(self.W(F.normalize(Zg, dim=-1)), dim=-1).mean(1), dim=-1)
        return F.normalize(self.W(F.normalize(P, dim=-1)), dim=-1) @ protos.T


def episode_scores(model, Zg, P, kind):
    if kind == "memory":
        params = model.write(Zg, create_graph=True)
        return -model.surprise(params, P)
    return model.scores(Zg, P)


def make_scorer(model, kind, device):
    if kind == "memory":
        return make_memory_scorer(model, device)

    def fn(G, gl, P, **_):
        classes, Zg = group_gallery(G, gl)
        with torch.no_grad():
            S = model.to(device).scores(Zg.to(device).float(), P.to(device).float())
        return S.cpu(), classes
    return fn


class Sampler:
    def __init__(self, emb, identity, mask, rng):
        self.emb, self.rng = emb, rng
        self.by_id = {c: np.where((identity == c) & mask)[0] for c in np.unique(identity[mask])}
        self.ids = sorted(self.by_id)

    def episode(self, n_known, n_unknown, n_gallery, q):
        sel = self.rng.choice(self.ids, n_known + n_unknown, replace=False)
        gal, prb, tgt = [], [], []
        for j, c in enumerate(sel):
            idx = self.rng.permutation(self.by_id[c])
            if j < n_known:
                gal.append(idx[:n_gallery])
                idx = idx[n_gallery:]
            take = idx[:q]
            prb.extend(take)
            tgt.extend([j if j < n_known else n_known] * len(take))
        Zg = self.emb[torch.as_tensor(np.stack(gal))]
        return Zg, self.emb[torch.as_tensor(np.array(prb))], torch.tensor(tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="split-backbone run dir with embeddings.pt")
    ap.add_argument("--model", choices=["memory", "proto"], default="memory")
    ap.add_argument("--tag", default="")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-known", type=int, default=8)
    ap.add_argument("--n-unknown", type=int, default=8)
    ap.add_argument("--n-gallery", type=int, default=2)
    ap.add_argument("--q", type=int, default=10)
    # memory hyperparameters. Defaults = the best untrained memory (arm 3, chosen on val):
    # identity projections, 512 hidden, 10 write steps, lr 0.5, momentum 0.9, forget 0.3.
    ap.add_argument("--d-key", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--proj-init", choices=["identity", "orthogonal"], default="identity")
    ap.add_argument("--freeze", nargs="*", default=[], choices=["proj", "init", "rates"],
                    help="memory parts NOT meta-learned (they stay at their start values)")
    ap.add_argument("--mem-lr", type=float, default=0.5)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--forget", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--by-source", action="store_true", help="also score each source separately")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Path(args.run)
    cfg = yaml.safe_load(open(run / "config.yaml"))
    blob = torch.load(run / "embeddings.pt", weights_only=False)
    assert (blob["split"] == "meta").any(), "not a split-backbone run (no split='meta' rows)"
    emb = blob["emb"].float().to(device)
    sampler = Sampler(emb, blob["identity"], blob["split"] == "meta", np.random.default_rng(args.seed))
    print(f"meta pool: {len(sampler.ids)} identities; model={args.model}")

    if args.model == "memory":
        model = AssocMemory(emb.shape[1], arch="mlp", hidden=args.hidden, d_key=args.d_key,
                            d_val=args.d_key, steps=args.steps, lr=args.mem_lr, momentum=args.momentum,
                            forget=args.forget, learn_proj=True, learn_init=True, learn_rates=True,
                            seed=args.seed, proj_init=args.proj_init,
                            proj_act=args.proj_init != "identity")
    else:
        model = ProtoHead(emb.shape[1], args.d_key)
        if args.proj_init == "identity" and args.d_key == emb.shape[1]:
            nn.init.eye_(model.W.weight)  # starts exactly at cosine_proto
    model.to(device)
    for pname, p in model.named_parameters():
        part = ("proj" if pname.startswith(("W_K", "W_V", "W.")) else
                "init" if pname.startswith("init") else "rates" if pname.startswith("raw_") else None)
        if part in args.freeze:
            p.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"meta-learned tensors: {[n for n, p in model.named_parameters() if p.requires_grad]}")
    # open-set softmax head: logits = [a*score_1..a*score_N, b]
    raw_a = nn.Parameter(torch.tensor(2.3, device=device))   # softplus -> ~2.4; scales scores
    b = nn.Parameter(torch.tensor(0.0, device=device))
    opt = torch.optim.Adam(trainable + [raw_a, b], lr=args.lr)

    name = f"meta_{args.model}" + (f"_{args.tag}" if args.tag else "")
    best, best_state, log = -1.0, None, []
    # Episode 0 = the starting point (untrained memory / plain cosine); it competes for selection.
    model.eval()
    v0 = run_protocol(blob, make_scorer(model, args.model, device))["val"]
    best, best_state = v0["dir@far0.05"], copy.deepcopy(model.state_dict())
    log.append({"episode": 0, "val_dir@far0.05": v0["dir@far0.05"], "val_auroc": v0["auroc"],
                "val_rank1": v0["rank1"]})
    print(f"episode=0 (start) val_dir@far0.05={v0['dir@far0.05']:.4f} val_auroc={v0['auroc']:.4f}")
    tic, n_skipped = time.time(), 0
    for ep in range(1, args.episodes + 1):
        model.train()
        Zg, P, tgt = sampler.episode(args.n_known, args.n_unknown, args.n_gallery, args.q)
        S = episode_scores(model, Zg, P, args.model)
        logits = torch.cat([F.softplus(raw_a) * S, b.expand(S.shape[0], 1)], 1)
        loss = F.cross_entropy(logits, tgt.to(device))
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            n_skipped += 1  # a diverged inner write; never let it update the meta-parameters
            continue
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(grad_norm):
            n_skipped += 1
            continue
        opt.step()
        if ep % args.eval_every == 0 or ep == args.episodes:
            model.eval()
            try:
                v = run_protocol(blob, make_scorer(model, args.model, device))["val"]
            except ValueError as e:  # this checkpoint's write rule diverges on real data
                if "NaN" not in str(e) and "infinity" not in str(e):
                    raise
                v = {"dir@far0.05": -1.0, "auroc": float("nan"), "rank1": float("nan")}
                print(f"episode={ep}: write diverged on val (non-finite scores); checkpoint rejected")
            rec = {"episode": ep, "skipped": n_skipped, "loss": round(loss.item(), 4), "val_dir@far0.05": v["dir@far0.05"],
                   "val_auroc": v["auroc"], "val_rank1": v["rank1"], "sec": round(time.time() - tic, 1)}
            if args.model == "memory":
                lr_, mom_, fgt_ = (x.item() for x in model.rates())
                rec.update({"mem_lr": round(lr_, 3), "momentum": round(mom_, 3), "forget": round(fgt_, 3)})
            log.append(rec)
            print(" ".join(f"{k}={v}" for k, v in rec.items()), flush=True)
            if v["dir@far0.05"] > best:
                best, best_state = v["dir@far0.05"], copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    scorer = make_scorer(model, args.model, device)
    results = {}
    groups = {"": blob}
    if args.by_source:
        groups.update({f"#{s}": b for s, b in source_blobs(blob, cfg["data"], cfg["seed"]).items()})
    for gname, gblob in groups.items():
        for vname, vblob in variant_blobs(gblob, cfg["data"], cfg["seed"]).items():
            results[name + gname + (f"@{vname}" if vname else "")] = run_protocol(vblob, scorer)
    out = run / name
    out.mkdir(exist_ok=True)
    torch.save(best_state, out / "model.pt")
    json.dump({"args": vars(args), "log": log, "results": results}, open(out / "metrics.json", "w"), indent=1)
    m = json.load(open(run / "metrics.json"))
    m["results"].update(results)
    json.dump(m, open(run / "metrics.json", "w"), indent=1)
    t = results[name]["test"]
    print(f"BEST val DIR@5% {best:.4f} | TEST rank1 {t['rank1']:.4f} auroc {t['auroc']:.4f} "
          f"dir@5% {t['dir@far0.05']:.4f} | g5 dir@5% {results.get(name + '@g5', {}).get('test', {}).get('dir@far0.05', float('nan')):.4f}")


if __name__ == "__main__":
    main()
