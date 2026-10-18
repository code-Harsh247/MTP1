import torch
import torch.nn.functional as F

from nlear.eval.openset import score_subspace
from nlear.memory import AssocMemory, group_gallery, make_memory_scorer


def _gallery(n_ids=4, g=2, d=32, seed=0):
    gen = torch.Generator().manual_seed(seed)
    G = torch.randn(n_ids * g, d, generator=gen)
    gl = [f"id{i}" for i in range(n_ids) for _ in range(g)]
    return G, gl


def test_linear_memory_converges_to_subspace_residual():
    # M_0 = 0, k = v = z, plain GD: W -> projection onto the gallery span, so
    # surprise = ||z - Pz||^2 = 1 - subspace_score^2 for unit probes (notes 8.3).
    G, gl = _gallery()
    P = torch.randn(20, 32, generator=torch.Generator().manual_seed(1))
    mem = AssocMemory(32, arch="linear", steps=400, lr=0.5, momentum=0.0, forget=0.0)
    S_mem, classes = make_memory_scorer(mem)(G, gl, P)
    S_sub, classes2 = score_subspace(G, gl, P)
    assert classes == classes2
    assert torch.allclose(-S_mem, 1 - S_sub.pow(2), atol=1e-3)


def test_batched_write_equals_per_identity_write():
    G, gl = _gallery(n_ids=3, g=2)
    mem = AssocMemory(32, arch="mlp", hidden=16, steps=3, lr=0.5, momentum=0.9, forget=0.05)
    _, Zg = group_gallery(G, gl)
    batched = mem.write(Zg)
    for i in range(3):
        single = mem.write(Zg[i:i + 1])
        for b, s in zip(batched, single):
            assert torch.allclose(b[i], s[0], atol=1e-5)


def test_momentum_and_forget_change_the_written_memory():
    G, gl = _gallery()
    _, Zg = group_gallery(G, gl)
    base = AssocMemory(32, arch="mlp", hidden=16, steps=4, lr=0.5, momentum=0.0, forget=0.0).write(Zg)
    mom = AssocMemory(32, arch="mlp", hidden=16, steps=4, lr=0.5, momentum=0.9, forget=0.0).write(Zg)
    fgt = AssocMemory(32, arch="mlp", hidden=16, steps=4, lr=0.5, momentum=0.0, forget=0.5).write(Zg)
    assert not torch.allclose(base[1], mom[1]) and not torch.allclose(base[1], fgt[1])


def test_meta_gradients_reach_projections_init_and_rates():
    G, gl = _gallery()
    _, Zg = group_gallery(G, gl)
    mem = AssocMemory(32, arch="mlp", hidden=16, d_key=16, d_val=16, steps=2, lr=0.5,
                      momentum=0.5, forget=0.1, learn_proj=True, learn_init=True, learn_rates=True)
    params = mem.write(Zg, create_graph=True)
    s = mem.surprise(params, torch.randn(5, 32))
    F.cross_entropy(-s, torch.tensor([0, 1, 2, 3, 0])).backward()
    for name, p in mem.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_meta_gradient_with_frozen_init_reaches_rates_only():
    G, gl = _gallery()
    _, Zg = group_gallery(G, gl)
    mem = AssocMemory(32, arch="mlp", hidden=16, steps=2, lr=0.5, momentum=0.5, forget=0.1,
                      learn_init=True, learn_rates=True)
    for name, p in mem.named_parameters():
        if name.startswith("init"):
            p.requires_grad_(False)
    s = mem.surprise(mem.write(Zg, create_graph=True), torch.randn(5, 32))
    F.cross_entropy(-s, torch.tensor([0, 1, 2, 3, 0])).backward()
    assert mem.raw_lr.grad is not None and mem.raw_forget.grad is not None
    assert mem.init0.grad is None


def test_initial_state_follows_module_device_and_dtype():
    mem = AssocMemory(32, arch="mlp", hidden=16).double()
    assert all(w.dtype == torch.float64 for w in mem.init)
    if torch.cuda.is_available():
        mem = mem.float().cuda()
        G, gl = _gallery()
        S, _ = make_memory_scorer(mem, device="cuda")(G, gl, torch.randn(3, 32))
        assert S.shape == (3, 4)


def test_unequal_gallery_sizes_are_rejected():
    G, gl = _gallery(n_ids=2, g=2)
    try:
        group_gallery(G[:3], gl[:3])
    except ValueError:
        return
    raise AssertionError("expected ValueError")
