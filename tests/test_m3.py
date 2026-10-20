import pytest
import torch
import torch.nn as nn

from nlear.m3 import M3, newton_schulz, split_params


def test_newton_schulz_approximately_orthogonalizes():
    torch.manual_seed(0)
    for shape in ((16, 64), (64, 16)):
        X = newton_schulz(torch.randn(*shape))
        small = min(shape)
        gram = X @ X.T if shape[0] <= shape[1] else X.T @ X
        # Muon's quintic NS targets singular values in ~[0.7, 1.2], not exactly 1
        s = torch.linalg.svdvals(X)
        assert gram.shape == (small, small) and s.min() > 0.5 and s.max() < 1.3


def _quadratic_run(alpha, steps=200, freq=4, lr=0.1):
    torch.manual_seed(0)
    W = nn.Parameter(torch.randn(8, 6))
    target = torch.randn(8, 6)
    opt = M3([W], lr=lr, alpha=alpha, freq=freq)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = ((W - target) ** 2).sum()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses, opt.state[W]


def test_m3_minimizes_a_quadratic():
    losses, _ = _quadratic_run(alpha=1.0)
    assert losses[-1] < 0.05 * losses[0]


def test_slow_memory_updates_only_every_freq_steps_and_alpha0_disables_it():
    torch.manual_seed(0)
    W = nn.Parameter(torch.randn(4, 3))
    opt = M3([W], lr=1e-3, alpha=0.5, freq=3)
    seen = []
    for i in range(1, 7):
        opt.zero_grad()
        (W ** 2).sum().backward()
        before = opt.state[W]["M2"].clone() if opt.state[W] else torch.zeros(4, 3)
        opt.step()
        seen.append(not torch.equal(before, opt.state[W]["M2"]))
    assert seen == [False, False, True, False, False, True]
    _, st = _quadratic_run(alpha=0.0, steps=10)
    assert torch.count_nonzero(st["M2"]) == 0


def test_conv_kernels_are_handled_as_matrices_and_vectors_rejected():
    conv = nn.Conv2d(3, 5, 3, bias=True)
    mats, rest = split_params(conv)
    assert [p.ndim for p in mats] == [4] and [p.ndim for p in rest] == [1]
    opt = M3(mats, lr=1e-3)
    conv(torch.randn(2, 3, 8, 8)).sum().backward()
    w0 = conv.weight.detach().clone()
    opt.step()
    assert conv.weight.shape == w0.shape and not torch.equal(conv.weight, w0)
    with pytest.raises(ValueError):
        M3([conv.bias], lr=1e-3).step()
