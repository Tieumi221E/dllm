"""Stratified and antithetic timestep sampling.

Known answers: each stratum holds exactly one draw; antithetic draws mirror
each other; all modes estimate E[f(t)] = 1/3 for f(t) = t^2 without bias,
and the variance-reducing modes have a smaller batch-mean variance.
"""

from __future__ import annotations

import pytest
import torch

from dllm import forward_process, sample_times


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def test_stratified_one_draw_per_stratum():
    for batch in (1, 7, 64):
        t = sample_times(batch, "stratified", generator=_gen(batch))
        strata = torch.floor(t * batch).long()
        assert sorted(strata.tolist()) == list(range(batch))


def test_antithetic_pairs_mirror():
    t = sample_times(10, "antithetic", generator=_gen(0))
    assert torch.allclose(t[5:], 1 - t[:5])
    with pytest.raises(ValueError):
        sample_times(7, "antithetic")
    with pytest.raises(ValueError):
        sample_times(4, "sobol")


def test_unbiased_and_variance_reducing():
    batch, trials = 8, 4000
    g = _gen(1)
    means = {}
    for mode in ("uniform", "stratified", "antithetic"):
        vals = torch.stack([
            (sample_times(batch, mode, generator=g) ** 2).mean() for _ in range(trials)
        ])
        means[mode] = (float(vals.mean()), float(vals.var()))
    for mode, (mean, _) in means.items():
        assert abs(mean - 1 / 3) < 0.01, mode
    assert means["stratified"][1] < 0.2 * means["uniform"][1]
    assert means["antithetic"][1] < means["uniform"][1]


def test_forward_process_uses_the_mode():
    ids = torch.zeros(16, 4, dtype=torch.long)
    m = forward_process(ids, 9, generator=_gen(2), t_sampling="stratified")
    assert sorted(torch.floor(m.t * 16).long().tolist()) == list(range(16))
    # default stays i.i.d. uniform and draws the same stream as before
    a = forward_process(ids, 9, generator=_gen(3))
    b = forward_process(ids, 9, generator=_gen(3), t_sampling="uniform")
    assert torch.equal(a.t, b.t) and torch.equal(a.masked_indices, b.masked_indices)
