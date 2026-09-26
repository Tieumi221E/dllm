"""Uniform-state corruption: selection rate, exclusions, untouched positions,
and uniformity of replacements."""

from __future__ import annotations

import pytest
import torch

from dllm import ClippedLinearSchedule, uniform_forward_process

V = 12


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def test_selection_rate_and_untouched_positions():
    ids = torch.randint(0, V, (3000, 32), generator=_gen(0))
    maskable = torch.ones_like(ids, dtype=torch.bool)
    maskable[:, :8] = False
    m = uniform_forward_process(ids, V, maskable=maskable, generator=_gen(1),
                                schedule=ClippedLinearSchedule(0.2, 0.6))
    assert not m.masked_indices[:, :8].any()
    assert torch.equal(m.noisy_ids[~m.masked_indices], ids[~m.masked_indices])
    rate = m.masked_indices[:, 8:].float().mean()
    assert abs(float(rate) - 0.4) < 5e-3
    assert m.p_mask.shape == ids.shape


def test_exclusions_never_sampled_and_uniform_replacements():
    ids = torch.zeros(4000, 16, dtype=torch.long)
    # excluded ids sit inside the vocabulary, so an index-vs-token mix-up shows
    m = uniform_forward_process(ids, V, t=torch.ones(4000), generator=_gen(2),
                                exclude_token_ids=(3, 7))
    replaced = m.noisy_ids[m.masked_indices]
    assert not bool(((replaced == 3) | (replaced == 7)).any())
    allowed = [i for i in range(V) if i not in (3, 7)]
    counts = torch.bincount(replaced, minlength=V)[allowed].float()
    expected = replaced.numel() / len(allowed)
    assert float(((counts - expected).abs() / expected).max()) < 0.05
    # a replacement may equal the original token (multinomial diffusion)
    assert bool((replaced == 0).any())


def test_validation():
    ids = torch.zeros(2, 4, dtype=torch.long)
    with pytest.raises(ValueError):
        uniform_forward_process(ids, 3, exclude_token_ids=(0, 1, 2))
    with pytest.raises(ValueError):
        uniform_forward_process(ids, V, t=torch.ones(3))
