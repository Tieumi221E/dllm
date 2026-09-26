"""Forward-process switches: clipped masking rates (and later ones).

Known answers: the clipped schedule with ``low=eps, high=1`` is the linear
schedule; the realized rate is uniform on ``[low, high]``; and under the same
seed ``forward_process`` reproduces Dream's reference ``q_sample(min, max)``
bit for bit (opt-in, see ``_dream_reference.py``).
"""

from __future__ import annotations

import pytest
import torch

from dllm import ClippedLinearSchedule, LinearSchedule, forward_process, get_schedule

from _dream_reference import load_module

MASK = 99


def test_clipped_reduces_to_linear():
    t = torch.rand(1000)
    for eps in (0.0, 1e-3, 0.2):
        a = ClippedLinearSchedule(eps, 1.0)
        b = LinearSchedule(eps)
        assert torch.equal(a.mask_prob(t), b.mask_prob(t))
        assert torch.equal(a.weight(t[t > 0]), b.weight(t[t > 0]))


def test_clipped_rate_is_uniform_on_interval():
    sched = ClippedLinearSchedule(0.3, 0.8)
    p = sched.mask_prob(torch.rand(200_000, generator=torch.Generator().manual_seed(0)))
    assert float(p.min()) >= 0.3 and float(p.max()) <= 0.8
    assert abs(float(p.mean()) - 0.55) < 2e-3
    assert abs(float(p.var()) - 0.5**2 / 12) < 1e-3  # U[a,b] variance
    assert sched.weight(0.5) == pytest.approx(1 / 0.55)


def test_clipped_validation_and_registry():
    for low, high in ((0.5, 0.5), (-0.1, 0.5), (0.2, 1.1), (0.8, 0.2)):
        with pytest.raises(ValueError):
            ClippedLinearSchedule(low, high)
    assert isinstance(get_schedule("clippedlinear"), ClippedLinearSchedule)


def test_clipped_forward_process_rate():
    ids = torch.zeros(4000, 64, dtype=torch.long)
    m = forward_process(
        ids, MASK, schedule=ClippedLinearSchedule(0.4, 0.6),
        generator=torch.Generator().manual_seed(1),
    )
    rate = m.masked_indices.float().mean(dim=1)
    assert float(m.p_mask.min()) >= 0.4 and float(m.p_mask.max()) <= 0.6
    assert abs(float(rate.mean()) - 0.5) < 5e-3


@pytest.mark.parametrize("low,high", [(0.0, 1.0), (0.2, 0.9), (0.45, 0.55)])
def test_clipped_matches_dream_q_sample(low, high):
    ref = load_module("gen_utils.py")
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, 50, (6, 20), generator=g)
    maskable = torch.arange(20).unsqueeze(0) >= torch.tensor([3, 5, 0, 8, 2, 11]).unsqueeze(1)
    for seed in range(5):
        torch.manual_seed(seed)
        x_t, t, t_mask = ref.q_sample(ids, maskable, MASK, min=low, max=high)
        ours = forward_process(
            ids, MASK, maskable=maskable, schedule=ClippedLinearSchedule(low, high),
            generator=torch.Generator().manual_seed(seed),
        )
        assert torch.equal(ours.p_mask[:, 0], t)
        assert torch.equal(ours.masked_indices, t_mask)
        assert torch.equal(ours.noisy_ids, x_t)
