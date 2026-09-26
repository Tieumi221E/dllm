"""CART token weights.

Known answers: hand-computed kernel values for a single clean neighbour,
and bit-for-bit agreement with Dream's reference ``context_adaptive_reweight``
plus its weight expression (opt-in, see ``_dream_reference.py``).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from dllm import cart_weights, masked_cross_entropy

from _dream_reference import load_function


def test_single_clean_neighbour_known_values():
    masked = torch.tensor([[True, False, True, True, True]])
    w = cart_weights(masked, cart_p=0.1)
    expected = torch.tensor([[0.05, 0.0, 0.05, 0.045, 0.0405]])
    assert torch.allclose(w, expected, atol=1e-7)


def test_weights_add_over_context_and_respect_context_mask():
    masked = torch.tensor([[False, True, False]])
    w = cart_weights(masked, cart_p=0.5)
    assert torch.allclose(w, torch.tensor([[0.0, 0.5, 0.0]]))  # 0.25 + 0.25
    ctx = torch.tensor([[True, True, False]])
    w = cart_weights(masked, cart_p=0.5, context=ctx)
    assert torch.allclose(w, torch.tensor([[0.0, 0.25, 0.0]]))


def test_validation():
    for p in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            cart_weights(torch.zeros(1, 3, dtype=torch.bool), cart_p=p)


@pytest.mark.parametrize("cart_p,length", [(0.1, 17), (0.8, 9), (0.3, 64)])
def test_matches_dream_reference(cart_p, length):
    reference = load_function("fsdp_sft_trainer.py", "context_adaptive_reweight")
    g = torch.Generator().manual_seed(length)
    masked = torch.rand(4, length, generator=g) < 0.6
    # Dream: weight = non_mask.type_as(W).matmul(W).masked_fill(non_mask, 0)
    matrix = reference(length, cart_p=cart_p)
    non_mask = ~masked
    ref = non_mask.type_as(matrix).matmul(matrix).masked_fill(non_mask, 0)
    assert torch.equal(cart_weights(masked, cart_p=cart_p), ref)


def test_cart_loss_matches_dream_normalization():
    # Dream: sum(ce * w over masked) / number of masked tokens
    torch.manual_seed(0)
    logits = torch.randn(2, 8, 11)
    target = torch.randint(0, 11, (2, 8))
    masked = torch.rand(2, 8) < 0.5
    masked[:, 0] = True
    w = cart_weights(masked)
    ce = F.cross_entropy(logits.view(-1, 11), target.view(-1), reduction="none").view(2, 8)
    ref = (ce * w * masked).sum() / masked.sum()
    got = masked_cross_entropy(logits, target, masked, token_weight=w, reduction="token_mean")
    assert torch.allclose(got, ref, atol=1e-6)
