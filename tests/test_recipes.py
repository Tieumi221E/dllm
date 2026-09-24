"""SFT recipe switches: content normalization, fixed response canvas,
complementary masking, and the autoregressive auxiliary term.

Every test checks a quantity that is known in advance, including the one
the "content" norm exists for: under "answer" normalization a fixed canvas
scales the response's loss by n_content / L, under "content" it does not.
"""

import torch
import torch.nn.functional as F

from dllm import (
    SFTCollator,
    complementary_view,
    diffusion_loss,
    forward_process,
    next_token_loss,
)

V, MASK, EOS = 40, 39, 38


def _samples():
    return [
        {"prompt_ids": [1, 2, 3], "response_ids": [4, 5]},
        {"prompt_ids": [1, 2], "response_ids": [6, 7, 8, 9]},
    ]


def test_collator_content_mask_without_canvas():
    b = SFTCollator(MASK, EOS, generator=torch.Generator().manual_seed(0))(_samples())
    # response + appended EOS; the batch pads with EOS beyond that
    assert b["content_lengths"].tolist() == [3, 5]
    assert torch.equal(b["content"] & ~b["maskable"], torch.zeros_like(b["content"]))
    # default behaviour unchanged: everything attended, pads maskable
    assert bool(b["attention_mask"].all())
    assert b["answer_lengths"].tolist() == [4, 5]


def test_collator_fixed_canvas():
    col = SFTCollator(MASK, EOS, response_canvas=8, generator=torch.Generator().manual_seed(0))
    b = col(_samples())
    # every sample's maskable window is exactly the canvas
    assert b["answer_lengths"].tolist() == [8, 8]
    assert b["content_lengths"].tolist() == [3, 5]
    # sample 0 has a longer prompt, so sample 1 ends one position earlier:
    # that trailing position is batch padding, neither attended nor maskable
    width = b["clean_ids"].shape[1]
    assert width == 3 + 8
    assert b["attention_mask"][1, -1].item() == 0
    assert not bool(b["maskable"][1, -1])
    assert bool(b["attention_mask"][0].all())
    # canvas contents: response, EOS, then EOS padding
    assert b["clean_ids"][0, 3:].tolist() == [4, 5, EOS] + [EOS] * 5


def test_collator_canvas_too_short_raises():
    col = SFTCollator(MASK, EOS, response_canvas=3)
    try:
        col(_samples())
    except ValueError:
        pass
    else:
        raise AssertionError("a response longer than the canvas must raise")


def test_content_norm_equals_answer_without_padding():
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 30, (3, 12))
    prompt = torch.tensor([2, 4, 3])
    maskable = torch.arange(12).unsqueeze(0) >= prompt.unsqueeze(1)
    m = forward_process(ids, MASK, maskable=maskable, generator=g)
    logits = torch.randn(3, 12, V)
    a = diffusion_loss(logits, ids, m.masked_indices, m.p_mask, norm="answer", maskable=maskable)
    c = diffusion_loss(
        logits, ids, m.masked_indices, m.p_mask, norm="content", maskable=maskable, content=maskable
    )
    assert torch.allclose(a, c, atol=1e-6)


def test_content_norm_manual_value():
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(0, 30, (2, 10))
    maskable = torch.zeros(2, 10, dtype=torch.bool)
    maskable[:, 3:] = True
    content = torch.zeros(2, 10, dtype=torch.bool)
    content[0, 3:5] = True
    content[1, 3:8] = True
    m = forward_process(ids, MASK, maskable=maskable, generator=g)
    logits = torch.randn(2, 10, V)
    got = diffusion_loss(
        logits, ids, m.masked_indices, m.p_mask, norm="content", maskable=maskable, content=content
    )
    ce = F.cross_entropy(logits.reshape(-1, V), ids.reshape(-1), reduction="none").view(2, 10)
    num = (ce * m.masked_indices / m.p_mask).sum(dim=1)
    ref = (num / content.sum(dim=1)).mean()
    assert torch.allclose(got, ref, atol=1e-6), (got, ref)


def test_content_norm_validation():
    ids = torch.randint(0, 30, (1, 6))
    logits = torch.randn(1, 6, V)
    mi = torch.ones(1, 6, dtype=torch.bool)
    pm = torch.full((1, 6), 0.5)
    for kwargs in (
        {},  # content missing
        {"content": torch.ones(1, 6, dtype=torch.bool), "maskable": torch.zeros(1, 6, dtype=torch.bool)},
    ):
        try:
            diffusion_loss(logits, ids, mi, pm, norm="content", **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


def test_canvas_dilution_known_case():
    """The mechanism the content norm fixes. Padding is predicted perfectly
    (zero CE), the response is not; every position is masked (p = 1). Then
    "answer" shrinks as n / L with the canvas, "content" stays constant."""
    n = 4
    results = {}
    for L in (8, 16, 32):
        ids = torch.full((1, L), EOS, dtype=torch.long)
        ids[0, :n] = torch.arange(n)
        logits = torch.zeros(1, L, V)
        logits[0, n:, EOS] = 50.0  # padding: CE ~ 0
        maskable = torch.ones(1, L, dtype=torch.bool)
        content = torch.zeros(1, L, dtype=torch.bool)
        content[0, :n] = True
        mi = torch.ones(1, L, dtype=torch.bool)
        pm = torch.ones(1, L)
        a = diffusion_loss(logits, ids, mi, pm, norm="answer", maskable=maskable)
        c = diffusion_loss(logits, ids, mi, pm, norm="content", maskable=maskable, content=content)
        results[L] = (a.item(), c.item())
    per_token = torch.log(torch.tensor(float(V))).item()  # uniform logits on the response
    for L, (a, c) in results.items():
        assert abs(c - per_token) < 1e-4, (L, c)
        assert abs(a - per_token * n / L) < 1e-4, (L, a)


def test_complementary_view_partitions_maskable():
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(0, 30, (4, 16))
    maskable = torch.arange(16).unsqueeze(0) >= torch.tensor([2, 5, 0, 9]).unsqueeze(1)
    m = forward_process(ids, MASK, maskable=maskable, generator=g)
    c = complementary_view(ids, m, MASK, maskable=maskable)
    assert not bool((m.masked_indices & c.masked_indices).any())
    assert torch.equal(m.masked_indices | c.masked_indices, maskable)
    assert torch.equal(c.noisy_ids, torch.where(c.masked_indices, MASK, ids))
    assert torch.allclose(c.p_mask, (1 - m.p_mask).clamp(min=1e-3))


def test_complementary_view_is_unbiased():
    """Each view's importance-weighted sum estimates the full CE sum."""
    torch.manual_seed(4)
    ids = torch.randint(0, 30, (1, 10))
    logits = torch.randn(1, 10, V)
    ce_total = F.cross_entropy(logits.reshape(-1, V), ids.reshape(-1), reduction="sum").item()
    t = torch.tensor([0.3])
    g = torch.Generator().manual_seed(5)
    est_a, est_b, trials = 0.0, 0.0, 4000
    for _ in range(trials):
        m = forward_process(ids, MASK, t=t, generator=g)
        c = complementary_view(ids, m, MASK)
        est_a += diffusion_loss(logits, ids, m.masked_indices, m.p_mask, norm="sum").item()
        est_b += diffusion_loss(logits, ids, c.masked_indices, c.p_mask, norm="sum").item()
    assert abs(est_a / trials - ce_total) / ce_total < 0.05
    assert abs(est_b / trials - ce_total) / ce_total < 0.05


def test_next_token_loss_matches_manual_shift():
    torch.manual_seed(6)
    ids = torch.randint(0, 30, (2, 7))
    logits = torch.randn(2, 7, V)
    sel = torch.zeros(2, 7, dtype=torch.bool)
    sel[0, 3:] = True
    sel[1, 1:5] = True
    got = next_token_loss(logits, ids, sel)
    ce = F.cross_entropy(
        logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1), reduction="none"
    ).view(2, 6)
    s = sel[:, 1:]
    ref = (ce * s).sum() / s.sum()
    assert torch.allclose(got, ref, atol=1e-6)
    # position 0 has no predecessor and is ignored
    sel0 = torch.zeros(2, 7, dtype=torch.bool)
    sel0[:, 0] = True
    assert next_token_loss(logits, ids, sel0).item() == 0.0
