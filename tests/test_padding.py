"""SFT padding strategies: EOS runs, ignored mask padding, dedicated pad
tokens, and per-batch random cutoff.

Known answers are hand-computed batches; ``eos_as_one`` is also compared
with Dream's reference ``q_sample(eos_token_id=...)`` under the same seed
(opt-in, see ``_dream_reference.py``).
"""

from __future__ import annotations

import random

import pytest
import torch

from dllm import ClippedLinearSchedule, SFTCollator, diffusion_loss, forward_process
from dllm.masking import trailing_run

from _dream_reference import load_module

MASK, EOS, P1, P2, P3 = 99, 98, 97, 96, 95


def _samples():
    return [
        {"prompt_ids": [1, 2, 3], "response_ids": [4, 5]},
        {"prompt_ids": [1, 2], "response_ids": [6, 7, 8, 9]},
    ]


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


# ----------------------------- trailing runs ------------------------------ #

def test_trailing_run_known_rows():
    ids = torch.tensor([[5, EOS, 6, EOS, EOS], [EOS, EOS, EOS, 1, 2], [3, 4, EOS, EOS, EOS]])
    maskable = torch.ones_like(ids, dtype=torch.bool)
    maskable[2, 4] = False  # a non-maskable tail breaks the run
    run = trailing_run(ids, maskable, EOS)
    assert run.tolist() == [
        [False, False, False, True, True],
        [False, False, False, False, False],
        [False, False, False, False, False],
    ]


def test_tied_run_is_all_or_nothing():
    ids = torch.tensor([[4, 5, 6, EOS, EOS, EOS, EOS, EOS]]).repeat(500, 1)
    m = forward_process(ids, MASK, generator=_gen(1), tie_trailing_token_id=EOS)
    tail = m.masked_indices[:, 3:]
    assert bool((tail.all(dim=1) | ~tail.any(dim=1)).all())
    # the run keeps its marginal masking rate p
    rate = tail[:, 0].float().mean() - m.p_mask[:, 0].mean()
    assert abs(float(rate)) < 0.06
    assert torch.equal(m.noisy_ids, torch.where(m.masked_indices, MASK, ids))


def test_eos_as_one_matches_dream_q_sample():
    ref = load_module("gen_utils.py")
    rows = [[1, 2, 3, 4, EOS, EOS, EOS], [1, 5, 6, 7, 8, 9, EOS], [1, 2, 3, EOS, EOS, EOS, EOS]]
    ids = torch.tensor(rows)
    maskable = torch.arange(7).unsqueeze(0) >= torch.tensor([1, 2, 1]).unsqueeze(1)
    for seed in range(20):
        torch.manual_seed(seed)
        x_t, _, t_mask = ref.q_sample(ids, maskable, MASK, eos_token_id=EOS)
        # q_sample's default rate range is [0, 1], not dllm's eps-floored one
        ours = forward_process(
            ids, MASK, maskable=maskable, schedule=ClippedLinearSchedule(0.0, 1.0),
            generator=_gen(seed), tie_trailing_token_id=EOS,
        )
        assert torch.equal(ours.masked_indices, t_mask)
        assert torch.equal(ours.noisy_ids, x_t)


# ------------------------------ pad modes -------------------------------- #

def test_eos_as_one_through_collator():
    col = SFTCollator(MASK, EOS, response_canvas=6, eos_as_one=True, generator=_gen(2))
    for _ in range(50):
        b = col(_samples())
        run = trailing_run(b["clean_ids"], b["maskable"], EOS)
        for i in range(2):
            vals = b["masked_indices"][i][run[i]]
            assert bool(vals.all()) or not bool(vals.any())


def test_mask_ignored_padding():
    b = SFTCollator(MASK, EOS, response_canvas=6, pad_mode="mask_ignored", generator=_gen(3))(
        _samples()
    )
    # sample 0: prompt 3, content [4, 5, EOS], then 3 padding positions
    assert b["clean_ids"][0, 3:].tolist() == [4, 5, EOS, MASK, MASK, MASK]
    assert b["maskable"][0].tolist() == [False] * 3 + [True] * 3 + [False] * 3
    assert torch.equal(b["maskable"], b["content"])
    assert bool(b["attention_mask"][0].all())  # padding stays attended
    # with maskable == content the two normalizations coincide
    logits = torch.randn(*b["clean_ids"].shape, 100)
    a = diffusion_loss(logits, b["clean_ids"], b["masked_indices"], b["p_mask"],
                       norm="answer", maskable=b["maskable"])
    c = diffusion_loss(logits, b["clean_ids"], b["masked_indices"], b["p_mask"],
                       norm="content", maskable=b["maskable"], content=b["content"])
    assert torch.allclose(a, c)


def test_dedicated_padding_cycles():
    b = SFTCollator(MASK, EOS, response_canvas=8, pad_mode="dedicated",
                    pad_token_ids=(P1, P2, P3), generator=_gen(4))(_samples())
    assert b["clean_ids"][0, 3:].tolist() == [4, 5, EOS, P1, P2, P3, P1, P2]
    assert b["clean_ids"][1, 2:10].tolist() == [6, 7, 8, 9, EOS, P1, P2, P3]
    # padding is maskable (it is trained) but not content
    assert b["maskable"][0, 6:].all() and not b["content"][0, 6:].any()
    assert b["content_lengths"].tolist() == [3, 5]
    # exactly one semantic EOS per sample
    assert (b["clean_ids"] == EOS).sum(dim=1).tolist() == [1, 1 + 1]  # +1: batch pad
    assert b["attention_mask"][1, -1].item() == 0


def test_pad_mode_validation():
    bad = [
        dict(pad_mode="zero"),
        dict(pad_mode="dedicated"),
        dict(pad_mode="dedicated", pad_token_ids=(EOS,)),
        dict(pad_token_ids=(P1,)),
        dict(pad_mode="mask_ignored", eos_as_one=True),
        dict(batch_cutoff=True, response_canvas=8),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            SFTCollator(MASK, EOS, **kwargs)


# ------------------------------ batch cutoff ------------------------------ #

def test_batch_cutoff_known_draws():
    samples = [
        {"prompt_ids": [1], "response_ids": [4, 5]},  # content 3
        {"prompt_ids": [1, 2], "response_ids": [6, 7, 8, 9, 10, 11]},  # content 7
    ]
    seen = set()
    rng = random.Random(0)
    col = SFTCollator(MASK, EOS, batch_cutoff=True, rng=rng, generator=_gen(5))
    for _ in range(40):
        b = col(samples)
        kept = int(b["answer_lengths"][0])
        seen.add(kept)
        assert kept in (3, 7)
        assert b["answer_lengths"].tolist() == [kept, kept]
        assert b["content_lengths"].tolist() == [min(3, kept), min(7, kept)]
        if kept == 3:
            # the long response is cut and loses its EOS, as in Dream
            assert b["clean_ids"][1, 2:5].tolist() == [6, 7, 8]
            assert EOS not in b["clean_ids"][1, 2:5].tolist()
        else:
            assert b["clean_ids"][0, 1:8].tolist() == [4, 5, EOS] + [EOS] * 4
    assert seen == {3, 7}
