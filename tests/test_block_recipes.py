"""Block SFT with content normalization and complementary masking.

Known answers: a response that fits in one block gets the same content mask
as full-canvas SFT with a canvas of one block; the complement partitions the
target block; forced-mask positions stay masked in both views and never
enter the loss.
"""

from __future__ import annotations

import random

import pytest
import torch

from dllm import BlockSFTCollator, SFTCollator, complementary_view, diffusion_loss

MASK, EOS, PAD = 99, 98, 0
BL = 4


def _col(canvas, seed=0):
    return BlockSFTCollator(
        MASK, EOS, PAD, block_length=BL, canvas=canvas,
        generator=torch.Generator().manual_seed(seed), rng=random.Random(seed),
    )


def test_single_block_matches_full_canvas_content():
    samples = [{"prompt_ids": [1, 2, 3], "response_ids": [4, 5]}]
    b = _col("truncated")(samples)
    f = SFTCollator(MASK, EOS, response_canvas=BL,
                    generator=torch.Generator().manual_seed(0))(samples)
    assert torch.equal(b["clean_ids"], f["clean_ids"])
    assert torch.equal(b["maskable"], f["maskable"])
    assert torch.equal(b["content"], f["content"])
    assert b["content_lengths"].tolist() == [3]


def test_content_in_each_block():
    # response 4,5,6,7,8 + EOS = 6 tokens -> blocks [4,5,6,7] and [8,EOS,EOS,EOS]
    samples = [{"prompt_ids": [1], "response_ids": [4, 5, 6, 7, 8]}]
    seen = {}
    for seed in range(20):
        b = _col("full", seed)(samples)
        k = int(b["maskable"][0].nonzero()[0]) // BL  # prompt len 1 -> offset
        seen[k] = b
    first, last = seen[0], seen[1]
    assert first["content_lengths"].tolist() == [4]
    assert last["content_lengths"].tolist() == [2]
    assert last["content"][0, 5:7].all() and not last["content"][0, 7:].any()
    # full canvas: block 0 as target forces block 1 masked, outside the loss
    assert first["force_mask"][0, 5:].all() and not first["maskable"][0, 5:].any()
    assert (first["input_ids"][0, 5:] == MASK).all()


def test_complement_with_forced_mask():
    samples = [{"prompt_ids": [1, 2], "response_ids": list(range(10, 20))}]
    for seed in range(20):
        b = _col("full", seed)(samples)
        if not b["force_mask"].any():
            continue
        from dllm.masking import MaskingOutput

        m = MaskingOutput(b["input_ids"], b["masked_indices"], b["p_mask"], b["t"])
        c = complementary_view(b["clean_ids"], m, MASK, maskable=b["maskable"],
                               always_masked=b["force_mask"])
        assert not bool((m.masked_indices & c.masked_indices).any())
        assert torch.equal(m.masked_indices | c.masked_indices, b["maskable"])
        assert (c.noisy_ids[b["force_mask"]] == MASK).all()
        assert not c.masked_indices[b["force_mask"]].any()
        # content norm on a block: denominator is the block's content count
        logits = torch.randn(*b["clean_ids"].shape, 100)
        loss = diffusion_loss(logits, b["clean_ids"], c.masked_indices, c.p_mask,
                              norm="content", maskable=b["maskable"], content=b["content"])
        assert torch.isfinite(loss)
        return
    raise AssertionError("no sample with a later block was drawn")


def test_always_masked_must_not_overlap_maskable():
    ids = torch.randint(0, 50, (1, 6))
    maskable = torch.ones(1, 6, dtype=torch.bool)
    from dllm import forward_process

    m = forward_process(ids, MASK, maskable=maskable)
    with pytest.raises(ValueError):
        complementary_view(ids, m, MASK, maskable=maskable, always_masked=maskable)
