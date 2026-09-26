"""Inference EOS stop for full-canvas decoding.

Known answer: an oracle that is most confident about its EOS commits it
first; with ``stop_at_eos`` everything after it is filled in the same step
and only the positions before it cost forwards.
"""

from __future__ import annotations

import pytest
import torch

from dllm import CanvasConfig, generate_canvas

VOCAB, MASK, EOS = 20, 19, 18
PROMPT = torch.tensor([[1, 2]])
# generated region: 5 6 7 EOS 9 10 11 12 (an oracle keeps writing after EOS)
TARGET = torch.tensor([5, 6, 7, EOS, 9, 10, 11, 12])


def oracle(ids, attention_mask=None):
    logits = torch.zeros(ids.shape[0], ids.shape[1], VOCAB)
    gen = torch.arange(ids.shape[1]) - PROMPT.shape[1]
    for i in range(ids.shape[1]):
        if 0 <= gen[i] < len(TARGET):
            tok = int(TARGET[gen[i]])
            logits[:, i, tok] = 10.0 if tok == EOS else 5.0 - 0.1 * int(gen[i])
    return logits


def _run(**kw):
    cfg = CanvasConfig(gen_length=8, block_length=8, steps=8, eos_token_id=EOS, **kw)
    return generate_canvas(oracle, PROMPT, MASK, cfg)


def test_stop_fills_after_eos_and_saves_forwards():
    base = _run()
    stop = _run(stop_at_eos=True)
    assert base.nfe == 8
    # step 0 commits EOS and fills the 4 positions after it; 3 remain
    assert stop.nfe == 1 + 3
    assert stop.canvas[0, 2:].tolist() == [5, 6, 7, EOS, EOS, EOS, EOS, EOS]
    assert stop.step_map[0, 3:].tolist() == [0] * 5
    assert stop.responses == base.responses == [[5, 6, 7]]


def test_stop_with_semi_ar_blocks_skips_later_blocks():
    cfg = CanvasConfig(gen_length=8, block_length=4, steps=8, eos_token_id=EOS,
                       stop_at_eos=True)
    out = generate_canvas(oracle, PROMPT, MASK, cfg)
    # block 0 (positions 0..3) holds the EOS; block 1 is filled, never run
    assert out.canvas[0, 6:].tolist() == [EOS] * 4
    assert out.nfe == 4


def test_stop_validation():
    for kw in (dict(record_trace=True), dict(suppress_eos_logits=True)):
        with pytest.raises(ValueError):
            _run(stop_at_eos=True, **kw)
    with pytest.raises(ValueError):
        generate_canvas(oracle, PROMPT, MASK,
                        CanvasConfig(gen_length=8, steps=8, stop_at_eos=True))


def test_leftmost_committed_eos_decides():
    # two EOS proposals, both committed in the first threshold step
    target = torch.tensor([5, 6, 7, EOS, 9, 10, EOS, 12])

    def two_eos(ids, attention_mask=None):
        logits = torch.zeros(ids.shape[0], ids.shape[1], VOCAB)
        for i in range(PROMPT.shape[1], ids.shape[1]):
            tok = int(target[i - PROMPT.shape[1]])
            logits[:, i, tok] = 10.0 if tok == EOS else 1.0
        return logits

    cfg = CanvasConfig(gen_length=8, block_length=8, steps=8, eos_token_id=EOS,
                       stop_at_eos=True, commit="threshold", threshold=0.9)
    out = generate_canvas(two_eos, PROMPT, MASK, cfg)
    # positions 4 and 5 lie after the leftmost EOS: filled, not decoded
    assert out.canvas[0, 2:].tolist() == [5, 6, 7, EOS, EOS, EOS, EOS, EOS]
    assert out.step_map[0, 4:6].tolist() == [0, 0]
