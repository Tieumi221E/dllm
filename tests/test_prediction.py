"""Prediction-field alignment: shifted (Dream-style) denoisers.

The known answer: a perfect shifted predictor, whose row i names the clean
token at i + 1, must reproduce its target exactly through the adapter and
full-canvas sampling. Declared as same-position it must not.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from dllm import (
    CanvasConfig,
    DenoiserInput,
    align_prediction_field,
    diffusion_loss,
    generate_canvas,
)
from dllm.adapters import AdapterCapabilityError, TransformersDenoiserAdapter

VOCAB, MASK = 32, 31
TARGET = torch.tensor([[3, 7, 1, 9, 4, 12, 5, 8, 2, 6]])


class ShiftedOracle(torch.nn.Module):
    """Row i puts its mass on TARGET[i + 1] (the last row on TARGET[0])."""

    def forward(self, input_ids=None, attention_mask=None, return_dict=True):
        batch, length = input_ids.shape
        nxt = torch.roll(TARGET[0, :length], shifts=-1)
        logits = torch.zeros(batch, length, VOCAB)
        logits[:, torch.arange(length), nxt] = 8.0
        return SimpleNamespace(logits=logits, past_key_values=None, hidden_states=None)


def _adapter(field):
    return TransformersDenoiserAdapter(
        ShiftedOracle(), prediction_field=field, default_topology="bidirectional"
    )


def test_align_known_values():
    logits = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    out = align_prediction_field(logits, "shifted")
    assert torch.equal(out[:, 1:], logits[:, :-1])
    assert torch.equal(out[:, 0], logits[:, 0])
    assert align_prediction_field(logits, "same_position") is logits
    for bad in ("next_token", "left"):
        try:
            align_prediction_field(logits, bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} must be rejected")


def test_shifted_oracle_reproduces_target_through_canvas():
    prompt = TARGET[:, :3]
    cfg = CanvasConfig(gen_length=7, block_length=7, steps=7)
    out = generate_canvas(_adapter("shifted"), prompt, MASK, cfg)
    assert torch.equal(out.canvas, TARGET)


def test_misdeclared_shifted_model_fails():
    # the negative control: same checkpoint, wrong declaration
    prompt = TARGET[:, :3]
    cfg = CanvasConfig(gen_length=7, block_length=7, steps=7)
    out = generate_canvas(_adapter("same_position"), prompt, MASK, cfg)
    assert not torch.equal(out.canvas, TARGET)
    assert torch.equal(out.canvas[0, 3:], torch.roll(TARGET[0], -1)[3:])


def test_shifted_capabilities_and_raw_execute():
    adapter = _adapter("shifted")
    assert adapter.capabilities.prediction_fields == frozenset({"same_position"})
    assert adapter.capabilities.native_prediction_field == "shifted"
    request = DenoiserInput(input_ids=TARGET.clone())
    raw = adapter.execute(request).logits
    aligned = adapter.denoise(DenoiserInput(input_ids=TARGET.clone())).logits
    assert torch.equal(aligned, align_prediction_field(raw, "shifted"))
    # the aligned field predicts every position after the first exactly
    assert torch.equal(aligned.argmax(-1)[0, 1:], TARGET[0, 1:])


def test_next_token_adapter_still_rejected():
    adapter = _adapter("next_token")
    assert adapter.capabilities.native_prediction_field == "next_token"
    try:
        adapter.denoise(DenoiserInput(input_ids=TARGET.clone()))
    except AdapterCapabilityError:
        return
    raise AssertionError("next-token logits must not be denoised")


def test_shifted_loss_matches_dream_training_formula():
    # Dream's official SFT: shift_logits = cat([logits[:, 0:1], logits[:, :-1]])
    torch.manual_seed(0)
    raw = torch.randn(2, 6, VOCAB)
    ids = torch.randint(0, 30, (2, 6))
    masked = torch.rand(2, 6) < 0.5
    masked[:, 0] = False
    p = torch.full((2, 6), 0.5)
    dream = torch.cat([raw[:, 0:1], raw[:, :-1]], dim=1)
    a = diffusion_loss(align_prediction_field(raw, "shifted"), ids, masked, p, norm="sum")
    b = diffusion_loss(dream, ids, masked, p, norm="sum")
    assert torch.equal(a, b)
