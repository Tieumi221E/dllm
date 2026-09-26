"""Prediction fields: which input position a logit row refers to.

A denoiser's output row ``i`` can mean one of three things:

- ``"same_position"``: row ``i`` predicts the clean token at position ``i``
  (LLaDA, SDAR, Nemotron-Labs-Diffusion, the reference transformer);
- ``"shifted"``: row ``i`` predicts the clean token at position ``i + 1``,
  i.e. a masked position is read from its predecessor's output. This is the
  AR-initialized convention of Dream, DiffuCoder, Dream-Coder, Fast-dLLM v2
  and DiffuLLaMA, which keep the next-token head of the source model;
- ``"next_token"``: an autoregressive model under causal attention. The row
  also predicts position ``i + 1``, but only from positions ``<= i``; it is
  not a denoiser and has no same-position view.

Samplers and losses in this package consume same-position logits.
:func:`align_prediction_field` is the single conversion point.
"""

from __future__ import annotations

import torch

PREDICTION_FIELDS = ("same_position", "shifted", "next_token")


def align_prediction_field(logits: torch.Tensor, field: str) -> torch.Tensor:
    """Return same-position logits for a ``(B, L, V)`` prediction field.

    ``"shifted"`` becomes ``cat([logits[:, :1], logits[:, :-1]], dim=1)``:
    row ``i`` of the result is row ``i - 1`` of the input. Position 0 has no
    predecessor and repeats row 0, the convention of Dream's official
    training and generation code; position 0 is a prompt token in every
    conditional use, so its row is never read.

    The alignment is defined on a sequence that starts at logical position
    0. A cached or windowed forward that starts at position ``s > 0`` must
    supply row ``s - 1`` itself; this function cannot recover it.
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape (B, L, V)")
    if field == "same_position":
        return logits
    if field == "shifted":
        return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    if field == "next_token":
        raise ValueError(
            "next-token logits come from causal attention and have no "
            "same-position view; use next_token_loss or causal verification"
        )
    raise ValueError(f"prediction field must be one of {PREDICTION_FIELDS}")


__all__ = ["PREDICTION_FIELDS", "align_prediction_field"]
