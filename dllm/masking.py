"""Forward (noising) process for masked diffusion.

``t ~ U(0,1)`` per sample, ``p = schedule.mask_prob(t)``, each maskable token
masked independently with probability ``p``. Returns a per-token ``p_mask``
matrix so losses can weight each token by its own masking probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

from .schedules import NoiseSchedule, get_schedule


@dataclass
class MaskingOutput:
    noisy_ids: torch.Tensor  # (B, L) input with [MASK] substituted
    masked_indices: torch.Tensor  # (B, L) bool - True where masked
    p_mask: torch.Tensor  # (B, L) float - per-token mask probability
    t: torch.Tensor  # (B,) float - sampled timesteps


def forward_process(
    input_ids: torch.Tensor,
    mask_token_id: int,
    maskable: Optional[torch.Tensor] = None,
    t: Optional[torch.Tensor] = None,
    schedule: Union[str, NoiseSchedule, None] = None,
    generator: Optional[torch.Generator] = None,
    min_one_mask: bool = False,
) -> MaskingOutput:
    """Apply the forward masking process.

    Args:
        input_ids: (B, L) clean token ids.
        maskable: (B, L) bool/int - positions eligible for masking. ``None``
            means every position. For SFT pass ``positions >= prompt_len``
            (prompt stays clean).
        t: optional (B,) timesteps in [0, 1]; sampled uniformly if None.
        schedule: noise schedule (default: linear, eps=1e-3).
        min_one_mask: force >=1 masked token per sample. Off by default -
            forcing a mask slightly biases the estimator; enable only for
            tiny-batch regimes where zero-mask samples are too wasteful.
    """
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be (B, L), got {tuple(input_ids.shape)}")
    device = input_ids.device
    bsz, seq_len = input_ids.shape
    sched = get_schedule(schedule)

    if maskable is None:
        maskable_b = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        maskable_b = maskable.to(device=device, dtype=torch.bool)

    if t is None:
        t = torch.rand(bsz, device=device, generator=generator)
    else:
        t = t.to(device=device, dtype=torch.float32)
        if t.shape != (bsz,):
            raise ValueError(f"t must be shape ({bsz},), got {tuple(t.shape)}")

    p_mask = sched.mask_prob(t).unsqueeze(1).expand(bsz, seq_len).contiguous()

    rand = torch.rand(bsz, seq_len, device=device, generator=generator)
    masked_indices = (rand < p_mask) & maskable_b

    if min_one_mask:
        needs = (~masked_indices.any(dim=1)) & maskable_b.any(dim=1)
        if needs.any():
            # pick one random maskable position for each sample that needs it
            scores = torch.rand(bsz, seq_len, device=device, generator=generator)
            scores = scores.masked_fill(~maskable_b, -1.0)
            pick = scores.argmax(dim=1)
            rows = torch.nonzero(needs, as_tuple=True)[0]
            masked_indices[rows, pick[rows]] = True

    noisy_ids = torch.where(masked_indices, mask_token_id, input_ids)
    return MaskingOutput(
        noisy_ids=noisy_ids, masked_indices=masked_indices, p_mask=p_mask, t=t
    )


def make_labels(
    input_ids: torch.Tensor,
    masked_indices: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Build a labels tensor with ``ignore_index`` outside masked positions."""
    labels = input_ids.clone()
    labels[~masked_indices] = ignore_index
    return labels


def random_truncate(
    input_ids: torch.Tensor,
    prob: float = 0.01,
    min_length: int = 1,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Random-length pretraining trick: with probability ``prob`` truncate the
    batch to a random length. Improves variable-length generation. Apply to
    the clean batch before :func:`forward_process`."""
    if prob <= 0.0:
        return input_ids
    r = torch.rand(1, generator=generator).item()
    if r < prob:
        L = input_ids.shape[1]
        length = int(torch.randint(min_length, L + 1, (1,), generator=generator).item())
        return input_ids[:, :length]
    return input_ids
