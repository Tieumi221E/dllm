"""Forward (noising) process for masked diffusion.

``t ~ U(0,1)`` per sample, ``p = schedule.mask_prob(t)``, each maskable token
masked independently with probability ``p``. Returns a per-token ``p_mask``
matrix so losses can weight each token by its own masking probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch

from .schedules import NoiseSchedule, get_schedule


@dataclass
class MaskingOutput:
    noisy_ids: torch.Tensor  # (B, L) input with [MASK] substituted
    masked_indices: torch.Tensor  # (B, L) bool - True where masked
    p_mask: torch.Tensor  # (B, L) float - per-token mask probability
    t: torch.Tensor  # (B,) float - sampled timesteps


_T_SAMPLING = ("uniform", "stratified", "antithetic")


def sample_times(
    batch_size: int,
    mode: str = "uniform",
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Per-sample timesteps ``t`` in [0, 1) for one batch.

    - ``"uniform"``: i.i.d. ``U(0, 1)``;
    - ``"stratified"``: one draw in each of the ``B`` equal strata
      ``[b/B, (b+1)/B)``, in random order (the low-discrepancy sampler of
      VDM / MDLM);
    - ``"antithetic"``: ``B/2`` draws ``u`` and their mirrors ``1 - u``
      (``B`` must be even).

    Every mode keeps each ``t`` marginally ``U(0, 1)``, so the batch-mean
    loss stays unbiased; stratified and antithetic draws only reduce its
    variance across batches.
    """
    if mode not in _T_SAMPLING:
        raise ValueError(f"t sampling must be one of {_T_SAMPLING}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if mode == "uniform":
        return torch.rand(batch_size, device=device, generator=generator)
    if mode == "stratified":
        u = torch.rand(batch_size, device=device, generator=generator)
        t = (torch.arange(batch_size, device=device) + u) / batch_size
        order = torch.randperm(batch_size, generator=generator).to(t.device)
        return t[order]
    if batch_size % 2:
        raise ValueError("antithetic sampling needs an even batch size")
    u = torch.rand(batch_size // 2, device=device, generator=generator)
    return torch.cat([u, 1.0 - u])


def forward_process(
    input_ids: torch.Tensor,
    mask_token_id: int,
    maskable: Optional[torch.Tensor] = None,
    t: Optional[torch.Tensor] = None,
    schedule: Union[str, NoiseSchedule, None] = None,
    generator: Optional[torch.Generator] = None,
    min_one_mask: bool = False,
    tie_trailing_token_id: Optional[int] = None,
    t_sampling: str = "uniform",
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
        tie_trailing_token_id: treat each row's trailing run of this token
            (maskable positions only) as one unit: the whole run takes the
            masking draw of its first position (Dream's ``treat_eos_as_one``).
            Every run position is still masked with marginal probability
            ``p``, so ``p_mask`` and the ``1/p`` weight are unchanged. Dream
            locates the run by counting non-EOS tokens, which agrees with
            this suffix rule whenever the token appears only at the end.
        t_sampling: how ``t`` is drawn when not given - "uniform",
            "stratified" or "antithetic" (see :func:`sample_times`).
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
        t = sample_times(bsz, t_sampling, generator=generator, device=device)
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

    if tie_trailing_token_id is not None:
        masked_indices = tie_trailing_run(
            input_ids, masked_indices, maskable_b, tie_trailing_token_id
        )

    noisy_ids = torch.where(masked_indices, mask_token_id, input_ids)
    return MaskingOutput(
        noisy_ids=noisy_ids, masked_indices=masked_indices, p_mask=p_mask, t=t
    )


def uniform_forward_process(
    input_ids: torch.Tensor,
    vocab_size: int,
    maskable: Optional[torch.Tensor] = None,
    t: Optional[torch.Tensor] = None,
    schedule: Union[str, NoiseSchedule, None] = None,
    generator: Optional[torch.Generator] = None,
    exclude_token_ids: Sequence[int] = (),
) -> MaskingOutput:
    """Uniform-state (multinomial) corruption instead of absorbing masking.

    Each maskable token is selected with probability ``p = schedule(t)`` and
    replaced by a token drawn uniformly from the vocabulary minus
    ``exclude_token_ids`` (e.g. mask, pad and other special ids). A
    replacement may equal the original token, as in multinomial diffusion.
    ``masked_indices`` marks the selected positions. There is no mask token
    in ``noisy_ids``, so a model cannot tell which positions were corrupted.

    DiffusionGemma (arXiv 2608.00146, sec. 4) trains on this process with
    cross-entropy over every canvas position divided by the canvas size,
    i.e. ``masked_cross_entropy(logits, clean, maskable,
    reduction="sample_mean")`` for one canvas per sample; Sumi-7B
    (arXiv 2606.19005) is a uniform-state model trained from scratch. The
    samplers in this package assume absorbing masks and do not decode such
    models yet.
    """
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be (B, L), got {tuple(input_ids.shape)}")
    device = input_ids.device
    bsz, seq_len = input_ids.shape
    excluded = set(int(i) for i in exclude_token_ids)
    allowed = torch.tensor(
        [i for i in range(vocab_size) if i not in excluded],
        dtype=torch.long,
        device=device,
    )
    if allowed.numel() == 0:
        raise ValueError("no token left to sample after exclusions")
    sched = get_schedule(schedule)
    maskable_b = (
        torch.ones_like(input_ids, dtype=torch.bool)
        if maskable is None
        else maskable.to(device=device, dtype=torch.bool)
    )
    if t is None:
        t = torch.rand(bsz, device=device, generator=generator)
    else:
        t = t.to(device=device, dtype=torch.float32)
        if t.shape != (bsz,):
            raise ValueError(f"t must be shape ({bsz},), got {tuple(t.shape)}")
    p_mask = sched.mask_prob(t).unsqueeze(1).expand(bsz, seq_len).contiguous()
    rand = torch.rand(bsz, seq_len, device=device, generator=generator)
    selected = (rand < p_mask) & maskable_b
    picks = torch.randint(
        0, allowed.numel(), (bsz, seq_len), device=device, generator=generator
    )
    noisy_ids = torch.where(selected, allowed[picks], input_ids)
    return MaskingOutput(
        noisy_ids=noisy_ids, masked_indices=selected, p_mask=p_mask, t=t
    )


def trailing_run(
    input_ids: torch.Tensor, maskable: torch.Tensor, token_id: int
) -> torch.Tensor:
    """(B, L) bool: each row's maximal suffix of maskable ``token_id``."""
    hit = (input_ids == token_id) & maskable.to(torch.bool)
    return torch.cumprod(hit.flip(1).to(torch.long), dim=1).flip(1).bool()


def tie_trailing_run(
    input_ids: torch.Tensor,
    masked_indices: torch.Tensor,
    maskable: torch.Tensor,
    token_id: int,
) -> torch.Tensor:
    """Give every position of the trailing run its first position's draw."""
    run = trailing_run(input_ids, maskable, token_id)
    lengths = run.sum(dim=1)
    has_run = lengths > 0
    if not bool(has_run.any()):
        return masked_indices
    first = (input_ids.shape[1] - lengths).clamp(max=input_ids.shape[1] - 1)
    draw = masked_indices.gather(1, first.unsqueeze(1)).expand_as(run)
    return torch.where(run & has_run.unsqueeze(1), draw, masked_indices)


def complementary_view(
    clean_ids: torch.Tensor,
    masking: MaskingOutput,
    mask_token_id: int,
    maskable: Optional[torch.Tensor] = None,
    min_prob: float = 1e-3,
    always_masked: Optional[torch.Tensor] = None,
) -> MaskingOutput:
    """The complementary masked view of a ``forward_process`` output.

    Every maskable position is masked in exactly one of the two views, so the
    pair supervises each response token once (complementary masking, as in
    DiffuCoder's coupled sampling, Fast-dLLM v2 and LLaDA2.0 SFT).

    The complement masks a position with marginal probability ``1 - p``, so
    its ``p_mask`` is ``1 - p`` and each view is, on its own, an unbiased
    importance-weighted estimate. ``1 - p`` is clamped at ``min_prob``: as
    ``p -> 1`` the complement's weight ``1 / (1 - p)`` diverges; the clamp
    trades a small bias for bounded variance, and pairs best with a clipped
    noise schedule (masking rates away from 0 and 1).

    Args:
        clean_ids: (B, L) clean token ids.
        masking: output of ``forward_process`` on ``clean_ids``.
        maskable: (B, L) bool - the same eligibility mask passed to
            ``forward_process``; ``None`` means every position.
        min_prob: lower bound for the complement's mask probability.
        always_masked: (B, L) bool - positions shown as the mask token in
            both views but outside the loss (the later blocks of a full-canvas
            block SFT batch, ``batch["force_mask"]``).
    """
    if clean_ids.shape != masking.masked_indices.shape:
        raise ValueError("clean_ids must match the masking output's shape")
    if not 0.0 < min_prob <= 1.0:
        raise ValueError("min_prob must be in (0, 1]")
    device = clean_ids.device
    if maskable is None:
        maskable_b = torch.ones_like(clean_ids, dtype=torch.bool)
    else:
        maskable_b = maskable.to(device=device, dtype=torch.bool)
    masked = maskable_b & ~masking.masked_indices.to(device=device, dtype=torch.bool)
    p_mask = (1.0 - masking.p_mask.to(device)).clamp(min=min_prob)
    shown = masked
    if always_masked is not None:
        if always_masked.shape != clean_ids.shape:
            raise ValueError("always_masked must match clean_ids")
        always = always_masked.to(device=device, dtype=torch.bool)
        if bool((always & maskable_b).any()):
            raise ValueError("always_masked positions cannot be maskable")
        shown = masked | always
    noisy_ids = torch.where(shown, mask_token_id, clean_ids)
    return MaskingOutput(
        noisy_ids=noisy_ids, masked_indices=masked, p_mask=p_mask, t=1.0 - masking.t
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
