"""Shared sampling primitives."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def add_gumbel_noise(
    logits: torch.Tensor,
    temperature: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Gumbel-max noise in float64: ``exp(logits) / (-log u)^T``.

    At T=1 this is exact Gumbel-max categorical sampling; at T!=1 it is not
    the same distribution as softmax(logits/T) - see
    ``sample_candidates(mode="multinomial")`` for that semantics.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand(
        logits.shape, dtype=torch.float64, device=logits.device, generator=generator
    )
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def sample_candidates(
    logits: torch.Tensor,
    temperature: float = 0.0,
    mode: str = "gumbel",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Draw a candidate token per position.

    mode="gumbel":      Gumbel-max semantics (see add_gumbel_noise).
    mode="multinomial": softmax(logits/T) categorical sampling (conventional
                        temperature semantics).
    T=0 is greedy argmax in both modes.
    """
    if temperature == 0:
        return logits.argmax(dim=-1)
    if mode == "gumbel":
        return add_gumbel_noise(logits, temperature, generator).argmax(dim=-1)
    if mode == "multinomial":
        probs = F.softmax(logits.float() / temperature, dim=-1)
        flat = probs.view(-1, probs.shape[-1])
        picks = torch.multinomial(flat, 1, generator=generator).squeeze(-1)
        return picks.view(logits.shape[:-1])
    raise ValueError(f"unknown sampling mode: {mode}")


def confidence_scores(
    logits: torch.Tensor,
    x0: torch.Tensor,
    kind: str = "prob",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Per-position commit confidence from the raw-logits softmax.

    kind: "prob" (p of the chosen token), "margin" (top1-top2),
    "neg_entropy", "random" (random-order baseline).
    """
    if kind == "random":
        return torch.rand(x0.shape, device=x0.device, generator=generator)
    p = F.softmax(logits.to(torch.float64), dim=-1)
    if kind == "prob":
        return torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
    if kind == "margin":
        top2 = torch.topk(p, 2, dim=-1).values
        return top2[..., 0] - top2[..., 1]
    if kind == "neg_entropy":
        return -torch.sum(p * torch.log(p + 1e-10), dim=-1)
    raise ValueError(f"unknown confidence kind: {kind}")


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Spread each row's mask count evenly over ``steps``.

    mask_index: (B, L) bool -> (B, steps) int64, rows sum to mask counts.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    out = base.expand(-1, steps).clone()
    idx = torch.arange(steps, device=mask_index.device).unsqueeze(0)
    out = out + (idx < remainder).long()
    return out


def split_steps(total_steps: int, num_blocks: int) -> list:
    """Distribute total steps across blocks (base + leading remainder)."""
    base, rem = divmod(total_steps, num_blocks)
    return [base + (i < rem) for i in range(num_blocks)]


def suppress_tokens_(logits: torch.Tensor, token_ids: Sequence[int]) -> torch.Tensor:
    """Set the given vocab ids to -inf in-place; returns logits."""
    if token_ids:
        ids = torch.as_tensor(list(token_ids), dtype=torch.long, device=logits.device)
        logits.index_fill_(-1, ids, float("-inf"))
    return logits


def select_topk_commits(
    confidence: torch.Tensor,  # (B, L), -inf outside candidates
    quota: torch.Tensor,  # (B,) int
) -> torch.Tensor:
    """Commit mask: per row, the ``quota[b]`` highest-confidence candidates."""
    B, L = confidence.shape
    commit = torch.zeros(B, L, dtype=torch.bool, device=confidence.device)
    for b in range(B):
        k = int(quota[b].item())
        if k > 0:
            _, sel = torch.topk(confidence[b], k=min(k, L))
            commit[b, sel] = True
    # guard: never "commit" a -inf slot (can happen if quota exceeds candidates)
    commit &= confidence > float("-inf")
    return commit


def select_threshold_commits(
    confidence: torch.Tensor,  # (B, L), -inf outside candidates
    threshold: float,
) -> torch.Tensor:
    """Commit everything >= threshold, at least the best one per row that
    still has candidates."""
    candidates = confidence > float("-inf")
    commit = candidates & (confidence >= threshold)
    need_fallback = candidates.any(dim=-1) & ~commit.any(dim=-1)
    if need_fallback.any():
        best = confidence.argmax(dim=-1)
        rows = torch.nonzero(need_fallback, as_tuple=True)[0]
        commit[rows, best[rows]] = True
    return commit


def strip_after_eos(seq: list, eos_token_id: Optional[int]) -> list:
    """EOS is a normal token during sampling; drop it and everything after
    it in the final output."""
    if eos_token_id is None:
        return seq
    try:
        return seq[: seq.index(eos_token_id)]
    except ValueError:
        return seq
