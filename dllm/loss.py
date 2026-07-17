"""Training loss and likelihood evaluation for masked diffusion.

``diffusion_loss``: masked cross-entropy with 1/p importance weighting and
explicit normalization choices (``norm="tokens"`` for pretraining,
``"answer"`` for SFT). ``mc_conditional_nll``: low-variance Monte-Carlo
estimator of the conditional NLL bound.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

_NORMS = ("tokens", "maskable", "answer", "masked", "sum")


def diffusion_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    masked_indices: torch.Tensor,
    p_mask: Optional[torch.Tensor] = None,
    norm: str = "tokens",
    maskable: Optional[torch.Tensor] = None,
    importance_weight: bool = True,
) -> torch.Tensor:
    """Masked cross-entropy with 1/p importance weighting.

    Args:
        logits: (B, L, V).
        target_ids: (B, L) clean token ids.
        masked_indices: (B, L) bool - which positions were masked.
        p_mask: (B, L) per-token mask probability (from ``forward_process``).
            Required when ``importance_weight=True``.
        norm: normalization -
            "tokens":   / (B*L)                          (pretraining)
            "maskable": / sum(maskable)
            "answer":   per-sample / row maskable count, then / B   (SFT)
            "masked":   / sum(masked)   (mean CE over masked tokens; only valid
                        with importance_weight=False - uniform weighting)
            "sum":      none.
        maskable: (B, L) bool - required for "maskable"/"answer" norms
            (e.g. the response window in SFT, incl. EOS padding).
        importance_weight: divide each token CE by its p (the bound's
            weight). False -> uniform weighting.
    """
    if norm not in _NORMS:
        raise ValueError(f"norm must be one of {_NORMS}")
    if importance_weight and p_mask is None:
        raise ValueError("p_mask is required when importance_weight=True")
    if norm in ("maskable", "answer") and maskable is None:
        raise ValueError(f"maskable is required for norm='{norm}'")
    if norm == "masked" and importance_weight:
        raise ValueError(
            "norm='masked' with importance_weight=True is biased (the 1/p "
            "weight is re-coupled to the realized mask count); use "
            "norm='tokens'/'answer', or importance_weight=False for uniform "
            "weighting"
        )

    B, L = target_ids.shape
    masked_indices = masked_indices.bool()

    if not masked_indices.any():
        return logits.sum() * 0.0  # keep graph, zero loss

    token_ce = F.cross_entropy(
        logits[masked_indices], target_ids[masked_indices], reduction="none"
    )
    if importance_weight:
        token_ce = token_ce / p_mask[masked_indices]

    if norm == "tokens":
        return token_ce.sum() / float(B * L)
    if norm == "maskable":
        denom = maskable.to(torch.float32).sum().clamp(min=1.0)
        return token_ce.sum() / denom
    if norm == "answer":
        row = masked_indices.nonzero(as_tuple=True)[0]
        lens = maskable.to(torch.float32).sum(dim=1).clamp(min=1.0)  # (B,)
        return (token_ce / lens[row]).sum() / float(B)
    if norm == "masked":
        return token_ce.sum() / float(masked_indices.sum().item())
    return token_ce.sum()


@torch.no_grad()
def mc_conditional_nll(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    mask_token_id: int,
    num_samples: int = 128,
    batch_size: int = 16,
    generator: Optional[torch.Generator] = None,
) -> dict:
    """Monte-Carlo conditional NLL bound.

    ``l ~ U{1..L}``; mask exactly ``l`` response tokens (uniform, without
    replacement); estimate ``(L/l) * sum over masked CE`` and average.

    Args:
        model_fn: callable mapping (N, L_total) ids -> (N, L_total, V) logits.
        prompt_ids: (Lp,) long tensor.
        response_ids: (Lr,) long tensor.
    Returns:
        dict with "nll" (sum bound over the response), "nll_per_token",
        "num_samples".
    """
    device = prompt_ids.device
    Lp, Lr = int(prompt_ids.numel()), int(response_ids.numel())
    if Lr == 0:
        raise ValueError("response is empty")

    estimates = []
    done = 0
    while done < num_samples:
        n = min(batch_size, num_samples - done)
        seqs = torch.empty(n, Lp + Lr, dtype=torch.long, device=device)
        seqs[:, :Lp] = prompt_ids
        seqs[:, Lp:] = response_ids
        ls = torch.randint(1, Lr + 1, (n,), generator=generator, device=device)
        masked = torch.zeros(n, Lr, dtype=torch.bool, device=device)
        for i in range(n):
            perm = torch.randperm(Lr, generator=generator, device=device)
            masked[i, perm[: int(ls[i])]] = True
        seqs[:, Lp:][masked] = mask_token_id

        logits = model_fn(seqs)[:, Lp:, :]  # (n, Lr, V)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            response_ids.unsqueeze(0).expand(n, -1).reshape(-1),
            reduction="none",
        ).view(n, Lr)
        ce = (ce * masked).sum(dim=1)  # sum over masked
        est = ce * (float(Lr) / ls.to(ce.dtype))  # (L/l) * sum
        estimates.append(est)
        done += n

    nll = torch.cat(estimates).mean().item()
    return {"nll": nll, "nll_per_token": nll / Lr, "num_samples": num_samples}
