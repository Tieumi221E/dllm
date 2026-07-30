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
_REDUCTIONS = ("none", "sum", "token_mean", "sample_mean")


def _broadcast_weight(
    weight: torch.Tensor,
    shape: torch.Size,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = torch.as_tensor(weight, device=device, dtype=dtype)
    if value.ndim == 1 and value.shape[0] == shape[0]:
        value = value.unsqueeze(1)
    try:
        return torch.broadcast_to(value, shape)
    except RuntimeError as exc:
        raise ValueError(
            f"{name} with shape {tuple(value.shape)} is not broadcastable "
            f"to {tuple(shape)}"
        ) from exc


def masked_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    selected: torch.Tensor,
    *,
    token_weight: Optional[torch.Tensor] = None,
    sample_weight: Optional[torch.Tensor] = None,
    reduction: str = "token_mean",
) -> torch.Tensor:
    """Cross-entropy over an explicit subset of token positions.

    This is an estimator-agnostic primitive for objectives that do not match
    :func:`diffusion_loss`.  Weighting and reduction are deliberately
    independent:

    - ``token_weight`` is broadcast over ``(B, L)`` before reduction;
    - ``sample_weight`` applies only to ``reduction="sample_mean"``;
    - ``sample_mean`` first averages selected tokens within each non-empty
      sample, then averages samples (weighted when requested).

    ``reduction="none"`` returns a dense ``(B, L)`` tensor with zeros outside
    ``selected``.  This low-level function permits arbitrary estimators;
    callers are responsible for their statistical interpretation.
    """
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {_REDUCTIONS}")
    if logits.ndim != 3:
        raise ValueError("logits must have shape (B, L, V)")
    if target_ids.shape != logits.shape[:2]:
        raise ValueError("target_ids must match logits' (B, L) dimensions")
    if selected.shape != target_ids.shape:
        raise ValueError("selected must have shape (B, L)")
    if sample_weight is not None and reduction != "sample_mean":
        raise ValueError("sample_weight requires reduction='sample_mean'")

    selected = selected.to(device=logits.device, dtype=torch.bool)
    targets = target_ids.to(device=logits.device)
    dense = logits[..., 0].float() * 0.0
    if selected.any():
        values = F.cross_entropy(
            logits[selected].float(), targets[selected], reduction="none"
        )
        if token_weight is not None:
            weights = _broadcast_weight(
                token_weight,
                target_ids.shape,
                name="token_weight",
                device=logits.device,
                dtype=values.dtype,
            )
            values = values * weights[selected]
        dense = dense.masked_scatter(selected, values)

    if reduction == "none":
        return dense
    if reduction == "sum":
        return dense.sum()

    counts = selected.sum(dim=1)
    if reduction == "token_mean":
        return dense.sum() / counts.sum().clamp(min=1).to(dense.dtype)

    valid = counts > 0
    if not valid.any():
        return dense.sum()
    per_sample = dense.sum(dim=1) / counts.clamp(min=1).to(dense.dtype)
    if sample_weight is None:
        return per_sample[valid].mean()
    weights = torch.as_tensor(
        sample_weight, device=logits.device, dtype=dense.dtype
    ).reshape(-1)
    if weights.shape[0] != logits.shape[0]:
        raise ValueError("sample_weight must contain one value per batch row")
    return (per_sample[valid] * weights[valid]).sum() / weights[valid].sum().clamp(
        min=torch.finfo(dense.dtype).eps
    )


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
    masked_indices = masked_indices.to(device=logits.device, dtype=torch.bool)
    token_weight = (
        None
        if not importance_weight
        else 1.0
        / _broadcast_weight(
            p_mask,
            target_ids.shape,
            name="p_mask",
            device=logits.device,
            dtype=torch.float32,
        )
    )
    token_ce = masked_cross_entropy(
        logits,
        target_ids,
        masked_indices,
        token_weight=token_weight,
        reduction="none",
    )

    if norm == "tokens":
        return token_ce.sum() / float(B * L)
    if norm == "maskable":
        denom = maskable.to(device=logits.device, dtype=torch.float32).sum().clamp(
            min=1.0
        )
        return token_ce.sum() / denom
    if norm == "answer":
        lens = (
            maskable.to(device=logits.device, dtype=torch.float32)
            .sum(dim=1)
            .clamp(min=1.0)
        )
        return (token_ce.sum(dim=1) / lens).sum() / float(B)
    if norm == "masked":
        return token_ce.sum() / masked_indices.sum().clamp(min=1).to(token_ce.dtype)
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
