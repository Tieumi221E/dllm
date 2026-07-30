"""Small runtime checks for third-party dllm protocol implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .models.protocol import (
    BlockCacheDenoiser,
    Denoiser,
    DenoiserInput,
    DenoiserOutput,
    PrefixCacheDenoiser,
    extract_logits,
)


class ContractViolation(ValueError):
    """A model advertises a protocol but violates its runtime contract."""


@dataclass(frozen=True)
class ContractReport:
    contract: str
    checks: Tuple[str, ...]


def _check_logits(logits: torch.Tensor, ids: torch.Tensor) -> None:
    if logits.dim() != 3:
        raise ContractViolation("logits must have shape (batch, length, vocab)")
    if logits.shape[:2] != ids.shape:
        raise ContractViolation("logits batch and length must match input ids")
    if logits.device != ids.device:
        raise ContractViolation("logits and input ids must share a device")
    if torch.isnan(logits).any():
        raise ContractViolation("logits cannot contain NaN")


@torch.no_grad()
def validate_denoiser(
    model,
    request: DenoiserInput,
) -> ContractReport:
    """Execute and validate the framework-neutral denoiser boundary."""
    if not isinstance(model, Denoiser):
        raise ContractViolation("model does not implement Denoiser")
    output = model.denoise(request)
    if not isinstance(output, DenoiserOutput):
        raise ContractViolation("Denoiser.denoise must return DenoiserOutput")
    ids = request.input_ids
    if ids is None:
        if request.inputs_embeds is None:
            raise ContractViolation("request has no model input")
        expected = request.inputs_embeds.shape[:2]
        if output.logits.dim() != 3:
            raise ContractViolation(
                "logits must have shape (batch, length, vocab)"
            )
        if output.logits.shape[:2] != expected:
            raise ContractViolation(
                "logits batch and length must match input embeddings"
            )
        if output.logits.device != request.inputs_embeds.device:
            raise ContractViolation(
                "logits and input embeddings must share a device"
            )
        if torch.isnan(output.logits).any():
            raise ContractViolation("logits cannot contain NaN")
    else:
        _check_logits(output.logits, ids)
    return ContractReport(
        contract="Denoiser",
        checks=("structured_output", "prediction_shape", "device", "no_nan"),
    )


@torch.no_grad()
def validate_block_cache_denoiser(
    model,
    prefix_ids: torch.Tensor,
    block_ids: torch.Tensor,
) -> ContractReport:
    """Exercise cache construction and one block extension."""
    if not isinstance(model, BlockCacheDenoiser):
        raise ContractViolation("model does not implement BlockCacheDenoiser")
    cache = model.build_kv_cache(prefix_ids)
    for method in ("index_select", "extend"):
        if not callable(getattr(cache, method, None)):
            raise ContractViolation(f"cache must provide {method}")
    semantics = getattr(cache, "semantics", None)
    if semantics is None or not bool(getattr(semantics, "exact", False)):
        raise ContractViolation(
            "ordered-prefix cache must declare exact semantics"
        )
    batch_indices = torch.arange(prefix_ids.shape[0], device=prefix_ids.device)
    selected = cache.index_select(batch_indices)
    if selected is None:
        raise ContractViolation("cache.index_select must return a cache")
    output = model.forward_block(block_ids, cache)
    _check_logits(extract_logits(output), block_ids)
    checks = [
        "cache_build",
        "batch_select",
        "exact_provenance",
        "block_shape",
    ]
    if isinstance(output, (tuple, list)) and len(output) > 1:
        extended = cache.extend(output[1])
        if extended is None:
            raise ContractViolation("cache.extend must return a cache")
        checks.append("cache_extend")
    return ContractReport(
        contract="BlockCacheDenoiser",
        checks=tuple(checks),
    )


@torch.no_grad()
def validate_prefix_cache_denoiser(
    model,
    canvas_ids: torch.Tensor,
    prefix_length: int,
) -> ContractReport:
    """Check that full-canvas prefix caching declares approximation."""
    if not isinstance(model, PrefixCacheDenoiser):
        raise ContractViolation("model does not implement PrefixCacheDenoiser")
    output = model.build_approximate_prefix_cache(
        canvas_ids,
        prefix_length=prefix_length,
    )
    if not isinstance(output, DenoiserOutput) or output.cache is None:
        raise ContractViolation(
            "prefix cache builder must return DenoiserOutput with cache"
        )
    semantics = getattr(output.cache, "semantics", None)
    if semantics is None or bool(getattr(semantics, "exact", True)):
        raise ContractViolation(
            "full-canvas prefix cache must declare approximate semantics"
        )
    _check_logits(output.logits, canvas_ids)
    return ContractReport(
        contract="PrefixCacheDenoiser",
        checks=("structured_output", "cache_present", "approximate_provenance"),
    )


__all__ = [
    "ContractReport",
    "ContractViolation",
    "validate_block_cache_denoiser",
    "validate_denoiser",
    "validate_prefix_cache_denoiser",
]
