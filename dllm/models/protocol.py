"""Framework-neutral denoiser inputs, outputs, and capability discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional, Protocol, runtime_checkable

import torch

from ..topology import AttentionTopology


@dataclass
class DenoiserInput:
    """A model request independent of a particular framework wrapper."""

    input_ids: Optional[torch.Tensor] = None
    inputs_embeds: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    topology: Optional[AttentionTopology] = None
    use_cache: bool = False
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.input_ids is None) == (self.inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids and inputs_embeds")


@dataclass
class DenoiserOutput:
    """The prediction field returned by a diffusion denoiser."""

    logits: torch.Tensor
    cache: Optional[Any] = None
    hidden_states: Optional[torch.Tensor] = None
    auxiliary: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelCapabilities:
    """Features callers may inspect instead of relying on model type checks."""

    attention_topologies: FrozenSet[str] = frozenset()
    explicit_position_ids: bool = False
    inputs_embeds: bool = False
    cache_semantics: FrozenSet[str] = frozenset()
    prediction_fields: FrozenSet[str] = frozenset()


@runtime_checkable
class Denoiser(Protocol):
    capabilities: ModelCapabilities

    def denoise(self, request: DenoiserInput) -> DenoiserOutput:
        ...


@runtime_checkable
class BlockCacheDenoiser(Protocol):
    """Model with exact ordered-prefix cache extension.

    Cache objects returned by ``build_kv_cache`` must provide
    ``index_select`` and ``extend``. ``forward_block`` returns either logits
    or a tuple whose first item is logits and second item is appendable cache
    state.
    """

    def build_kv_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        ...

    def forward_block(
        self,
        block_ids: torch.Tensor,
        cache: Any,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        ...


@runtime_checkable
class PrefixCacheDenoiser(BlockCacheDenoiser, Protocol):
    """Model that can build an explicitly approximate full-canvas prefix."""

    def build_approximate_prefix_cache(
        self,
        input_ids: torch.Tensor,
        prefix_length: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> DenoiserOutput:
        ...


def extract_logits(output: Any) -> torch.Tensor:
    """Normalize common tensor, structured, and tuple model outputs."""
    if isinstance(output, torch.Tensor):
        return output
    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(output, (tuple, list)) and output and isinstance(
        output[0], torch.Tensor
    ):
        return output[0]
    raise TypeError("model output does not contain a logits tensor")


__all__ = [
    "Denoiser",
    "BlockCacheDenoiser",
    "PrefixCacheDenoiser",
    "DenoiserInput",
    "DenoiserOutput",
    "ModelCapabilities",
    "extract_logits",
]
