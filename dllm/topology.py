"""Ordered attention topologies for diffusion language models.

An attention topology separates dependency structure from tensor layout.
Tokens are assigned to ordered groups: a query may attend to keys in its own
group and every earlier group. Tokens in the same group remain bidirectional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


def ordered_attention_mask(
    query_group_ids: torch.Tensor,
    key_group_ids: torch.Tensor,
    query_valid: Optional[torch.Tensor] = None,
    key_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compile ordered groups into an SDPA boolean mask.

    ``True`` means that the query-key pair is allowed. Group tensors have
    shape ``(B, L)`` and may use broadcasting only in the batch dimension.
    """
    if query_group_ids.dim() != 2 or key_group_ids.dim() != 2:
        raise ValueError("group ids must have shape (batch, length)")
    if query_group_ids.shape[0] != key_group_ids.shape[0]:
        raise ValueError("query and key group ids must have the same batch size")
    if query_group_ids.device != key_group_ids.device:
        raise ValueError("query and key group ids must be on the same device")

    allowed = key_group_ids.unsqueeze(1) <= query_group_ids.unsqueeze(2)
    if key_valid is not None:
        if key_valid.shape != key_group_ids.shape:
            raise ValueError("key_valid must match key_group_ids")
        allowed = allowed & key_valid.bool().unsqueeze(1)
    if query_valid is not None:
        if query_valid.shape != query_group_ids.shape:
            raise ValueError("query_valid must match query_group_ids")
        allowed = allowed & query_valid.bool().unsqueeze(2)
    return allowed.unsqueeze(1)


def _valid_tensor(
    valid: Optional[torch.Tensor],
    batch_size: Optional[int],
    length: Optional[int],
    device: Optional[torch.device],
) -> torch.Tensor:
    if valid is not None:
        if valid.dim() != 2:
            raise ValueError("valid must have shape (batch, length)")
        if batch_size is not None and valid.shape[0] != batch_size:
            raise ValueError("batch_size does not match valid")
        if length is not None and valid.shape[1] != length:
            raise ValueError("length does not match valid")
        return valid.bool()
    if batch_size is None or length is None:
        raise ValueError("provide valid or both batch_size and length")
    return torch.ones(batch_size, length, dtype=torch.bool, device=device)


@dataclass(frozen=True)
class AttentionTopology:
    """An ordered partition of a token sequence.

    A query in group ``g`` attends to keys in groups ``<= g``. This one rule
    represents bidirectional, causal, block-causal, prefix-LM, and arbitrary
    ordered-block attention without coupling those modes to a model class.
    """

    group_ids: torch.Tensor
    valid: Optional[torch.Tensor] = None
    name: str = "ordered"

    def __post_init__(self) -> None:
        if self.group_ids.dim() != 2:
            raise ValueError("group_ids must have shape (batch, length)")
        if self.group_ids.dtype == torch.bool or self.group_ids.is_floating_point():
            raise TypeError("group_ids must use an integer dtype")
        valid = (
            torch.ones_like(self.group_ids, dtype=torch.bool)
            if self.valid is None
            else self.valid.bool()
        )
        if valid.shape != self.group_ids.shape:
            raise ValueError("valid must match group_ids")
        if valid.device != self.group_ids.device:
            raise ValueError("valid and group_ids must be on the same device")
        if valid.any() and (self.group_ids[valid] < 0).any():
            raise ValueError("valid tokens cannot have negative group ids")
        object.__setattr__(self, "valid", valid)

    @property
    def batch_size(self) -> int:
        return int(self.group_ids.shape[0])

    @property
    def length(self) -> int:
        return int(self.group_ids.shape[1])

    def attention_mask(self) -> torch.Tensor:
        """Return a boolean ``(B, 1, L, L)`` SDPA mask."""
        return ordered_attention_mask(
            self.group_ids, self.group_ids, self.valid, self.valid
        )

    def with_valid(self, valid: torch.Tensor) -> "AttentionTopology":
        """Compose an additional padding/validity mask."""
        if valid.shape != self.group_ids.shape:
            raise ValueError("valid must match group_ids")
        return AttentionTopology(
            self.group_ids, self.valid & valid.bool(), name=self.name
        )

    def index_select(self, idx: torch.Tensor) -> "AttentionTopology":
        return AttentionTopology(
            self.group_ids[idx], self.valid[idx], name=self.name
        )

    @classmethod
    def bidirectional(
        cls,
        valid: Optional[torch.Tensor] = None,
        *,
        batch_size: Optional[int] = None,
        length: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> "AttentionTopology":
        valid = _valid_tensor(valid, batch_size, length, device)
        return cls(torch.zeros_like(valid, dtype=torch.long), valid, "bidirectional")

    @classmethod
    def causal(
        cls,
        valid: Optional[torch.Tensor] = None,
        *,
        batch_size: Optional[int] = None,
        length: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> "AttentionTopology":
        valid = _valid_tensor(valid, batch_size, length, device)
        groups = torch.arange(valid.shape[1], device=valid.device).expand(
            valid.shape[0], -1
        )
        return cls(groups, valid, "causal")

    @classmethod
    def block_causal(
        cls,
        block_size: int,
        valid: Optional[torch.Tensor] = None,
        *,
        batch_size: Optional[int] = None,
        length: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> "AttentionTopology":
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        valid = _valid_tensor(valid, batch_size, length, device)
        groups = (
            torch.arange(valid.shape[1], device=valid.device) // block_size
        ).expand(valid.shape[0], -1)
        return cls(groups, valid, "block_causal")

    @classmethod
    def from_boundaries(
        cls,
        boundaries: Sequence[int],
        length: int,
        *,
        batch_size: int = 1,
        valid: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        name: str = "block_causal",
    ) -> "AttentionTopology":
        """Build ordered segments from exclusive end offsets.

        For example, ``boundaries=[4, 8, 12]`` creates three bidirectional
        groups of length four. The final boundary may be omitted.
        """
        if length < 0:
            raise ValueError("length must be non-negative")
        if device is None and valid is not None:
            device = valid.device
        if valid is not None:
            if valid.dim() != 2 or valid.shape[1] != length:
                raise ValueError("valid must have shape (batch, length)")
            if batch_size == 1 and valid.shape[0] != 1:
                batch_size = int(valid.shape[0])
            elif valid.shape[0] != batch_size:
                raise ValueError("batch_size does not match valid")
        bounds = torch.as_tensor(list(boundaries), dtype=torch.long, device=device)
        if bounds.numel() and (
            (bounds <= 0).any()
            or (bounds[1:] <= bounds[:-1]).any()
            or int(bounds[-1]) > length
        ):
            raise ValueError("boundaries must be increasing offsets within length")
        pos = torch.arange(length, device=bounds.device)
        groups = torch.searchsorted(bounds, pos, right=True).expand(batch_size, -1)
        if valid is None:
            valid = torch.ones(
                batch_size, length, dtype=torch.bool, device=groups.device
            )
        return cls(groups, valid, name)


__all__ = ["AttentionTopology", "ordered_attention_mask"]
