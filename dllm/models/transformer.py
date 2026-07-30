"""Topology-aware Transformer backbone for masked diffusion.

- attention: MHA or GQA via ``num_kv_heads``; optional projection/FF biases
  (``attn_bias``/``ff_bias``) for older checkpoints;
- positions: learned absolute or RoPE with explicit ``position_ids``;
- attention: bidirectional by default, with ordered causal/block topologies;
- KV cache for exact block-wise decoding with dependency metadata.

State-dict layout: ``token_emb / pos_emb / layers.N.attn.{q,k,v,o}_proj /
layers.N.norm1|norm2 / layers.N.ff.w1|w2|w3 / norm / head``.

``build_kv_cache`` + ``forward_block`` is exact under the ordered topology
stored in the cache: the frozen prefix never attends to a later block. Reusing
it for a fully bidirectional canvas remains an explicitly named approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..execution import (
    CacheSemantics,
    EXACT_ORDERED,
)
from ..topology import AttentionTopology, ordered_attention_mask
from .protocol import (
    DenoiserInput,
    DenoiserOutput,
    ModelCapabilities,
)

KVPair = Tuple[torch.Tensor, torch.Tensor]  # (K, V): (B, n_kv, L, head_dim)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.w2 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.w3 = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(self.dropout(self.w1(x) * F.silu(self.w2(x))))


class Rotary(nn.Module):
    """Standard RoPE, applied to q/k at explicit absolute positions."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def cos_sin(self, positions: torch.Tensor, dtype: torch.dtype):
        # positions: (L,) or (B, L) absolute indices
        freqs = positions.float().unsqueeze(-1) * self.inv_freq
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q/k: (B, H, L, hd); cos/sin: (L, hd) or (B, L, hd)
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    else:
        raise ValueError("RoPE positions must have shape (L,) or (B, L)")
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


@dataclass
class KVCache:
    """Per-layer prefix KV plus dependency and position provenance."""

    kvs: List[KVPair] = field(default_factory=list)
    key_mask: Optional[torch.Tensor] = None  # (B, L_prefix) bool, True = attend
    position_ids: Optional[torch.Tensor] = None  # (B, L_prefix)
    group_ids: Optional[torch.Tensor] = None  # (B, L_prefix), ordered topology
    semantics: CacheSemantics = EXACT_ORDERED

    @property
    def length(self) -> int:
        return 0 if not self.kvs else int(self.kvs[0][0].shape[2])

    def extend(
        self,
        block_kvs: List[KVPair],
        block_mask: Optional[torch.Tensor] = None,
        *,
        position_ids: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.Tensor] = None,
    ) -> "KVCache":
        """Return a new cache with a committed block appended."""
        if len(block_kvs) != len(self.kvs):
            raise ValueError("block_kvs must contain one entry per cached layer")
        if not block_kvs:
            return self
        batch = int(block_kvs[0][0].shape[0])
        block_length = int(block_kvs[0][0].shape[2])
        new_kvs = [
            (torch.cat([pk, bk], dim=2), torch.cat([pv, bv], dim=2))
            for (pk, pv), (bk, bv) in zip(self.kvs, block_kvs)
        ]
        device = block_kvs[0][0].device

        if block_mask is not None:
            if block_mask.shape != (batch, block_length):
                raise ValueError("block_mask must match the appended block")
            block_mask = block_mask.bool()
        if self.key_mask is None and block_mask is None:
            new_mask = None
        else:
            prefix_mask = (
                self.key_mask
                if self.key_mask is not None
                else torch.ones(batch, self.length, dtype=torch.bool, device=device)
            )
            suffix_mask = (
                block_mask
                if block_mask is not None
                else torch.ones(batch, block_length, dtype=torch.bool, device=device)
            )
            new_mask = torch.cat([prefix_mask, suffix_mask], dim=1)

        if position_ids is None:
            position_ids = _next_position_ids(self, block_length, device)
        if position_ids.shape != (batch, block_length):
            raise ValueError("position_ids must match the appended block")
        prefix_positions = (
            self.position_ids
            if self.position_ids is not None
            else torch.arange(self.length, device=device).expand(batch, -1)
        )

        if group_ids is None:
            group_ids = _next_group_ids(self, block_length, device)
        if group_ids.shape != (batch, block_length):
            raise ValueError("group_ids must match the appended block")
        prefix_groups = (
            self.group_ids
            if self.group_ids is not None
            else torch.zeros(batch, self.length, dtype=torch.long, device=device)
        )
        return KVCache(
            kvs=new_kvs,
            key_mask=new_mask,
            position_ids=torch.cat([prefix_positions, position_ids], dim=1),
            group_ids=torch.cat([prefix_groups, group_ids], dim=1),
            semantics=self.semantics,
        )

    def index_select(self, idx: torch.Tensor) -> "KVCache":
        """Slice the batch dimension (e.g. keep only still-active samples)."""
        return KVCache(
            kvs=[(k[idx], v[idx]) for k, v in self.kvs],
            key_mask=None if self.key_mask is None else self.key_mask[idx],
            position_ids=(
                None if self.position_ids is None else self.position_ids[idx]
            ),
            group_ids=None if self.group_ids is None else self.group_ids[idx],
            semantics=self.semantics,
        )

    def crop(self, length: int) -> "KVCache":
        """Return the prefix of this cache without mutating the original."""
        if length < 0 or length > self.length:
            raise ValueError(
                f"cache crop length must be in [0, {self.length}]"
            )
        return KVCache(
            kvs=[
                (key[:, :, :length], value[:, :, :length])
                for key, value in self.kvs
            ],
            key_mask=(
                None if self.key_mask is None else self.key_mask[:, :length]
            ),
            position_ids=(
                None
                if self.position_ids is None
                else self.position_ids[:, :length]
            ),
            group_ids=(
                None
                if self.group_ids is None
                else self.group_ids[:, :length]
            ),
            semantics=self.semantics,
        )

    def with_semantics(self, semantics: CacheSemantics) -> "KVCache":
        """Return this cache with new execution provenance."""
        return KVCache(
            kvs=self.kvs,
            key_mask=self.key_mask,
            position_ids=self.position_ids,
            group_ids=self.group_ids,
            semantics=semantics,
        )


def _cache_valid(cache: KVCache, batch: int, device: torch.device) -> torch.Tensor:
    if cache.key_mask is not None:
        return cache.key_mask.bool()
    return torch.ones(batch, cache.length, dtype=torch.bool, device=device)


def _next_position_ids(
    cache: KVCache, length: int, device: torch.device
) -> torch.Tensor:
    batch = int(cache.kvs[0][0].shape[0]) if cache.kvs else 0
    steps = torch.arange(length, device=device).unsqueeze(0)
    if cache.position_ids is None:
        return steps.expand(batch, -1) + cache.length
    if cache.length == 0:
        return steps.expand(batch, -1)
    valid = _cache_valid(cache, batch, device)
    last = cache.position_ids.masked_fill(~valid, -1).max(dim=1).values + 1
    return last.unsqueeze(1) + steps


def _next_group_ids(
    cache: KVCache, length: int, device: torch.device
) -> torch.Tensor:
    batch = int(cache.kvs[0][0].shape[0]) if cache.kvs else 0
    if cache.group_ids is None:
        next_group = torch.ones(batch, dtype=torch.long, device=device)
    elif cache.length == 0:
        next_group = torch.zeros(batch, dtype=torch.long, device=device)
    else:
        valid = _cache_valid(cache, batch, device)
        next_group = cache.group_ids.masked_fill(~valid, -1).max(dim=1).values + 1
    return next_group.unsqueeze(1).expand(-1, length)


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        attention_dropout: float,
        bias: bool = False,
        rotary: Optional[Rotary] = None,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.attn_drop_p = attention_dropout
        self.rotary = rotary

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,  # bool/additive (B,1,Lq,Lk)
        past_kv: Optional[KVPair] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, KVPair]:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.rotary is not None:
            if position_ids is None:
                position_ids = torch.arange(L, device=x.device)
            cos, sin = self.rotary.cos_sin(position_ids, x.dtype)
            q, k = _apply_rope(q, k, cos, sin)

        new_kv: KVPair = (k, v)  # pre-expansion, current tokens only
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        if attn_bias is not None and attn_bias.dtype not in (torch.bool, q.dtype):
            attn_bias = attn_bias.to(q.dtype)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out), new_kv


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        attention_dropout: float,
        resid_dropout: float,
        bias: bool = False,
        ff_bias: bool = False,
        rotary: Optional[Rotary] = None,
    ) -> None:
        super().__init__()
        self.attn = Attention(
            hidden_size,
            num_heads,
            num_kv_heads,
            attention_dropout,
            bias=bias,
            rotary=rotary,
        )
        self.attn_dropout = nn.Dropout(resid_dropout)
        self.ff = SwiGLUFeedForward(
            hidden_size, intermediate_size, resid_dropout, bias=ff_bias
        )
        self.ff_dropout = nn.Dropout(resid_dropout)
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        past_kv: Optional[KVPair] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, KVPair]:
        attn_out, new_kv = self.attn(
            self.norm1(x),
            attn_bias=attn_bias,
            past_kv=past_kv,
            position_ids=position_ids,
        )
        x = x + self.attn_dropout(attn_out)
        x = x + self.ff_dropout(self.ff(self.norm2(x)))
        return x, new_kv


def _padding_mask(attention_mask: torch.Tensor) -> Optional[torch.Tensor]:
    """Convert ``(B, Lk)`` validity to an SDPA key mask."""
    am = attention_mask.bool()
    if am.all():
        return None
    return am.unsqueeze(1).unsqueeze(2)


def _topology_mask(topology: AttentionTopology) -> Optional[torch.Tensor]:
    mask = topology.attention_mask()
    return None if mask.all() else mask


class DiffusionTransformer(nn.Module):
    """Reference mask-predictor with topology-neutral attention."""

    capabilities = ModelCapabilities(
        attention_topologies=frozenset(
            {"bidirectional", "causal", "block_causal", "ordered"}
        ),
        explicit_position_ids=True,
        inputs_embeds=True,
        cache_semantics=frozenset(
            {"exact_ordered", "exact_block_causal"}
        ),
        prediction_fields=frozenset({"same_position"}),
    )

    def __init__(
        self,
        vocab_size: int,
        max_position_embeddings: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        position_embedding: str = "learned",  # "learned" | "rope"
        rope_theta: float = 10000.0,
        attn_bias: bool = False,
        ff_bias: bool = False,
        emb_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        tie_embeddings: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if position_embedding not in ("learned", "rope"):
            raise ValueError("position_embedding must be 'learned' or 'rope'")
        num_kv_heads = num_kv_heads or num_heads
        intermediate_size = intermediate_size or 4 * hidden_size

        self.position_embedding = position_embedding
        self.max_pos = max_position_embeddings
        self.hidden_size = hidden_size
        self.gradient_checkpointing = gradient_checkpointing

        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        if position_embedding == "learned":
            self.pos_emb = nn.Embedding(max_position_embeddings, hidden_size)
            rotary = None
        else:
            self.pos_emb = None
            rotary = Rotary(hidden_size // num_heads, theta=rope_theta)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_heads,
                    num_kv_heads,
                    intermediate_size,
                    attention_dropout,
                    resid_dropout,
                    bias=attn_bias,
                    ff_bias=ff_bias,
                    rotary=rotary,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        if self.pos_emb is not None:
            nn.init.normal_(self.pos_emb.weight, std=0.02)
        if tie_embeddings:
            self.head.weight = self.token_emb.weight

    # ------------------------------------------------------------------ #

    def _check_positions(self, position_ids: torch.Tensor) -> None:
        if position_ids.dtype == torch.bool or position_ids.is_floating_point():
            raise TypeError("position_ids must use an integer dtype")
        if position_ids.numel() == 0:
            return
        if (position_ids < 0).any():
            raise ValueError("position_ids cannot be negative")
        end_pos = int(position_ids.max()) + 1
        if end_pos > self.max_pos:
            raise ValueError(
                f"sequence end {end_pos} exceeds max_position_embeddings={self.max_pos}"
                " (refusing to clamp positions silently)"
            )

    def _embed(
        self,
        input_ids: Optional[torch.Tensor] = None,
        *,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        pos_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids and inputs_embeds")
        if input_ids is not None:
            if input_ids.dim() != 2:
                raise ValueError("input_ids must have shape (batch, length)")
            batch, length = input_ids.shape
            x = self.token_emb(input_ids)
        else:
            assert inputs_embeds is not None
            if inputs_embeds.dim() != 3 or inputs_embeds.shape[-1] != self.hidden_size:
                raise ValueError(
                    "inputs_embeds must have shape (batch, length, hidden_size)"
                )
            batch, length, _ = inputs_embeds.shape
            x = inputs_embeds

        if position_ids is None:
            position_ids = torch.arange(
                pos_offset, pos_offset + length, device=x.device
            ).expand(batch, -1)
        if position_ids.shape != (batch, length):
            raise ValueError("position_ids must match the input shape")
        if position_ids.device != x.device:
            raise ValueError("position_ids and inputs must be on the same device")
        self._check_positions(position_ids)
        if self.pos_emb is not None:
            x = x + self.pos_emb(position_ids)
        return self.emb_dropout(x), position_ids

    @staticmethod
    def _prepare_attention(
        batch: int,
        length: int,
        device: torch.device,
        attention_mask: Optional[torch.Tensor],
        attn_bias: Optional[torch.Tensor],
        topology: Optional[AttentionTopology],
    ) -> Tuple[Optional[torch.Tensor], Optional[AttentionTopology]]:
        if attn_bias is not None and topology is not None:
            raise ValueError("attn_bias and topology are mutually exclusive")
        if attention_mask is not None:
            if attention_mask.shape != (batch, length):
                raise ValueError("attention_mask must match the input shape")
            if attention_mask.device != device:
                raise ValueError("attention_mask and inputs must be on the same device")
        if topology is not None:
            if topology.group_ids.shape != (batch, length):
                raise ValueError("topology must match the input shape")
            if topology.group_ids.device != device:
                raise ValueError("topology and inputs must be on the same device")
            if attention_mask is not None:
                topology = topology.with_valid(attention_mask)
            return _topology_mask(topology), topology
        if attn_bias is not None:
            return attn_bias, None
        if attention_mask is not None:
            return _padding_mask(attention_mask), None
        return None, None

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        return_kvs: bool = False,
        *,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        topology: Optional[AttentionTopology] = None,
        return_dict: bool = False,
    ):
        """Predict every input position under an explicit attention contract.

        Args:
            attention_mask: (B, L) 1/True for real tokens (padding excluded).
            attn_bias: legacy raw boolean/additive attention mask. It cannot be
                combined with ``topology`` and still overrides ``attention_mask``.
            topology: ordered dependency structure; defaults to bidirectional.
            position_ids: explicit logical positions, independent of tensor columns.
            return_kvs: also return per-layer (K, V) for the whole sequence
                through the legacy tuple API.
            return_dict: return :class:`DenoiserOutput`; otherwise preserve the
                1.1 tensor/tuple return convention.
        Returns:
            logits (B, L, V), or (logits, kvs) when ``return_kvs``.
        """
        raw_attention_bias = attn_bias is not None
        x, position_ids = self._embed(
            input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
        )
        batch, length, _ = x.shape
        attn_bias, resolved_topology = self._prepare_attention(
            batch,
            length,
            x.device,
            attention_mask,
            attn_bias,
            topology,
        )
        kvs: List[KVPair] = []
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and not return_kvs:
                def layer_forward(
                    hidden: torch.Tensor, current_layer: TransformerBlock = layer
                ) -> torch.Tensor:
                    return current_layer(
                        hidden,
                        attn_bias=attn_bias,
                        position_ids=position_ids,
                    )[0]

                x = checkpoint(layer_forward, x, use_reentrant=False)
            else:
                x, kv = layer(
                    x, attn_bias=attn_bias, position_ids=position_ids
                )
                if return_kvs:
                    kvs.append(kv)
        logits = self.head(self.norm(x))
        if return_dict:
            cache = None
            if return_kvs:
                if raw_attention_bias:
                    raise ValueError(
                        "a raw attn_bias cannot provide reusable cache provenance; "
                        "use AttentionTopology or the legacy tuple output"
                    )
                valid = (
                    resolved_topology.valid
                    if resolved_topology is not None
                    else (
                        attention_mask.bool()
                        if attention_mask is not None
                        else torch.ones(
                            batch, length, dtype=torch.bool, device=x.device
                        )
                    )
                )
                groups = (
                    resolved_topology.group_ids
                    if resolved_topology is not None
                    else torch.zeros(
                        batch, length, dtype=torch.long, device=x.device
                    )
                )
                cache = KVCache(
                    kvs=kvs,
                    key_mask=None if valid.all() else valid,
                    position_ids=position_ids,
                    group_ids=groups,
                    semantics=EXACT_ORDERED,
                )
            return DenoiserOutput(logits=logits, cache=cache)
        return (logits, kvs) if return_kvs else logits

    def denoise(self, request: DenoiserInput) -> DenoiserOutput:
        """Execute a framework-neutral denoiser request."""
        return self.forward(
            request.input_ids,
            attention_mask=request.attention_mask,
            inputs_embeds=request.inputs_embeds,
            position_ids=request.position_ids,
            topology=request.topology,
            return_kvs=request.use_cache,
            return_dict=True,
            **request.model_kwargs,
        )

    # -------------------------- KV-cache API -------------------------- #

    def build_kv_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        position_ids: Optional[torch.Tensor] = None,
        topology: Optional[AttentionTopology] = None,
    ) -> KVCache:
        """Encode a prefix for exact ordered/block-causal extension."""
        x, position_ids = self._embed(
            input_ids,
            position_ids=position_ids,
        )
        batch, length, _ = x.shape
        bias, resolved_topology = self._prepare_attention(
            batch,
            length,
            x.device,
            attention_mask,
            None,
            topology,
        )
        kvs: List[KVPair] = []
        for layer in self.layers:
            x, kv = layer(x, attn_bias=bias, position_ids=position_ids)
            kvs.append(kv)
        valid = (
            resolved_topology.valid
            if resolved_topology is not None
            else (
                attention_mask.bool()
                if attention_mask is not None
                else torch.ones(
                    batch, length, dtype=torch.bool, device=x.device
                )
            )
        )
        groups = (
            resolved_topology.group_ids
            if resolved_topology is not None
            else torch.zeros(batch, length, dtype=torch.long, device=x.device)
        )
        return KVCache(
            kvs=kvs,
            key_mask=None if valid.all() else valid,
            position_ids=position_ids,
            group_ids=groups,
            semantics=EXACT_ORDERED,
        )

    def build_approximate_prefix_cache(
        self,
        input_ids: torch.Tensor,
        prefix_length: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> DenoiserOutput:
        """Freeze a prefix encoded inside a bidirectional full canvas.

        This is the explicit approximation used by windowed full-canvas
        decoding: the returned prefix states are not recomputed as the active
        block changes.
        """
        if prefix_length < 0 or prefix_length > input_ids.shape[1]:
            raise ValueError("prefix_length must be within input_ids")
        output = self.forward(
            input_ids,
            attention_mask=attention_mask,
            return_kvs=True,
            return_dict=True,
        )
        if output.cache is None:
            raise RuntimeError("full-canvas forward did not return a cache")
        cache = output.cache.crop(prefix_length).with_semantics(
            CacheSemantics.approximate_for(
                "bidirectional_canvas",
                "prefix states are frozen while the active window changes",
            )
        )
        return DenoiserOutput(logits=output.logits, cache=cache)

    def forward_block(
        self,
        block_ids: torch.Tensor,
        cache: KVCache,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        position_ids: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ):
        """Forward one block against an ordered-prefix cache.

        The block attends to [prefix + block]; the prefix representation is
        frozen (never sees the block). Returns block logits and the block's
        per-layer KVs (pre-expansion) for :meth:`KVCache.extend`. Structured
        output returns the already-extended cache.
        """
        if block_ids.dim() != 2:
            raise ValueError("block_ids must have shape (batch, length)")
        batch, length = block_ids.shape
        if len(cache.kvs) != len(self.layers):
            raise ValueError("cache must contain one entry per model layer")
        if cache.kvs and cache.kvs[0][0].shape[0] != batch:
            raise ValueError("cache and block must have the same batch size")
        if attention_mask is None:
            block_valid = torch.ones(
                batch, length, dtype=torch.bool, device=block_ids.device
            )
        else:
            if attention_mask.shape != block_ids.shape:
                raise ValueError("attention_mask must match block_ids")
            if attention_mask.device != block_ids.device:
                raise ValueError(
                    "attention_mask and block_ids must be on the same device"
                )
            block_valid = attention_mask.bool()
        if position_ids is None:
            position_ids = _next_position_ids(cache, length, block_ids.device)
        if group_ids is None:
            group_ids = _next_group_ids(cache, length, block_ids.device)
        if group_ids.shape != block_ids.shape:
            raise ValueError("group_ids must match block_ids")
        if group_ids.device != block_ids.device:
            raise ValueError("group_ids and block_ids must be on the same device")
        if group_ids.dtype == torch.bool or group_ids.is_floating_point():
            raise TypeError("group_ids must use an integer dtype")

        x, position_ids = self._embed(
            block_ids, position_ids=position_ids
        )
        prefix_valid = _cache_valid(cache, batch, block_ids.device)
        prefix_groups = (
            cache.group_ids
            if cache.group_ids is not None
            else torch.zeros(
                batch, cache.length, dtype=torch.long, device=block_ids.device
            )
        )
        if cache.semantics.exact and cache.length:
            comparable = prefix_valid.any(dim=1) & block_valid.any(dim=1)
            prefix_max = prefix_groups.masked_fill(
                ~prefix_valid, torch.iinfo(prefix_groups.dtype).min
            ).max(dim=1).values
            block_min = group_ids.masked_fill(
                ~block_valid, torch.iinfo(group_ids.dtype).max
            ).min(dim=1).values
            if bool((comparable & (block_min <= prefix_max)).any()):
                raise ValueError(
                    "exact ordered cache extension requires valid block "
                    "groups to follow all valid cached groups"
                )
        key_groups = torch.cat([prefix_groups, group_ids], dim=1)
        key_valid = torch.cat([prefix_valid, block_valid], dim=1)
        bias = ordered_attention_mask(
            group_ids, key_groups, block_valid, key_valid
        )
        if bias.all():
            bias = None

        new_kvs: List[KVPair] = []
        for layer, pkv in zip(self.layers, cache.kvs):
            x, kv = layer(
                x,
                attn_bias=bias,
                past_kv=pkv,
                position_ids=position_ids,
            )
            new_kvs.append(kv)
        logits = self.head(self.norm(x))
        if return_dict:
            extended = cache.extend(
                new_kvs,
                None if block_valid.all() else block_valid,
                position_ids=position_ids,
                group_ids=group_ids,
            )
            return DenoiserOutput(logits=logits, cache=extended)
        return logits, new_kvs
