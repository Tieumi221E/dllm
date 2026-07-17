"""Bidirectional Transformer backbone for masked diffusion.

- attention: MHA or GQA via ``num_kv_heads``; optional projection/FF biases
  (``attn_bias``/``ff_bias``) for older checkpoints;
- positions: learned absolute or RoPE (``position_embedding``);
- KV cache for block-wise decoding; the padding mask travels inside the
  cache, and position overflow raises instead of clamping.

State-dict layout: ``token_emb / pos_emb / layers.N.attn.{q,k,v,o}_proj /
layers.N.norm1|norm2 / layers.N.ff.w1|w2|w3 / norm / head``.

Cache semantics: ``build_kv_cache`` + ``forward_block`` is *block-causal* -
the prefix never attends to the new block. It equals a full forward under a
block-causal mask (tested); it is an approximation of a fully bidirectional
canvas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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
    """Standard RoPE, applied to q/k with an arbitrary position offset."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def cos_sin(self, positions: torch.Tensor, dtype: torch.dtype):
        # positions: (L,) absolute indices
        freqs = torch.outer(positions.float(), self.inv_freq)  # (L, hd/2)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q/k: (B, H, L, hd); cos/sin: (L, hd)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


@dataclass
class KVCache:
    """Per-layer prefix KV plus the prefix key-padding mask."""

    kvs: List[KVPair] = field(default_factory=list)
    key_mask: Optional[torch.Tensor] = None  # (B, L_prefix) bool, True = attend

    @property
    def length(self) -> int:
        return 0 if not self.kvs else int(self.kvs[0][0].shape[2])

    def extend(
        self, block_kvs: List[KVPair], block_mask: Optional[torch.Tensor] = None
    ) -> "KVCache":
        """Return a new cache with a committed block appended."""
        new_kvs = [
            (torch.cat([pk, bk], dim=2), torch.cat([pv, bv], dim=2))
            for (pk, pv), (bk, bv) in zip(self.kvs, block_kvs)
        ]
        new_mask = self.key_mask
        if new_mask is not None:
            B = new_mask.shape[0]
            Lb = block_kvs[0][0].shape[2]
            bm = (
                block_mask
                if block_mask is not None
                else torch.ones(B, Lb, dtype=torch.bool, device=new_mask.device)
            )
            new_mask = torch.cat([new_mask, bm], dim=1)
        return KVCache(kvs=new_kvs, key_mask=new_mask)

    def index_select(self, idx: torch.Tensor) -> "KVCache":
        """Slice the batch dimension (e.g. keep only still-active samples)."""
        return KVCache(
            kvs=[(k[idx], v[idx]) for k, v in self.kvs],
            key_mask=None if self.key_mask is None else self.key_mask[idx],
        )


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
        attn_bias: Optional[torch.Tensor] = None,  # additive, broadcastable (B,1,Lq,Lk)
        past_kv: Optional[KVPair] = None,
        pos_offset: int = 0,
    ) -> Tuple[torch.Tensor, KVPair]:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.rotary is not None:
            pos = torch.arange(pos_offset, pos_offset + L, device=x.device)
            cos, sin = self.rotary.cos_sin(pos, x.dtype)
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
        pos_offset: int = 0,
    ) -> Tuple[torch.Tensor, KVPair]:
        attn_out, new_kv = self.attn(
            self.norm1(x), attn_bias=attn_bias, past_kv=past_kv, pos_offset=pos_offset
        )
        x = x + self.attn_dropout(attn_out)
        x = x + self.ff_dropout(self.ff(self.norm2(x)))
        return x, new_kv


def _padding_bias(
    attention_mask: torch.Tensor, dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """(B, Lk) bool/int -> additive (B,1,1,Lk) bias, or None if nothing masked."""
    am = attention_mask.bool()
    if am.all():
        return None
    bias = torch.zeros(am.shape[0], 1, 1, am.shape[1], dtype=dtype, device=am.device)
    return bias.masked_fill(~am.unsqueeze(1).unsqueeze(2), torch.finfo(dtype).min)


class DiffusionTransformer(nn.Module):
    """Bidirectional (no causal mask) Transformer mask-predictor."""

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

    def _check_length(self, end_pos: int) -> None:
        if end_pos > self.max_pos:
            raise ValueError(
                f"sequence end {end_pos} exceeds max_position_embeddings={self.max_pos}"
                " (refusing to clamp positions silently)"
            )

    def _embed(self, input_ids: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        b, s = input_ids.shape
        self._check_length(pos_offset + s)
        x = self.token_emb(input_ids)
        if self.pos_emb is not None:
            pos = torch.arange(pos_offset, pos_offset + s, device=input_ids.device)
            x = x + self.pos_emb(pos).unsqueeze(0)
        return self.emb_dropout(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        return_kvs: bool = False,
    ):
        """Full bidirectional forward.

        Args:
            attention_mask: (B, L) 1/True for real tokens (padding excluded).
            attn_bias: optional explicit additive bias (overrides attention_mask).
            return_kvs: also return per-layer (K, V) for the whole sequence
                (used to build a full-canvas prefix cache).
        Returns:
            logits (B, L, V), or (logits, kvs) when ``return_kvs``.
        """
        x = self._embed(input_ids)
        if attn_bias is None and attention_mask is not None:
            attn_bias = _padding_bias(attention_mask, x.dtype)
        kvs: List[KVPair] = []
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and not return_kvs:
                x, _ = checkpoint(layer, x, attn_bias, use_reentrant=False)
            else:
                x, kv = layer(x, attn_bias=attn_bias)
                if return_kvs:
                    kvs.append(kv)
        logits = self.head(self.norm(x))
        return (logits, kvs) if return_kvs else logits

    # -------------------------- KV-cache API -------------------------- #

    def build_kv_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> KVCache:
        """Encode a prefix (prefix-only context) and return its KV cache."""
        x = self._embed(input_ids)
        bias = None
        key_mask = None
        if attention_mask is not None:
            key_mask = attention_mask.bool()
            bias = _padding_bias(attention_mask, x.dtype)
        kvs: List[KVPair] = []
        for layer in self.layers:
            x, kv = layer(x, attn_bias=bias)
            kvs.append(kv)
        return KVCache(kvs=kvs, key_mask=key_mask)

    def forward_block(
        self,
        block_ids: torch.Tensor,
        cache: KVCache,
    ) -> Tuple[torch.Tensor, List[KVPair]]:
        """Forward one block against a prefix cache (block-causal semantics).

        The block attends to [prefix + block]; the prefix representation is
        frozen (never sees the block). Returns block logits and the block's
        per-layer KVs (pre-expansion) for :meth:`KVCache.extend`.
        """
        B, s = block_ids.shape
        offset = cache.length
        x = self._embed(block_ids, pos_offset=offset)
        bias = None
        if cache.key_mask is not None:
            full_mask = torch.cat(
                [
                    cache.key_mask,
                    torch.ones(B, s, dtype=torch.bool, device=block_ids.device),
                ],
                dim=1,
            )
            bias = _padding_bias(full_mask, x.dtype)
        new_kvs: List[KVPair] = []
        for layer, pkv in zip(self.layers, cache.kvs):
            x, kv = layer(x, attn_bias=bias, past_kv=pkv, pos_offset=offset)
            new_kvs.append(kv)
        return self.head(self.norm(x)), new_kvs
