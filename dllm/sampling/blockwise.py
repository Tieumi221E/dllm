"""Incremental block-wise sampler (truncated canvas + KV cache).

Blocks are appended one at a time; future blocks *do not exist* in the
sequence, so the committed prefix can be KV-cached exactly under block-causal
attention. Committed block KVs are recomputed from the final (clean) tokens
before being appended to the cache.

This regime matches models trained with a truncated-canvas block SFT
(``data.BlockSFTCollator(canvas="truncated")``). It is NOT the full-canvas
semantics - for models trained on a full canvas, prefer :mod:`.canvas`
(optionally with ``prefix_cache=True``).

Per-sample EOS early exit: finished samples get EOS-filled blocks so the
shared cache stays aligned across the batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from ..models.transformer import DiffusionTransformer
from .utils import (
    confidence_scores,
    get_num_transfer_tokens,
    sample_candidates,
    select_threshold_commits,
    select_topk_commits,
    strip_after_eos,
    suppress_tokens_,
)

_COMMITS = ("transfer", "threshold")


@dataclass
class BlockwiseConfig:
    gen_length: int = 256
    block_length: int = 32
    steps_per_block: int = 16
    temperature: float = 0.0
    sampling: str = "gumbel"  # "gumbel" | "multinomial"
    commit: str = "transfer"
    confidence: str = "prob"
    threshold: float = 0.9
    allow_mask_prediction: bool = False
    suppress_token_ids: Sequence[int] = ()
    eos_token_id: Optional[int] = None  # enables early exit + stripping
    use_cache: bool = True  # performance only: attention is always
    # block-causal, under which the cache is
    # exact; pair training with
    # :func:`block_causal_bias`


def block_causal_bias(
    boundaries: Sequence[int],
    length: int,
    device,
    dtype=torch.float32,
) -> torch.Tensor:
    """Additive staircase bias: position in segment j attends to segments <= j.

    ``boundaries``: segment end offsets, e.g. [prompt_len, prompt_len+BL, ...].
    Also useful in training loops to make truncated-canvas block SFT exactly
    cache-consistent.
    """
    bounds = torch.as_tensor(list(boundaries), device=device)
    pos = torch.arange(length, device=device)
    seg = torch.searchsorted(bounds, pos, right=True)  # segment index
    allow = seg.unsqueeze(1) >= seg.unsqueeze(0)  # row >= col
    bias = torch.zeros(1, 1, length, length, dtype=dtype, device=device)
    return bias.masked_fill(~allow.unsqueeze(0).unsqueeze(0), torch.finfo(dtype).min)


@dataclass
class BlockwiseOutput:
    sequences: List[List[int]]  # per sample, EOS-stripped responses
    canvas: torch.Tensor  # (B, Lp + n_blocks*block_len) raw
    step_map: torch.Tensor  # (B, gen) global commit step (-1 unset)
    nfe: int = 0


@torch.no_grad()
def generate_blockwise(
    model: DiffusionTransformer,
    prompt_ids: torch.Tensor,
    mask_token_id: int,
    config: Optional[BlockwiseConfig] = None,
    num_samples: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> BlockwiseOutput:
    """Generate with incremental blocks.

    Args:
        prompt_ids: (Lp,) or (B, Lp). All prompts must share the same length
            (no padding support in the cache path - pad upstream if needed).
        num_samples: if given and prompt is a single row, replicate it.
    """
    cfg = config or BlockwiseConfig()
    if cfg.commit not in _COMMITS:
        raise ValueError(f"commit must be one of {_COMMITS}")
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if num_samples is not None and prompt_ids.shape[0] == 1:
        prompt_ids = prompt_ids.expand(num_samples, -1).contiguous()
    device = prompt_ids.device
    B, Lp = prompt_ids.shape
    BL = cfg.block_length
    n_blocks = (cfg.gen_length + BL - 1) // BL

    suppress_ids = list(cfg.suppress_token_ids)
    if not cfg.allow_mask_prediction:
        suppress_ids.append(mask_token_id)

    eos = cfg.eos_token_id
    eos_hit = torch.zeros(B, dtype=torch.bool, device=device)
    committed: List[torch.Tensor] = []
    step_map = torch.full((B, n_blocks * BL), -1, dtype=torch.long, device=device)

    nfe = 0
    global_step = 0
    cache = model.build_kv_cache(prompt_ids) if cfg.use_cache else None
    if cfg.use_cache:
        nfe += 1
    prefix_no_cache = prompt_ids  # grows as blocks commit (no-cache path)

    for blk in range(n_blocks):
        if eos is not None and bool(eos_hit.all()):
            break
        actual = min(BL, cfg.gen_length - blk * BL)
        fill_id = eos if eos is not None else mask_token_id
        block = torch.full((B, actual), fill_id, dtype=torch.long, device=device)
        active = (
            ~eos_hit
            if eos is not None
            else torch.ones(B, dtype=torch.bool, device=device)
        )

        if active.any():
            act_idx = active.nonzero(as_tuple=True)[0]
            nb = int(act_idx.numel())
            blk_x = torch.full(
                (nb, actual), mask_token_id, dtype=torch.long, device=device
            )
            quota = get_num_transfer_tokens(
                torch.ones(nb, actual, dtype=torch.bool, device=device),
                cfg.steps_per_block,
            )
            cache_act = cache.index_select(act_idx) if cfg.use_cache else None
            prefix_act = prefix_no_cache[act_idx] if not cfg.use_cache else None

            i = 0
            max_iters = (
                cfg.steps_per_block
                if cfg.commit != "threshold"
                else actual + cfg.steps_per_block
            )
            while i < max_iters:
                masked = blk_x == mask_token_id
                if not masked.any():
                    break

                if cfg.use_cache:
                    logits, _ = model.forward_block(blk_x, cache_act)
                else:
                    full = torch.cat([prefix_act, blk_x], dim=1)
                    bounds = (
                        [Lp] + [Lp + (j + 1) * BL for j in range(blk)] + [full.shape[1]]
                    )
                    bias = block_causal_bias(
                        bounds, full.shape[1], device, torch.float32
                    )
                    logits = model(full, attn_bias=bias)[:, -actual:, :]
                nfe += 1
                logits = logits.clone()
                suppress_tokens_(logits, suppress_ids)

                x0 = sample_candidates(logits, cfg.temperature, cfg.sampling, generator)
                conf = confidence_scores(logits, x0, cfg.confidence, generator).float()

                cand = conf.masked_fill(~masked, float("-inf"))
                if cfg.commit == "transfer":
                    commit = select_topk_commits(
                        cand, quota[:, min(i, cfg.steps_per_block - 1)]
                    )
                else:
                    commit = select_threshold_commits(cand, cfg.threshold)
                blk_x = torch.where(commit, x0, blk_x)
                gpos = step_map[act_idx, blk * BL : blk * BL + actual]
                gpos[commit] = global_step
                step_map[act_idx, blk * BL : blk * BL + actual] = gpos
                global_step += 1
                i += 1

            block[act_idx] = blk_x

        # commit: recompute the block's KV from FINAL tokens (all samples)
        if cfg.use_cache:
            _, block_kvs = model.forward_block(block, cache)
            nfe += 1
            cache = cache.extend(block_kvs)
        else:
            prefix_no_cache = torch.cat([prefix_no_cache, block], dim=1)
        committed.append(block)

        if eos is not None:
            eos_hit |= (block == eos).any(dim=1)

    if not committed:
        empty = torch.empty(B, 0, dtype=torch.long, device=device)
        return BlockwiseOutput(
            sequences=[[] for _ in range(B)],
            canvas=torch.cat([prompt_ids, empty], dim=1),
            step_map=step_map,
            nfe=nfe,
        )

    gen = torch.cat(committed, dim=1)
    canvas = torch.cat([prompt_ids, gen], dim=1)
    sequences = [strip_after_eos(row.tolist(), eos) for row in gen]
    return BlockwiseOutput(
        sequences=sequences, canvas=canvas, step_map=step_map, nfe=nfe
    )
