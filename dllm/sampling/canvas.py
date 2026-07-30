"""Full-canvas sampler (reverse process on [prompt | gen_length x MASK]).

Blocks are decoded left-to-right within one canvas, so attention always sees
the future masked region. This is the reference sampler; use
:mod:`.blockwise` only for models trained on truncated canvases.

Commit strategies:
- "transfer":  per-step quota, commit top-confidence; committed tokens never
               return to MASK;
- "threshold": parallel decoding - commit everything with confidence >=
               threshold (min 1 per step).

Prefix cache (``prefix_cache=True``): at each block start one full-canvas
forward builds the cache, chopped at the block start; inner steps recompute
only a window (block .. block+further_horizon). Higher fidelity than
incremental caching because the cached prefix has seen the masked future.
Requires ``forward(..., return_kvs=True)`` + ``forward_block``
(DiffusionTransformer). Not combinable with CFG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from ..execution import CacheSemantics
from ..models.protocol import extract_logits
from ..models.transformer import DiffusionTransformer, KVCache
from .trace import (
    TokenDistribution,
    TopKPrediction,
    TrajectorySample,
    TrajectoryStep,
)
from .utils import (
    confidence_scores,
    get_num_transfer_tokens,
    sample_candidates,
    select_threshold_commits,
    select_topk_commits,
    split_steps,
    strip_after_eos,
    suppress_tokens_,
)

_COMMITS = ("transfer", "threshold")


@dataclass
class CanvasConfig:
    gen_length: int = 128
    block_length: Optional[int] = None  # None -> gen_length (no semi-AR)
    steps: int = 128
    temperature: float = 0.0
    sampling: str = "gumbel"  # "gumbel" | "multinomial"
    commit: str = "transfer"
    confidence: str = "prob"  # prob | margin | neg_entropy | random
    threshold: float = 0.9  # for commit="threshold"
    cfg_scale: float = 0.0
    allow_mask_prediction: bool = False  # False -> mask logit set to -inf
    suppress_token_ids: Sequence[int] = ()  # e.g. PAD
    eos_token_id: Optional[int] = None
    suppress_eos_logits: bool = False  # forbid EOS in predictions
    suppress_eos_confidence: bool = False  # commit EOS last
    prefix_cache: bool = False
    further_horizon: Optional[int] = (
        None  # window size beyond block end; None -> suffix
    )
    record_trace: bool = False
    trace_topk: int = 0  # 0 stores states/log-probs only; >0 adds compact dists


@dataclass
class CanvasOutput:
    canvas: torch.Tensor  # (B, Lp+gen) final canvas
    responses: List[List[int]]  # per sample, EOS-stripped
    step_map: torch.Tensor  # (B, gen) commit step per position (-1 unset)
    nfe: int = 0
    traces: Optional[List[TrajectorySample]] = None


def _model_logits(model, ids, attention_mask=None):
    out = (
        model(ids)
        if attention_mask is None
        else model(ids, attention_mask=attention_mask)
    )
    return extract_logits(out)


@torch.no_grad()
def generate_canvas(
    model,
    prompt_ids: torch.Tensor,
    mask_token_id: int,
    config: Optional[CanvasConfig] = None,
    attention_mask: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    **overrides,
) -> CanvasOutput:
    """Run the reverse process on a full canvas.

    Args:
        model: mask predictor - callable returning logits (or ``.logits``).
            With ``prefix_cache=True`` it must be a DiffusionTransformer.
        prompt_ids: (B, Lp) or (Lp,) long tensor.
        attention_mask: (B, Lp) for left-padded prompt batches; extended with
            ones over the generation region.
    """
    cfg = config or CanvasConfig()
    if overrides:
        cfg = dataclass_replace(cfg, **overrides)
    if cfg.commit not in _COMMITS:
        raise ValueError(f"commit must be one of {_COMMITS}")
    if cfg.prefix_cache and cfg.cfg_scale > 0:
        raise ValueError("prefix_cache is not combinable with CFG")
    if cfg.prefix_cache and not isinstance(model, DiffusionTransformer):
        raise TypeError("prefix_cache requires a dllm DiffusionTransformer")
    if cfg.trace_topk < 0:
        raise ValueError("trace_topk must be non-negative")
    if cfg.trace_topk and not cfg.record_trace:
        raise ValueError("trace_topk requires record_trace=True")
    if cfg.suppress_eos_logits or cfg.suppress_eos_confidence:
        if cfg.eos_token_id is None:
            raise ValueError("eos_token_id required for EOS suppression options")

    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    device = prompt_ids.device
    B, Lp = prompt_ids.shape
    G = cfg.gen_length
    block_len = cfg.block_length or G
    if G % block_len != 0:
        raise ValueError("gen_length must be a multiple of block_length")
    num_blocks = G // block_len
    steps_list = split_steps(cfg.steps, num_blocks)

    x = torch.full((B, Lp + G), mask_token_id, dtype=torch.long, device=device)
    x[:, :Lp] = prompt_ids
    prompt_index = x != mask_token_id

    full_attn = None
    if attention_mask is not None:
        full_attn = torch.cat(
            [
                attention_mask.to(device),
                torch.ones(B, G, dtype=attention_mask.dtype, device=device),
            ],
            dim=1,
        )

    suppress_ids = list(cfg.suppress_token_ids)
    if not cfg.allow_mask_prediction:
        suppress_ids.append(mask_token_id)
    if cfg.suppress_eos_logits:
        suppress_ids.append(cfg.eos_token_id)

    step_map = torch.full((B, G), -1, dtype=torch.long, device=device)
    traces = None
    if cfg.record_trace:
        traces = [
            TrajectorySample(prompt=prompt_ids[b].tolist(), final=[]) for b in range(B)
        ]

    nfe = 0
    global_step = 0

    def forward_logits(inp: torch.Tensor, attn) -> torch.Tensor:
        nonlocal nfe
        nfe += 1
        if cfg.cfg_scale > 0.0:
            un_x = inp.clone()
            un_x[prompt_index] = mask_token_id
            both = torch.cat([inp, un_x], dim=0)
            attn2 = None if attn is None else torch.cat([attn, attn], dim=0)
            logits = _model_logits(model, both, attn2)
            cond, uncond = torch.chunk(logits, 2, dim=0)
            return uncond + (cfg.cfg_scale + 1) * (cond - uncond)
        return _model_logits(model, inp, attn)

    for blk in range(num_blocks):
        s, e = Lp + blk * block_len, Lp + (blk + 1) * block_len
        cur_steps = steps_list[blk]
        if cur_steps == 0:
            cur_steps = 1
        block_mask0 = x[:, s:e] == mask_token_id
        quota = get_num_transfer_tokens(block_mask0, cur_steps)

        # -- prefix cache: one full forward per block, then window updates --
        cache: Optional[KVCache] = None
        window_end = (
            Lp + G
            if cfg.further_horizon is None
            else min(e + cfg.further_horizon, Lp + G)
        )

        i = 0
        max_iters = cur_steps if cfg.commit != "threshold" else block_len + cur_steps
        while i < max_iters:
            if (x[:, s:e] == mask_token_id).sum() == 0:
                break

            if cfg.prefix_cache:
                if cache is None:
                    logits_full, kvs = model(
                        x, attention_mask=full_attn, return_kvs=True
                    )
                    nfe += 1
                    cache = KVCache(
                        kvs=[(k[:, :, :s], v[:, :, :s]) for k, v in kvs],
                        key_mask=None if full_attn is None else full_attn[:, :s].bool(),
                        position_ids=torch.arange(s, device=x.device)
                        .unsqueeze(0)
                        .expand(x.shape[0], -1),
                        group_ids=torch.zeros(
                            x.shape[0], s, dtype=torch.long, device=x.device
                        ),
                        semantics=CacheSemantics.approximate_for(
                            "bidirectional_canvas",
                            "prefix states are frozen while the active window changes",
                        ),
                    )
                    logits = logits_full
                    lo = 0  # logits indexed over full canvas
                else:
                    win_logits, _ = model.forward_block(x[:, s:window_end], cache)
                    nfe += 1
                    logits = win_logits
                    lo = s  # logits indexed from s
            else:
                logits = forward_logits(x, full_attn)
                lo = 0

            # region of interest: current block, in logits coordinates
            blk_logits = logits[:, s - lo : e - lo, :].clone()
            suppress_tokens_(blk_logits, suppress_ids)

            x0 = sample_candidates(blk_logits, cfg.temperature, cfg.sampling, generator)
            conf = confidence_scores(blk_logits, x0, cfg.confidence, generator).float()
            if cfg.suppress_eos_confidence:
                conf = conf.masked_fill(x0 == cfg.eos_token_id, float("-inf"))

            blk_masked = x[:, s:e] == mask_token_id
            cand_conf = conf.masked_fill(~blk_masked, float("-inf"))
            if cfg.commit == "transfer":
                q = quota[:, min(i, cur_steps - 1)]
                commit = select_topk_commits(cand_conf, q)
            else:
                commit = select_threshold_commits(cand_conf, cfg.threshold)

            if cfg.record_trace:
                _record(
                    traces,
                    x,
                    mask_token_id,
                    commit,
                    blk_logits,
                    x0,
                    s,
                    global_step,
                    blk,
                    cfg.trace_topk,
                )

            xb = x[:, s:e]
            xb[commit] = x0[commit]
            x[:, s:e] = xb
            sm = step_map[:, s - Lp : e - Lp]
            sm[commit] = global_step
            step_map[:, s - Lp : e - Lp] = sm

            global_step += 1
            i += 1

    responses = [strip_after_eos(row.tolist(), cfg.eos_token_id) for row in x[:, Lp:]]
    if cfg.record_trace:
        for b in range(B):
            traces[b].final = x[b, Lp:].tolist()
    return CanvasOutput(
        canvas=x, responses=responses, step_map=step_map, nfe=nfe, traces=traces
    )


def _record(
    traces,
    x,
    mask_token_id,
    commit,
    blk_logits,
    x0,
    s,
    step,
    blk,
    trace_topk,
):
    logp = F.log_softmax(blk_logits.float(), dim=-1)
    probs = logp.exp()
    lp_x0 = torch.gather(logp, -1, x0.unsqueeze(-1)).squeeze(-1)
    B, L = x.shape
    masked_all = x == mask_token_id
    for b in range(len(traces)):
        committed_full = [False] * L
        cm = commit[b]
        clp = {}
        for j in range(cm.shape[0]):
            if cm[j]:
                committed_full[s + j] = True
                clp[s + j] = float(lp_x0[b, j])
        active = masked_all[b, s : s + cm.shape[0]]
        active_probs = probs[b, active]
        meta = {
            "masked_count": int(active.sum().item()),
            "commit_count": int(cm.sum().item()),
        }
        distributions = []
        if active_probs.numel():
            selected_probs = torch.gather(
                probs[b], -1, x0[b].unsqueeze(-1)
            ).squeeze(-1)
            meta["pmax_mean"] = float(selected_probs[active].mean().item())
            top2 = torch.topk(
                active_probs, k=min(2, active_probs.shape[-1]), dim=-1
            ).values
            margin = (
                top2[:, 0] - top2[:, 1]
                if top2.shape[-1] == 2
                else top2[:, 0]
            )
            meta["margin_mean"] = float(margin.mean().item())

            if trace_topk:
                k = min(trace_topk, active_probs.shape[-1])
                values, indices = torch.topk(active_probs, k=k, dim=-1)
                normalized = values / values.sum(dim=-1, keepdim=True).clamp(
                    min=torch.finfo(values.dtype).tiny
                )
                entropy = -(normalized * normalized.clamp_min(1e-12).log()).sum(
                    dim=-1
                )
                meta["entropy_topk_mean"] = float(entropy.mean().item())
                active_positions = torch.nonzero(active, as_tuple=True)[0]
                for row, relative in enumerate(active_positions.tolist()):
                    distributions.append(
                        TokenDistribution(
                            position=s + int(relative),
                            topk=[
                                TopKPrediction(
                                    token_id=int(token_id),
                                    probability=float(probability),
                                )
                                for token_id, probability in zip(
                                    indices[row].tolist(), values[row].tolist()
                                )
                            ],
                        )
                    )
        traces[b].steps.append(
            TrajectoryStep(
                step=step,
                block=blk,
                tokens=x[b].tolist(),
                masked=masked_all[b].tolist(),
                committed=committed_full,
                commit_logprob=clp,
                distributions=distributions,
                meta=meta,
            )
        )


def dataclass_replace(cfg: CanvasConfig, **kw) -> CanvasConfig:
    import dataclasses

    return dataclasses.replace(cfg, **kw)
