"""Trajectory-decomposed log-probabilities for RL.

A subtle correctness requirement that is easy to violate: the reconstructed
per-step states MUST match the canvas regime of the sampler that produced the
rollout (a bidirectional model's logits change with whether future [MASK]
tokens are present) -

- rollouts from :func:`~dllm.sampling.generate_blockwise`
  (incremental canvas)  -> ``canvas="incremental"``: the state at a step is
  truncated at the end of the current block; future blocks do not exist;
- rollouts from :func:`~dllm.sampling.generate_canvas`
  (full canvas)         -> ``canvas="full"``: future positions are present as
  [MASK] tokens.

Step collapsing (``collapse``) merges consecutive commit steps to reduce the
number of forward passes. ``collapse="block"`` gives one state per block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Union

import torch
import torch.nn.functional as F


@dataclass
class StepLogProb:
    step: int  # collapsed step value
    positions: torch.Tensor  # (n,) absolute positions (over prompt+gen)
    logp: torch.Tensor  # (n,) log p(token | reconstructed state)


def _collapse_step_map(
    step_map: torch.Tensor, block_ids: torch.Tensor, collapse: Union[str, int]
) -> torch.Tensor:
    """Map raw commit steps to collapsed step values (monotone per sequence)."""
    if collapse == "none":
        return step_map.clone()
    if collapse == "block":
        return block_ids.clone()
    if isinstance(collapse, int) and collapse > 0:
        # merge every `collapse` consecutive distinct steps, within blocks
        out = torch.full_like(step_map, -1)
        for b in torch.unique(block_ids):
            sel = block_ids == b
            steps = torch.unique(step_map[sel & (step_map >= 0)], sorted=True)
            for rank, sv in enumerate(steps.tolist()):
                out[sel & (step_map == sv)] = int(b) * 10**6 + rank // collapse
        return out
    raise ValueError("collapse must be 'none', 'block', or a positive int")


@torch.no_grad()
def trajectory_logprobs(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    prompt_ids: torch.Tensor,  # (Lp,)
    gen_ids: torch.Tensor,  # (Lg,) final generated tokens (untruncated canvas region)
    step_map: torch.Tensor,  # (Lg,) commit step per position (-1 = never)
    mask_token_id: int,
    block_length: int,
    canvas: str = "incremental",
    collapse: Union[str, int] = "block",
    chunk: int = 8,
) -> List[StepLogProb]:
    """Reconstruct per-step states and compute log p(committed tokens | state).

    Every generated position with ``step_map >= 0`` appears in exactly one
    returned entry. Use the same ``canvas`` as the rollout sampler.

    ``model_fn``: (N, L) ids -> (N, L, V) logits.
    """
    if canvas not in ("incremental", "full"):
        raise ValueError("canvas must be 'incremental' or 'full'")
    device = prompt_ids.device
    Lp, Lg = int(prompt_ids.numel()), int(gen_ids.numel())
    full = torch.cat([prompt_ids, gen_ids])
    pos_idx = torch.arange(Lg, device=device)
    block_ids = pos_idx // block_length

    csteps = _collapse_step_map(step_map.to(device), block_ids, collapse)
    valid = step_map.to(device) >= 0

    # ordered unique collapsed steps (blocks are decoded left-to-right, and
    # within a block steps increase, so sort by (block, step))
    keys = torch.unique(torch.stack([block_ids[valid], csteps[valid]], dim=1), dim=0)
    order = torch.argsort(keys[:, 0] * 10**9 + keys[:, 1])
    keys = keys[order]

    states, targets_list, meta = [], [], []
    for b_val, s_val in keys.tolist():
        in_block = block_ids == b_val
        tgt = valid & in_block & (csteps == s_val)
        if not tgt.any():
            continue
        # masked at this state: current block's not-yet-committed positions
        not_committed = in_block & (~valid | (csteps >= s_val))
        if canvas == "incremental":
            end = Lp + min((b_val + 1) * block_length, Lg)
            state = full[:end].clone()
            m = not_committed[: end - Lp]
            state[Lp:end][m] = mask_token_id
        else:
            state = full.clone()
            future = block_ids > b_val
            state[Lp:][not_committed | future] = mask_token_id
        states.append(state)
        targets_list.append(tgt)
        meta.append((int(b_val), int(s_val)))

    out: List[StepLogProb] = []
    i = 0
    while i < len(states):
        # batch a contiguous run of equal-length states (never pad - padding
        # would pollute bidirectional attention)
        run = []
        for j in range(i, len(states)):
            if int(states[j].numel()) == int(states[i].numel()) and len(run) < chunk:
                run.append(j)
            else:
                break
        batch = torch.stack([states[j] for j in run])
        logits = model_fn(batch)
        for k, j in enumerate(run):
            tgt = targets_list[j]
            positions = Lp + torch.nonzero(tgt, as_tuple=True)[0]
            rows = logits[k, positions, :].float()
            logp = F.log_softmax(rows, dim=-1)
            token_lp = logp.gather(1, full[positions].unsqueeze(1)).squeeze(1)
            out.append(StepLogProb(step=meta[j][1], positions=positions, logp=token_lp))
        i = run[-1] + 1
    return out
