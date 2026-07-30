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

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, List, Union

import torch
import torch.nn.functional as F

from .models.protocol import extract_logits


@dataclass
class StepLogProb:
    step: int  # collapsed step value
    positions: torch.Tensor  # (n,) absolute positions (over prompt+gen)
    logp: torch.Tensor  # (n,) log p(token | reconstructed state)


@dataclass
class TrajectoryState:
    """One reconstructed denoising state and its token action."""

    step: int
    block: int
    input_ids: torch.Tensor  # (L_state,)
    positions: torch.Tensor  # (n,) absolute target positions
    target_ids: torch.Tensor  # (n,)

    def to(self, device: Union[str, torch.device]) -> "TrajectoryState":
        """Copy the state tensors to ``device`` for storage or scoring."""
        return TrajectoryState(
            step=self.step,
            block=self.block,
            input_ids=self.input_ids.to(device),
            positions=self.positions.to(device),
            target_ids=self.target_ids.to(device),
        )


@dataclass
class PPOObjective:
    """Differentiable PPO objective plus detached diagnostics."""

    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    clip_fraction: torch.Tensor
    ratio_mean: torch.Tensor


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


def trajectory_states(
    prompt_ids: torch.Tensor,  # (Lp,)
    gen_ids: torch.Tensor,  # (Lg,) final generated tokens (untruncated canvas region)
    step_map: torch.Tensor,  # (Lg,) commit step per position (-1 = never)
    mask_token_id: int,
    block_length: int,
    canvas: str = "incremental",
    collapse: Union[str, int] = "block",
) -> List[TrajectoryState]:
    """Reconstruct the model inputs and token actions for a rollout.

    This function is model-free and does not retain an autograd graph.  The
    returned states can therefore be reused for old-policy scoring, current
    policy scoring, trajectory distillation, or diagnostics.
    """
    if canvas not in ("incremental", "full"):
        raise ValueError("canvas must be 'incremental' or 'full'")
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    if prompt_ids.ndim != 1 or gen_ids.ndim != 1 or step_map.ndim != 1:
        raise ValueError("prompt_ids, gen_ids, and step_map must be 1-D")
    if gen_ids.shape != step_map.shape:
        raise ValueError("gen_ids and step_map must have the same length")
    device = prompt_ids.device
    Lp, Lg = int(prompt_ids.numel()), int(gen_ids.numel())
    full = torch.cat([prompt_ids, gen_ids.to(device)])
    pos_idx = torch.arange(Lg, device=device)
    block_ids = pos_idx // block_length
    step_map = step_map.to(device)

    csteps = _collapse_step_map(step_map, block_ids, collapse)
    valid = step_map >= 0
    if not valid.any():
        return []

    # ordered unique collapsed steps (blocks are decoded left-to-right, and
    # within a block steps increase, so sort by (block, step))
    keys = torch.unique(torch.stack([block_ids[valid], csteps[valid]], dim=1), dim=0)
    order = torch.argsort(keys[:, 0] * 10**9 + keys[:, 1])
    keys = keys[order]

    states = []
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
        positions = Lp + torch.nonzero(tgt, as_tuple=True)[0]
        states.append(
            TrajectoryState(
                step=int(s_val),
                block=int(b_val),
                input_ids=state,
                positions=positions,
                target_ids=full[positions],
            )
        )
    return states


def _model_logits(model_fn, input_ids: torch.Tensor) -> torch.Tensor:
    return extract_logits(model_fn(input_ids))


def score_trajectory_states(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    states: List[TrajectoryState],
    *,
    chunk: int = 8,
    with_grad: bool = False,
) -> List[StepLogProb]:
    """Score pre-built trajectory states, optionally retaining gradients."""
    if chunk <= 0:
        raise ValueError("chunk must be positive")

    out: List[StepLogProb] = []
    i = 0
    grad_context = nullcontext() if with_grad else torch.no_grad()
    with grad_context:
        while i < len(states):
            # Batch only a contiguous run of equal-length states. Padding would
            # alter bidirectional attention unless every adapter handled it.
            run = []
            for j in range(i, len(states)):
                if (
                    states[j].input_ids.numel() == states[i].input_ids.numel()
                    and len(run) < chunk
                ):
                    run.append(j)
                else:
                    break
            batch = torch.stack([states[j].input_ids for j in run])
            logits = _model_logits(model_fn, batch)
            for k, j in enumerate(run):
                state = states[j]
                rows = logits[k, state.positions, :].float()
                logp = F.log_softmax(rows, dim=-1)
                token_lp = logp.gather(
                    1, state.target_ids.unsqueeze(1)
                ).squeeze(1)
                out.append(
                    StepLogProb(
                        step=state.step,
                        positions=state.positions,
                        logp=token_lp,
                    )
                )
            i = run[-1] + 1
    return out


def trajectory_logprobs(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    prompt_ids: torch.Tensor,
    gen_ids: torch.Tensor,
    step_map: torch.Tensor,
    mask_token_id: int,
    block_length: int,
    canvas: str = "incremental",
    collapse: Union[str, int] = "block",
    chunk: int = 8,
    with_grad: bool = False,
) -> List[StepLogProb]:
    """Compute token log-probabilities on reconstructed rollout states.

    Every generated position with ``step_map >= 0`` appears exactly once.
    ``with_grad=False`` preserves the inexpensive scoring behavior; set it to
    ``True`` for policy optimization.
    """
    states = trajectory_states(
        prompt_ids,
        gen_ids,
        step_map,
        mask_token_id,
        block_length,
        canvas=canvas,
        collapse=collapse,
    )
    return score_trajectory_states(
        model_fn, states, chunk=chunk, with_grad=with_grad
    )


def ppo_clip_objective(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantage: Union[float, torch.Tensor],
    *,
    clip_eps: float = 0.2,
    beta: float = 0.01,
    kl_estimator: str = "k3",
) -> PPOObjective:
    """Token-level clipped PPO objective used by trajectory RL.

    ``kl_estimator="k3"`` uses ``exp(-Δ)-1+Δ`` for
    ``Δ = logp - old_logp``.  ``"k1"`` retains the sampled ``Δ`` estimator,
    and ``"none"`` disables the KL term.
    """
    if logp.shape != old_logp.shape:
        raise ValueError("logp and old_logp must have the same shape")
    if not 0.0 <= clip_eps < 1.0:
        raise ValueError("clip_eps must be in [0, 1)")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    if kl_estimator not in ("k3", "k1", "none"):
        raise ValueError("kl_estimator must be 'k3', 'k1', or 'none'")
    if logp.numel() == 0:
        zero = logp.sum()
        return PPOObjective(zero, zero, zero, zero.detach(), zero.detach())

    old_logp = old_logp.to(device=logp.device, dtype=logp.dtype)
    advantage = torch.as_tensor(
        advantage, device=logp.device, dtype=logp.dtype
    )
    delta = logp - old_logp
    ratio = delta.exp()
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    policy_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()

    if kl_estimator == "k3":
        kl = (-delta).exp() - 1.0 + delta
    elif kl_estimator == "k1":
        kl = delta
    else:
        kl = torch.zeros_like(delta)
    kl_loss = beta * kl.mean()
    return PPOObjective(
        loss=policy_loss + kl_loss,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        clip_fraction=(
            (ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)
        )
        .float()
        .mean()
        .detach(),
        ratio_mean=ratio.mean().detach(),
    )
