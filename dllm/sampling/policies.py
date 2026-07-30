"""Commit-policy contracts shared by diffusion samplers.

A commit policy decides *which masked positions become tokens* after a model
forward. Candidate-token sampling and confidence estimation remain separate
mechanisms, so position policies can evolve without changing token semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Union, runtime_checkable

import torch

from .utils import (
    get_num_transfer_tokens,
    select_threshold_commits,
    select_topk_commits,
)


@dataclass(frozen=True)
class CommitState:
    """Inputs visible to a position-selection policy for one denoising step."""

    confidence: torch.Tensor  # (B, L), meaningful only at candidates
    candidates: torch.Tensor  # (B, L) bool, currently masked positions
    initial_mask: torch.Tensor  # (B, L) bool, positions masked at block start
    step: int
    steps: int  # planned steps; policies may ignore this value

    def __post_init__(self) -> None:
        if self.confidence.ndim != 2:
            raise ValueError("confidence must have shape (batch, length)")
        if self.candidates.shape != self.confidence.shape:
            raise ValueError("candidates must match confidence")
        if self.initial_mask.shape != self.confidence.shape:
            raise ValueError("initial_mask must match confidence")
        if self.candidates.dtype != torch.bool:
            raise TypeError("candidates must be boolean")
        if self.initial_mask.dtype != torch.bool:
            raise TypeError("initial_mask must be boolean")
        if self.candidates.device != self.confidence.device:
            raise ValueError("candidates and confidence must share a device")
        if self.initial_mask.device != self.confidence.device:
            raise ValueError("initial_mask and confidence must share a device")
        if bool((self.candidates & ~self.initial_mask).any()):
            raise ValueError("candidates must be a subset of initial_mask")
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if self.steps <= 0:
            raise ValueError("steps must be positive")


@dataclass(frozen=True)
class CommitDecision:
    """Positions selected by a policy and optional action log-probability.

    ``selection_logprob`` is one scalar per batch row for the complete
    position-selection action. Deterministic policies leave it unset.
    """

    commit: torch.Tensor  # (B, L) bool
    selection_logprob: Optional[torch.Tensor] = None  # (B,)


@runtime_checkable
class CommitPolicy(Protocol):
    """Framework-neutral position-selection policy."""

    def select(self, state: CommitState) -> CommitDecision:
        ...


@dataclass(frozen=True)
class QuotaCommitPolicy:
    """Commit a fixed per-step quota of the highest-confidence positions."""

    def select(self, state: CommitState) -> CommitDecision:
        quota = get_num_transfer_tokens(state.initial_mask, state.steps)
        current = quota[:, min(state.step, state.steps - 1)]
        return CommitDecision(
            select_topk_commits(
                state.confidence,
                current,
                candidates=state.candidates,
            )
        )


@dataclass(frozen=True)
class ThresholdCommitPolicy:
    """Commit every position above a threshold, with a progress floor."""

    threshold: float = 0.9

    def select(self, state: CommitState) -> CommitDecision:
        return CommitDecision(
            select_threshold_commits(
                state.confidence,
                self.threshold,
                candidates=state.candidates,
            )
        )


CommitSpec = Union[str, CommitPolicy]


def resolve_commit_policy(
    policy: CommitSpec,
    *,
    threshold: float = 0.9,
) -> CommitPolicy:
    """Resolve legacy string shorthands or validate a policy object."""

    if isinstance(policy, str):
        if policy in ("transfer", "quota"):
            return QuotaCommitPolicy()
        if policy == "threshold":
            return ThresholdCommitPolicy(threshold=threshold)
        raise ValueError(
            "commit must be 'transfer', 'quota', 'threshold', "
            "or a CommitPolicy"
        )
    if isinstance(policy, CommitPolicy):
        return policy
    raise TypeError("commit must be a string or CommitPolicy")


def apply_commit_policy(
    policy: CommitPolicy,
    state: CommitState,
) -> CommitDecision:
    """Run a policy and enforce sampler progress and action-shape invariants."""

    decision = policy.select(state)
    if not isinstance(decision, CommitDecision):
        raise TypeError("CommitPolicy.select must return CommitDecision")
    commit = decision.commit
    if commit.shape != state.candidates.shape:
        raise ValueError("policy commit mask must match candidates")
    if commit.dtype != torch.bool:
        raise TypeError("policy commit mask must be boolean")
    if commit.device != state.candidates.device:
        raise ValueError("policy commit mask must share the sampler device")
    if bool((commit & ~state.candidates).any()):
        raise ValueError("a commit policy cannot select non-candidate positions")

    active = state.candidates.any(dim=-1)
    stalled = active & ~commit.any(dim=-1)
    if bool(stalled.any()):
        raise ValueError(
            "a commit policy must select at least one position for every "
            "active batch row"
        )

    selection_logprob = decision.selection_logprob
    if selection_logprob is not None:
        expected = (state.confidence.shape[0],)
        if selection_logprob.shape != expected:
            raise ValueError(
                f"selection_logprob must have shape {expected}"
            )
        if selection_logprob.device != state.confidence.device:
            raise ValueError(
                "selection_logprob and confidence must share a device"
            )
    return decision


__all__ = [
    "CommitDecision",
    "CommitPolicy",
    "CommitSpec",
    "CommitState",
    "QuotaCommitPolicy",
    "ThresholdCommitPolicy",
    "apply_commit_policy",
    "resolve_commit_policy",
]
