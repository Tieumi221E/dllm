"""Linear self-speculation with framework-neutral model execution.

The same model drafts a block under diffusion-style attention, then verifies
it causally. Only the longest draft prefix matching greedy causal predictions
is accepted, followed by one verifier token. Consequently, draft sampling
changes the execution schedule but not the emitted greedy sequence.

Model-specific attention modes, adapters, and cache types live behind
``SelfSpecBackend``. Reference backends live in :mod:`.backends`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import torch

from .policies import (
    CommitPolicy,
    CommitSpec,
    CommitState,
    apply_commit_policy,
    resolve_commit_policy,
)
from .utils import (
    confidence_scores,
    sample_candidates,
    suppress_tokens_,
)


@dataclass
class SelfSpecStep:
    """Logits and cache returned by a causal backend operation."""

    logits: torch.Tensor
    cache: Any


@runtime_checkable
class SelfSpecBackend(Protocol):
    """Execution boundary required by linear self-speculation.

    ``draft_logits`` must align logits with the token positions in
    ``block_ids`` and must not mutate the accepted prefix cache.
    ``causal_verify`` returns next-token logits: row ``i`` predicts the token
    after ``block_ids[i]``.
    """

    def causal_prefill(self, prompt_ids: torch.Tensor) -> SelfSpecStep:
        ...

    def draft_logits(
        self, block_ids: torch.Tensor, cache: Any
    ) -> torch.Tensor:
        ...

    def causal_verify(
        self, block_ids: torch.Tensor, cache: Any
    ) -> SelfSpecStep:
        ...

    def crop_cache(self, cache: Any, length: int) -> Any:
        ...


@dataclass
class SelfSpecConfig:
    max_new_tokens: int = 128
    block_length: int = 32
    draft_steps: int = 1
    temperature: float = 0.0
    sampling: str = "gumbel"
    commit: CommitSpec = "threshold"
    confidence: str = "prob"
    threshold: float = 0.0
    allow_mask_prediction: bool = False
    suppress_token_ids: Sequence[int] = ()
    terminal_token_ids: Sequence[int] = ()


@dataclass
class SelfSpecStats:
    drafted_tokens: int = 0
    accepted_draft_tokens: int = 0
    emitted_tokens: int = 0
    draft_forwards: int = 0
    verifier_forwards: int = 0
    iterations: int = 0

    @property
    def nfe(self) -> int:
        return self.draft_forwards + self.verifier_forwards

    @property
    def draft_acceptance(self) -> float:
        return self.accepted_draft_tokens / max(1, self.drafted_tokens)

    @property
    def mean_accepted_draft(self) -> float:
        return self.accepted_draft_tokens / max(1, self.iterations)

    def as_dict(self) -> dict:
        return {
            "drafted_tokens": self.drafted_tokens,
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "emitted_tokens": self.emitted_tokens,
            "draft_forwards": self.draft_forwards,
            "verifier_forwards": self.verifier_forwards,
            "iterations": self.iterations,
            "nfe": self.nfe,
            "draft_acceptance": self.draft_acceptance,
            "mean_accepted_draft": self.mean_accepted_draft,
        }


@dataclass
class SelfSpecOutput:
    token_ids: torch.Tensor  # (1, generated length), terminal token included
    stats: SelfSpecStats

    @property
    def sequences(self) -> list:
        return [row.tolist() for row in self.token_ids]

    @property
    def nfe(self) -> int:
        return self.stats.nfe


def _check_logits(
    logits: torch.Tensor,
    batch: int,
    length: int,
    operation: str,
) -> None:
    if logits.ndim != 3 or logits.shape[:2] != (batch, length):
        raise ValueError(
            f"{operation} logits must have shape (batch, length, vocab)"
        )


@torch.no_grad()
def generate_self_speculative(
    backend: SelfSpecBackend,
    prompt_ids: torch.Tensor,
    mask_token_id: int,
    config: Optional[SelfSpecConfig] = None,
    generator: Optional[torch.Generator] = None,
    **overrides,
) -> SelfSpecOutput:
    """Generate the backend's exact greedy causal sequence via block drafts.

    The current cache-cropping algorithm intentionally requires batch size 1;
    variable accepted lengths need a paged/ragged cache backend rather than
    padding disguised as equivalent semantics.
    """

    cfg = config or SelfSpecConfig()
    if overrides:
        cfg = dataclass_replace(cfg, **overrides)
    if cfg.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if cfg.block_length <= 0:
        raise ValueError("block_length must be positive")
    if cfg.draft_steps <= 0:
        raise ValueError("draft_steps must be positive")
    policy: CommitPolicy = resolve_commit_policy(
        cfg.commit, threshold=cfg.threshold
    )

    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("linear self-speculation requires batch size 1")
    if prompt_ids.shape[1] == 0:
        raise ValueError("prompt_ids cannot be empty")

    stats = SelfSpecStats()
    if cfg.max_new_tokens == 0:
        empty = prompt_ids.new_empty((1, 0))
        return SelfSpecOutput(token_ids=empty, stats=stats)

    suppress_ids = list(cfg.suppress_token_ids)
    if not cfg.allow_mask_prediction:
        suppress_ids.append(mask_token_id)
    terminals = {int(token) for token in cfg.terminal_token_ids}

    prefill = backend.causal_prefill(prompt_ids)
    _check_logits(
        prefill.logits,
        batch=1,
        length=prompt_ids.shape[1],
        operation="causal prefill",
    )
    stats.verifier_forwards += 1
    first_logits = prefill.logits[:, -1:, :].clone()
    suppress_tokens_(first_logits, suppress_ids)
    next_token = first_logits.argmax(dim=-1)
    generated = [next_token]
    total_generated = 1
    cache_length = int(prompt_ids.shape[1])
    cache = prefill.cache

    if int(next_token.item()) in terminals:
        stats.emitted_tokens = 1
        return SelfSpecOutput(token_ids=next_token, stats=stats)

    while total_generated < cfg.max_new_tokens:
        proposal_length = min(
            cfg.block_length,
            cfg.max_new_tokens - total_generated + 1,
        )
        block = torch.full(
            (1, proposal_length),
            mask_token_id,
            dtype=torch.long,
            device=prompt_ids.device,
        )
        block[:, 0] = next_token
        initial_mask = block == mask_token_id

        draft_step = 0
        while bool((block == mask_token_id).any()):
            draft_logits = backend.draft_logits(block, cache)
            _check_logits(
                draft_logits,
                batch=1,
                length=proposal_length,
                operation="draft",
            )
            stats.draft_forwards += 1
            draft_logits = draft_logits.clone()
            suppress_tokens_(draft_logits, suppress_ids)
            candidates = sample_candidates(
                draft_logits,
                cfg.temperature,
                cfg.sampling,
                generator,
            )
            confidence = confidence_scores(
                draft_logits,
                candidates,
                cfg.confidence,
                generator,
            ).float()
            masked = block == mask_token_id
            decision = apply_commit_policy(
                policy,
                CommitState(
                    confidence=confidence,
                    candidates=masked,
                    initial_mask=initial_mask,
                    step=draft_step,
                    steps=cfg.draft_steps,
                ),
            )
            block = torch.where(decision.commit, candidates, block)
            draft_step += 1

        stats.drafted_tokens += max(0, proposal_length - 1)
        stats.iterations += 1

        verified = backend.causal_verify(block, cache)
        _check_logits(
            verified.logits,
            batch=1,
            length=proposal_length,
            operation="causal verification",
        )
        stats.verifier_forwards += 1
        verify_logits = verified.logits.clone()
        suppress_tokens_(verify_logits, suppress_ids)
        causal_tokens = verify_logits.argmax(dim=-1)

        matched = 0
        for index in range(proposal_length - 1):
            if int(causal_tokens[0, index]) != int(block[0, index + 1]):
                break
            matched += 1
        accepted = min(
            matched + 1,
            cfg.max_new_tokens - total_generated,
        )
        accepted_tokens = causal_tokens[:, :accepted]

        emitted_accepted = accepted
        terminal_index = next(
            (
                index
                for index, token in enumerate(accepted_tokens[0].tolist())
                if token in terminals
            ),
            None,
        )
        if terminal_index is not None:
            emitted_accepted = terminal_index + 1
            accepted_tokens = accepted_tokens[:, :emitted_accepted]

        generated.append(accepted_tokens)
        total_generated += emitted_accepted
        stats.accepted_draft_tokens += min(matched, emitted_accepted)

        if terminal_index is not None:
            break

        cache_length += accepted
        cache = backend.crop_cache(verified.cache, cache_length)
        next_token = causal_tokens[:, accepted - 1 : accepted]

    token_ids = torch.cat(generated, dim=1)[:, : cfg.max_new_tokens]
    stats.emitted_tokens = int(token_ids.shape[1])
    return SelfSpecOutput(token_ids=token_ids, stats=stats)


def dataclass_replace(cfg: SelfSpecConfig, **kwargs) -> SelfSpecConfig:
    import dataclasses

    return dataclasses.replace(cfg, **kwargs)


__all__ = [
    "SelfSpecBackend",
    "SelfSpecConfig",
    "SelfSpecOutput",
    "SelfSpecStats",
    "SelfSpecStep",
    "generate_self_speculative",
]
