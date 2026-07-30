"""Trajectory recording structures.

One structure serves both consumers:
- constrained-order / trajectory training: needs, per
  step, the canvas state, which positions were committed, and the commit-time
  log-probability (ELBO-proxy selection);
- trajectory-decomposed RL: needs the per-position commit step (``step_map``).

Token *ids* are recorded (not strings) - decode at the boundary if needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TopKPrediction:
    """One token in a recorded predictive distribution."""

    token_id: int
    probability: float

    def to_dict(self) -> dict:
        return {"token_id": self.token_id, "probability": self.probability}

    @classmethod
    def from_dict(cls, value: dict) -> "TopKPrediction":
        return cls(
            token_id=int(value["token_id"]),
            probability=float(value["probability"]),
        )


@dataclass
class TokenDistribution:
    """A compact top-k distribution at one absolute canvas position."""

    position: int
    topk: List[TopKPrediction] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "topk": [entry.to_dict() for entry in self.topk],
        }

    @classmethod
    def from_dict(cls, value: dict) -> "TokenDistribution":
        return cls(
            position=int(value["position"]),
            topk=[
                TopKPrediction.from_dict(entry) for entry in value.get("topk", [])
            ],
        )


@dataclass
class TrajectoryStep:
    step: int  # global step index
    block: int  # block index this step worked on
    tokens: List[int]  # canvas BEFORE the commit (full sequence)
    masked: List[bool]  # mask state BEFORE the commit
    committed: List[bool]  # which positions were committed this step
    commit_logprob: Dict[int, float] = field(default_factory=dict)
    # position -> log p(committed token | state); basis for ELBO-proxy selection
    distributions: List[TokenDistribution] = field(default_factory=list)
    # Optional log-probability of the position-selection action. Deterministic
    # top-k commit policies leave this unset; stochastic policies can record it.
    selection_logprob: Optional[float] = None
    meta: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "block": self.block,
            "tokens": self.tokens,
            "masked": self.masked,
            "committed": self.committed,
            "commit_logprob": {str(k): v for k, v in self.commit_logprob.items()},
            "distributions": [dist.to_dict() for dist in self.distributions],
            "selection_logprob": self.selection_logprob,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "TrajectoryStep":
        return cls(
            step=int(value["step"]),
            block=int(value["block"]),
            tokens=[int(token) for token in value.get("tokens", [])],
            masked=[bool(item) for item in value.get("masked", [])],
            committed=[bool(item) for item in value.get("committed", [])],
            commit_logprob={
                int(position): float(logp)
                for position, logp in value.get("commit_logprob", {}).items()
            },
            distributions=[
                TokenDistribution.from_dict(dist)
                for dist in value.get("distributions", [])
            ],
            selection_logprob=(
                None
                if value.get("selection_logprob") is None
                else float(value["selection_logprob"])
            ),
            meta=value.get("meta"),
        )


@dataclass
class TrajectorySample:
    prompt: List[int]
    final: List[int]  # final canvas (response region, incl. EOS pads)
    steps: List[TrajectoryStep] = field(default_factory=list)

    @property
    def step_map(self) -> List[int]:
        """Per response-position commit step (-1 if never committed)."""
        n = len(self.final)
        out = [-1] * n
        offset = len(self.prompt)
        for st in self.steps:
            for i, c in enumerate(st.committed):
                if c and i >= offset:
                    out[i - offset] = st.step
        return out

    def content_logprob_mean(self, eos_token_id: Optional[int] = None) -> float:
        """ELBO proxy: mean commit-time logprob over response tokens before
        the first EOS - a practical instantiation of the ELBO-based
        trajectory selection for constrained-order training."""
        offset = len(self.prompt)
        end = len(self.final)
        if eos_token_id is not None and eos_token_id in self.final:
            end = self.final.index(eos_token_id)
        lps = []
        for st in self.steps:
            for pos, lp in st.commit_logprob.items():
                rel = int(pos) - offset
                if 0 <= rel < end:
                    lps.append(lp)
        return sum(lps) / len(lps) if lps else float("-inf")

    def summary(self, eos_token_id: Optional[int] = None) -> dict:
        """Return model-agnostic rollout statistics.

        Confidence statistics are included when the sampler recorded them in
        ``step.meta``.  Token and selection log-probabilities remain separate:
        deterministic commit policies generally have no selection log-prob.
        """
        response_length = len(self.final)
        offset = len(self.prompt)
        cumulative = 0
        progress = []
        commit_counts = []
        token_logprobs = []
        weighted_logprob = 0.0
        weighted_step_count = 0.0
        selection_logprobs = []
        pmax = []
        margins = []
        entropies = []

        end = response_length
        if eos_token_id is not None and eos_token_id in self.final:
            end = self.final.index(eos_token_id)
        content_logprobs = []

        for step in self.steps:
            count = sum(
                bool(committed)
                for position, committed in enumerate(step.committed)
                if position >= offset
            )
            commit_counts.append(count)
            cumulative += count
            progress.append(min(cumulative, response_length))
            for position, logp in step.commit_logprob.items():
                token_logprobs.append(float(logp))
                weighted_logprob += float(logp) * float(step.step)
                weighted_step_count += float(step.step)
                relative = int(position) - offset
                if 0 <= relative < end:
                    content_logprobs.append(float(logp))
            if step.selection_logprob is not None:
                selection_logprobs.append(float(step.selection_logprob))
            if step.meta:
                if "pmax_mean" in step.meta:
                    pmax.append(float(step.meta["pmax_mean"]))
                if "margin_mean" in step.meta:
                    margins.append(float(step.meta["margin_mean"]))
                if "entropy_topk_mean" in step.meta:
                    entropies.append(float(step.meta["entropy_topk_mean"]))

        n_steps = len(self.steps)
        denom = max(response_length * n_steps, 1)
        result = {
            "steps": n_steps,
            "auc_progress": float(sum(progress) / denom),
            "idle_ratio": float(
                sum(count == 0 for count in commit_counts) / max(n_steps, 1)
            ),
            "logprob_sum": float(sum(token_logprobs)),
            "logprob_mean": float(
                sum(token_logprobs) / max(len(token_logprobs), 1)
            ),
            "commit_tokens": len(token_logprobs),
            "logprob_weighted": float(weighted_logprob),
            "weighted_step_count": float(weighted_step_count),
            "content_logprob_mean": float(
                sum(content_logprobs) / max(len(content_logprobs), 1)
            ),
            "content_commit_tokens": len(content_logprobs),
        }
        if selection_logprobs:
            result["selection_logprob_sum"] = float(sum(selection_logprobs))
        if pmax:
            result["pmax_mean"] = float(sum(pmax) / len(pmax))
            result["pmax_slope"] = float(pmax[-1] - pmax[0])
        if margins:
            result["margin_mean"] = float(sum(margins) / len(margins))
        if entropies:
            result["entropy_topk_mean"] = float(sum(entropies) / len(entropies))
        return result

    def to_dict(self) -> dict:
        return {
            "schema": "dllm.trajectory.v1",
            "prompt": self.prompt,
            "final": self.final,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, value: dict) -> "TrajectorySample":
        return cls(
            prompt=[int(token) for token in value.get("prompt", [])],
            final=[int(token) for token in value.get("final", [])],
            steps=[
                TrajectoryStep.from_dict(step) for step in value.get("steps", [])
            ],
        )
