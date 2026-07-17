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
class TrajectoryStep:
    step: int  # global step index
    block: int  # block index this step worked on
    tokens: List[int]  # canvas BEFORE the commit (full sequence)
    masked: List[bool]  # mask state BEFORE the commit
    committed: List[bool]  # which positions were committed this step
    commit_logprob: Dict[int, float] = field(default_factory=dict)
    # position -> log p(committed token | state); basis for ELBO-proxy selection
    meta: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "block": self.block,
            "tokens": self.tokens,
            "masked": self.masked,
            "committed": self.committed,
            "commit_logprob": {str(k): v for k, v in self.commit_logprob.items()},
            "meta": self.meta,
        }


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

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "final": self.final,
            "steps": [s.to_dict() for s in self.steps],
        }
