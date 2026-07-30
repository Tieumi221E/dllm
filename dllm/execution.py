"""Execution metadata shared by cache implementations and samplers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CacheSemantics:
    """Declare the dependency structure preserved by a cache.

    ``exact=False`` requires a short description of the approximation so a
    speed path cannot silently change model semantics.
    """

    topology: str
    exact: bool
    approximation: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.topology:
            raise ValueError("topology must be non-empty")
        if self.exact and self.approximation is not None:
            raise ValueError("an exact cache cannot declare an approximation")
        if not self.exact and not self.approximation:
            raise ValueError("an approximate cache must explain the approximation")

    @classmethod
    def exact_for(cls, topology: str) -> "CacheSemantics":
        return cls(topology=topology, exact=True)

    @classmethod
    def approximate_for(
        cls, topology: str, approximation: str
    ) -> "CacheSemantics":
        return cls(
            topology=topology, exact=False, approximation=approximation
        )


EXACT_ORDERED = CacheSemantics.exact_for("ordered")
# Compatibility constant for callers that name the common fixed-block case.
EXACT_BLOCK_CAUSAL = CacheSemantics.exact_for("block_causal")


__all__ = [
    "CacheSemantics",
    "EXACT_ORDERED",
    "EXACT_BLOCK_CAUSAL",
]
