"""Noise schedules for masked diffusion.

A schedule defines ``mask_prob(t)`` for ``t in [0, 1]`` and the matching
per-token loss weight ``weight(t) = p'(t)/p(t)`` (the NELBO weight).
``t`` is always sampled uniformly; to reshape the noise distribution, change
the schedule - the weight then stays consistent with the likelihood bound.
"""

from __future__ import annotations

import math
from typing import Dict, Type, Union

import torch

Number = Union[float, torch.Tensor]

_REGISTRY: Dict[str, Type["NoiseSchedule"]] = {}


class NoiseSchedule:
    """Base class. Subclasses implement ``mask_prob`` and ``mask_prob_derivative``."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _REGISTRY[cls.__name__] = cls
        name = cls.__name__.replace("Schedule", "").lower()
        _REGISTRY[name] = cls

    # -- to implement ------------------------------------------------------
    def mask_prob(self, t: Number) -> Number:
        raise NotImplementedError

    def mask_prob_derivative(self, t: Number) -> Number:
        raise NotImplementedError

    # -- derived -----------------------------------------------------------
    def weight(self, t: Number) -> Number:
        """Generic NELBO weight w(t) = p'(t)/p(t)."""
        return self.mask_prob_derivative(t) / self.mask_prob(t)

    def __call__(self, t: Number) -> Number:
        return self.mask_prob(t)


class LinearSchedule(NoiseSchedule):
    """Linear schedule p(t) = (1-eps)*t + eps with weight 1/p(t)."""

    def __init__(self, eps: float = 1e-3):
        if not 0.0 <= eps < 1.0:
            raise ValueError("eps must be in [0, 1)")
        self.eps = eps

    def mask_prob(self, t: Number) -> Number:
        return (1.0 - self.eps) * t + self.eps

    def mask_prob_derivative(self, t: Number) -> Number:
        if isinstance(t, torch.Tensor):
            return torch.full_like(t, 1.0 - self.eps)
        return 1.0 - self.eps

    def weight(self, t: Number) -> Number:
        return 1.0 / self.mask_prob(t)


class CosineSchedule(NoiseSchedule):
    """Cosine schedule p(t) = eps + (1-eps)*(1 - cos(pi*t/2)); generic weight."""

    def __init__(self, eps: float = 1e-3):
        self.eps = eps

    def mask_prob(self, t: Number) -> Number:
        if isinstance(t, torch.Tensor):
            return self.eps + (1.0 - self.eps) * (1.0 - torch.cos(math.pi * t / 2))
        return self.eps + (1.0 - self.eps) * (1.0 - math.cos(math.pi * t / 2))

    def mask_prob_derivative(self, t: Number) -> Number:
        if isinstance(t, torch.Tensor):
            return (1.0 - self.eps) * (math.pi / 2) * torch.sin(math.pi * t / 2)
        return (1.0 - self.eps) * (math.pi / 2) * math.sin(math.pi * t / 2)


class ClippedLinearSchedule(NoiseSchedule):
    """Masking rate uniform on ``[low, high]``: p(t) = low + (high-low)*t.

    Clipping keeps the rate away from the extremes, where the gradient is
    high-variance (almost nothing masked) or carries only marginal
    information (almost everything masked). ``low=eps, high=1`` is exactly
    ``LinearSchedule(eps)``. The per-token importance weight stays ``1/p``,
    which keeps each ``t``'s estimate unbiased for that ``t``; clipping
    changes which rates are trained, not the estimator.

    References: clipped masking rates, BD3-LM (arXiv 2503.09573, sec. 5.2);
    ``q_sample(min, max)`` in Dream's official trainer; the SFT mask-ratio
    bandwidth of LLaDA2.0 (arXiv 2512.15745, sec. 5.1).
    """

    def __init__(self, low: float = 1e-3, high: float = 1.0):
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("need 0 <= low < high <= 1")
        self.low = low
        self.high = high

    def mask_prob(self, t: Number) -> Number:
        return self.low + (self.high - self.low) * t

    def mask_prob_derivative(self, t: Number) -> Number:
        if isinstance(t, torch.Tensor):
            return torch.full_like(t, self.high - self.low)
        return self.high - self.low

    def weight(self, t: Number) -> Number:
        return 1.0 / self.mask_prob(t)


def get_schedule(schedule: Union[str, NoiseSchedule, None]) -> NoiseSchedule:
    """Resolve a schedule by instance, name ('linear', 'cosine', 'clippedlinear'),
    or None (linear)."""
    if schedule is None:
        return LinearSchedule()
    if isinstance(schedule, NoiseSchedule):
        return schedule
    if isinstance(schedule, str):
        key = schedule.lower()
        if key not in _REGISTRY:
            raise KeyError(f"Unknown schedule '{schedule}'. Known: {sorted(_REGISTRY)}")
        return _REGISTRY[key]()
    raise TypeError(f"Cannot resolve schedule from {type(schedule)}")
