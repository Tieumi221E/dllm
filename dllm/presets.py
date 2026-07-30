"""Discoverable, composable configuration presets.

Presets are separated into model shapes, algorithm recipes, and external
integrations so new combinations do not require a monolithic entry for every
experiment. The three pre-1.3.2 names remain complete compatibility presets.

Usage::

    from dllm.presets import compose_presets

    config = compose_presets(
        "model/ref-small-gqa-rope",
        "recipe/blockwise-exact",
    )
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Mapping, Optional


@dataclass(frozen=True)
class PresetInfo:
    """Metadata for discovery without materializing a mutable config."""

    name: str
    category: str
    description: str
    requires: FrozenSet[str] = frozenset()
    reference: Optional[str] = None


_PRESETS: Dict[str, dict] = {}
_INFO: Dict[str, PresetInfo] = {}


def _register(
    name: str,
    category: str,
    description: str,
    config: Mapping[str, Any],
    *,
    requires=(),
    reference: Optional[str] = None,
) -> None:
    if name in _PRESETS:
        raise ValueError(f"duplicate preset: {name}")
    _PRESETS[name] = deepcopy(dict(config))
    _INFO[name] = PresetInfo(
        name=name,
        category=category,
        description=description,
        requires=frozenset(requires),
        reference=reference,
    )


# ------------------------ model architecture shapes ----------------------- #

_SMALL_MHA_MODEL = dict(
    vocab_size=50260,
    max_position_embeddings=128,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    num_kv_heads=12,
    intermediate_size=3072,
    position_embedding="learned",
    attn_bias=True,
    ff_bias=True,
)
_SMALL_GQA_MODEL = dict(
    vocab_size=4096,
    max_position_embeddings=2048,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    num_kv_heads=2,
    intermediate_size=3072,
    position_embedding="learned",
    attn_bias=False,
)
_SMALL_GQA_ROPE_MODEL = {
    **_SMALL_GQA_MODEL,
    "position_embedding": "rope",
}

_register(
    "model/ref-small-mha",
    "model",
    "Reference MHA Transformer with learned positions and legacy biases.",
    {"model": _SMALL_MHA_MODEL},
    requires={"same_position"},
)
_register(
    "model/ref-small-gqa",
    "model",
    "Reference GQA Transformer with learned positions.",
    {"model": _SMALL_GQA_MODEL},
    requires={"same_position"},
)
_register(
    "model/ref-small-gqa-rope",
    "model",
    "Reference GQA Transformer with RoPE for new training runs.",
    {"model": _SMALL_GQA_ROPE_MODEL},
    requires={"same_position", "explicit_position_ids"},
)


# ----------------------------- algorithm recipes -------------------------- #

_register(
    "recipe/mdlm-pretrain",
    "recipe",
    "Importance-weighted masked-diffusion pretraining objective.",
    {"loss": {"norm": "tokens"}},
    requires={"same_position", "bidirectional"},
    reference="https://arxiv.org/abs/2406.07524",
)
_register(
    "recipe/sft-full",
    "recipe",
    "Answer-normalized SFT on a full bidirectional canvas.",
    {"loss": {"norm": "answer"}},
    requires={"same_position", "bidirectional"},
)
_register(
    "recipe/full-transfer",
    "recipe",
    "Full-canvas fixed-quota decoding with deterministic candidates.",
    {
        "canvas": {
            "gen_length": 128,
            "block_length": 128,
            "steps": 128,
            "temperature": 0.0,
            "sampling": "gumbel",
            "commit": "transfer",
            "confidence": "prob",
        }
    },
    requires={"same_position", "bidirectional"},
    reference="https://arxiv.org/abs/2502.09992",
)
_register(
    "recipe/full-threshold",
    "recipe",
    "Full-canvas confidence-threshold decoding with dynamic parallelism.",
    {
        "canvas": {
            "gen_length": 128,
            "block_length": 128,
            "steps": 128,
            "temperature": 0.0,
            "sampling": "gumbel",
            "commit": "threshold",
            "confidence": "prob",
            "threshold": 0.9,
        }
    },
    requires={"same_position", "bidirectional"},
)
_register(
    "recipe/semiar-full",
    "recipe",
    "Left-to-right blocks denoised inside a persistent full canvas.",
    {
        "canvas": {
            "gen_length": 256,
            "block_length": 32,
            "steps": 128,
            "temperature": 0.0,
            "sampling": "gumbel",
            "commit": "transfer",
            "confidence": "prob",
        }
    },
    requires={"same_position", "bidirectional"},
)
_register(
    "recipe/blockwise-exact",
    "recipe",
    "Incremental block decoding paired with truncated-canvas SFT.",
    {
        "blockwise": {
            "block_length": 32,
            "steps_per_block": 32,
            "gen_length": 384,
            "temperature": 1.0,
            "sampling": "gumbel",
            "commit": "transfer",
            "confidence": "prob",
        },
        "block_sft": {"canvas": "truncated"},
        "loss": {"norm": "answer"},
    },
    requires={"same_position", "block_causal", "exact_ordered"},
    reference="https://arxiv.org/abs/2503.09573",
)
_register(
    "recipe/self-spec-linear",
    "recipe",
    "Linear diffusion drafting with causal greedy verification.",
    {
        "self_spec": {
            "max_new_tokens": 256,
            "block_length": 32,
            "draft_steps": 1,
            "temperature": 0.0,
            "sampling": "gumbel",
            "commit": "threshold",
            "confidence": "prob",
            "threshold": 0.0,
        }
    },
    requires={"next_token", "causal", "block_causal", "cache_crop"},
)
_register(
    "recipe/trajectory-rollout",
    "recipe",
    "Stochastic full-canvas rollout with compact predictive traces.",
    {
        "canvas": {
            "gen_length": 256,
            "block_length": 256,
            "steps": 128,
            "temperature": 0.6,
            "sampling": "gumbel",
            "commit": "transfer",
            "confidence": "prob",
            "record_trace": True,
            "trace_topk": 8,
        }
    },
    requires={"same_position", "bidirectional"},
)


# --------------------------- external integrations ------------------------ #

_register(
    "integration/llada-8b",
    "integration",
    "Official LLaDA-8B-Instruct checkpoint metadata.",
    {
        "hf_model": "GSAI-ML/LLaDA-8B-Instruct",
        "mask_token_id": 126336,
    },
    requires={"same_position", "bidirectional", "transformers"},
    reference="https://github.com/ML-GSAI/LLaDA",
)


# ------------------------- compatibility presets -------------------------- #

_register(
    "small-mha",
    "legacy",
    "Complete pre-1.3.2 small-MHA configuration.",
    {
        "model": _SMALL_MHA_MODEL,
        "canvas": {
            "commit": "transfer",
            "sampling": "gumbel",
            "temperature": 1.0,
            "confidence": "prob",
        },
        "loss": {"norm": "tokens"},
    },
)
_register(
    "small-gqa",
    "legacy",
    "Complete pre-1.3.2 small-GQA blockwise configuration.",
    {
        "model": _SMALL_GQA_MODEL,
        "blockwise": {
            "block_length": 32,
            "steps_per_block": 32,
            "gen_length": 384,
            "temperature": 1.0,
            "sampling": "gumbel",
            "commit": "transfer",
            "confidence": "prob",
        },
        "block_sft": {"canvas": "truncated"},
        "loss": {"norm": "answer"},
    },
)
_register(
    "llada-8b",
    "legacy",
    "Complete pre-1.3.2 LLaDA-8B rollout configuration.",
    {
        "hf_model": "GSAI-ML/LLaDA-8B-Instruct",
        "mask_token_id": 126336,
        "canvas": {
            "gen_length": 256,
            "block_length": 256,
            "steps": 128,
            "temperature": 0.0,
            "sampling": "gumbel",
            "commit": "transfer",
            "confidence": "prob",
        },
        "canvas_trajectory": {
            "gen_length": 256,
            "block_length": 256,
            "steps": 128,
            "temperature": 0.6,
            "sampling": "gumbel",
            "commit": "transfer",
            "confidence": "prob",
            "record_trace": True,
        },
        "loss": {"norm": "answer"},
    },
)


def get_preset(name: str) -> dict:
    """Return an isolated mutable copy of a preset configuration."""
    if name not in _PRESETS:
        raise KeyError(f"unknown preset '{name}'; available: {sorted(_PRESETS)}")
    return deepcopy(_PRESETS[name])


def get_preset_info(name: str) -> PresetInfo:
    """Return immutable discovery metadata for one preset."""
    if name not in _INFO:
        raise KeyError(f"unknown preset '{name}'; available: {sorted(_INFO)}")
    return _INFO[name]


def list_presets(
    category: Optional[str] = None,
    *,
    include_legacy: bool = True,
):
    """List preset names, optionally filtered by category."""
    names = []
    for name, info in _INFO.items():
        if not include_legacy and info.category == "legacy":
            continue
        if category is not None and info.category != category:
            continue
        names.append(name)
    return sorted(names)


def _merge(
    target: Dict[str, Any],
    incoming: Mapping[str, Any],
    *,
    path: str,
    replace: bool,
) -> None:
    for key, value in incoming.items():
        location = f"{path}.{key}" if path else key
        if key not in target:
            target[key] = deepcopy(value)
            continue
        current = target[key]
        if isinstance(current, dict) and isinstance(value, Mapping):
            _merge(current, value, path=location, replace=replace)
        elif replace:
            target[key] = deepcopy(value)
        elif current != value:
            raise ValueError(f"conflicting preset value at {location}")


def compose_presets(
    *names: str,
    overrides: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Merge orthogonal presets and reject silent configuration conflicts.

    ``overrides`` is the only source allowed to replace an existing value.
    """
    if not names:
        raise ValueError("provide at least one preset name")
    result: Dict[str, Any] = {}
    for name in names:
        _merge(result, get_preset(name), path="", replace=False)
    if overrides is not None:
        _merge(result, overrides, path="", replace=True)
    return result


__all__ = [
    "PresetInfo",
    "compose_presets",
    "get_preset",
    "get_preset_info",
    "list_presets",
]
