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


# Checkpoint facts below were read from each repository's config.json,
# generation_config.json, tokenizer_config.json and modeling / generation
# code at the recorded revision (2026-09-27). ``prediction_field`` follows
# dllm.prediction; ``topology`` is the attention the checkpoint was trained
# with. Where the config and tokenizer disagree, both are recorded.


def _checkpoint(
    name: str,
    description: str,
    *,
    hf_model: str,
    revision: str,
    license: str,
    prediction_field: str,
    topology: str,
    tokens: Mapping[str, Any],
    reference: str,
    block_length: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    config: Dict[str, Any] = {
        "hf_model": hf_model,
        "revision": revision,
        "license": license,
        "prediction_field": prediction_field,
        "topology": topology,
        **tokens,
    }
    if block_length is not None:
        config["block_length"] = block_length
    config.update(extra or {})
    _register(
        name,
        "integration",
        description,
        config,
        requires={prediction_field, topology, "transformers"},
        reference=reference,
    )


_DREAM_REF = "https://github.com/DreamLM/Dream"
_DREAM_DECODING = {"commit_policy": "timestep_quota", "timestep_eps": 1e-3}

_checkpoint(
    "integration/dream-v0-instruct-7b",
    "Dream-v0-Instruct-7B: full-canvas masked diffusion from Qwen2.5-7B.",
    hf_model="Dream-org/Dream-v0-Instruct-7B",
    revision="05334cb9fa",
    license="apache-2.0",
    prediction_field="shifted",
    topology="bidirectional",
    tokens={"mask_token_id": 151666, "eos_token_id": 151643,
            "pad_token_id": 151643, "turn_end_token_id": 151645},
    reference=_DREAM_REF,
    extra=_DREAM_DECODING,
)
_checkpoint(
    "integration/dream-coder-v0-instruct-7b",
    "Dream-Coder-v0-Instruct-7B: Dream architecture from Qwen2.5-Coder-7B.",
    hf_model="Dream-org/Dream-Coder-v0-Instruct-7B",
    revision="5d9e88c723",
    license="apache-2.0",
    prediction_field="shifted",
    topology="bidirectional",
    tokens={"mask_token_id": 151666, "eos_token_id": 151643,
            "pad_token_id": 151643, "turn_end_token_id": 151645,
            "bos_token_id": 151665},
    reference=_DREAM_REF,
    extra=_DREAM_DECODING,
)
_checkpoint(
    "integration/diffucoder-7b-instruct",
    "DiffuCoder-7B-Instruct: Dream architecture, dedicated <|dlm_pad|> padding.",
    hf_model="apple/DiffuCoder-7B-Instruct",
    revision="4fdd458006",
    license="apple-amlr",
    prediction_field="shifted",
    topology="bidirectional",
    # config.json pad_token_id is 151643; the tokenizer's pad is <|dlm_pad|>
    tokens={"mask_token_id": 151666, "eos_token_id": 151643,
            "pad_token_id": 151667, "config_pad_token_id": 151643,
            "turn_end_token_id": 151645},
    reference="https://github.com/apple/ml-diffucoder",
    extra=_DREAM_DECODING,
)
_checkpoint(
    "integration/illada-8b-instruct",
    "iLLaDA-8B-Instruct: masked diffusion trained from scratch (12T tokens).",
    hf_model="GSAI-ML/iLLaDA-8B-Instruct",
    revision="5769f04922",
    license="apache-2.0",
    prediction_field="same_position",
    topology="bidirectional",
    tokens={"mask_token_id": 5, "eos_token_id": 2, "pad_token_id": 1,
            "bos_token_id": 0},
    reference="https://github.com/ML-GSAI/LLaDA",
)
_checkpoint(
    "integration/fast-dllm-v2-7b",
    "Fast-dLLM v2 7B: block diffusion converted from Qwen2.5-7B-Instruct.",
    hf_model="Efficient-Large-Model/Fast_dLLM_v2_7B",
    revision="0661abf5f9",
    license="apache-2.0",
    prediction_field="shifted",
    topology="block_causal",
    block_length=32,
    # config pad/eos is <|im_end|>; the tokenizer's pad is <|endoftext|>
    tokens={"mask_token_id": 151665, "eos_token_id": 151645,
            "pad_token_id": 151645, "tokenizer_pad_token_id": 151643},
    reference="https://github.com/NVlabs/Fast-dLLM",
    extra={"sub_block_length": 8, "sft_padding": "mask_ignored",
           "complementary_mask": True},
)
_checkpoint(
    "integration/sdar-8b-chat",
    "SDAR-8B-Chat: block diffusion from Qwen3-8B (card licence apache-2.0, "
    "GitHub MIT).",
    hf_model="JetLM/SDAR-8B-Chat",
    revision="ac4528d2c0",
    license="apache-2.0",
    prediction_field="same_position",
    topology="block_causal",
    block_length=4,
    tokens={"mask_token_id": 151669, "eos_token_id": 151643,
            "pad_token_id": 151643, "turn_end_token_id": 151645},
    reference="https://github.com/JetAstra/SDAR",
)
_checkpoint(
    "integration/trado-8b-instruct",
    "TraDo-8B-Instruct: SDAR architecture with TraceRL post-training.",
    hf_model="Gen-Verse/TraDo-8B-Instruct",
    revision="2d37bd3ebb",
    license="mit",
    prediction_field="same_position",
    topology="block_causal",
    block_length=4,
    tokens={"mask_token_id": 151669, "eos_token_id": 151643,
            "pad_token_id": 151643, "turn_end_token_id": 151645},
    reference="https://github.com/Gen-Verse/dLLM-RL",
)
_checkpoint(
    "integration/nemotron-labs-diffusion-8b",
    "Nemotron-Labs-Diffusion-8B: one set of weights for block diffusion, "
    "AR and self-speculative decoding.",
    hf_model="nvidia/Nemotron-Labs-Diffusion-8B",
    revision="16c67f0560",
    license="nvidia-nemotron-open-model-license",
    prediction_field="same_position",
    topology="block_causal",
    block_length=32,
    tokens={"mask_token_id": 100, "eos_token_id": 11, "bos_token_id": 1},
    reference="https://github.com/NVlabs/Nemotron-Labs-Diffusion",
    extra={"decoding_modes": ["block_diffusion", "autoregressive",
                              "self_speculative"]},
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
