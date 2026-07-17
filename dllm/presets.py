"""Configuration presets for common regimes.

Each preset bundles model kwargs and sampler/loss settings for a given
training/inference regime.

Usage:
    from dllm.presets import get_preset
    p = get_preset("small-gqa")
    model = DiffusionTransformer(**p["model"])
    out = generate_blockwise(model, prompt, mask_id, BlockwiseConfig(**p["blockwise"]))
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict

_PRESETS: Dict[str, dict] = {
    # ~110M MHA model, learned positions, with projection/FF biases.
    "small-mha": {
        "model": dict(
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
        ),
        "canvas": dict(
            commit="transfer", sampling="gumbel", temperature=1.0, confidence="prob"
        ),
        "loss": dict(norm="tokens"),
    },
    # ~105M GQA model (12 Q / 2 KV heads), incremental block-wise inference.
    "small-gqa": {
        "model": dict(
            vocab_size=4096,
            max_position_embeddings=2048,
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            num_kv_heads=2,
            intermediate_size=3072,
            position_embedding="learned",
            attn_bias=False,
        ),
        "blockwise": dict(
            block_length=32,
            steps_per_block=32,
            gen_length=384,
            temperature=1.0,
            sampling="gumbel",
            commit="transfer",
            confidence="prob",
        ),
        "block_sft": dict(canvas="truncated"),
        "loss": dict(norm="answer"),
    },
    # LLaDA-8B-Instruct backbone (HF AutoModel); full-canvas sampling.
    "llada-8b": {
        "hf_model": "GSAI-ML/LLaDA-8B-Instruct",
        "mask_token_id": 126336,
        "canvas": dict(
            gen_length=256,
            block_length=256,
            steps=128,
            temperature=0.0,
            sampling="gumbel",
            commit="transfer",
            confidence="prob",
        ),
        "canvas_trajectory": dict(
            gen_length=256,
            block_length=256,
            steps=128,
            temperature=0.6,
            sampling="gumbel",
            commit="transfer",
            confidence="prob",
            record_trace=True,
        ),
        "loss": dict(norm="answer"),
    },
}


def get_preset(name: str) -> dict:
    if name not in _PRESETS:
        raise KeyError(f"unknown preset '{name}'; available: {sorted(_PRESETS)}")
    return deepcopy(_PRESETS[name])


def list_presets():
    return sorted(_PRESETS)
