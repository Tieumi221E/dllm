from .transformer import DiffusionTransformer, KVCache
from .protocol import (
    BlockCacheDenoiser,
    Denoiser,
    DenoiserInput,
    DenoiserOutput,
    ModelCapabilities,
    PrefixCacheDenoiser,
    extract_logits,
)

__all__ = [
    "DiffusionTransformer",
    "KVCache",
    "BlockCacheDenoiser",
    "Denoiser",
    "DenoiserInput",
    "DenoiserOutput",
    "ModelCapabilities",
    "PrefixCacheDenoiser",
    "extract_logits",
    "DiffusionConfig",
    "DiffusionTransformerLM",
    "DiffusionLMOutput",
]


def __getattr__(name):
    # transformers is optional; import the HF wrapper lazily
    if name in ("DiffusionConfig", "DiffusionTransformerLM", "DiffusionLMOutput"):
        from . import hf

        return getattr(hf, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
