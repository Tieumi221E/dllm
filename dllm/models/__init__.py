from .transformer import DiffusionTransformer, KVCache

__all__ = [
    "DiffusionTransformer",
    "KVCache",
    "DiffusionConfig",
    "DiffusionTransformerLM",
]


def __getattr__(name):
    # transformers is optional; import the HF wrapper lazily
    if name in ("DiffusionConfig", "DiffusionTransformerLM", "DiffusionLMOutput"):
        from . import hf

        return getattr(hf, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
