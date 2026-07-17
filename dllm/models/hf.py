"""HuggingFace wrapper: config + PreTrainedModel around DiffusionTransformer.

Config field names and defaults are kept stable across versions, so
checkpoints saved by predecessor wrappers (``config.json`` + weights under
the ``model.`` prefix) load via ``DiffusionTransformerLM.from_pretrained``
unchanged; newer fields (``position_embedding``, ``attn_bias``, ``ff_bias``)
default to the original behaviour.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple

import torch

try:
    from transformers import PretrainedConfig, PreTrainedModel
except ImportError as e:  # pragma: no cover
    raise ImportError("dllm.models.hf requires `transformers`") from e

from .transformer import DiffusionTransformer


class DiffusionLMOutput(NamedTuple):
    logits: torch.FloatTensor
    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None


class DiffusionConfig(PretrainedConfig):
    model_type = "diffusion_transformer"

    def __init__(
        self,
        vocab_size=4096,
        max_position_embeddings=2048,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        num_kv_heads=2,
        intermediate_size=3072,
        emb_dropout=0.0,
        resid_dropout=0.0,
        attention_dropout=0.0,
        tie_embeddings=True,
        position_embedding="learned",
        rope_theta=10000.0,
        attn_bias=False,
        ff_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size
        self.emb_dropout = emb_dropout
        self.resid_dropout = resid_dropout
        self.attention_dropout = attention_dropout
        self.tie_embeddings = tie_embeddings
        self.position_embedding = position_embedding
        self.rope_theta = rope_theta
        self.attn_bias = attn_bias
        self.ff_bias = ff_bias


class DiffusionTransformerLM(PreTrainedModel):
    config_class = DiffusionConfig

    def __init__(self, config: DiffusionConfig):
        super().__init__(config)
        self.model = DiffusionTransformer(
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            intermediate_size=config.intermediate_size,
            position_embedding=config.position_embedding,
            rope_theta=config.rope_theta,
            attn_bias=config.attn_bias,
            ff_bias=getattr(config, "ff_bias", False),
            emb_dropout=config.emb_dropout,
            resid_dropout=config.resid_dropout,
            attention_dropout=config.attention_dropout,
            tie_embeddings=config.tie_embeddings,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> DiffusionLMOutput:
        logits = self.model(input_ids, attention_mask=attention_mask)
        return DiffusionLMOutput(logits=logits)
