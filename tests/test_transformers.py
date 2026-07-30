"""Optional Transformers integration tests; no model download required."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dllm import AttentionTopology, DenoiserInput
from dllm.adapters import TransformersDenoiserAdapter
from dllm.validation import validate_denoiser


def test_tiny_bert_masked_lm_adapter():
    try:
        from transformers import BertConfig, BertForMaskedLM
    except ImportError:
        return

    model = BertForMaskedLM(
        BertConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
        )
    ).eval()
    adapter = TransformersDenoiserAdapter(
        model,
        prediction_field="same_position",
        default_topology="bidirectional",
    )
    ids = torch.tensor([[1, 2, 3, 4]])
    topology = AttentionTopology.bidirectional(batch_size=1, length=4)
    report = validate_denoiser(
        adapter,
        DenoiserInput(input_ids=ids, topology=topology),
    )
    assert report.contract == "Denoiser"


if __name__ == "__main__":
    test_tiny_bert_masked_lm_adapter()
    print("PASS  test_tiny_bert_masked_lm_adapter")
