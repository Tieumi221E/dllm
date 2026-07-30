"""Contract tests for framework adapters and extension validation."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dllm import (
    AttentionTopology,
    CanvasConfig,
    DenoiserInput,
    DiffusionTransformer,
    generate_canvas,
)
from dllm.adapters import (
    AdapterCapabilityError,
    TransformersDenoiserAdapter,
)
from dllm.validation import (
    validate_block_cache_denoiser,
    validate_denoiser,
    validate_prefix_cache_denoiser,
)


VOCAB = 32
MASK = 30


class FakeTransformersModel(torch.nn.Module):
    """Small HF-shaped module with deterministic same-position logits."""

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
        return_dict=True,
        logit_scale=1.0,
        attention_mode=None,
    ):
        if input_ids is None:
            batch, length = inputs_embeds.shape[:2]
            token_ids = inputs_embeds[..., 0].long().remainder(VOCAB)
        else:
            batch, length = input_ids.shape
            token_ids = input_ids
        logits = torch.zeros(batch, length, VOCAB, device=token_ids.device)
        targets = (token_ids + 1).remainder(VOCAB)
        logits.scatter_(-1, targets.unsqueeze(-1), 4.0 * logit_scale)
        cache = ("fake-cache", length) if use_cache else None
        return SimpleNamespace(
            logits=logits,
            past_key_values=cache,
            hidden_states=None,
        )


def test_transformers_adapter_runs_canvas_and_contract():
    adapter = TransformersDenoiserAdapter(
        FakeTransformersModel(),
        prediction_field="same_position",
        default_topology="bidirectional",
        model_kwargs={"logit_scale": 0.5},
    )
    ids = torch.tensor([[1, 2, 3, 4]])
    topology = AttentionTopology.bidirectional(batch_size=1, length=4)
    report = validate_denoiser(
        adapter,
        DenoiserInput(input_ids=ids, topology=topology),
    )
    assert report.contract == "Denoiser"
    output = generate_canvas(
        adapter,
        ids[:, :2],
        MASK,
        CanvasConfig(gen_length=4, block_length=4, steps=2),
    )
    assert output.canvas.shape == (1, 6)
    assert (output.canvas[:, 2:] != MASK).all()


def test_next_token_adapter_is_an_explicit_negative_control():
    adapter = TransformersDenoiserAdapter(
        FakeTransformersModel(),
        prediction_field="next_token",
        default_topology="causal",
    )
    ids = torch.tensor([[1, 2, 3]])
    request = DenoiserInput(
        input_ids=ids,
        topology=AttentionTopology.causal(batch_size=1, length=3),
        use_cache=True,
    )
    output = adapter.execute(request)
    assert output.logits.shape == (1, 3, VOCAB)
    assert output.cache == ("fake-cache", 3)
    try:
        adapter.denoise(request)
    except AdapterCapabilityError:
        pass
    else:
        raise AssertionError("next-token logits must not satisfy denoising")
    try:
        generate_canvas(
            adapter,
            ids[:, :2],
            MASK,
            CanvasConfig(gen_length=2, block_length=2, steps=1),
        )
    except TypeError:
        pass
    else:
        raise AssertionError("AR adapter must not enter full-canvas sampling")


def test_adapter_rejects_undeclared_topology_and_managed_kwargs():
    adapter = TransformersDenoiserAdapter(
        FakeTransformersModel(),
        prediction_field="same_position",
        default_topology="bidirectional",
    )
    ids = torch.tensor([[1, 2, 3]])
    causal = AttentionTopology.causal(batch_size=1, length=3)
    try:
        adapter.denoise(DenoiserInput(input_ids=ids, topology=causal))
    except AdapterCapabilityError:
        pass
    else:
        raise AssertionError("expected topology capability error")

    try:
        adapter.denoise(
            DenoiserInput(
                input_ids=ids,
                model_kwargs={"input_ids": ids},
            )
        )
    except ValueError:
        pass
    else:
        raise AssertionError("managed model kwargs must not be overridden")

    multimode = TransformersDenoiserAdapter(
        FakeTransformersModel(),
        prediction_field="same_position",
        default_topology="bidirectional",
        attention_topologies={"bidirectional", "causal"},
        topology_adapter=lambda topology: {"attention_mode": topology.name},
    )
    output = multimode.denoise(
        DenoiserInput(input_ids=ids, topology=causal)
    )
    assert output.logits.shape == (1, 3, VOCAB)


def test_reference_cache_contract_validators():
    model = DiffusionTransformer(
        vocab_size=VOCAB,
        max_position_embeddings=32,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        intermediate_size=32,
    ).eval()
    prefix = torch.tensor([[1, 2, 3]])
    block = torch.tensor([[4, 5]])
    report = validate_block_cache_denoiser(model, prefix, block)
    assert report.contract == "BlockCacheDenoiser"

    canvas = torch.tensor([[1, 2, 3, MASK, MASK]])
    report = validate_prefix_cache_denoiser(model, canvas, prefix_length=3)
    assert report.contract == "PrefixCacheDenoiser"


if __name__ == "__main__":
    tests = sorted(
        (name, value)
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    for _, test in tests:
        test()
        print("PASS ", test.__name__)
    print(f"\n{len(tests)}/{len(tests)} passed")
