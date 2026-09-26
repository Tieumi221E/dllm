"""Dream decoding: timestep quota policy and conformance with the reference.

The unit tests need nothing external. The conformance test compares
``generate_canvas`` (shifted adapter + ``TimestepQuotaCommitPolicy``) with
Dream's own ``_sample`` loop, step by step, on a context-dependent fake
model. It runs only when the reference files are available (see
``_dream_reference.py``) and ``transformers`` is installed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dllm import CanvasConfig, TimestepQuotaCommitPolicy, generate_canvas
from dllm.adapters import TransformersDenoiserAdapter
from dllm.sampling import CommitState, apply_commit_policy

from _dream_reference import load_module

VOCAB, MASK = 40, 39


def _state(n, length, step, steps):
    cand = torch.zeros(1, length, dtype=torch.bool)
    cand[0, :n] = True
    conf = torch.rand(1, length)
    return CommitState(conf, cand, cand.clone(), step, steps)


def test_quota_is_idle_early_and_complete_at_the_end():
    policy = TimestepQuotaCommitPolicy()
    # 3 masked, 8 steps: int(3 * (1 - s/t)) is 0 on the first step
    first = apply_commit_policy(policy, _state(3, 6, 0, 8))
    assert int(first.commit.sum()) == 0
    last = apply_commit_policy(policy, _state(3, 6, 7, 8))
    assert int(last.commit.sum()) == 3
    single = apply_commit_policy(policy, _state(5, 6, 0, 1))
    assert int(single.commit.sum()) == 5


def test_quota_known_values():
    # t = linspace(1, 1e-3, 5); step 1 has t=0.75025, s=0.5005:
    # int(8 * (1 - 0.5005/0.75025)) = int(2.663...) = 2
    policy = TimestepQuotaCommitPolicy()
    got = apply_commit_policy(policy, _state(8, 8, 1, 4))
    assert int(got.commit.sum()) == 2


def test_idle_steps_only_for_declaring_policies():
    class Lazy:
        def select(self, state):
            from dllm import CommitDecision

            return CommitDecision(torch.zeros_like(state.candidates))

    with pytest.raises(ValueError):
        apply_commit_policy(Lazy(), _state(3, 6, 0, 8))
    # a declaring policy may idle, but not on its last planned step
    with pytest.raises(ValueError):

        class LazyIdle(Lazy):
            allows_idle_steps = True

        apply_commit_policy(LazyIdle(), _state(3, 6, 7, 8))


class ContextFake(torch.nn.Module):
    """Raw rows depend on the token at each position and on the canvas mean,
    so the commit order changes the result."""

    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = torch.randn(VOCAB, VOCAB, generator=g)
        self.ctx = torch.randn(VOCAB, VOCAB, generator=g)
        self.pos = torch.randn(64, VOCAB, generator=g)

    def raw(self, ids):
        length = ids.shape[1]
        logits = self.emb[ids] + self.ctx[ids].mean(dim=1, keepdim=True)
        logits = logits + self.pos[:length]
        logits[..., MASK] = -1e4  # neither side should ever predict MASK
        return logits

    def forward(self, input_ids=None, attention_mask=None, return_dict=True):
        return SimpleNamespace(
            logits=self.raw(input_ids), past_key_values=None, hidden_states=None
        )


def _dllm_run(model, prompt, gen, steps):
    adapter = TransformersDenoiserAdapter(
        model, prediction_field="shifted", default_topology="bidirectional"
    )
    cfg = CanvasConfig(
        gen_length=gen,
        block_length=gen,
        steps=steps,
        temperature=0.0,
        commit=TimestepQuotaCommitPolicy(eps=1e-3),
        confidence="prob",
        allow_mask_prediction=True,
    )
    return generate_canvas(adapter, prompt, MASK, cfg)


def test_timestep_policy_decodes_to_completion():
    out = _dllm_run(ContextFake(0), torch.tensor([[1, 2, 3]]), gen=12, steps=20)
    assert (out.canvas != MASK).all()
    assert int(out.step_map.max()) <= 19


def _load_reference():
    pytest.importorskip("transformers")
    return load_module("generation_utils.py")


@pytest.mark.parametrize(
    "seed,prompt_len,gen,steps",
    [(0, 3, 8, 8), (1, 4, 12, 5), (2, 2, 10, 24), (3, 5, 16, 16), (4, 3, 9, 3)],
)
def test_conformance_with_dream_reference(seed, prompt_len, gen, steps):
    ref = _load_reference()
    model = ContextFake(seed)
    g = torch.Generator().manual_seed(100 + seed)
    prompt = torch.randint(0, MASK, (1, prompt_len), generator=g)

    class FakeDream(ref.DreamGenerationMixin):
        device = torch.device("cpu")

        def __call__(self, x, attention_mask, tok_idx):
            return SimpleNamespace(logits=model.raw(x))

    config = ref.DreamGenerationConfig(
        max_length=prompt_len + gen,
        steps=steps,
        eps=1e-3,
        alg="maskgit_plus",
        temperature=0.0,
        mask_token_id=MASK,
        return_dict_in_generate=True,
        output_history=True,
    )
    result = FakeDream()._sample(
        prompt, None, config, lambda i, x, logits: x, lambda i, x, logits: logits
    )
    # reference commit step per generated position
    ref_steps = torch.full((gen,), -1, dtype=torch.long)
    for i, state in enumerate(result.history):
        newly = (state[0, prompt_len:] != MASK) & (ref_steps == -1)
        ref_steps[newly] = i

    out = _dllm_run(model, prompt, gen, steps)
    assert torch.equal(out.canvas, result.sequences)
    # idle steps advance the step counter on both sides, so the commit
    # step of every generated position must match exactly
    assert torch.equal(out.step_map[0], ref_steps)
