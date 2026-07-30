"""Smoke + alignment tests for dllm.

Run:  python tests/test_smoke.py   (or pytest tests/)

The two most important tests:
- test_loss_matches_reference_*: our loss reproduces the reference
  reference code numerically (pretrain and SFT normalization);
- test_kv_cache_exactness: build_kv_cache + forward_block equals a full
  forward under an explicit block-causal attention bias.
"""

import math
import sys
import os

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dllm import (
    BlockSFTCollator,
    BlockwiseConfig,
    CanvasConfig,
    DiffusionTransformer,
    LinearSchedule,
    PretrainCollator,
    SFTCollator,
    diffusion_loss,
    forward_process,
    generate_blockwise,
    generate_canvas,
    get_schedule,
    masked_cross_entropy,
    mc_conditional_nll,
    ppo_clip_objective,
    trajectory_logprobs,
    trajectory_states,
)
from dllm.sampling.utils import get_num_transfer_tokens, split_steps

torch.manual_seed(0)

V, MASK, EOS, PAD = 64, 60, 61, 62


def tiny_model(pos="learned", kv_heads=2, bias=False, max_pos=256):
    torch.manual_seed(7)
    return DiffusionTransformer(
        vocab_size=V,
        max_position_embeddings=max_pos,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=kv_heads,
        intermediate_size=64,
        position_embedding=pos,
        attn_bias=bias,
    ).eval()


# --------------------------- schedules / masking --------------------------- #


def test_linear_schedule_weight():
    s = LinearSchedule(eps=1e-3)
    t = torch.tensor([0.0, 0.5, 1.0])
    p = s.mask_prob(t)
    assert torch.allclose(p, torch.tensor([1e-3, 0.5005, 1.0]))
    assert torch.allclose(s.weight(t), 1.0 / p)
    assert get_schedule("linear").__class__ is LinearSchedule


def test_forward_process_stats():
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 50, (64, 128))
    t = torch.full((64,), 0.5)
    m = forward_process(ids, MASK, t=t, generator=g)
    rate = m.masked_indices.float().mean().item()
    assert abs(rate - 0.5005) < 0.02, rate
    assert (m.noisy_ids[m.masked_indices] == MASK).all()
    assert (m.noisy_ids[~m.masked_indices] == ids[~m.masked_indices]).all()
    assert m.p_mask.shape == ids.shape and (m.p_mask == m.p_mask[:, :1]).all()
    # maskable respected
    maskable = torch.zeros_like(ids, dtype=torch.bool)
    maskable[:, :10] = True
    m2 = forward_process(ids, MASK, maskable=maskable, t=t, generator=g)
    assert not m2.masked_indices[:, 10:].any()


# ----------------------- loss vs reference implementation ------------------ #


def _reference_pretrain_loss(logits, input_ids, masked_indices, p_mask):
    # verbatim reference implementation (see README references)
    token_loss = (
        F.cross_entropy(
            logits[masked_indices], input_ids[masked_indices], reduction="none"
        )
        / p_mask[masked_indices]
    )
    return token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])


def _reference_sft_loss(logits, input_ids, masked_indices, p_mask, answer_lengths_row):
    token_loss = (
        F.cross_entropy(
            logits[masked_indices], input_ids[masked_indices], reduction="none"
        )
        / p_mask[masked_indices]
    )
    answer_lengths = answer_lengths_row.unsqueeze(1).repeat(1, input_ids.shape[1])
    return torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]


def test_loss_matches_reference_pretrain():
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 50, (4, 32))
    m = forward_process(ids, MASK, generator=g)
    logits = torch.randn(4, 32, V)
    ours = diffusion_loss(logits, ids, m.masked_indices, m.p_mask, norm="tokens")
    ref = _reference_pretrain_loss(logits, ids, m.masked_indices, m.p_mask)
    assert torch.allclose(ours, ref, atol=1e-6), (ours, ref)


def test_loss_matches_reference_sft():
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(0, 50, (4, 32))
    prompt_len = torch.tensor([5, 8, 3, 10])
    maskable = torch.arange(32).unsqueeze(0) >= prompt_len.unsqueeze(1)
    m = forward_process(ids, MASK, maskable=maskable, generator=g)
    logits = torch.randn(4, 32, V)
    ours = diffusion_loss(
        logits, ids, m.masked_indices, m.p_mask, norm="answer", maskable=maskable
    )
    ref = _reference_sft_loss(
        logits, ids, m.masked_indices, m.p_mask, maskable.sum(dim=1).float()
    )
    assert torch.allclose(ours, ref, atol=1e-6), (ours, ref)


def test_loss_zero_when_nothing_masked():
    ids = torch.randint(0, 50, (2, 8))
    logits = torch.randn(2, 8, V, requires_grad=True)
    none = torch.zeros(2, 8, dtype=torch.bool)
    loss = diffusion_loss(logits, ids, none, torch.full((2, 8), 0.5), norm="tokens")
    assert loss.item() == 0.0
    loss.backward()  # graph intact


def test_loss_rejects_biased_combo():
    ids = torch.randint(0, 50, (2, 8))
    logits = torch.randn(2, 8, V)
    mi = torch.zeros(2, 8, dtype=torch.bool)
    mi[:, :3] = True
    pm = torch.full((2, 8), 0.5)
    try:
        diffusion_loss(logits, ids, mi, pm, norm="masked")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for masked+importance_weight")
    # uniform weighting with masked norm is fine
    loss = diffusion_loss(logits, ids, mi, None, norm="masked", importance_weight=False)
    assert torch.isfinite(loss)


def test_masked_cross_entropy_composable_reductions():
    logits = torch.randn(3, 5, V, requires_grad=True)
    targets = torch.randint(0, V, (3, 5))
    selected = torch.zeros(3, 5, dtype=torch.bool)
    selected[0, :2] = True
    selected[1, :4] = True
    token_weight = torch.tensor([2.0, 0.5, 1.0]).unsqueeze(1)
    sample_weight = torch.tensor([1.0, 3.0, 9.0])

    dense = masked_cross_entropy(
        logits,
        targets,
        selected,
        token_weight=token_weight,
        reduction="none",
    )
    row_mean = dense.sum(dim=1) / selected.sum(dim=1).clamp(min=1)
    expected = (row_mean[0] + 3.0 * row_mean[1]) / 4.0
    actual = masked_cross_entropy(
        logits,
        targets,
        selected,
        token_weight=token_weight,
        sample_weight=sample_weight,
        reduction="sample_mean",
    )
    assert torch.allclose(actual, expected)
    actual.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


# --------------------------------- model ---------------------------------- #


def test_model_variants_forward():
    for pos in ("learned", "rope"):
        for kv in (4, 2):
            model = tiny_model(pos=pos, kv_heads=kv)
            ids = torch.randint(0, V, (3, 20))
            out = model(ids)
            assert out.shape == (3, 20, V)
    # padding mask changes result
    model = tiny_model()
    ids = torch.randint(0, V, (2, 12))
    am = torch.ones(2, 12, dtype=torch.long)
    am[:, 8:] = 0
    a = model(ids, attention_mask=am)
    b = model(ids)
    assert not torch.allclose(a[:, :8], b[:, :8], atol=1e-5)


def test_position_overflow_raises():
    model = tiny_model(max_pos=16)
    try:
        model(torch.randint(0, V, (1, 17)))
    except ValueError:
        return
    raise AssertionError("expected ValueError for overlong sequence")


def test_kv_cache_exactness():
    """cache path == full forward under explicit block-causal bias."""
    for pos in ("learned", "rope"):
        model = tiny_model(pos=pos)
        Lp, Lb = 11, 5
        prefix = torch.randint(0, V, (2, Lp))
        block = torch.randint(0, V, (2, Lb))
        cache = model.build_kv_cache(prefix)
        blk_logits, _ = model.forward_block(block, cache)

        full = torch.cat([prefix, block], dim=1)
        L = Lp + Lb
        bias = torch.zeros(1, 1, L, L)
        bias[:, :, :Lp, Lp:] = float("-inf")  # prefix cannot see block
        ref = model(full, attn_bias=bias)
        assert torch.allclose(blk_logits, ref[:, Lp:, :], atol=1e-4), pos


def test_kv_cache_extend_and_mask():
    model = tiny_model()
    prefix = torch.randint(0, V, (2, 6))
    am = torch.ones(2, 6, dtype=torch.long)
    am[1, 4:] = 0  # padded prompt
    cache = model.build_kv_cache(prefix, attention_mask=am)
    b1 = torch.randint(0, V, (2, 4))
    logits1, kvs1 = model.forward_block(b1, cache)
    cache2 = cache.extend(kvs1)
    assert cache2.length == 10 and cache2.key_mask.shape == (2, 10)
    b2 = torch.randint(0, V, (2, 4))
    logits2, _ = model.forward_block(b2, cache2)
    assert logits2.shape == (2, 4, V)


# -------------------------------- sampling --------------------------------- #


def test_quota_helpers():
    mi = torch.ones(3, 10, dtype=torch.bool)
    mi[1, 5:] = False
    q = get_num_transfer_tokens(mi, 4)
    assert (q.sum(dim=1) == mi.sum(dim=1)).all()
    assert split_steps(10, 4) == [3, 3, 2, 2]


def test_canvas_transfer_greedy():
    model = tiny_model()
    prompt = torch.randint(0, V, (2, 7))
    cfg = CanvasConfig(
        gen_length=16,
        block_length=8,
        steps=8,
        temperature=0.0,
        commit="transfer",
        eos_token_id=EOS,
    )
    out = generate_canvas(model, prompt, MASK, cfg)
    assert out.canvas.shape == (2, 7 + 16)
    assert (out.canvas[:, 7:] != MASK).all()  # everything decoded
    assert (out.step_map >= 0).all()
    # block ordering: all commits in block 0 happen before block 1
    assert out.step_map[:, :8].max() < out.step_map[:, 8:].min()
    # commits per step match the quota (8 masks / 4 steps = 2 per row per step)
    from collections import Counter

    for row in out.step_map.tolist():
        counts = Counter(row)
        assert all(c == 2 for c in counts.values()), counts


def test_canvas_modes_run():
    model = tiny_model()
    prompt = torch.randint(0, V, (1, 5))
    for commit in ("transfer", "threshold"):
        for sampling in ("gumbel", "multinomial"):
            cfg = CanvasConfig(
                gen_length=8,
                block_length=4,
                steps=4,
                temperature=1.0,
                sampling=sampling,
                commit=commit,
                threshold=0.5,
                eos_token_id=EOS,
            )
            out = generate_canvas(
                model, prompt, MASK, cfg, generator=torch.Generator().manual_seed(3)
            )
            assert (out.canvas[:, 5:] != MASK).all(), (commit, sampling)
    # CFG + gumbel + trace
    cfg = CanvasConfig(
        gen_length=8,
        block_length=8,
        steps=8,
        temperature=0.7,
        sampling="gumbel",
        commit="transfer",
        cfg_scale=1.0,
        record_trace=True,
        eos_token_id=EOS,
    )
    out = generate_canvas(model, prompt, MASK, cfg)
    tr = out.traces[0]
    assert len(tr.steps) > 0 and len(tr.final) == 8
    sm = tr.step_map
    assert all(s >= 0 for s in sm)
    assert math.isfinite(tr.content_logprob_mean(EOS))
    restored = type(tr).from_dict(tr.to_dict())
    assert restored.to_dict() == tr.to_dict()


def test_trace_topk_and_summary():
    model = tiny_model()
    prompt = torch.randint(0, V, (1, 5))
    out = generate_canvas(
        model,
        prompt,
        MASK,
        CanvasConfig(
            gen_length=8,
            block_length=4,
            steps=4,
            record_trace=True,
            trace_topk=3,
            eos_token_id=EOS,
        ),
    )
    trace = out.traces[0]
    assert trace.steps[0].distributions
    assert len(trace.steps[0].distributions[0].topk) == 3
    summary = trace.summary(EOS)
    assert summary["commit_tokens"] == 8
    assert summary["steps"] == len(trace.steps)
    assert "pmax_mean" in summary and "margin_mean" in summary


def test_canvas_prefix_cache_runs():
    model = tiny_model()
    prompt = torch.randint(0, V, (1, 6))
    base = dict(
        gen_length=12,
        block_length=4,
        steps=12,
        temperature=0.0,
        commit="transfer",
        eos_token_id=EOS,
    )
    out_nc = generate_canvas(model, prompt, MASK, CanvasConfig(**base))
    out_c = generate_canvas(
        model, prompt, MASK, CanvasConfig(prefix_cache=True, **base)
    )
    out_w = generate_canvas(
        model, prompt, MASK, CanvasConfig(prefix_cache=True, further_horizon=4, **base)
    )
    assert (out_c.canvas[:, 6:] != MASK).all()
    assert (out_w.canvas[:, 6:] != MASK).all()
    # cache is an approximation - shapes/termination matter, not equality
    assert out_c.canvas.shape == out_nc.canvas.shape


def test_blockwise_cache_equals_nocache():
    """use_cache is a pure speed knob: with T=0 the cache path must equal the
    block-causal recompute path (default) exactly."""
    model = tiny_model()
    prompt = torch.randint(0, V, (2, 6))
    base = dict(
        gen_length=12,
        block_length=4,
        steps_per_block=4,
        temperature=0.0,
        commit="transfer",
        eos_token_id=None,
    )
    a = generate_blockwise(model, prompt, MASK, BlockwiseConfig(use_cache=True, **base))
    b = generate_blockwise(
        model, prompt, MASK, BlockwiseConfig(use_cache=False, **base)
    )
    assert torch.equal(a.canvas, b.canvas)
    assert torch.equal(a.step_map, b.step_map)


def test_blockwise_eos_early_exit():
    model = tiny_model()
    prompt = torch.randint(0, V, (1, 4))
    cfg = BlockwiseConfig(
        gen_length=16,
        block_length=4,
        steps_per_block=4,
        temperature=1.0,
        sampling="multinomial",
        commit="transfer",
        eos_token_id=EOS,
    )
    out = generate_blockwise(
        model,
        prompt,
        MASK,
        cfg,
        num_samples=3,
        generator=torch.Generator().manual_seed(5),
    )
    assert len(out.sequences) == 3
    for seq in out.sequences:
        assert EOS not in seq  # stripped
    assert (out.canvas[:, :4] == prompt.expand(3, -1)).all()


# ----------------------------------- RL ------------------------------------ #


def test_trajectory_logprobs_partition():
    model = tiny_model()
    prompt = torch.randint(0, V, (1, 5))
    cfg = BlockwiseConfig(
        gen_length=8,
        block_length=4,
        steps_per_block=2,
        temperature=0.0,
        commit="transfer",
        eos_token_id=None,
    )
    out = generate_blockwise(model, prompt, MASK, cfg)
    gen = out.canvas[0, 5:]
    sm = out.step_map[0]

    def model_fn(ids):
        return model(ids)

    steps = trajectory_logprobs(
        model_fn,
        prompt[0],
        gen,
        sm,
        MASK,
        block_length=4,
        canvas="incremental",
        collapse="none",
    )
    covered = torch.zeros(8, dtype=torch.bool)
    for st in steps:
        rel = st.positions - 5
        assert not covered[rel].any()  # each position exactly once
        covered[rel] = True
        assert torch.isfinite(st.logp).all()
    assert covered.all()

    # block collapse: one entry per block
    steps_b = trajectory_logprobs(
        model_fn,
        prompt[0],
        gen,
        sm,
        MASK,
        block_length=4,
        canvas="incremental",
        collapse="block",
    )
    assert len(steps_b) == 2
    # full-canvas mode also runs
    steps_f = trajectory_logprobs(
        model_fn,
        prompt[0],
        gen,
        sm,
        MASK,
        block_length=4,
        canvas="full",
        collapse="block",
    )
    assert len(steps_f) == 2

    states = trajectory_states(
        prompt[0],
        gen,
        sm,
        MASK,
        block_length=4,
        canvas="incremental",
        collapse="block",
    )
    assert len(states) == 2
    assert states[0].input_ids.numel() == 5 + 4
    assert states[1].input_ids.numel() == 5 + 8
    assert states[0].to("cpu").input_ids.device.type == "cpu"


def test_trajectory_logprobs_gradient_and_ppo():
    model = tiny_model()
    prompt = torch.randint(0, V, (1, 3))
    gen = torch.randint(0, V - 4, (4,))
    step_map = torch.tensor([0, 0, 1, 1])
    scored = trajectory_logprobs(
        model,
        prompt[0],
        gen,
        step_map,
        MASK,
        block_length=4,
        canvas="full",
        collapse="none",
        with_grad=True,
    )
    logp = torch.cat([step.logp for step in scored])
    old_logp = logp.detach() - 0.1
    objective = ppo_clip_objective(logp, old_logp, advantage=1.0)
    objective.loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert objective.ratio_mean.item() > 1.0


# ------------------------------- data / eval ------------------------------- #


def test_collators():
    g = torch.Generator().manual_seed(9)
    pc = PretrainCollator(MASK, PAD, random_length_prob=0.0, generator=g)
    b = pc([[1, 2, 3, 4, 5], [6, 7, 8]])
    assert b["input_ids"].shape == (2, 5)
    assert not b["masked_indices"][1, 3:].any()  # pad never masked

    sc = SFTCollator(MASK, EOS, generator=g)
    b = sc(
        [
            {"prompt_ids": [1, 2, 3], "response_ids": [4, 5]},
            {"prompt_ids": [1, 2], "response_ids": [4, 5, 6, 7, 8]},
        ]
    )
    ids = b["clean_ids"]
    assert ids.shape[1] == 8  # 2 + 5 + eos
    assert not b["masked_indices"][0, :3].any()  # prompt clean
    assert (ids[0, 6:] == EOS).all()  # EOS-padded
    assert b["maskable"][0, 6:].all()  # EOS pads maskable
    assert b["answer_lengths"].tolist() == [5, 6]

    for canvas in ("truncated", "full"):
        bc = BlockSFTCollator(
            MASK, EOS, PAD, block_length=4, canvas=canvas, generator=g
        )
        b = bc(
            [
                {"prompt_ids": [1, 2], "response_ids": [3, 4, 5]},
                {"prompt_ids": [1], "response_ids": [3, 4, 5, 6, 7]},
            ]
        )
        assert b["maskable"].sum(dim=1).tolist() == [4, 4]
        assert b["masked_indices"].any()
        loss = diffusion_loss(
            torch.randn(*b["clean_ids"].shape, V),
            b["clean_ids"],
            b["masked_indices"],
            b["p_mask"],
            norm="answer",
            maskable=b["maskable"],
        )
        assert torch.isfinite(loss)


def test_mc_conditional_nll_uniform_model():
    """A constant-logits model must give NLL/token == log V."""

    class Uniform:
        def __call__(self, ids):
            return torch.zeros(ids.shape[0], ids.shape[1], V)

    r = mc_conditional_nll(
        Uniform(),
        torch.randint(0, V, (6,)),
        torch.randint(0, V, (10,)),
        MASK,
        num_samples=32,
        batch_size=8,
        generator=torch.Generator().manual_seed(11),
    )
    assert abs(r["nll_per_token"] - math.log(V)) < 1e-4, r


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa
            failed += 1
            import traceback

            print(f"FAIL  {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
