# dllm

A self-contained **masked diffusion language model (dLLM) toolkit** -
training, inference, evaluation, and RL utilities in one importable package
(`import dllm`), designed as the diffusion counterpart of a mature
autoregressive training/inference stack.

Every functional default follows an authoritative reference implementation
(see [References](#references)); every documented alternative is an explicit
switch, never a silent behaviour change.

```
dllm/
  schedules.py      noise schedules (linear eps / cosine) with consistent ELBO weights
  masking.py        forward noising process + random-length trick
  loss.py           diffusion_loss (explicit normalizations) + MC likelihood
  data.py           PretrainCollator / SFTCollator / BlockSFTCollator
  topology.py       ordered bidirectional / causal / block-causal dependencies
  execution.py      exactness metadata shared by cache implementations
  models/           topology-aware DiffusionTransformer, denoiser protocol,
                    exact block-causal KV cache, HF wrapper
  sampling/         full-canvas / incremental / self-speculative generation,
                    commit policies, and trajectory recording
  rl.py             trajectory state reconstruction + differentiable PPO primitives
  presets.py        regime presets (small-mha / small-gqa / llada-8b)
docs/architecture.md capability boundaries and admission policy
tests/test_smoke.py focused semantic and numerical tests
```

The long-term package boundary and support tiers are described in
[Architecture](docs/architecture.md). The stable core owns reusable dLLM
mechanisms; model/framework integrations are adapters, while datasets,
task-specific rewards, and complete experiment loops stay downstream.
See [Changelog](CHANGELOG.md) for versioned API changes.

## Install

```bash
pip install -e .          # add [hf] for the transformers wrapper
python tests/test_smoke.py
```

## Quickstart

### Pretraining step

```python
from dllm import PretrainCollator, diffusion_loss

collator = PretrainCollator(mask_token_id=MASK, pad_token_id=PAD,
                            random_length_prob=0.01)      # variable-length trick
batch = collator(list_of_token_id_lists)
logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])
loss = diffusion_loss(logits, batch["clean_ids"], batch["masked_indices"],
                      batch["p_mask"], norm="tokens")     # sum(CE/p) / (B*L)
```

### SFT step (prompt clean, EOS-padding protocol)

```python
from dllm import SFTCollator
collator = SFTCollator(mask_token_id=MASK, eos_token_id=EOS)
batch = collator([{"prompt_ids": p, "response_ids": r}, ...])
logits = model(batch["input_ids"])
loss = diffusion_loss(logits, batch["clean_ids"], batch["masked_indices"],
                      batch["p_mask"], norm="answer", maskable=batch["maskable"])
```

### Semi-AR block SFT

```python
from dllm import BlockSFTCollator
collator = BlockSFTCollator(MASK, EOS, PAD, block_length=32,
                            canvas="truncated")   # or "full"
batch = collator(samples)
loss = diffusion_loss(model(batch["input_ids"]), batch["clean_ids"],
                      batch["masked_indices"], batch["p_mask"],
                      norm="answer", maskable=batch["maskable"])   # bound-consistent
# uniform weighting variant: diffusion_loss(..., importance_weight=False,
#                                            norm="masked")
```

To make the truncated regime exactly KV-cache-consistent at inference, train
with an ordered block topology:

```python
from dllm import AttentionTopology

topology = AttentionTopology.from_boundaries(
    [prompt_length, prompt_length + 32, prompt_length + 64],
    length=batch["input_ids"].shape[1],
    batch_size=batch["input_ids"].shape[0],
    device=batch["input_ids"].device,
)
logits = model(batch["input_ids"], topology=topology)
```

`dllm.sampling.block_causal_bias` remains as a compatibility wrapper for
training code that expects a raw additive mask.

### Attention and model contracts

```python
from dllm import AttentionTopology, DenoiserInput

# Same model, different dependency structure. Tokens inside one group are
# bidirectional; a query in group g sees keys in groups <= g.
topology = AttentionTopology.block_causal(
    block_size=32, batch_size=input_ids.shape[0],
    length=input_ids.shape[1], device=input_ids.device,
)
position_ids = logical_positions  # independent of padded tensor columns
logits = model(input_ids, position_ids=position_ids, topology=topology)

# Framework-neutral structured output and capability discovery.
out = model.denoise(DenoiserInput(
    input_ids=input_ids,
    position_ids=position_ids,
    topology=topology,
    use_cache=True,
))
print(model.capabilities, out.logits.shape, out.cache.semantics)
```

`AttentionTopology.bidirectional`, `.causal`, `.block_causal`, and
`.from_boundaries` compile to the same low-level SDPA mask contract.
`forward` still returns a logits tensor by default, and `return_kvs=True`
still returns the 1.1 `(logits, kvs)` tuple.

### Sampling

```python
from dllm import CanvasConfig, generate_canvas

out = generate_canvas(model, prompt_ids, MASK, CanvasConfig(
    gen_length=256, block_length=32, steps=128,
    temperature=0.0, commit="transfer",     # top-k quota commit
    eos_token_id=EOS))
print(out.responses[0])                      # EOS-stripped token ids
```

Fast paths: `CanvasConfig(prefix_cache=True, further_horizon=64)` (windowed
prefix cache), `commit="threshold", threshold=0.9` (confidence-threshold
parallel decoding), or `generate_blockwise(...)` for truncated-canvas models
(exact KV cache, per-sample EOS early exit).

Quota and threshold are built-in `CommitPolicy` implementations rather than
sampler branches. `commit="transfer"` and `commit="threshold"` remain stable
shorthands; a custom policy can return a `CommitDecision` with an optional
position-selection log-probability for trajectory learning.

### Linear self-speculation

Hybrid checkpoints that support causal next-token prediction and
bidirectional within-block drafting can use the same weights for both:

```python
from dllm import (
    SelfSpecConfig,
    TopologySelfSpecBackend,
    generate_self_speculative,
)

backend = TopologySelfSpecBackend(model, draft_shift=True)
out = generate_self_speculative(
    backend,
    prompt_ids,
    MASK,
    SelfSpecConfig(max_new_tokens=256, block_length=32),
)
print(out.sequences[0], out.stats.draft_acceptance)
```

The default is a one-pass diffusion draft
(`commit="threshold", threshold=0.0`). Every block is then checked causally;
only the longest matching prefix plus one verifier token is accepted. The
emitted sequence therefore equals the backend's greedy causal sequence under
the same token-suppression and stopping rules. Model-specific attention
switches, LoRA routing, and framework cache types belong in a
`SelfSpecBackend`, not in the orchestration loop. The reference backend
currently requires batch size 1 because exact variable-length acceptance
needs a ragged or paged cache for batching.

### Likelihood evaluation

```python
from dllm import mc_conditional_nll
r = mc_conditional_nll(lambda ids: model(ids), prompt, response, MASK,
                       num_samples=128)
print(r["nll_per_token"])
```

### RL log-probs (trajectory decomposition)

```python
from dllm import trajectory_logprobs
out = generate_blockwise(model, prompt, MASK, cfg, num_samples=8)
steps = trajectory_logprobs(lambda ids: model(ids), prompt, out.canvas[0, Lp:],
                            out.step_map[0], MASK, block_length=32,
                            canvas="incremental")   # MUST match the rollout regime
```

Pass `with_grad=True` when the returned log-probabilities are part of a policy
objective. `trajectory_states(...)` exposes the model-free reconstruction
separately, and `ppo_clip_objective(...)` provides the token-level clipped PPO
term without taking ownership of backward or optimizer steps.

### Trajectory data for constrained-order training

```python
out = generate_canvas(model, prompt, MASK,
                      CanvasConfig(..., temperature=0.6, record_trace=True,
                                   trace_topk=8))
traj = out.traces[0]
traj.content_logprob_mean(EOS)   # ELBO-proxy selection score
traj.step_map                    # commit order -> (x_t, x_0) training pairs
traj.summary(EOS)                # rollout progress/confidence/log-prob stats
```

## Switch reference

| Knob | Default (authoritative) | Alternatives |
|---|---|---|
| loss norm | `"tokens"` (pretrain), `"answer"` (SFT) | `"maskable"`; `"masked"` (uniform weighting only - the biased 1/p combination is rejected) |
| loss weight | `1/p` (bound-consistent) | `importance_weight=False` (uniform block SFT) |
| schedule | `LinearSchedule(eps=1e-3)` | `CosineSchedule` (weight derived consistently) |
| t sampling | uniform, always | - (non-uniform t without re-weighting is not a likelihood bound; use a schedule instead) |
| min_one_mask | off | on (tiny-batch efficiency; slight bias) |
| EOS in SFT | maskable, learnable, counted | `non_maskable_ids` (EOS-collapse mitigation for tiny models) |
| commit | `"transfer"` (top-k quota) | `"threshold"` (parallel decoding), custom `CommitPolicy` |
| temperature | `"gumbel"` fp64 | `"multinomial"` = softmax(logits/T) (different distribution at T!=1) |
| confidence | `"prob"` raw-softmax | `"margin"`, `"neg_entropy"`, `"random"` |
| canvas | full canvas (`generate_canvas`) | incremental (`generate_blockwise`) - truncated-canvas regime |
| blockwise attention | block-causal (KV cache exact; `use_cache` is speed-only) | - |
| topology | bidirectional ordered group 0 | causal, fixed block, arbitrary ordered boundaries, raw mask escape hatch |
| positions | `"learned"` | `"rope"` (recommended for new training) |
| attention | GQA (`num_kv_heads`) or MHA; optional `attn_bias`/`ff_bias` | |

## Design notes (pitfalls this API prevents)

- **Normalization**: the 1/p-importance-weighted sum must be normalized by a
  *fixed* denominator (total tokens, or per-sample answer length), never by
  the realized mask count - and never both per-sample-count-divided *and*
  1/p-weighted (that up-weights low-noise samples by ~1/t).
- **Canvas consistency**: a bidirectional model's logits depend on whether
  future [MASK] tokens are present. Training collator, sampler, and RL
  log-prob reconstruction must share one canvas regime; the API names it
  explicitly everywhere (`canvas="truncated" | "full"`).
- **Cache honesty**: `generate_blockwise`'s cache is exact (block-causal by
  construction, tested against an explicit staircase mask);
  `generate_canvas(prefix_cache=True)` is a windowed approximation of the
  fully bidirectional canvas. Cache objects carry this exact/approximate
  provenance instead of hiding a semantic change behind a speed flag.
- **Dependency vs layout**: logical positions, padding, and attention groups
  are separate tensors. Left padding or a future edit process therefore does
  not need to pretend that physical columns are semantic positions.
- **No silent failures**: position overflow raises instead of clamping;
  padded prompts carry their mask through the KV cache; the mask token is
  never predictable unless explicitly allowed.
- **Draft/verify exactness**: self-speculative drafts may be stochastic, but
  accepted tokens are causal-verifier tokens. Cache cropping discards every
  rejected suffix instead of allowing a draft to leak into later steps.

## Verification

`python tests/test_smoke.py` - focused tests including numeric equality with the
reference loss implementation (pretraining + SFT normalizations), ordered
topology == explicit attention masks, multi-block KV-cache == full topology
recomputation (learned & RoPE), padding invariance with explicit positions,
cache-on/off generation equality, trajectory serialization and differentiable
policy scoring, custom commit-policy actions, self-speculation equality with
greedy causal decoding, and exact recovery of log V by the MC likelihood
estimator on a uniform-logits model.

## References

The defaults of this toolkit follow these works; the mapping is noted in the
switch table above.

- **LLaDA** - Nie et al., *Large Language Diffusion Models*,
  [arXiv:2502.09992](https://arxiv.org/abs/2502.09992);
  [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA).
  Training objective and normalizations (Eq. 3/5), MC likelihood estimator
  (Eq. 6), transfer-top-k reverse process, Gumbel-max sampling, EOS protocol,
  random-length pretraining trick.
- **TraceRL / dLLM-RL** - Wang et al.,
  [arXiv:2509.06949](https://arxiv.org/abs/2509.06949);
  [Gen-Verse/dLLM-RL](https://github.com/Gen-Verse/dLLM-RL).
  Trajectory-decomposed RL log-probabilities, windowed prefix cache,
  margin/neg-entropy confidence, full-canvas block-SFT construction.
- **Fast-dLLM** - Wu et al.,
  [arXiv:2505.22618](https://arxiv.org/abs/2505.22618);
  [NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM).
  Confidence-threshold parallel decoding; block-level caching.
- **Nemotron-Labs-Diffusion** -
  [NVlabs/Nemotron-Labs-Diffusion](https://github.com/NVlabs/Nemotron-Labs-Diffusion).
  Single-model causal/diffusion mode switching and linear self-speculation.
- **dllm** - [ZHZisZZ/dllm](https://github.com/ZHZisZZ/dllm).
  Schedule/weight formulation (w(t) = -alpha'/(1-alpha)) and normalization options.
- Low-precision Gumbel-max degradation (float64 rationale):
  [arXiv:2409.02908](https://arxiv.org/abs/2409.02908).

## License

MIT. See [LICENSE](LICENSE).
