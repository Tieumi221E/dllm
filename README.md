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
  models/           DiffusionTransformer (MHA|GQA x learned|RoPE, exact KV cache),
                    HF wrapper
  sampling/         generate_canvas (full canvas) / generate_blockwise
                    (incremental block decoding) / trajectory recording
  rl.py             trajectory_logprobs (canvas-consistent reconstruction)
  presets.py        regime presets (small-mha / small-gqa / llada-8b)
tests/test_smoke.py 19 tests incl. numeric equality with the reference loss code
```

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
loss = diffusion_loss(model(batch["input_ids"]).logits, batch["clean_ids"],
                      batch["masked_indices"], batch["p_mask"],
                      norm="answer", maskable=batch["maskable"])   # bound-consistent
# uniform weighting variant: diffusion_loss(..., importance_weight=False,
#                                            norm="masked")
```

To make the truncated regime exactly KV-cache-consistent at inference, train
with the staircase attention bias: `from dllm.sampling import block_causal_bias`.

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

### Likelihood evaluation

```python
from dllm import mc_conditional_nll
r = mc_conditional_nll(lambda ids: model(ids).logits, prompt, response, MASK,
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

### Trajectory data for constrained-order training

```python
out = generate_canvas(model, prompt, MASK,
                      CanvasConfig(..., temperature=0.6, record_trace=True))
traj = out.traces[0]
traj.content_logprob_mean(EOS)   # ELBO-proxy selection score
traj.step_map                    # commit order -> (x_t, x_0) training pairs
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
| commit | `"transfer"` (top-k quota) | `"threshold"` (parallel decoding) |
| temperature | `"gumbel"` fp64 | `"multinomial"` = softmax(logits/T) (different distribution at T!=1) |
| confidence | `"prob"` raw-softmax | `"margin"`, `"neg_entropy"`, `"random"` |
| canvas | full canvas (`generate_canvas`) | incremental (`generate_blockwise`) - truncated-canvas regime |
| blockwise attention | block-causal (KV cache exact; `use_cache` is speed-only) | - |
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
  fully bidirectional canvas - documented as such.
- **No silent failures**: position overflow raises instead of clamping;
  padded prompts carry their mask through the KV cache; the mask token is
  never predictable unless explicitly allowed.

## Verification

`python tests/test_smoke.py` - 19 tests, including numeric equality with the
reference loss implementation (pretraining + SFT normalizations), KV-cache ==
explicit block-causal forward (learned & RoPE), cache-on/off generation
equality, and exact recovery of log V by the MC likelihood estimator on a
uniform-logits model.

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
- **dllm** - [ZHZisZZ/dllm](https://github.com/ZHZisZZ/dllm).
  Schedule/weight formulation (w(t) = -alpha'/(1-alpha)) and normalization options.
- Low-precision Gumbel-max degradation (float64 rationale):
  [arXiv:2409.02908](https://arxiv.org/abs/2409.02908).

## License

MIT. See [LICENSE](LICENSE).
