# Architecture

`dllm` is an algorithm kernel for discrete diffusion language modeling. It is
not a collection of complete training applications: data acquisition, task
prompts, rewards, experiment tracking, distributed launchers, and benchmark
reporting remain in downstream projects.

The boundary is intentional. A small core makes semantic differences visible
and testable, while adapters can support fast-moving model and serving stacks
without making them mandatory dependencies.

The resulting relationship is an extension layer over the established
PyTorch/Transformers ecosystem. Frameworks continue to own pretrained model
loading, distributed optimization, device placement, quantization, and
production serving; `dllm` owns diffusion-specific objectives, topology,
state transitions, cache provenance, and generation policies.

## Layers

| Layer | Stable responsibility | Extension direction |
|---|---|---|
| Process | Time schedules, mask corruption, maskable regions | Structured and non-absorbing corruption |
| Objective | Token losses, MDLM importance weighting, explicit reductions | Block-diffusion and learned-schedule estimators |
| Model | Denoiser and cache protocols, prediction field, explicit positions, reference transformer | Checkpoint and framework adapters |
| Topology | Ordered attention groups independent of tensor layout | Sparse, edit-aware, and multimodal dependency compilers |
| Sampling | Full-canvas, incremental, and linear self-speculative generation; explicit proposal and commit policies | Learned position policies and batched speculation |
| Execution | Exact ordered-prefix cache and documented approximate prefix cache | Hierarchical and adaptive cache adapters |
| Trajectory | Token actions, commit order, optional predictive distributions | Position-action probabilities and distillation targets |
| Post-training | State reconstruction and differentiable PPO primitives | GRPO/DPO adapters without task-specific rewards |
| Evaluation | Monte Carlo conditional NLL | Common generation and likelihood harness adapters |

The axes are independent. For example, confidence-threshold decoding is a
commit policy, not a model family; block-causal attention is a topology, not a
cache flag; and a cache implementation must state which topology it preserves
exactly.

## Regimes

The package currently treats these regimes explicitly:

- **Absorbing-mask diffusion** (MDLM, LLaDA, Dream): bidirectional denoising on
  a fixed canvas.
- **Semi-autoregressive decoding**: blocks are opened left-to-right and
  denoised in parallel within a block.
- **Block-causal diffusion** (BD3LM-style): prefix blocks cannot attend to the
  active block, which permits exact inter-block KV caching.
- **Linear self-speculation**: one model drafts with block-bidirectional
  attention and verifies causally, preserving the verifier's greedy sequence.

Several active directions fit the architecture but are not yet stable APIs:

- cache-native AR-to-diffusion conversion and models that switch among causal,
  block-causal, and bidirectional modes;
- adaptive or hierarchical caching and paged/ragged speculative batching;
- trajectory distillation and learned parallel-decoding policies;
- edit processes with insertion and deletion;
- multimodal block diffusion.

These belong in the stable core only after their state transition and
attention semantics can be expressed independently of one checkpoint or
training framework.

## Contracts

### Objective contract

`diffusion_loss` is the high-level, bound-aware MDLM objective. It rejects a
realized-mask denominator combined with `1/p` weighting because that changes
the estimator.

`masked_cross_entropy` is the low-level escape hatch. It independently
controls token weights, sample weights, and reduction, so constrained
trajectory training and legacy objectives can share the implementation
without being mislabeled as an ELBO.

### Model contract

A denoiser consumes token IDs or embeddings, logical position IDs, an
attention topology, and optional execution state. It returns a prediction
field (`logits`) plus optional cache and auxiliary data. It does not own the
noise schedule, commit policy, EOS handling, reward, or optimizer.

`DenoiserInput`, `DenoiserOutput`, and `ModelCapabilities` are the
framework-neutral boundary. The reference `DiffusionTransformer` implements
that boundary while preserving its tensor-returning 1.1 `forward` API.
Adapters may expose framework-native outputs as long as `extract_logits`
normalizes the prediction field.

Fast samplers depend on structural protocols rather than the reference model
class. `BlockCacheDenoiser` defines exact ordered-prefix cache construction
and block extension; `PrefixCacheDenoiser` additionally defines explicit
approximate prefix construction for a changing bidirectional canvas.

### Framework-adapter contract

An adapter must declare the model's prediction field and default attention
topology. Same-position prediction may enter full-canvas denoising;
next-token prediction may enter the generic execution boundary and causal
verification, but is rejected by diffusion canvas sampling.

The generic Transformers adapter maps supported inputs and normalizes logits,
cache, hidden states, and auxiliary outputs. It does not modify weights,
rewrite attention, or claim framework-native caches are exact. Unsupported
logical positions, topology changes, or cache operations fail unless a
model-specific hook implements them.

Framework cache compatibility is more than accepting a `past_key_values`
argument. Exact adapters must preserve padding validity, logical cache
positions, ordered groups, crop semantics, and exactness provenance.

### Topology contract

`AttentionTopology` is an ordered partition. A query in group `g` attends to
keys in groups `<= g`, while tokens in one group remain bidirectional. This
single invariant expresses:

- bidirectional diffusion (all tokens in group 0);
- causal attention (one increasing group per token);
- block-causal attention (one increasing group per block);
- prefix-LM and arbitrary ordered segments (caller-supplied groups).

Padding is composed through `valid`; `position_ids` are independent logical
coordinates. Raw `attn_bias` remains a low-level escape hatch, but cannot be
combined with a topology because doing so would make dependency provenance
ambiguous.

### Canvas contract

Training, rollout, and policy scoring must use the same canvas regime:

- `full`: future positions exist as mask tokens and participate in
  bidirectional attention;
- `incremental`: future blocks do not yet exist;
- block-causal attention: future blocks may exist in storage, but earlier
  blocks cannot attend to them.

Changing this contract changes model logits, not just runtime performance.

### Commit-policy contract

Candidate-token sampling, confidence estimation, and position selection are
separate operations. A `CommitPolicy` receives confidence, the current
candidate mask, the block's initial mask, and step metadata. It returns a
boolean commit mask plus an optional per-sample selection-action
log-probability.

Every policy must select only current candidates and make progress for each
active sample. The sampler validates both invariants. `QuotaCommitPolicy` and
`ThresholdCommitPolicy` implement the established transfer and confidence
threshold rules; custom stochastic or learned policies use the same boundary
without entering sampler control flow.

### Cache contract

A cache implementation must declare whether it is:

- **exact** for the stated attention topology;
- **approximate**, with the changed dependency made explicit;
- a pure execution optimization whose cache-on and cache-off results agree.

`build_kv_cache` plus `forward_block` is exact for ordered dependencies in
which cached groups precede the active group. Block-causal decoding is the
standard use of that invariant. Full-canvas prefix caching is an
approximation because cached prefix states are not recomputed after the
active block changes. `build_approximate_prefix_cache` makes that model-owned
choice explicit. `KVCache.semantics` declares the provenance, while cached key
validity, positions, and ordered group IDs travel with the tensors.

### Preset contract

Presets are organized along independent `model`, `recipe`, and `integration`
axes. Composition recursively merges non-conflicting fields and rejects
conflicts unless the caller supplies an explicit override. Each preset carries
category, stability, and requirement metadata so examples do not silently
become support claims. Legacy complete presets remain unchanged.

### Self-speculation contract

`generate_self_speculative` owns only the model-independent state machine:
causal seed, diffusion draft, causal verification, longest matching prefix,
one verifier token, and rejected-cache cropping. A `SelfSpecBackend` owns
attention-mode switching, prediction-field alignment, optional adapter
routing, and framework cache operations.

The reference topology backend lives with other sampling backends, separate
from the framework-neutral orchestration state machine. Third-party backends
therefore need not inherit from or import the reference transformer.

The accepted output is the backend's greedy causal sequence under identical
token suppression and stopping. Draft randomness may change acceptance and
NFE but not emitted tokens. The reference topology backend is deliberately
batch-1 until variable accepted lengths can be represented without padding
away cache semantics.

### Trajectory contract

A trajectory records the state before each transition, committed token
actions, token log-probabilities, and optional compact distributions.
Selection-action log-probability is separate from token log-probability. This
distinction keeps the schema usable for methods that learn which positions to
unmask, while deterministic top-k policies may leave it unset.

Token IDs are the storage format. Token strings and task metadata are boundary
concerns handled by applications.

## Admission policy

A new method enters the stable package when it satisfies all of the following:

1. Its reusable mechanism is separable from a particular dataset, reward, or
   launcher.
2. Its semantics can be stated as an invariant and covered by a focused test.
3. It has an authoritative specification or implementation that can be
   compared without copying project-specific code.
4. Optional framework dependencies stay behind an adapter or extra.
5. The public API names approximation, bias, and incompatibility instead of
   hiding them behind a speed flag.

Otherwise it remains an application implementation or an experimental
adapter. This keeps the package open to new dLLM directions without turning
the stable surface into a paper-by-paper registry.

## Landscape references

The architecture is checked against several distinct implementation lines:

- [LLaDA](https://github.com/ML-GSAI/LLaDA) and
  [Dream](https://github.com/DreamLM/Dream) for full-canvas masked diffusion;
- [BD3LM](https://github.com/kuleshov-group/bd3lms) and
  [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) for block diffusion,
  cache-compatible attention, and parallel decoding;
- [LLaDA 2.0](https://github.com/inclusionAI/LLaDA2.0) and
  [Nemotron-Labs-Diffusion](https://github.com/NVlabs/Nemotron-Labs-Diffusion)
  for AR conversion, multi-mode attention, and single-model
  self-speculation;
- [d3LLM](https://github.com/hao-ai-lab/d3LLM) for trajectory distillation;
- [Mask-Aware Policy Gradients](https://arxiv.org/abs/2607.15200) for the
  distinction between token and position-selection actions;
- [ZHZisZZ/dllm](https://github.com/ZHZisZZ/dllm) as a broader recipe-oriented
  toolkit with MDLM, BD3LM, Edit Flows, evaluation, and Trainer integrations.

These repositories are references for interfaces and invariants, not source
templates. Model-specific code remains behind adapters unless it demonstrates
a reusable contract.
