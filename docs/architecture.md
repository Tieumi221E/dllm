# Architecture

`dllm` is an algorithm kernel for discrete diffusion language modeling. It is
not a collection of complete training applications: data acquisition, task
prompts, rewards, experiment tracking, distributed launchers, and benchmark
reporting remain in downstream projects.

The boundary is intentional. A small core makes semantic differences visible
and testable, while adapters can support fast-moving model and serving stacks
without making them mandatory dependencies.

## Layers

| Layer | Stable responsibility | Extension direction |
|---|---|---|
| Process | Time schedules, mask corruption, maskable regions | Structured and non-absorbing corruption |
| Objective | Token losses, MDLM importance weighting, explicit reductions | Block-diffusion and learned-schedule estimators |
| Model | Reference transformer, attention bias, position handling | Protocols for Hugging Face and cache-native models |
| Sampling | Full-canvas and incremental block generation, proposal and commit policies | Stochastic position policies, speculative decoding |
| Execution | Exact block-causal cache and documented approximate prefix cache | Hierarchical and adaptive cache adapters |
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

Several active directions fit the architecture but are not yet stable APIs:

- cache-native AR-to-diffusion conversion and models that switch among causal,
  block-causal, and bidirectional modes;
- adaptive or hierarchical caching and speculative block decoding;
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

### Canvas contract

Training, rollout, and policy scoring must use the same canvas regime:

- `full`: future positions exist as mask tokens and participate in
  bidirectional attention;
- `incremental`: future blocks do not yet exist;
- block-causal attention: future blocks may exist in storage, but earlier
  blocks cannot attend to them.

Changing this contract changes model logits, not just runtime performance.

### Cache contract

A cache implementation must declare whether it is:

- **exact** for the stated attention topology;
- **approximate**, with the changed dependency made explicit;
- a pure execution optimization whose cache-on and cache-off results agree.

`build_kv_cache` plus `forward_block` is exact for block-causal attention.
Full-canvas prefix caching is an approximation because cached prefix states
are not recomputed after the active block changes.

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
  for AR conversion and multi-mode attention;
- [d3LLM](https://github.com/hao-ai-lab/d3LLM) for trajectory distillation;
- [Mask-Aware Policy Gradients](https://arxiv.org/abs/2607.15200) for the
  distinction between token and position-selection actions;
- [ZHZisZZ/dllm](https://github.com/ZHZisZZ/dllm) as a broader recipe-oriented
  toolkit with MDLM, BD3LM, Edit Flows, evaluation, and Trainer integrations.

These repositories are references for interfaces and invariants, not source
templates. Model-specific code remains behind adapters unless it demonstrates
a reusable contract.
