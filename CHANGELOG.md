# Changelog

## 1.3.1

Version 1.3.1 consolidates the 1.3 generation architecture without adding a
new decoding method.

### Added

- `BlockCacheDenoiser` and `PrefixCacheDenoiser` structural protocols for
  cache-capable sampler integrations.
- `DiffusionTransformer.build_approximate_prefix_cache(...)`, which owns the
  windowed full-canvas approximation and its provenance.
- `EXACT_ORDERED`, the precise cache semantic for causal, fixed-block, and
  arbitrary ordered-prefix extension.

### Changed

- Canvas and blockwise cache paths now depend on structural protocols instead
  of checking for the reference Transformer class.
- The reference `TopologySelfSpecBackend` moved to
  `dllm.sampling.backends`; the framework-neutral state machine no longer
  imports model code.
- Quota and threshold policies now reuse the public selector primitives with
  an explicit candidate mask; suppressed `-inf` scores no longer duplicate
  selection logic or cause ambiguous membership.
- Exact caches reject extensions whose valid groups do not strictly follow
  the cached groups, preventing an invalid speed path from being mislabeled
  as exact.
- The root wildcard API now favors common workflows. Advanced extension
  records and resolver helpers remain available from `dllm.sampling`, and
  their existing direct root attributes remain import-compatible.

### Compatibility

- Existing high-level imports and the legacy two-argument selector calls are
  unchanged.
- `EXACT_BLOCK_CAUSAL` remains available for callers naming that specific
  regime.
- Model state dictionaries and training, generation, and RL tensor outputs
  are unchanged; cache metadata now names the broader ordered invariant.

## 1.3.0

Version 1.3 separates generation decisions from sampler control flow and adds
a model-independent linear self-speculation state machine.

### Added

- `CommitPolicy`, `CommitState`, and `CommitDecision` contracts.
- `QuotaCommitPolicy` and `ThresholdCommitPolicy` built-in strategies.
- Optional position-selection action log-probabilities from custom policies
  into recorded trajectories.
- `SelfSpecBackend` and `generate_self_speculative` for causal
  seed/draft/verify/accept orchestration.
- `TopologySelfSpecBackend`, a reference backend for the topology-aware
  `DiffusionTransformer`.
- Non-mutating `KVCache.crop(...)` with provenance preservation.

### Changed

- Full-canvas and incremental samplers now share the same commit-policy path.
- `CanvasConfig.commit` and `BlockwiseConfig.commit` accept custom policy
  objects in addition to string shorthands.
- Invalid zero-step configurations and policies that stall or select
  non-candidate positions now raise explicit errors.

### Compatibility

- `"transfer"` and `"threshold"` retain their 1.2 behavior and defaults.
- Existing sampler outputs, model state dictionaries, and downstream import
  paths are unchanged.
- Self-speculation is opt-in and adds no framework dependency.

## 1.2.0

Version 1.2 makes dependency structure and model execution explicit without
changing the default 1.1 forward or sampling APIs.

### Added

- `AttentionTopology`, an ordered-group representation covering
  bidirectional, causal, block-causal, prefix, and custom segmented attention.
- Explicit `position_ids` and `inputs_embeds` support in the reference
  transformer.
- Framework-neutral `DenoiserInput`, `DenoiserOutput`, `Denoiser`, and
  `ModelCapabilities` contracts.
- `CacheSemantics` plus position, validity, and topology provenance in
  `KVCache`.
- Structured `denoise(...)` and `return_dict=True` execution paths.

### Changed

- Full, causal, and block-causal execution now share one topology compiler.
- `build_kv_cache` and `forward_block` preserve logical positions and ordered
  groups across committed blocks.
- The full-canvas prefix cache declares its approximation explicitly.
- The Hugging Face wrapper accepts explicit positions and attention topology.

### Compatibility

- `model(input_ids)` still returns a logits tensor.
- `model(..., return_kvs=True)` and `forward_block(...)` still return their
  existing tuples.
- `block_causal_bias(...)` remains available as a raw-mask compatibility
  wrapper.
- Existing state-dict parameter names and model configuration fields are
  unchanged.

## 1.1.0

- Added portable trajectory records with compact top-k predictive
  distributions and selection-action log-probabilities.
- Separated model-free trajectory state reconstruction from differentiable
  policy scoring.
- Added composable masked cross-entropy reductions and a token-level clipped
  PPO primitive.

## 1.0.0

- Initial self-contained toolkit for masked diffusion training, full-canvas
  and blockwise sampling, likelihood evaluation, trajectory scoring, and a
  reference Transformer with exact block-causal caching.
