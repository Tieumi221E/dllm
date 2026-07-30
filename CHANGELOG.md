# Changelog

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
