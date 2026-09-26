"""Checkpoint integration presets: internal consistency.

The token ids themselves were read from the checkpoints' own files (see the
comment above ``_checkpoint`` in ``dllm/presets.py``); these tests check that
every preset is usable as metadata: a known prediction field and topology,
a block length for block topologies, a mask id distinct from EOS and pad,
and requirements that name what the checkpoint needs.
"""

from __future__ import annotations

from dllm import PREDICTION_FIELDS, get_preset, get_preset_info, list_presets

CHECKPOINTS = {
    "integration/dream-v0-instruct-7b": ("shifted", 151666),
    "integration/dream-coder-v0-instruct-7b": ("shifted", 151666),
    "integration/diffucoder-7b-instruct": ("shifted", 151666),
    "integration/illada-8b-instruct": ("same_position", 5),
    "integration/fast-dllm-v2-7b": ("shifted", 151665),
    "integration/sdar-8b-chat": ("same_position", 151669),
    "integration/trado-8b-instruct": ("same_position", 151669),
    "integration/nemotron-labs-diffusion-8b": ("same_position", 100),
}


def test_all_checkpoint_presets_registered():
    names = set(list_presets(category="integration"))
    assert set(CHECKPOINTS) <= names
    assert "integration/llada-8b" in names  # the pre-existing preset stays


def test_checkpoint_presets_are_consistent():
    for name, (field, mask) in CHECKPOINTS.items():
        cfg = get_preset(name)
        info = get_preset_info(name)
        assert cfg["prediction_field"] == field in PREDICTION_FIELDS, name
        assert cfg["mask_token_id"] == mask, name
        assert cfg["mask_token_id"] not in (cfg["eos_token_id"], cfg.get("pad_token_id")), name
        assert cfg["topology"] in ("bidirectional", "block_causal"), name
        if cfg["topology"] == "block_causal":
            assert cfg["block_length"] > 0, name
        else:
            assert "block_length" not in cfg, name
        assert {field, cfg["topology"], "transformers"} <= info.requires, name
        assert cfg["hf_model"] and cfg["revision"] and cfg["license"], name
        assert info.reference and info.reference.startswith("https://"), name


def test_dream_family_points_to_timestep_decoding():
    for name, (field, _) in CHECKPOINTS.items():
        cfg = get_preset(name)
        if cfg["hf_model"].startswith(("Dream-org/", "apple/DiffuCoder")):
            assert cfg["commit_policy"] == "timestep_quota", name


def test_timestep_quota_shorthand_resolves():
    from dllm import TimestepQuotaCommitPolicy
    from dllm.sampling import resolve_commit_policy

    assert isinstance(resolve_commit_policy("timestep_quota"), TimestepQuotaCommitPolicy)
