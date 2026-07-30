"""Preset discovery, composition, and compatibility tests."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dllm import (
    compose_presets,
    get_preset,
    get_preset_info,
    list_presets,
)


def test_legacy_presets_remain_complete():
    small_mha = get_preset("small-mha")
    assert small_mha["model"]["num_kv_heads"] == 12
    assert small_mha["canvas"]["temperature"] == 1.0
    assert small_mha["loss"] == {"norm": "tokens"}

    small_gqa = get_preset("small-gqa")
    assert small_gqa["model"]["num_kv_heads"] == 2
    assert small_gqa["block_sft"] == {"canvas": "truncated"}
    assert small_gqa["blockwise"]["gen_length"] == 384

    llada = get_preset("llada-8b")
    assert llada["mask_token_id"] == 126336
    assert llada["canvas_trajectory"]["record_trace"]


def test_presets_are_isolated_and_discoverable():
    first = get_preset("model/ref-small-gqa-rope")
    first["model"]["hidden_size"] = 1
    second = get_preset("model/ref-small-gqa-rope")
    assert second["model"]["hidden_size"] == 768

    recipes = list_presets(category="recipe")
    assert "recipe/full-threshold" in recipes
    assert "recipe/blockwise-exact" in recipes
    assert "small-gqa" not in recipes
    assert set(list_presets(category="legacy")) == {
        "small-gqa",
        "small-mha",
        "llada-8b",
    }

    info = get_preset_info("recipe/full-threshold")
    assert info.category == "recipe"
    assert {"same_position", "bidirectional"} <= info.requires


def test_orthogonal_presets_compose_without_silent_conflicts():
    combined = compose_presets(
        "model/ref-small-gqa-rope",
        "recipe/blockwise-exact",
    )
    assert combined["model"]["position_embedding"] == "rope"
    assert combined["blockwise"]["block_length"] == 32

    try:
        compose_presets("recipe/full-transfer", "recipe/full-threshold")
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting recipes must fail")

    overridden = compose_presets(
        "recipe/full-threshold",
        overrides={"canvas": {"threshold": 0.8, "gen_length": 64}},
    )
    assert overridden["canvas"]["threshold"] == 0.8
    assert overridden["canvas"]["gen_length"] == 64
    assert overridden["canvas"]["commit"] == "threshold"


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
