"""Opt-in access to Dream's reference code for conformance tests.

Set ``DLLM_DREAM_REFERENCE_DIR`` to a directory holding unmodified copies of:

- ``generation_utils.py`` from a Dream-v0 checkpoint on the Hugging Face Hub
  (https://huggingface.co/Dream-org/Dream-v0-Instruct-7B);
- ``gen_utils.py`` (``src/diffllm/``) and ``fsdp_sft_trainer.py``
  (``src/trainer/``) from https://github.com/DreamLM/Dream.

The files are not vendored. Tests that need them skip when they are absent.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import os

import pytest
import torch

ENV = "DLLM_DREAM_REFERENCE_DIR"


def reference_path(filename: str) -> str:
    root = os.environ.get(ENV)
    path = os.path.join(root, filename) if root else ""
    if not path or not os.path.exists(path):
        pytest.skip(f"set {ENV} to a directory containing Dream's {filename}")
    return path


def load_module(filename: str):
    path = reference_path(filename)
    spec = importlib.util.spec_from_file_location(
        "dream_ref_" + filename.replace(".py", ""), path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_function(filename: str, name: str):
    """Extract one top-level function from a file whose imports are not
    installable here (the trainer imports verl), and execute only it."""
    path = reference_path(filename)
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            np = pytest.importorskip("numpy")  # the reference code uses numpy
            namespace = {"np": np, "torch": torch, "math": math}
            code = compile(ast.Module(body=[node], type_ignores=[]), path, "exec")
            exec(code, namespace)
            return namespace[name]
    raise LookupError(f"{name} not found in {filename}")
