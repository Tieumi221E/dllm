"""Adapter for Transformers-style models that already support denoising.

This module does not convert autoregressive weights into a diffusion model.
It normalizes the call and output boundary for models whose checkpoint and
attention implementation already provide the declared prediction field and
topology.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import torch

from ..models.protocol import (
    DenoiserInput,
    DenoiserOutput,
    ModelCapabilities,
    extract_logits,
)
from ..topology import AttentionTopology

TopologyAdapter = Callable[[AttentionTopology], Mapping[str, Any]]


class AdapterCapabilityError(ValueError):
    """A wrapped model cannot satisfy a requested semantic capability."""


class TransformersDenoiserAdapter:
    """Wrap a Transformers-style model behind the :class:`Denoiser` contract.

    Args:
        model: A callable model with a ``forward`` method and HF-style output.
        prediction_field: ``"same_position"`` for a diffusion denoiser or
            ``"next_token"`` for an autoregressive model.
        default_topology: Dependency structure used when no custom mask is
            passed to the wrapped model.
        attention_topologies: Topologies the wrapped model can implement.
            Defaults to only ``default_topology``.
        topology_adapter: Optional conversion from an ``AttentionTopology`` to
            model-specific forward kwargs. Without it, only
            ``default_topology`` is accepted.
        model_kwargs: Stable extra kwargs included in each model call.

    A next-token adapter may use :meth:`execute` for compatibility checks, but
    :meth:`denoise` rejects it so full-canvas generation cannot silently use
    autoregressive logits as same-position predictions.
    """

    _PREDICTION_FIELDS = frozenset({"same_position", "next_token"})
    _RESERVED = frozenset(
        {
            "input_ids",
            "inputs_embeds",
            "attention_mask",
            "position_ids",
            "use_cache",
            "return_dict",
        }
    )

    def __init__(
        self,
        model: Any,
        *,
        prediction_field: str,
        default_topology: str,
        attention_topologies: Optional[Iterable[str]] = None,
        topology_adapter: Optional[TopologyAdapter] = None,
        model_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if prediction_field not in self._PREDICTION_FIELDS:
            raise ValueError(
                "prediction_field must be 'same_position' or 'next_token'"
            )
        if not default_topology:
            raise ValueError("default_topology must be non-empty")
        supported_topologies = frozenset(
            attention_topologies
            if attention_topologies is not None
            else {default_topology}
        )
        if default_topology not in supported_topologies:
            raise ValueError(
                "attention_topologies must include default_topology"
            )
        if not callable(getattr(model, "forward", None)):
            raise TypeError("model must provide a callable forward method")
        stable_kwargs = dict(model_kwargs or {})
        overlap = self._RESERVED.intersection(stable_kwargs)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"model_kwargs cannot override managed fields: {names}")

        self.model = model
        self.prediction_field = prediction_field
        self.default_topology = default_topology
        self.attention_topologies = supported_topologies
        self.topology_adapter = topology_adapter
        self.model_kwargs = stable_kwargs

        signature = inspect.signature(model.forward)
        self._parameters = signature.parameters
        self._accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in self._parameters.values()
        )
        self.capabilities = ModelCapabilities(
            attention_topologies=supported_topologies,
            explicit_position_ids=self._accepts("position_ids"),
            inputs_embeds=self._accepts("inputs_embeds"),
            prediction_fields=frozenset({prediction_field}),
        )

    def _accepts(self, name: str) -> bool:
        return self._accepts_kwargs or name in self._parameters

    def _set_kwarg(
        self,
        kwargs: Dict[str, Any],
        name: str,
        value: Any,
        *,
        required: bool,
    ) -> None:
        if value is None:
            return
        if not self._accepts(name):
            if required:
                raise AdapterCapabilityError(
                    f"wrapped model does not accept {name!r}"
                )
            return
        kwargs[name] = value

    @staticmethod
    def _attention_mask(request: DenoiserInput) -> Optional[torch.Tensor]:
        mask = request.attention_mask
        if request.topology is None:
            return mask
        valid = request.topology.valid
        if mask is None:
            return valid
        if mask.shape != valid.shape:
            raise ValueError("attention_mask must match topology.valid")
        if mask.device != valid.device:
            raise ValueError("attention_mask and topology must share a device")
        return mask.bool() & valid

    def execute(self, request: DenoiserInput) -> DenoiserOutput:
        """Run the wrapped model without changing prediction alignment."""
        topology_kwargs: Dict[str, Any] = {}
        if request.topology is not None:
            if request.topology.name not in self.attention_topologies:
                raise AdapterCapabilityError(
                    f"wrapped model does not declare topology "
                    f"{request.topology.name!r}"
                )
            if request.topology.name != self.default_topology:
                if self.topology_adapter is None:
                    raise AdapterCapabilityError(
                        f"wrapped model defaults to {self.default_topology!r}, "
                        f"not {request.topology.name!r}"
                    )
                topology_kwargs.update(self.topology_adapter(request.topology))

        managed = self._RESERVED.intersection(request.model_kwargs)
        if managed:
            names = ", ".join(sorted(managed))
            raise ValueError(
                f"request.model_kwargs cannot override managed fields: {names}"
            )
        kwargs = dict(self.model_kwargs)
        kwargs.update(request.model_kwargs)
        for name, value in topology_kwargs.items():
            if name in kwargs:
                raise ValueError(f"duplicate topology kwarg: {name}")
            if not self._accepts(name):
                raise AdapterCapabilityError(
                    f"wrapped model does not accept topology kwarg {name!r}"
                )
            kwargs[name] = value

        self._set_kwarg(
            kwargs, "input_ids", request.input_ids, required=request.input_ids is not None
        )
        self._set_kwarg(
            kwargs,
            "inputs_embeds",
            request.inputs_embeds,
            required=request.inputs_embeds is not None,
        )
        attention_mask = self._attention_mask(request)
        attention_required = bool(
            attention_mask is not None and not attention_mask.bool().all()
        )
        self._set_kwarg(
            kwargs,
            "attention_mask",
            attention_mask,
            required=attention_required,
        )
        self._set_kwarg(
            kwargs,
            "position_ids",
            request.position_ids,
            required=request.position_ids is not None,
        )
        self._set_kwarg(
            kwargs,
            "use_cache",
            True if request.use_cache else None,
            required=request.use_cache,
        )
        self._set_kwarg(kwargs, "return_dict", True, required=False)

        raw = self.model(**kwargs)
        logits = extract_logits(raw)
        cache = getattr(raw, "past_key_values", None)
        hidden_states = getattr(raw, "hidden_states", None)
        return DenoiserOutput(
            logits=logits,
            cache=cache,
            hidden_states=hidden_states,
        )

    def denoise(self, request: DenoiserInput) -> DenoiserOutput:
        """Run a same-position denoiser, rejecting next-token checkpoints."""
        if self.prediction_field != "same_position":
            raise AdapterCapabilityError(
                "denoising requires same-position logits; the wrapped model "
                f"declares {self.prediction_field!r}"
            )
        return self.execute(request)


__all__ = [
    "AdapterCapabilityError",
    "TransformersDenoiserAdapter",
]
