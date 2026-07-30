"""Reference execution backends for framework-neutral sampling loops."""

from __future__ import annotations

import torch

from ..models.transformer import DiffusionTransformer, KVCache
from ..topology import AttentionTopology
from .speculative import SelfSpecStep


class TopologySelfSpecBackend:
    """Self-spec backend for the topology-aware reference Transformer.

    ``draft_shift=True`` aligns a next-token-trained prediction field with
    masked block positions. Set it to ``False`` when the draft mode already
    predicts the token at its own position.
    """

    def __init__(
        self,
        model: DiffusionTransformer,
        *,
        draft_shift: bool = True,
    ) -> None:
        self.model = model
        self.draft_shift = draft_shift

    def causal_prefill(self, prompt_ids: torch.Tensor) -> SelfSpecStep:
        batch, length = prompt_ids.shape
        topology = AttentionTopology.causal(
            batch_size=batch,
            length=length,
            device=prompt_ids.device,
        )
        output = self.model(
            prompt_ids,
            topology=topology,
            return_kvs=True,
            return_dict=True,
        )
        if output.cache is None:
            raise RuntimeError("causal prefill did not return a cache")
        return SelfSpecStep(output.logits, output.cache)

    def draft_logits(self, block_ids: torch.Tensor, cache: KVCache) -> torch.Tensor:
        logits, _ = self.model.forward_block(block_ids, cache)
        if self.draft_shift and logits.shape[1] > 1:
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return logits

    def causal_verify(self, block_ids: torch.Tensor, cache: KVCache) -> SelfSpecStep:
        batch, length = block_ids.shape
        if cache.group_ids is None or cache.length == 0:
            first_group = torch.zeros(batch, dtype=torch.long, device=block_ids.device)
        else:
            first_group = cache.group_ids.max(dim=1).values + 1
        group_ids = first_group.unsqueeze(1) + torch.arange(
            length, dtype=torch.long, device=block_ids.device
        ).unsqueeze(0)
        output = self.model.forward_block(
            block_ids,
            cache,
            group_ids=group_ids,
            return_dict=True,
        )
        if output.cache is None:
            raise RuntimeError("causal verification did not return a cache")
        return SelfSpecStep(output.logits, output.cache)

    def crop_cache(self, cache: KVCache, length: int) -> KVCache:
        return cache.crop(length)


__all__ = ["TopologySelfSpecBackend"]
