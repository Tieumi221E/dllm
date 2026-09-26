"""Collators: pretraining, SFT, and block-wise semi-AR SFT.

Each collator returns clean tensors plus the masking artefacts, ready for
:func:`dllm.loss.diffusion_loss`:

    batch = collator(samples)
    logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])
    loss = diffusion_loss(logits, batch["clean_ids"], batch["masked_indices"],
                          batch["p_mask"], norm=..., maskable=batch["maskable"])

SFT pads short pairs with EOS as part of the response (maskable, attended,
counted in ``answer_lengths``) so the model learns length control. Block SFT
uses ``norm="answer"`` for bound-consistent weighting, or
``importance_weight=False, norm="masked"`` for uniform weighting.

``SFTCollator`` also returns ``content`` (the real response plus its one
terminal EOS, without padding) for ``norm="content"``, and can pad every
response to a fixed ``response_canvas`` so training matches a fixed
generation length at inference. Its padding switches:

- ``pad_mode="eos"`` (default): padding is EOS, maskable and in the loss
  (LLaDA, Dream); ``eos_as_one=True`` masks each trailing EOS run as one unit
  (Dream's ``treat_eos_as_one``);
- ``pad_mode="mask_ignored"``: padding is the mask token, attended but
  neither maskable nor in the loss (Fast-dLLM v2);
- ``pad_mode="dedicated"``: padding cycles through ``pad_token_ids``,
  maskable and in the loss; one id is VoidPadding's ``[VOID]``, several ids
  are Rainbow Padding. Forbid those ids at inference with
  ``suppress_token_ids``;
- ``batch_cutoff=True``: per batch, one sample's response length is drawn
  as the common response length; longer responses are cut, shorter ones
  padded (Dream's ``perbatch_cutoff_type="random_with_input_pad"``).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Set, Union

import torch

from .masking import forward_process, random_truncate
from .schedules import NoiseSchedule, get_schedule


def _pad_stack(seqs: List[List[int]], pad_id: int, max_length: Optional[int] = None):
    if max_length is not None:
        seqs = [s[:max_length] for s in seqs]
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    attn = torch.zeros(len(seqs), width, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, : len(s)] = 1
    return ids, attn


class PretrainCollator:
    """Masked-diffusion pretraining batches from token-id lists."""

    def __init__(
        self,
        mask_token_id: int,
        pad_token_id: int,
        max_length: Optional[int] = None,
        schedule: Union[str, NoiseSchedule, None] = None,
        random_length_prob: float = 0.0,  # 0.01 recommended for pretraining
        non_maskable_ids: Optional[Set[int]] = None,
        min_one_mask: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.schedule = get_schedule(schedule)
        self.random_length_prob = random_length_prob
        self.non_maskable_ids = non_maskable_ids or set()
        self.min_one_mask = min_one_mask
        self.generator = generator

    def __call__(self, batch: List[List[int]]) -> Dict[str, torch.Tensor]:
        ids, attn = _pad_stack(batch, self.pad_token_id, self.max_length)
        ids = random_truncate(ids, self.random_length_prob, generator=self.generator)
        attn = attn[:, : ids.shape[1]]

        maskable = attn.bool()
        if self.non_maskable_ids:
            for tok in self.non_maskable_ids:
                maskable &= ids != tok

        m = forward_process(
            ids,
            self.mask_token_id,
            maskable=maskable,
            schedule=self.schedule,
            generator=self.generator,
            min_one_mask=self.min_one_mask,
        )
        return {
            "input_ids": m.noisy_ids,
            "clean_ids": ids,
            "attention_mask": attn,
            "maskable": maskable,
            "masked_indices": m.masked_indices,
            "p_mask": m.p_mask,
            "t": m.t,
        }


class SFTCollator:
    """SFT batches from {prompt_ids, response_ids} samples (prompt kept clean)."""

    def __init__(
        self,
        mask_token_id: int,
        eos_token_id: int,
        max_length: Optional[int] = None,
        schedule: Union[str, NoiseSchedule, None] = None,
        append_eos: bool = True,  # ensure the response ends with EOS
        non_maskable_ids: Optional[Set[int]] = None,  # EOS-collapse mitigation
        min_one_mask: bool = False,
        generator: Optional[torch.Generator] = None,
        response_canvas: Optional[int] = None,
        pad_mode: str = "eos",
        pad_token_ids: Sequence[int] = (),
        eos_as_one: bool = False,
        batch_cutoff: bool = False,
        rng: Optional[random.Random] = None,
    ):
        """
        Args:
            response_canvas: if set, every response (after the EOS is
                appended) is padded with EOS to exactly this many tokens, the
                fixed generation length used at inference. Positions after a
                sample's canvas that exist only because another sample in the
                batch is longer are batch padding: not attended, not
                maskable. A response longer than the canvas raises.
            pad_mode: "eos", "mask_ignored" or "dedicated" (module docstring).
            pad_token_ids: the dedicated padding ids, used cyclically.
            eos_as_one: tie each trailing EOS run into one masking unit
                (only with ``pad_mode="eos"``).
            batch_cutoff: draw a common response length per batch from the
                batch's content lengths (with ``rng``); not combinable with
                ``response_canvas``. Dream draws from response lengths that
                exclude tokens equal to its pad id; here the drawn lengths
                include the terminal EOS.
        """
        if response_canvas is not None and response_canvas < 1:
            raise ValueError("response_canvas must be a positive length")
        if pad_mode not in ("eos", "mask_ignored", "dedicated"):
            raise ValueError("pad_mode must be 'eos', 'mask_ignored' or 'dedicated'")
        if pad_mode == "dedicated":
            if not pad_token_ids:
                raise ValueError("pad_mode='dedicated' needs pad_token_ids")
            if eos_token_id in pad_token_ids or mask_token_id in pad_token_ids:
                raise ValueError("dedicated pad ids must differ from EOS and mask")
        elif pad_token_ids:
            raise ValueError("pad_token_ids requires pad_mode='dedicated'")
        if eos_as_one and pad_mode != "eos":
            raise ValueError("eos_as_one applies to pad_mode='eos' only")
        if batch_cutoff and response_canvas is not None:
            raise ValueError("batch_cutoff and response_canvas are exclusive")
        self.pad_mode = pad_mode
        self.pad_token_ids = tuple(pad_token_ids)
        self.eos_as_one = eos_as_one
        self.batch_cutoff = batch_cutoff
        self.rng = rng or random.Random()
        self.mask_token_id = mask_token_id
        self.eos_token_id = eos_token_id
        self.max_length = max_length
        self.schedule = get_schedule(schedule)
        self.append_eos = append_eos
        self.non_maskable_ids = non_maskable_ids or set()
        self.min_one_mask = min_one_mask
        self.generator = generator
        self.response_canvas = response_canvas

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        pairs = []
        for ex in batch:
            p, r = list(ex["prompt_ids"]), list(ex["response_ids"])
            if self.append_eos and (not r or r[-1] != self.eos_token_id):
                r = r + [self.eos_token_id]
            pairs.append((p, r))
        canvas = self.response_canvas
        if self.batch_cutoff:
            canvas = self.rng.choice([len(r) for _, r in pairs])
            pairs = [(p, r[:canvas]) for p, r in pairs]

        seqs, prompt_lens, content_lens = [], [], []
        for p, r in pairs:
            content_lens.append(len(r))
            if canvas is not None:
                if len(r) > canvas:
                    raise ValueError(
                        f"response of {len(r)} tokens exceeds response_canvas="
                        f"{canvas}"
                    )
                r = r + [self.eos_token_id] * (canvas - len(r))
            seqs.append(p + r)
            prompt_lens.append(len(p))

        # pad short pairs with EOS *as answer tokens*
        width = max(len(s) for s in seqs)
        if self.max_length is not None:
            width = min(width, self.max_length)
        ids = torch.full((len(seqs), width), self.eos_token_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            s = s[:width]
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        prompt_lens_t = torch.tensor(prompt_lens, dtype=torch.long)
        content_lens_t = torch.tensor(content_lens, dtype=torch.long)

        positions = torch.arange(width).unsqueeze(0)
        maskable = positions >= prompt_lens_t.unsqueeze(1)  # response + EOS pads
        if canvas is not None:
            # a fixed canvas ends every sample at prompt + canvas; anything
            # beyond it is batch padding
            in_canvas = positions < (prompt_lens_t + canvas).unsqueeze(1)
            maskable &= in_canvas
            attn = in_canvas.to(torch.long).expand(len(seqs), width).clone()
        else:
            attn = torch.ones_like(ids)
        content = maskable & (
            positions < (prompt_lens_t + content_lens_t).unsqueeze(1)
        )
        padding = maskable & ~content
        if self.pad_mode == "mask_ignored":
            ids = ids.masked_fill(padding, self.mask_token_id)
            maskable = maskable & ~padding
        elif self.pad_mode == "dedicated":
            table = torch.tensor(self.pad_token_ids, dtype=torch.long)
            order = (padding.cumsum(dim=1) - 1).clamp(min=0) % len(table)
            ids = torch.where(padding, table[order], ids)
        if self.non_maskable_ids:
            for tok in self.non_maskable_ids:
                maskable &= ids != tok

        m = forward_process(
            ids,
            self.mask_token_id,
            maskable=maskable,
            schedule=self.schedule,
            generator=self.generator,
            min_one_mask=self.min_one_mask,
            tie_trailing_token_id=self.eos_token_id if self.eos_as_one else None,
        )
        answer_lengths = maskable.sum(dim=1)
        return {
            "input_ids": m.noisy_ids,
            "clean_ids": ids,
            "attention_mask": attn,
            "maskable": maskable,
            "masked_indices": m.masked_indices,
            "p_mask": m.p_mask,
            "t": m.t,
            "prompt_lengths": prompt_lens_t,
            "answer_lengths": answer_lengths,
            "content": content,
            "content_lengths": content.sum(dim=1),
        }


class BlockSFTCollator:
    """Semi-AR block SFT: predict one block given clean prompt + prefix blocks.

    For each sample: pad the response with EOS to a block multiple, pick a
    random block k, mask within it by the schedule; earlier blocks stay clean.
    ``canvas="truncated"`` drops everything after block k (pairs with
    blockwise inference); ``canvas="full"`` keeps later blocks fully masked
    (pairs with canvas inference).

    Besides the masking artefacts the batch carries ``content`` (the target
    block's real response tokens plus the terminal EOS, without the EOS
    padding that completes the last block) for ``norm="content"``, and
    ``force_mask`` (the later blocks of a full canvas: masked in the input,
    never in the loss). Padding only completes the last block, so every
    target block holds at least one content token. For complementary
    masking pass ``force_mask`` as ``always_masked`` to
    :func:`dllm.masking.complementary_view`.
    """

    def __init__(
        self,
        mask_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        block_length: int = 32,
        canvas: str = "truncated",
        max_length: Optional[int] = None,
        schedule: Union[str, NoiseSchedule, None] = None,
        min_one_mask: bool = True,  # a fully-unmasked block has no signal
        generator: Optional[torch.Generator] = None,
        rng: Optional[random.Random] = None,
    ):
        if canvas not in ("truncated", "full"):
            raise ValueError("canvas must be 'truncated' or 'full'")
        self.mask_token_id = mask_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.block_length = block_length
        self.canvas = canvas
        self.max_length = max_length
        self.schedule = get_schedule(schedule)
        self.min_one_mask = min_one_mask
        self.generator = generator
        self.rng = rng or random.Random()

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        BL = self.block_length
        seqs = []
        for ex in batch:
            p, r = list(ex["prompt_ids"]), list(ex["response_ids"])
            if not r or r[-1] != self.eos_token_id:
                r = r + [self.eos_token_id]
            n_content = len(r)
            r = r + [self.eos_token_id] * ((-len(r)) % BL)
            n_blocks = len(r) // BL
            k = self.rng.randrange(n_blocks)
            if self.canvas == "truncated":
                seq = p + r[: (k + 1) * BL]
            else:
                seq = p + r
            if self.max_length is not None and len(seq) > self.max_length:
                # drop rather than silently truncate through the target block
                continue
            block_start = len(p) + k * BL
            content_end = len(p) + n_content
            mrow = [False] * len(seq)
            for j in range(block_start, block_start + BL):
                mrow[j] = "content" if j < content_end else True
            if self.canvas == "full":
                # later blocks: fully masked context, never in the loss
                for j in range(block_start + BL, len(seq)):
                    mrow[j] = None  # sentinel: force-mask, not maskable
            seqs.append((seq, mrow))

        if not seqs:
            raise ValueError("all samples exceeded max_length")

        ids, attn = _pad_stack([s for s, _ in seqs], self.pad_token_id)
        width = ids.shape[1]
        B = ids.shape[0]
        maskable = torch.zeros(B, width, dtype=torch.bool)
        content = torch.zeros(B, width, dtype=torch.bool)
        force_mask = torch.zeros(B, width, dtype=torch.bool)
        for i, (_, mrow) in enumerate(seqs):
            for j, v in enumerate(mrow):
                if v is True or v == "content":
                    maskable[i, j] = True
                    content[i, j] = v == "content"
                elif v is None:
                    force_mask[i, j] = True

        m = forward_process(
            ids,
            self.mask_token_id,
            maskable=maskable,
            schedule=self.schedule,
            generator=self.generator,
            min_one_mask=self.min_one_mask,
        )
        noisy = m.noisy_ids
        noisy[force_mask] = self.mask_token_id

        return {
            "input_ids": noisy,
            "clean_ids": ids,
            "attention_mask": attn,
            "maskable": maskable,
            "masked_indices": m.masked_indices,
            "p_mask": m.p_mask,
            "t": m.t,
            "answer_lengths": maskable.sum(dim=1),  # == BL per sample
            "content": content,
            "content_lengths": content.sum(dim=1),
            "force_mask": force_mask,
        }
