"""dllm - a self-contained masked-diffusion LM toolkit.

Training :  schedules / forward_process / diffusion_loss / collators
Inference:  generate_canvas (full canvas) / generate_blockwise (incremental)
Evaluation: mc_conditional_nll
RL:         trajectory_logprobs (canvas-consistent state reconstruction)
Models:     DiffusionTransformer (MHA/GQA x learned/RoPE, KV cache), HF wrapper

References for the algorithms are listed in the README.
"""

__version__ = "1.0.0"

from .schedules import NoiseSchedule, LinearSchedule, CosineSchedule, get_schedule
from .masking import MaskingOutput, forward_process, make_labels, random_truncate
from .loss import diffusion_loss, mc_conditional_nll
from .data import PretrainCollator, SFTCollator, BlockSFTCollator
from .models import DiffusionTransformer, KVCache
from .sampling import (
    CanvasConfig,
    CanvasOutput,
    generate_canvas,
    BlockwiseConfig,
    BlockwiseOutput,
    generate_blockwise,
    TrajectorySample,
    TrajectoryStep,
)
from .rl import StepLogProb, trajectory_logprobs
from .presets import get_preset, list_presets

__all__ = [
    "__version__",
    # schedules & masking
    "NoiseSchedule",
    "LinearSchedule",
    "CosineSchedule",
    "get_schedule",
    "MaskingOutput",
    "forward_process",
    "make_labels",
    "random_truncate",
    # loss & eval
    "diffusion_loss",
    "mc_conditional_nll",
    # data
    "PretrainCollator",
    "SFTCollator",
    "BlockSFTCollator",
    # models
    "DiffusionTransformer",
    "KVCache",
    # sampling
    "CanvasConfig",
    "CanvasOutput",
    "generate_canvas",
    "BlockwiseConfig",
    "BlockwiseOutput",
    "generate_blockwise",
    "TrajectorySample",
    "TrajectoryStep",
    # rl
    "StepLogProb",
    "trajectory_logprobs",
    # presets
    "get_preset",
    "list_presets",
]
