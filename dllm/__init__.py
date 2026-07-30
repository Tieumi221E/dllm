"""dllm - a self-contained masked-diffusion LM toolkit.

Training :  schedules / forward_process / diffusion_loss / collators
Inference:  full-canvas / incremental / self-speculative generation
Evaluation: mc_conditional_nll
RL:         trajectory_logprobs (canvas-consistent state reconstruction)
Models:     topology-aware DiffusionTransformer, denoiser protocol, HF wrapper

References for the algorithms are listed in the README.
"""

__version__ = "1.3.0"

from .schedules import NoiseSchedule, LinearSchedule, CosineSchedule, get_schedule
from .masking import MaskingOutput, forward_process, make_labels, random_truncate
from .loss import diffusion_loss, masked_cross_entropy, mc_conditional_nll
from .data import PretrainCollator, SFTCollator, BlockSFTCollator
from .topology import AttentionTopology, ordered_attention_mask
from .execution import CacheSemantics, EXACT_BLOCK_CAUSAL
from .models import (
    Denoiser,
    DenoiserInput,
    DenoiserOutput,
    DiffusionTransformer,
    KVCache,
    ModelCapabilities,
    extract_logits,
)
from .sampling import (
    CanvasConfig,
    CanvasOutput,
    generate_canvas,
    BlockwiseConfig,
    BlockwiseOutput,
    generate_blockwise,
    CommitDecision,
    CommitPolicy,
    CommitSpec,
    CommitState,
    QuotaCommitPolicy,
    ThresholdCommitPolicy,
    apply_commit_policy,
    resolve_commit_policy,
    SelfSpecBackend,
    SelfSpecConfig,
    SelfSpecOutput,
    SelfSpecStats,
    SelfSpecStep,
    TopologySelfSpecBackend,
    generate_self_speculative,
    TrajectorySample,
    TrajectoryStep,
    TokenDistribution,
    TopKPrediction,
)
from .rl import (
    PPOObjective,
    StepLogProb,
    TrajectoryState,
    ppo_clip_objective,
    score_trajectory_states,
    trajectory_logprobs,
    trajectory_states,
)
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
    "masked_cross_entropy",
    "mc_conditional_nll",
    # data
    "PretrainCollator",
    "SFTCollator",
    "BlockSFTCollator",
    # models
    "AttentionTopology",
    "ordered_attention_mask",
    "CacheSemantics",
    "EXACT_BLOCK_CAUSAL",
    "Denoiser",
    "DenoiserInput",
    "DenoiserOutput",
    "DiffusionTransformer",
    "KVCache",
    "ModelCapabilities",
    "extract_logits",
    # sampling
    "CanvasConfig",
    "CanvasOutput",
    "generate_canvas",
    "BlockwiseConfig",
    "BlockwiseOutput",
    "generate_blockwise",
    "CommitDecision",
    "CommitPolicy",
    "CommitSpec",
    "CommitState",
    "QuotaCommitPolicy",
    "ThresholdCommitPolicy",
    "apply_commit_policy",
    "resolve_commit_policy",
    "SelfSpecBackend",
    "SelfSpecConfig",
    "SelfSpecOutput",
    "SelfSpecStats",
    "SelfSpecStep",
    "TopologySelfSpecBackend",
    "generate_self_speculative",
    "TrajectorySample",
    "TrajectoryStep",
    "TokenDistribution",
    "TopKPrediction",
    # rl
    "PPOObjective",
    "StepLogProb",
    "TrajectoryState",
    "ppo_clip_objective",
    "score_trajectory_states",
    "trajectory_logprobs",
    "trajectory_states",
    # presets
    "get_preset",
    "list_presets",
]
