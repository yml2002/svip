"""Configuration management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    data_dir: str = "data/shared_data/preprocessed_fixed"
    # train_splits: list[str] = field(default_factory=lambda: ["train", "test"])
    train_splits: list[str] = field(default_factory=lambda: ["train"])
    val_split: str = "val"
    video_length: int = 120       # raw frames per NPZ (fixed by dataset)
    sampled_frames: int = 32      # frames after valid-frame extraction + uniform downsample
    max_persons: int = 16
    cache_data: bool = False
    max_samples: Optional[int] = None
    data_ratio: float = 1.0
    augmentation: "AugmentationConfig" = field(default_factory=lambda: AugmentationConfig())


@dataclass
class AugmentationConfig:
    enabled: bool = True
    horizontal_flip_prob: float = 0.5
    brightness: float = 0.12
    contrast: float = 0.12
    saturation: float = 0.10
    noise_std: float = 0.02


@dataclass
class DropoutConfig:
    features: float = 0.10
    gatv2: float = 0.20
    temporal: float = 0.20
    scoring: float = 0.20


@dataclass
class BBoxGeomConfig:
    enabled: bool = True
    feature_dim: int = 64
    hidden_dim: int = 64
    fuse_scale: float = 1.0
    dropout_prob: float = 0.0


@dataclass
class DinoConfig:
    enabled: bool = True
    model_dir: str = "data/models/dinov2-base"
    feature_dim: int = 768
    freeze: bool = False
    unfreeze_layers: int = 1
    image_size: int = 196


@dataclass
class FeatureConfig:
    dino: DinoConfig = field(default_factory=DinoConfig)
    bbox_geom: BBoxGeomConfig = field(default_factory=BBoxGeomConfig)
    fused_dim: int = 768


@dataclass
class GATv2Config:
    enabled: bool = True
    graph_type: str = "gatv2"  # "gatv2" or "gcn"
    hidden_dim: int = 512
    num_layers: int = 2
    heads: int = 4
    topk_neighbors: int = 8
    use_spatial_edges: bool = True
    use_temporal_edges: bool = True
    use_edge_features: bool = True


@dataclass
class ScoringConfig:
    hidden_dim: int = 256


@dataclass
class IntrinsicConfig:
    use_motion_priors: bool = True
    use_max_pool: bool = True
    use_attention_pool: bool = True


@dataclass
class RelationConfig:
    enabled: bool = True
    delta_scale: float = 0.85
    use_adaptive_gate: bool = True
    gate_bias: float = 0.90


@dataclass
class LossConfig:
    beta: float = 1.0
    importance_weight: float = 1.0
    preference_weight: float = 0.25
    intrinsic_aux_weight: float = 0.20
    relation_aux_weight: float = 0.05
    scene_consistency_weight: float = 0.02


@dataclass
class GlobalContextConfig:
    """Open-world scene context extracted from keyframes."""
    enabled: bool = True
    num_keyframes: int = 8
    context_dim: int = 256
    num_heads: int = 4
    num_layers: int = 1
    dropout: float = 0.15
    num_prototypes: int = 8


@dataclass
class ModelConfig:
    dropout: DropoutConfig = field(default_factory=DropoutConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    gatv2: GATv2Config = field(default_factory=GATv2Config)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    intrinsic: IntrinsicConfig = field(default_factory=IntrinsicConfig)
    relation: RelationConfig = field(default_factory=RelationConfig)
    global_context: GlobalContextConfig = field(default_factory=GlobalContextConfig)
    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    backbone_lr_scale: float = 0.05
    weight_decay: float = 5e-4
    betas: tuple = (0.9, 0.999)
    min_lr: float = 1e-5
    backbone_warmup_epochs: int = 1
    backbone_train_mode: str = "attn_ln"

    seed: int = 2026

    num_epochs: int = 15
    batch_size: int = 64
    accumulation_steps: int = 1
    roi_chunk: int = 32768
    export_train_predictions: bool = False

    use_mixed_precision: bool = True
    activation_checkpointing: bool = False
    max_grad_norm: float = 3.0

    distributed: bool = False
    local_rank: int = 0
    world_size: int = 1
    find_unused_parameters: bool = True

    num_workers: int = 8
    pin_memory: bool = True
    save_checkpoints: bool = False

    early_stop: Optional[int] = 2

    debug: bool = False
    debug_output_dir: str = "debug"


@dataclass
class ExperimentConfig:
    output_dir: str = "outputs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def get_default_config() -> ExperimentConfig:
    return ExperimentConfig()
