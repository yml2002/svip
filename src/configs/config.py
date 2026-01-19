"""Configuration management (minimal core + preserved training shell).

Notes
- This is intentionally compatible with the existing training runtime/trainer utilities.
- We keep a stable config surface for logging/visualization/training, while the model core
    switches to a minimal DINOv2 (local HF) + GATv2 + Transformer + ranking-loss formulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    data_dir: str = "data/shared_data/preprocessed_fixed"
    video_length: int = 120
    max_persons: int = 20
    image_height: int = 192
    image_width: int = 336
    cache_data: bool = True  # 设置True为启用特征缓存
    max_samples: Optional[int] = None
    data_ratio: float = 1.0


@dataclass
class DropoutConfig:
    features: float = 0.10
    gatv2: float = 0.10
    temporal: float = 0.20
    scoring: float = 0.20
    gate: float = 0.20


@dataclass
class BBoxGeomConfig:
    feature_dim: int = 256
    hidden_dim: int = 512


@dataclass
class DinoConfig:
    """DINO (HuggingFace) settings (offline-only).

    Example local directory: `data/models/dinov2-base`.
    """

    enabled: bool = True
    model_dir: str = "data/models/dinov2-base"
    feature_dim: int = 768
    freeze: bool = True
    image_size: int = 192


@dataclass
class FeatureConfig:
    dino: DinoConfig = field(default_factory=DinoConfig)
    bbox_geom: BBoxGeomConfig = field(default_factory=BBoxGeomConfig)
    fused_dim: int = 1024


@dataclass
class GATv2Config:
    enabled: bool = True
    hidden_dim: int = 512
    num_layers: int = 2
    heads: int = 4
    use_residual: bool = True


@dataclass
class TemporalTransformerConfig:
    enabled: bool = True
    d_model: int = 768
    nhead: int = 12
    num_layers: int = 2
    dim_feedforward: int = 1024
    use_event_token: bool = True
    event_num_layers: int = 1
    agg_heads: int = 4
    agg_out_dim: int = 512
    pooling: Optional[str] = "mean"
    transformer_layers: int = 1
    use_video_transformer: bool = True


@dataclass
class ScoringConfig:
    hidden_dim: int = 128
    temperature: float = 1.0


@dataclass
class LossConfig:
    beta: float = 1.0
    importance_weight: float = 1.0
    preference_weight: float = 0.3
    pairwise_weight: float = 0.0
    pairwise_margin: float = 0.2


@dataclass
class ModelConfig:
    dropout: DropoutConfig = field(default_factory=DropoutConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    gatv2: GATv2Config = field(default_factory=GATv2Config)
    temporal: TemporalTransformerConfig = field(default_factory=TemporalTransformerConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class TrainingConfig:
    learning_rate: float = 5e-5
    weight_decay: float = 5e-4
    betas: tuple = (0.9, 0.999)
    min_lr: float = 1e-5

    num_epochs: int = 30
    batch_size: int = 8
    accumulation_steps: int = 2
    roi_chunk: int = 256

    enable_dual_head: bool = True
    gate_hidden_dim: int = 128

    use_mixed_precision: bool = True
    max_grad_norm: float = 3.0

    distributed: bool = False
    local_rank: int = 0
    world_size: int = 1
    find_unused_parameters: bool = True

    num_workers: int = 2
    pin_memory: bool = True

    early_stop: Optional[int] = 3

    debug: bool = False
    debug_output_dir: str = "debug"
    debug_log_interval: int = 1  # Log timing every N batches in debug mode
    save_model: bool = False


@dataclass
class ExperimentConfig:
    output_dir: str = "outputs"

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def get_default_config() -> ExperimentConfig:
    return ExperimentConfig()
