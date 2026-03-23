"""Configuration management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    data_dir: str = "data/shared_data/preprocessed_fixed"
    train_splits: list[str] = field(default_factory=lambda: ["train", "test"])
    # train_splits: list[str] = field(default_factory=lambda: ["train"])
    val_split: str = "val"
    video_length: int = 120
    max_persons: int = 16
    cache_data: bool = False
    max_samples: Optional[int] = None
    data_ratio: float = 1.0


@dataclass
class DropoutConfig:
    features: float = 0.10
    gatv2: float = 0.20
    temporal: float = 0.20
    scoring: float = 0.20


@dataclass
class BBoxGeomConfig:
    enabled: bool = True
    feature_dim: int = 128
    hidden_dim: int = 128
    spatial_edge_dim: int = 32
    temporal_edge_dim: int = 16


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
    hidden_dim: int = 512
    num_layers: int = 2
    heads: int = 4
    topk_neighbors: int = 4
    temporal_window: int = 3
    use_temporal_edges: bool = True
    use_edge_features: bool = True


@dataclass
class ScoringConfig:
    hidden_dim: int = 256
    temperature: float = 1.0


@dataclass
class SelfBranchConfig:
    enabled: bool = True


@dataclass
class RelationConfig:
    enabled: bool = True


@dataclass
class LossConfig:
    beta: float = 1.0
    importance_weight: float = 1.0
    preference_weight: float = 0.0


@dataclass
class ModelConfig:
    dropout: DropoutConfig = field(default_factory=DropoutConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    gatv2: GATv2Config = field(default_factory=GATv2Config)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    self_branch: SelfBranchConfig = field(default_factory=SelfBranchConfig)
    relation: RelationConfig = field(default_factory=RelationConfig)
    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class TrainingConfig:
    learning_rate: float = 5e-5
    weight_decay: float = 5e-4
    betas: tuple = (0.9, 0.999)
    min_lr: float = 1e-5

    num_epochs: int = 15
    batch_size: int = 16
    accumulation_steps: int = 4
    roi_chunk: int = 512
    export_train_predictions: bool = False

    use_mixed_precision: bool = True
    activation_checkpointing: bool = False
    max_grad_norm: float = 3.0

    distributed: bool = False
    local_rank: int = 0
    world_size: int = 1
    find_unused_parameters: bool = True

    num_workers: int = 4
    pin_memory: bool = True
    save_checkpoints: bool = False

    early_stop: Optional[int] = 3

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
