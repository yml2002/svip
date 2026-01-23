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
    cache_data: bool = False
    # cache_data: bool = True  # 设置True为启用特征缓存
    max_samples: Optional[int] = None
    data_ratio: float = 1.0


@dataclass
class DropoutConfig:
    features: float = 0.10
    gatv2: float = 0.10
    temporal: float = 0.20
    scoring: float = 0.25
    gate: float = 0.20
    st_graph: float = 0.15  # 时空图


@dataclass
class BBoxGeomConfig:
    """BBox几何特征编码.
    
    功能: 14维基础特征 (x,y,w,h,area,运动等) → feature_dim
    设计:
    - feature_dim=256: 辅助特征，不应过大 (vs DINOv2 768维)
    - hidden_dim=256: 匹配14维输入的简单性 (18x扩张，原36x过大)
    """
    feature_dim: int = 256
    hidden_dim: int = 256  # 匹配14维输入的简单性


@dataclass
class DinoConfig:
    """DINO (HuggingFace) settings (offline-only).

    Example local directory: `data/models/dinov2-base`.
    
    微调策略 (基于SVM分析):
    - freeze=False: 启用微调 (SVM 75.86% > 深度模型 68.32% → 需要任务特定特征)
    - finetune_layers=2: 仅微调最后2层 (12层总共, ~17%参数)
    - 降低过拟合风险,同时适配MSG-VIP任务
    """

    enabled: bool = True
    model_dir: str = "data/models/dinov2-base"
    feature_dim: int = 768
    freeze: bool = False  # 启用微调
    finetune_layers: int = 2  # 恢复峰值能力
    image_size: int = 192


@dataclass
class FeatureConfig:
    dino: DinoConfig = field(default_factory=DinoConfig)
    bbox_geom: BBoxGeomConfig = field(default_factory=BBoxGeomConfig)
    fused_dim: int = 1024


@dataclass
class SpatiotemporalGraphConfig:
    """Unified spatiotemporal graph modeling - 时空联合建模.
    
    设计理念 (基于SVM 75.86%的启示):
    - hidden_dim=768: 适度压缩 (1024→76825%), 平衡信息保留和泛化
    - num_layers=2: 问题接近线性可分，不需要太深
    - num_heads=12: 适配768维 (64维/head, 标准设置)
    - 参数量: ~15M (原6.8M的2.2x, 避免过拟合)
    - dropout: 从DroupoutConfig.st_graph统一管理
    """
    enabled: bool = True  # 启用新架构
    hidden_dim: int = 768  # 适度压缩25%
    num_layers: int = 2  # 小数据集下更稳健
    num_heads: int = 12  # 12头 (768/12=64维/head)
    temporal_window: int = 10  # 时序窗口：前后10帧
    
    # Memory optimization for large sequences
    use_chunked_attention: bool = True  # 启用分块attention（省显存）
    chunk_size: int = 30  # 每次处理30帧
    chunk_threshold: int = 1000  # TN>1000时启用chunking


@dataclass
class GATv2Config:
    enabled: bool = False  # 替换为时空图
    hidden_dim: int = 512
    num_layers: int = 3
    heads: int = 4
    use_residual: bool = True


@dataclass
class TemporalTransformerConfig:
    """Temporal modeling config.
    
    Event Token设计理念:
    - ST-Graph: bottom-up, 局部时空交互 (person-centric)
    - Event Token: top-down, 全局事件语义 (event-centric)
    - 互补关系: 个体特征 + 事件上下文 = 在该事件中的重要度
    """
    enabled: bool = False  # 被ST-Graph替代
    d_model: int = 768
    nhead: int = 8
    num_layers: int = 1
    dim_feedforward: int = 1024
    
    # Event Token配置
    use_event_token: bool = True  # 启用hierarchical event context
    event_type: str = "segment"  # "segment" (hierarchical) or "frame" (legacy)
    num_segments: int = 6  # 120帧/6 = 20帧/段
    event_dim: int = 512  # Event token维度
    event_num_layers: int = 1
    agg_heads: int = 4
    agg_out_dim: int = 512
    pooling: Optional[str] = "attention"
    transformer_layers: int = 1
    use_video_transformer: bool = False  # 时空图已包含


@dataclass
class ScoringConfig:
    """评分模块配置.
    
    设计理念:
    - hidden_dim=256: 减少从512的压缩率 (原128太小)
    - SVM用全部512维达到75.86%，说明维度很重要
    """
    hidden_dim: int = 512  # 提升评分表达,冲峰值
    temperature: float = 1.0


@dataclass
class LossConfig:
    beta: float = 1.0
    importance_weight: float = 1.0
    preference_weight: float = 0.3
    pairwise_weight: float = 0.05
    pairwise_margin: float = 0.2


@dataclass
class ModelConfig:
    dropout: DropoutConfig = field(default_factory=DropoutConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    gatv2: GATv2Config = field(default_factory=GATv2Config)
    st_graph: SpatiotemporalGraphConfig = field(default_factory=SpatiotemporalGraphConfig)
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
    batch_size: int = 16
    accumulation_steps: int = 2
    roi_chunk: int = 256

    enable_dual_head: bool = True  # 保留dual head
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
