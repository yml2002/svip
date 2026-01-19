# SVIP - Social Video Importance Prediction

视频中重要人物识别与排序系统，基于多模态特征融合和时空图建模。

## 项目概述

本项目实现了MSG_VIP任务的深度学习解决方案，目标是从视频中自动识别出最重要的人物。采用**分离式时空建模**架构：空间维度使用图注意力网络（GATv2）建模人物间社交关系，时间维度使用Transformer捕捉行为演化。

### 核心功能

- **多模态特征提取**：DINOv2视觉特征 + BBox几何特征
- **空间建模**：GATv2逐帧建模人物社交关系图
- **时间建模**：Transformer跨帧捕捉每个人的行为演化
- **Dual-head评分**：区分自身重要性（self）和社交关系重要性（rel），通过gate动态融合
- **特征缓存系统**：两阶段训练，预提取DINOv2特征加速迭代

## 架构设计

```
输入：视频帧 + BBox + Person Mask
  ↓
DINOv2 Feature Extraction (可缓存)
  ↓
BBox Geometry Features
  ↓
Multimodal Fusion (1024-dim)
  ↓
GATv2 (空间图建模，2层×4头) → 社交关系特征
  ↓
Temporal Transformer (时序建模，2层×12头) → 行为演化
  ↓
Event Token Context (全局上下文)
  ↓
Video-level Aggregator (attention pooling)
  ↓
Dual-head Scoring:
  - Self branch: 自身重要性
  - Rel branch: 社交重要性
  - Gate: 动态融合权重
  ↓
输出：重要性排序
```

## 主要改动（当前版本）

### 1. 特征缓存系统 (`src/utils/feature_cache.py`)
- **FeatureExtractor**: 提取并缓存DINOv2+BBox特征
- **智能缓存失效**: 基于config MD5 hash自动检测配置变化
- **两阶段训练**: 
  - Stage 1: 提取特征到 `feature_cache_tmp/{hash}/`
  - Stage 2: 从缓存加载，加速训练迭代
- **持久化缓存**: 禁用自动清理，支持跨训练复用

### 2. 调试性能分析 (`src/training/loops.py`)
- `--debug` 模式：逐batch记录详细timing
  - data loading time
  - forward pass time
  - loss computation time
  - backward pass time
  - optimizer step time

### 3. DDP稳定性改进 (`src/train.py`)
- 随机端口选择（29500-65535）避免端口冲突
- 启动时清理陈旧的DDP环境变量

### 4. Bug修复
- **Early stopping**: 修复off-by-one错误（`>` → `>=`）
- **GPU设备传输**: 修复缓存特征未正确转移到GPU的问题
- **VideoLevelAggregator**: 保留复杂projection layer用于复现实验

### 5. 配置管理 (`src/configs/config.py`)
- 新增 `cache_data: bool = True` 控制特征缓存
- 新增 `debug_log_interval: int = 1` 控制调试日志频率
- 完整的temporal transformer配置（event_token、agg_heads等）

## 环境要求

```bash
# Python 3.8+
torch >= 2.0
transformers  # 用于DINOv2
torch_geometric  # 用于GATv2
```

## 使用方法

### 基础训练
```bash
python src/train.py --batch_size 8 --num_epochs 30 --learning_rate 5e-5
```

### 启用调试模式
```bash
python src/train.py --debug  # 详细记录每个batch的timing
```

### 配置特征缓存
编辑 `src/configs/config.py`:
```python
cache_data: bool = True   # 启用缓存（首次会提取51分钟）
cache_data: bool = False  # 禁用缓存，每次从原始数据提取
```

## 项目结构

```
src/
├── configs/          # 配置管理
│   └── config.py     # ExperimentConfig (data/model/training)
├── data/             # 数据加载
│   └── dataloader.py # MSGVIPDataset
├── models/           # 核心模型
│   ├── importance_ranker.py  # 主模型
│   ├── vision_encoder.py     # DINOv2封装
│   ├── bbox_geom.py          # BBox几何特征
│   ├── gatv2.py              # 图注意力网络
│   ├── event_context.py      # EventTokenContext
│   └── video_aggregator.py   # VideoLevelAggregator
├── training/         # 训练流程
│   ├── runtime.py    # TrainingRuntime (编排)
│   ├── trainer.py    # MemoryEfficientTrainer
│   ├── loops.py      # train/val epoch循环
│   └── loss.py       # CombinedLoss
├── utils/            # 工具函数
│   ├── feature_cache.py      # 特征缓存系统
│   ├── output_manager.py     # 输出管理
│   └── visualization.py      # 可视化
└── train.py          # 训练入口

docs/
├── msgvip_architecture_overview.md  # 架构详细文档
└── msgvip_edge_rebuild_plan.md      # 边构建计划
```

## 性能特征

### 计算复杂度
- **Batch配置**: 8个视频 × 120帧 × 20人 = 19,200个ROI crops
- **主要瓶颈**: Temporal Transformer（O(N²)复杂度）
  - 单次forward: (160序列, 120帧, 768维) × 2层 × 12头
  - Dual-head模式: 计算量翻倍
- **典型timing** (启用缓存后):
  - Data loading: ~0.1ms
  - Forward pass: ~27s (Transformer主导)
  - Loss compute: ~770ms
  - Backward: ~16s
  - Optimizer: ~30ms

### 优化建议
1. **降采样帧数**: 120→60帧可减半计算量
2. **减少Transformer层数**: 2→1层
3. **关闭dual_head**: 如gate权重不显著则可节省50%计算
4. **Hierarchical temporal**: 局部→全局分层建模

## 实验目标

当前目标：**准确率超越75%**

实验路径：
1. ✅ 复现src0117的69%准确率（验证VideoLevelAggregator问题）
2. 修复output_projection恢复至71%
3. 迭代优化突破75%

## 版本历史

### src-acc71 (Baseline)
- 71%准确率的稳定版本

### src0117 (Problematic)
- 69%准确率，包含VideoLevelAggregator的problematic projection

### src (Current)
- 特征缓存系统
- 性能调试工具
- DDP稳定性改进
- 准备复现实验

## License

研究项目代码

## 联系方式

Repository: https://github.com/yml2002/svip
