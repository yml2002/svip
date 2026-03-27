# SVIP: Spatio-Temporal Video Importance Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/pytorch-1.12+-red.svg)](https://pytorch.org/)
[![CUDA 11.0+](https://img.shields.io/badge/cuda-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SVIP** 是一个基于双分支架构的视频人物重要性排序系统，通过显式分离个体显著性和社交动态重要性，实现对视频中人物重要性的精准评估。

## 🎯 任务定义

**视频人物重要性排序**：给定包含多个人物的社交场景视频片段，模型需要输出每个人物的重要性得分，识别出谁是场景中的主要关注对象。

### 核心挑战

1. **个体与群体的平衡**：人物重要性既取决于个体自身的显著性，也受其在群体中的角色影响
2. **静态与动态的结合**：重要性评估需要同时考虑静态特征和动态变化
3. **场景适应性**：不同场景有不同的"重要性标准"
4. **复杂社交关系**：人物间的相互影响和群体动态难以建模

## 🏗️ 架构设计

### 核心创新

我们的SVIP架构采用**双分支分解**设计，将人物重要性分解为两个互补维度：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       SVIP 双分支架构                                    │
│                                                                         │
│  输入视频 → 特征提取主干 → ┬─ 个体分支(静态显著性) → 个体得分           │
│                          │                                             │
│                          └─ 关系分支(社交动态) → 关系得分               │
│                                    │                                    │
│                                    └─ KCGC场景上下文                    │
│                                    └─ 时空图注意力网络                  │
│                                    └─ 全局关系上下文                    │
│                                                                         │
│  个体得分 + 关系得分 → 温度调节softmax → 重要性排序                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### 关键模块

| 模块 | 创新点 | 解决的问题 |
|------|--------|------------|
| **双分支分解** | 个体(静态) + 关系(动态)评分相加 | 清晰分离：个体显著性 vs 社交动态重要性 |
| **KCGC** | 全帧DINOv2 CLS作为场景上下文，交叉注意力注入 | 场景感知的重要性评估 |
| **时空图结构** | 联合(person, frame)节点 + 空间边 + 时间边 | 同时建模瞬时空间关系和时间外观动态 |
| **全局关系上下文** | Transformer编码器 + 位置编码 | 捕获全局社交关系结构 |
| **特征职责分离** | 静态几何 + 时间运动特征的清晰分工 | 避免特征冗余和干扰 |

## 📊 实验结果

### 消融实验 (experiments_20260325_223734)

| 实验 | Rank@1 | Rank@2 | Rank@3 | vs Full |
|------|------|------|------|------|
| Full (Ours) | **73.75** | 90.73 | 97.14 | — |
| w/o Temporal | 73.48 | 90.73 | 97.09 | -0.27 |
| GCN | 73.64 | **90.94** | 97.25 | -0.11 |
| w/o Global Context | 72.72 | 90.40 | 97.04 | -1.02 |
| w/o Relation | 70.08 | 88.63 | 96.87 | -3.67 |
| w/o Geometry | 71.97 | 90.73 | **97.36** | -1.78 |
| w/o GAT | 72.88 | 90.30 | 96.98 | -0.86 |
| w/o Spatial | 72.83 | 90.89 | 97.04 | -0.92 |

### 多GPU实验 (experiments_20260326_184233)

| 实验 | Rank@1 | Rank@2 | Rank@3 | vs Full |
|------|------|------|------|------|
| Full (Ours) | **73.10** | 90.35 | 96.66 | — |
| no_temporal | 72.72 | **90.51** | 96.87 | -0.38 |
| gcn | 72.51 | 90.30 | **97.30** | -0.59 |
| no_global_ctx | 72.45 | 89.97 | 96.50 | -0.65 |
| no_relation | 69.33 | 89.16 | 96.39 | -3.77 |
| no_geom | 72.13 | 90.08 | 97.03 | -0.97 |
| no_gat | 72.61 | 90.51 | 96.55 | -0.49 |
| no_spatial | 73.05 | 90.19 | 96.71 | -0.05 |

### 关键发现

1. **关系模块最关键**：移除整个关系分支导致性能下降3.67-3.77%，证明社交动态建模的必要性
2. **几何特征重要**：静态几何信息贡献0.97-1.78%的性能提升
3. **场景上下文有价值**：KCGC模块提供0.65-1.02%的性能提升
4. **稳定性良好**：不同种子下结果稳定，Rank@1标准差仅为0.23-0.46%

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+
- 其他依赖见`requirements.txt`

### 安装

```bash
# 克隆项目
git clone <repository-url>
cd svip

# 创建conda环境
conda create -n svip python=3.8
conda activate svip

# 安装依赖
pip install -r requirements.txt
```

### 训练命令

#### 单卡测试

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n svip python src/train.py \
    --batch_size 2 \
    --num_epochs 1 \
    --data_ratio 0.005 \
    --accumulation_steps 16
```

#### 单卡训练

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n svip python src/train.py \
    --batch_size 8 \
    --num_epochs 10 \
    --data_ratio 1.0 \
    --accumulation_steps 8
```

#### 多卡DDP训练

```bash
torchrun --nproc_per_node=2 src/train.py \
    --batch_size 4 \
    --num_epochs 10 \
    --data_ratio 1.0 \
    --accumulation_steps 16
```

### 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--batch_size` | 批次大小 | 64 |
| `--accumulation_steps` | 梯度累积步数 | 1 |
| `--num_epochs` | 训练轮数 | 15 |
| `--data_ratio` | 数据使用比例 | 1.0 |
| `--roi_chunk` | ROI处理分块大小 | 16384 |
| `--learning_rate` | 学习率 | 5e-5 |
| `--num_workers` | 数据加载线程数 | 8 |
| `--early_stop` | 早停轮数 | 2 |
| `--importance_weight` | 重要性损失权重 | 1.0 |
| `--preference_weight` | 偏好损失权重 | 0.0 |
| `--logit_temperature` | 温度参数 | 1.0 |

## 📁 项目结构

```
svip/
├── README.md                    # 项目说明文档
├── requirements.txt             # 依赖包列表
├── LICENSE                      # 许可证文件
├── src/                         # 源代码目录
│   ├── config.py               # 配置管理
│   ├── train.py                # 训练入口
│   ├── experiments.py          # 实验管理
│   ├── ARCHITECTURE.md         # 架构详细文档
│   ├── RESULTS.md              # 实验结果
│   ├── data/                   # 数据处理
│   │   ├── dataset.py         # 数据集定义
│   │   ├── splits.py          # 数据划分
│   │   └── dataloader.py      # 数据加载器
│   ├── models/                 # 模型定义
│   │   ├── ranker.py          # 主模型
│   │   ├── vision_encoder.py  # DINOv2视觉编码器
│   │   ├── bbox_geom.py       # 边界框几何编码器
│   │   ├── gatv2.py           # 时空图注意力网络
│   │   ├── global_context.py  # KCGC场景上下文
│   │   └── roi.py             # ROI裁剪工具
│   ├── engine/                 # 训练引擎
│   │   ├── trainer.py         # 训练器
│   │   ├── loops.py           # 训练循环
│   │   ├── loss.py            # 损失函数
│   │   └── metrics.py         # 评估指标
│   └── utils/                  # 工具函数
├── data/                       # 数据目录
├── outputs/                    # 输出目录
│   └── experiments_*/         # 实验结果
└── docs/                       # 文档目录
```

## 🔧 核心算法流程

### 1. 特征提取

```
输入视频帧 → ROI裁剪 → DINOv2 → 视觉特征(768维)
         ↓
    边界框几何 → 几何编码器 → 几何特征(128维)
         ↓
    特征拼接 → 融合MLP → 融合特征(768维)
```

### 2. 双分支处理

```
融合特征 → 个体分支(时间平均+MLP) → 个体得分
      ↓
    KCGC场景增强 → 时空图注意力 → 全局关系上下文 → 时间平均 → 关系得分
```

### 3. 输出融合

```
个体得分 + 关系得分 → logits → softmax(温度调节) → 重要性排序
```

## 📈 模型复杂度

| 模块 | 参数量 | 可训练参数 | 用途 |
|------|--------|------------|------|
| VisionEncoder (DINOv2-Base) | 86.580M | 7.089M (最后1层) | 视觉特征提取 |
| BBoxGeomEncoder | 0.018M | 0.018M | 几何编码 |
| 特征融合MLP | 0.690M | 0.690M | 多模态集成 |
| 个体评分MLP | 0.199M | 0.199M | 静态显著性 |
| SpatioTemporalGATv2 | 1.509M | 1.509M | 社交交互建模 |
| GlobalRelationContext | 0.526M | 0.526M | 全局关系结构 |
| 关系评分MLP | 0.133M | 0.133M | 动态重要性 |
| KCGC (交叉注意力) | 1.774M | 1.774M | 场景上下文 |
| **总计** | **91.4M** | **11.9M** | **完整系统** |

## 🎯 应用场景

本系统适用于：

1. **视频摘要生成**：自动识别重要人物，生成聚焦摘要
2. **智能视频剪辑**：基于人物重要性的自动剪辑
3. **内容推荐系统**：个性化视频内容推荐
4. **社交行为分析**：视频中的社交关系和动态分析
5. **安防监控**：重要人物识别和异常行为检测
6. **会议记录**：自动识别发言人和主要参与者

## 📋 实验配置

### 默认配置

- **GAT Layers**: 2层
- **GAT Heads**: 4个头
- **Temporal Window**: 3
- **Spatial TopK**: 4
- **Learning Rate**: 1e-4
- **DINOv2 Unfreeze**: 1（解冻最后一层）
- **K-Fold**: 8

### 超参数敏感性

通过大量实验验证了不同超参数对性能的影响：

- **GAT层数**: 2层最佳，更深网络容易过拟合
- **学习率**: 1e-4最优，过小收敛慢，过大不稳定
- **解冻策略**: 解冻DINOv2层至关重要，冻结时性能下降5.07%

## 🔍 详细文档

- **[架构设计文档](src/ARCHITECTURE.md)**：详细的架构设计说明
- **[实验结果](src/RESULTS.md)**：完整的实验结果和分析
- **[配置说明](src/config.py)**：配置参数详细说明

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出改进建议。请遵循以下步骤：

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看[LICENSE](LICENSE)文件了解详情。

## 🙏 致谢

感谢以下开源项目的支持：
- [DINOv2](https://github.com/facebookresearch/dinov2) - 视觉特征提取
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) - 图神经网络
- [PyTorch](https://pytorch.org/) - 深度学习框架

## 📞 联系方式

- 项目维护者：[维护者姓名]
- 邮箱：[邮箱地址]
- 项目链接：[GitHub仓库地址]

---

**注意**：本项目以研究迭代为主，README优先反映当前可运行实现与配置。若后续改动配置/训练逻辑，请同步更新本文件。