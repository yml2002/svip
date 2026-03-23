# MSG_VIP 模型架构与实验说明

## 模型概述

视频重要人物识别（Most Significant Person in Video）。给定一段视频中多个被跟踪的人物，预测谁是最重要的人。

## 架构设计

### 整体流程

```
输入: frames (B,T,3,H,W) + bboxes (B,T,N,4) + person_mask (B,T,N)
  │
  ├─ DINOv2 ROI Crop → 每个人的视觉特征 (B,T,N,768)
  ├─ BBoxGeom Encoder → 每个人的几何特征 (B,T,N,128)
  │     含: 位置、大小、面积、位移、速度、加速度 (14维 → MLP → 128维)
  └─ Feature Fusion → fused (B,T,N,768)
       │
       ├─ Self Branch: temporal mean pool → scoring head → self_scores (B,N)
       │     独立看每个人的外观+几何，不看其他人
       │
       └─ Relation Branch: Spatio-Temporal GATv2 → temporal attn pool → scoring head → rel_scores (B,N)
             看人物之间的时空交互关系
             │
             最终: logits = self_scores + rel_scores → softmax → 排序
```

### Self 分支

- 输入: fused 特征 (B,T,N,768)
- 时间维度 masked mean pooling → (B,N,768)
- Scoring head: LayerNorm → Linear(768,256) → ReLU → Dropout → Linear(256,1)
- 输出: 每个人的独立重要性分数
- **核心**: 完全逐人独立，不看其他人

### Relation 分支 — 时空联合图 (Spatio-Temporal GATv2)

#### 图结构
- **节点**: 每个 (person_i, frame_t) 是一个节点
- **空间边 (intra-frame)**: 同一帧内人物之间，基于 bbox 中心距离的 Top-K 近邻
- **时间边 (inter-frame)**: 同一人在相邻帧之间，窗口大小可配置

#### 边特征
- **空间边特征** (6维): delta_cx, delta_cy, delta_w, delta_h, center_dist, IoU
- **时间边特征** (4维): delta_cx, delta_cy, delta_area, speed
- 边特征通过 MLP 投影后注入 GATv2Conv 的注意力计算

#### 消息传递
- GATv2Conv × 2 layers, 4 heads, residual connection + LayerNorm
- 每层: node_feat → GATv2(node_feat, edge_index, edge_attr) → ELU → Dropout → LayerNorm(residual)

#### 时间聚合
- Learned temporal attention pooling: per-person 在时间维度上加权求和
- 输出: (B, N, 512)
- Scoring head: LayerNorm → Linear(512,256) → ReLU → Dropout → Linear(256,1)

### 特征提取

- **DINOv2-Base**: 预训练视觉 backbone，默认解冻最后 1 层
- **BBoxGeom Encoder**: 14维几何特征 (x1,y1,x2,y2,cx,cy,w,h,area,dcx,dcy,disp,speed,accel) → MLP → 128维
- **Fusion**: cat(768+128) → Linear(896,768) → LayerNorm → ReLU → Dropout

### 损失函数

- CrossEntropyLoss 在最终 fused logits 上
- 无独立分支 CE（确保消融有效）
- 可选: PreferenceOptimizationLoss (默认关闭)

---

## 实验设计

### 运行方式

所有实验通过统一脚本 `src/run_experiments.py` 运行:

```bash
# 一次性运行全部实验（消融 + 多seed + 超参数）
python src/run_experiments.py all --batch_size 32 --num_epochs 15 --nproc_per_node 7

# 快速测试
python src/run_experiments.py all --batch_size 32 --num_epochs 1 --data_ratio 0.05

# 单独运行某类实验
python src/run_experiments.py ablation --batch_size 32 --num_epochs 15
python src/run_experiments.py seed --batch_size 32 --num_epochs 15
python src/run_experiments.py hyperparam --batch_size 32 --num_epochs 15

# 只运行特定实验
python src/run_experiments.py ablation --experiments full,self_only
python src/run_experiments.py hyperparam --experiments lr_1e5,lr_5e5,lr_1e4
```

### 实验 1: 消融实验 (Table — 6行)

| 实验 | 改动 | 回答的问题 |
|------|------|-----------|
| Full | 完整模型 | baseline |
| Self Only | 去掉 Relation 分支 | 时空关系建模的整体贡献 |
| w/o ST-Graph | GAT → MLP (去掉图结构) | 图结构 vs 简单投影 |
| w/o Temporal Edges | GAT 只有空间边 | 时序连接的贡献 |
| w/o Edge Features | GAT 有边但无几何边特征 | 边特征的贡献 |
| w/o Geometry | 去掉 BBoxGeom，纯视觉 | 几何特征的贡献 |

### 实验 2: 多 Seed 稳定性 (Table 1 的 mean±std)

- 实验: full 模型
- Seeds: [42, 3407, 2026]
- 报告: Rank@1/2/3 的 mean ± std

### 实验 3: 超参数敏感性 (Table/Figure)

| 超参数 | 测试值 | 默认值 |
|--------|--------|--------|
| GAT 层数 | {1, **2**, 3} | 2 |
| 注意力头数 | {2, **4**, 8} | 4 |
| 时间窗口 | {1, **3**, 5} | 3 |
| 空间 Top-K | {2, **4**, all} | 4 |
| 学习率 | {1e-5, 2e-5, **5e-5**, 1e-4, 2e-4} | 5e-5 |
| DINOv2 解冻层数 | {0(frozen), **1**} | 1 |

---

## 默认超参数

```yaml
# 数据
video_length: 120
max_persons: 16

# DINOv2
model: dinov2-base (768-d)
image_size: 196
freeze: false
unfreeze_layers: 1

# BBoxGeom
feature_dim: 128
spatial_edge_dim: 32
temporal_edge_dim: 16

# GATv2
hidden_dim: 512
num_layers: 2
heads: 4
topk_neighbors: 4
temporal_window: 3

# Scoring
hidden_dim: 256
temperature: 1.0

# Training
learning_rate: 5e-5
weight_decay: 5e-4
batch_size: 32 (× 7 GPUs)
num_epochs: 15
early_stop: 3
max_grad_norm: 3.0
mixed_precision: true
optimizer: AdamW
scheduler: CosineAnnealingLR
```

---

## 项目文件结构

```
src/
  configs/config.py          — 所有配置 dataclass
  data/dataloader.py         — NPZ 数据加载、时序重采样、slot 排列
  models/
    importance_ranker.py     — 主模型 (Self + Relation 分支)
    vision_encoder.py        — DINOv2 backbone wrapper
    bbox_geom.py             — 几何特征编码 + 边特征计算
    gatv2.py                 — 时空联合图 GATv2
  training/
    loss.py                  — CE + 可选 Preference loss
    loops.py                 — 训练/验证循环
    trainer.py               — Trainer (DDP, checkpoint, logging)
    runtime.py               — 训练运行时 (配置加载, 数据构建)
  train.py                   — 训练入口
  run_experiments.py         — 统一实验运行器 (消融/seed/超参数)
  run_ablation.py            — 旧版消融脚本 (保留兼容)
```
