# MSG_VIP 项目技术总览

> 目的：快速理解 `proj_msgvip` 的目录组织、配置体系、训练/推理流程，以及各模块之间的参数与数据流转关系，便于二次开发与调试。

## 1. 顶层目录速览

| 目录 | 说明 |
| --- | --- |
| `src/configs/` | 纯 Python 配置（`ExperimentConfig`）描述数据、模型、图、训练等所有超参。 |
| `src/train.py` | 训练入口脚本，解析 CLI、装配输出目录、初始化分布式与 `TrainingRuntime`。 |
| `src/data/` | 数据集与 DataLoader 封装（`MSGVIPDataset`、`create_dataloader`）。 |
| `src/features/` | 各模态特征提取器与多模态融合器。 |
| `src/graph/` | 图构建、边特征、GNN、记忆模块。 |
| `src/models/` | 核心 `MSGVIPModel` 及打分头。 |
| `src/training/` | 运行时、Trainer、训练/验证循环、损失与优化器封装。 |
| `src/evaluation/` | 评价器与指标计算。 |
| `src/utils/` | 日志、可视化、分布式评估、模型工具等。 |
| `outputs/` | 训练运行产出：ckpt、日志、可视化、预测等，按时间戳/实验名组织。 |

## 2. 配置系统 (`src/configs/config.py`)

- `ExperimentConfig` 聚合七大子配置：`data`、`model`（features/graph/memory/temporal/scoring）、`training`、`evaluation`、`logging`、`hardware`、`experiment` 元信息。
- 每个子配置使用 `@dataclass`，保证 IDE 类型提示、默认值、嵌套结构明确。
- CLI 仅覆盖关键训练参数（数据目录、batch、lr、epochs 等），其余仍由配置集中管理。`src/train.py` 会在解析后把 override 写回 `config.training`，后续模块直接读取。

## 3. 训练入口 (`src/train.py`)

1. 配置 CUDA/TensorRT 环境变量、显存分配策略。
2. 解析 CLI：支持数据路径、batch、lr、epochs、warmup、loss 相关权重、冻结模态等参数；新增 `--graph_aux_weight / --graph_aux_temperature / --graph_aux_detach_teacher` 用于控制图辅助 KL 蒸馏强度。
3. 初始化分布式上下文 `DistributedContext`（单/多卡自动兼容）。
4. 创建 `TrainingRuntime`，注入：参数、Project root、分布式信息、输出目录构建器、CUDA allocator。
5. `main()` 只负责 orchestration，具体流程交给 `TrainingRuntime.run()`。

示例运行命令：

```bash
python3 ./src/train.py --batch_size 128 --learning_rate 1e-4 --num_epochs 15 --num_workers 2 --data_ratio 1.0 --ranking_weight 0.3
```

## 4. 数据与加载 (`src/data/dataloader.py`)

- `MSGVIPDataset` 读取指定 split 下的 `.npz`，字段包含视频帧、bbox、person mask、target index、scene category 等。
- 训练阶段会随机打乱人物槽位，避免模型记住固定 ID。
- 预处理：
  - 帧转成 `float16` 并标准化到 `[0,1]`，形状变为 `(T, 3, H, W)`。
  - bbox 剪裁+归一化，mask 掩盖无效槽位。
- `create_dataloader` 统一构建，分布式时自动接入 `DistributedSampler`。

## 5. 模态特征模块 (`src/features/`)

| 模块 | 关键逻辑 |
| --- | --- |
| `AppearanceExtractor` | ROIAlign 裁剪人物，使用 TIMM MobileNetV3（可加载本地权重），输出 `(B,T,N,feature_dim)`，可冻结。 |
| `PoseExtractor` | ViTPose (state_dict .pth) + ST-GCN（原生 GPU 推理），产生关键点 embedding 及动作特征。 |
| `FaceFeatureExtractor` | YOLOv8-face 检测嘴部 + MobileNetV3 编码，附带头部姿态、说话活动、嘴部速度等辅助量。 |
| `SpatialExtractor` | 计算 bbox 几何特征与相对关系（距离、面积份额等）。 |
| `TrajectoryExtractor` | 通过 bbox 中心轨迹计算速度/加速度/方向/统计量，窗口聚合后输出轨迹 embedding 与 `motion_features`。 |
| `MultiModalExtractor` | 根据 `FeatureUsageConfig` 组装上述 extractor，输出各模态 embedding、辅助字典，并使用 concat/add/attention 融合获得 `fused` 节点特征。 |

## 6. 图构建与 GNN (`src/graph/`)

1. `GraphBuilder`
   - 通过 `RuleBasedEdgeBuilder` 基于规则（spatial/attention/synchronization/trigger）直接生成多关系邻接矩阵（稀疏 top-k），不做 learnable edge builder。
   - 时间建模不在图里做，统一交给后续的 TemporalMemory/HierarchicalMemory。
   - 先基于 `person_mask.any(dim=1)` 动态压缩有效人物，再在计算完成后用选择矩阵恢复到原始槽位，避免对空槽位构建 `N×N` 边（显著降低显存/算量）。
   - Edge builder 完成阈值、自环过滤与归一化/Mask，并只保留可学习部分；冗余的启发式与 `use_edge_*`/`use_vectorized_extraction` / `edge_types` 列表等配置项已删除，只保留 `edges.*` 布尔开关。
2. （已移除）`EdgeFeatureAggregator` / `EdgeFeatureFusion` / `EdgeMemoryModule`
   - 这条分支在最终链路中已剔除，避免引入额外变量与训练不稳定。
3. `RGTRefiner`（Graph Transformer）
   - 每个边类型分配一组 GAT 层，输出后按照 concat/add/attention 聚合，并与残差投影相加。

## 7. 模型前向 (`src/models/msgvip_model.py`)

典型流程：

```
frames, bboxes, person_mask
   ↓ MultiModalExtractor → fused + {appearance, pose, face, spatial, trajectory}
   ↓ _prepare_graph_features() 抽取 head_directions / speech / action / motion → GraphBuilder
a d j a c e n c i e s ───┐
                         ├─ EdgeFeatureAggregator → EdgeMemoryModule → EdgeFeatureFusion
fused node feats ────────┘
   ↓ RGTRefiner（replace 语义固定，config 不再暴露 mode）
   ↓ TemporalMemory (GRU / Hierarchical)
   ↓ VideoLevelAggregator (attention pooling)
   ↓ ScoringHead (attention/MLP，多 head，可加入 speaking stats)
   ↓ importance logits/scores
```

- `apply_feature_freeze_policy` 根据配置冻结/解冻模态 extractor。
- 输出包含 logits、softmax scores、视频级特征、更新后的短期记忆。

### 7.1 🧩 拿到特征后，后处理到底做了什么？（关键细节）

这里把“特征提取完成之后”的主干链路按**输入/输出张量形状**与**mask 语义**拆开说清楚，方便你定位 Rank@1 平台化到底是卡在“特征 / 图 / RGT / memory / scoring”哪一段。

#### A. 统一约定：$N$ 是 padded slots

- DataLoader 会把每帧的人数 pad 到固定 $N=\texttt{config.data.max_persons}$。
- `person_mask` 标记哪些 slots 真正有效：
   - 常见形状：`person_mask` 为 `(B, T, N)`，bool。
   - 有效人数统计：`valid_counts = person_mask.sum(-1)`，得到 `(B,T)`。

这意味着：**任何对 $N\times N$ 的全量边槽位处理都会在实际有效人数远小于 $N$ 时浪费巨大。**（DDP OOM 就是典型后果。）

#### B. MultiModalExtractor 的输出（节点级）

MultiModalExtractor 输出结构是“一个主节点 embedding + 若干辅助特征字典”，核心节点张量一般是：

- `node_embeddings` / `fused`：形状 `(B, T, N, D_node)`
- 以及供规则边用的辅助特征（同样对齐 `(B,T,N,*)`）：
   - `head_directions`（注意力/视线边）
   - `speech_features`（trigger/response 边）
   - `action_features`、`motion_features`（同步边 / 边特征聚合）

所有这些张量都仍然包含 padded 的无效 slots；后续必须用 `person_mask` 过滤。

#### C. GraphBuilder：规则图构建（多关系邻接）

1) **压缩有效人物（可选但默认会做）**

- GraphBuilder 会从 `person_mask` 推导哪些人物在该 clip 内出现过：`valid = person_mask.any(dim=1)`，形状 `(B,N)`。
- 若有效人数 $N_{valid} < N$，会把每个 batch 的有效人物压缩到一个“紧凑的 $N_{valid}$ 轴”，再去算规则边，避免对全量 $N$ 做 $N\times N$。
- 算完后再用 selector matrix **展开回 full slot**，保证下游模块仍然拿到 `(B,T,N,N)` 形状的邻接矩阵（但无效区域被 mask 为 0）。

2) **RuleBasedEdgeBuilder 产出多关系邻接**

- 返回字典：`adjacency_matrices: Dict[str, Tensor]`，每个 edge_type 的邻接形状通常是 `(B, T, N, N)`
- 每类边是规则生成 + top-k 稀疏化 + mask：
   - `spatial`：距离/相对位置
   - `attention`：头朝向/视线
   - `synchronization`：运动一致性
   - `trigger`：语音触发/响应

3) **图统计落盘/日志**

- 每个 epoch（main process）会写入 `records/graph_stats.csv`（均度、密度等）。
- 当 `config.training.debug=True` 时，会额外打印 `[GraphStats] ...` 的 info 日志，便于肉眼快速发现图退化（例如密度趋近 0 或趋近 1）。

#### D. RGTRefiner：Graph Transformer（mode=replace）

RGT 的输入是节点特征 + 上一步的邻接（多关系）。关键点：

- **mode=replace**：输出的节点表示主要由图消息传递结果构成，而不是“原节点+一点点图修饰”。
- 内部会产生注意力矩阵/分布的统计（例如熵、最大概率、非零比例），每个 epoch 写入：`records/rgt_stats.csv`。

这一步能回答一个关键问题：

> 图是“有边但信息没走动”（注意力塌缩/极端尖锐）还是“边本身就不对”（候选边密度异常）？

#### E. EdgeMemory：从 $N^2$ 到 TopK 稀疏 edge slots（避免 OOM）

EdgeMemory 的核心目标是：给“人-人关系”引入时间记忆。但如果直接把 edge feature 做成 `(B,T,N,N,D)` 再展平，会变成 `N^2` 槽位，DDP 下极易爆显存。

当前实现已经收敛为：

- 只从 rule graph 中取 TopK（每个 node 取 K 个邻居），形成稀疏 edge slots
- 边槽总数约为 $E=N\cdot K$，而不是 $N^2$
- EdgeMemory 的输入变为 `(B, T, E, D_edge)`，并缓存 `last_edge_memory` 供调试

trainer 会把 edge slots 数、NaN 比例等写入 `records/debug_stats.csv`（默认每次训练都会写，用于快速定位问题来源）。

#### F. Temporal / Aggregation / Scoring：从节点到 clip-level 预测

1) Temporal memory（GRU / Hierarchical）

- 作用：把 `(B,T,N,D)` 变成带时间上下文的表示（仍对齐 person slots）。

2) VideoLevelAggregator

- 把时间维聚合成 clip-level 表示，输出一般是 `(B, N, D')` 或直接 `(B, N)` 的 logits 输入。

3) ScoringHead

- 输出每个 person slot 的 importance logits：`logits` 形状通常是 `(B, N)`
- `person_mask` 会在 loss/metrics 里屏蔽无效人（否则 padded slots 会污染指标）。

#### G. Loss/metrics 如何用 mask（避免 padded 污染）

训练时 CombinedLoss 会在计算 CE / ranking / contrastive 等损失时，使用 `person_mask` 仅对有效人槽位计入；验证 Rank@K 同理。

与定位最相关的 records：

- `records/train_metrics.csv`：rank@k / acc / loss
- `records/graph_stats.csv`：图稀疏度、度分布等
- `records/rgt_stats.csv`：注意力分布诊断
- `records/debug_stats.csv`（默认常开）：有效人数统计、edge slots 数、NaN 比例等

## 8. 训练运行时 (`src/training/`)

- `TrainingRuntime`
  1. `_prepare_environment()`：创建输出目录（run/checkpoints/logs/...），设置日志级别。
  2. `_load_and_override_config()`：把 CLI 传参写回 `ExperimentConfig`，检查 batch/world_size 兼容。
  3. `_prepare_dataloaders()`：可按 `--data_ratio` 子采样。
  4. `_build_trainer()`：实例化模型、`CombinedLoss`、优化器/调度器、`MemoryEfficientTrainer`。
  5. `_resume_if_needed()`、`_run_training_loop()`。
- `MemoryEfficientTrainer`
  - 负责混合精度、梯度累积、梯度裁剪、DDP 包装。
  - 调用 `train_epoch` / `validate_epoch`（`src/training/loops.py`）执行循环，并记录 batch/epoch 级指标、可视化、预测 CSV。
  - 内置 EarlyStopping、检查点保存、梯度/特征监控、Debug dump。
- `loss_functions.CombinedLoss` 聚合：重要度交叉熵、排序损失、对比损失、图辅助 KL 蒸馏（Graph aux logits 通过 softmax 与主路径 logits 做温度化 KL，对无效人槽自动 Mask），可通过配置/CLI 调整权重与温度。
- `optimizers` 提供 AdamW/SGD 等创建函数，并支持余弦/线性 warmup 调度。

## 9. 评估 & 指标 (`src/evaluation/`)

- `TrainingMetrics` 实时统计训练/验证准确率、Rank@K、mAP、Precision/Recall/F1、MRR 等。
- `MSGVIPEvaluator` 用于验证阶段，结合 `metrics.py` 的多种统计。
- `predictor.py`（若后续扩展推理服务）复用相同模型与数据管线。

## 10. 参数传递与覆盖链路

```
 src/configs/config.py -> ExperimentConfig default
    │
 src/train.py 解析 CLI
        │ 覆盖 data_dir/batch/lr/epochs/warmup/... 并标记 _data_dir_overridden
        ▼
 TrainingRuntime._load_and_override_config()
        │
  - 写入 per-rank batch / world_size
  - 可选：freeze_extractors、loss weights、logit temperature
        ▼
 Trainer / Model / Feature Extractor 直接读取 config
```

- 模态模块通过 `config.model.features.<modality>` 获取维度、窗口和模型路径。
- 图模块读取 `config.model.graph` 中的阈值、edges 开关、GNN 结构。
- 训练器依赖 `config.training` 控制混合精度、梯度累积、剪裁、调度器、早停等。

## 11. 完整流程串联

1. **准备阶段**：`train.py` 解析参数 → 初始化分布式 → `TrainingRuntime` 构建输出、日志、配置、DataLoader、Trainer。
2. **epoch 循环**：
   - `train_epoch`：取 batch → `MSGVIPModel` 前向 → `CombinedLoss` 反传 → 梯度裁剪/累积 → 记录指标/可视化。
   - 周期性校验：`validate_epoch` 计算指标、保存最佳模型，必要时早停。
3. **评估输出**：在 `outputs/<run>/predictions`、`records`、`visualizations` 下固化预测、指标曲线、诊断信息。

## 12. 快速排错建议

- **数据问题**：`MSGVIPDataset` 对缺失字段会抛 `ValueError`，可先在 `data/shared_data/preprocessed_fixed/<split>` 下检查 NPZ 完整性。
- **CUDA / PyTorch（原生 GPU）**：ViTPose/YOLO 依赖 GPU；ViTPose 期望使用 state_dict 风格的 checkpoint（主流 .pth），不再依赖序列化的 nn.Module，若初始化失败请核实 `data/models/...` 权重是否存在、CUDA 与 PyTorch 版本是否匹配。
- **分布式**：确保 `batch_size` 能被 `WORLD_SIZE` 整除；`DistributedContext` 会强制 per-rank batch。
- **边构建**：若缺少规则边所需的特征（例如同步边依赖 `motion_features`），应直接报错暴露问题；项目默认不做“静默 fallback 造边/补零”。

---

通过以上梳理，可将 MSG_VIP 看作“多模态特征 → 多通道图 → 记忆增强 → 视频级评分”流水线，配置集中在 `ExperimentConfig`，训练入口则通过 `TrainingRuntime` 统一调度。