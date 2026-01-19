# MSGVIP 图关系（Edges）重构设计方案（Rule-based + 可学习/Transformer）

日期：2025-12-15

---

## 2025-12-16 更新：目录证据、硬基线与“最终定版（精简）架构”

这一节用于记录我们在 2025-12-16 的关键结论：**不再以“多开关可选”为目标，而是以“效果优先 + 代码精简 + 可诊断证据充分”为目标**。后续将按此节进行代码级推倒重构（删除冗余路径）。

### 1) 满数据硬基线（必须超过）

从日志 `outputs/logs/graph-identity-fulldata` 解析得到：

- Val Rank@1 best = **63.28**
- Val Rank@1 last = **58.26**

后续 full-data 新方案至少要稳定超过 **58.26（val_last）**，并以超过 **63.28（val_best）** 为第一阶段门槛。

### 2) 8 个实验的输出目录证据（不看曲线，直接看 records）

我们以 `outputs/diagnostics/diagnose_summary.csv` 为索引，进入对应 `run_dir/records/graph_stats.csv` 检索图统计，得到的最关键现象是：**图会在训练后期出现 0 边/极稀疏/边类型塌缩**。

代表性证据（0.05/5epoch/b32，主要用于结构诊断，不用于与 full-data 数值横比）：

- `baseline_full_rule`：last epoch `latest num_edges=0`，且训练过程中存在 `edges_zero_epochs=1`
- `baseline_full_hybrid`：last epoch `latest num_edges=0`，`edges_zero_epochs=2`
- `baseline_full_rule_rgt_refiner_gate_bias`：last epoch `latest num_edges=0`，`edges_zero_epochs=4`
- `baseline_full_rule_rgt_replace_gate_bias`：last epoch `latest num_edges=5`（极稀疏）

同时，多数 run 的 last epoch 边类型统计中，边会**塌缩为单一 trigger**（spatial/attention/synchronization 缺失或接近 0）。

### 3) 可证伪根因归因（为何准确率不涨/不稳）

第一性根因不是“GNN vs Transformer 之争”，而是：

1) 边构建与后处理（topk/threshold/normalize/门控）组合在训练过程中会使 adjacency 退化为全 0 或极稀疏；
2) 图退化后跨人关系推理等价于“无消息传递”，模型回退为仅靠节点自身特征 + 时序；
3) 因此 Rank@1 不涨或 best/last 极不稳定。

验证方式：无需看曲线，只需检查 `records/graph_stats.csv` 的 `num_edges/density/edge_type` 是否出现“归零/塌缩”。

### 4) 最终定版（精简）架构：以效果为优先

特征提取后主链路（固定，不再提供多套可选）：

1) **Rule edges**：只负责产生候选/权重（不做传播）；保证“仅在有效人物 mask 内产生边”。
2) **RGT（Relational Graph Transformer）**：作为默认且唯一跨人消息传递模块（倾向 replace 模式），以关系边驱动注意力传播。
3) **Temporal/Memory**：跨时间累积证据（VIP 的证据是时间连续的）。
4) **Video-level aggregator**：将时序证据聚合为最终 per-person 决策表征。
5) **Scoring head**：输出每个人的重要性分数（节点自身 + 关系已被写入表征）。

工程约束：

- 不写“fallback 造边”逻辑（允许某些样本/某些关系无边），但必须输出诊断证据。
- 将移除 learnable/hybrid edge builder、feature_only/memory_only/graph_only 等冗余分支与 CLI 开关。

### 5) 重构后必须保留的最小诊断记录（输出证据，不改变行为）

重构后仍必须写入 `outputs/<run>/records/`：

- `train_metrics.csv`：每 epoch 的 rank@k / loss
- `graph_stats.csv`：每 epoch 的 num_edges/density/avg_degree + edge_type 占比 + 0 边出现次数
- `rgt_stats.csv`：每 epoch 的 attention 有效候选比例、注意力熵/尖锐度（用于定位“注意力塌缩/无邻居传播”）

---

## 背景与结论（来自诊断矩阵）

在 `data_ratio=0.1, epochs=8` 的可控消融诊断中：

- `graph_only(normal + topk16)` 长期低位横盘（best Rank@1≈28），说明“跨人 message passing + 当前 learnable edges”系统性注入噪声。
- `graph_identity` 曲线健康且最终更高（Rank@1≈53），反证 GNN 计算栈本身并非原罪；主要问题集中在 **跨人边的构建与标定**。
- edge-only 消融显示 `only_attention` 最差，提示 attention 通道最可能在融合中主导并污染节点表征。

因此：

> 本次重构目标不是“逃避图”，而是用**正确的关系逻辑**构边，并在工程上强约束图的稀疏度、稳定性与可诊断性，使跨人关系成为正贡献。

---

## 设计目标（Design Goals）

1. **可解释**：四类边（空间 / 注视注意力 / 同步跟随 / 触发因果）具有明确物理或语义含义。
2. **可控**：每类边输出的稀疏度、数值范围、时间汇总策略可配置、可复现。
3. **稳定训练**：避免“全连接中等权重 + row normalize”导致的噪声扩散；强制 top-k、边权尖锐化、self-loop 优先。
4. **可诊断**：记录每类边的统计（度、权重分布、topk 命中率/漂移），支持 ablation（only/disable）。
5. **兼容现有系统**：不推倒现有 GNN/Memory，只替换 edge builder；保持 `GraphBuilder` 输出格式 `Dict[str, Tensor(B,T,N,N)]`。

---

## “边建模契约”（Edge Modeling Contract）

### 输入

- `bboxes`: (B, T, N, 4)  
- `person_mask`: (B, T, N)
- 预先提取/整理的图特征（来自 `MultiModalExtractor`+图特征准备阶段）：
  - `head_directions`: (B, T, N, 3) 头部朝向（单位向量或可归一化）
  - `speech_features`: (B, T, N, 2) 说话相关（如能量/概率）
  - `action_features`: (B, T, N, D_pose) 动作/姿态相关
  - `motion_features`: (B, T, N, D_traj) 轨迹/运动相关
  - （可选）`spatial_features`: 如相对位置、尺度、场景坐标等

### 输出

- 多通道邻接 `adjacency_matrices: Dict[str, Tensor]`
  - 每个通道 shape: (B, T, N, N)
  - 值域: [0, 1]
  - 有向/无向：
    - spatial: 无向（对称）
    - attention/gaze: 有向 i→j（i 看 j）
    - synchronization: 无向或双向
    - trigger/causal: 有向 i→j（i 触发 j）

### 稀疏化（Hard Constraint）

- 每个节点每帧最多保留 `topk` 条出边（不含 self-loop）；`topk` 建议 4~8。
- 对于无向边，采用对称化：保留 (i,j) 或 (j,i) 后对称填充。

### 时间汇总

支持两级策略（按任务选择）：

- **frame-level adjacency**：每帧构边，GNN 自己在 T 维处理。
- **window/EMA 汇总**：对 `A_{t}` 做 EMA/mean/max，适用于持续关系（跟随/同步/对话）。

### 归一化

- 明确优先级：`self-loop` > inter-person edges。
- 建议：边构建先输出未归一化权重，后处理阶段再用 row-normalize/softmax，但必须在稀疏化后。

---

## 路线 A：规则/物理启发的四类边（第一优先落地）

### 1) Spatial（空间交互先验，建议无向）

每帧计算：

- bbox center: $c_i=(x_i,y_i)$
- 距离核：$w_d=\exp(-\|c_i-c_j\|^2/\sigma^2)$
- 尺度兼容：$w_s=\exp(-|\log(area_i/area_j)|)$
- IoU：$w_{iou}=IoU(box_i,box_j)$

合成：

$$A^{spatial}_{ij}=\text{clip}(\alpha w_d + \beta w_{iou} + \gamma w_s, 0, 1)$$

再进行 top-k 稀疏化与对称化。

### 2) Attention / Gaze（注视边，有向）

用几何直接计算“i 在看向 j”的强度：

- head direction：$v_i$（归一化）
- 指向 j 的方向：$u_{i\to j}=\frac{c_j-c_i}{\|c_j-c_i\|}$
- 夹角相似：$s=\max(0, v_i\cdot u_{i\to j})$
- 距离衰减：$g=\exp(-d_{ij}/\tau)$

$$A^{attn}_{i\to j}=s^{\eta}\cdot g$$

再对每个 i 做 top-k。

时间汇总（可选）：
- `mean`：频率
- `EMA`：持续性

### 3) Synchronization / Follow（同步/跟随，建议无向/双向）

基于运动学一致性：

- 从 bbox center 得到速度向量 $\Delta c_i(t)=c_i(t)-c_i(t-1)$
- 方向一致：$\text{cos}(\overline{\Delta c}_i, \overline{\Delta c}_j)$

$$A^{sync}_{ij}=ReLU(cos)\cdot \exp(-d_{ij}/\tau)$$

跟随（有向）可选：用滞后相关

$$A^{follow}_{i\to j}=\max_{\Delta\in[1..L]} corr(\Delta c_i(t), \Delta c_j(t-\Delta))$$

### 4) Trigger / Causal（触发因果，有向）

基于事件检测 + 时序窗口统计：

- speech onset：$E^{spk}_i(t)$
- action onset：$E^{act}_i(t)$
- response(被触发)：例如 j 的 speech onset/动作增长

$$A^{trig}_{i\to j}\propto\sum_t E_i(t)\cdot R_j(t+\Delta)$$

并支持惩罚“独立发生”以降低伪因果。

---

## 路线 B：Transformer/关系学习（第二阶段）

核心原则：**规则边提供候选邻居与 bias，Transformer 学细粒度权重**。

### Graph Transformer with Bias

对每个 t：

- $Q_i=W_Q h_i, K_j=W_K h_j, V_j=W_V h_j$
- logits：$l_{i\to j}=(Q_i\cdot K_j)/\sqrt{d}+b(i,j)$
  - $b(i,j)$ 来自规则边（距离、视线、互动先验）

并仅在候选集 top-k 上做 softmax，实现稀疏 attention。

---

## 与现有 MSGVIP 对接方案

1. 保留：`GraphBuilder` 输出字典、`only_edge_types/disable_edge_types`、`adjacency_mode`、`_post_process_adjacency`。
2. 新增：`RuleBasedEdgeBuilder`（替代 `LearnableEdgeBuilder`），并作为默认且唯一边构建实现。
3. 训练流程不变：GNN/Mem/Head 不动，先验证“正确边 + message passing”是否稳定增益。

### 最小验证矩阵（建议）

- `graph_only + rule_edges`：验证跨人关系是否变正贡献
- `baseline_full + rule_edges`：验证 full pipeline 是否不再崩
- `only_attention(rule)`：验证 attention 通道从最毒变为有益/至少不拖后腿

---

## 工程要求（Enterprise Quality Gates）

- 单元测试：形状、值域、mask、对角线、自环、稀疏度。
- 记录：每个通道的 avg_degree/density/avg_weight/topk_keep_ratio。
- 可复现：所有超参入 `run_meta/run_config`；诊断脚本可一键对比。
