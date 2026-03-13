# SVIP - Social Video Importance Prediction

视频中重要人物识别与排序系统（MSG_VIP），基于 DINOv2 + 几何特征 + 图建模 + 时序建模。

## 项目概述

当前主干模型流程：

1. 输入 `frames/bboxes/person_mask/frame_mask`
2. 从人框 ROI 提取 DINOv2 视觉特征
3. 提取 BBox 几何特征并融合
4. GATv2 做逐帧社交关系建模
5. Temporal Transformer + Event Token 建模时序上下文
6. Video-level attention pooling 聚合
7. Dual-head 打分（self + rel）并 gate 融合
8. 输出人物重要性 logits 和排序

损失函数当前为：
- `importance_loss + preference_loss`

## 当前代码状态（与实现一致）

- Dual-head 开启（`enable_dual_head=True`）
- 激活重计算开启（`activation_checkpointing=True`）
- `max_persons=16`
- `video_length=120`（可改，且现在会真实触发时序重采样）
- `save_checkpoints=False`（默认不保存 `last.pt/best.pt`）

参考配置文件：`src/configs/config.py`

## 训练命令

单卡示例：

```bash
CUDA_VISIBLE_DEVICES=1 python src/train.py --batch_size 4 --num_epochs 10 --data_ratio 1.0 --accumulation_steps 16
```

双卡 DDP 示例：

```bash
torchrun --nproc_per_node=2 src/train.py --batch_size 4 --num_epochs 10 --data_ratio 1.0 --accumulation_steps 16
```

常用参数：
- `--batch_size`
- `--accumulation_steps`
- `--num_epochs`
- `--data_ratio`
- `--roi_chunk`
- `--learning_rate`
- `--num_workers`
- `--early_stop`
- `--importance_weight`
- `--preference_weight`
- `--logit_temperature`

## 时序下采样（T 控制）

现在 `config.data.video_length` 会在 dataloader 中实际生效。

实现位置：`src/data/dataloader.py`

- `self.num_frames = config.data.video_length`
- 若样本原始 `T != num_frames`，触发 `_resample_temporal_arrays(...)`
- 同步重采样字段：
  - `frames`
  - `bboxes`
  - `person_ids`
  - `person_mask`
  - `frame_mask`
- 采样方式：确定性均匀采样（`np.linspace + round`）
- 日志只打印一次（train split, rank0, worker0）

## 显存相关优化（当前已接入）

1. CUDA allocator 默认开启：
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,expandable_segments:True`
- 位置：`src/train.py`

2. ROI 路径优化：
- 仅对有效人物做 ROI crop（valid-only）
- DINO 前向分块（`roi_chunk`）
- 位置：`src/models/importance_ranker.py`

3. Activation checkpointing：
- 对时序高峰模块做重计算换显存
- 位置：`src/models/importance_ranker.py`

## Checkpoint 保存策略

配置项：`training.save_checkpoints`

- `False`（当前默认）：不保存 `last.pt` / `best.pt`
- `True`：
  - 每个 epoch 覆盖更新 `last.pt`
  - val 提升时更新 `best.pt`

实现位置：`src/training/trainer.py`

## 输出目录

每次运行会创建：

- `outputs/<timestamp>/logs/`
- `outputs/<timestamp>/records/`
- `outputs/<timestamp>/visualizations/`
- `outputs/<timestamp>/predictions/`
- `outputs/<timestamp>/configs/run_config.json`
- `outputs/<timestamp>/checkpoints/`（仅当 `save_checkpoints=True` 时有模型文件）

## 项目结构

```text
src/
  configs/config.py
  data/dataloader.py
  models/
    importance_ranker.py
    vision_encoder.py
    bbox_geom.py
    gatv2.py
    event_context.py
    video_aggregator.py
  training/
    runtime.py
    trainer.py
    loops.py
    loss.py
  train.py
```

## 说明

- 本仓库以研究迭代为主，README 优先反映当前可运行实现与配置。
- 若后续改动配置/训练逻辑，请同步更新本文件。
