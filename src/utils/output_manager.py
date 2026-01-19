"""Output manager.

Centralizes output artifacts under a run directory:
- records (metrics.csv/jsonl)
- predictions (CSV per split/epoch)

Trainer/loops should call into this module rather than implementing file formats.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


class OutputManager:
    def __init__(self, *, records_dir: Optional[Path], predictions_dir: Optional[Path]) -> None:
        self.records_dir = Path(records_dir) if records_dir is not None else None
        self.predictions_dir = Path(predictions_dir) if predictions_dir is not None else None

        if self.records_dir is not None:
            self.records_dir.mkdir(parents=True, exist_ok=True)
            self._csv_path = self.records_dir / "losses.csv"
            self._json_path = self.records_dir / "metrics.jsonl"
        else:
            self._csv_path = None
            self._json_path = None

        self._csv_header_written = bool(self._csv_path and self._csv_path.exists())
        self._csv_header: Optional[list[str]] = None
        if self._csv_header_written and self._csv_path is not None:
            header_line = self._csv_path.open("r", encoding="utf-8").readline().strip()
            self._csv_header = header_line.split(",") if header_line else None

        self._classwise_dir: Optional[Path] = None
        if self.predictions_dir is not None:
            self._classwise_dir = self.predictions_dir / "classwise"

    @staticmethod
    def _round4(x: Any) -> Any:
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return round(x, 4)
        return x

    def log_losses(self, payload: Dict[str, Any]) -> None:
        if self.records_dir is None or self._csv_path is None:
            return

        flat = {k: v for k, v in payload.items() if isinstance(v, (int, float, str)) or v is None}
        if not flat:
            return

        flat = {k: self._round4(v) for k, v in flat.items()}
        flat.setdefault("epoch", None)

        if not self._csv_header_written:
            header = list(flat.keys())
            with self._csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerow(flat)
            self._csv_header_written = True
            self._csv_header = header
            return

        header = self._csv_header or list(flat.keys())
        with self._csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writerow({k: flat.get(k) for k in header})

    def log_metrics(self, payload: Dict[str, Any]) -> None:
        if self.records_dir is None or self._json_path is None:
            return

        payload = dict(payload)
        payload.setdefault("epoch", None)
        payload = {k: self._round4(v) for k, v in payload.items()}

        with self._json_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_epoch(self, payload: Dict[str, Any]) -> None:
        if self.records_dir is None or self._csv_path is None or self._json_path is None:
            return

        self.log_metrics(payload)

    def export_predictions_csv(
        self,
        *,
        split: str,
        epoch: int,
        predicted_index: torch.Tensor,
        targets: torch.Tensor,
        video_ids: List[str],
        scene_categories: List[str],
        person_ids: Optional[List[List[int]]],
        rankk_cache: Dict[int, float],
    ) -> None:
        if self.predictions_dir is None:
            return

        out_dir = self.predictions_dir / str(split)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"epoch{int(epoch):03d}.csv"

        pred_cpu = predicted_index.detach().long().cpu()
        targets_cpu = targets.detach().long().cpu()

        B = int(targets_cpu.shape[0])
        if len(video_ids) < B:
            video_ids = list(video_ids) + [""] * (B - len(video_ids))
        if len(scene_categories) < B:
            scene_categories = list(scene_categories) + [""] * (B - len(scene_categories))
        if person_ids is None or len(person_ids) < B:
            person_ids = (list(person_ids) if person_ids is not None else []) + [None] * (B - (len(person_ids) if person_ids is not None else 0))

        correct_count = 0
        valid_sample_count = 0

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "video_id",
                    "scene_category",
                    "predicted_index",
                    "target_index",
                    "predicted_person_id",
                    "target_person_id",
                    "correct",
                ]
            )

            for i in range(B):
                target_idx = int(targets_cpu[i].item())

                predicted_idx = int(pred_cpu[i].item())

                predicted_person_id = ""
                target_person_id = ""
                pid_seq = person_ids[i]
                if pid_seq is not None:
                    if 0 <= predicted_idx < len(pid_seq):
                        predicted_person_id = pid_seq[predicted_idx]
                    if 0 <= target_idx < len(pid_seq):
                        target_person_id = pid_seq[target_idx]

                valid_sample_count += 1
                if predicted_idx >= 0 and predicted_idx == target_idx:
                    correct_count += 1
                    correct_flag = 1
                else:
                    correct_flag = 0

                writer.writerow(
                    [
                        video_ids[i],
                        scene_categories[i],
                        predicted_idx if predicted_idx >= 0 else "",
                        target_idx if target_idx >= 0 else "",
                        predicted_person_id,
                        target_person_id,
                        correct_flag,
                    ]
                )
            total_samples = B
            invalid_sample_count = total_samples - valid_sample_count
            acc_value = correct_count / valid_sample_count if valid_sample_count > 0 else float("nan")
            writer.writerow(
                [
                    "SUMMARY_TOTALS",
                    (
                        f"valid_sample_count={valid_sample_count}; correct={correct_count}; acc={acc_value:.4f}; invalid={invalid_sample_count}"
                    ),
                ]
            )

            r1 = float(rankk_cache.get(1, 0.0))
            r2 = float(rankk_cache.get(2, 0.0))
            r3 = float(rankk_cache.get(3, 0.0))
            writer.writerow(["SUMMARY_RANKS", f"rank@1={r1:.4f}; rank@2={r2:.4f}; rank@3={r3:.4f}"])

    def log_classwise_metrics(self, epoch: int, class_metrics: Dict[str, Dict[str, float]] | None) -> None:
        if self.predictions_dir is None or not class_metrics:
            return

        class_dir = self._classwise_dir
        class_dir.mkdir(parents=True, exist_ok=True)

        for class_name, metrics in sorted((class_metrics or {}).items()):
            normalized = str(class_name).replace(" ", "_").lower()
            file_path = class_dir / f"{normalized}_metrics.csv"
            is_new = not file_path.exists()

            def safe_percent(v: Any) -> float:
                x = float(v)
                return round(x, 3) if math.isfinite(x) else float("nan")

            with file_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if is_new:
                    writer.writerow([
                        "epoch",
                        "rank@1",
                        "rank@2",
                        "rank@3",
                        "rank1_correct_count",
                        "total_count",
                    ])

                writer.writerow(
                    [
                        int(epoch),
                        safe_percent(metrics.get("rank@1")),
                        safe_percent(metrics.get("rank@2")),
                        safe_percent(metrics.get("rank@3")),
                        int(metrics.get("rank1_correct", 0) or 0),
                        int(metrics.get("total", 0) or 0),
                    ]
                )
