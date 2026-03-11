"""Training-time visualization (loss/acc/rank curves).

Kept close to src-ref behavior: produces pngs under output_paths['visualizations'].
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import logging

import matplotlib

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.pyplot").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cycler
import numpy as np


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_metric_plots(
    visualization_dir: Optional[Path],
    epoch: int,
    *,
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    train_accs: Sequence[float],
    val_accs: Sequence[float],
    rank1_values: Sequence[float],
    rank2_values: Sequence[float],
    rank3_values: Sequence[float],
    main_process: bool,
    logger,
    plot_validation: bool = False,
) -> None:
    if visualization_dir is None or not main_process or not train_losses:
        return

    try:
        _ensure_dir(visualization_dir)
    except Exception as exc:
        logger.warning("Failed to prepare visualization directory: %s", exc)
        return

    epochs = np.arange(1, len(train_losses) + 1, dtype=np.int32)
    train_losses_arr = np.asarray(train_losses, dtype=np.float64)
    val_losses_arr = np.asarray(val_losses, dtype=np.float64) if (plot_validation and val_losses) else np.array([], dtype=np.float64)
    train_accs_arr = np.asarray(train_accs, dtype=np.float64) if train_accs else np.array([], dtype=np.float64)
    val_accs_arr = np.asarray(val_accs, dtype=np.float64) if val_accs else np.array([], dtype=np.float64)
    rank1_arr = np.asarray(rank1_values, dtype=np.float64) if rank1_values else np.array([], dtype=np.float64)
    rank2_arr = np.asarray(rank2_values, dtype=np.float64) if rank2_values else np.array([], dtype=np.float64)
    rank3_arr = np.asarray(rank3_values, dtype=np.float64) if rank3_values else np.array([], dtype=np.float64)

    palette = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]
    style_context = {
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.prop_cycle": cycler("color", palette),
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
    }

    def save_line_plot(
        filename: str,
        series: List[Tuple[str, np.ndarray, str]],
        ylabel: str,
        *,
        ylim: Optional[Tuple[float, float]] = None,
        percent: bool = False,
    ) -> None:
        with plt.rc_context(style_context):
            fig, ax = plt.subplots(figsize=(11, 4.2), dpi=220)
            has_data = False
            for label, values, color in series:
                if values.size == 0:
                    continue
                mask = np.isfinite(values)
                if not mask.any():
                    continue
                ax.plot(epochs[mask], values[mask], label=label, linewidth=2.0, marker="o", markersize=4, color=color)
                has_data = True
            if not has_data:
                plt.close(fig)
                return
            ax.set_xlabel("Epoch")
            ax.set_ylabel(f"{ylabel}{' (%)' if percent else ''}")
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.set_title(f"{ylabel} vs. Epoch")
            if len(series) > 1:
                ax.legend(loc="best")
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_xlim(left=0.5)
            fig.tight_layout()
            fig.savefig(visualization_dir / filename, bbox_inches="tight")
            plt.close(fig)

    loss_series = [("Train Loss", train_losses_arr, "#1f77b4")]
    if plot_validation and val_losses_arr.size and np.isfinite(val_losses_arr).any():
        loss_series.append(("Val Loss", val_losses_arr, "#d62728"))
    save_line_plot("loss_curve.png", loss_series, "Loss")

    acc_series = [
        ("Train Acc", train_accs_arr, "#2ca02c"),
        ("Val Acc", val_accs_arr, "#9467bd"),
    ]
    save_line_plot("accuracy_curve.png", acc_series, "Accuracy", ylim=(0, 1.0))

    if rank1_arr.size and np.isfinite(rank1_arr).any():
        save_line_plot("rank1_curve.png", [("Rank@1", rank1_arr, "#1f77b4")], "Rank@1", ylim=(0, 100), percent=True)
    if rank2_arr.size and np.isfinite(rank2_arr).any():
        save_line_plot("rank2_curve.png", [("Rank@2", rank2_arr, "#17becf")], "Rank@2", ylim=(0, 100), percent=True)
    if rank3_arr.size and np.isfinite(rank3_arr).any():
        save_line_plot("rank3_curve.png", [("Rank@3", rank3_arr, "#9467bd")], "Rank@3", ylim=(0, 100), percent=True)
