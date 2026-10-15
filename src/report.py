"""Result plots for one run directory.  Run: python -m src.report reports/raion"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import KYIV_CITY, REPORTS_DIR
from .eda import GRID, INK_2, SURFACE  # noqa: F401  (also applies the shared rcParams)

# Fixed slot per model, so a model keeps its color in every chart.
MODEL_COLORS = {"persistence": "#eb6834", "seasonal": "#1baf7a", "gbm": "#2a78d6", "gbm_text": "#4a3aa7"}
LABELS = {"persistence": "persistence", "seasonal": "seasonal (28d)", "gbm": "GBM", "gbm_text": "GBM + Telegram"}


def plot_auc(metrics: pd.DataFrame, out: Path) -> None:
    models = [m for m in MODEL_COLORS if m in set(metrics["model"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (subset, title) in zip(axes, [("all", "All hours"), ("quiet_now", "No alert at forecast time")]):
        d = metrics[metrics["subset"] == subset].pivot(index="H", columns="model", values="roc_auc")
        x = np.arange(len(d.index))
        width = 0.8 / len(models)
        for i, m in enumerate(models):
            ax.bar(x + (i - (len(models) - 1) / 2) * width, d[m], width=width * 0.92,
                   color=MODEL_COLORS[m], label=LABELS[m])
        ax.set_xticks(x, [f"{h} h" for h in d.index])
        ax.set_title(title)
        ax.set_xlabel("forecast horizon")
        ax.set_ylim(0.75, 1.0)
    axes[0].set_ylabel("ROC-AUC (test period)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncols=len(models), fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "results_auc.png", dpi=150)
    plt.close(fig)


def plot_calibration(preds: pd.DataFrame, out: Path) -> None:
    model = "gbm_text" if "gbm_text" in preds else "gbm"
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], color=INK_2, linestyle="--", linewidth=1, label="perfect")
    for H, color in zip([1, 6, 24], ["#2a78d6", "#eb6834", "#1baf7a"]):
        g = preds[preds["H"] == H]
        bins = pd.cut(g[model], np.linspace(0, 1, 11), include_lowest=True)
        cal = g.groupby(bins, observed=True).agg(p=(model, "mean"), y=("y", "mean"))
        ax.plot(cal["p"], cal["y"], marker="o", markersize=5, linewidth=2, color=color, label=f"{H} h")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(f"Calibration, {LABELS[model]}")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "results_calibration.png", dpi=150)
    plt.close(fig)


def plot_kyiv_week(preds: pd.DataFrame, out: Path) -> None:
    model = "gbm_text" if "gbm_text" in preds else "gbm"
    g = preds[(preds["H"] == 1)].xs(KYIV_CITY, level="unit").sort_index()
    g = g[g.index >= g.index.max() - pd.Timedelta(days=7)]
    t = g.index.tz_convert("Europe/Kyiv")
    period = f"{t.min():%d %b} – {t.max():%d %b %Y}"
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.fill_between(t, 0, g["y"], step="post", color="#e34948", alpha=0.18, linewidth=0,
                    label="alert during the next hour (actual)")
    ax.step(t, g[model], where="post", color=MODEL_COLORS[model], linewidth=1.6,
            label=f"{LABELS[model]}: P(alert within 1 h)")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("probability")
    ax.set_title(f"Kyiv city, {period}: 1-hour-ahead forecast vs actual alerts")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9, ncols=2)
    fig.tight_layout()
    fig.savefig(out / "results_kyiv_week.png", dpi=150)
    plt.close(fig)


def main(out: Path) -> None:
    metrics = pd.read_csv(out / "metrics.csv")
    preds = pd.read_csv(out / "predictions.csv.gz", index_col=["time", "unit"], parse_dates=["time"])
    plot_auc(metrics, out)
    plot_calibration(preds, out)
    plot_kyiv_week(preds, out)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPORTS_DIR / "raion")
