"""Exploratory plots -> reports/eda_*.png.  Run: python -m src.eda"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import HISTORY_START, REPORTS_DIR
from .data import forecast_units, load_alerts, minute_activity

# Reference palette (dataviz skill), light mode; categorical slots in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def plot_levels_by_month(alerts: pd.DataFrame) -> None:
    """Shows the Nov 2025 switch from oblast-level to raion-level alerts."""
    m = alerts.assign(month=alerts["started_at"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp())
    counts = m.groupby(["month", "level"]).size().unstack(fill_value=0)[["oblast", "raion", "hromada"]]
    fig, ax = plt.subplots(figsize=(10, 4))
    bottom = None
    for color, level in zip(SERIES, counts.columns):
        ax.bar(counts.index, counts[level], width=25, bottom=bottom, color=color, label=level,
               edgecolor=SURFACE, linewidth=1)
        bottom = counts[level] if bottom is None else bottom + counts[level]
    ax.axvline(HISTORY_START.tz_localize(None), color=INK_2, linestyle="--", linewidth=1)
    ax.annotate("training window starts\n(raion-level regime)", (HISTORY_START.tz_localize(None), ax.get_ylim()[1] * 0.9),
                xytext=(-8, 0), textcoords="offset points", ha="right", va="top", color=INK_2, fontsize=9)
    ax.set_title("Alert records per month, by level")
    ax.set_ylabel("alerts")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eda_levels_by_month.png", dpi=150)
    plt.close(fig)


def plot_hour_profile(activity: pd.DataFrame) -> None:
    """Share of raions under alert by local hour of day."""
    local_hour = activity.index.tz_convert("Europe/Kyiv").hour
    share = activity.mean(axis=1).groupby(local_hour).mean() * 100
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(share.index, share.values, color=SERIES[0], width=0.8, edgecolor=SURFACE, linewidth=1)
    peak = share.idxmax()
    ax.annotate(f"{share[peak]:.0f}% at {peak:02d}:00", (peak, share[peak]), xytext=(0, 4),
                textcoords="offset points", ha="center", color=INK_2, fontsize=9)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("hour of day (Kyiv time)")
    ax.set_ylabel("% of raions under alert")
    ax.set_title(f"Alerts follow a daily cycle ({HISTORY_START:%b %Y} – {activity.index[-1]:%b %Y})")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eda_hour_profile.png", dpi=150)
    plt.close(fig)


def plot_daily_share(activity: pd.DataFrame) -> None:
    daily = activity.mean(axis=1).resample("D").mean() * 100
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(daily.index, daily.values, color=SERIES[0], linewidth=1, alpha=0.45)
    ax.plot(daily.index, daily.rolling(7).mean().values, color=SERIES[0], linewidth=2, label="7-day mean")
    ax.set_ylabel("% of raion-time under alert")
    ax.set_title("Daily share of raion-time under alert")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eda_daily_share.png", dpi=150)
    plt.close(fig)


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    alerts = load_alerts()
    data_end = alerts["finished_at"].max().floor("D")
    units = forecast_units(alerts, HISTORY_START)
    activity = minute_activity(alerts, units, HISTORY_START, data_end)
    plot_levels_by_month(alerts[alerts["started_at"] >= "2023-01-01"])
    plot_hour_profile(activity)
    plot_daily_share(activity)
    share = activity.mean().sort_values(ascending=False) * 100
    print("most / least alerted units (% of time):")
    print(share.head(5).round(1).to_string(), "\n...\n", share.tail(5).round(1).to_string(), sep="")


if __name__ == "__main__":
    main()
