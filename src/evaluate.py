"""Walk-forward backtest on the most recent period.

For a fold starting at ``fold_start`` and horizon H, the model is trained only on rows with
``time + H <= fold_start``: their targets were fully observed before the forecasts are made.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def folds(data_end: pd.Timestamp, test_days: int, fold_days: int = 7) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    test_start = data_end - pd.Timedelta(days=test_days)
    edges = pd.date_range(test_start, data_end, freq=f"{fold_days}D")
    if edges[-1] < data_end:
        edges = edges.append(pd.DatetimeIndex([data_end]))
    return list(zip(edges[:-1], edges[1:]))


def split(df: pd.DataFrame, H: int, fold_start, fold_end) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = df.index.get_level_values("time")
    y = df[f"y_{H}h"]
    train = df[(t + pd.Timedelta(hours=H) <= fold_start) & y.notna()]
    test = df[(t >= fold_start) & (t < fold_end) & y.notna()]
    return train, test


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    out = {"n": len(y), "base_rate": float(y.mean()), "brier": brier_score_loss(y, p)}
    if 0 < y.mean() < 1:
        out["roc_auc"] = roc_auc_score(y, p)
        out["pr_auc"] = average_precision_score(y, p)
    return out


def backtest(df: pd.DataFrame, models: dict, horizons: list[int], test_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (metrics table, out-of-sample predictions)."""
    data_end = df.index.get_level_values("time").max() + pd.Timedelta(hours=1)
    preds = []
    for H in horizons:
        for fold_start, fold_end in folds(data_end, test_days):
            train, test = split(df, H, fold_start, fold_end)
            if test.empty:
                continue
            out = pd.DataFrame({"H": H, "y": test[f"y_{H}h"].to_numpy(),
                                "active_now": test["active_now"].to_numpy()}, index=test.index)
            for name, make in models.items():
                out[name] = make().fit(train, train[f"y_{H}h"]).predict(test)
            preds.append(out)
            print(f"H={H}h fold {fold_start:%m-%d}: train={len(train):,} test={len(test):,}", flush=True)
    preds = pd.concat(preds)

    rows = []
    for H, g in preds.groupby("H"):
        for subset, mask in [("all", np.ones(len(g), bool)), ("quiet_now", g["active_now"].to_numpy() == 0)]:
            for name in models:
                m = metrics(g["y"].to_numpy()[mask], g[name].to_numpy()[mask])
                rows.append({"H": H, "subset": subset, "model": name, **m})
    return pd.DataFrame(rows), preds
