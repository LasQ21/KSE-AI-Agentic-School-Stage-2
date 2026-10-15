"""End-to-end pipeline: alerts (+ Telegram) -> features -> walk-forward backtest -> reports/.

Examples:
    python main.py                                   # alerts CSV only, test = last 28 days
    python main.py --telegram kpszsu war_monitor     # + text features and ablation
    python main.py --sources sirens_csv alerts_in_ua # extend history (needs ALERTS_IN_UA_TOKEN)
    python main.py --level oblast --sources sirens_csv oblast_csv --test-days 29   # fresh Sep-Oct test
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import HISTORY_START, HORIZONS, OBLAST_CSV_GAP, REPORTS_DIR
from src.data import (forecast_units, hromada_activity, load_alerts, minute_activity, oblast_activity,
                      oblast_units)
from src.evaluate import backtest
from src.features import build_features
from src.models import GBM, persistence, seasonal

NON_FEATURES = {"y_1h", "y_3h", "y_6h", "y_24h"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["raion", "oblast"], default="raion")
    ap.add_argument("--sources", nargs="+", default=["sirens_csv"])
    ap.add_argument("--telegram", nargs="*", default=[], help="channels scraped with src.telegram")
    ap.add_argument("--test-days", type=int, default=28)
    ap.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    ap.add_argument("--data-end", help="override the data end (UTC), e.g. 2026-09-07")
    ap.add_argument("--out", help="report sub-directory (default: <level>)")
    ap.add_argument("--variants", action="store_true",
                    help="also score recency-weighted and recalibrated variants of the main GBM (ablation)")
    args = ap.parse_args()
    if "oblast_csv" in args.sources and args.level != "oblast":
        ap.error("oblast_csv has no raion information: use --level oblast")

    alerts = load_alerts(args.sources)
    # Alerts still active at snapshot time are missing from the CSV, so the last (partial) day is
    # dropped: otherwise its final hours would look artificially quiet.
    csv_alerts = alerts if args.sources == ["sirens_csv"] else load_alerts(["sirens_csv"])
    data_end = csv_alerts["finished_at"].max().floor("D")
    gaps = []
    if "alerts_in_ua" in args.sources:
        from src.alerts_in_ua import coverage_start, fetched_at
        api_start = coverage_start()
        if api_start > data_end:
            gaps.append((data_end, api_start))  # neither source covers this period
        data_end = fetched_at().floor("h")  # the API includes ongoing alerts (finished_at = null)
    if "oblast_csv" in args.sources:
        gaps.append((data_end, OBLAST_CSV_GAP[1]))  # official CSV ends, third-party data inaccurate
        data_end = alerts["finished_at"].max().floor("D")  # same snapshot caveat as the CSV
    if args.data_end:
        data_end = pd.Timestamp(args.data_end, tz="UTC")
    if args.level == "raion":
        units = forecast_units(csv_alerts, HISTORY_START)  # fixed unit set, independent of the source
    else:
        units = oblast_units()
    print(f"alerts: {len(alerts):,} rows, units: {len(units)}, data end: {data_end}, gaps: {gaps}")

    posts = None
    if args.telegram:
        from src.telegram import load_channel
        posts = pd.concat([load_channel(c) for c in args.telegram], ignore_index=True)
        posts = posts[posts["published_at"] < data_end]

    if args.level == "raion":
        activity = minute_activity(alerts, units, HISTORY_START, data_end)
        hromada = hromada_activity(alerts, units, HISTORY_START, data_end)
    else:  # any alert level counts; the flattened data has no hromada information
        activity = oblast_activity(alerts, units, HISTORY_START, data_end)
        hromada = pd.DataFrame(np.zeros(activity.shape, dtype=bool), index=activity.index, columns=activity.columns)
    df = build_features(activity, hromada, units, data_end, posts, gaps)
    if args.level == "oblast":
        df = df.drop(columns=[c for c in df.columns if c.startswith("hrom")])

    base_features = [c for c in df.columns if c not in NON_FEATURES and not c.startswith("tg_")]
    models = {"persistence": persistence, "seasonal": seasonal, "gbm": lambda: GBM(base_features)}
    if posts is not None:
        all_features = [c for c in df.columns if c not in NON_FEATURES]
        models["gbm_text"] = lambda: GBM(all_features)
    if args.variants:
        main_name, main_features = ("gbm_text", all_features) if posts is not None else ("gbm", base_features)
        for suffix, kw in [("hl14", {"half_life_days": 14}), ("hl45", {"half_life_days": 45}),
                           ("iso", {"recalibrate": "isotonic"}), ("platt", {"recalibrate": "platt"})]:
            models[f"{main_name}_{suffix}"] = lambda kw=kw: GBM(main_features, **kw)

    table, preds = backtest(df, models, args.horizons, args.test_days)
    out = REPORTS_DIR / (args.out or args.level)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "metrics.csv", index=False)
    preds.to_csv(out / "predictions.csv.gz")
    print(f"saved to {out}")
    pd.set_option("display.width", 200)
    print(table.pivot_table(index=["subset", "H"], columns="model", values=["roc_auc", "brier"]).round(4))


if __name__ == "__main__":
    main()
