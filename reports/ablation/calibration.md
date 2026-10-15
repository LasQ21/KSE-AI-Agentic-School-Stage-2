# Calibration fixes under the Sep–Oct regime shift: negative result

**Problem.** On the fresh test (2026-09-16 → 10-15) the 24 h probabilities of GBM + Telegram are too low: predicted 0.15 → observed 0.42 (`reports/raion_fresh/results_calibration.png`). The model was trained on a calmer period.

**Tried** (all leakage-safe: everything is fitted on training rows whose targets end before the fold start; covered by `tests/test_leakage.py::test_recalibration_data_ends_before_fold_start`):

| Variant | What it does |
|---|---|
| `hl14`, `hl45` | recency-weighted training, sample weight `0.5 ** (age / half_life)` |
| `iso`, `platt` | isotonic / Platt recalibration on the last 7 days of the training rows, using a hold-out copy of the model |

Reproduce: `python main.py <usual args> --variants --out ablation/<name>`. Raw metrics are in `calib_fresh_24h/`, `calib_fresh_short/` and `calib_aug/`.

## Result: no variant is adopted

Change vs GBM + Telegram, hours with no alert at forecast time (Δ Brier < 0 and Δ AUC > 0 are improvements), from `calibration_deltas_quiet_now.csv`:

| Variant | Brier better | AUC better | mean Δ Brier | mean Δ AUC |
|---|---|---|---|---|
| hl14 | 0 / 8 | 3 / 8 | +0.0087 | −0.0054 |
| hl45 | 0 / 8 | 2 / 8 | +0.0035 | −0.0017 |
| isotonic | 0 / 8 | 0 / 8 | +0.0079 | −0.0146 |
| Platt | 0 / 8 | 0 / 8 | +0.0083 | −0.0121 |

(8 = 2 tests × 4 horizons.)

## Why recalibration fails here

By week of the fresh test, 24 h horizon (AUC / Brier):

| Week | no fix | isotonic | Platt |
|---|---|---|---|
| 09-16 | 0.963 / **0.081** | 0.961 / 0.156 | 0.963 / 0.156 |
| 09-23 | 0.967 / **0.061** | 0.953 / 0.078 | 0.952 / 0.075 |
| 09-30 | 0.967 / **0.064** | 0.964 / 0.067 | 0.965 / 0.066 |
| 10-07 | 0.960 / 0.075 | 0.958 / **0.071** | 0.960 / 0.072 |

* In the first week, the last 7 days of training data fall *before* the 07–15 Sep data gap. The calibrator therefore learns the calmer August regime and pushes probabilities down even further: Brier almost doubles.
* In weeks 2–3 the calibration window is right after the gap. There the long-window features are still warming up, so the hold-out predictions the calibrator learns from are distorted.
* Only in week 4, with three weeks of continuous post-gap data, does recalibration beat the raw model.

So recalibration needs several weeks of clean data in the new regime. With a 9-day gap just before the test period, that is not available. Recency weighting shifts weight to the most recent pre-gap weeks, which are also the old regime, so it cannot fix the base rate either.

## What would plausibly work (not done)

Online recalibration started 3+ weeks after a gap, or a base-rate prior updated from the newest *observed* hours (alert history up to `t−1` is always available and is not affected by the target delay).
