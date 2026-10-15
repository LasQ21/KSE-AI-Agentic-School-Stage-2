"""Baselines and the gradient-boosting model. All of them share fit(train, y) / predict(test)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .config import HORIZONS

CATEGORICAL = ["unit_id", "oblast_id"]


class GroupRateBaseline:
    """P(y=1 | group) estimated on (the tail of) the training data, with a global fallback."""

    def __init__(self, keys: list[str], last_days: int | None = None):
        self.keys, self.last_days = keys, last_days

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.last_days:
            t = X.index.get_level_values("time")
            recent = t >= t.max() - pd.Timedelta(days=self.last_days)
            X, y = X[recent], y[recent]
        self.global_ = float(y.mean())
        self.rates_ = y.groupby([X[k].to_numpy() for k in self.keys]).mean()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        keys = pd.MultiIndex.from_arrays([X[k].to_numpy() for k in self.keys]) if len(self.keys) > 1 \
            else pd.Index(X[self.keys[0]].to_numpy())
        return self.rates_.reindex(keys).fillna(self.global_).to_numpy()


def persistence():
    """'What is happening now will keep happening': P(y | unit, alert active now)."""
    return GroupRateBaseline(["unit_id", "active_now"])


def seasonal():
    """'Same as usual at this hour': P(y | unit, local hour) over the last 28 days."""
    return GroupRateBaseline(["unit_id", "hour_local"], last_days=28)


class GBM:
    """Histogram gradient boosting with two optional fixes for a shift in base rates.

    half_life_days: recency weights ``0.5 ** (age / half_life)``. Age is counted back from the fold
        start; the weights are rescaled to mean 1, so the reference point only fixes the scale.
    recalibrate: "isotonic" or "platt". The last ``recal_days`` of the training rows are held out, a
        copy of the model is trained on the rows whose targets end before that window (whatever H
        is, so the buffer is the longest horizon), and its predictions on the window fit a calibrator
        that is applied to the model trained on all rows. Everything comes from the training rows,
        which ``evaluate.split`` limits to targets that end before the fold start.
    """

    def __init__(self, features: list[str], seed: int = 0, half_life_days: float | None = None,
                 recalibrate: str | None = None, recal_days: int = 7):
        self.features, self.seed = features, seed
        self.half_life_days, self.recalibrate, self.recal_days = half_life_days, recalibrate, recal_days
        self.model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=200,
            l2_regularization=1.0, categorical_features=[f in CATEGORICAL for f in features],
            early_stopping=False, random_state=seed,
        )

    def _weights(self, X: pd.DataFrame) -> np.ndarray | None:
        if not self.half_life_days:
            return None
        t = X.index.get_level_values("time")
        age_days = (t.max() - t) / pd.Timedelta(days=1)
        w = 0.5 ** (np.asarray(age_days, dtype=float) / self.half_life_days)
        return w / w.mean()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X[self.features], y, sample_weight=self._weights(X))
        self.calibrator_ = None
        if self.recalibrate:
            self._fit_calibrator(X, y)
        return self

    def _fit_calibrator(self, X: pd.DataFrame, y: pd.Series) -> None:
        t = X.index.get_level_values("time")
        self.calib_start_ = t.max() + pd.Timedelta(hours=1) - pd.Timedelta(days=self.recal_days)
        calib = t >= self.calib_start_
        early = t + pd.Timedelta(hours=max(HORIZONS)) <= self.calib_start_
        self.calib_times_ = t[calib]
        if not early.any() or y[calib].nunique() < 2:
            return  # nothing to learn from: keep the raw probabilities
        holdout = GBM(self.features, self.seed, self.half_life_days).fit(X[early], y[early])
        p, yc = holdout.predict(X[calib]), y[calib].to_numpy()
        if self.recalibrate == "isotonic":
            self.calibrator_ = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, yc)
        elif self.recalibrate == "platt":
            self.calibrator_ = LogisticRegression(C=1e6).fit(_logit(p)[:, None], yc)
        else:
            raise ValueError(f"unknown recalibration: {self.recalibrate}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = self.model.predict_proba(X[self.features])[:, 1]
        if self.calibrator_ is None:
            return p
        if self.recalibrate == "isotonic":
            return self.calibrator_.predict(p)
        return self.calibrator_.predict_proba(_logit(p)[:, None])[:, 1]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))
