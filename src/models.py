"""Baselines and the gradient-boosting model. All of them share fit(train, y) / predict(test)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

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
    def __init__(self, features: list[str], seed: int = 0):
        self.features = features
        self.model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=200,
            l2_regularization=1.0, categorical_features=[f in CATEGORICAL for f in features],
            early_stopping=False, random_state=seed,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X[self.features], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.features])[:, 1]
