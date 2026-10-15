"""Leakage tests: features at time t must not change when anything at or after t changes."""
import numpy as np
import pandas as pd
import pytest

from src.config import HORIZONS
from src.evaluate import folds, split
from src.features import build_features
from src.models import GBM

START = pd.Timestamp("2026-01-01", tz="UTC")
DAYS = 40
T0 = START + pd.Timedelta(days=35, hours=13)  # an hour boundary inside the range
UNITS = pd.DataFrame({"oblast": ["Сумська область", "Сумська область", "м. Київ"],
                      "raion": ["Сумський район", "Шосткинський район", np.nan]})
UNITS["unit"] = ["Сумська область / Сумський район", "Сумська область / Шосткинський район", "м. Київ"]


def _activity(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(START, periods=DAYS * 24 * 60, freq="min")
    # Random alert episodes: a Markov chain per unit gives realistic runs of 1s.
    a = np.zeros((len(idx), len(UNITS)), dtype=bool)
    for j in range(len(UNITS)):
        flips = rng.random(len(idx)) < 0.01
        a[:, j] = np.cumsum(flips) % 2 == 1
    return pd.DataFrame(a, index=idx, columns=UNITS["unit"])


def _posts(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = START + pd.to_timedelta(np.sort(rng.integers(0, DAYS * 24 * 3600, 3000)), unit="s")
    texts = rng.choice(["Шахеди на Сумщині", "Ракета курсом на Київ", "Балістика", "Відбій"], 3000)
    return pd.DataFrame({"message_id": range(3000), "published_at": times, "edited": False, "text": texts})


def _corrupt_from(df: pd.DataFrame, t0, seed: int) -> pd.DataFrame:
    out = df.copy()
    future = out.index >= t0
    out.loc[future] = np.random.default_rng(seed).random((future.sum(), out.shape[1])) < 0.5
    return out


def test_features_ignore_future():
    end = START + pd.Timedelta(days=DAYS)
    act, hrom, posts = _activity(1), _activity(2), _posts(3)
    base = build_features(act, hrom, UNITS, end, posts)

    act2, hrom2 = _corrupt_from(act, T0, 4), _corrupt_from(hrom, T0, 5)
    posts2 = posts.copy()
    future = posts2["published_at"] >= T0
    posts2.loc[future, "text"] = "Масована атака: ракети, балістика, шахеди на Сумщині і Київ"
    # A post published exactly at T0 must not be visible at T0.
    edge = pd.DataFrame({"message_id": [99999], "published_at": [T0], "edited": [False],
                         "text": ["Ракети на Київ і Сумщину"]})
    posts2 = pd.concat([posts2, edge], ignore_index=True)
    changed = build_features(act2, hrom2, UNITS, end, posts2)

    feature_cols = [c for c in base.columns if not c.startswith("y_")]
    t = base.index.get_level_values("time")
    pd.testing.assert_frame_equal(base.loc[t <= T0, feature_cols], changed.loc[t <= T0, feature_cols])
    # Sanity: the corruption is visible after T0, so the test is not vacuous.
    assert not base.loc[t > T0, feature_cols].equals(changed.loc[t > T0, feature_cols])


def test_targets_look_only_forward_and_respect_data_end():
    end = START + pd.Timedelta(days=DAYS)
    df = build_features(_activity(1), _activity(2), UNITS, end)
    t = df.index.get_level_values("time")
    for H in HORIZONS:
        y = df[f"y_{H}h"]
        assert y[t + pd.Timedelta(hours=H) > end].isna().all()
        assert y[t + pd.Timedelta(hours=H) <= end].notna().all()


@pytest.mark.parametrize("H", HORIZONS)
def test_train_targets_end_before_fold_start(H):
    end = START + pd.Timedelta(days=DAYS)
    df = build_features(_activity(1), _activity(2), UNITS, end)
    for fold_start, fold_end in folds(end, test_days=14):
        train, test = split(df, H, fold_start, fold_end)
        t_train = train.index.get_level_values("time")
        assert (t_train + pd.Timedelta(hours=H) <= fold_start).all()
        assert (test.index.get_level_values("time") >= fold_start).all()


@pytest.mark.parametrize("H", HORIZONS)
@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_recalibration_data_ends_before_fold_start(H, method):
    end = START + pd.Timedelta(days=DAYS)
    df = build_features(_activity(1), _activity(2), UNITS, end)
    features = [c for c in df.columns if not c.startswith("y_")]
    for fold_start, fold_end in folds(end, test_days=14):
        train, test = split(df, H, fold_start, fold_end)
        model = GBM(features, recalibrate=method).fit(train, train[f"y_{H}h"])
        calib = model.calib_times_
        y_calib = train.loc[train.index.get_level_values("time") >= model.calib_start_, f"y_{H}h"]
        # Not vacuous: a calibrator is fitted unless the window has one class (synthetic y_24h == 1).
        assert (model.calibrator_ is not None) == (y_calib.nunique() == 2)
        # The calibration window is the tail of the training rows: its targets end before the fold.
        assert len(calib) > 0 and (calib + pd.Timedelta(hours=H) <= fold_start).all()
        assert calib.min() >= model.calib_start_ > fold_start - pd.Timedelta(days=8)
        # Corrupting everything from the fold start on must not change the forecasts.
        corrupted = df.copy()
        future = corrupted.index.get_level_values("time") >= fold_start
        corrupted.loc[future, f"y_{H}h"] = 1 - corrupted.loc[future, f"y_{H}h"]
        train2, _ = split(corrupted, H, fold_start, fold_end)
        model2 = GBM(features, recalibrate=method).fit(train2, train2[f"y_{H}h"])
        np.testing.assert_array_equal(model.predict(test), model2.predict(test))


def test_gaps_mask_features_and_targets():
    end = START + pd.Timedelta(days=DAYS)
    gap = (START + pd.Timedelta(days=20), START + pd.Timedelta(days=22))
    df = build_features(_activity(1), _activity(2), UNITS, end, gaps=[gap])
    t = df.index.get_level_values("time")
    # active_now at t reads minute t-1, so it is unknown for t in (gap start, gap end].
    in_gap = (t > gap[0]) & (t <= gap[1])
    assert df.loc[in_gap, "active_now"].isna().all()
    assert df.loc[t == gap[0], "active_now"].notna().all()
    # Hour-level features read the last minute -> unknown inside the gap.
    assert df.loc[in_gap, "act_1h"].isna().all()
    # Rate features use the known part of the window: a 2-day gap leaves 5 of 7 days known...
    assert df.loc[t == gap[1] + pd.Timedelta(days=1), "act_168h"].notna().all()
    # ...and they equal the active share over exactly those known hours.
    t1 = gap[1] + pd.Timedelta(days=1)
    act = _activity(1)
    win = act[(act.index >= t1 - pd.Timedelta(days=7)) & (act.index < t1)]
    win = win[(win.index < gap[0]) | (win.index >= gap[1])]
    np.testing.assert_allclose(df.xs(t1, level="time")["act_168h"].to_numpy(), win.mean().to_numpy(), rtol=1e-5)
    # A target window that ends inside the gap is unknown, too.
    assert df.loc[(t > gap[0] - pd.Timedelta(hours=24)) & (t < gap[0]), "y_24h"].isna().all()
    # Look-backs reaching before the start of history are unknown, not zero:
    # act_720h needs 15 known days, so it is NaN for the first 15 days.
    assert df.loc[t < START + pd.Timedelta(days=15), "act_720h"].isna().all()
    assert df.loc[(t >= START + pd.Timedelta(days=15)) & (t < gap[0]), "act_720h"].notna().all()
