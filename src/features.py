"""Hourly feature/target table: one row per (forecast time t, unit).

Anti-leakage contract (checked by tests/test_leakage.py):
* every feature for time t uses only minutes strictly before t (minute index < t);
* Telegram posts count only if ``published_at < t``;
* targets y_H look at [t, t+H) and are NaN when t+H is beyond the data end.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import HORIZONS, KYIV_CITY

LAG_HOURS = [1, 3, 6, 12, 24, 72, 168]

# Oblast mentions in Air Force posts ("на Київщині", "курс Житомирщина", "Запоріжжя", ...).
OBLAST_PATTERNS = {  # oblast names ("на Київщині") + oblast-centre cities ("Харків: БпЛА")
    "Вінницька область": r"вінниччин|вінниц",
    "Волинська область": r"волин|луцьк",
    "Дніпропетровська область": r"дніпропетровщин|дніпровщин|дніпр(?:о|а|і)\b",
    "Донецька область": r"донеччин|донецьк|краматорськ|слов.янськ",
    "Житомирська область": r"житомир",
    "Закарпатська область": r"закарпатт|ужгород",
    "Запорізька область": r"запоріжж|запорізьк",
    "Івано-Франківська область": r"івано-франківськ|прикарпатт",
    "Київська область": r"київщин",
    "Кіровоградська область": r"кіровоградщин|кропивницьк",
    "Львівська область": r"львів",
    "Миколаївська область": r"миколаїв",
    "Одеська область": r"одещин|одес",
    "Полтавська область": r"полтав",
    "Рівненська область": r"\bрівн(?:е|ого|ому|енщин)",
    "Сумська область": r"сумщин|\bсум(?:и|ах|ам)?\b",
    "Тернопільська область": r"тернопіл",
    "Харківська область": r"харків",
    "Херсонська область": r"херсон",
    "Хмельницька область": r"хмельниччин|хмельницьк",
    "Черкаська область": r"черкащин|черкас",
    "Чернівецька область": r"буковин|чернівеччин|чернівц",
    "Чернігівська область": r"чернігів",
    KYIV_CITY: r"\bки(?:їв|єв|єві|єва)\b|столиц",
}
THREAT_PATTERNS = {
    "drone": r"бпла|шахед|дрон|ударн",
    "missile": r"ракет|\bкр\b|крилат|калібр|х-101|іскандер",
    "ballistic": r"балістик",
    "aviation": r"міг-31|ту-95|ту-22|ту-160|су-34|\bкаб\b|авіабомб|тактичн",
    "all_clear": r"відбій|чисто",
}


def _hourly_sum(minutes: np.ndarray) -> np.ndarray:
    """[minutes x k] -> [hours x k] sums; the minute count must be a multiple of 60."""
    n, k = minutes.shape
    return minutes.reshape(n // 60, 60, k).sum(axis=1)


def _window_sum(hourly: np.ndarray, w: int) -> np.ndarray:
    """out[h] = sum of hourly[h-w : h]  (the w complete hours *before* hour h)."""
    c = np.vstack([np.zeros((1, hourly.shape[1])), np.cumsum(hourly, axis=0)])
    h = np.arange(hourly.shape[0])
    return c[h] - c[np.maximum(h - w, 0)]


def _run_lengths(active: np.ndarray, at: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each column, sampled at minute indices ``at``: (current streak length, minutes since last active).

    Values are measured at minute ``at`` (inclusive), capped at 7 days.
    """
    cap = 7 * 24 * 60
    idx = np.arange(active.shape[0])
    streak = np.empty((len(at), active.shape[1]), dtype=np.float32)
    since = np.empty_like(streak)
    for j in range(active.shape[1]):
        a = active[:, j]
        last_false = np.maximum.accumulate(np.where(~a, idx, -1))
        last_true = np.maximum.accumulate(np.where(a, idx, -cap))
        streak[:, j] = np.minimum(at - last_false[at], cap)
        since[:, j] = np.minimum(at - last_true[at], cap)
    return streak, since


def _look_back_hours(feature: str) -> int:
    """How many hours of alert history before t a feature reads (0 = none, e.g. Telegram)."""
    if feature.startswith(("tg_", "act_", "hrom_", "seas_")):
        return 0  # Telegram is continuous; rate features use only the known part of their window
    if feature in ("streak_min", "since_last_min"):
        return 7 * 24
    m = re.search(r"_(\d+)h$", feature)
    return int(m.group(1)) if m else 1  # *_now features read the last minute -> 1 hour


def telegram_hourly(posts: pd.DataFrame, hours: pd.DatetimeIndex, drop_edited: bool = True) -> pd.DataFrame:
    """[hour x feature] counts of posts *published within* each hour [h, h+1)."""
    if drop_edited:
        posts = posts[~posts["edited"]]
    text = posts["text"].str.lower()
    bins = posts["published_at"].dt.floor("h")
    cols = {"tg_posts": pd.Series(1, index=posts.index)}
    for name, pat in {**{f"tg_ob::{k}": v for k, v in OBLAST_PATTERNS.items()},
                      **{f"tg_{k}": v for k, v in THREAT_PATTERNS.items()}}.items():
        cols[name] = text.str.contains(pat, regex=True).astype(int)
    df = pd.DataFrame(cols)
    df["hour"] = bins
    return df.groupby("hour").sum().reindex(hours, fill_value=0)


def build_features(
    activity: pd.DataFrame,
    hromada: pd.DataFrame,
    units: pd.DataFrame,
    data_end: pd.Timestamp,
    posts: pd.DataFrame | None = None,
    gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = (),
) -> pd.DataFrame:
    """Long table indexed by (time, unit) with features and targets ``y_{H}h``.

    ``activity``/``hromada`` are [minute x unit] boolean matrices starting at an hour boundary.
    Rows run from 7 days after the start (so that lags exist) to ``data_end``.
    ``gaps`` are periods without alert data (the time before the start counts as one, too):
    * rate features (act_*, hrom_*, seas_*) are computed over the *known* part of their window and
      become NaN only if less than half of the window is known;
    * other features and targets become NaN if their window touches unknown hours at all.
    """
    start = activity.index[0]
    n_hours = len(activity) // 60
    act = activity.to_numpy()[: n_hours * 60]
    hrom = hromada.to_numpy()[: n_hours * 60]
    hours = pd.date_range(start, periods=n_hours, freq="h")
    hourly = _hourly_sum(act.astype(np.int32)).astype(np.float32)  # active minutes per hour
    hourly_h = _hourly_sum(hrom.astype(np.int32)).astype(np.float32)
    n_units = act.shape[1]

    unknown = np.zeros(n_hours, dtype=np.int32)
    for g0, g1 in gaps:
        lo = max(int((g0 - start) // pd.Timedelta(hours=1)), 0)
        hi = min(int(np.ceil((g1 - start) / pd.Timedelta(hours=1))), n_hours)
        unknown[lo:hi] = 1
    known = (1 - unknown).astype(np.float32)[:, None]
    cu = np.concatenate([[0], np.cumsum(unknown)])

    def touches_unknown(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Window [lo, hi) of hours touches a gap or lies (partly) before the start."""
        return (lo < 0) | (cu[np.clip(hi, 0, n_hours)] - cu[np.clip(lo, 0, n_hours)] > 0)

    def known_rate(hourly_minutes: np.ndarray, w: int) -> np.ndarray:
        """Active share of the known minutes among the w hours before t; NaN if < half is known."""
        num = _window_sum(hourly_minutes * known, w)[rows_h]
        den = _window_sum(known, w)[rows_h]  # before-start hours are not counted as known
        with np.errstate(invalid="ignore", divide="ignore"):
            out = num / (60 * den)
        out[np.broadcast_to(den < w / 2, out.shape)] = np.nan
        return out

    first = 7 * 24  # need a week of history for the longest lag
    rows_h = np.arange(first, n_hours)  # hour index h == forecast time t = hours[h]
    t_minute = rows_h * 60
    prev_minute = t_minute - 1  # last minute fully observed before t

    feats: dict[str, np.ndarray] = {}
    feats["active_now"] = act[prev_minute].astype(np.float32)
    feats["hromada_now"] = hrom[prev_minute].astype(np.float32)
    streak, since = _run_lengths(act, prev_minute)
    feats["streak_min"], feats["since_last_min"] = streak, since
    for w in LAG_HOURS + [720]:
        feats[f"act_{w}h"] = known_rate(hourly, w)
    for w in [1, 6, 24]:
        feats[f"hrom_{w}h"] = known_rate(hourly_h, w)

    # Onsets: number of alert starts in the previous hours (alerts tend to come in waves).
    onsets = np.zeros_like(act, dtype=np.int8)
    onsets[1:] = act[1:] & ~act[:-1]
    hourly_on = _hourly_sum(onsets.astype(np.int32)).astype(np.float32)

    # Oblast and national context.
    oblast_of = units["oblast"].to_numpy()
    nat_now = feats["active_now"].mean(axis=1, keepdims=True)
    feats["nat_active_now"] = np.repeat(nat_now, n_units, axis=1)
    for w in [1, 3, 6]:
        on = _window_sum(hourly_on, w)[rows_h].sum(axis=1, keepdims=True) / n_units
        feats[f"nat_onsets_{w}h"] = np.repeat(on, n_units, axis=1)
    ob_now = np.zeros_like(feats["active_now"])
    ob_on1 = np.zeros_like(feats["active_now"])
    on1 = _window_sum(hourly_on, 1)[rows_h]
    for ob in np.unique(oblast_of):
        cols = oblast_of == ob
        ob_now[:, cols] = feats["active_now"][:, cols].mean(axis=1, keepdims=True)
        ob_on1[:, cols] = on1[:, cols].mean(axis=1, keepdims=True)
    feats["oblast_active_now"], feats["oblast_onsets_1h"] = ob_now, ob_on1

    # Telegram text features (oblast-specific mentions + national threat types).
    # Channels post 2-3x more in 2026 autumn than in winter, so raw counts drift out of the training
    # range. All features are scale-free: shares of posts, or volume relative to the channel's own
    # last-7-days rate.
    if posts is not None:
        tg = telegram_hourly(posts, hours)

        def past(w: int) -> pd.DataFrame:  # sums over the w hours before h (posts published < t)
            return pd.DataFrame(tg.rolling(w, min_periods=1).sum().shift(1).fillna(0).to_numpy()[rows_h],
                                columns=tg.columns)

        week_rate = past(168)["tg_posts"].to_numpy() / 168
        for w in [1, 3, 6]:
            tgw = past(w)
            n_posts = tgw["tg_posts"].to_numpy()
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = n_posts / (w * week_rate)
                feats[f"tg_volume_rel_{w}h"] = np.repeat(np.nan_to_num(rel, nan=0.0)[:, None], n_units, axis=1).astype(np.float32)
                for c in [c for c in tg.columns if not c.startswith("tg_ob::") and c != "tg_posts"]:
                    share = np.nan_to_num(tgw[c].to_numpy() / n_posts, nan=0.0)
                    feats[f"{c}_share_{w}h"] = np.repeat(share[:, None], n_units, axis=1).astype(np.float32)
                ob = np.stack([tgw[f"tg_ob::{o}"].to_numpy() for o in oblast_of], axis=1) / n_posts[:, None]
            feats[f"tg_oblast_share_{w}h"] = np.nan_to_num(ob, nan=0.0).astype(np.float32)

    # Targets: any active minute in [t, t+H).
    any_hour = (hourly > 0).astype(np.float32)
    c = np.vstack([np.zeros((1, n_units)), np.cumsum(any_hour, axis=0)])

    def window_any(h0: np.ndarray, H: int) -> np.ndarray:
        """Any active minute in hours [h0, h0+H)."""
        return ((c[np.minimum(h0 + H, n_hours)] - c[np.clip(h0, 0, n_hours)]) > 0).astype(np.float32)

    targets = {}
    end_h = int((data_end - start) / pd.Timedelta(hours=1))
    for H in HORIZONS:
        y = window_any(rows_h, H)
        y[rows_h + H > end_h] = np.nan
        targets[f"y_{H}h"] = y

    # Unit x time-of-day memory: share of the previous k days with an alert in the same window
    # [t-24d, t-24d+H). For H <= 24 that window ends at or before t, so it is already observed.
    for H in [h for h in HORIZONS if h <= 24]:
        for days in [7, 28]:
            # Only days whose window is fully known count; NaN if fewer than half of the days do.
            total = np.zeros((len(rows_h), n_units), dtype=np.float32)
            n_ok = np.zeros(len(rows_h), dtype=np.float32)
            for d in range(1, days + 1):
                ok = ~touches_unknown(rows_h - 24 * d, rows_h - 24 * d + H)
                total[ok] += window_any(rows_h - 24 * d, H)[ok]
                n_ok += ok
            with np.errstate(invalid="ignore", divide="ignore"):
                seas = total / n_ok[:, None]
            seas[n_ok < days / 2] = np.nan
            feats[f"seas_{H}h_{days}d"] = seas

    # Mask the remaining features that would read unknown hours.
    for name, values in feats.items():
        look_back = _look_back_hours(name)
        if look_back:
            values[touches_unknown(rows_h - look_back, rows_h)] = np.nan
    for H in HORIZONS:
        targets[f"y_{H}h"][touches_unknown(rows_h, rows_h + H)] = np.nan

    times = hours[rows_h]
    keep = times < data_end
    index = pd.MultiIndex.from_product([times[keep], units["unit"]], names=["time", "unit"])
    out = {k: v[keep].reshape(-1) for k, v in {**feats, **targets}.items()}
    df = pd.DataFrame(out, index=index)
    local = df.index.get_level_values("time").tz_convert("Europe/Kyiv")
    df["hour_local"] = local.hour
    df["dow"] = local.dayofweek
    df["unit_id"] = np.tile(np.arange(n_units), keep.sum())
    df["oblast_id"] = np.tile(pd.factorize(oblast_of)[0], keep.sum())
    return df
