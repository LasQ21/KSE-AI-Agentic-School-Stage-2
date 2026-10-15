"""Loading alert intervals and turning them into a per-unit minute-level activity matrix.

Every alert source returns the same frame: columns ``oblast, raion, hromada, level, started_at,
finished_at`` (tz-aware UTC; ``raion``/``hromada`` are NaN above their level). That lets us swap
the GitHub CSV for the alerts.in.ua API later without touching features or evaluation.

Forecast unit = raion (plus Kyiv city). A raion is "under alert" when either its own
raion-level alert or an oblast-level alert of its oblast is active. Hromada-level alerts
cover only part of a raion, so they are not part of the target (they are used as a feature).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from .config import KYIV_CITY, OBLAST_CSV_GAP, OBLASTS, RAW_DIR, SIRENS_CSV_URL

COLUMNS = ["oblast", "raion", "hromada", "level", "started_at", "finished_at"]


def download_sirens_csv(force: bool = False) -> None:
    path = RAW_DIR / "official_data_uk.csv"
    if path.exists() and not force:
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(SIRENS_CSV_URL, timeout=120)
    resp.raise_for_status()
    path.write_bytes(resp.content)


def load_sirens_csv() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "official_data_uk.csv")
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)
    df = df[df["oblast"].isin(OBLASTS)]
    return df[COLUMNS].reset_index(drop=True)


def load_oblast_csv() -> pd.DataFrame:
    """Official alerts flattened to oblast level (no raion info), 2026-01-01 .. 2026-10-12.

    Received from a third party. Verified against the GitHub CSV on Aug 2026: every record matches
    an official record exactly (same oblast and start/finish to the second), i.e. it is the union of
    all alert levels per oblast. Only the part after OBLAST_CSV_GAP is used; the gap itself is known
    to be inaccurate (e.g. a separate "Київ" region with synthetic 30-minute durations).
    """
    df = pd.read_csv(RAW_DIR / "friend" / "alerts.csv", encoding="utf-8-sig").iloc[:, :3]
    df = df.rename(columns={"region": "oblast"})
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)
    df = df[df["oblast"].isin(OBLASTS) & (df["started_at"] >= OBLAST_CSV_GAP[1])]
    df["raion"], df["hromada"], df["level"] = np.nan, np.nan, "flat"
    return df[COLUMNS].reset_index(drop=True)


def load_alerts(sources: list[str] | None = None) -> pd.DataFrame:
    """Concatenate alert intervals from the requested sources (deduplicated)."""
    sources = sources or ["sirens_csv"]
    frames = []
    for src in sources:
        if src == "sirens_csv":
            download_sirens_csv()
            frames.append(load_sirens_csv())
        elif src == "oblast_csv":
            frames.append(load_oblast_csv())
        elif src == "alerts_in_ua":
            from .alerts_in_ua import load_alerts_in_ua  # needs ALERTS_IN_UA_TOKEN
            frames.append(load_alerts_in_ua())
        else:
            raise ValueError(f"unknown alert source: {src}")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["started_at"])
    # The same alert may come from two sources with second-level jitter in timestamps.
    df["_key"] = df["started_at"].dt.floor("min")
    df = df.drop_duplicates(subset=["oblast", "raion", "hromada", "level", "_key"]).drop(columns="_key")
    return df.sort_values("started_at").reset_index(drop=True)


def unit_name(oblast: str, raion: str | float) -> str:
    return oblast if oblast == KYIV_CITY else f"{oblast} / {raion}"


def forecast_units(alerts: pd.DataFrame, since: pd.Timestamp, min_alerts: int = 20) -> pd.DataFrame:
    """Raions (+ Kyiv city) that had at least ``min_alerts`` raion-level alerts since ``since``.

    Occupied raions with almost no alerts would add only noise.
    """
    recent = alerts[alerts["started_at"] >= since]
    rai = recent[recent["level"] == "raion"].groupby(["oblast", "raion"]).size()
    units = rai[rai >= min_alerts].reset_index()[["oblast", "raion"]]
    kyiv = pd.DataFrame({"oblast": [KYIV_CITY], "raion": [np.nan]})
    units = pd.concat([units, kyiv], ignore_index=True)
    units["unit"] = [unit_name(o, r) for o, r in zip(units["oblast"], units["raion"])]
    return units.sort_values("unit").reset_index(drop=True)


def oblast_units() -> pd.DataFrame:
    """One unit per oblast (+ Kyiv city): the only granularity the oblast CSV supports."""
    units = pd.DataFrame({"oblast": OBLASTS, "raion": np.nan})
    units["unit"] = units["oblast"]
    return units


def oblast_activity(alerts: pd.DataFrame, units: pd.DataFrame, start, end) -> pd.DataFrame:
    """[minute x oblast] True if an alert of *any* level is active in the oblast.

    Same definition as the flattened oblast CSV, so official-CSV history and fresh oblast-CSV
    data can be concatenated without changing the meaning of the target.
    """
    index = pd.date_range(start, end, freq="min", inclusive="left")
    out = np.zeros((len(index), len(units)), dtype=bool)
    for j, ob in enumerate(units["oblast"]):
        out[:, j] = _interval_mask(alerts[alerts["oblast"] == ob], start, end, len(index))
    return pd.DataFrame(out, index=index, columns=units["unit"].tolist())


def _interval_mask(sub: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, n: int) -> np.ndarray:
    """Boolean minute mask: True where at least one interval of ``sub`` is active."""
    one_min = pd.Timedelta(minutes=1)
    # Timedelta division is independent of the datetime unit (pandas 3 infers us/s, not ns).
    s = ((sub["started_at"] - start) // one_min).to_numpy(dtype=np.int64)
    f = ((sub["finished_at"].fillna(end) - start) // one_min).to_numpy(dtype=np.int64)
    s, f = np.clip(s, 0, n), np.clip(f, 0, n)
    delta = np.zeros(n + 1, dtype=np.int32)
    np.add.at(delta, s, 1)
    np.add.at(delta, f, -1)
    return np.cumsum(delta[:-1]) > 0


def minute_activity(
    alerts: pd.DataFrame, units: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Boolean matrix [minute x unit]. An alert with no ``finished_at`` is active until ``end``."""
    index = pd.date_range(start, end, freq="min", inclusive="left")
    n = len(index)
    oblast_level = alerts[alerts["level"] == "oblast"]
    oblast_masks = {
        ob: _interval_mask(oblast_level[oblast_level["oblast"] == ob], start, end, n)
        for ob in units["oblast"].unique()
    }
    out = np.zeros((n, len(units)), dtype=bool)
    raion_level = alerts[alerts["level"] == "raion"]
    for j, (ob, ra) in enumerate(zip(units["oblast"], units["raion"])):
        own = raion_level[(raion_level["oblast"] == ob) & (raion_level["raion"] == ra)]
        out[:, j] = oblast_masks[ob] | _interval_mask(own, start, end, n)
    return pd.DataFrame(out, index=index, columns=units["unit"].tolist())


def hromada_activity(alerts: pd.DataFrame, units: pd.DataFrame, start, end) -> pd.DataFrame:
    """[minute x unit] True if any hromada-level alert inside the raion is active (feature only)."""
    index = pd.date_range(start, end, freq="min", inclusive="left")
    n = len(index)
    hrom = alerts[alerts["level"] == "hromada"]
    out = np.zeros((n, len(units)), dtype=bool)
    for j, (ob, ra) in enumerate(zip(units["oblast"], units["raion"])):
        sub = hrom[(hrom["oblast"] == ob) & (hrom["raion"] == ra)]
        if len(sub):
            out[:, j] = _interval_mask(sub, start, end, n)
    return pd.DataFrame(out, index=index, columns=units["unit"].tolist())
