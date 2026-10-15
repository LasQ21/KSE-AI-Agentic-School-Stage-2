"""alerts.in.ua API adapter: fresh alert history (the GitHub CSV stops on 2026-09-07).

API facts (from https://devs.alerts.in.ua, not from memory):
* GET https://api.alerts.in.ua/v1/regions/{uid}/alerts/month_ago.json, ``Authorization: Bearer <token>``;
* only ``month_ago`` exists, so we can get roughly the last 30 days;
* history endpoint limit: 2 requests/minute; repeated 429s can get the token blocked.

Usage:  export ALERTS_IN_UA_TOKEN=...;  python -m src.alerts_in_ua      (downloads raw JSON)
Raw responses are cached in data/raw/alerts_in_ua/ so the API is hit only once per UID.
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd
import requests

from .config import KYIV_CITY, OBLASTS, RAW_DIR, ROOT
from .data import COLUMNS

API = "https://api.alerts.in.ua/v1/regions/{uid}/alerts/month_ago.json"
CACHE = RAW_DIR / "alerts_in_ua"
UID_TABLE = ROOT / "data" / "ref" / "alerts_in_ua_uids.csv"  # the API's official UID spreadsheet
DELAY_S = 35  # stays under 2 requests/minute

# Raions renamed after the GitHub dataset fixed its names: dataset name -> alerts.in.ua name.
RAION_ALIASES = {
    "Володимир-Волинський район": "Володимирський район",
    "Красноградський район": "Берестинський район",
    "Червоноградський район": "Шептицький район",
}


def _norm(name: str) -> str:
    return str(name).replace("’", "'").replace("ʼ", "'").strip()


def uid_table() -> pd.DataFrame:
    d = pd.read_csv(UID_TABLE, skiprows=3)
    d["key"] = d["Назва"].map(_norm)
    return d


def oblast_uids() -> dict[str, int]:
    d = uid_table()
    d = d[d["Тип"].isin(["Область", "Місто з спеціальним статусом"])]
    m = dict(zip(d["key"], d["UID"]))
    return {ob: int(m[_norm(ob)]) for ob in OBLASTS}


def raion_uids(raions: list[str]) -> dict[str, int]:
    d = uid_table()
    m = dict(zip(d.loc[d["Тип"] == "Район", "key"], d.loc[d["Тип"] == "Район", "UID"]))
    out = {}
    for r in raions:
        key = _norm(RAION_ALIASES.get(r, r))
        if key not in m:
            raise KeyError(f"no alerts.in.ua UID for raion {r!r}")
        out[r] = int(m[key])
    return out


def fetch(uid: int, token: str) -> list[dict]:
    path = CACHE / f"{uid}.json"
    if path.exists():
        return json.loads(path.read_text())["alerts"]
    for attempt in range(5):
        try:
            resp = requests.get(API.format(uid=uid), headers={"Authorization": f"Bearer {token}"}, timeout=60)
            break
        except requests.RequestException as exc:  # connection resets happen; the cache makes reruns cheap
            print(f"uid {uid}: {exc!r}, retrying in {DELAY_S}s", flush=True)
            time.sleep(DELAY_S)  # a retry still counts against the 2 requests/minute limit
    else:
        raise RuntimeError(f"uid {uid}: giving up after 5 attempts")
    if resp.status_code == 429:
        raise RuntimeError("alerts.in.ua rate limit hit; wait a few minutes before retrying")
    resp.raise_for_status()
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(resp.text)
    time.sleep(DELAY_S)
    return resp.json()["alerts"]


def get_token() -> str:
    """ALERTS_IN_UA_TOKEN from the environment or from the git-ignored .env file."""
    token = os.environ.get("ALERTS_IN_UA_TOKEN")
    env = ROOT / ".env"
    if not token and env.exists():
        for line in env.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "ALERTS_IN_UA_TOKEN":
                token = value.strip()
    if not token:
        raise SystemExit("put ALERTS_IN_UA_TOKEN=... into .env (git-ignored); never commit the token")
    return token


def download(units: pd.DataFrame | None = None) -> None:
    """Fetch every oblast; if oblast responses lack raion-level alerts, fetch raions too."""
    token = get_token()
    records = []
    for ob, uid in oblast_uids().items():
        alerts = fetch(uid, token)
        records += alerts
        print(f"{ob}: {len(alerts)} alerts", flush=True)
    types = pd.Series([a.get("location_type") for a in records]).value_counts()
    print("location types in oblast responses:", types.to_dict())
    if "raion" not in types and units is not None:
        print("no raion-level alerts in oblast responses -> fetching raions one by one (~1h)")
        for raion, uid in raion_uids(units["raion"].dropna().tolist()).items():
            print(f"{raion}: {len(fetch(uid, token))} alerts", flush=True)


def load_alerts_in_ua() -> pd.DataFrame:
    """All cached responses -> the common alert frame (air-raid alerts only)."""
    rows = []
    for path in sorted(CACHE.glob("*.json")):
        rows += json.loads(path.read_text())["alerts"]
    df = pd.DataFrame(rows).drop_duplicates("id")
    # Real responses differ from the docs: raion-level alerts carry the raion name in
    # ``location_title`` (``location_raion`` is absent), only hromada alerts have ``location_raion``,
    # and every "city" alert duplicates its hromada's alert (same minute), so cities are dropped.
    df = df[(df["alert_type"] == "air_raid") & (df["location_type"] != "city")]
    if "location_raion" not in df:
        df["location_raion"] = None
    reverse = {_norm(v): k for k, v in RAION_ALIASES.items()}
    to_dataset_oblast = {_norm(ob): ob for ob in OBLASTS}
    raion_name = df["location_title"].where(df["location_type"] == "raion", df["location_raion"])

    out = pd.DataFrame({
        "oblast": df["location_oblast"].map(lambda s: to_dataset_oblast.get(_norm(s))),
        "raion": raion_name.map(lambda s: reverse.get(_norm(s), _norm(s)) if pd.notna(s) else s),
        "hromada": df["location_title"].where(df["location_type"] == "hromada"),
        "level": df["location_type"].map({"oblast": "oblast", "raion": "raion", "hromada": "hromada"}),
        "started_at": pd.to_datetime(df["started_at"], utc=True),
        "finished_at": pd.to_datetime(df["finished_at"], utc=True),
    })
    # Kyiv city alerts are oblast-level in the GitHub dataset's convention.
    kyiv = df["location_title"].map(_norm) == _norm(KYIV_CITY)
    out.loc[kyiv.to_numpy(), ["oblast", "level", "raion", "hromada"]] = [KYIV_CITY, "oblast", None, None]
    # Dataset raion names use the typographic apostrophe.
    out["raion"] = out["raion"].map(lambda s: s.replace("'", "’") if isinstance(s, str) else s)
    unknown = out["oblast"].isna() | out["level"].isna()
    if unknown.any():
        print(f"alerts_in_ua: dropping {unknown.sum()} alerts with unmapped oblast/level")
    return out.loc[~unknown, COLUMNS].reset_index(drop=True)


def _fetch_times() -> list[pd.Timestamp]:
    """Snapshot time of every cached response, from its ``meta.last_updated_at``.

    (File mtimes would change on ``git clone`` and silently shift the gap and the data end.)
    """
    files = sorted(CACHE.glob("*.json"))
    if not files:
        raise SystemExit("no cached alerts.in.ua responses: run `python -m src.alerts_in_ua` first")
    return [pd.to_datetime(json.loads(p.read_text())["meta"]["last_updated_at"], format="%Y/%m/%d %H:%M:%S %z")
            .tz_convert("UTC") for p in files]


def coverage_start() -> pd.Timestamp:
    """From here on every cached month_ago response covers the period (latest fetch - 1 month)."""
    return (max(_fetch_times()) - pd.DateOffset(months=1)).ceil("D")


def fetched_at() -> pd.Timestamp:
    """Data is complete only up to the earliest fetch: later alerts are missing from older files."""
    return min(_fetch_times())


if __name__ == "__main__":
    from .config import HISTORY_START
    from .data import forecast_units, load_alerts
    download(forecast_units(load_alerts(["sirens_csv"]), HISTORY_START))
