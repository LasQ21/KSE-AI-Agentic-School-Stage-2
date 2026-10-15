"""Scraping public Telegram channels via the t.me/s/<channel> web preview (no login, no API key).

Leakage note: the web preview shows the *current* text of a post. If a post was edited after
publication, its text may contain information from after ``published_at``. We store the
``edited`` flag and the feature builder drops edited posts by default.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import RAW_DIR

HEADERS = {"User-Agent": "Mozilla/5.0 (research project; air-alert forecasting)"}
COLUMNS = ["message_id", "published_at", "edited", "text"]


def _path(channel: str, part: str = ""):
    return RAW_DIR / f"telegram_{channel}{part}.csv"


def parse_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for msg in soup.select("div.tgme_widget_message[data-post]"):
        when = msg.select_one("a.tgme_widget_message_date time")
        if when is None:
            continue
        text_el = msg.select_one("div.tgme_widget_message_text")
        meta = msg.select_one("span.tgme_widget_message_meta")
        rows.append({
            "message_id": int(msg["data-post"].rsplit("/", 1)[1]),
            "published_at": pd.Timestamp(when["datetime"]).tz_convert("UTC"),
            "edited": bool(meta and "edited" in meta.get_text(" ", strip=True).lower()),
            "text": text_el.get_text(" ", strip=True) if text_el else "",
        })
    return rows


def fetch_page(channel: str, before: int | None = None, retries: int = 6) -> list[dict]:
    url = f"https://t.me/s/{channel}" + (f"?before={before}" if before else "")
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
        except requests.RequestException as exc:  # timeouts, dropped connections
            print(f"{url}: {exc!r}, retrying", flush=True)
            time.sleep(5 * (attempt + 1))
            continue
        if resp.status_code == 429:  # rate limited: back off
            time.sleep(10 * (attempt + 1))
            continue
        resp.raise_for_status()
        return parse_page(resp.text)
    raise RuntimeError(f"giving up on {url}")


def _save(rows: list[dict], path) -> None:
    df = pd.DataFrame(rows, columns=COLUMNS).drop_duplicates("message_id").sort_values("message_id")
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)  # atomic: a crash never leaves a half-written file


def scrape_channel(channel: str, since: pd.Timestamp, before: int | None = None,
                   stop_id: int = 0, part: str = "", delay: float = 1.0, checkpoint: int = 25) -> None:
    """Walk backwards from ``before`` (default: newest post) until ``since`` or ``stop_id``.

    Saves a checkpoint every ``checkpoint`` pages and resumes from the oldest saved post, so an
    interrupted run loses at most a few pages.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = _path(channel, part)
    rows: list[dict] = []
    if path.exists():
        rows = load_file(path).to_dict("records")
        oldest = min(rows, key=lambda r: r["message_id"])
        if oldest["published_at"] < since or oldest["message_id"] <= stop_id:
            print(f"{channel}{part}: already complete ({len(rows)} posts)")
            return
        before = oldest["message_id"]
        print(f"{channel}{part}: resuming before id {before} ({oldest['published_at']})", flush=True)

    pages = 0
    while True:
        page = fetch_page(channel, before)
        if not page:
            break
        rows.extend(r for r in page if r["message_id"] > stop_id)
        oldest = min(page, key=lambda r: r["message_id"])
        before = oldest["message_id"]
        pages += 1
        if oldest["published_at"] < since or before <= stop_id:
            break
        if pages % checkpoint == 0:
            _save(rows, path)
            print(f"{channel}{part}: {len(rows)} posts, reached {oldest['published_at']}", flush=True)
        time.sleep(delay)
    _save(rows, path)
    print(f"{channel}{part}: done, {len(rows)} posts", flush=True)


def load_file(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"text": str})
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df["text"] = df["text"].fillna("")
    return df


def load_channel(channel: str) -> pd.DataFrame:
    """All scraped parts of a channel, deduplicated."""
    parts = sorted(RAW_DIR.glob(f"telegram_{channel}*.csv"))
    if not parts:
        raise FileNotFoundError(f"no scraped data for {channel}: run python -m src.telegram {channel}")
    df = pd.concat([load_file(p) for p in parts], ignore_index=True)
    return df.drop_duplicates("message_id", keep="last").sort_values("message_id").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("channel")
    ap.add_argument("--since", default="2025-10-25")
    ap.add_argument("--before", type=int, help="start below this message id (default: newest)")
    ap.add_argument("--stop-id", type=int, default=0, help="stop at this message id")
    ap.add_argument("--part", default="", help="file suffix, so parallel segments do not collide")
    args = ap.parse_args()
    scrape_channel(args.channel, pd.Timestamp(args.since, tz="UTC"), args.before, args.stop_id, args.part)
    out = load_channel(args.channel)
    print(f"{args.channel}: {len(out)} posts, {out['published_at'].min()} .. {out['published_at'].max()}, "
          f"edited={out['edited'].mean():.1%}")
