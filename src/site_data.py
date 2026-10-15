"""Builds the data files for the website map (docs/geo.js, docs/forecast.js).

* Geometry: OCHA COD-AB Ukraine (CC BY-IGO), downloaded from
  https://data.humdata.org/dataset/cod-ab-ukr  (``ukr_admin1.geojson``, ``ukr_admin2.geojson``).
  The files are large, so they are not committed; pass the folder with ``--geo``.
* Forecasts: the out-of-sample walk-forward predictions of a raion-level run
  (``reports/<out>/predictions.csv.gz``). Oblast numbers are derived from them.

    python -m src.site_data --geo path/to/geojson_dir --pred reports/raion_fresh
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .config import HORIZONS, ROOT

S = 200.0                    # map units per degree of latitude
LAT_TOP = 52.5
LON_LEFT = 22.0
COS = float(np.cos(np.radians(48.5)))
EPS = 0.9                    # simplification tolerance, map units (about 0.0045 degrees)
MIN_RING = 2.5               # drop rings smaller than this (map units) in both directions
Q = 77                       # forecast probabilities are stored in Q+1 steps, one character each
MODEL = "gbm_text"

RENAMED_RAIONS = {           # raion names in the model's data -> names in the 2025 boundaries
    "Володимир-Волинський": "Володимирський",
    "Червоноградський": "Шептицький",
    "Красноградський": "Берестинський",
}
KYIV_CITY = "м. Київ"


def norm(s: str) -> str:
    return s.replace("’", "'").replace("ʼ", "'")


def project(lon, lat):
    return np.column_stack([(np.asarray(lon) - LON_LEFT) * COS * S, (LAT_TOP - np.asarray(lat)) * S])


def rdp(pts: np.ndarray, eps: float) -> np.ndarray:
    n = len(pts)
    if n < 3:
        return pts
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        p, q = pts[a], pts[b]
        seg = pts[a + 1:b] - p
        d = q - p
        norm_d = np.hypot(*d)
        if norm_d == 0:
            dist = np.hypot(seg[:, 0], seg[:, 1])
        else:
            dist = np.abs(seg[:, 0] * d[1] - seg[:, 1] * d[0]) / norm_d
        k = int(np.argmax(dist))
        if dist[k] > eps:
            m = a + 1 + k
            keep[m] = True
            stack.append((a, m))
            stack.append((m, b))
    return pts[keep]


def simplify_ring(ring) -> np.ndarray | None:
    pts = project([c[0] for c in ring], [c[1] for c in ring])
    if np.ptp(pts[:, 0]) < MIN_RING and np.ptp(pts[:, 1]) < MIN_RING:
        return None
    k = int(np.argmax(np.hypot(*(pts - pts[0]).T)))        # split a closed ring at its far point
    out = np.vstack([rdp(pts[:k + 1], EPS), rdp(pts[k:], EPS)[1:]])
    out = np.round(out).astype(int)
    keep = np.r_[True, np.any(np.diff(out, axis=0) != 0, axis=1)]
    out = out[keep]
    return out if len(out) >= 4 else None


def path_d(geom) -> str:
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    parts = []
    for poly in polys:
        for ring in poly:
            r = simplify_ring(ring)
            if r is None:
                continue
            d = np.diff(r, axis=0)
            parts.append(f"M{r[0][0]} {r[0][1]}l" + " ".join(f"{a} {b}" for a, b in d) + "z")
    return "".join(parts)


def centroid(geom) -> list[int]:
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    best = max(polys, key=lambda p: len(p[0]))
    pts = project([c[0] for c in best[0]], [c[1] for c in best[0]])
    c = pts.mean(axis=0)
    return [int(round(c[0])), int(round(c[1]))]


def build(geo_dir: Path, pred_dir: Path, out: Path) -> None:
    a1 = json.load(open(geo_dir / "ukr_admin1.geojson"))["features"]
    a2 = json.load(open(geo_dir / "ukr_admin2.geojson"))["features"]

    pred = pd.read_csv(pred_dir / "predictions.csv.gz", parse_dates=["time"])
    unit_names = sorted(pred["unit"].unique())

    # ---- match forecast units to boundary polygons -------------------------------------
    by_key = {}
    for f in a2:
        pr = f["properties"]
        by_key[(norm(pr["adm1_name1"]) + " область", norm(pr["adm2_name1"]) + " район")] = pr
    a1_by_name = {norm(f["properties"]["adm1_name1"]) + " область": f["properties"] for f in a1}
    kyiv_city_adm2 = next(f["properties"] for f in a2 if f["properties"]["adm2_name1"] == "Київ")
    kyiv_city_adm1 = next(f["properties"] for f in a1 if f["properties"]["adm1_name1"] == "Київ")

    units, unit_of_pcode = [], {}
    for name in unit_names:
        if name == KYIV_CITY:
            units.append({"k": name, "r": kyiv_city_adm2["adm2_pcode"], "o": kyiv_city_adm1["adm1_pcode"],
                          "uk": "м. Київ", "en": "Kyiv city", "ouk": "м. Київ", "oen": "Kyiv city"})
            unit_of_pcode[kyiv_city_adm2["adm2_pcode"]] = len(units) - 1
            continue
        obl, rai = (norm(x) for x in name.split(" / "))
        rai_key = rai
        for old, new in RENAMED_RAIONS.items():
            if rai.startswith(old):
                rai_key = new + " район"
        pr = by_key[(obl, rai_key)]                      # KeyError = a unit without a polygon
        o = a1_by_name[obl]
        units.append({"k": name, "r": pr["adm2_pcode"], "o": pr["adm1_pcode"],
                      "uk": pr["adm2_name1"] + " район", "en": pr["adm2_name"] + " raion",
                      "ouk": o["adm1_name1"] + " область", "oen": o["adm1_name"] + " oblast"})
        unit_of_pcode[pr["adm2_pcode"]] = len(units) - 1

    # ---- geometry ------------------------------------------------------------------------
    raions = []
    for f in a2:
        pr = f["properties"]
        u = unit_of_pcode.get(pr["adm2_pcode"], -1)
        raions.append({"id": pr["adm2_pcode"], "u": u, "uk": pr["adm2_name1"], "en": pr["adm2_name"],
                       "o": pr["adm1_pcode"], "d": path_d(f["geometry"])})
    oblasts = []
    for f in a1:
        pr = f["properties"]
        oblasts.append({"id": pr["adm1_pcode"], "uk": pr["adm1_name1"], "en": pr["adm1_name"],
                        "d": path_d(f["geometry"]), "c": centroid(f["geometry"])})
    ky = next(f for f in a2 if f["properties"]["adm2_pcode"] == kyiv_city_adm2["adm2_pcode"])
    geo = {"w": int(np.ceil((40.3 - LON_LEFT) * COS * S)), "h": int(np.ceil((LAT_TOP - 44.0) * S)),
           "raions": raions, "oblasts": oblasts, "kyiv": centroid(ky["geometry"]),
           "units": [{k: v for k, v in u.items() if k != "k"} for u in units]}
    (out / "geo.js").write_text("// Generated by src/site_data.py. Boundaries: OCHA COD-AB Ukraine, CC BY-IGO.\n"
                                "window.GEO=" + json.dumps(geo, ensure_ascii=False, separators=(",", ":")) + ";\n")

    # ---- forecasts -----------------------------------------------------------------------
    t0 = pred["time"].min()
    n = int((pred["time"].max() - t0) / pd.Timedelta(hours=1)) + 1
    uidx = {u["k"]: i for i, u in enumerate(units)}
    pred["ui"] = pred["unit"].map(uidx)
    pred["ti"] = ((pred["time"] - t0) / pd.Timedelta(hours=1)).astype(int)

    def enc(vals, missing):
        return "".join(chr(47) if m else chr(48 + int(round(v * Q))) for v, m in zip(vals, missing))

    P, Y, valid = {}, {}, {}
    for H in HORIZONS:
        d = pred[pred["H"] == H]
        pm = np.full((len(units), n), np.nan)
        ym = np.full((len(units), n), np.nan)
        pm[d["ui"], d["ti"]] = d[MODEL].to_numpy()
        ym[d["ui"], d["ti"]] = d["y"].to_numpy()
        P[H] = [enc(np.nan_to_num(r), np.isnan(r)) for r in pm]
        Y[H] = ["".join("." if np.isnan(v) else str(int(v)) for v in r) for r in ym]
        valid[H] = int(d["ti"].max()) + 1
    d1 = pred[pred["H"] == 1]
    am = np.full((len(units), n), np.nan)
    am[d1["ui"], d1["ti"]] = d1["active_now"].to_numpy()
    A = ["".join("." if np.isnan(v) else str(int(v)) for v in r) for r in am]

    # ---- how well do oblast numbers (noisy-OR over raions) hold up? ---------------------
    pred["obl"] = pred["unit"].map(lambda s: units[uidx[s]]["o"])
    check = {}
    for H in HORIZONS:
        d = pred[pred["H"] == H].dropna(subset=["y", MODEL])
        g = d.groupby(["time", "obl"])
        agg = pd.DataFrame({"p": g[MODEL].apply(lambda s: 1 - np.prod(1 - s)), "y": g["y"].max()})
        check[H] = {"auc": round(float(roc_auc_score(agg["y"], agg["p"])), 3),
                    "brier": round(float(brier_score_loss(agg["y"], agg["p"])), 3),
                    "base": round(float(agg["y"].mean()), 3), "meanp": round(float(agg["p"].mean()), 3)}

    fc = {"t0": t0.isoformat(), "n": n, "valid": valid, "q": Q, "model": MODEL, "check": check,
          "p": P, "y": Y, "a": A}
    (out / "forecast.js").write_text("// Generated by src/site_data.py from " + str(pred_dir.relative_to(ROOT)) +
                                     "/predictions.csv.gz (walk-forward, out of sample).\nwindow.FORECAST=" +
                                     json.dumps(fc, separators=(",", ":")) + ";\n")
    print("units", len(units), "raions", len(raions), "oblasts", len(oblasts), "hours", n, valid)
    print({k: (out / k).stat().st_size // 1024 for k in ["geo.js", "forecast.js"]}, "KB")
    print("oblast aggregate check:", check)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo", required=True, type=Path, help="folder with ukr_admin1.geojson and ukr_admin2.geojson")
    ap.add_argument("--pred", default=ROOT / "reports" / "raion_fresh", type=Path)
    ap.add_argument("--out", default=ROOT / "docs", type=Path)
    a = ap.parse_args()
    build(a.geo, a.pred.resolve(), a.out)
