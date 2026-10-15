"""Project-wide constants. Everything time-related is UTC."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

SIRENS_CSV_URL = (
    "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/"
    "main/datasets/official_data_uk.csv"
)

# Alerts became almost exclusively raion-level from Nov 2025, so we start training there
# to keep one consistent "regime" (see README, "Data").
HISTORY_START = pd.Timestamp("2025-11-01", tz="UTC")

# Forecast horizons in hours.
HORIZONS = [1, 3, 6, 24]

KYIV_CITY = "м. Київ"

# Luhansk oblast has 2 records in the whole dataset (occupied), so it is excluded.
OBLASTS = [
    "Вінницька область", "Волинська область", "Дніпропетровська область",
    "Донецька область", "Житомирська область", "Закарпатська область",
    "Запорізька область", "Івано-Франківська область", "Київська область",
    "Кіровоградська область", "Львівська область", "Миколаївська область",
    "Одеська область", "Полтавська область", "Рівненська область",
    "Сумська область", "Тернопільська область", "Харківська область",
    "Херсонська область", "Хмельницька область", "Черкаська область",
    "Чернівецька область", "Чернігівська область", "м. Київ",
]

# The third-party oblast-level CSV is inaccurate between these moments (user-reported, confirmed:
# 3-4x fewer records per day, a separate "Київ" region with synthetic 30-minute alerts). Neither
# source covers the period reliably, so it is treated as missing data.
OBLAST_CSV_GAP = (pd.Timestamp("2026-09-07", tz="UTC"), pd.Timestamp("2026-09-13", tz="UTC"))
