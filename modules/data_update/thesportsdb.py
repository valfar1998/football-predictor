"""Fixtures coppe da TheSportsDB (fallback quando altre fonti mancano)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
CUPS_DIR = RAW_DIR / "fd" / "cups"
KEY_PATH = RAW_DIR / "thesportsdb.key"
CACHE_PATH = CUPS_DIR / "thesportsdb_fixtures.csv"
BASE = "https://www.thesportsdb.com/api/v1/json"
UA = "Mozilla/5.0 (compatible; football-predictor/1.0; +local)"

# IDs noti e stabili per coppe UEFA (v1 free key=3).
DEFAULT_LEAGUE_IDS = (4480, 4481)


def _api_key() -> str:
    cands: list[str] = []
    for key in ("THESPORTSDB_API_KEY", "THESPORTSDB_KEY"):
        val = (os.environ.get(key) or "").strip()
        if val:
            cands.append(val)
    if KEY_PATH.exists():
        val = KEY_PATH.read_text(encoding="utf-8").strip()
        if val:
            cands.append(val)
    for val in cands:
        if len(val) == 32 and all(c in "0123456789abcdefABCDEF" for c in val):
            continue  # chiave API-Football salvata per errore
        return val
    return "123"


def save_api_key(key: str) -> Path:
    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    KEY_PATH.write_text(key.strip(), encoding="utf-8")
    return KEY_PATH


def _get(path: str, *, key: str) -> dict:
    req = Request(f"{BASE}/{key}/{path}", headers={"User-Agent": UA})
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _event_to_row(ev: dict) -> dict | None:
    from modules.data_update.cups import classify_cup, resolve_known_team

    league = str(ev.get("strLeague") or "").strip()
    cup = classify_cup(league)
    if not cup:
        return None
    country, league_name, code = cup
    date_s = str(ev.get("dateEvent") or "").strip()
    if not date_s:
        return None
    date = pd.to_datetime(date_s, errors="coerce")
    if pd.isna(date):
        return None
    home_raw = str(ev.get("strHomeTeam") or "").strip()
    away_raw = str(ev.get("strAwayTeam") or "").strip()
    home = resolve_known_team(home_raw) or home_raw
    away = resolve_known_team(away_raw) or away_raw
    if not home or not away:
        return None
    tm = str(ev.get("strTime") or "").strip()
    if len(tm) >= 5:
        tm = tm[:5]
    return {
        "date": pd.Timestamp(date).normalize(),
        "time": tm,
        "home_team": home,
        "away_team": away,
        "country": country,
        "league": league_name,
        "div": code,
        "source": "fixtures-cups-thesportsdb",
        "venue": str(ev.get("strVenue") or "").strip(),
        "venue_city": str(ev.get("strCity") or ev.get("strCountry") or "").strip(),
        "venue_neutral": False,
        "odd_home": None,
        "odd_draw": None,
        "odd_away": None,
        "odd_over_25": None,
        "odd_under_25": None,
    }


def download_cup_fixtures(
    *,
    key: str | None = None,
    league_ids: tuple[int, ...] = DEFAULT_LEAGUE_IDS,
) -> dict:
    """Scarica prossimi eventi coppe da eventsnextleague (TheSportsDB v1)."""
    key = (key or _api_key()).strip() or "3"
    rows: list[dict] = []
    errors: list[str] = []
    for lid in league_ids:
        try:
            data = _get(f"eventsnextleague.php?id={lid}", key=key)
            events = data.get("events") or []
            for ev in events:
                row = _event_to_row(ev)
                if row:
                    rows.append(row)
            time.sleep(0.35)
        except HTTPError as exc:
            errors.append(f"{lid}: HTTP {exc.code}")
            if exc.code == 429:
                break
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            errors.append(f"{lid}: {exc}")
    CUPS_DIR.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.DataFrame(rows).drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
        df.to_csv(CACHE_PATH, index=False)
        try:
            from modules.data_update.venues import update_home_venues

            update_home_venues(df)
        except Exception:
            pass
    return {
        "n_cup_files": 1 if rows else 0,
        "n_cup_fixtures": len(rows),
        "source": "thesportsdb",
        "key": "custom" if key != "3" else "free-demo",
        "errors": errors,
    }


def load_cup_fixtures() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(CACHE_PATH, parse_dates=["date"])
    df["source"] = "fixtures-cups-thesportsdb"
    return df.dropna(subset=["date", "home_team", "away_team"])
