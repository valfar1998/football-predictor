"""Fallback coppe da API-Football (api-sports)."""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
CUPS_DIR = RAW_DIR / "fd" / "cups"
KEY_PATH = RAW_DIR / "api-football.key"
CACHE_PATH = CUPS_DIR / "api_football_fixtures.csv"
BASE = "https://v3.football.api-sports.io"
UA = "Mozilla/5.0 (compatible; football-predictor/1.0; +local)"

# API-Football league ids: CL, EL, ECL, Libertadores
LEAGUE_IDS = (2, 3, 848, 13)


def _api_key() -> str | None:
    for k in ("API_FOOTBALL_KEY", "APISPORTS_KEY"):
        val = (os.environ.get(k) or "").strip()
        if val:
            return val
    if KEY_PATH.exists():
        val = KEY_PATH.read_text(encoding="utf-8").strip()
        if val:
            return val
    return None


def save_api_key(key: str) -> Path:
    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    KEY_PATH.write_text(key.strip(), encoding="utf-8")
    return KEY_PATH


def _get(path: str, *, key: str, params: dict[str, str]) -> dict:
    q = urlencode(params)
    req = Request(
        f"{BASE}{path}?{q}",
        headers={"x-apisports-key": key, "User-Agent": UA},
    )
    with urlopen(req, timeout=35) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fixture_row(item: dict) -> dict | None:
    from modules.data_update.cups import classify_cup, resolve_known_team

    lg = item.get("league") or {}
    lg_name = str(lg.get("name") or "").strip()
    cup = classify_cup(lg_name)
    if not cup:
        return None
    country, league_name, code = cup
    fx = item.get("fixture") or {}
    teams = item.get("teams") or {}
    dt = pd.to_datetime(fx.get("date"), errors="coerce", utc=True)
    if pd.isna(dt):
        return None
    dt_local = dt.tz_convert(None)
    home_raw = str((teams.get("home") or {}).get("name") or "").strip()
    away_raw = str((teams.get("away") or {}).get("name") or "").strip()
    home = resolve_known_team(home_raw) or home_raw
    away = resolve_known_team(away_raw) or away_raw
    if not home or not away:
        return None
    ven = (fx.get("venue") or {}) if isinstance(fx.get("venue"), dict) else {}
    return {
        "date": pd.Timestamp(dt_local).normalize(),
        "time": pd.Timestamp(dt_local).strftime("%H:%M"),
        "home_team": home,
        "away_team": away,
        "country": country,
        "league": league_name,
        "div": code,
        "source": "fixtures-cups-api-football",
        "venue": str(ven.get("name") or "").strip(),
        "venue_city": str(ven.get("city") or "").strip(),
        "venue_neutral": False,
        "odd_home": None,
        "odd_draw": None,
        "odd_away": None,
        "odd_over_25": None,
        "odd_under_25": None,
    }


def download_cup_fixtures(*, key: str | None = None, days: int = 10) -> dict:
    key = (key or _api_key() or "").strip()
    if not key:
        return {"n_cup_files": 0, "token": False, "source": "api-football"}
    today = date.today()
    date_from = today.isoformat()
    date_to = (today + timedelta(days=max(2, days))).isoformat()
    rows: list[dict] = []
    errors: list[str] = []
    for lid in LEAGUE_IDS:
        try:
            data = _get(
                "/fixtures",
                key=key,
                params={
                    "league": str(lid),
                    "season": str(today.year),
                    "from": date_from,
                    "to": date_to,
                },
            )
            errs = data.get("errors") or {}
            if errs:
                errors.append(f"{lid}: {errs}")
                continue
            for item in data.get("response") or []:
                row = _fixture_row(item)
                if row:
                    rows.append(row)
        except HTTPError as exc:
            errors.append(f"{lid}: HTTP {exc.code}")
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
        "token": True,
        "source": "api-football",
        "errors": errors,
    }


def load_cup_fixtures() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(CACHE_PATH, parse_dates=["date"])
    df["source"] = "fixtures-cups-api-football"
    return df.dropna(subset=["date", "home_team", "away_team"])
