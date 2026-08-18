"""Calendario mondiale: tutte le partite del giorno (non solo coppe)."""

from __future__ import annotations

import json
import os
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CACHE = RAW / "fixtures" / "world.csv"
UA = "Mozilla/5.0 (compatible; football-predictor/1.0; +local)"
TSDB_BASE = "https://www.thesportsdb.com/api/v1/json"
APIF_BASE = "https://v3.football.api-sports.io"
_OPENFOOTBALL_BASE = "https://raw.githubusercontent.com/openfootball"
_OPENFOOTBALL_FILES = (
    ("england/master/2026-27/1-premierleague.txt", "Inghilterra", "Premier League"),
    ("england/master/2026-27/2-championship.txt", "Inghilterra", "Championship"),
    ("italy/master/2026-27/1-seriea.txt", "Italia", "Serie A"),
    ("espana/master/2026-27/1-liga.txt", "Spagna", "La Liga"),
    ("deutschland/master/2026-27/1-bundesliga.txt", "Germania", "Bundesliga"),
)
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def _tsdb_key() -> str:
    """TheSportsDB free key is 123. Ignore chiavi API-Football salvate per errore."""
    cands: list[str] = []
    for k in ("THESPORTSDB_API_KEY", "THESPORTSDB_KEY"):
        val = (os.environ.get(k) or "").strip()
        if val:
            cands.append(val)
    p = RAW / "thesportsdb.key"
    if p.exists():
        val = p.read_text(encoding="utf-8").strip()
        if val:
            cands.append(val)
    for val in cands:
        if len(val) == 32 and all(c in "0123456789abcdefABCDEF" for c in val):
            continue
        return val
    return "123"


def _apif_key() -> str | None:
    for k in ("API_FOOTBALL_KEY", "APISPORTS_KEY"):
        val = (os.environ.get(k) or "").strip()
        if val:
            return val
    p = RAW / "api-football.key"
    if p.exists():
        val = p.read_text(encoding="utf-8").strip()
        if val:
            return val
    return None


def _get_json(url: str, headers: dict[str, str] | None = None) -> dict:
    req = Request(url, headers={"User-Agent": UA, **(headers or {})})
    with urlopen(req, timeout=35) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _row(
    *,
    when,
    tm: str,
    home: str,
    away: str,
    country: str,
    league: str,
    source: str,
    venue: str = "",
    venue_city: str = "",
    venue_neutral: bool = False,
) -> dict | None:
    from modules.data_update.cups import resolve_known_team

    dt = pd.to_datetime(when, errors="coerce")
    if pd.isna(dt) or not home or not away:
        return None
    if getattr(dt, "tzinfo", None):
        dt = dt.tz_convert(None)
    return {
        "date": pd.Timestamp(dt).normalize(),
        "time": tm or pd.Timestamp(dt).strftime("%H:%M"),
        "home_team": resolve_known_team(home) or home,
        "away_team": resolve_known_team(away) or away,
        "country": country or "",
        "league": league or "",
        "div": "",
        "source": source,
        "venue": venue or "",
        "venue_city": venue_city or "",
        "venue_neutral": bool(venue_neutral),
        "odd_home": None,
        "odd_draw": None,
        "odd_away": None,
        "odd_over_25": None,
        "odd_under_25": None,
    }


def _tsdb_day(day: date, key: str) -> list[dict]:
    url = f"{TSDB_BASE}/{key}/eventsday.php?d={day.isoformat()}&s=Soccer"
    data = _get_json(url)
    out: list[dict] = []
    for ev in data.get("events") or []:
        tm = str(ev.get("strTime") or "")[:5]
        row = _row(
            when=ev.get("dateEvent") or day.isoformat(),
            tm=tm,
            home=str(ev.get("strHomeTeam") or "").strip(),
            away=str(ev.get("strAwayTeam") or "").strip(),
            country=str(ev.get("strCountry") or ev.get("strLeagueAlternate") or "").strip(),
            league=str(ev.get("strLeague") or "").strip(),
            source="fixtures-world-thesportsdb",
            venue=str(ev.get("strVenue") or "").strip(),
            venue_city=str(ev.get("strCity") or ev.get("strCountry") or "").strip(),
        )
        if row:
            out.append(row)
    return out


def _apif_day(day: date, key: str) -> list[dict]:
    q = urlencode({"date": day.isoformat()})
    data = _get_json(f"{APIF_BASE}/fixtures?{q}", headers={"x-apisports-key": key})
    errs = data.get("errors") or {}
    if errs:
        raise RuntimeError(str(errs))
    out: list[dict] = []
    for item in data.get("response") or []:
        fx = item.get("fixture") or {}
        lg = item.get("league") or {}
        teams = item.get("teams") or {}
        status = str((fx.get("status") or {}).get("short") or "")
        if status in {"FT", "AET", "PEN", "CANC", "ABD", "AWD", "WO"}:
            continue
        ven = fx.get("venue") or {}
        row = _row(
            when=fx.get("date"),
            tm="",
            home=str((teams.get("home") or {}).get("name") or "").strip(),
            away=str((teams.get("away") or {}).get("name") or "").strip(),
            country=str(lg.get("country") or "").strip(),
            league=str(lg.get("name") or "").strip(),
            source="fixtures-world-api-football",
            venue=str(ven.get("name") or "").strip(),
            venue_city=str(ven.get("city") or "").strip(),
        )
        if row:
            out.append(row)
    return out


def _parse_openfootball_txt(text: str, *, country: str, league: str, days: int) -> list[dict]:
    today = pd.Timestamp.now().normalize()
    until = today + pd.Timedelta(days=max(1, days))
    year = int(today.year)
    cur_date: pd.Timestamp | None = None
    cur_time = "15:00"
    out: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "=", "▪", "[", "-")):
            continue
        parts = line.split()
        if parts and parts[0][:3] in _WEEKDAYS and len(parts) >= 3 and parts[1][:3] in _MONTHS:
            try:
                day_n = int(parts[2])
                month = _MONTHS[parts[1][:3]]
                if len(parts) >= 4 and parts[3].isdigit() and len(parts[3]) == 4:
                    year = int(parts[3])
                cur_date = pd.Timestamp(year=year, month=month, day=day_n)
                continue
            except (ValueError, KeyError):
                pass
        if " v " not in line:
            continue
        tm = cur_time
        rest = line
        if len(line) >= 5 and line[0].isdigit() and ":" in line[:5]:
            tm = line[:5]
            rest = line[5:].strip()
            cur_time = tm
        home, _, away = rest.partition(" v ")
        home, away = home.strip(), away.strip()
        if cur_date is None or not home or not away:
            continue
        if cur_date < today or cur_date > until:
            continue
        when = cur_date + pd.Timedelta(hours=int(tm[:2]), minutes=int(tm[3:5]))
        row = _row(
            when=when,
            tm=tm,
            home=home,
            away=away,
            country=country,
            league=league,
            source="fixtures-world-openfootball",
        )
        if row:
            out.append(row)
    return out


def _openfootball_upcoming(*, days: int) -> list[dict]:
    out: list[dict] = []
    for rel, country, league in _OPENFOOTBALL_FILES:
        url = f"{_OPENFOOTBALL_BASE}/{rel}"
        try:
            req = Request(url, headers={"User-Agent": UA})
            with urlopen(req, timeout=25) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except Exception:
            continue
        out.extend(_parse_openfootball_txt(text, country=country, league=league, days=days))
    return out


def _openligadb_upcoming(*, days: int) -> list[dict]:
    shortcuts = ("bl1", "bl2", "bl3", "dfb")
    names = {
        "bl1": ("Germania", "Bundesliga"),
        "bl2": ("Germania", "2. Bundesliga"),
        "bl3": ("Germania", "3. Liga"),
        "dfb": ("Germania", "DFB-Pokal"),
    }
    today = pd.Timestamp.now().normalize()
    until = today + pd.Timedelta(days=max(1, days))
    out: list[dict] = []
    for code in shortcuts:
        try:
            data = _get_json(f"https://api.openligadb.de/getmatchdata/{code}")
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        country, league = names[code]
        for m in data:
            if m.get("matchIsFinished"):
                continue
            when = m.get("matchDateTimeUTC") or m.get("matchDateTime")
            dt = pd.to_datetime(when, errors="coerce")
            if pd.isna(dt):
                continue
            if getattr(dt, "tzinfo", None):
                dt = dt.tz_convert(None)
            if dt.normalize() < today or dt.normalize() > until:
                continue
            t1 = ((m.get("team1") or {}).get("teamName") or "").strip()
            t2 = ((m.get("team2") or {}).get("teamName") or "").strip()
            row = _row(
                when=dt,
                tm=pd.Timestamp(dt).strftime("%H:%M"),
                home=t1,
                away=t2,
                country=country,
                league=league,
                source="fixtures-world-openligadb",
            )
            if row:
                out.append(row)
    return out


def download_world_fixtures(*, days: int = 14) -> dict:
    """Scarica tutte le partite di calcio per i prossimi N giorni."""
    days = max(1, min(14, int(days)))
    today = date.today()
    rows: list[dict] = []
    errors: list[str] = []
    tsdb_n = 0
    apif_n = 0
    oldb_n = 0
    of_n = 0
    tsdb_key = _tsdb_key()
    apif_key = _apif_key()
    tsdb_days = min(3, days)
    apif_ok = bool(apif_key)

    for i in range(days):
        day = today + timedelta(days=i)
        if i < tsdb_days:
            try:
                part = _tsdb_day(day, tsdb_key)
                rows.extend(part)
                tsdb_n += len(part)
            except HTTPError as exc:
                errors.append(f"tsdb {day}: HTTP {exc.code}")
                if exc.code == 429:
                    tsdb_days = i
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                errors.append(f"tsdb {day}: {exc}")
            time.sleep(0.35)

        if apif_ok:
            try:
                part = _apif_day(day, apif_key)
                rows.extend(part)
                apif_n += len(part)
            except HTTPError as exc:
                errors.append(f"apif {day}: HTTP {exc.code}")
                if exc.code in {403, 429}:
                    apif_ok = False
            except Exception as exc:
                msg = str(exc)
                errors.append(f"apif {day}: {exc}")
                if "Free plans do not have access to this date" in msg:
                    apif_ok = False

    try:
        oldb = _openligadb_upcoming(days=days)
        rows.extend(oldb)
        oldb_n = len(oldb)
    except Exception as exc:
        errors.append(f"openligadb: {exc}")

    try:
        ofb = _openfootball_upcoming(days=days)
        rows.extend(ofb)
        of_n = len(ofb)
    except Exception as extra:
        errors.append(f"openfootball: {extra}")

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    n_unique = 0
    if rows:
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "home_team", "away_team"])
        df = df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
        n_unique = int(len(df))
        df.to_csv(CACHE, index=False)
        try:
            from modules.data_update.venues import update_home_venues

            update_home_venues(df)
        except Exception:
            pass
    print(
        f"ok calendario mondiale: {n_unique} partite "
        f"({days} giorni, tsdb={tsdb_n}, apif={apif_n}, oldb={oldb_n}, ofb={of_n})"
    )
    return {
        "n_world_fixtures": n_unique,
        "n_tsdb": tsdb_n,
        "n_apif": apif_n,
        "n_openligadb": oldb_n,
        "n_openfootball": of_n,
        "days": days,
        "errors": errors,
        "path": str(CACHE),
    }


def load_world_fixtures() -> pd.DataFrame:
    if not CACHE.exists():
        return pd.DataFrame()
    df = pd.read_csv(CACHE, parse_dates=["date"])
    return df.dropna(subset=["date", "home_team", "away_team"])
