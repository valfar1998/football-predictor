"""Coppe continentali: Champions, Libertadores, AFC CL e analoghe.

football-data.co.uk non le pubblica. Il calendario coppe arriva da
GET https://api.football-data.org/v4/matches (token gratis), da TheSportsDB
come fallback e, se c'è cache, da AsianBetSoccer.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CUPS_DIR = ROOT / "data" / "raw" / "fd" / "cups"
TOKEN_PATH = ROOT / "data" / "raw" / "football-data.org.token"
ORG_BASE = "https://api.football-data.org/v4"
UA = "Mozilla/5.0 (compatible; football-predictor/1.0; +local)"

# regex (specifiche prima), paese, nome UI, codice
CUP_PATTERNS: list[tuple[re.Pattern[str], str, str, str]] = [
    (re.compile(r"champions league quali|cl quali|qualificazion[ei] champions", re.I), "Europa", "Qualificazioni Champions League", "CLQ"),
    (re.compile(r"europa league quali|el quali|qualificazion[ei] europa", re.I), "Europa", "Qualificazioni Europa League", "ELQ"),
    (re.compile(r"conference league quali|qualificazion[ei] conference", re.I), "Europa", "Qualificazioni Conference League", "COLQ"),
    (re.compile(r"uefa super ?cup|supercoppa (uefa|europea)", re.I), "Europa", "Supercoppa UEFA", "ESC"),
    (re.compile(r"conference league|uecl\b", re.I), "Europa", "Conference League", "UECL"),
    (re.compile(r"europa league|uefa cup|coppa uefa", re.I), "Europa", "Europa League", "EL"),
    (re.compile(r"afc champions league (two|2)|afc cl ?2|afc cup\b", re.I), "Asia", "AFC Champions League Two", "ACL2"),
    (re.compile(r"afc champions|acl elite|afc elite|champions league asiatica", re.I), "Asia", "AFC Champions League", "ACL"),
    (re.compile(r"concacaf champion|champions cup concacaf|coppa dei campioni concacaf", re.I), "Nordamerica", "CONCACAF Champions Cup", "CCL"),
    (re.compile(r"\bcaf confeder|coppa confederazione", re.I), "Africa", "CAF Confederation Cup", "CAFCC"),
    (re.compile(r"\bcaf champions|champions league africana", re.I), "Africa", "CAF Champions League", "CAFCL"),
    (re.compile(r"uefa champions|champions league|ucl\b|liga dei campioni|coppa dei campioni", re.I), "Europa", "Champions League", "CL"),
    (re.compile(r"recopa", re.I), "Sudamerica", "Recopa Sudamericana", "REC"),
    (re.compile(r"sudamericana", re.I), "Sudamerica", "Copa Sudamericana", "CS"),
    (re.compile(r"libertadores", re.I), "Sudamerica", "Copa Libertadores", "CLI"),
    (re.compile(r"copa america", re.I), "Sudamerica", "Copa America", "CA"),
    (re.compile(r"asian cup|coppa d.?asia", re.I), "Asia", "Coppa d'Asia", "ASC"),
    (re.compile(r"africa cup|coppa d.?africa|\bafcon\b", re.I), "Africa", "Coppa d'Africa", "AC"),
    (re.compile(r"leagues cup", re.I), "Nordamerica", "Leagues Cup", "LCUP"),
    (re.compile(r"club world|mondiale per club|fifa intercontinental|coppa intercontinentale", re.I), "Mondo", "Mondiale per club", "FCWC"),
    (re.compile(r"world cup quali|qualificazion[ei] mondial", re.I), "Mondo", "Qualificazioni Mondiali", "WCQ"),
    (re.compile(r"european championship|europei\b|\beuro 20", re.I), "Europa", "Europei", "EC"),
    (re.compile(r"world cup|mondiali", re.I), "Mondo", "Mondiali", "WC"),
]

# Codici competizioni da richiedere: se alcuni non sono accettati dal piano/API,
# la funzione fa fallback a /matches non filtrato e filtra localmente.
ORG_MATCH_CODES = ("CL", "EL", "ECL", "UECL", "UEL", "UCL", "EC", "WC", "CLI", "CS")

ORG_CODE_META = {
    "CL": ("Europa", "Champions League"),
    "UCL": ("Europa", "Champions League"),
    "EL": ("Europa", "Europa League"),
    "UEL": ("Europa", "Europa League"),
    "ECL": ("Europa", "Conference League"),
    "UECL": ("Europa", "Conference League"),
    "CLQ": ("Europa", "Qualificazioni Champions League"),
    "ELQ": ("Europa", "Qualificazioni Europa League"),
    "COLQ": ("Europa", "Qualificazioni Conference League"),
    "ESC": ("Europa", "Supercoppa UEFA"),
    "CLI": ("Sudamerica", "Copa Libertadores"),
    "CS": ("Sudamerica", "Copa Sudamericana"),
    "ACL": ("Asia", "AFC Champions League"),
    "FCWC": ("Mondo", "Mondiale per club"),
    "CA": ("Sudamerica", "Copa America"),
    "EC": ("Europa", "Europei"),
    "WC": ("Mondo", "Mondiali"),
}

from modules.data_update.team_names import (  # noqa: F401
    _norm_key,
    known_team_index,
    resolve_known_team,
)

# compat: alias storici usati da test/import
from modules.data_update.team_names import SOURCE_TEAM_ALIASES as CUP_TEAM_ALIASES


def classify_cup(league: str | None) -> tuple[str, str, str] | None:
    raw = str(league or "").strip()
    if not raw:
        return None
    for pattern, country, name, code in CUP_PATTERNS:
        if pattern.search(raw):
            return country, name, code
    return None


def asian_cup_fixtures(rows: list[dict] | None = None) -> pd.DataFrame:
    if rows is None:
        from modules.data_update.asian_odds import load_asian_odds

        rows = load_asian_odds()
    out: list[dict] = []
    for row in rows or []:
        meta = classify_cup(row.get("league"))
        if not meta:
            continue
        country, league, code = meta
        date = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.isna(date):
            continue
        home = resolve_known_team(row.get("home") or "") or str(row.get("home") or "").strip()
        away = resolve_known_team(row.get("away") or "") or str(row.get("away") or "").strip()
        if not home or not away:
            continue
        out.append(
            {
                "date": date,
                "time": row.get("time") or "",
                "home_team": home,
                "away_team": away,
                "country": country,
                "league": league,
                "div": code,
                "odd_home": row.get("odd_1"),
                "odd_draw": row.get("odd_x"),
                "odd_away": row.get("odd_2"),
                "odd_over_25": row.get("odd_over"),
                "odd_under_25": row.get("odd_under"),
                "source": "fixtures-cups-asian",
            }
        )
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out)


def _org_token() -> str | None:
    for key in ("FOOTBALL_DATA_ORG_TOKEN", "FOOTBALL_DATA_API_KEY"):
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if raw.startswith("#") or "=" not in raw:
                continue
            key, val = raw.split("=", 1)
            if key.strip() in {"FOOTBALL_DATA_ORG_TOKEN", "FOOTBALL_DATA_API_KEY"}:
                token = val.strip().strip("'").strip('"')
                if token:
                    return token
    if TOKEN_PATH.exists():
        token = TOKEN_PATH.read_text(encoding="utf-8").strip()
        if token:
            return token
    return None


def org_token_configured() -> bool:
    return bool(_org_token())


def save_org_token(token: str) -> Path:
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(token.strip(), encoding="utf-8")
    return TOKEN_PATH


def _org_get(path: str, token: str, params: dict[str, str] | None = None) -> dict:
    query = f"?{urlencode(params)}" if params else ""
    req = Request(
        f"{ORG_BASE}{path}{query}",
        headers={"User-Agent": UA, "X-Auth-Token": token},
    )
    with urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _org_matches_window(token: str, date_from: date, date_to: date) -> list[dict]:
    # Prima senza filtro competizioni: copre campionati + coppe nella finestra.
    params = {"dateFrom": date_from.isoformat(), "dateTo": date_to.isoformat()}
    try:
        data = _org_get("/matches", token, params)
        return data.get("matches") or []
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:240]
        print(f"skip /v4/matches {date_from}->{date_to}: HTTP {exc.code} {detail}")
        params["competitions"] = ",".join(ORG_MATCH_CODES)
        data = _org_get("/matches", token, params)
        return data.get("matches") or []


def _org_league_meta(match: dict) -> tuple[str, str, str] | None:
    comp = match.get("competition") or {}
    code = str(comp.get("code") or "").strip().upper()
    name = str(comp.get("name") or "").strip()
    ctype = str(comp.get("type") or "").strip().upper()
    area = str((match.get("area") or {}).get("name") or "Europa")
    if code in ORG_CODE_META:
        country, league = ORG_CODE_META[code]
        return country, league, code
    classified = classify_cup(name) or classify_cup(code)
    if classified:
        return classified
    if name or code:
        return area, name or code, code or ctype or "LEA"
    return None


def _org_match_row(match: dict, *, finished: bool) -> dict | None:
    meta = _org_league_meta(match)
    if not meta:
        return None
    country, league, code = meta
    home_raw = ((match.get("homeTeam") or {}).get("shortName") or (match.get("homeTeam") or {}).get("name") or "").strip()
    away_raw = ((match.get("awayTeam") or {}).get("shortName") or (match.get("awayTeam") or {}).get("name") or "").strip()
    home = resolve_known_team(home_raw) or home_raw
    away = resolve_known_team(away_raw) or away_raw
    utc = match.get("utcDate")
    when = pd.to_datetime(utc, errors="coerce", utc=True)
    if pd.isna(when) or not home or not away:
        return None
    local = when.tz_convert(None) if getattr(when, "tzinfo", None) else when
    venue = match.get("venue")
    if isinstance(venue, dict):
        venue_name = str(venue.get("name") or "").strip()
        venue_city = str(venue.get("city") or "").strip()
    else:
        venue_name = str(venue or "").strip()
        venue_city = ""
    item = {
        "date": pd.Timestamp(local).normalize(),
        "time": pd.Timestamp(local).strftime("%H:%M"),
        "home_team": home,
        "away_team": away,
        "country": country,
        "league": league,
        "div": code,
        "source": "fd.org:" + (code or "matches"),
        "venue": venue_name,
        "venue_city": venue_city,
        "venue_neutral": False,
    }
    if finished:
        score = (match.get("score") or {}).get("fullTime") or {}
        hg, ag = score.get("home"), score.get("away")
        if hg is None or ag is None:
            return None
        item["home_goals"] = hg
        item["away_goals"] = ag
    return item


def download_org_cups(*, token: str | None = None, days: int = 10) -> dict:
    """GET /v4/matches su una finestra di date (piano free: max ~10 giorni)."""
    token = token or _org_token()
    if not token:
        print("skip coppe football-data.org: token mancante (FOOTBALL_DATA_ORG_TOKEN)")
        return {"n_cup_files": 0, "token": False}
    today = date.today()
    target_days = max(2, int(days))
    windows: list[tuple[date, date]] = []
    start = today
    left = target_days
    while left > 0:
        span = min(10, left)
        end = start + timedelta(days=span)
        windows.append((start, end))
        start = end + timedelta(days=1)
        left -= span
    try:
        matches: list[dict] = []
        for d1, d2 in windows:
            matches.extend(_org_matches_window(token, d1, d2))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"skip coppe football-data.org: {exc}")
        return {"n_cup_files": 0, "token": True, "error": str(exc)}

    seen: set[tuple[str, str, str]] = set()
    uniq: list[dict] = []
    for m in matches:
        fx = m.get("fixture") or {}
        key = (str(m.get("utcDate") or ""), str((m.get("homeTeam") or {}).get("name") or ""), str((m.get("awayTeam") or {}).get("name") or ""))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(m)
    matches = uniq
    live = {"SCHEDULED", "TIMED", "IN_PLAY", "PAUSED", "LIVE"}
    fixtures = [row for m in matches if m.get("status") in live for row in [_org_match_row(m, finished=False)] if row]
    results = [row for m in matches if m.get("status") == "FINISHED" for row in [_org_match_row(m, finished=True)] if row]
    CUPS_DIR.mkdir(parents=True, exist_ok=True)
    n_files = 0
    if fixtures:
        pd.DataFrame(fixtures).to_csv(CUPS_DIR / "fixtures.csv", index=False)
        n_files += 1
        try:
            from modules.data_update.venues import update_home_venues

            update_home_venues(pd.DataFrame(fixtures))
        except Exception:
            pass
    if results:
        pd.DataFrame(results).to_csv(CUPS_DIR / "results.csv", index=False)
        n_files += 1
    comps = sorted({r["league"] for r in fixtures})
    print(f"ok /v4/matches: {len(fixtures)} coppe in programma ({', '.join(comps) or '—'}; {len(results)} risultati)")
    return {
        "n_cup_files": n_files,
        "token": True,
        "n_matches": len(matches),
        "n_cup_fixtures": len(fixtures),
        "competitions": comps,
        "permission": None,
    }


def load_org_cup_fixtures() -> pd.DataFrame:
    path = CUPS_DIR / "fixtures.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    df["source"] = df.get("source", "fixtures-cups-org")
    return df


def load_org_cup_results() -> pd.DataFrame:
    path = CUPS_DIR / "results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    return df.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])


def load_cup_fixtures() -> pd.DataFrame:
    try:
        from modules.data_update.thesportsdb import load_cup_fixtures as load_thesportsdb_cups

        tsdb = load_thesportsdb_cups()
    except Exception:
        tsdb = pd.DataFrame()
    try:
        from modules.data_update.api_football import load_cup_fixtures as load_api_football_cups

        apif = load_api_football_cups()
    except Exception:
        apif = pd.DataFrame()
    frames = [load_org_cup_fixtures(), asian_cup_fixtures(), tsdb, apif]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])
    today = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    df = df[df["date"] >= today]
    return df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
