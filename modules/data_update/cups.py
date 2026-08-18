"""Coppe continentali: Champions, Libertadores, AFC CL e analoghe.

football-data.co.uk non le pubblica. Il calendario arriva da AsianBetSoccer
(già usato per le quote). Con token FOOTBALL_DATA_ORG_TOKEN si scaricano anche
risultati storici da football-data.org.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CUPS_DIR = ROOT / "data" / "raw" / "fd" / "cups"
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

# football-data.org: coppe club + nazionali principali
ORG_CUP_CODES = (
    "CL",
    "EL",
    "UCL",
    "CLQ",
    "ELQ",
    "COLQ",
    "ESC",
    "CLI",
    "CS",
    "ACL",
    "FCWC",
    "CA",
    "EC",
    "WC",
)

ORG_CODE_META = {
    "CL": ("Europa", "Champions League"),
    "EL": ("Europa", "Europa League"),
    "UCL": ("Europa", "Conference League"),
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

CUP_TEAM_ALIASES = {
    "inter milan": "Inter",
    "internazionale": "Inter",
    "fc internazionale": "Inter",
    "fc internazionale milano": "Inter",
    "ac milan": "Milan",
    "manchester united": "Man United",
    "man utd": "Man United",
    "man united": "Man United",
    "manchester city": "Man City",
    "man city": "Man City",
    "atletico madrid": "Ath Madrid",
    "atlético madrid": "Ath Madrid",
    "atl madrid": "Ath Madrid",
    "atleti": "Ath Madrid",
    "athletic bilbao": "Ath Bilbao",
    "athletic club": "Ath Bilbao",
    "bayern munich": "Bayern Munich",
    "bayern munchen": "Bayern Munich",
    "fc bayern munich": "Bayern Munich",
    "fc bayern": "Bayern Munich",
    "borussia dortmund": "Dortmund",
    "bvb": "Dortmund",
    "paris saint germain": "Paris SG",
    "paris saint-germain": "Paris SG",
    "psg": "Paris SG",
    "sporting lisbon": "Sp Lisbon",
    "sporting cp": "Sp Lisbon",
    "sporting portugal": "Sp Lisbon",
    "psv": "PSV Eindhoven",
    "olympiacos": "Olympiakos",
    "olympiakos": "Olympiakos",
    "flamengo": "Flamengo RJ",
    "cr flamengo": "Flamengo RJ",
    "club america": "Club America",
    "chivas": "Guadalajara Chivas",
    "cd guadalajara": "Guadalajara Chivas",
    "tigres": "Tigres UANL",
    "red bull salzburg": "Salzburg",
    "rb salzburg": "Salzburg",
}

_NOISE = re.compile(r"\b(fc|cf|ac|sc|afc|bk|sk|fk|cd|de|the|club|calcio|ssc|us|cf)\b", re.I)


def classify_cup(league: str | None) -> tuple[str, str, str] | None:
    raw = str(league or "").strip()
    if not raw:
        return None
    for pattern, country, name, code in CUP_PATTERNS:
        if pattern.search(raw):
            return country, name, code
    return None


def _norm_key(name: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", str(name).lower())
    s = _NOISE.sub(" ", s)
    return " ".join(s.split())


def resolve_known_team(name: str, known: Iterable[str] | dict[str, str] | None = None) -> str | None:
    """Allinea il nome Asian/API allo spelling football-data.co.uk."""
    from modules.dataset_loader.loader import TEAM_ALIASES, normalize_team

    raw = str(name or "").strip()
    if not raw:
        return None
    aliased = CUP_TEAM_ALIASES.get(_norm_key(raw)) or TEAM_ALIASES.get(" ".join(raw.lower().split()))
    fallback = aliased or normalize_team(raw)
    if known is None:
        return fallback

    if isinstance(known, dict):
        index = known
        canonical = set(known.values())
    else:
        canonical = set(known)
        index = {_norm_key(t): t for t in canonical}

    if aliased and aliased in canonical:
        return aliased
    if fallback in canonical:
        return fallback
    key = _norm_key(raw)
    if key in index:
        return index[key]
    key2 = _norm_key(fallback)
    if key2 in index:
        return index[key2]
    return None


def known_team_index(names: Iterable[str]) -> dict[str, str]:
    idx: dict[str, str] = {}
    for name in names:
        idx[_norm_key(name)] = name
    return idx


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
    return os.environ.get("FOOTBALL_DATA_ORG_TOKEN") or os.environ.get("FOOTBALL_DATA_API_KEY")


def _org_get(path: str, token: str) -> dict:
    req = Request(
        f"{ORG_BASE}{path}",
        headers={"User-Agent": UA, "X-Auth-Token": token},
    )
    with urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _org_matches_frame(matches: list[dict], code: str, *, finished: bool) -> pd.DataFrame:
    country, league = ORG_CODE_META.get(code, ("Altro", code))
    rows = []
    for m in matches:
        home = ((m.get("homeTeam") or {}).get("shortName") or (m.get("homeTeam") or {}).get("name") or "").strip()
        away = ((m.get("awayTeam") or {}).get("shortName") or (m.get("awayTeam") or {}).get("name") or "").strip()
        utc = m.get("utcDate")
        date = pd.to_datetime(utc, errors="coerce", utc=True)
        if pd.isna(date) or not home or not away:
            continue
        local = date.tz_convert(None) if getattr(date, "tzinfo", None) else date
        score = (m.get("score") or {}).get("fullTime") or {}
        item = {
            "date": pd.Timestamp(local).normalize(),
            "time": pd.Timestamp(local).strftime("%H:%M"),
            "home_team": home,
            "away_team": away,
            "country": country,
            "league": league,
            "div": code,
            "source": "fd.org:" + code,
        }
        if finished:
            hg, ag = score.get("home"), score.get("away")
            if hg is None or ag is None:
                continue
            item["home_goals"] = hg
            item["away_goals"] = ag
        rows.append(item)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def download_org_cups(*, token: str | None = None) -> dict:
    """Scarica coppe da football-data.org se c'è un token gratuito."""
    token = token or _org_token()
    if not token:
        print("skip coppe football-data.org: imposta FOOTBALL_DATA_ORG_TOKEN")
        return {"n_cup_files": 0, "token": False}
    CUPS_DIR.mkdir(parents=True, exist_ok=True)
    fixtures: list[pd.DataFrame] = []
    results: list[pd.DataFrame] = []
    errors: list[str] = []
    for i, code in enumerate(ORG_CUP_CODES):
        if i:
            time.sleep(6.5)
        try:
            data = _org_get(f"/competitions/{code}/matches", token)
        except HTTPError as exc:
            errors.append(f"{code}: HTTP {exc.code}")
            print(f"skip coppa {code}: HTTP {exc.code}")
            continue
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            errors.append(f"{code}: {exc}")
            print(f"skip coppa {code}: {exc}")
            continue
        matches = data.get("matches") or []
        scheduled = [m for m in matches if m.get("status") in {"SCHEDULED", "TIMED", "IN_PLAY"}]
        finished = [m for m in matches if m.get("status") == "FINISHED"]
        fx = _org_matches_frame(scheduled, code, finished=False)
        rs = _org_matches_frame(finished, code, finished=True)
        if not fx.empty:
            fixtures.append(fx)
        if not rs.empty:
            results.append(rs)
        print(f"ok coppa {code}: {len(scheduled)} in programma, {len(finished)} risultati")
    n_files = 0
    if fixtures:
        pd.concat(fixtures, ignore_index=True).to_csv(CUPS_DIR / "fixtures.csv", index=False)
        n_files += 1
    if results:
        pd.concat(results, ignore_index=True).to_csv(CUPS_DIR / "results.csv", index=False)
        n_files += 1
    return {"n_cup_files": n_files, "token": True, "errors": errors}


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
    frames = [load_org_cup_fixtures(), asian_cup_fixtures()]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])
    today = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    df = df[df["date"] >= today]
    return df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
