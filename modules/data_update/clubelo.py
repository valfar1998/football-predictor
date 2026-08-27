"""Elo esterno ClubElo. Non entra in EV/Kelly: solo quadro analisi.

L'API storica ``http://api.clubelo.com/YYYY-MM-DD`` da molte reti (TIM incluso)
accetta il TCP e poi non risponde: timeout a ogni Solo quote, cache mai scritta.
Il sito ``https://clubelo.com/Ranking`` (Cloudflare) invece risponde; è la fonte
principale. L'API resta un fallback con timeout corto.
"""

from __future__ import annotations

import io
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "clubelo.csv"
API = "http://api.clubelo.com/{day}"
RANKING_URL = "https://clubelo.com/Ranking"
UA = "Mozilla/5.0 (compatible; football-predictor/1.0; +local)"
CACHE_MAX_AGE_H = 72
API_TIMEOUT_S = 6

# football-data.co.uk → ClubElo (nomi che non coincidono dopo la normalizzazione)
_ELO_ALIASES = {
    "ath madrid": "atletico",
    "bayern munich": "bayern",
    "sp lisbon": "sporting",
    "nottm forest": "forest",
    "sheffield united": "sheffield",
    "psv eindhoven": "psv",
    "flamengo rj": "flamengo",
}
_SKIP_FUZZY = {
    "united", "city", "real", "sport", "club", "athletic", "sporting",
    "inter", "milan",
}

_ROW_HTML = re.compile(
    r'alt="([A-Z]{3})"[^>]*>\s*</a>\s*<small>\s*(\d+)\s*</small>'
    r'<a href="/([^"]+)">([^<]+).*?</td>\',\s*\'([0-9.]+)\'',
)

_DF: pd.DataFrame | None = None
_MTIME: float | None = None


def _cache_fresh() -> bool:
    if not CACHE.exists() or CACHE.stat().st_size < 200:
        return False
    age_h = (time.time() - CACHE.stat().st_mtime) / 3600.0
    return age_h < CACHE_MAX_AGE_H


def _save_cache(df: pd.DataFrame) -> pd.DataFrame:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    payload = df.copy()
    payload["_fetched_at"] = datetime.now(timezone.utc).isoformat()
    payload.to_csv(CACHE, index=False, encoding="utf-8")
    global _DF, _MTIME
    _DF, _MTIME = df, CACHE.stat().st_mtime
    return df


def _df_from_api_csv(text: str) -> pd.DataFrame:
    if "Club" not in text[:200] and "," not in text.split("\n", 1)[0]:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text))
    if df.empty or "Club" not in df.columns:
        return pd.DataFrame()
    return df


def _df_from_ranking_html(html: str) -> pd.DataFrame:
    m = re.search(r"const eloData = (\[.*?\]);", html, re.S)
    if not m:
        return pd.DataFrame()
    rows: list[dict] = []
    for country, rank, slug, club, elo in _ROW_HTML.findall(m.group(1)):
        try:
            elo_f = float(elo)
            rank_i = int(rank)
        except ValueError:
            continue
        name = str(club).replace("\xa0", " ").strip()
        if not name:
            continue
        rows.append(
            {
                "Rank": rank_i,
                "Club": name,
                "Country": country,
                "Elo": elo_f,
                "Slug": slug,
            }
        )
    return pd.DataFrame(rows)


def _fetch_ranking_html(*, timeout: int = 20) -> pd.DataFrame:
    try:
        from modules.data_update.http_client import fetch_bytes

        raw = fetch_bytes(RANKING_URL, timeout=timeout)
    except Exception as exc:
        print(f"skip ClubElo sito: {exc}")
        return pd.DataFrame()
    df = _df_from_ranking_html(raw.decode("utf-8", "replace"))
    if df.empty:
        print("skip ClubElo sito: ranking non parsato")
    return df


def _fetch_api_csv(*, day: date | None = None, timeout: int = API_TIMEOUT_S) -> pd.DataFrame:
    day = day or date.today()
    url = API.format(day=day.isoformat())
    req = Request(url, headers={"User-Agent": UA})
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except (URLError, TimeoutError, OSError) as exc:
        print(f"skip ClubElo API: {exc}")
        return pd.DataFrame()
    text = raw.decode("utf-8", "replace")
    df = _df_from_api_csv(text)
    if df.empty:
        print("skip ClubElo API: risposta non CSV")
    return df


def fetch_clubelo(*, day: date | None = None, timeout: int = API_TIMEOUT_S) -> pd.DataFrame:
    if _cache_fresh():
        df = load_clubelo()
        if df is not None and not df.empty:
            print(f"ok ClubElo cache ({len(df)} club)")
            return df

    df = _fetch_ranking_html()
    source = "sito"
    if df.empty:
        df = _fetch_api_csv(day=day, timeout=timeout)
        source = "api"
    if df.empty:
        stale = load_clubelo()
        if stale is not None and not stale.empty:
            print(f"skip ClubElo: uso cache vecchia ({len(stale)} club)")
            return stale
        print("skip ClubElo: nessuna risposta")
        return stale if stale is not None else pd.DataFrame()

    _save_cache(df)
    print(f"ok ClubElo {len(df)} club ({source})")
    return df


def load_clubelo() -> pd.DataFrame:
    global _DF, _MTIME
    if not CACHE.exists():
        return pd.DataFrame()
    mtime = CACHE.stat().st_mtime
    if _DF is not None and _MTIME == mtime:
        return _DF
    df = pd.read_csv(CACHE, encoding="utf-8")
    _DF, _MTIME = df, mtime
    return df


def _elo_row(df: pd.DataFrame, idx: int) -> dict | None:
    row = df.iloc[int(idx)]
    try:
        elo = float(row["Elo"])
    except (TypeError, ValueError, KeyError):
        return None
    rank = row.get("Rank") if "Rank" in df.columns else None
    return {
        "club": str(row["Club"]),
        "elo": round(elo, 1),
        "country": str(row.get("Country") or ""),
        "rank": None if rank is None or pd.isna(rank) else int(rank),
    }


def _club_index(df: pd.DataFrame) -> dict[str, int]:
    from modules.data_update.cups import _norm_key

    keys: dict[str, int] = {}
    for i, club in enumerate(df["Club"].astype(str)):
        k = _norm_key(club)
        if k:
            keys.setdefault(k, i)
    if "Slug" not in df.columns:
        return keys
    for i, slug in enumerate(df["Slug"].astype(str)):
        spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", slug)
        for probe in (slug, spaced):
            k = _norm_key(probe)
            if k:
                keys.setdefault(k, i)
    return keys


def lookup_elo(name: str, df: pd.DataFrame | None = None) -> dict | None:
    from modules.data_update.cups import _norm_key, resolve_known_team

    df = load_clubelo() if df is None else df
    if df is None or df.empty or "Club" not in df.columns:
        return None
    raw = str(name or "").strip()
    if not raw:
        return None
    keys = _club_index(df)
    candidates = [raw, resolve_known_team(raw) or ""]
    tried: set[str] = set()
    for cand in candidates:
        k = _norm_key(cand)
        if not k or k in tried:
            continue
        tried.add(k)
        alias = _ELO_ALIASES.get(k)
        for probe in (k, alias or ""):
            if not probe:
                continue
            idx = keys.get(probe)
            if idx is not None:
                hit = _elo_row(df, idx)
                if hit:
                    return hit
        if k in _SKIP_FUZZY or len(k) < 6:
            continue
        hits = [i for club, i in keys.items() if k in club or club in k]
        if len(hits) == 1:
            hit = _elo_row(df, hits[0])
            if hit:
                return hit
    return None
