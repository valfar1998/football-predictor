"""Elo esterno ClubElo (CSV pubblico). Non entra in EV/Kelly: solo quadro analisi."""

from __future__ import annotations

import io
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "clubelo.csv"
API = "http://api.clubelo.com/{day}"
UA = "Mozilla/5.0 (compatible; football-predictor/1.0; +local)"

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

_DF: pd.DataFrame | None = None
_MTIME: float | None = None


def fetch_clubelo(*, day: date | None = None, timeout: int = 12) -> pd.DataFrame:
    day = day or date.today()
    url = API.format(day=day.isoformat())
    req = Request(url, headers={"User-Agent": UA})
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except (URLError, TimeoutError, OSError) as exc:
        print(f"skip ClubElo: {exc}")
        return load_clubelo()
    text = raw.decode("utf-8", "replace")
    if "Club" not in text[:200] and "," not in text.split("\n", 1)[0]:
        print("skip ClubElo: risposta non CSV")
        return load_clubelo()
    df = pd.read_csv(io.StringIO(text))
    if df.empty or "Club" not in df.columns:
        print("skip ClubElo: CSV vuoto")
        return load_clubelo()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    payload = df.copy()
    payload["_fetched_at"] = datetime.now(timezone.utc).isoformat()
    payload.to_csv(CACHE, index=False)
    global _DF, _MTIME
    _DF, _MTIME = df, CACHE.stat().st_mtime
    print(f"ok ClubElo {len(df)} club")
    return df


def load_clubelo() -> pd.DataFrame:
    global _DF, _MTIME
    if not CACHE.exists():
        return pd.DataFrame()
    mtime = CACHE.stat().st_mtime
    if _DF is not None and _MTIME == mtime:
        return _DF
    df = pd.read_csv(CACHE)
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


def lookup_elo(name: str, df: pd.DataFrame | None = None) -> dict | None:
    from modules.data_update.cups import _norm_key, resolve_known_team

    df = load_clubelo() if df is None else df
    if df is None or df.empty or "Club" not in df.columns:
        return None
    raw = str(name or "").strip()
    if not raw:
        return None
    clubs = df["Club"].astype(str)
    keys = {_norm_key(c): i for i, c in enumerate(clubs)}
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
