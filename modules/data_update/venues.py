"""Stadio reale vs casa: cache dagli fixture che già hanno `venue`.

Niente scraping (Soccerway / Transfermarkt / Marca). football-data.co.uk
non pubblica lo stadio: usiamo football-data.org, API-Football e TheSportsDB
quando il campo c'è.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
VENUE_CACHE = PROCESSED / "home_venues.csv"

NEUTRAL_LEAGUE_HINTS = (
    "super cup",
    "supercoppa",
    "supercopa",
    "supercoupe",
    "community shield",
    "club world",
    "mundial de clubes",
    "fifa intercontinental",
    "uefa super",
    "recopa",
    "trophee des champions",
    "trophée des champions",
    "finalissima",
    "neutral venue",
    "campo neutro",
)


def _norm_text(s: Any) -> str:
    raw = str(s or "").strip().lower()
    for ch in ("stadium", "stadio", "arena", "park", "ground", "estadio", "estádio"):
        raw = raw.replace(ch, " ")
    return " ".join(raw.split())


def _truthy(v: Any) -> bool:
    if v is True:
        return True
    if v is False or v is None:
        return False
    if isinstance(v, float) and pd.isna(v):
        return False
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def venue_from_item(
    *,
    name: Any = None,
    city: Any = None,
    neutral: Any = None,
) -> dict[str, Any]:
    return {
        "venue": str(name or "").strip(),
        "venue_city": str(city or "").strip(),
        "venue_neutral": _truthy(neutral),
    }


def league_looks_neutral(league: str | None) -> bool:
    blob = f" {str(league or '').strip().lower()} "
    return any(h in blob for h in NEUTRAL_LEAGUE_HINTS)


def load_home_venues() -> dict[str, dict[str, Any]]:
    if not VENUE_CACHE.exists():
        return {}
    df = pd.read_csv(VENUE_CACHE)
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        team = str(row.get("home_team") or "").strip()
        if team:
            out[team] = row.to_dict()
    return out


def usual_home_venue(home: str, idx: dict[str, dict[str, Any]] | None = None) -> str:
    idx = idx if idx is not None else load_home_venues()
    row = idx.get(str(home or "").strip()) or {}
    return str(row.get("venue") or "").strip()


def update_home_venues(fixtures: pd.DataFrame | None) -> int:
    """Aggiorna lo stadio-casa più frequente (partite non neutre)."""
    if fixtures is None or fixtures.empty or "venue" not in fixtures.columns:
        return 0
    df = fixtures.copy()
    df["venue"] = df["venue"].fillna("").astype(str).str.strip()
    df = df[df["venue"].str.len() >= 3]
    if "venue_neutral" in df.columns:
        df = df[~df["venue_neutral"].map(_truthy)]
    if "league" in df.columns:
        df = df[~df["league"].astype(str).map(league_looks_neutral)]
    if df.empty:
        return 0

    counts = (
        df.groupby(["home_team", "venue"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["home_team", "n"], ascending=[True, False])
    )
    best = counts.drop_duplicates("home_team", keep="first")

    old = pd.DataFrame()
    if VENUE_CACHE.exists():
        old = pd.read_csv(VENUE_CACHE)
    if old.empty:
        merged = best[["home_team", "venue", "n"]]
    else:
        merged = pd.concat([old, best[["home_team", "venue", "n"]]], ignore_index=True)
        merged = (
            merged.groupby(["home_team", "venue"], dropna=False)["n"]
            .sum()
            .reset_index()
            .sort_values(["home_team", "n"], ascending=[True, False])
            .drop_duplicates("home_team", keep="first")
        )
    VENUE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(VENUE_CACHE, index=False)
    return int(len(merged))


def classify_venue(
    *,
    home: str,
    venue: str | None,
    league: str | None = None,
    city: str | None = None,
    explicit_neutral: Any = False,
    usual: str | None = None,
) -> dict[str, Any]:
    """Flag + penalità probabilità casa: neutro −2%, alternativo −1%."""
    venue_s = str(venue or "").strip()
    usual_s = str(usual if usual is not None else usual_home_venue(home)).strip()
    city_s = str(city or "").strip()
    notes: list[str] = []

    if _truthy(explicit_neutral) or league_looks_neutral(league):
        notes.append("competizione / flag da stadio neutro")
        return {
            "flag": "campo_neutro",
            "venue": venue_s,
            "venue_city": city_s,
            "usual": usual_s,
            "penalty_pct": -0.02,
            "notes": notes,
        }

    if not venue_s:
        return {
            "flag": "n/d",
            "venue": "",
            "venue_city": city_s,
            "usual": usual_s,
            "penalty_pct": 0.0,
            "notes": ["stadio assente (football-data.co.uk non lo pubblica)"],
        }

    if not usual_s:
        notes.append(f"venue {venue_s}: nessuno stadio-casa in cache, niente penalità")
        return {
            "flag": "casa_normale",
            "venue": venue_s,
            "venue_city": city_s,
            "usual": "",
            "penalty_pct": 0.0,
            "notes": notes,
        }

    ratio = difflib.SequenceMatcher(None, _norm_text(venue_s), _norm_text(usual_s)).ratio()
    if ratio >= 0.72 or _norm_text(venue_s) in _norm_text(usual_s) or _norm_text(usual_s) in _norm_text(venue_s):
        notes.append(f"stadio casa {usual_s}")
        return {
            "flag": "casa_normale",
            "venue": venue_s,
            "venue_city": city_s,
            "usual": usual_s,
            "penalty_pct": 0.0,
            "notes": notes,
        }

    if ratio <= 0.45:
        notes.append(f"venue {venue_s} ≠ stadio ufficiale {usual_s}")
        return {
            "flag": "campo_neutro",
            "venue": venue_s,
            "venue_city": city_s,
            "usual": usual_s,
            "penalty_pct": -0.02,
            "notes": notes,
        }

    notes.append(f"venue {venue_s} stadio alternativo vs {usual_s}")
    return {
        "flag": "stadio_alternativo",
        "venue": venue_s,
        "venue_city": city_s,
        "usual": usual_s,
        "penalty_pct": -0.01,
        "notes": notes,
    }
