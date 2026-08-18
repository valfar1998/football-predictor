"""Classifica Sofascore via soccerdata (forza attuale). Solo quadro, non EV/Kelly."""

from __future__ import annotations

import difflib
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
TEAM_CACHE = PROCESSED / "sofascore_team_context.csv"

SOFA_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
]


def _norm(name: str) -> str:
    from modules.data_update.cups import _norm_key

    return _norm_key(name or "")


def _reserve_mismatch(query: str, hit: str) -> bool:
    def flag(k: str) -> bool:
        t = f" {k} "
        return any(s in t for s in (" ii ", " iii ", " u21 ", " u19 ", " u23 ", " reserves ", " amateur "))

    return flag(_norm(query)) != flag(_norm(hit))


def download_sofascore_context(*, seasons: list[int] | None = None) -> dict[str, Any]:
    seasons = seasons or [date.today().year - 1, date.today().year]
    try:
        import soccerdata as sd
    except Exception as exc:
        return {"ok": False, "n_teams": 0, "error": f"soccerdata non disponibile: {exc}"}

    try:
        sofa = sd.Sofascore(leagues=SOFA_LEAGUES, seasons=seasons, headless=True)
        table = sofa.read_league_table()
    except Exception as exc:
        return {"ok": False, "n_teams": 0, "error": str(exc)}

    if table is None or len(table) == 0:
        return {"ok": True, "n_teams": 0, "error": "Sofascore vuoto"}

    df = table.reset_index()
    rename = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"team"}:
            rename[c] = "team"
        elif cl in {"league"}:
            rename[c] = "league"
        elif cl in {"season"}:
            rename[c] = "season"
        elif cl in {"mp", "p", "played"}:
            rename[c] = "mp"
        elif cl in {"pts", "points"}:
            rename[c] = "pts"
        elif cl in {"gf"}:
            rename[c] = "gf"
        elif cl in {"ga"}:
            rename[c] = "ga"
        elif cl in {"gd"}:
            rename[c] = "gd"
        elif cl in {"w"}:
            rename[c] = "w"
        elif cl in {"d"}:
            rename[c] = "d"
        elif cl in {"l"}:
            rename[c] = "l"
    out = df.rename(columns=rename)
    if "team" not in out.columns:
        return {"ok": False, "n_teams": 0, "error": "schema Sofascore inatteso"}
    for c in ("mp", "pts", "gf", "ga", "gd", "w", "d", "l"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "mp" in out.columns:
        n = out["mp"].replace(0, pd.NA)
        out["ppg"] = (out.get("pts") / n).round(3)
        out["gd_pg"] = (out.get("gd") / n).round(3)
    if "season" in out.columns:
        out = out.sort_values("season").groupby("team", as_index=False).tail(1)
    out["team_norm"] = out["team"].map(_norm)
    out = out.drop_duplicates(subset=["team_norm"], keep="last")
    keep = [c for c in ("team", "league", "season", "mp", "pts", "gf", "ga", "gd", "ppg", "gd_pg", "w", "d", "l", "team_norm") if c in out.columns]
    out = out[keep].copy()
    out["fetched_at"] = pd.Timestamp.utcnow().isoformat()
    TEAM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(TEAM_CACHE, index=False)
    return {"ok": True, "n_teams": int(len(out)), "path": str(TEAM_CACHE), "seasons": seasons}


def load_sofascore_team_index() -> dict[str, dict[str, Any]]:
    if not TEAM_CACHE.exists():
        return {}
    df = pd.read_csv(TEAM_CACHE)
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        k = _norm(str(row.get("team") or row.get("team_norm") or ""))
        if k:
            out[k] = row.to_dict()
    return out


def lookup_sofascore_team(name: str, idx: dict[str, dict[str, Any]] | None = None) -> dict[str, Any] | None:
    k = _norm(name)
    if not k:
        return None
    idx = idx or load_sofascore_team_index()
    if k in idx:
        return idx[k]
    hit = difflib.get_close_matches(k, list(idx.keys()), n=1, cutoff=0.88)
    if hit:
        row = idx[hit[0]]
        if row and not _reserve_mismatch(name, hit[0]):
            return row
    return None
