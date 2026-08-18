"""Contesto xG da Understat via soccerdata (solo analisi quadro)."""

from __future__ import annotations

from datetime import date
import difflib
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
TEAM_CACHE = PROCESSED / "understat_team_context.csv"

UNDERSTAT_LEAGUES = [
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

    q, h = _norm(query), _norm(hit)
    return flag(q) != flag(h)


def download_understat_context(*, seasons: list[int] | None = None) -> dict[str, Any]:
    seasons = seasons or [date.today().year - 1, date.today().year]
    try:
        import soccerdata as sd
    except Exception as exc:
        return {"ok": False, "n_teams": 0, "error": f"soccerdata non disponibile: {exc}"}

    try:
        us = sd.Understat(leagues=UNDERSTAT_LEAGUES, seasons=seasons)
        tm = us.read_team_match_stats().reset_index()
    except Exception as exc:
        return {"ok": False, "n_teams": 0, "error": str(exc)}

    if tm is None or tm.empty:
        return {"ok": True, "n_teams": 0, "error": "Understat vuoto"}

    rows: list[dict[str, Any]] = []
    for side in ("home", "away"):
        rows.append(
            tm[
                [
                    "league",
                    "season",
                    f"{side}_team",
                    f"{side}_xg",
                    f"{'away' if side == 'home' else 'home'}_xg",
                    f"{side}_goals",
                    f"{'away' if side == 'home' else 'home'}_goals",
                    "date",
                ]
            ].rename(
                columns={
                    f"{side}_team": "team",
                    f"{side}_xg": "xg_for",
                    f"{'away' if side == 'home' else 'home'}_xg": "xg_against",
                    f"{side}_goals": "g_for",
                    f"{'away' if side == 'home' else 'home'}_goals": "g_against",
                }
            )
        )
    long = pd.concat(rows, ignore_index=True)
    for c in ("xg_for", "xg_against", "g_for", "g_against"):
        long[c] = pd.to_numeric(long[c], errors="coerce")
    long = long.dropna(subset=["team"])
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    long = long.sort_values("date")
    agg = (
        long.groupby("team", as_index=False)
        .agg(
            n_matches=("team", "size"),
            xg_for=("xg_for", "mean"),
            xg_against=("xg_against", "mean"),
            g_for=("g_for", "mean"),
            g_against=("g_against", "mean"),
            last_match=("date", "max"),
        )
        .copy()
    )
    agg["xg_diff"] = agg["xg_for"] - agg["xg_against"]
    agg["team_norm"] = agg["team"].map(_norm)
    agg["fetched_at"] = pd.Timestamp.utcnow().isoformat()
    agg = agg.drop_duplicates(subset=["team_norm"], keep="last")
    TEAM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(TEAM_CACHE, index=False)
    return {"ok": True, "n_teams": int(len(agg)), "path": str(TEAM_CACHE), "seasons": seasons}


def load_understat_team_index() -> dict[str, dict[str, Any]]:
    if not TEAM_CACHE.exists():
        return {}
    df = pd.read_csv(TEAM_CACHE)
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        k = _norm(str(row.get("team") or row.get("team_norm") or ""))
        if k:
            out[k] = row.to_dict()
    return out


def lookup_understat_team(name: str, idx: dict[str, dict[str, Any]] | None = None) -> dict[str, Any] | None:
    k = _norm(name)
    if not k:
        return None
    idx = idx or load_understat_team_index()
    if k in idx:
        return idx[k]
    hit = difflib.get_close_matches(k, list(idx.keys()), n=1, cutoff=0.88)
    if hit:
        row = idx[hit[0]]
        if row and not _reserve_mismatch(name, hit[0]):
            return row
    return None
