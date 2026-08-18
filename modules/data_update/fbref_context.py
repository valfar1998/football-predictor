"""Contesto squadra da FBref via soccerdata (supporto al quadro, non EV/Kelly)."""

from __future__ import annotations

from datetime import date
import difflib
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
TEAM_CACHE = PROCESSED / "fbref_team_context.csv"

# Copertura reale di soccerdata.FBref senza login premium.
FBREF_LEAGUES = [
    "Big 5 European Leagues Combined",
    "INT-World Cup",
    "INT-European Championship",
]


def _norm(name: str) -> str:
    from modules.data_update.cups import _norm_key

    return _norm_key(name or "")


def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    out = df.copy()
    cols: list[str] = []
    for c in out.columns:
        if not isinstance(c, tuple):
            cols.append(str(c))
            continue
        a, b = c
        a = str(a or "").strip()
        b = str(b or "").strip()
        cols.append(f"{a}_{b}".strip("_"))
    out.columns = cols
    return out


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def download_fbref_context(*, seasons: list[int] | None = None) -> dict[str, Any]:
    """Scarica statistiche squadra FBref utili al quadro analisi."""
    seasons = seasons or [date.today().year - 1, date.today().year]
    try:
        import soccerdata as sd
    except Exception as exc:
        return {"ok": False, "n_teams": 0, "error": f"soccerdata non disponibile: {exc}"}

    try:
        fb = sd.FBref(leagues=FBREF_LEAGUES, seasons=seasons, headless=True)
        team = fb.read_team_season_stats(stat_type="standard")
    except Exception as exc:
        return {"ok": False, "n_teams": 0, "error": str(exc)}

    if team is None or len(team) == 0:
        return {"ok": True, "n_teams": 0, "error": "FBref vuoto"}

    df = team.reset_index()
    df = _flatten_cols(df)

    # Colonne attese da soccerdata (in parte MultiIndex flatten).
    pick = {
        "Poss": "poss",
        "Performance_Gls": "gls",
        "Performance_Ast": "ast",
        "Performance_G-PK": "g_nonpen",
        "Performance_CrdY": "cards_y",
        "Performance_CrdR": "cards_r",
        "Per 90 Minutes_Gls": "gls_p90",
        "Per 90 Minutes_Ast": "ast_p90",
        "Per 90 Minutes_G+A": "ga_p90",
    }
    cols: dict[str, str] = {}
    for c in df.columns:
        cc = str(c)
        if cc in pick:
            tgt = pick[cc]
            if tgt:
                cols[cc] = tgt
        elif cc.lower() == "team":
            cols[cc] = "team"
        elif cc.lower() == "league":
            cols[cc] = "league"
        elif cc.lower() == "season":
            cols[cc] = "season"
        elif cc == "Playing Time_90s":
            cols[cc] = "n90"
        elif cc == "Age":
            cols[cc] = "age"

    out = df[[c for c in cols if c in df.columns]].rename(columns=cols).copy()
    if out.empty or "team" not in out.columns:
        return {"ok": False, "n_teams": 0, "error": "schema FBref inatteso"}

    for c in ("poss", "gls", "ast", "g_nonpen", "cards_y", "cards_r", "gls_p90", "ast_p90", "ga_p90", "n90", "age"):
        if c in out.columns:
            out[c] = _to_num(out[c])

    # Tiene l'ultima stagione disponibile per ogni team.
    if "season" in out.columns:
        out = out.sort_values("season").groupby("team", as_index=False).tail(1)

    out["team_norm"] = out["team"].map(_norm)
    out = out.drop_duplicates(subset=["team_norm"], keep="last")
    out["fetched_at"] = pd.Timestamp.utcnow().isoformat()

    TEAM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(TEAM_CACHE, index=False)
    return {"ok": True, "n_teams": int(len(out)), "path": str(TEAM_CACHE), "seasons": seasons}


def load_fbref_team_index() -> dict[str, dict[str, Any]]:
    if not TEAM_CACHE.exists():
        return {}
    df = pd.read_csv(TEAM_CACHE)
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        k = _norm(str(row.get("team") or row.get("team_norm") or ""))
        if not k:
            continue
        out[k] = row.to_dict()
    return out


def team_context(name: str) -> dict[str, Any] | None:
    return lookup_team_context(name)


def lookup_team_context(name: str, idx: dict[str, dict[str, Any]] | None = None) -> dict[str, Any] | None:
    k = _norm(name)
    if not k:
        return None
    idx = idx or load_fbref_team_index()
    exact = idx.get(k)
    if exact:
        return exact
    if not idx:
        return None
    keys = list(idx.keys())
    hit = difflib.get_close_matches(k, keys, n=1, cutoff=0.88)
    if hit:
        return idx.get(hit[0])
    return None
