"""StatsBomb open data: più stagioni + mapping nomi aggressivo. Solo quadro."""

from __future__ import annotations

import difflib
import re
import warnings
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
TEAM_CACHE = PROCESSED / "statsbomb_team_context.csv"

# Alias → nome canonico StatsBomb / football-data
SB_ALIASES: dict[str, str] = {
    "man utd": "manchester united",
    "manchester utd": "manchester united",
    "man united": "manchester united",
    "man city": "manchester city",
    "spurs": "tottenham hotspur",
    "tottenham": "tottenham hotspur",
    "wolves": "wolverhampton wanderers",
    "nottingham forest": "nottingham forest",
    "inter": "inter milan",
    "internazionale": "inter milan",
    "ac milan": "milan",
    "atletico madrid": "atlético madrid",
    "atletico": "atlético madrid",
    "athletic bilbao": "athletic club",
    "athletic club bilbao": "athletic club",
    "bayern": "bayern munich",
    "bayern munchen": "bayern munich",
    "psg": "paris saint germain",
    "paris sg": "paris saint germain",
    "paris saint-germain": "paris saint germain",
    "sporting cp": "sporting lisbon",
    "sporting lisboa": "sporting lisbon",
}


def _norm(name: str) -> str:
    from modules.data_update.cups import _norm_key

    return _norm_key(name or "")


def _alias_keys(name: str) -> list[str]:
    k = _norm(name)
    if not k:
        return []
    out = [k]
    # strip common tokens
    stripped = k
    for tok in (" fc", " cf", " afc", " sc", " united", " city", " club"):
        if stripped.endswith(tok) and len(stripped) > len(tok) + 2:
            stripped = stripped[: -len(tok)].strip()
            if stripped:
                out.append(stripped)
    if k in SB_ALIASES:
        out.append(_norm(SB_ALIASES[k]))
    for alias, canon in SB_ALIASES.items():
        if _norm(canon) == k or alias == k:
            out.append(_norm(alias))
            out.append(_norm(canon))
    # dedupe
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _reserve_mismatch(query: str, hit: str) -> bool:
    def flag(k: str) -> bool:
        t = f" {k} "
        return any(s in t for s in (" ii ", " iii ", " u21 ", " u19 ", " u23 ", " reserves ", " amateur "))

    return flag(_norm(query)) != flag(_norm(hit))


def _season_start(name: object) -> int:
    m = re.match(r"(\d{4})", str(name or ""))
    return int(m.group(1)) if m else 0


def _team_name(val: object) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if isinstance(val, dict):
        return str(val.get("home_team_name") or val.get("away_team_name") or val.get("team_name") or "").strip()
    return str(val).strip()


def download_statsbomb_context(
    *,
    min_season: int = 2015,
    seasons_per_comp: int = 3,
    on_progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Aggrega gol/punti dalle ultime N stagioni open data per competizione."""
    from modules.progress_report import emit

    emit(on_progress, 0.05, "StatsBomb: competizioni…")
    try:
        from statsbombpy import sb
    except Exception as exc:
        return {"ok": False, "n_teams": 0, "error": f"statsbombpy non disponibile: {exc}"}

    warnings.filterwarnings("ignore", message="Please be aware")
    warnings.filterwarnings("ignore", category=UserWarning, module="statsbombpy")
    try:
        comps = sb.competitions()
    except Exception as exc:
        return {"ok": False, "n_teams": 0, "error": str(exc)}
    if comps is None or comps.empty:
        return {"ok": True, "n_teams": 0, "error": "StatsBomb open data vuoto"}

    gender = comps.get("competition_gender", pd.Series(dtype=str)).fillna("male")
    youth = comps.get("competition_youth", pd.Series(dtype=bool)).fillna(False)
    intl = comps.get("competition_international", pd.Series(dtype=bool)).fillna(False)
    comps = comps[(gender == "male") & (~youth.astype(bool)) & (~intl.astype(bool))].copy()
    comps["_year"] = comps["season_name"].map(_season_start)
    comps = comps[comps["_year"] >= int(min_season)]
    if comps.empty:
        return {"ok": True, "n_teams": 0, "error": "nessuna competizione club recente"}

    selected = (
        comps.sort_values("_year")
        .groupby("competition_name", as_index=False)
        .tail(max(1, int(seasons_per_comp)))
    )
    parts: list[pd.DataFrame] = []
    errors: list[str] = []
    rows_sel = list(selected.iterrows())
    n_sel = max(1, len(rows_sel))
    for i, (_, row) in enumerate(rows_sel):
        emit(
            on_progress,
            0.15 + 0.7 * (i / n_sel),
            f"{row.get('competition_name')} {row.get('season_name')}",
        )
        try:
            matches = sb.matches(competition_id=int(row["competition_id"]), season_id=int(row["season_id"]))
        except Exception as exc:
            errors.append(f"{row.get('competition_name')} {row.get('season_name')}: {exc}")
            continue
        if matches is None or matches.empty:
            continue
        m = matches.copy()
        m["home_team"] = m["home_team"].map(_team_name)
        m["away_team"] = m["away_team"].map(_team_name)
        m["home_score"] = pd.to_numeric(m["home_score"], errors="coerce")
        m["away_score"] = pd.to_numeric(m["away_score"], errors="coerce")
        m = m.dropna(subset=["home_team", "away_team", "home_score", "away_score"])
        m["competition"] = str(row.get("competition_name") or "")
        m["season"] = str(row.get("season_name") or "")
        parts.append(m[["home_team", "away_team", "home_score", "away_score", "competition", "season"]])

    if not parts:
        # fallback: keep previous cache if download fails
        if TEAM_CACHE.exists():
            try:
                old = pd.read_csv(TEAM_CACHE)
                return {
                    "ok": True,
                    "n_teams": int(len(old)),
                    "n_matches": 0,
                    "n_competitions": 0,
                    "path": str(TEAM_CACHE),
                    "errors": errors or ["download fallito, cache precedente"],
                    "from_cache": True,
                }
            except Exception:
                pass
        return {"ok": False, "n_teams": 0, "error": "; ".join(errors) or "nessun match StatsBomb"}

    games = pd.concat(parts, ignore_index=True)
    rows: list[dict[str, Any]] = []
    for side, gf, ga in (
        ("home_team", "home_score", "away_score"),
        ("away_team", "away_score", "home_score"),
    ):
        chunk = games[[side, gf, ga, "competition", "season"]].rename(
            columns={side: "team", gf: "g_for", ga: "g_against"}
        )
        chunk["pts"] = 0
        chunk.loc[chunk["g_for"] > chunk["g_against"], "pts"] = 3
        chunk.loc[chunk["g_for"] == chunk["g_against"], "pts"] = 1
        rows.append(chunk)
    long = pd.concat(rows, ignore_index=True)
    long["team"] = long["team"].astype(str).str.strip()
    long = long[long["team"] != ""]
    agg = (
        long.groupby("team", as_index=False)
        .agg(
            n=("g_for", "size"),
            g_for=("g_for", "sum"),
            g_against=("g_against", "sum"),
            pts=("pts", "sum"),
            competition=("competition", "last"),
            season=("season", "last"),
        )
    )
    agg["gd"] = agg["g_for"] - agg["g_against"]
    agg["gf_pg"] = (agg["g_for"] / agg["n"]).round(3)
    agg["ga_pg"] = (agg["g_against"] / agg["n"]).round(3)
    agg["ppg"] = (agg["pts"] / agg["n"]).round(3)
    agg["gd_pg"] = (agg["gd"] / agg["n"]).round(3)
    agg["team_norm"] = agg["team"].map(_norm)
    agg = agg.drop_duplicates(subset=["team_norm"], keep="last")
    agg["fetched_at"] = pd.Timestamp.utcnow().isoformat()
    TEAM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(TEAM_CACHE, index=False)
    emit(on_progress, 1.0, f"OK · {len(agg)} squadre")
    return {
        "ok": True,
        "n_teams": int(len(agg)),
        "n_matches": int(len(games)),
        "n_competitions": int(selected["competition_name"].nunique()),
        "n_season_rows": int(len(selected)),
        "path": str(TEAM_CACHE),
        "errors": errors,
        "from_cache": False,
    }


def load_statsbomb_team_index() -> dict[str, dict[str, Any]]:
    if not TEAM_CACHE.exists():
        return {}
    df = pd.read_csv(TEAM_CACHE)
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        k = _norm(str(row.get("team") or row.get("team_norm") or ""))
        if k:
            out[k] = row.to_dict()
            for ak in _alias_keys(str(row.get("team") or "")):
                out.setdefault(ak, row.to_dict())
    return out


def lookup_statsbomb_team(name: str, idx: dict[str, dict[str, Any]] | None = None) -> dict[str, Any] | None:
    keys = _alias_keys(name)
    if not keys:
        return None
    idx = idx or load_statsbomb_team_index()
    for k in keys:
        if k in idx:
            return idx[k]
    # fuzzy più aggressivo
    pool = list(idx.keys())
    for k in keys:
        hit = difflib.get_close_matches(k, pool, n=1, cutoff=0.82)
        if hit:
            row = idx[hit[0]]
            if row and not _reserve_mismatch(name, str(row.get("team") or hit[0])):
                return row
    return None
