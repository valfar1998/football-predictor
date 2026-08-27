"""Contesto xG da Understat via soccerdata (solo analisi quadro)."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from modules.data_update.sd_compat import quiet_soccerdata, season_codes

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
TEAM_CACHE = PROCESSED / "understat_team_context.csv"
PLAYER_CACHE = PROCESSED / "understat_player_xg.csv"

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


def download_understat_context(
    *,
    seasons: list[int] | None = None,
    on_progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    from modules.progress_report import emit

    seasons = season_codes(seasons)
    emit(on_progress, 0.05, "Understat: avvio…")
    try:
        import soccerdata as sd
    except Exception as exc:
        return {"ok": False, "n_teams": 0, "error": f"soccerdata non disponibile: {exc}"}

    try:
        emit(on_progress, 0.15, "Team match stats…")
        with quiet_soccerdata():
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
    player_info: dict[str, Any] = {}
    try:
        emit(on_progress, 0.75, "Player xG…")
        player_info = download_understat_players(us=us, on_progress=on_progress)
    except Exception as exc:
        player_info = {"ok": False, "error": str(exc)}
        print(f"skip Understat players: {exc}")
    emit(on_progress, 1.0, f"OK · {len(agg)} squadre")
    return {
        "ok": True,
        "n_teams": int(len(agg)),
        "path": str(TEAM_CACHE),
        "seasons": seasons,
        "players": player_info,
    }


def download_understat_players(
    *,
    us=None,
    seasons: list[int] | None = None,
    on_progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """xG giocatore stagione (Big 5) per mercati marcatori."""
    from modules.progress_report import emit

    seasons = season_codes(seasons)
    emit(on_progress, 0.78, "Player season stats…")
    try:
        import soccerdata as sd
    except Exception as exc:
        return {"ok": False, "n_players": 0, "error": str(exc)}
    if us is None:
        try:
            with quiet_soccerdata():
                us = sd.Understat(leagues=UNDERSTAT_LEAGUES, seasons=seasons)
        except Exception as exc:
            return {"ok": False, "n_players": 0, "error": str(exc)}
    try:
        with quiet_soccerdata():
            raw = us.read_player_season_stats(force_cache=True)
    except Exception as exc:
        return {"ok": False, "n_players": 0, "error": str(exc)}
    if raw is None or len(raw) == 0:
        return {"ok": False, "n_players": 0, "error": "vuoto"}
    df = raw.reset_index() if hasattr(raw, "reset_index") else pd.DataFrame(raw)
    # colonne tipiche soccerdata Understat
    rename = {}
    low = {str(c).lower(): c for c in df.columns}
    for want, aliases in (
        ("player", ("player", "player_name", "name")),
        ("team", ("team", "team_title", "team_name")),
        ("xg", ("xg", "x_g", "npxg")),
        ("xa", ("xa", "x_a", "xa")),
        ("goals", ("goals", "g", "npg")),
        ("shots", ("shots", "sh")),
        ("time", ("time", "minutes", "min")),
    ):
        for a in aliases:
            if a in low:
                rename[low[a]] = want
                break
    part = df.rename(columns=rename)
    keep = [c for c in ("player", "team", "xg", "xa", "goals", "shots", "time", "league", "season") if c in part.columns]
    if "player" not in keep or "team" not in keep:
        return {"ok": False, "n_players": 0, "error": f"colonne: {list(part.columns)[:12]}"}
    out = part[keep].copy()
    for c in ("xg", "xa", "goals", "shots", "time"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["xg"] = out["xg"].fillna(0)
    out["team_sum"] = out.groupby("team")["xg"].transform("sum")
    out["share"] = (out["xg"] / out["team_sum"].replace(0, pd.NA)).fillna(0).clip(0, 0.45)
    out["team_norm"] = out["team"].map(_norm)
    out["player_norm"] = out["player"].astype(str).map(_norm)
    out["fetched_at"] = pd.Timestamp.utcnow().isoformat()
    PLAYER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PLAYER_CACHE, index=False)
    emit(on_progress, 0.95, f"Player xG: {len(out)}")
    return {"ok": True, "n_players": int(len(out)), "path": str(PLAYER_CACHE)}


def load_understat_player_index() -> dict[str, list[dict[str, Any]]]:
    if not PLAYER_CACHE.exists():
        return {}
    df = pd.read_csv(PLAYER_CACHE)
    out: dict[str, list[dict[str, Any]]] = {}
    for _, row in df.iterrows():
        team = _norm(str(row.get("team_norm") or row.get("team") or ""))
        player = str(row.get("player") or "").strip()
        if not team or not player:
            continue
        try:
            xg = float(row.get("xg") or 0)
            share = float(row.get("share") or 0)
        except (TypeError, ValueError):
            continue
        if xg < 0.8 and share < 0.05:
            continue
        out.setdefault(team, []).append(
            {"player": player, "xg": xg, "share": max(0.0, min(0.45, share)), "source": "understat"}
        )
    for team, rows in out.items():
        rows.sort(key=lambda r: -float(r["xg"]))
        out[team] = rows[:15]
    return out


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
