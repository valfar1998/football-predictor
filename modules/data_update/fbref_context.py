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
PLAYER_CACHE = PROCESSED / "fbref_player_contrib.csv"

# Copertura reale di soccerdata.FBref senza login premium.
FBREF_LEAGUES = [
    "Big 5 European Leagues Combined",
    "INT-World Cup",
    "INT-European Championship",
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


def _first_col(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    low = {str(c).lower().replace(" ", "_"): str(c) for c in df.columns}
    for a in aliases:
        key = a.lower().replace(" ", "_")
        if key in low:
            return low[key]
        for k, orig in low.items():
            if key in k:
                return orig
    return None


def _stat_frame(fb, stat_type: str, *, opponent: bool = False) -> pd.DataFrame:
    raw = fb.read_team_season_stats(stat_type=stat_type, opponent_stats=opponent)
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    return _flatten_cols(raw.reset_index())


def _per90(series: pd.Series, n90: pd.Series) -> pd.Series:
    n = _to_num(n90).replace(0, pd.NA)
    return _to_num(series) / n


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

    extra_ok: list[str] = []
    try:
        shoot = _stat_frame(fb, "shooting")
        if not shoot.empty:
            dist_c = _first_col(shoot, ("Average_Shot_Distance", "Dist", "Standard_Dist"))
            sh_c = _first_col(shoot, ("Standard_Sh", "Sh"))
            tcol = _first_col(shoot, ("team",))
            if tcol:
                part = shoot[[tcol] + [c for c in (dist_c, sh_c) if c]].copy().rename(columns={tcol: "team"})
                if dist_c:
                    part["shot_dist"] = _to_num(part[dist_c])
                if sh_c:
                    part["shots"] = _to_num(part[sh_c])
                keep = [c for c in ("team", "shot_dist", "shots") if c in part.columns]
                out = out.merge(part[keep].drop_duplicates("team"), on="team", how="left")
                extra_ok.append("shooting")
    except Exception as exc:
        print(f"skip FBref shooting: {exc}")

    try:
        misc = _stat_frame(fb, "misc")
        if not misc.empty:
            tcol = _first_col(misc, ("team",))
            crs_c = _first_col(misc, ("Performance_Crs", "Crs", "Crosses"))
            rec_c = _first_col(misc, ("Performance_Recov", "Recov", "Recoveries"))
            if tcol:
                part = misc[[tcol]].copy().rename(columns={tcol: "team"})
                if crs_c:
                    part["crosses"] = _to_num(misc[crs_c].values)
                if rec_c:
                    part["recov"] = _to_num(misc[rec_c].values)
                if "n90" in out.columns:
                    part = part.merge(out[["team", "n90"]], on="team", how="left")
                    if "crosses" in part.columns:
                        part["crosses_p90"] = _per90(part["crosses"], part["n90"])
                    if "recov" in part.columns:
                        part["recov_p90"] = _per90(part["recov"], part["n90"])
                keep = [c for c in ("team", "crosses_p90", "recov_p90") if c in part.columns]
                out = out.merge(part[keep].drop_duplicates("team"), on="team", how="left")
                extra_ok.append("misc")
    except Exception as exc:
        print(f"skip FBref misc: {exc}")

    try:
        opp = _stat_frame(fb, "misc", opponent=True)
        if not opp.empty:
            tcol = _first_col(opp, ("team",))
            crs_c = _first_col(opp, ("Performance_Crs", "Crs", "Crosses"))
            if tcol and crs_c:
                part = opp[[tcol, crs_c]].copy().rename(columns={tcol: "team", crs_c: "crosses_conc"})
                part["crosses_conc"] = _to_num(part["crosses_conc"])
                if "n90" in out.columns:
                    part = part.merge(out[["team", "n90"]], on="team", how="left")
                    part["crosses_conc_p90"] = _per90(part["crosses_conc"], part["n90"])
                keep = [c for c in ("team", "crosses_conc_p90") if c in part.columns]
                out = out.merge(part[keep].drop_duplicates("team"), on="team", how="left")
                extra_ok.append("misc_against")
    except Exception as exc:
        print(f"skip FBref misc against: {exc}")

    # Tiene l'ultima stagione disponibile per ogni team.
    if "season" in out.columns:
        out = out.sort_values("season").groupby("team", as_index=False).tail(1)

    out["team_norm"] = out["team"].map(_norm)
    out = out.drop_duplicates(subset=["team_norm"], keep="last")
    out["fetched_at"] = pd.Timestamp.utcnow().isoformat()

    TEAM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(TEAM_CACHE, index=False)
    try:
        n_pl = _save_player_contrib(fb)
        extra_ok.append(f"players:{n_pl}")
    except Exception as exc:
        print(f"skip FBref players: {exc}")
        n_pl = 0
    return {
        "ok": True,
        "n_teams": int(len(out)),
        "n_players": n_pl,
        "path": str(TEAM_CACHE),
        "seasons": seasons,
        "extra": extra_ok,
    }


def _save_player_contrib(fb) -> int:
    raw = fb.read_player_season_stats(stat_type="standard")
    if raw is None or len(raw) == 0:
        return 0
    df = _flatten_cols(raw.reset_index())
    tcol = _first_col(df, ("team",))
    pcol = _first_col(df, ("player", "Player"))
    if not tcol or not pcol:
        return 0
    min_c = _first_col(df, ("Playing_Time_Min", "Min"))
    xg_c = _first_col(df, ("Expected_xG", "xG"))
    xa_c = _first_col(df, ("Expected_xAG", "Expected_xA", "xAG", "xA"))
    part = pd.DataFrame({"team": df[tcol].astype(str), "player": df[pcol].astype(str)})
    part["minutes"] = _to_num(df[min_c]) if min_c else 0
    part["xg"] = _to_num(df[xg_c]) if xg_c else 0
    part["xa"] = _to_num(df[xa_c]) if xa_c else 0
    part["impact"] = part["xg"].fillna(0) + part["xa"].fillna(0) + part["minutes"].fillna(0) / 900.0
    part = part.dropna(subset=["team", "player"])
    part["team_sum"] = part.groupby("team")["impact"].transform("sum")
    part["share"] = (part["impact"] / part["team_sum"].replace(0, pd.NA)).fillna(0).clip(0, 0.35)
    part["team_norm"] = part["team"].map(_norm)
    part["player_norm"] = part["player"].map(_norm)
    part["fetched_at"] = pd.Timestamp.utcnow().isoformat()
    PLAYER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    part.to_csv(PLAYER_CACHE, index=False)
    return int(len(part))


def load_player_contrib_index() -> dict[tuple[str, str], float]:
    if not PLAYER_CACHE.exists():
        return {}
    df = pd.read_csv(PLAYER_CACHE)
    out: dict[tuple[str, str], float] = {}
    for _, row in df.iterrows():
        tk = _norm(str(row.get("team_norm") or row.get("team") or ""))
        pk = _norm(str(row.get("player_norm") or row.get("player") or ""))
        if tk and pk:
            try:
                out[(tk, pk)] = float(row.get("share") or 0)
            except (TypeError, ValueError):
                continue
    return out


def player_share(team: str, player: str, idx: dict[tuple[str, str], float] | None = None) -> float:
    idx = idx or load_player_contrib_index()
    tk, pk = _norm(team), _norm(player)
    if (tk, pk) in idx:
        return idx[(tk, pk)]
    for (t, p), v in idx.items():
        if t == tk and (p in pk or pk in p) and min(len(p), len(pk)) >= 5:
            return v
    return 0.0


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
        row = idx.get(hit[0])
        team = str((row or {}).get("team") or (row or {}).get("team_norm") or hit[0])
        if row and not _reserve_mismatch(name, team):
            return row
    return None
