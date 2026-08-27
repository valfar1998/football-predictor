"""Contesto squadra da FBref via soccerdata (supporto al quadro, non EV/Kelly)."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from modules.data_update.sd_compat import quiet_soccerdata, season_codes

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
TEAM_CACHE = PROCESSED / "fbref_team_context.csv"
PLAYER_CACHE = PROCESSED / "fbref_player_contrib.csv"
MATCH_RATES_CACHE = PROCESSED / "fbref_match_side_rates.csv"

# Copertura reale di soccerdata.FBref senza login premium.
# Solo leghe club: Mondiali/Europei hanno schema diverso (niente colonna Club)
# e mescolati a Premier/La Liga fanno scattare i warning soccerdata.
FBREF_LEAGUES = [
    "Big 5 European Leagues Combined",
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
    with quiet_soccerdata():
        raw = fb.read_team_season_stats(stat_type=stat_type, opponent_stats=opponent)
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    return _flatten_cols(raw.reset_index())


def _per90(series: pd.Series, n90: pd.Series) -> pd.Series:
    n = _to_num(n90).replace(0, pd.NA)
    return _to_num(series) / n


def download_fbref_context(
    *,
    seasons: list[int] | None = None,
    match_logs: bool = False,
    on_progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Scarica statistiche squadra FBref utili al quadro analisi.

    ``match_logs=True`` scarica anche i team match logs (lento; cards/corners).
    Di default False: i rate FD (`side_rates`) bastano per la maggior parte delle leghe.
    """
    from modules.progress_report import emit

    seasons = season_codes(seasons)
    emit(on_progress, 0.02, "Avvio FBref…")
    try:
        import soccerdata as sd
    except Exception as exc:
        return {"ok": False, "n_teams": 0, "error": f"soccerdata non disponibile: {exc}"}

    try:
        emit(on_progress, 0.08, "Connessione FBref / standard…")
        with quiet_soccerdata():
            fb = sd.FBref(leagues=FBREF_LEAGUES, seasons=seasons, headless=True)
            team = fb.read_team_season_stats(stat_type="standard")
    except Exception as exc:
        return {"ok": False, "n_teams": 0, "error": str(exc)}

    if team is None or len(team) == 0:
        emit(on_progress, 1.0, "FBref vuoto")
        return {"ok": True, "n_teams": 0, "error": "FBref vuoto"}

    emit(on_progress, 0.25, "Standard ok · shooting…")
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
        emit(on_progress, 0.35, "Shooting…")
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
        emit(on_progress, 0.50, "Misc (cross/recuperi)…")
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
        emit(on_progress, 0.62, "Misc against…")
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
        emit(on_progress, 0.78, "Player contrib…")
        n_pl = _save_player_contrib(fb)
        extra_ok.append(f"players:{n_pl}")
    except Exception as exc:
        print(f"skip FBref players: {exc}")
        n_pl = 0
    match_info: dict[str, Any] = {"ok": False, "skipped": True}
    if match_logs:
        try:
            emit(on_progress, 0.85, "Match logs (lento)…")
            match_info = download_fbref_match_logs(fb=fb, seasons=seasons, last_n=12, on_progress=on_progress)
            if match_info.get("ok"):
                extra_ok.append(f"match_logs:{match_info.get('n_teams', 0)}")
        except Exception as exc:
            print(f"skip FBref match logs: {exc}")
            match_info = {"ok": False, "error": str(exc)}
    emit(on_progress, 1.0, f"OK · {len(out)} squadre")
    return {
        "ok": True,
        "n_teams": int(len(out)),
        "n_players": n_pl,
        "path": str(TEAM_CACHE),
        "seasons": seasons,
        "extra": extra_ok,
        "match_logs": match_info,
    }


def download_fbref_match_logs(
    *,
    fb=None,
    seasons: list[int] | None = None,
    last_n: int = 12,
    on_progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Match logs FBref (misc + schedule) → media cartellini/corner recenti per squadra."""
    from modules.progress_report import emit

    seasons = season_codes(seasons)
    emit(on_progress, 0.05, "Match logs: avvio…")
    try:
        import soccerdata as sd
    except Exception as exc:
        return {"ok": False, "n_teams": 0, "error": f"soccerdata non disponibile: {exc}"}

    if fb is None:
        try:
            emit(on_progress, 0.08, "Connessione FBref…")
            with quiet_soccerdata():
                fb = sd.FBref(leagues=FBREF_LEAGUES, seasons=seasons, headless=True)
        except Exception as exc:
            return {"ok": False, "n_teams": 0, "error": str(exc)}

    frames: list[pd.DataFrame] = []
    stats = ("misc", "schedule")
    for si, stat in enumerate(stats):
        try:
            emit(on_progress, 0.15 + 0.35 * si, f"Match logs «{stat}» (può richiedere minuti)…")
            with quiet_soccerdata():
                raw = fb.read_team_match_stats(stat_type=stat, force_cache=True)
            emit(on_progress, 0.30 + 0.35 * si, f"Match logs «{stat}» ricevuti")
            if raw is None or len(raw) == 0:
                continue
            part = _flatten_cols(raw.reset_index())
            part["_stat"] = stat
            frames.append(part)
        except Exception as exc:
            print(f"skip FBref match {stat}: {exc}")
            emit(on_progress, 0.30 + 0.35 * si, f"Skip {stat}: {exc}")

    if not frames:
        return {"ok": False, "n_teams": 0, "error": "match logs vuoti"}

    df = frames[0]
    for extra in frames[1:]:
        # merge on common keys if possible
        keys = [c for c in ("team", "date", "league", "season", "game") if c in df.columns and c in extra.columns]
        if not keys:
            # try Team / Date variants
            t1 = _first_col(df, ("team", "Team"))
            t2 = _first_col(extra, ("team", "Team"))
            d1 = _first_col(df, ("date", "Date"))
            d2 = _first_col(extra, ("date", "Date"))
            if t1 and t2 and d1 and d2:
                extra = extra.rename(columns={t2: t1, d2: d1})
                keys = [t1, d1]
        if keys:
            overlap = [c for c in extra.columns if c not in df.columns or c in keys]
            df = df.merge(extra[overlap], on=keys, how="outer", suffixes=("", "_y"))
        else:
            df = pd.concat([df, extra], ignore_index=True)

    tcol = _first_col(df, ("team", "Team"))
    dcol = _first_col(df, ("date", "Date"))
    if not tcol:
        return {"ok": False, "n_teams": 0, "error": "colonna team assente"}

    cards_c = None
    for alias in ("Performance_CrdY", "CrdY", "Cards_Yellow", "Yellow", "cards_y"):
        cards_c = _first_col(df, (alias,))
        if cards_c:
            break
    red_c = _first_col(df, ("Performance_CrdR", "CrdR", "Cards_Red", "Red"))
    corner_c = None
    for alias in ("Performance_CK", "CK", "Corners", "Corner Kicks", "corner_kicks"):
        corner_c = _first_col(df, (alias,))
        if corner_c:
            break

    work = pd.DataFrame({"team": df[tcol].astype(str)})
    if dcol:
        work["date"] = pd.to_datetime(df[dcol], errors="coerce")
    else:
        work["date"] = pd.NaT
    work["cards_y"] = _to_num(df[cards_c]) if cards_c else pd.NA
    work["cards_r"] = _to_num(df[red_c]) if red_c else pd.NA
    work["corners"] = _to_num(df[corner_c]) if corner_c else pd.NA
    work = work.dropna(subset=["team"])
    work["team_norm"] = work["team"].map(_norm)
    work = work.sort_values("date")
    work["rank"] = work.groupby("team_norm").cumcount(ascending=False)
    recent = work[work["rank"] < last_n]
    if recent.empty:
        recent = work
    agg = recent.groupby(["team_norm", "team"], as_index=False).agg(
        n=("team", "size"),
        cards_y_avg=("cards_y", "mean"),
        cards_r_avg=("cards_r", "mean"),
        corners_avg=("corners", "mean"),
    )
    agg = agg.drop_duplicates("team_norm", keep="first")
    agg["fetched_at"] = pd.Timestamp.utcnow().isoformat()
    MATCH_RATES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(MATCH_RATES_CACHE, index=False)

    # arricchisci anche team context se presente
    if TEAM_CACHE.exists() and not agg.empty:
        try:
            team = pd.read_csv(TEAM_CACHE)
            team["team_norm"] = team["team"].map(_norm) if "team_norm" not in team.columns else team["team_norm"]
            merge_cols = ["team_norm", "cards_y_avg", "cards_r_avg", "corners_avg", "n"]
            m = agg[merge_cols].rename(
                columns={
                    "cards_y_avg": "match_cards_y_avg",
                    "cards_r_avg": "match_cards_r_avg",
                    "corners_avg": "match_corners_avg",
                    "n": "match_n",
                }
            )
            team = team.drop(columns=[c for c in ("match_cards_y_avg", "match_cards_r_avg", "match_corners_avg", "match_n") if c in team.columns], errors="ignore")
            team = team.merge(m, on="team_norm", how="left")
            team.to_csv(TEAM_CACHE, index=False)
        except Exception as exc:
            print(f"skip merge match rates into team cache: {exc}")

    emit(on_progress, 1.0, f"Match logs OK · {len(agg)} squadre")
    return {
        "ok": True,
        "n_teams": int(len(agg)),
        "path": str(MATCH_RATES_CACHE),
        "has_cards": bool(cards_c),
        "has_corners": bool(corner_c),
    }


def load_fbref_match_side_index() -> dict[str, dict[str, Any]]:
    if not MATCH_RATES_CACHE.exists():
        return {}
    df = pd.read_csv(MATCH_RATES_CACHE)
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        key = _norm(str(row.get("team_norm") or row.get("team") or ""))
        if not key:
            continue
        out[key] = {
            "team": row.get("team"),
            "n": int(row["n"]) if pd.notna(row.get("n")) else 0,
            "cards_y_avg": float(row["cards_y_avg"]) if pd.notna(row.get("cards_y_avg")) else None,
            "cards_r_avg": float(row["cards_r_avg"]) if pd.notna(row.get("cards_r_avg")) else None,
            "corners_avg": float(row["corners_avg"]) if pd.notna(row.get("corners_avg")) else None,
            "source": "fbref_match",
        }
    return out


def lookup_fbref_match_side(team: str, idx: dict[str, dict[str, Any]] | None = None) -> dict[str, Any] | None:
    idx = idx if idx is not None else load_fbref_match_side_index()
    key = _norm(team)
    if key in idx:
        return idx[key]
    for k, v in idx.items():
        if key and (key in k or k in key) and min(len(k), len(key)) >= 5:
            return v
    return None


def _save_player_contrib(fb) -> int:
    with quiet_soccerdata():
        raw = fb.read_player_season_stats(stat_type="standard")
    if raw is None or len(raw) == 0:
        return 0
    df = _flatten_cols(raw.reset_index())
    tcol = _first_col(df, ("team",))
    pcol = _first_col(df, ("player", "Player"))
    if not tcol or not pcol:
        return 0
    min_c = _first_col(df, ("Playing_Time_Min", "Min", "Playing Time_Min"))
    # evita match ambigui tipo sottostringa corta "xg" su colonne sbagliate
    xg_c = None
    for alias in ("Expected_xG", "Expected_npxG", "Per 90 Minutes_xG", "xG"):
        cand = _first_col(df, (alias,))
        if cand and (str(cand).lower().endswith("xg") or "expected" in str(cand).lower()):
            xg_c = cand
            break
    if xg_c is None:
        for c in df.columns:
            cl = str(c).lower()
            if cl.endswith("_xg") or cl.endswith(" xg") or cl == "xg":
                xg_c = str(c)
                break
    xa_c = _first_col(df, ("Expected_xAG", "Expected_xA", "xAG", "xA"))
    gls_c = _first_col(df, ("Performance_Gls", "Gls", "Goals"))
    part = pd.DataFrame({"team": df[tcol].astype(str), "player": df[pcol].astype(str)})
    part["minutes"] = _to_num(df[min_c]) if min_c else 0
    part["xg"] = _to_num(df[xg_c]) if xg_c else 0.0
    part["xa"] = _to_num(df[xa_c]) if xa_c else 0.0
    part["gls"] = _to_num(df[gls_c]) if gls_c else 0.0
    part = part.dropna(subset=["team", "player"])
    # se xG assente (Big5 a volte vuoto su standard), usa gol come proxy share
    use = part["xg"].fillna(0)
    if float(use.sum() or 0) < 5:
        use = part["gls"].fillna(0)
        part["xg"] = use  # memorizza proxy nei campi xg per scorers
    part["xg_sum"] = use.groupby(part["team"]).transform("sum")
    part["share"] = (use / part["xg_sum"].replace(0, pd.NA)).fillna(0).clip(0, 0.40)
    part["impact"] = use + 0.35 * part["xa"].fillna(0)
    part["team_sum"] = part.groupby("team")["impact"].transform("sum")
    part["share_impact"] = (part["impact"] / part["team_sum"].replace(0, pd.NA)).fillna(0).clip(0, 0.35)
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
