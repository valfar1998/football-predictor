"""WhoScored: assenze confermate (preview) e stile se presente nella pagina.

soccerdata.WhoScored legge i preview Selenium, non Wyscout.
Limite: solo prossime partite Big 5 (lento). Non entra in EV/Kelly.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from lxml import html

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MISS_CACHE = PROCESSED / "whoscored_missing.csv"
STYLE_CACHE = PROCESSED / "whoscored_style.csv"

WS_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
]
MAX_PREVIEWS = 18


def _norm(name: str) -> str:
    from modules.data_update.cups import _norm_key

    return _norm_key(name or "")


def _texts(nodes) -> list[str]:
    out = []
    for n in nodes:
        t = " ".join(str(n.text_content() or "").split())
        if t and t.lower() not in {"strengths", "weaknesses", "style of play"}:
            out.append(t)
    return out


def _parse_preview_style(path: Path, home: str, away: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        tree = html.parse(str(path))
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for team, side in ((home, "home"), (away, "away")):
        strengths = _texts(tree.xpath(f"//div[@id='missing-players']/..//*[contains(@class,'strengths')]//td"))
        # fallback generic
        if not strengths:
            strengths = _texts(tree.xpath("//*[contains(@class,'strengths')]//li | //*[contains(@class,'strengths')]//td"))
        weak = _texts(tree.xpath("//*[contains(@class,'weaknesses')]//li | //*[contains(@class,'weaknesses')]//td"))
        style = _texts(tree.xpath("//*[contains(@class,'styleofplay') or contains(@class,'style-of-play')]//li | //*[contains(@class,'styleofplay')]//td"))
        if strengths or weak or style:
            rows.append(
                {
                    "team": team,
                    "side": side,
                    "strengths": " | ".join(strengths[:6]),
                    "weaknesses": " | ".join(weak[:6]),
                    "style": " | ".join(style[:8]),
                    "team_norm": _norm(team),
                }
            )
    return rows


def download_whoscored_context(*, seasons: list[int] | None = None) -> dict[str, Any]:
    seasons = seasons or [date.today().year - 1, date.today().year]
    try:
        import soccerdata as sd
        from soccerdata._config import DATA_DIR
    except Exception as exc:
        return {"ok": False, "n_missing": 0, "error": f"soccerdata non disponibile: {exc}"}

    try:
        ws = sd.WhoScored(leagues=WS_LEAGUES, seasons=seasons, headless=True)
        sched = ws.read_schedule().reset_index()
    except Exception as exc:
        return {"ok": False, "n_missing": 0, "error": str(exc)}

    if sched is None or sched.empty:
        return {"ok": True, "n_missing": 0, "error": "WhoScored schedule vuoto"}

    date_col = "date" if "date" in sched.columns else ("Date" if "Date" in sched.columns else None)
    id_col = "game_id" if "game_id" in sched.columns else None
    home_col = "home_team" if "home_team" in sched.columns else "home"
    away_col = "away_team" if "away_team" in sched.columns else "away"
    if not date_col or not id_col or home_col not in sched.columns:
        return {"ok": False, "n_missing": 0, "error": f"schema schedule inatteso: {list(sched.columns)[:12]}"}

    sched[date_col] = pd.to_datetime(sched[date_col], errors="coerce")
    today = pd.Timestamp.now().normalize()
    window = sched[(sched[date_col] >= today) & (sched[date_col] <= today + pd.Timedelta(days=10))]
    window = window.sort_values(date_col).head(MAX_PREVIEWS)
    ids = [int(x) for x in window[id_col].dropna().tolist()]
    if not ids:
        return {"ok": True, "n_missing": 0, "n_games": 0, "error": "nessun preview in finestra 10 giorni"}

    try:
        missing = ws.read_missing_players(match_id=ids)
    except Exception as exc:
        return {"ok": False, "n_missing": 0, "error": str(exc), "n_games": len(ids)}

    if missing is None or (hasattr(missing, "empty") and missing.empty):
        miss_df = pd.DataFrame()
    else:
        miss_df = missing.reset_index() if hasattr(missing, "reset_index") else pd.DataFrame(missing)

    if not miss_df.empty:
        miss_df["team_norm"] = miss_df.get("team", pd.Series(dtype=str)).astype(str).map(_norm)
        miss_df["player_norm"] = miss_df.get("player", pd.Series(dtype=str)).astype(str).map(_norm)
        # allinea date/squadre dal schedule
        meta = window[[id_col, date_col, home_col, away_col]].copy()
        meta = meta.rename(columns={id_col: "game_id", date_col: "date", home_col: "home_team", away_col: "away_team"})
        if "game_id" in miss_df.columns:
            miss_df = miss_df.merge(meta, on="game_id", how="left")
        miss_df["fetched_at"] = pd.Timestamp.utcnow().isoformat()
        PROCESSED.mkdir(parents=True, exist_ok=True)
        miss_df.to_csv(MISS_CACHE, index=False)

    styles: list[dict[str, Any]] = []
    for _, g in window.iterrows():
        gid = g[id_col]
        lg = str(g.get("league") or "")
        seas = str(g.get("season") or "")
        path = DATA_DIR / "WhoScored" / "previews" / f"{lg}_{seas}" / f"{gid}.html"
        styles.extend(_parse_preview_style(path, str(g[home_col]), str(g[away_col])))
    if styles:
        st = pd.DataFrame(styles).drop_duplicates(subset=["team_norm"], keep="last")
        st["fetched_at"] = pd.Timestamp.utcnow().isoformat()
        st.to_csv(STYLE_CACHE, index=False)

    return {
        "ok": True,
        "n_missing": 0 if miss_df.empty else int(len(miss_df)),
        "n_games": len(ids),
        "n_style": len(styles),
        "path": str(MISS_CACHE),
        "seasons": seasons,
    }


def lookup_whoscored_match(home: str, away: str, kickoff=None) -> dict[str, Any]:
    empty = {"home": [], "away": [], "style_home": None, "style_away": None}
    if not MISS_CACHE.exists() and not STYLE_CACHE.exists():
        return empty
    hn, an = _norm(home), _norm(away)
    out = dict(empty)
    if MISS_CACHE.exists():
        df = pd.read_csv(MISS_CACHE)
        df["team_norm"] = df.get("team_norm", df.get("team", "")).astype(str).map(_norm)
        day = None
        if kickoff is not None:
            day = pd.to_datetime(kickoff, errors="coerce")
            if pd.notna(day) and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df[df["date"].dt.normalize() == pd.Timestamp(day).normalize()]
        hmask = df["team_norm"] == hn
        amask = df["team_norm"] == an
        if "home_team" in df.columns:
            hmask = hmask | (df["home_team"].astype(str).map(_norm) == hn)
            amask = amask | (df["away_team"].astype(str).map(_norm) == an)
        out["home"] = df[hmask].to_dict("records")
        out["away"] = df[amask].to_dict("records")
    if STYLE_CACHE.exists():
        st = pd.read_csv(STYLE_CACHE)
        st["team_norm"] = st["team_norm"].astype(str)
        hr = st[st["team_norm"] == hn]
        ar = st[st["team_norm"] == an]
        if not hr.empty:
            out["style_home"] = hr.iloc[-1].to_dict()
        if not ar.empty:
            out["style_away"] = ar.iloc[-1].to_dict()
    return out
