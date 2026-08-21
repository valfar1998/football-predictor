"""Contesto FotMob via API web non ufficiale (`/api/data`). Solo quadro, non EV/Kelly.

Niente dipendenza sportly. Cache locale: classifica Big 5+ e indice partite 7 giorni.
Lineup/xG matchDetails solo on-demand (singola partita), non in bulk sul calendario.
"""

from __future__ import annotations

import difflib
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
TEAM_CACHE = PROCESSED / "fotmob_team_context.csv"
MATCH_CACHE = PROCESSED / "fotmob_matches.json"
BASE = "https://www.fotmob.com/api/data"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

# ID ufficiali FotMob (stabili da anni).
FOTMOB_LEAGUES: dict[str, int] = {
    "Premier League": 47,
    "Championship": 48,
    "La Liga": 87,
    "Bundesliga": 54,
    "Serie A": 55,
    "Ligue 1": 53,
    "Eredivisie": 57,
    "Champions League": 42,
    "Europa League": 73,
    "MLS": 130,
    "Brasileirao": 268,
}


def _norm(name: str) -> str:
    from modules.data_update.cups import _norm_key

    return _norm_key(name or "")


def _reserve_mismatch(query: str, hit: str) -> bool:
    def flag(k: str) -> bool:
        t = f" {k} "
        return any(s in t for s in (" ii ", " iii ", " u21 ", " u19 ", " u23 ", " reserves ", " amateur "))

    return flag(_norm(query)) != flag(_norm(hit))


def _get_json(path: str, params: dict | None = None, timeout: int = 20) -> dict:
    q = ("?" + urlencode({k: v for k, v in (params or {}).items() if v is not None})) if params else ""
    url = f"{BASE}/{path.lstrip('/')}{q}"
    req = Request(
        url,
        headers={
            "User-Agent": UA,
            "Accept": "application/json",
            "Referer": "https://www.fotmob.com/",
            "Origin": "https://www.fotmob.com",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _table_rows(league_payload: dict) -> list[dict]:
    tables = league_payload.get("table") or []
    if not tables:
        return []
    data = (tables[0] or {}).get("data") or {}
    tab = data.get("table") or {}
    if isinstance(tab, dict):
        rows = tab.get("all") or []
    elif isinstance(tab, list):
        rows = tab
    else:
        rows = []
    return rows if isinstance(rows, list) else []


def download_fotmob_context(*, days: int = 7, league_ids: dict[str, int] | None = None) -> dict[str, Any]:
    """Scarica classifiche + indice partite. Nessuna chiave API."""
    league_ids = league_ids or FOTMOB_LEAGUES
    errors: list[str] = []
    team_rows: list[dict[str, Any]] = []

    for name, lid in league_ids.items():
        try:
            payload = _get_json("leagues", {"id": lid})
            for row in _table_rows(payload):
                played = int(row.get("played") or 0)
                pts = float(row.get("pts") or 0)
                gd = float(row.get("goalConDiff") or 0)
                team_rows.append(
                    {
                        "team": row.get("name") or row.get("shortName"),
                        "team_id": row.get("id"),
                        "league": name,
                        "league_id": lid,
                        "played": played,
                        "pts": pts,
                        "ppg": round(pts / played, 3) if played else 0.0,
                        "gd": gd,
                        "gd_pg": round(gd / played, 3) if played else 0.0,
                        "w": row.get("wins"),
                        "d": row.get("draws"),
                        "l": row.get("losses"),
                        "scores": row.get("scoresStr"),
                    }
                )
            time.sleep(0.35)
        except (HTTPError, URLError, TimeoutError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"league {name}: {exc}")

    team_df = pd.DataFrame(team_rows)
    if not team_df.empty:
        team_df["team_norm"] = team_df["team"].map(_norm)
        team_df["fetched_at"] = datetime.now(timezone.utc).isoformat()
        team_df = team_df.drop_duplicates(subset=["team_norm"], keep="first")
        TEAM_CACHE.parent.mkdir(parents=True, exist_ok=True)
        team_df.to_csv(TEAM_CACHE, index=False)

    matches: list[dict[str, Any]] = []
    today = date.today()
    for i in range(max(1, int(days))):
        day = today + timedelta(days=i)
        key = day.strftime("%Y%m%d")
        try:
            payload = _get_json("matches", {"date": key})
            for league in payload.get("leagues") or []:
                lg_name = league.get("name") or ""
                lg_id = league.get("id")
                for m in league.get("matches") or []:
                    home = m.get("home") or {}
                    away = m.get("away") or {}
                    status = m.get("status") or {}
                    utc = str(status.get("utcTime") or "")[:10]
                    matches.append(
                        {
                            "match_id": m.get("id"),
                            "date": utc or day.isoformat(),
                            "league": lg_name,
                            "league_id": lg_id,
                            "home": home.get("name") or home.get("longName"),
                            "away": away.get("name") or away.get("longName"),
                            "home_id": home.get("id"),
                            "away_id": away.get("id"),
                            "started": bool(status.get("started")),
                            "finished": bool(status.get("finished")),
                            "score": status.get("scoreStr"),
                        }
                    )
            time.sleep(0.25)
        except (HTTPError, URLError, TimeoutError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"matches {key}: {exc}")

    MATCH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    MATCH_CACHE.write_text(
        json.dumps(
            {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "n": len(matches),
                "matches": matches,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "ok": not errors or bool(team_rows) or bool(matches),
        "n_teams": int(len(team_df)) if not team_df.empty else 0,
        "n_matches": len(matches),
        "path": str(TEAM_CACHE),
        "matches_path": str(MATCH_CACHE),
        "errors": errors,
        "note": "API non ufficiale /api/data — solo quadro",
    }


def load_fotmob_team_index() -> dict[str, dict[str, Any]]:
    if not TEAM_CACHE.exists():
        return {}
    df = pd.read_csv(TEAM_CACHE)
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        k = _norm(str(row.get("team") or row.get("team_norm") or ""))
        if k:
            out[k] = row.to_dict()
    return out


def lookup_fotmob_team(name: str, idx: dict[str, dict[str, Any]] | None = None) -> dict[str, Any] | None:
    k = _norm(name)
    if not k:
        return None
    idx = idx or load_fotmob_team_index()
    if k in idx:
        return idx[k]
    hit = difflib.get_close_matches(k, list(idx.keys()), n=1, cutoff=0.88)
    if hit:
        row = idx[hit[0]]
        if row and not _reserve_mismatch(name, str(row.get("team") or hit[0])):
            return row
    return None


def load_fotmob_matches() -> list[dict[str, Any]]:
    if not MATCH_CACHE.exists():
        return []
    try:
        data = json.loads(MATCH_CACHE.read_text(encoding="utf-8"))
        return list(data.get("matches") or [])
    except Exception:
        return []


def lookup_fotmob_match(
    home: str,
    away: str,
    kickoff_date: str | None = None,
    matches: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    matches = matches if matches is not None else load_fotmob_matches()
    if not matches:
        return None
    hk, ak = _norm(home), _norm(away)
    day = (kickoff_date or "")[:10]
    best = None
    best_score = 0.0
    for m in matches:
        mh, ma = _norm(str(m.get("home") or "")), _norm(str(m.get("away") or ""))
        if not mh or not ma:
            continue
        if day and str(m.get("date") or "")[:10] not in {day, ""}:
            # tollera ±1 giorno timezone
            try:
                d0 = date.fromisoformat(day)
                d1 = date.fromisoformat(str(m.get("date") or "")[:10])
                if abs((d0 - d1).days) > 1:
                    continue
            except ValueError:
                continue
        s_h = difflib.SequenceMatcher(None, hk, mh).ratio()
        s_a = difflib.SequenceMatcher(None, ak, ma).ratio()
        score = 0.5 * (s_h + s_a)
        if score > best_score and s_h >= 0.72 and s_a >= 0.72:
            if _reserve_mismatch(home, str(m.get("home") or "")) or _reserve_mismatch(away, str(m.get("away") or "")):
                continue
            best_score = score
            best = dict(m)
            best["match_score"] = round(score, 3)
    return best


def fetch_match_details(match_id: int | str) -> dict[str, Any] | None:
    """Dettaglio on-demand: xG, lineup available, momentum presente. Non usare in loop massivo."""
    try:
        md = _get_json("matchDetails", {"matchId": match_id})
    except (HTTPError, URLError, TimeoutError, ValueError):
        return None
    content = md.get("content") or {}
    xg_h = xg_a = None
    poss_h = poss_a = None
    shots_h = shots_a = None
    periods = ((content.get("stats") or {}).get("Periods") or {}).get("All") or {}
    for block in periods.get("stats") or []:
        for row in block.get("stats") or []:
            key = str(row.get("key") or "")
            stats = row.get("stats") or [None, None]
            if key == "expected_goals" and stats[0] not in (None, "") and stats[1] not in (None, ""):
                try:
                    xg_h, xg_a = float(stats[0]), float(stats[1])
                except (TypeError, ValueError):
                    pass
            if key == "BallPossesion":
                try:
                    poss_h, poss_a = float(stats[0]), float(stats[1])
                except (TypeError, ValueError):
                    pass
            if key == "total_shots":
                try:
                    shots_h, shots_a = float(stats[0]), float(stats[1])
                except (TypeError, ValueError):
                    pass
    lineup = content.get("lineup") or {}
    has_lineup = bool(lineup.get("homeTeam") and lineup.get("awayTeam"))
    mom = content.get("momentum") or {}
    return {
        "match_id": match_id,
        "xg_home": xg_h,
        "xg_away": xg_a,
        "poss_home": poss_h,
        "poss_away": poss_a,
        "shots_home": shots_h,
        "shots_away": shots_a,
        "has_lineup": has_lineup,
        "has_momentum": bool(mom.get("main")),
        "has_shotmap": bool(content.get("shotmap")),
    }


DETAILS_CACHE = PROCESSED / "fotmob_details_cache.json"


def _load_details_cache() -> dict[str, Any]:
    if not DETAILS_CACHE.exists():
        return {}
    try:
        return json.loads(DETAILS_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_details_cache(data: dict[str, Any]) -> None:
    DETAILS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    DETAILS_CACHE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def enrich_top_picks_fotmob(
    rows: list[dict[str, Any]],
    *,
    min_score: int = 7,
    max_n: int = 12,
    max_age_hours: float = 6.0,
) -> dict[str, Any]:
    """Scarica matchDetails solo per top picks (voto≥min_score). Cache 6h."""
    cache = _load_details_cache()
    now = datetime.now(timezone.utc)
    ranked = sorted(
        (
            r
            for r in rows
            if (r.get("score_unified") or r.get("score") or 0) >= min_score
            and ((r.get("prediction") or {}).get("fotmob_context") or {}).get("match", {}).get("match_id")
        ),
        key=lambda r: float(r.get("score_unified") or r.get("score") or 0),
        reverse=True,
    )[:max_n]

    enriched = 0
    errors: list[str] = []
    for r in ranked:
        mid = str(
            (((r.get("prediction") or {}).get("fotmob_context") or {}).get("match") or {}).get("match_id")
        )
        if not mid:
            continue
        entry = cache.get(mid) or {}
        fresh = False
        try:
            ts = str(entry.get("fetched_at") or "")
            if ts:
                fetched = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if fetched.tzinfo is None:
                    fetched = fetched.replace(tzinfo=timezone.utc)
                fresh = (now - fetched).total_seconds() < max_age_hours * 3600
        except ValueError:
            fresh = False
        if fresh and entry.get("details"):
            details = entry["details"]
        else:
            try:
                details = fetch_match_details(mid)
                time.sleep(0.35)
            except Exception as exc:
                errors.append(f"{mid}: {exc}")
                details = None
            if details:
                cache[mid] = {"fetched_at": now.isoformat(), "details": details}
        if not details:
            continue
        pred = dict(r.get("prediction") or {})
        fm = dict(pred.get("fotmob_context") or {})
        fm["details"] = details
        pred["fotmob_context"] = fm
        r["prediction"] = pred
        r["fotmob_details"] = details
        enriched += 1

    if enriched:
        _save_details_cache(cache)
    return {"ok": True, "n_enriched": enriched, "n_candidates": len(ranked), "errors": errors}
