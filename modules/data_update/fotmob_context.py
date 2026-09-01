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
from typing import Any, Callable
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


def _get_json(path: str, params: dict | None = None, timeout: int = 20, retries: int = 3) -> dict:
    from modules.data_update.http_client import fetch_json

    qpath = path.lstrip("/")
    last_exc: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            return fetch_json(
                f"{BASE}/{qpath}",
                params=params,
                headers={
                    "Accept": "application/json",
                    "Referer": "https://www.fotmob.com/",
                    "Origin": "https://www.fotmob.com",
                },
                timeout=timeout,
            )
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            last_exc = exc
            time.sleep(0.4 * (attempt + 1))
    raise last_exc or RuntimeError(f"FotMob GET failed: {BASE}/{qpath}")


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


def download_fotmob_context(
    *,
    days: int = 7,
    league_ids: dict[str, int] | None = None,
    on_progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Scarica classifiche + indice partite. Nessuna chiave API."""
    from modules.progress_report import emit

    league_ids = league_ids or FOTMOB_LEAGUES
    errors: list[str] = []
    team_rows: list[dict[str, Any]] = []
    emit(on_progress, 0.05, "FotMob classifiche…")

    items = list(league_ids.items())
    for i, (name, lid) in enumerate(items):
        emit(on_progress, 0.08 + 0.35 * (i / max(1, len(items))), f"Lega {name}")
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
    elif TEAM_CACHE.exists():
        errors.append("classifiche vuote: cache precedente mantenuta")

    matches: list[dict[str, Any]] = []
    today = date.today()
    n_days = max(1, int(days))
    for i in range(n_days):
        day = today + timedelta(days=i)
        key = day.strftime("%Y%m%d")
        emit(on_progress, 0.45 + 0.35 * (i / n_days), f"Match day {day.isoformat()}")
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

    if matches:
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

    xg_info: dict[str, Any] = {}
    try:
        emit(on_progress, 0.88, "xG rolling…")
        xg_info = download_fotmob_xg_rolling(days_back=14, max_details=60)
    except Exception as exc:
        xg_info = {"ok": False, "error": str(exc), "n_teams": 0}

    emit(on_progress, 1.0, f"OK · {len(team_rows)} squadre · {len(matches)} match")
    return {
        "ok": not errors or bool(team_rows) or bool(matches),
        "n_teams": int(len(team_df)) if not team_df.empty else 0,
        "n_matches": len(matches),
        "n_xg_teams": int(xg_info.get("n_teams") or 0),
        "path": str(TEAM_CACHE),
        "matches_path": str(MATCH_CACHE),
        "errors": errors,
        "xg": xg_info,
        "note": "API non ufficiale /api/data — solo quadro (+ xG rolling)",
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
        md = _get_json("matchDetails", {"matchId": match_id}) or {}
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
    home_starters, away_starters = _extract_lineup_names(lineup)
    mom = content.get("momentum") or {}
    # momentum summary (se presente)
    mom_main = mom.get("main") if isinstance(mom, dict) else None
    mom_pts = None
    if isinstance(mom_main, list) and mom_main:
        try:
            vals = [float(x.get("value") if isinstance(x, dict) else x) for x in mom_main[-12:]]
            mom_pts = round(sum(vals) / len(vals), 3) if vals else None
        except (TypeError, ValueError):
            mom_pts = None
    shotmap = content.get("shotmap") or {}
    shots_map_n = None
    if isinstance(shotmap, dict):
        sh = shotmap.get("shots") or shotmap.get("home") or []
        if isinstance(sh, list):
            shots_map_n = len(sh)
    # corner/cards se presenti nelle stats match
    cards_h = cards_a = corners_h = corners_a = None
    for block in periods.get("stats") or []:
        for row in block.get("stats") or []:
            key = str(row.get("key") or "").lower()
            stats = row.get("stats") or [None, None]
            try:
                if "corner" in key:
                    corners_h, corners_a = float(stats[0]), float(stats[1])
                if key in {"yellow_cards", "yellowcards"} or ("yellow" in key and "card" in key):
                    cards_h, cards_a = float(stats[0]), float(stats[1])
            except (TypeError, ValueError):
                pass
    return {
        "match_id": match_id,
        "xg_home": xg_h,
        "xg_away": xg_a,
        "poss_home": poss_h,
        "poss_away": poss_a,
        "shots_home": shots_h,
        "shots_away": shots_a,
        "has_lineup": has_lineup,
        "lineup_home": home_starters,
        "lineup_away": away_starters,
        "n_starters_home": len(home_starters),
        "n_starters_away": len(away_starters),
        "has_momentum": bool(mom.get("main")),
        "has_shotmap": bool(shotmap),
        "momentum_avg": mom_pts,
        "shotmap_n": shots_map_n,
        "cards_home": cards_h,
        "cards_away": cards_a,
        "corners_home": corners_h,
        "corners_away": corners_a,
    }


def _extract_lineup_names(lineup: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Estrae nomi titolari da struttura FotMob (varianti API)."""

    def side_names(side: Any) -> list[str]:
        if not isinstance(side, dict):
            return []
        names: list[str] = []
        # formati tipici
        for key in ("starters", "startXI", "lineup", "players"):
            block = side.get(key)
            if isinstance(block, list):
                for p in block:
                    if not isinstance(p, dict):
                        continue
                    # skip panche se flag
                    if p.get("substitute") is True or p.get("isSub") is True:
                        continue
                    if str(p.get("role") or "").lower() in {"sub", "substitute", "bench"}:
                        continue
                    nm = p.get("name") or p.get("playerName") or (p.get("player") or {}).get("name")
                    if not nm and isinstance(p.get("player"), dict):
                        nm = p["player"].get("name")
                    if nm:
                        names.append(str(nm).strip())
                if names:
                    break
        # nested under "members"
        if not names and isinstance(side.get("members"), list):
            for p in side["members"]:
                if not isinstance(p, dict):
                    continue
                if p.get("isStarter") is False:
                    continue
                nm = p.get("name") or (p.get("player") or {}).get("name")
                if nm:
                    names.append(str(nm).strip())
        # dedupe preserve order, max 11
        seen = set()
        out: list[str] = []
        for n in names:
            k = n.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(n)
            if len(out) >= 11:
                break
        return out

    home = side_names(lineup.get("homeTeam") or lineup.get("home") or {})
    away = side_names(lineup.get("awayTeam") or lineup.get("away") or {})
    return home, away


XG_CACHE = PROCESSED / "fotmob_xg_rolling.csv"
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
            and (((r.get("prediction") or {}).get("fotmob_context") or {}).get("match") or {}).get("match_id")
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


def download_fotmob_xg_rolling(
    *,
    days_back: int = 21,
    max_details: int = 80,
    league_ids: dict[str, int] | None = None,
) -> dict[str, Any]:
    """xG rolling da matchDetails di partite finite (rate-limited)."""
    league_ids = league_ids or {
        k: v for k, v in FOTMOB_LEAGUES.items() if k in {
            "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "Championship",
        }
    }
    allow = set(league_ids.values())
    cache = _load_details_cache()
    now = datetime.now(timezone.utc)
    finished: list[dict[str, Any]] = []
    errors: list[str] = []

    for i in range(1, max(2, int(days_back) + 1)):
        day = date.today() - timedelta(days=i)
        key = day.strftime("%Y%m%d")
        try:
            payload = _get_json("matches", {"date": key})
            time.sleep(0.2)
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            errors.append(f"{key}: {exc}")
            continue
        for league in payload.get("leagues") or []:
            if league.get("id") not in allow:
                continue
            for m in league.get("matches") or []:
                st = m.get("status") or {}
                if not st.get("finished"):
                    continue
                home = m.get("home") or {}
                away = m.get("away") or {}
                finished.append(
                    {
                        "match_id": m.get("id"),
                        "home": home.get("name") or home.get("longName"),
                        "away": away.get("name") or away.get("longName"),
                        "date": str(st.get("utcTime") or day.isoformat())[:10],
                    }
                )
        if len(finished) >= max_details * 2:
            break

    # dedupe + limit
    seen: set[str] = set()
    todo: list[dict] = []
    for m in finished:
        mid = str(m.get("match_id") or "")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        todo.append(m)
        if len(todo) >= max_details:
            break

    team_acc: dict[str, dict[str, float]] = {}
    used = 0
    for m in todo:
        mid = str(m["match_id"])
        entry = cache.get(mid) or {}
        details = entry.get("details") if isinstance(entry, dict) else None
        fresh = False
        try:
            ts = str(entry.get("fetched_at") or "")
            if ts and details:
                fetched = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if fetched.tzinfo is None:
                    fetched = fetched.replace(tzinfo=timezone.utc)
                fresh = (now - fetched).total_seconds() < 24 * 3600
        except ValueError:
            fresh = bool(details)
        if not fresh:
            details = fetch_match_details(mid)
            time.sleep(0.35)
            if details:
                cache[mid] = {"fetched_at": now.isoformat(), "details": details}
        if not details or details.get("xg_home") is None or details.get("xg_away") is None:
            continue
        used += 1
        for team, xf, xa in (
            (m.get("home"), details["xg_home"], details["xg_away"]),
            (m.get("away"), details["xg_away"], details["xg_home"]),
        ):
            if not team:
                continue
            k = _norm(str(team))
            row = team_acc.setdefault(k, {"team": str(team), "xg_for": 0.0, "xg_against": 0.0, "n": 0.0})
            row["xg_for"] += float(xf)
            row["xg_against"] += float(xa)
            row["n"] += 1.0

    _save_details_cache(cache)
    rows = []
    for k, row in team_acc.items():
        n = max(1.0, row["n"])
        rows.append(
            {
                "team": row["team"],
                "team_norm": k,
                "n": int(row["n"]),
                "xg_for": round(row["xg_for"] / n, 3),
                "xg_against": round(row["xg_against"] / n, 3),
                "xg_diff": round((row["xg_for"] - row["xg_against"]) / n, 3),
                "fetched_at": now.isoformat(),
            }
        )
    if rows:
        df = pd.DataFrame(rows)
        XG_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(XG_CACHE, index=False)
    elif XG_CACHE.exists():
        # keep previous
        try:
            old = pd.read_csv(XG_CACHE)
            return {
                "ok": True,
                "n_teams": int(len(old)),
                "n_details": used,
                "from_cache": True,
                "errors": errors,
                "path": str(XG_CACHE),
            }
        except Exception:
            pass
    return {
        "ok": bool(rows),
        "n_teams": len(rows),
        "n_details": used,
        "n_candidates": len(todo),
        "from_cache": False,
        "errors": errors,
        "path": str(XG_CACHE),
    }


def load_fotmob_xg_index() -> dict[str, dict[str, Any]]:
    if not XG_CACHE.exists():
        return {}
    df = pd.read_csv(XG_CACHE)
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        k = _norm(str(row.get("team") or row.get("team_norm") or ""))
        if k:
            out[k] = row.to_dict()
    return out


def lookup_fotmob_xg(name: str, idx: dict[str, dict[str, Any]] | None = None) -> dict[str, Any] | None:
    k = _norm(name)
    if not k:
        return None
    idx = idx or load_fotmob_xg_index()
    if k in idx:
        return idx[k]
    hit = difflib.get_close_matches(k, list(idx.keys()), n=1, cutoff=0.86)
    if hit and not _reserve_mismatch(name, hit[0]):
        return idx[hit[0]]
    return None


def _player_name_from_event(ev: dict[str, Any]) -> str | None:
    if not isinstance(ev, dict):
        return None
    for key in ("playerName", "name", "title"):
        val = ev.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    pl = ev.get("player")
    if isinstance(pl, dict):
        for key in ("name", "shortName", "fullName"):
            val = pl.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    elif isinstance(pl, str) and pl.strip():
        return pl.strip()
    return None


def _is_goal_event(ev: dict[str, Any]) -> bool:
    if not isinstance(ev, dict):
        return False
    typ = str(ev.get("type") or ev.get("eventType") or ev.get("title") or "").lower()
    if "own" in typ and "goal" in typ:
        return False
    if "penalty missed" in typ or "miss" in typ:
        return False
    if "goal" in typ or typ in {"g", "goal"}:
        return True
    if ev.get("isGoal") is True:
        return True
    return False


def _iter_fotmob_events(md: dict[str, Any]) -> list[dict[str, Any]]:
    """Raccoglie eventi gol da strutture FotMob matchDetails (varianti API)."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(ev: Any) -> None:
        if not isinstance(ev, dict):
            return
        name = _player_name_from_event(ev)
        if not name or not _is_goal_event(ev):
            return
        minute = ev.get("time") or ev.get("minute") or ev.get("min") or 9999
        try:
            minute_f = float(minute)
        except (TypeError, ValueError):
            minute_f = 9999.0
        key = f"{name.lower()}|{minute_f}"
        if key in seen:
            return
        seen.add(key)
        out.append({"player": name, "minute": minute_f, "raw": ev})

    header = md.get("header") or {}
    for block in (header.get("events"), header.get("Events")):
        if isinstance(block, list):
            for ev in block:
                add(ev)
        elif isinstance(block, dict):
            for sub in block.values():
                if isinstance(sub, list):
                    for ev in sub:
                        add(ev)

    content = md.get("content") or {}
    mf = content.get("matchFacts") or {}
    for block in (mf.get("events"), mf.get("Events"), content.get("events")):
        if isinstance(block, list):
            for ev in block:
                add(ev)

    liveticker = content.get("liveticker") or {}
    for block in (liveticker.get("events"), liveticker.get("Events")):
        if isinstance(block, list):
            for ev in block:
                add(ev)

    out.sort(key=lambda x: float(x.get("minute") or 9999))
    return out


def extract_goal_scorers(match_id: int | str, *, md: dict[str, Any] | None = None) -> dict[str, Any]:
    """Marcatori reale partita: anytime list + first scorer (esclusi autogol)."""
    payload = md
    if payload is None:
        try:
            payload = _get_json("matchDetails", {"matchId": match_id}) or {}
        except (HTTPError, URLError, TimeoutError, ValueError, OSError):
            return {"ok": False, "match_id": match_id, "scorers": [], "first_scorer": None}
    events = _iter_fotmob_events(payload or {})
    scorers = [str(e["player"]) for e in events if e.get("player")]
    first = scorers[0] if scorers else None
    return {
        "ok": bool(scorers),
        "match_id": match_id,
        "scorers": scorers,
        "first_scorer": first,
        "n_goals": len(scorers),
    }


def scorer_hit(picked_player: str, scorers: list[str], *, mode: str = "anytime") -> bool | None:
    """True/False se matchabile; None se lista marcatori vuota."""
    from modules.advisor.scorers import _name_match

    if not scorers:
        return None
    target = str(picked_player or "").strip()
    if not target:
        return None
    if mode == "first":
        return _name_match(target, scorers[0])
    return any(_name_match(target, s) for s in scorers)

