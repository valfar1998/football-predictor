"""Sofascore post-match: risultati FT + stats/incidenti per settle e analisi.

Fonte primaria Big 5 (schedule soccerdata + API event). Soft-fail; non entra in EV/Kelly.
"""

from __future__ import annotations

import json
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from modules.data_update.sd_compat import assert_soccerdata_available, quiet_soccerdata, season_codes
from modules.data_update.sofascore_context import SOFA_LEAGUES

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
STATS_CACHE = PROCESSED / "sofascore_match_stats.json"

SOFASCORE_API = "https://api.sofascore.com/api/v1/"
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"

# Chiavi normalizzate (lower) → campi flat home/away
_STAT_KEYS: dict[str, tuple[str, str]] = {
    "yellow cards": ("HY", "AY"),
    "yellow card": ("HY", "AY"),
    "red cards": ("HR", "AR"),
    "red card": ("HR", "AR"),
    "corner kicks": ("HC", "AC"),
    "corners": ("HC", "AC"),
    "ball possession": ("possession_home", "possession_away"),
    "possession": ("possession_home", "possession_away"),
    "expected goals": ("xg_home", "xg_away"),
    "expected goals (xg)": ("xg_home", "xg_away"),
    "xg": ("xg_home", "xg_away"),
    "total shots": ("shots_home", "shots_away"),
    "shots": ("shots_home", "shots_away"),
    "shots on target": ("sot_home", "sot_away"),
    "shots on goal": ("sot_home", "sot_away"),
    "fouls": ("fouls_home", "fouls_away"),
    "big chances": ("big_chances_home", "big_chances_away"),
}


def _load_cache() -> dict[str, Any]:
    if not STATS_CACHE.exists():
        return {}
    try:
        return json.loads(STATS_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict[str, Any]) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    STATS_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _event_get(path: str, *, timeout: int = 20) -> dict[str, Any] | None:
    """GET api.sofascore.com/api/v1/{path} soft-fail."""
    url = SOFASCORE_API + path.lstrip("/")
    try:
        from modules.data_update.http_client import fetch_json

        return fetch_json(
            url,
            headers={"User-Agent": _UA, "Accept": "application/json"},
            timeout=timeout,
            retries=2,
        )
    except Exception:
        return None


def fetch_event_statistics(event_id: int | str) -> dict[str, Any] | None:
    data = _event_get(f"event/{int(event_id)}/statistics")
    if not isinstance(data, dict):
        return None
    return data


def fetch_event_incidents(event_id: int | str) -> dict[str, Any] | None:
    data = _event_get(f"event/{int(event_id)}/incidents")
    if not isinstance(data, dict):
        return None
    return data


def _norm_stat_name(name: str) -> str:
    s = re.sub(r"\s+", " ", str(name or "").strip().lower())
    s = s.replace("%", "").strip()
    return s


def parse_statistics_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Estrae stats FT per nome (non per indice di gruppo)."""
    out: dict[str, Any] = {}
    if not payload:
        return out
    periods = payload.get("statistics") or []
    if not isinstance(periods, list):
        return out
    # Preferisci "ALL" / full match; altrimenti primo blocco
    block = None
    for p in periods:
        if not isinstance(p, dict):
            continue
        period = str(p.get("period") or "").upper()
        if period in {"ALL", "FULL", "FT", "MATCH"}:
            block = p
            break
    if block is None and periods:
        block = periods[0] if isinstance(periods[0], dict) else None
    if not block:
        return out
    for group in block.get("groups") or []:
        if not isinstance(group, dict):
            continue
        for item in group.get("statisticsItems") or []:
            if not isinstance(item, dict):
                continue
            key = _norm_stat_name(str(item.get("name") or item.get("key") or ""))
            mapped = _STAT_KEYS.get(key)
            if not mapped:
                continue
            hk, ak = mapped
            hv, av = item.get("homeValue"), item.get("awayValue")
            if hv is None and item.get("home") is not None:
                hv = item.get("home")
            if av is None and item.get("away") is not None:
                av = item.get("away")
            try:
                if hv is not None:
                    out[hk] = float(hv) if "." in str(hv) else int(float(hv))
            except (TypeError, ValueError):
                out[hk] = hv
            try:
                if av is not None:
                    out[ak] = float(av) if "." in str(av) else int(float(av))
            except (TypeError, ValueError):
                out[ak] = av
    return out


def scorer_names_from_incidents(incidents: dict[str, Any] | list | None) -> dict[str, Any]:
    """Lista marcatori (esclusi autogol) + first scorer da timeline Sofascore."""
    raw = incidents
    if isinstance(incidents, dict):
        raw = incidents.get("incidents") or incidents.get("events") or []
    if not isinstance(raw, list):
        return {"ok": False, "scorers": [], "first_scorer": None, "n_goals": 0}

    def _is_goal(ev: dict[str, Any]) -> bool:
        if ev.get("isOwnGoal") or str(ev.get("incidentClass") or "").lower() == "owngoal":
            return False
        itype = str(ev.get("incidentType") or ev.get("type") or "").lower()
        iclass = str(ev.get("incidentClass") or "").lower()
        if ev.get("isGoal"):
            return True
        if itype in {"goal", "penalty"}:
            return True
        if iclass in {"goal", "penalty"}:
            return True
        return False

    timed: list[tuple[int, str]] = []
    for ev in raw:
        if not isinstance(ev, dict) or not _is_goal(ev):
            continue
        player = ""
        pobj = ev.get("player") or ev.get("playerName")
        if isinstance(pobj, dict):
            player = str(pobj.get("name") or pobj.get("shortName") or "").strip()
        elif pobj:
            player = str(pobj).strip()
        if not player:
            continue
        t = ev.get("time")
        if isinstance(t, dict):
            minute = int(t.get("minute") or t.get("seconds") or 0)
        else:
            try:
                minute = int(t or 0)
            except (TypeError, ValueError):
                minute = 0
        timed.append((minute, player))
    timed.sort(key=lambda x: x[0])
    scorers = [p for _, p in timed]
    return {
        "ok": bool(scorers),
        "scorers": scorers,
        "first_scorer": scorers[0] if scorers else None,
        "n_goals": len(scorers),
    }


def _schedule_finished(*, days_back: int = 7, seasons: list[str] | None = None) -> pd.DataFrame:
    """Schedule Big 5 FT recente via soccerdata."""
    try:
        sd = assert_soccerdata_available()
    except Exception:
        return pd.DataFrame()
    seasons = season_codes(seasons)
    try:
        with quiet_soccerdata():
            sofa = sd.Sofascore(leagues=SOFA_LEAGUES, seasons=seasons)
            raw = sofa.read_schedule(force_cache=False)
    except Exception as exc:
        print(f"skip Sofascore schedule: {exc}")
        return pd.DataFrame()
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    df = raw.reset_index()
    # colonne attese: date, home_team, away_team, home_score, away_score, game_id
    rename = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"home_team", "home"}:
            rename[c] = "home_team"
        elif cl in {"away_team", "away"}:
            rename[c] = "away_team"
        elif cl in {"home_score", "home_goals"}:
            rename[c] = "home_score"
        elif cl in {"away_score", "away_goals"}:
            rename[c] = "away_score"
        elif cl in {"game_id", "id"}:
            rename[c] = "game_id"
        elif cl == "date":
            rename[c] = "date"
    df = df.rename(columns=rename)
    need = {"date", "home_team", "away_team", "home_score", "away_score", "game_id"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()
    df["home_goals"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_goals", "away_goals", "game_id"])
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(days_back)))
    df = df[df["date"] >= cutoff]
    df["sofascore_match_id"] = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["sofascore_match_id"])
    return df.reset_index(drop=True)


def fetch_sofascore_results(
    *,
    days_back: int = 7,
    with_side_stats: bool = True,
    max_stat_fetch: int = 80,
    seasons: list[str] | None = None,
) -> pd.DataFrame:
    """DataFrame compatibile con settle_from_results (+ sofascore_match_id, opz. HY/HC)."""
    sched = _schedule_finished(days_back=days_back, seasons=seasons)
    if sched.empty:
        return pd.DataFrame()

    cache = _load_cache()
    rows: list[dict[str, Any]] = []
    fetched = 0
    for _, fx in sched.iterrows():
        eid = int(fx["sofascore_match_id"])
        day = pd.Timestamp(fx["date"]).strftime("%Y-%m-%d")
        rec: dict[str, Any] = {
            "date": day,
            "home_team": str(fx["home_team"]),
            "away_team": str(fx["away_team"]),
            "home_goals": int(fx["home_goals"]),
            "away_goals": int(fx["away_goals"]),
            "sofascore_match_id": eid,
            "league": str(fx.get("league") or ""),
        }
        cached = cache.get(str(eid)) if isinstance(cache.get(str(eid)), dict) else None
        stats_flat: dict[str, Any] = {}
        if cached and isinstance(cached.get("stats"), dict):
            stats_flat = dict(cached["stats"])
        elif with_side_stats and fetched < max_stat_fetch:
            payload = fetch_event_statistics(eid)
            stats_flat = parse_statistics_payload(payload)
            fetched += 1
            if payload is not None:
                entry = dict(cached or {})
                entry.update(
                    {
                        "sofascore_match_id": eid,
                        "date": day,
                        "home": rec["home_team"],
                        "away": rec["away_team"],
                        "home_goals": rec["home_goals"],
                        "away_goals": rec["away_goals"],
                        "stats": stats_flat,
                        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                )
                cache[str(eid)] = entry
            time.sleep(0.12)
        for k in ("HY", "AY", "HR", "AR", "HC", "AC"):
            if k in stats_flat and stats_flat[k] is not None:
                try:
                    rec[k] = int(float(stats_flat[k]))
                except (TypeError, ValueError):
                    pass
        # Tiri totali per settle SHOT*
        for src, dst in (("shots_home", "SH"), ("shots_away", "SA")):
            if src in stats_flat and stats_flat[src] is not None:
                try:
                    rec[dst] = int(float(stats_flat[src]))
                    rec[src] = rec[dst]
                except (TypeError, ValueError):
                    pass
        rows.append(rec)

    if fetched:
        _save_cache(cache)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def normalized_stats_bundle(
    event_id: int,
    *,
    home: str = "",
    away: str = "",
    date_s: str = "",
    home_goals: int | None = None,
    away_goals: int | None = None,
) -> dict[str, Any]:
    """Scarica (o riusa cache) statistics + scorers per un event_id."""
    cache = _load_cache()
    key = str(int(event_id))
    entry = dict(cache.get(key) or {}) if isinstance(cache.get(key), dict) else {}
    need_stats = not isinstance(entry.get("stats"), dict) or not entry.get("stats")
    need_scorers = not entry.get("scorers")
    if need_stats:
        payload = fetch_event_statistics(event_id)
        entry["stats"] = parse_statistics_payload(payload)
        time.sleep(0.1)
    if need_scorers:
        inc = fetch_event_incidents(event_id)
        names = scorer_names_from_incidents(inc)
        entry["scorers"] = names.get("scorers") or []
        entry["first_scorer"] = names.get("first_scorer")
        entry["incidents_ok"] = bool(names.get("ok"))
        time.sleep(0.1)
    entry.update(
        {
            "sofascore_match_id": int(event_id),
            "date": date_s or entry.get("date") or "",
            "home": home or entry.get("home") or "",
            "away": away or entry.get("away") or "",
            "home_goals": home_goals if home_goals is not None else entry.get("home_goals"),
            "away_goals": away_goals if away_goals is not None else entry.get("away_goals"),
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    cache[key] = entry
    _save_cache(cache)
    return entry


def enrich_settled_sofascore_stats(*, limit: int = 60, days_back: int = 14) -> dict[str, Any]:
    """Scrive sofascore_stats JSON sulle righe history con sofascore_match_id (o match da schedule)."""
    from modules.data_update import history as hist

    conn = hist._connect()
    updated = 0
    skipped = 0
    errors: list[str] = []
    try:
        hist._migrate_jsonl(conn)
        cutoff = (date.today() - timedelta(days=max(1, int(days_back)))).isoformat()
        rows = conn.execute(
            """
            SELECT match_key, date, home, away, home_goals, away_goals,
                   sofascore_match_id, sofascore_stats
            FROM matches
            WHERE date >= ?
              AND (
                sofascore_match_id IS NOT NULL
                OR (home_goals IS NOT NULL AND away_goals IS NOT NULL)
              )
            ORDER BY date DESC
            LIMIT ?
            """,
            (cutoff, int(limit) * 3),
        ).fetchall()
    except Exception as exc:
        conn.close()
        return {"ok": False, "updated": 0, "error": str(exc)}

    # Mappa schedule recente id → (date, home, away) per match senza id
    id_by_key: dict[str, int] = {}
    try:
        sched = _schedule_finished(days_back=days_back)
        from modules.data_update.team_names import resolve_known_team

        for _, fx in sched.iterrows():
            day = pd.Timestamp(fx["date"]).strftime("%Y-%m-%d")
            home = resolve_known_team(str(fx["home_team"])) or str(fx["home_team"])
            away = resolve_known_team(str(fx["away_team"])) or str(fx["away_team"])
            eid = int(fx["sofascore_match_id"])
            id_by_key[f"{day}|{home}|{away}"] = eid
            id_by_key[f"{day}|{fx['home_team']}|{fx['away_team']}"] = eid
    except Exception:
        pass

    try:
        for rec in rows:
            if updated >= limit:
                break
            raw_stats = rec["sofascore_stats"] if "sofascore_stats" in rec.keys() else None
            if raw_stats and str(raw_stats).strip().startswith("{"):
                skipped += 1
                continue
            mid = rec["sofascore_match_id"] if "sofascore_match_id" in rec.keys() else None
            if mid is None:
                mid = id_by_key.get(str(rec["match_key"] or ""))
            if mid is None:
                skipped += 1
                continue
            try:
                mid_i = int(mid)
            except (TypeError, ValueError):
                skipped += 1
                continue
            try:
                bundle = normalized_stats_bundle(
                    mid_i,
                    home=str(rec["home"] or ""),
                    away=str(rec["away"] or ""),
                    date_s=str(rec["date"] or ""),
                    home_goals=int(rec["home_goals"]) if rec["home_goals"] is not None else None,
                    away_goals=int(rec["away_goals"]) if rec["away_goals"] is not None else None,
                )
                payload = {
                    "sofascore_match_id": mid_i,
                    "stats": bundle.get("stats") or {},
                    "scorers": bundle.get("scorers") or [],
                    "first_scorer": bundle.get("first_scorer"),
                    "fetched_at": bundle.get("fetched_at"),
                }
                conn.execute(
                    """
                    UPDATE matches
                    SET sofascore_match_id=COALESCE(sofascore_match_id, ?),
                        sofascore_stats=?
                    WHERE match_key=?
                    """,
                    (mid_i, json.dumps(payload, ensure_ascii=False), rec["match_key"]),
                )
                updated += 1
            except Exception as exc:
                errors.append(f"{rec['match_key']}: {exc}")
        conn.commit()
    finally:
        conn.close()
    return {
        "ok": True,
        "updated": updated,
        "skipped": skipped,
        "errors": errors[:8],
        "cache": str(STATS_CACHE),
    }


def extract_goal_scorers_sofascore(event_id: int | str) -> dict[str, Any]:
    """Marcatori da incidents Sofascore (stesso shape di FotMob extract_goal_scorers)."""
    try:
        eid = int(event_id)
    except (TypeError, ValueError):
        return {"ok": False, "match_id": event_id, "scorers": [], "first_scorer": None}
    cache = _load_cache()
    cached = cache.get(str(eid))
    if isinstance(cached, dict) and cached.get("scorers"):
        scorers = list(cached.get("scorers") or [])
        return {
            "ok": bool(scorers),
            "match_id": eid,
            "scorers": scorers,
            "first_scorer": cached.get("first_scorer") or (scorers[0] if scorers else None),
            "n_goals": len(scorers),
            "source": "sofascore",
        }
    inc = fetch_event_incidents(eid)
    names = scorer_names_from_incidents(inc)
    entry = dict(cached) if isinstance(cached, dict) else {"sofascore_match_id": eid}
    entry["scorers"] = names.get("scorers") or []
    entry["first_scorer"] = names.get("first_scorer")
    entry["fetched_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cache[str(eid)] = entry
    _save_cache(cache)
    return {
        "ok": bool(names.get("ok")),
        "match_id": eid,
        "scorers": names.get("scorers") or [],
        "first_scorer": names.get("first_scorer"),
        "n_goals": int(names.get("n_goals") or 0),
        "source": "sofascore",
    }
