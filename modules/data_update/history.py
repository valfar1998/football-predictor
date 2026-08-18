"""Storico locale SQLite: tutte le partite (anche N/D) + esiti per il voto unificato.

Non usa MySQL: per un'app locale SQLite evita server, password e dipendenze extra.
Dopo MIN_GLOBAL_SETTLED partite chiuse e MIN_TEAM_MATCHES per squadra, lo storico
entra nel voto unificato con peso HISTORY_WEIGHT (12%).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
JSONL = PROCESSED / "our_history.jsonl"
DB = PROCESSED / "our_history.sqlite"

MIN_TEAM_MATCHES = 6
MIN_GLOBAL_SETTLED = 30
HISTORY_WEIGHT = 0.12

_CREATE = """
CREATE TABLE IF NOT EXISTS matches (
    match_key TEXT PRIMARY KEY,
    date TEXT,
    time TEXT,
    home TEXT,
    away TEXT,
    league TEXT,
    country TEXT,
    pick TEXT,
    action TEXT,
    score INTEGER,
    score_unified INTEGER,
    ev_cons REAL,
    probability REAL,
    odds_source TEXT,
    skip_reason TEXT,
    covered INTEGER,
    home_goals INTEGER,
    away_goals INTEGER,
    result TEXT,
    hit INTEGER,
    saved_at TEXT,
    settled_at TEXT
)
"""


def _key(row: dict[str, Any]) -> str:
    return f"{row.get('date')}|{row.get('home')}|{row.get('away')}"


def _connect() -> sqlite3.Connection:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_CREATE)
    return conn


def _migrate_jsonl(conn: sqlite3.Connection) -> None:
    n = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    if n or not JSONL.exists():
        return
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for line in JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not rec.get("match_key"):
            rec["match_key"] = _key(rec)
        if not rec.get("match_key") or rec["match_key"].startswith("None|") or rec["match_key"].endswith("|None"):
            continue
        rec["covered"] = 1 if rec.get("covered") else 0
        _upsert(conn, rec, now, keep_result=True)
    conn.commit()


def _upsert(conn: sqlite3.Connection, rec: dict[str, Any], now: str, *, keep_result: bool) -> None:
    if not rec.get("match_key"):
        rec["match_key"] = _key(rec)
    rec.setdefault("time", "")
    rec.setdefault("league", "")
    rec.setdefault("country", "")
    rec.setdefault("pick", None)
    rec.setdefault("action", None)
    rec.setdefault("score", None)
    rec.setdefault("score_unified", None)
    rec.setdefault("ev_cons", None)
    rec.setdefault("probability", None)
    rec.setdefault("odds_source", None)
    rec.setdefault("skip_reason", None)
    rec.setdefault("covered", 0)
    rec.setdefault("home_goals", None)
    rec.setdefault("away_goals", None)
    rec.setdefault("result", None)
    rec.setdefault("hit", None)
    rec.setdefault("saved_at", now)
    rec.setdefault("settled_at", None)
    prev = conn.execute("SELECT * FROM matches WHERE match_key=?", (rec["match_key"],)).fetchone()
    if prev and keep_result and prev["result"]:
        rec["result"] = prev["result"]
        rec["home_goals"] = prev["home_goals"]
        rec["away_goals"] = prev["away_goals"]
        rec["hit"] = prev["hit"]
        rec["settled_at"] = prev["settled_at"]
    conn.execute(
        """
        INSERT INTO matches (
            match_key, date, time, home, away, league, country, pick, action,
            score, score_unified, ev_cons, probability, odds_source, skip_reason,
            covered, home_goals, away_goals, result, hit, saved_at, settled_at
        ) VALUES (
            :match_key, :date, :time, :home, :away, :league, :country, :pick, :action,
            :score, :score_unified, :ev_cons, :probability, :odds_source, :skip_reason,
            :covered, :home_goals, :away_goals, :result, :hit, :saved_at, :settled_at
        )
        ON CONFLICT(match_key) DO UPDATE SET
            time=excluded.time, league=excluded.league, country=excluded.country,
            pick=excluded.pick, action=excluded.action, score=excluded.score,
            score_unified=excluded.score_unified, ev_cons=excluded.ev_cons,
            probability=excluded.probability, odds_source=excluded.odds_source,
            skip_reason=excluded.skip_reason, covered=excluded.covered,
            saved_at=excluded.saved_at,
            home_goals=COALESCE(matches.home_goals, excluded.home_goals),
            away_goals=COALESCE(matches.away_goals, excluded.away_goals),
            result=COALESCE(matches.result, excluded.result),
            hit=COALESCE(matches.hit, excluded.hit),
            settled_at=COALESCE(matches.settled_at, excluded.settled_at)
        """,
        rec,
    )


def load_history() -> list[dict[str, Any]]:
    conn = _connect()
    try:
        _migrate_jsonl(conn)
        rows = conn.execute("SELECT * FROM matches ORDER BY date, home").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def archive_upcoming(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Salva/aggiorna tutte le partite del calendario, comprese le N/D."""
    conn = _connect()
    try:
        _migrate_jsonl(conn)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        added = 0
        from modules.data_update.team_names import resolve_known_team

        for row in rows:
            home = resolve_known_team(row.get("home") or "") or row.get("home")
            away = resolve_known_team(row.get("away") or "") or row.get("away")
            rec = {
                "match_key": _key({"date": row.get("date"), "home": home, "away": away}),
                "date": row.get("date"),
                "time": row.get("time"),
                "home": home,
                "away": away,
                "league": row.get("league"),
                "country": row.get("country"),
                "pick": row.get("pick"),
                "action": row.get("action"),
                "score": row.get("score"),
                "score_unified": row.get("score_unified"),
                "ev_cons": row.get("ev_cons"),
                "probability": row.get("probability"),
                "odds_source": row.get("odds_source"),
                "skip_reason": row.get("skip_reason"),
                "covered": 1 if row.get("action") not in {"n/d", "invalido", None} and not row.get("skip_reason") else 0,
                "home_goals": None,
                "away_goals": None,
                "result": None,
                "hit": None,
                "saved_at": now,
                "settled_at": None,
            }
            exists = conn.execute("SELECT 1 FROM matches WHERE match_key=?", (rec["match_key"],)).fetchone()
            if not exists:
                added += 1
            _upsert(conn, rec, now, keep_result=True)
        conn.commit()
        n = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        return {"n_history": int(n), "added": added, "updated": len(rows) - added, "path": str(DB)}
    finally:
        conn.close()


def settle_from_results(results: pd.DataFrame) -> dict[str, Any]:
    """Chiude i match quando arrivano i gol (nomi allineati al dizionario FD)."""
    if results is None or results.empty:
        return {"settled": 0}
    from modules.data_update.team_names import resolve_known_team

    conn = _connect()
    try:
        _migrate_jsonl(conn)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        settled = 0
        for _, fx in results.iterrows():
            try:
                day = pd.Timestamp(fx["date"]).strftime("%Y-%m-%d")
                hg, ag = int(fx["home_goals"]), int(fx["away_goals"])
            except (TypeError, ValueError, KeyError):
                continue
            home = resolve_known_team(str(fx.get("home_team") or "")) or str(fx.get("home_team") or "").strip()
            away = resolve_known_team(str(fx.get("away_team") or "")) or str(fx.get("away_team") or "").strip()
            if hg > ag:
                res = "1"
            elif hg < ag:
                res = "2"
            else:
                res = "X"
            keys = [f"{day}|{home}|{away}"]
            raw_h, raw_a = str(fx.get("home_team") or "").strip(), str(fx.get("away_team") or "").strip()
            if raw_h and raw_a:
                keys.append(f"{day}|{raw_h}|{raw_a}")
            rec = None
            for k in keys:
                rec = conn.execute("SELECT * FROM matches WHERE match_key=?", (k,)).fetchone()
                if rec:
                    break
            if rec is None:
                rec = conn.execute(
                    "SELECT * FROM matches WHERE date=? AND home=? AND away=? AND result IS NULL",
                    (day, home, away),
                ).fetchone()
            if rec is None and raw_h and raw_a:
                rec = conn.execute(
                    "SELECT * FROM matches WHERE date=? AND home=? AND away=? AND result IS NULL",
                    (day, raw_h, raw_a),
                ).fetchone()
            if not rec or rec["result"]:
                continue
            pick = str(rec["pick"] or "")
            hit = 1 if pick in {"1", "X", "2"} and pick == res else 0
            conn.execute(
                """
                UPDATE matches
                SET home_goals=?, away_goals=?, result=?, hit=?, settled_at=?
                WHERE match_key=?
                """,
                (hg, ag, res, hit, now, rec["match_key"]),
            )
            settled += 1
        conn.commit()
        n = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        return {"settled": settled, "n_history": int(n)}
    finally:
        conn.close()


def settle_pending() -> dict[str, Any]:
    """Chiude i match archiviati usando coppe (org) e risultati football-data.co.uk."""
    conn = _connect()
    try:
        _migrate_jsonl(conn)
        pending = conn.execute(
            "SELECT MIN(date) AS min_d FROM matches WHERE result IS NULL AND date IS NOT NULL"
        ).fetchone()
        min_d = pending["min_d"] if pending else None
    finally:
        conn.close()

    settled = 0
    try:
        from modules.data_update.cups import load_org_cup_results

        settled += int(settle_from_results(load_org_cup_results()).get("settled") or 0)
    except Exception:
        pass
    try:
        from modules.data_update.parse import load_historical

        hist = load_historical(min_date=str(min_d or "2025-07-01"))
        settled += int(settle_from_results(hist).get("settled") or 0)
    except Exception:
        pass
    summary = history_summary()
    summary["settled"] = settled
    return summary


def _team_form(conn: sqlite3.Connection, team: str) -> dict[str, Any] | None:
    rows = conn.execute(
        "SELECT home, away, home_goals, away_goals, result FROM matches WHERE result IS NOT NULL AND (home=? OR away=?)",
        (team, team),
    ).fetchall()
    if len(rows) < MIN_TEAM_MATCHES:
        return None
    pts = gf = ga = 0
    for r in rows:
        if r["home"] == team:
            g_for, g_against = int(r["home_goals"]), int(r["away_goals"])
        else:
            g_for, g_against = int(r["away_goals"]), int(r["home_goals"])
        gf += g_for
        ga += g_against
        if g_for > g_against:
            pts += 3
        elif g_for == g_against:
            pts += 1
    n = len(rows)
    return {
        "team": team,
        "n": n,
        "ppg": round(pts / n, 3),
        "gd_pg": round((gf - ga) / n, 3),
        "gf_pg": round(gf / n, 3),
        "ga_pg": round(ga / n, 3),
    }


def lookup_history_match(home: str, away: str) -> dict[str, Any]:
    """Segnale per il quadro/voto: pronto solo dopo abbastanza esiti locali."""
    empty = {"ready": False, "n_global": 0, "home": None, "away": None}
    if not DB.exists() and not JSONL.exists():
        return empty
    from modules.data_update.team_names import resolve_known_team

    home = resolve_known_team(home) or home
    away = resolve_known_team(away) or away
    conn = _connect()
    try:
        _migrate_jsonl(conn)
        n_global = conn.execute("SELECT COUNT(*) FROM matches WHERE result IS NOT NULL").fetchone()[0]
        h = _team_form(conn, home)
        a = _team_form(conn, away)
        ready = int(n_global) >= MIN_GLOBAL_SETTLED and h is not None and a is not None
        return {
            "ready": ready,
            "n_global": int(n_global),
            "min_team": MIN_TEAM_MATCHES,
            "min_global": MIN_GLOBAL_SETTLED,
            "weight": HISTORY_WEIGHT,
            "home": h,
            "away": a,
        }
    finally:
        conn.close()


def history_summary() -> dict[str, Any]:
    conn = _connect()
    try:
        _migrate_jsonl(conn)
        n = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        settled = conn.execute("SELECT COUNT(*) FROM matches WHERE result IS NOT NULL").fetchone()[0]
        nd = conn.execute("SELECT COUNT(*) FROM matches WHERE action IN ('n/d', 'invalido')").fetchone()[0]
        return {
            "n_history": int(n),
            "n_settled": int(settled),
            "n_nd": int(nd),
            "ready": int(settled) >= MIN_GLOBAL_SETTLED,
            "min_global": MIN_GLOBAL_SETTLED,
            "min_team": MIN_TEAM_MATCHES,
            "weight": HISTORY_WEIGHT,
            "path": str(DB),
        }
    finally:
        conn.close()
