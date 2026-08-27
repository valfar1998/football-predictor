"""Storico locale SQLite: tutte le partite (anche N/D) + esiti per il voto unificato.

Non usa MySQL: per un'app locale SQLite evita server, password e dipendenze extra.
Dopo MIN_GLOBAL_SETTLED partite chiuse e MIN_TEAM_MATCHES per squadra, lo storico
entra nel voto unificato con peso da HISTORY_WEIGHT (12%) fino a HISTORY_WEIGHT_MAX (18%)
quando ci sono abbastanza esiti globali e di lega.
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
PLAYS_CSV = ROOT / "storico_giocate.csv"
MONTHLY_SUCCESS_CSV = ROOT / "storico_successo_mensile.csv"
HIGH_VOTES = (7, 8, 9, 10)

MIN_TEAM_MATCHES = 6
MIN_GLOBAL_SETTLED = 30
MIN_GLOBAL_BOOST = 80
MIN_LEAGUE_SETTLED = 20
HISTORY_WEIGHT = 0.12
HISTORY_WEIGHT_MAX = 0.18

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
    settled_at TEXT,
    quota_pick REAL,
    agree_share REAL,
    data_edge REAL,
    move_rank REAL,
    residual REAL,
    adj_ev REAL,
    data_factors TEXT,
    no_bet_reasons TEXT,
    pick_group TEXT,
    model_cluster TEXT,
    ev_sharp REAL
)
"""

_EXTRA_COLS = {
    "quota_pick": "REAL",
    "agree_share": "REAL",
    "data_edge": "REAL",
    "move_rank": "REAL",
    "residual": "REAL",
    "adj_ev": "REAL",
    "data_factors": "TEXT",
    "no_bet_reasons": "TEXT",
    "pick_group": "TEXT",
    "model_cluster": "TEXT",
    "ev_sharp": "REAL",
    "context_partial": "INTEGER",
    "synthetic_backfill": "INTEGER",
    "clv": "REAL",
    "quota_close": "REAL",
    "beat_close": "INTEGER",
    "quadro_agree_n": "INTEGER",
    "quadro_votes_n": "INTEGER",
}


def _quota_from_row(row: dict[str, Any], pick: str | None = None) -> float | None:
    """Estrae quota pick da quota_pick, odds dict, odd_1/x/2 o markets."""
    q = _float_or_none(row.get("quota_pick"))
    if q is not None:
        return q
    if not isinstance(row.get("odds"), bool):
        q = _float_or_none(row.get("odds"))
        if q is not None:
            return q
    odds = row.get("odds")
    pk = str(pick or row.get("pick") or "").strip().upper()
    code_map = {
        "1": ("odd_1", "1"),
        "X": ("odd_x", "X"),
        "2": ("odd_2", "2"),
        "O2.5": ("odd_over_25", "over_2.5"),
        "U2.5": ("odd_under_25", "under_2.5"),
    }
    if pk in code_map:
        flat_key, dict_key = code_map[pk]
        q = _float_or_none(row.get(flat_key))
        if q is not None:
            return q
        if isinstance(odds, dict):
            q = _float_or_none(odds.get(dict_key))
            if q is not None:
                return q
    if isinstance(odds, dict) and pk:
        for key in (pk, pk.lower(), f"over_{pk[1:]}" if pk.startswith("O") else None, f"under_{pk[1:]}" if pk.startswith("U") else None):
            if not key:
                continue
            q = _float_or_none(odds.get(key))
            if q is not None:
                return q
    markets = row.get("markets")
    if isinstance(markets, list):
        for m in markets:
            if isinstance(m, dict) and str(m.get("code") or "").upper() == pk:
                q = _float_or_none(m.get("odds"))
                if q is not None:
                    return q
    return _float_or_none(row.get("fair_odds"))


def _ensure_columns(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(matches)").fetchall()}
    for name, typ in _EXTRA_COLS.items():
        if name not in cols:
            conn.execute(f"ALTER TABLE matches ADD COLUMN {name} {typ}")


def _key(row: dict[str, Any]) -> str:
    return f"{row.get('date')}|{row.get('home')}|{row.get('away')}"


def _connect() -> sqlite3.Connection:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_CREATE)
    _ensure_columns(conn)
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


def _json_dump(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, str):
        return val
    try:
        return json.dumps(val, ensure_ascii=False)
    except Exception:
        return None


def _row_to_dict(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    for key in ("data_factors", "no_bet_reasons"):
        raw = d.get(key)
        if isinstance(raw, str) and raw.strip().startswith(("[", "{")):
            try:
                d[key] = json.loads(raw)
            except json.JSONDecodeError:
                pass
    return d


def _upsert(conn: sqlite3.Connection, rec: dict[str, Any], now: str, *, keep_result: bool) -> None:
    if not rec.get("match_key"):
        rec["match_key"] = _key(rec)
    for k, default in (
        ("time", ""),
        ("league", ""),
        ("country", ""),
        ("pick", None),
        ("action", None),
        ("score", None),
        ("score_unified", None),
        ("ev_cons", None),
        ("probability", None),
        ("odds_source", None),
        ("skip_reason", None),
        ("covered", 0),
        ("home_goals", None),
        ("away_goals", None),
        ("result", None),
        ("hit", None),
        ("saved_at", now),
        ("settled_at", None),
        ("quota_pick", None),
        ("agree_share", None),
        ("data_edge", None),
        ("move_rank", None),
        ("residual", None),
        ("adj_ev", None),
        ("data_factors", None),
        ("no_bet_reasons", None),
        ("pick_group", None),
        ("model_cluster", None),
        ("ev_sharp", None),
        ("context_partial", 0),
        ("synthetic_backfill", 0),
        ("clv", None),
        ("quota_close", None),
        ("beat_close", None),
        ("quadro_agree_n", None),
        ("quadro_votes_n", None),
    ):
        rec.setdefault(k, default)
    rec["data_factors"] = _json_dump(rec.get("data_factors"))
    rec["no_bet_reasons"] = _json_dump(rec.get("no_bet_reasons"))
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
            covered, home_goals, away_goals, result, hit, saved_at, settled_at,
            quota_pick, agree_share, data_edge, move_rank, residual, adj_ev,
            data_factors, no_bet_reasons, pick_group, model_cluster, ev_sharp,
            context_partial, synthetic_backfill, clv, quota_close, beat_close,
            quadro_agree_n, quadro_votes_n
        ) VALUES (
            :match_key, :date, :time, :home, :away, :league, :country, :pick, :action,
            :score, :score_unified, :ev_cons, :probability, :odds_source, :skip_reason,
            :covered, :home_goals, :away_goals, :result, :hit, :saved_at, :settled_at,
            :quota_pick, :agree_share, :data_edge, :move_rank, :residual, :adj_ev,
            :data_factors, :no_bet_reasons, :pick_group, :model_cluster, :ev_sharp,
            :context_partial, :synthetic_backfill, :clv, :quota_close, :beat_close,
            :quadro_agree_n, :quadro_votes_n
        )
        ON CONFLICT(match_key) DO UPDATE SET
            time=excluded.time, league=excluded.league, country=excluded.country,
            pick=excluded.pick, action=excluded.action, score=excluded.score,
            score_unified=excluded.score_unified, ev_cons=excluded.ev_cons,
            probability=excluded.probability, odds_source=excluded.odds_source,
            skip_reason=excluded.skip_reason, covered=excluded.covered,
            saved_at=excluded.saved_at,
            quota_pick=COALESCE(excluded.quota_pick, matches.quota_pick),
            agree_share=COALESCE(excluded.agree_share, matches.agree_share),
            data_edge=COALESCE(excluded.data_edge, matches.data_edge),
            move_rank=COALESCE(excluded.move_rank, matches.move_rank),
            residual=COALESCE(excluded.residual, matches.residual),
            adj_ev=COALESCE(excluded.adj_ev, matches.adj_ev),
            data_factors=COALESCE(excluded.data_factors, matches.data_factors),
            no_bet_reasons=COALESCE(excluded.no_bet_reasons, matches.no_bet_reasons),
            pick_group=COALESCE(excluded.pick_group, matches.pick_group),
            model_cluster=COALESCE(excluded.model_cluster, matches.model_cluster),
            ev_sharp=COALESCE(excluded.ev_sharp, matches.ev_sharp),
            clv=COALESCE(excluded.clv, matches.clv),
            quota_close=COALESCE(excluded.quota_close, matches.quota_close),
            beat_close=COALESCE(excluded.beat_close, matches.beat_close),
            quadro_agree_n=COALESCE(excluded.quadro_agree_n, matches.quadro_agree_n),
            quadro_votes_n=COALESCE(excluded.quadro_votes_n, matches.quadro_votes_n),
            context_partial=CASE
                WHEN matches.synthetic_backfill=0 AND excluded.synthetic_backfill=1 THEN matches.context_partial
                ELSE COALESCE(excluded.context_partial, matches.context_partial)
            END,
            synthetic_backfill=CASE
                WHEN matches.synthetic_backfill=0 THEN 0
                ELSE COALESCE(excluded.synthetic_backfill, matches.synthetic_backfill)
            END,
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
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def _action_label(action: Any) -> str:
    a = str(action or "").strip().lower()
    return {
        "gioca": "Gioca",
        "no_bet": "No bet",
        "invalido": "Invalido",
        "n/d": "N/D",
        "nd": "N/D",
    }.get(a, a or "—")


def _presa_label(hit: Any) -> str:
    if hit is None or hit == "":
        return "In attesa"
    try:
        return "Sì" if int(hit) == 1 else "No"
    except (TypeError, ValueError):
        return "In attesa"


def _pct_eu(val: Any) -> str:
    """Percentuale con virgola (Excel IT)."""
    if val is None or val == "":
        return ""
    try:
        x = float(val)
    except (TypeError, ValueError):
        return ""
    if abs(x) <= 1.5:
        x *= 100.0
    return f"{x:.1f}".replace(".", ",")


def _num_eu(val: Any, digits: int = 2) -> str:
    if val is None or val == "":
        return ""
    try:
        x = float(val)
    except (TypeError, ValueError):
        return ""
    return f"{x:.{digits}f}".replace(".", ",")


def export_plays_csv(*, path: Path | None = None, only_gioca: bool = True) -> dict[str, Any]:
    """Scrive storico_giocate.csv leggibile (IT), aggiornato dopo ogni settle."""
    out = path or PLAYS_CSV
    rows = load_history()
    if only_gioca:
        rows = [r for r in rows if str(r.get("action") or "").strip().lower() == "gioca"]
    rows = sorted(
        rows,
        key=lambda r: (
            str(r.get("date") or ""),
            str(r.get("time") or ""),
            str(r.get("home") or ""),
        ),
        reverse=True,
    )

    export_rows: list[dict[str, Any]] = []
    # Hit rate cumulata in ordine cronologico (dalla più vecchia)
    chrono = list(reversed(rows))
    run_n = run_hits = 0
    cum_by_key: dict[str, str] = {}
    for r in chrono:
        key = str(r.get("match_key") or f"{r.get('date')}|{r.get('home')}|{r.get('away')}|{r.get('pick')}")
        if r.get("hit") is None:
            cum_by_key[key] = ""
            continue
        run_n += 1
        run_hits += 1 if int(r.get("hit") or 0) == 1 else 0
        cum_by_key[key] = f"{run_hits}/{run_n} ({100.0 * run_hits / run_n:.0f}%)".replace(".", ",")

    for r in rows:
        key = str(r.get("match_key") or f"{r.get('date')}|{r.get('home')}|{r.get('away')}|{r.get('pick')}")
        hit = r.get("hit")
        quota = _float_or_none(r.get("quota_pick"))
        profit = ""
        if hit is not None and quota is not None:
            try:
                if int(hit) == 1:
                    profit = _num_eu(quota - 1.0)
                else:
                    profit = _num_eu(-1.0)
            except (TypeError, ValueError):
                profit = ""
        hg, ag = r.get("home_goals"), r.get("away_goals")
        scoreline = ""
        if hg is not None and ag is not None:
            scoreline = f"{int(hg)}-{int(ag)}"
        backfill = "Sì" if int(r.get("synthetic_backfill") or 0) else "No"
        export_rows.append(
            {
                "Data": str(r.get("date") or "")[:10],
                "Ora": str(r.get("time") or ""),
                "Casa": r.get("home") or "",
                "Trasferta": r.get("away") or "",
                "Campionato": r.get("league") or "",
                "Paese": r.get("country") or "",
                "Consiglio": r.get("pick") or "",
                "Mercato": r.get("pick_group") or "",
                "Azione": _action_label(r.get("action")),
                "Voto": r.get("score_unified") if r.get("score_unified") is not None else r.get("score") or "",
                "Quota": _num_eu(quota),
                "Probabilita_%": _pct_eu(r.get("probability")),
                "EV_%": _pct_eu(r.get("ev_cons") if r.get("ev_cons") is not None else r.get("ev")),
                "Risultato": scoreline,
                "Esito_1X2": r.get("result") or "",
                "Presa": _presa_label(hit),
                "Profitto_1u": profit,
                "Hit_rate_cumulata": cum_by_key.get(key, ""),
                "Fonte_quote": r.get("odds_source") or "",
                "Backfill": backfill,
                "Chiusa_il": str(r.get("settled_at") or "")[:19].replace("T", " "),
            }
        )

    df = pd.DataFrame(export_rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig", sep=";")
    settled = sum(1 for r in rows if r.get("hit") is not None)
    hits = sum(1 for r in rows if r.get("hit") is not None and int(r.get("hit") or 0) == 1)
    monthly = export_monthly_success_csv()
    return {
        "ok": True,
        "path": str(out),
        "n": len(export_rows),
        "n_settled": settled,
        "n_hits": hits,
        "hit_rate": round(hits / settled, 4) if settled else None,
        "monthly_success": monthly,
    }


def _score_unified_of(row: dict[str, Any]) -> int | None:
    for key in ("score_unified", "score"):
        v = row.get(key)
        if v is None or v == "":
            continue
        try:
            s = int(v)
            if 1 <= s <= 10:
                return s
        except (TypeError, ValueError):
            continue
    return None


def _month_of(row: dict[str, Any]) -> str | None:
    raw = str(row.get("date") or row.get("settled_at") or "")[:10]
    if len(raw) < 7:
        return None
    return raw[:7]


def _bucket_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    pending = [r for r in items if r.get("hit") is None]
    closed = [r for r in items if r.get("hit") is not None]
    hits = sum(1 for r in closed if int(r.get("hit") or 0) == 1)
    n_closed = len(closed)
    n_pending = len(pending)
    rate = round(hits / n_closed, 4) if n_closed else None
    return {
        "n_total": len(items),
        "n_closed": n_closed,
        "n_hits": hits,
        "n_misses": n_closed - hits,
        "n_pending": n_pending,
        "hit_rate": rate,
        "success_pct": "-" if rate is None else f"{rate * 100:.1f}".replace(".", ",") + "%",
        "label": f"{hits}/{n_closed}" if n_closed else "0/0",
    }


def export_monthly_success_csv(*, path: Path | None = None) -> dict[str, Any]:
    """CSV mese × voto 7/8/9/10: % successo delle sole giocate consigliate (action=gioca)."""
    out = path or MONTHLY_SUCCESS_CSV
    rows = [
        r
        for r in load_history()
        if str(r.get("action") or "").strip().lower() == "gioca"
        and (_score_unified_of(r) in HIGH_VOTES)
    ]

    by_month: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        month = _month_of(r)
        if not month:
            continue
        by_month.setdefault(month, []).append(r)

    export_rows: list[dict[str, Any]] = []
    for month in sorted(by_month.keys(), reverse=True):
        month_rows = by_month[month]
        for vote in HIGH_VOTES:
            subset = [r for r in month_rows if _score_unified_of(r) == vote]
            st = _bucket_stats(subset)
            export_rows.append(
                {
                    "Mese": month,
                    "Voto_unificato": vote,
                    "Giocate_totali": st["n_total"],
                    "Chiuse": st["n_closed"],
                    "Prese": st["n_hits"],
                    "Sbagliate": st["n_misses"],
                    "In_attesa": st["n_pending"],
                    "Successo_%": st["success_pct"],
                    "Dettaglio": st["label"] if st["n_closed"] else "nessuna chiusa",
                }
            )
        all7 = _bucket_stats(month_rows)
        export_rows.append(
            {
                "Mese": month,
                "Voto_unificato": "7-10 insieme",
                "Giocate_totali": all7["n_total"],
                "Chiuse": all7["n_closed"],
                "Prese": all7["n_hits"],
                "Sbagliate": all7["n_misses"],
                "In_attesa": all7["n_pending"],
                "Successo_%": all7["success_pct"],
                "Dettaglio": all7["label"] if all7["n_closed"] else "nessuna chiusa",
            }
        )

    # Riepilogo globale
    for vote in HIGH_VOTES:
        subset = [r for r in rows if _score_unified_of(r) == vote]
        st = _bucket_stats(subset)
        export_rows.append(
            {
                "Mese": "TOTALE",
                "Voto_unificato": vote,
                "Giocate_totali": st["n_total"],
                "Chiuse": st["n_closed"],
                "Prese": st["n_hits"],
                "Sbagliate": st["n_misses"],
                "In_attesa": st["n_pending"],
                "Successo_%": st["success_pct"],
                "Dettaglio": st["label"] if st["n_closed"] else "nessuna chiusa",
            }
        )
    tot = _bucket_stats(rows)
    export_rows.append(
        {
            "Mese": "TOTALE",
            "Voto_unificato": "7-10 insieme",
            "Giocate_totali": tot["n_total"],
            "Chiuse": tot["n_closed"],
            "Prese": tot["n_hits"],
            "Sbagliate": tot["n_misses"],
            "In_attesa": tot["n_pending"],
            "Successo_%": tot["success_pct"],
            "Dettaglio": tot["label"] if tot["n_closed"] else "nessuna chiusa",
        }
    )

    # Tabella larga mese × % per voto (più comoda in Excel)
    wide_rows: list[dict[str, Any]] = []
    for month in sorted(by_month.keys(), reverse=True):
        month_rows = by_month[month]
        wide: dict[str, Any] = {"Mese": month}
        for vote in HIGH_VOTES:
            st = _bucket_stats([r for r in month_rows if _score_unified_of(r) == vote])
            wide[f"Voto_{vote}_chiuse"] = st["n_closed"]
            wide[f"Voto_{vote}_prese"] = st["n_hits"]
            wide[f"Voto_{vote}_successo_%"] = st["success_pct"]
            wide[f"Voto_{vote}_in_attesa"] = st["n_pending"]
        all7 = _bucket_stats(month_rows)
        wide["Totale_7_10_chiuse"] = all7["n_closed"]
        wide["Totale_7_10_prese"] = all7["n_hits"]
        wide["Totale_7_10_successo_%"] = all7["success_pct"]
        wide["Totale_7_10_in_attesa"] = all7["n_pending"]
        wide_rows.append(wide)

    wide_tot: dict[str, Any] = {"Mese": "TOTALE"}
    for vote in HIGH_VOTES:
        st = _bucket_stats([r for r in rows if _score_unified_of(r) == vote])
        wide_tot[f"Voto_{vote}_chiuse"] = st["n_closed"]
        wide_tot[f"Voto_{vote}_prese"] = st["n_hits"]
        wide_tot[f"Voto_{vote}_successo_%"] = st["success_pct"]
        wide_tot[f"Voto_{vote}_in_attesa"] = st["n_pending"]
    wide_tot["Totale_7_10_chiuse"] = tot["n_closed"]
    wide_tot["Totale_7_10_prese"] = tot["n_hits"]
    wide_tot["Totale_7_10_successo_%"] = tot["success_pct"]
    wide_tot["Totale_7_10_in_attesa"] = tot["n_pending"]
    wide_rows.append(wide_tot)

    out.parent.mkdir(parents=True, exist_ok=True)
    # Un solo CSV chiaro: una riga = mese + voto (7/8/9/10 + totale)
    long_df = pd.DataFrame(export_rows)
    long_df.to_csv(out, index=False, encoding="utf-8-sig", sep=";")
    # Vista larga separata (una riga = un mese) — comoda per confronti
    wide_path = out.with_name("storico_successo_mensile_largo.csv")
    pd.DataFrame(wide_rows).to_csv(wide_path, index=False, encoding="utf-8-sig", sep=";")

    return {
        "ok": True,
        "path": str(out),
        "wide_path": str(wide_path),
        "n_months": len(by_month),
        "n_plays": len(rows),
        "n_closed": tot["n_closed"],
        "n_hits": tot["n_hits"],
        "hit_rate": tot["hit_rate"],
    }


def archive_upcoming(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Salva/aggiorna tutte le partite del calendario, comprese le N/D (logging ricco)."""
    conn = _connect()
    try:
        _migrate_jsonl(conn)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        added = 0
        from modules.data_update.team_names import resolve_known_team

        for row in rows:
            home = resolve_known_team(row.get("home") or "") or row.get("home")
            away = resolve_known_team(row.get("away") or "") or row.get("away")
            pred = row.get("prediction") if isinstance(row.get("prediction"), dict) else {}
            ds = row.get("data_signal") or pred.get("data_signal") or {}
            sa = row.get("source_agreement") or {}
            if not sa and isinstance(row.get("play"), dict):
                sa = row["play"].get("source_agreement") or {}
            residual = row.get("residual_ev") or {}
            if not residual and isinstance(row.get("play"), dict):
                residual = row["play"].get("residual_ev") or {}
            move = row.get("market_move") or {}
            move_rank = None
            try:
                from modules.data_update.asian_odds import MOVE_RANK

                move_rank = MOVE_RANK.get(move.get("movement_level") or "Stabile", 0)
            except Exception:
                move_rank = row.get("move_rank")
            quota = _quota_from_row(row, pick=row.get("pick"))
            clv_v, qc, beat = _clv_archive_fields(row, pick=row.get("pick"), quota=quota)
            factors = None
            if isinstance(ds, dict):
                factors = ds.get("factors")
            ds_ready = bool(isinstance(ds, dict) and ds.get("ready"))
            agree_v = sa.get("agree_share") if isinstance(sa, dict) else row.get("agree_share")
            quadro = row.get("quadro") if isinstance(row.get("quadro"), dict) else {}
            if not quadro and isinstance(row.get("prediction"), dict):
                quadro = (row["prediction"].get("quadro") or {}) if row.get("prediction") else {}
            q_an = quadro.get("agree_n") or row.get("quadro_agree_n")
            q_vn = quadro.get("votes_n") or row.get("quadro_votes_n")
            try:
                q_an = int(q_an) if q_an is not None else None
            except (TypeError, ValueError):
                q_an = None
            try:
                q_vn = int(q_vn) if q_vn is not None else None
            except (TypeError, ValueError):
                q_vn = None
            context_partial = 0 if (ds_ready and quota and factors and agree_v is not None) else 1
            try:
                su_int = int(row.get("score_unified")) if row.get("score_unified") is not None else None
            except (TypeError, ValueError):
                su_int = None
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
                "score_unified": su_int,
                "ev_cons": row.get("ev_cons"),
                "ev_sharp": row.get("ev_sharp"),
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
                "quota_pick": quota,
                "agree_share": sa.get("agree_share") if isinstance(sa, dict) else row.get("agree_share"),
                "data_edge": ds.get("edge") if isinstance(ds, dict) else row.get("data_edge"),
                "move_rank": move_rank,
                "residual": residual.get("residual") if isinstance(residual, dict) else row.get("residual"),
                "adj_ev": residual.get("adj_ev") if isinstance(residual, dict) else row.get("adj_ev"),
                "data_factors": factors,
                "no_bet_reasons": row.get("no_bet_reasons"),
                "pick_group": row.get("pick_group"),
                "model_cluster": pred.get("model_cluster") or row.get("model_cluster"),
                "context_partial": int(context_partial),
                "synthetic_backfill": 0,
                "clv": clv_v,
                "quota_close": qc,
                "beat_close": beat,
                "quadro_agree_n": q_an,
                "quadro_votes_n": q_vn,
            }
            exists = conn.execute("SELECT 1 FROM matches WHERE match_key=?", (rec["match_key"],)).fetchone()
            if not exists:
                added += 1
            _upsert(conn, rec, now, keep_result=True)
        conn.commit()
        n = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        return {"n_history": int(n), "added": added, "updated": len(rows) - added, "path": str(DB), "rich": True}
    finally:
        conn.close()


def _float_or_none(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
        if x >= 1.01:
            return x
    except (TypeError, ValueError):
        pass
    return None


def _quota_close_from_move(row: dict[str, Any], pick: str | None) -> float | None:
    move = row.get("market_move") or {}
    if not move:
        return None
    pk = str(pick or row.get("pick") or "").strip().upper()
    grp = str(row.get("pick_group") or "1x2").lower()
    if pk == "1":
        return _float_or_none(move.get("odd_1"))
    if pk == "X":
        return _float_or_none(move.get("odd_x"))
    if pk == "2":
        return _float_or_none(move.get("odd_2"))
    if grp == "ou" or pk.startswith("O"):
        if "U" in pk and not pk.startswith("O"):
            return _float_or_none(move.get("odd_under"))
        if pk.startswith("O") and "GOL" not in pk:
            return _float_or_none(move.get("odd_over"))
    if pk.startswith("U"):
        return _float_or_none(move.get("odd_under"))
    return None


def _clv_archive_fields(
    row: dict[str, Any],
    *,
    pick: str | None,
    quota: float | None,
) -> tuple[float | None, float | None, int | None]:
    from modules.advisor.staking import beat_close, clv_prob

    play = row.get("play") if isinstance(row.get("play"), dict) else {}
    clv_v = row.get("clv")
    if clv_v is None:
        clv_v = play.get("clv")
    try:
        clv_v = float(clv_v) if clv_v is not None else None
    except (TypeError, ValueError):
        clv_v = None
    qc = _float_or_none(row.get("quota_close")) or _quota_close_from_move(row, pick)
    if clv_v is None and quota and qc:
        clv_v = clv_prob(float(quota), float(qc))
    beat: int | None = None
    if row.get("beat_close") is not None:
        beat = int(row.get("beat_close") or 0)
    elif quota and qc:
        bc = beat_close(float(quota), float(qc))
        if bc is not None:
            beat = 1 if bc else 0
    return clv_v, qc, beat


def _side_stats_needed(pick: str) -> bool:
    p = str(pick or "").strip().upper()
    return p.startswith("CARD") or p.startswith("CORN")


def _int_stat(v: Any) -> int | None:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _hit_for_pick(
    pick: str,
    *,
    res: str,
    hg: int,
    ag: int,
    tot: int,
    hy: int | None = None,
    ay: int | None = None,
    hr: int | None = None,
    ar: int | None = None,
    hc: int | None = None,
    ac: int | None = None,
) -> int | None:
    """Valuta hit su gol, cartellini e corner. None se stats side mancanti."""
    p = str(pick or "").strip().upper().replace(" ", "")
    if p in {"1", "X", "2"}:
        return 1 if p == res else 0
    if p in {"1X", "12", "X2"}:
        ok = (p == "1X" and res in {"1", "X"}) or (p == "12" and res in {"1", "2"}) or (p == "X2" and res in {"X", "2"})
        return 1 if ok else 0
    if p in {"GOL", "BTTS", "BTTSYES"}:
        return 1 if hg > 0 and ag > 0 else 0
    if p in {"NOGOL", "BTTSNO", "NG"}:
        return 1 if hg == 0 or ag == 0 else 0
    # Over/Under totali: O2.5 U1.5 …
    if p.startswith("O") and p[1:].replace(".", "", 1).isdigit():
        try:
            line = float(p[1:])
            return 1 if tot > line else 0
        except ValueError:
            pass
    if p.startswith("U") and p[1:].replace(".", "", 1).isdigit():
        try:
            line = float(p[1:])
            return 1 if tot < line else 0
        except ValueError:
            pass
    # Multigol MG1-2, MG2-3, MG3+, …
    if p.startswith("MG"):
        body = p[2:].replace("PLUS", "+")
        if body.endswith("+") or body.endswith("PLUS"):
            try:
                n = int("".join(ch for ch in body if ch.isdigit()))
                return 1 if tot >= n else 0
            except ValueError:
                return 0
        if "-" in body:
            a, b = body.split("-", 1)
            try:
                lo, hi = int(a), int(b)
                return 1 if lo <= tot <= hi else 0
            except ValueError:
                return 0
    if p in {"PARI", "EVEN", "GOALSEVEN"}:
        return 1 if tot % 2 == 0 else 0
    if p in {"DISPARI", "ODD", "GOALSODD"}:
        return 1 if tot % 2 == 1 else 0
    if p in {"AH01", "AH0H", "AHHOME0"}:
        return 1 if hg > ag else 0
    if p in {"AH02", "AH0A", "AHAWAY0"}:
        return 1 if ag > hg else 0
    # Cartellini CARDO3.5 / CARDU3.5
    if p.startswith("CARDO") or p.startswith("CARDU"):
        if hy is None or ay is None:
            return None
        cards = int(hy) + int(ay) + int(hr or 0) + int(ar or 0)
        body = p[5:].replace("PLUS", "+")
        try:
            line = float(body)
        except ValueError:
            return 0
        if p.startswith("CARDO"):
            return 1 if cards > line else 0
        return 1 if cards < line else 0
    # Corner CORNO8.5 / CORNU8.5
    if p.startswith("CORNO") or p.startswith("CORNU"):
        if hc is None or ac is None:
            return None
        corners = int(hc) + int(ac)
        body = p[5:].replace("PLUS", "+")
        try:
            line = float(body)
        except ValueError:
            return 0
        if p.startswith("CORNO"):
            return 1 if corners > line else 0
        return 1 if corners < line else 0
    # Exact score 2-1
    if "-" in p and p[0].isdigit():
        try:
            a, b = p.split("-", 1)
            return 1 if int(a) == hg and int(b) == ag else 0
        except ValueError:
            pass
    # Se pick non riconosciuto ma era 1X2-like fallito
    return 0 if pick else 0


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
            tot = int(hg) + int(ag)
            hy = _int_stat(fx.get("home_yellow") if "home_yellow" in fx.index else fx.get("HY"))
            ay = _int_stat(fx.get("away_yellow") if "away_yellow" in fx.index else fx.get("AY"))
            hr = _int_stat(fx.get("home_red") if "home_red" in fx.index else fx.get("HR"))
            ar = _int_stat(fx.get("away_red") if "away_red" in fx.index else fx.get("AR"))
            hc = _int_stat(fx.get("home_corners") if "home_corners" in fx.index else fx.get("HC"))
            ac = _int_stat(fx.get("away_corners") if "away_corners" in fx.index else fx.get("AC"))
            if _side_stats_needed(pick) and (
                (pick.upper().startswith("CARD") and (hy is None or ay is None))
                or (pick.upper().startswith("CORN") and (hc is None or ac is None))
            ):
                continue
            hit = _hit_for_pick(
                pick,
                res=res,
                hg=int(hg),
                ag=int(ag),
                tot=tot,
                hy=hy,
                ay=ay,
                hr=hr,
                ar=ar,
                hc=hc,
                ac=ac,
            )
            if hit is None:
                continue
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


def _fetch_world_results(*, days_back: int = 3) -> pd.DataFrame:
    """Scarica i risultati degli ultimi N giorni da TheSportsDB e API-Football."""
    import json
    import os
    from datetime import date, timedelta
    from urllib.request import Request, urlopen

    UA = "Mozilla/5.0 (compatible; football-predictor/1.0; +local)"
    TSDB_BASE = "https://www.thesportsdb.com/api/v1/json"
    APIF_BASE = "https://v3.football.api-sports.io"

    def _tsdb_key() -> str:
        for k in ("THESPORTSDB_API_KEY", "THESPORTSDB_KEY"):
            val = (os.environ.get(k) or "").strip()
            if val and not (len(val) == 32 and all(c in "0123456789abcdefABCDEF" for c in val)):
                return val
        p = ROOT / "data" / "raw" / "thesportsdb.key"
        if p.exists():
            val = p.read_text(encoding="utf-8").strip()
            if val and not (len(val) == 32 and all(c in "0123456789abcdefABCDEF" for c in val)):
                return val
        return "123"

    def _apif_key() -> str | None:
        for k in ("API_FOOTBALL_KEY", "APISPORTS_KEY"):
            val = (os.environ.get(k) or "").strip()
            if val:
                return val
        p = ROOT / "data" / "raw" / "api-football.key"
        if p.exists():
            val = p.read_text(encoding="utf-8").strip()
            if val:
                return val
        return None

    from modules.data_update.team_names import resolve_known_team

    rows: list[dict] = []
    today = date.today()
    tsdb_key = _tsdb_key()
    apif_key = _apif_key()

    for i in range(1, days_back + 1):
        day = today - timedelta(days=i)
        day_s = day.isoformat()

        # TheSportsDB: eventsday restituisce anche partite con risultati
        try:
            url = f"{TSDB_BASE}/{tsdb_key}/eventsday.php?d={day_s}&s=Soccer"
            req = Request(url, headers={"User-Agent": UA})
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            for ev in data.get("events") or []:
                home_raw = str(ev.get("strHomeTeam") or "").strip()
                away_raw = str(ev.get("strAwayTeam") or "").strip()
                hg = ev.get("intHomeScore")
                ag = ev.get("intAwayScore")
                if not home_raw or not away_raw or hg is None or ag is None:
                    continue
                try:
                    hg, ag = int(hg), int(ag)
                except (TypeError, ValueError):
                    continue
                rows.append({
                    "date": pd.Timestamp(day_s),
                    "home_team": resolve_known_team(home_raw) or home_raw,
                    "away_team": resolve_known_team(away_raw) or away_raw,
                    "home_goals": hg,
                    "away_goals": ag,
                })
        except Exception:
            pass

        # API-Football: risultati del giorno
        if apif_key:
            try:
                from urllib.parse import urlencode
                q = urlencode({"date": day_s})
                req = Request(
                    f"{APIF_BASE}/fixtures?{q}",
                    headers={"User-Agent": UA, "x-apisports-key": apif_key},
                )
                with urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                for item in data.get("response") or []:
                    fx = item.get("fixture") or {}
                    status = str((fx.get("status") or {}).get("short") or "")
                    if status not in {"FT", "AET", "PEN"}:
                        continue
                    teams = item.get("teams") or {}
                    goals = item.get("goals") or {}
                    home_raw = str((teams.get("home") or {}).get("name") or "").strip()
                    away_raw = str((teams.get("away") or {}).get("name") or "").strip()
                    hg = goals.get("home")
                    ag = goals.get("away")
                    if not home_raw or not away_raw or hg is None or ag is None:
                        continue
                    try:
                        hg, ag = int(hg), int(ag)
                    except (TypeError, ValueError):
                        continue
                    rows.append({
                        "date": pd.Timestamp(day_s),
                        "home_team": resolve_known_team(home_raw) or home_raw,
                        "away_team": resolve_known_team(away_raw) or away_raw,
                        "home_goals": hg,
                        "away_goals": ag,
                    })
            except Exception:
                pass

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    return df


def settle_pending() -> dict[str, Any]:
    """Chiude i match archiviati usando coppe (org), football-data.co.uk e risultati mondiali."""
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
    # Fonti mondiali: TheSportsDB + API-Football per i risultati degli ultimi giorni
    # Coprono le 700+ partite internazionali non presenti nei CSV football-data.co.uk
    try:
        world = _fetch_world_results(days_back=3)
        if not world.empty:
            n = int(settle_from_results(world).get("settled") or 0)
            settled += n
            if n:
                print(f"storico locale: {n} partite chiuse da fonti mondiali (TSDB/API-Football)")
    except Exception as exc:
        print(f"skip world results settle: {exc}")
    summary = history_summary()
    summary["settled"] = settled
    # Apprendimento continuo da esiti chiusi (bins, residual, pesi, soglie)
    try:
        from modules.advisor.online_learn import learn_from_settled

        learn = learn_from_settled()
        summary["online_learn"] = {
            k: learn.get(k) for k in ("ok", "n_settled", "error", "fitted_at") if k in learn
        }
        summary["online_learn_steps"] = learn.get("steps")
    except Exception as exc:
        summary["online_learn_error"] = str(exc)
    try:
        from modules.advisor.analysis_outcomes import refresh_analysis_outcomes

        ao = refresh_analysis_outcomes()
        summary["analysis_outcomes"] = {
            "n_in_pool": ao.get("n_in_pool"),
            "n_with_score_unified": ao.get("n_with_score_unified"),
            "updated_at": ao.get("updated_at"),
        }
    except Exception as exc:
        summary["analysis_outcomes_error"] = str(exc)
    try:
        summary["plays_csv"] = export_plays_csv()
    except Exception as exc:
        summary["plays_csv_error"] = str(exc)
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


def _history_weight(n_global: int, n_league: int = 0) -> float:
    w = HISTORY_WEIGHT
    if int(n_global) >= MIN_GLOBAL_BOOST:
        w = 0.15
    if int(n_league) >= MIN_LEAGUE_SETTLED:
        w += 0.03
    return min(HISTORY_WEIGHT_MAX, w)


def lookup_history_match(home: str, away: str, league: str | None = None) -> dict[str, Any]:
    """Segnale per il quadro/voto: pronto solo dopo abbastanza esiti locali."""
    empty = {"ready": False, "n_global": 0, "n_league": 0, "home": None, "away": None, "weight": HISTORY_WEIGHT}
    if not DB.exists() and not JSONL.exists():
        return empty
    from modules.data_update.team_names import resolve_known_team

    home = resolve_known_team(home) or home
    away = resolve_known_team(away) or away
    conn = _connect()
    try:
        _migrate_jsonl(conn)
        n_global = conn.execute("SELECT COUNT(*) FROM matches WHERE result IS NOT NULL").fetchone()[0]
        n_league = 0
        lg = str(league or "").strip()
        if lg:
            n_league = conn.execute(
                "SELECT COUNT(*) FROM matches WHERE result IS NOT NULL AND league = ?",
                (lg,),
            ).fetchone()[0]
        h = _team_form(conn, home)
        a = _team_form(conn, away)
        ready = int(n_global) >= MIN_GLOBAL_SETTLED and h is not None and a is not None
        return {
            "ready": ready,
            "n_global": int(n_global),
            "n_league": int(n_league),
            "league": lg or None,
            "min_team": MIN_TEAM_MATCHES,
            "min_global": MIN_GLOBAL_SETTLED,
            "weight": _history_weight(int(n_global), int(n_league)),
            "home": h,
            "away": a,
        }
    finally:
        conn.close()


def enrich_clv_from_matches_csv(*, max_rows: int = 2000) -> dict[str, Any]:
    """Compila clv/quota_close da open→close in matches.csv (righe senza CLV)."""
    from modules.advisor.staking import beat_close, clv_prob

    path = PROCESSED / "matches.csv"
    if not path.is_file():
        return {"ok": False, "updated": 0, "error": "matches.csv assente"}
    df = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    if df.empty:
        return {"ok": False, "updated": 0}

    def _f(row, *cols):
        for c in cols:
            if c in row.index and pd.notna(row[c]):
                try:
                    v = float(row[c])
                    if v >= 1.01:
                        return v
                except (TypeError, ValueError):
                    continue
        return None

    idx: dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        try:
            day = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
            h = str(row["home_team"]).strip()
            a = str(row["away_team"]).strip()
            idx[f"{day}|{h}|{a}"] = row
        except (TypeError, ValueError, KeyError):
            continue

    conn = _connect()
    updated = 0
    try:
        rows = conn.execute(
            "SELECT match_key, date, home, away, pick, quota_pick, clv FROM matches "
            "WHERE hit IS NOT NULL AND (clv IS NULL OR quota_close IS NULL) LIMIT ?",
            (int(max_rows),),
        ).fetchall()
        for rec in rows:
            pick = str(rec["pick"] or "").upper()
            if pick not in {"1", "X", "2"}:
                continue
            key = f"{rec['date']}|{rec['home']}|{rec['away']}"
            row = idx.get(key)
            if row is None:
                continue
            if pick == "1":
                open_o = _f(row, "odd_home", "odd_home_close")
                close_o = _f(row, "odd_home_close", "odd_home")
            elif pick == "X":
                open_o = _f(row, "odd_draw", "odd_draw_close")
                close_o = _f(row, "odd_draw_close", "odd_draw")
            else:
                open_o = _f(row, "odd_away", "odd_away_close")
                close_o = _f(row, "odd_away_close", "odd_away")
            bet = open_o or rec["quota_pick"]
            if not bet or not close_o:
                continue
            clv = clv_prob(float(bet), float(close_o))
            bc = beat_close(float(bet), float(close_o))
            beat = 1 if bc else 0 if bc is False else None
            conn.execute(
                "UPDATE matches SET clv=?, quota_close=?, beat_close=? WHERE match_key=?",
                (clv, close_o, beat, rec["match_key"]),
            )
            updated += 1
        conn.commit()
    finally:
        conn.close()
    return {"ok": True, "updated": updated}


def history_summary() -> dict[str, Any]:
    conn = _connect()
    try:
        _migrate_jsonl(conn)
        n = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        settled = conn.execute("SELECT COUNT(*) FROM matches WHERE result IS NOT NULL").fetchone()[0]
        nd = conn.execute("SELECT COUNT(*) FROM matches WHERE action IN ('n/d', 'invalido')").fetchone()[0]
        rich = conn.execute(
            """
            SELECT COUNT(*) FROM matches
            WHERE result IS NOT NULL
              AND quota_pick IS NOT NULL
              AND (ev_cons IS NOT NULL OR ev_sharp IS NOT NULL)
              AND data_factors IS NOT NULL
              AND agree_share IS NOT NULL
            """
        ).fetchone()[0]
        synth = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE synthetic_backfill=1"
        ).fetchone()[0]
        return {
            "n_history": int(n),
            "n_settled": int(settled),
            "n_nd": int(nd),
            "n_rich": int(rich),
            "n_rich_target": 80,
            "n_synthetic": int(synth),
            "ready": int(settled) >= MIN_GLOBAL_SETTLED,
            "min_global": MIN_GLOBAL_SETTLED,
            "min_team": MIN_TEAM_MATCHES,
            "weight": _history_weight(int(settled), 0),
            "path": str(DB),
        }
    finally:
        conn.close()
