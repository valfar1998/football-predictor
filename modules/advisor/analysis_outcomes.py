"""Storico esiti analisi: hit rate per voto unificato, azione, mercato + andamento giocate.

Si aggiorna automaticamente dopo ogni settle (e backfill).
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "data" / "processed" / "analysis_outcomes.json"


def _score_int(rec: dict[str, Any], key: str) -> int | None:
    v = rec.get(key)
    if v is None:
        return None
    try:
        s = int(v)
        if 1 <= s <= 10:
            return s
    except (TypeError, ValueError):
        pass
    return None


def _valid_quota(val: Any) -> float | None:
    if isinstance(val, bool) or val is None:
        return None
    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    return num if num >= 1.01 else None


def _bucket_rows(
    rows: list[dict[str, Any]],
    key_fn,
    *,
    min_n: int = 1,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        k = key_fn(r)
        if k is None:
            continue
        groups[str(k)].append(r)
    out: list[dict] = []
    for k, items in groups.items():
        n = len(items)
        if n < min_n:
            continue
        hits = sum(1 for x in items if int(x.get("hit") or 0) == 1)
        out.append(
            {
                "key": k,
                "n": n,
                "hits": hits,
                "misses": n - hits,
                "hit_rate": round(hits / n, 4) if n else 0.0,
                "label": f"{hits}/{n}",
            }
        )

    def _sort_key(row: dict) -> tuple:
        k = row["key"]
        try:
            return (0, -int(k))
        except ValueError:
            return (1, k)

    return sorted(out, key=_sort_key)


def _recent_rows(rows: list[dict[str, Any]], *, limit: int = 25) -> list[dict[str, Any]]:
    chunk = sorted(rows, key=lambda r: str(r.get("settled_at") or r.get("date") or ""))[-limit:]
    out = []
    for r in reversed(chunk):
        su = _score_int(r, "score_unified")
        sc = _score_int(r, "score")
        quota = _valid_quota(r.get("quota_pick"))
        out.append(
            {
                "date": r.get("date"),
                "match": f"{r.get('home')} – {r.get('away')}",
                "pick": r.get("pick"),
                "action": r.get("action"),
                "quota": quota,
                "score_unified": su,
                "score": sc,
                "hit": int(r.get("hit") or 0),
                "result": r.get("result"),
                "scoreline": (
                    f"{r.get('home_goals')}-{r.get('away_goals')}"
                    if r.get("home_goals") is not None
                    else None
                ),
                "live": not int(r.get("synthetic_backfill") or 0),
                "quadro": (
                    f"{int(r['quadro_agree_n'])}/{int(r['quadro_votes_n'])}"
                    if r.get("quadro_agree_n") is not None and r.get("quadro_votes_n")
                    else None
                ),
                "trainable": bool(
                    quota is not None
                    and (r.get("ev_cons") is not None or r.get("ev_sharp") is not None)
                    and r.get("data_factors")
                    and r.get("agree_share") is not None
                ),
            }
        )
    return out


def _week_key(rec: dict[str, Any]) -> str | None:
    raw = str(rec.get("date") or rec.get("settled_at") or "")[:10]
    if len(raw) < 10:
        return None
    try:
        day = datetime.fromisoformat(raw).date()
    except ValueError:
        return None
    iso = day.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _gioca_progress(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Hit rate delle sole giocate consigliate, con andamento settimanale e rolling."""
    gioca = [
        r
        for r in rows
        if str(r.get("action") or "").strip().lower() == "gioca" and r.get("hit") is not None
    ]
    gioca = sorted(gioca, key=lambda r: str(r.get("date") or r.get("settled_at") or ""))
    n = len(gioca)
    hits = sum(1 for r in gioca if int(r.get("hit") or 0) == 1)
    overall = round(hits / n, 4) if n else None

    week_groups: dict[str, list[dict]] = defaultdict(list)
    for r in gioca:
        wk = _week_key(r)
        if wk:
            week_groups[wk].append(r)
    by_week: list[dict[str, Any]] = []
    for wk in sorted(week_groups):
        items = week_groups[wk]
        h = sum(1 for x in items if int(x.get("hit") or 0) == 1)
        m = len(items)
        by_week.append(
            {
                "key": wk,
                "n": m,
                "hits": h,
                "misses": m - h,
                "hit_rate": round(h / m, 4) if m else 0.0,
                "label": f"{h}/{m}",
            }
        )

    def _window(k: int) -> dict[str, Any] | None:
        if n < 1:
            return None
        chunk = gioca[-k:] if n >= k else gioca
        h = sum(1 for r in chunk if int(r.get("hit") or 0) == 1)
        m = len(chunk)
        return {"n": m, "hits": h, "hit_rate": round(h / m, 4) if m else None, "label": f"{h}/{m}"}

    last10 = _window(10)
    last20 = _window(20)
    prev = None
    if n >= 20:
        older = gioca[:-10]
        h = sum(1 for r in older if int(r.get("hit") or 0) == 1)
        m = len(older)
        prev = {"n": m, "hits": h, "hit_rate": round(h / m, 4) if m else None, "label": f"{h}/{m}"}

    trend = None
    if last10 and last10.get("hit_rate") is not None and prev and prev.get("hit_rate") is not None:
        delta = float(last10["hit_rate"]) - float(prev["hit_rate"])
        if delta >= 0.05:
            trend = "in miglioramento"
        elif delta <= -0.05:
            trend = "in calo"
        else:
            trend = "stabile"

    cumulative: list[dict[str, Any]] = []
    run_h = 0
    for i, r in enumerate(gioca, start=1):
        run_h += 1 if int(r.get("hit") or 0) == 1 else 0
        cumulative.append(
            {
                "n": i,
                "date": r.get("date"),
                "hit_rate": round(run_h / i, 4),
                "label": f"{run_h}/{i}",
            }
        )

    high = [r for r in gioca if (_score_int(r, "score_unified") or 0) >= 8]
    high_hits = sum(1 for r in high if int(r.get("hit") or 0) == 1)

    return {
        "n": n,
        "hits": hits,
        "misses": n - hits,
        "hit_rate": overall,
        "label": f"{hits}/{n}" if n else "0/0",
        "trend": trend,
        "last_10": last10,
        "last_20": last20,
        "before_last_10": prev,
        "by_week": by_week,
        "cumulative": cumulative[-60:],
        "high_vote": {
            "n": len(high),
            "hits": high_hits,
            "hit_rate": round(high_hits / len(high), 4) if high else None,
            "label": f"{high_hits}/{len(high)}" if high else "0/0",
        },
    }


def build_analysis_outcomes_report(
    *,
    live_only: bool = False,
    trainable_only: bool = False,
    require_score_unified: bool = False,
) -> dict[str, Any]:
    from modules.data_update.history import load_history
    from modules.advisor.learn_policy import is_live, is_trainable

    all_rows = load_history()
    settled = [r for r in all_rows if r.get("hit") is not None]
    pool = list(settled)
    if live_only:
        pool = [r for r in pool if is_live(r)]
    if trainable_only:
        pool = [r for r in pool if is_trainable(r)]
    if require_score_unified:
        pool = [r for r in pool if _score_int(r, "score_unified") is not None]

    with_unified = [r for r in pool if _score_int(r, "score_unified") is not None]
    with_score = [r for r in pool if _score_int(r, "score") is not None]

    by_unified = _bucket_rows(with_unified, lambda r: _score_int(r, "score_unified"))
    by_score = _bucket_rows(with_score, lambda r: _score_int(r, "score"))
    by_action = _bucket_rows(pool, lambda r: str(r.get("action") or "n/d").lower())
    by_market = _bucket_rows(pool, lambda r: str(r.get("pick_group") or "1x2").lower())
    by_quadro = _bucket_rows(
        pool,
        lambda r: (
            f"{int(r['quadro_agree_n'])}/{int(r['quadro_votes_n'])}"
            if r.get("quadro_agree_n") is not None and r.get("quadro_votes_n")
            else None
        ),
    )
    by_unified_action = _bucket_rows(
        with_unified,
        lambda r: f"{_score_int(r, 'score_unified')}|{str(r.get('action') or 'n/d').lower()}",
        min_n=1,
    )

    gioca = _gioca_progress(pool)

    highlights: list[str] = []
    if gioca["n"] >= 1:
        rate = f"{gioca['hit_rate']:.0%}" if gioca["hit_rate"] is not None else "—"
        line = f"Giocate consigliate: {gioca['label']} ({rate})"
        if gioca.get("trend"):
            line += f" · trend {gioca['trend']}"
        highlights.append(line)
        if gioca.get("last_10") and gioca["last_10"]["n"] >= 5:
            highlights.append(
                f"Ultime {gioca['last_10']['n']} giocate: {gioca['last_10']['label']} "
                f"({gioca['last_10']['hit_rate']:.0%})"
            )
        hv = gioca.get("high_vote") or {}
        if hv.get("n", 0) >= 1:
            hr = hv.get("hit_rate")
            highlights.append(
                f"Giocate voto ≥8: {hv['label']}"
                + (f" ({hr:.0%})" if hr is not None else "")
            )
    for row in by_unified:
        if int(row["key"]) >= 8 and row["n"] >= 2:
            highlights.append(
                f"Voto unificato {row['key']}: {row['label']} prese ({row['hit_rate']:.0%})"
            )
    for row in by_quadro:
        if str(row.get("key", "")).startswith("10/") and row["n"] >= 1:
            highlights.append(
                f"Quadro {row['key']} fonti: {row['label']} prese ({row['hit_rate']:.0%})"
            )

    total_hits = sum(1 for r in pool if int(r.get("hit") or 0) == 1)
    return {
        "ok": True,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "filters": {
            "live_only": live_only,
            "trainable_only": trainable_only,
            "require_score_unified": require_score_unified,
        },
        "n_settled_total": len(settled),
        "n_in_pool": len(pool),
        "n_with_score_unified": len(with_unified),
        "n_with_score": len(with_score),
        "overall_hits": total_hits,
        "overall_hit_rate": round(total_hits / len(pool), 4) if pool else None,
        "gioca": gioca,
        "highlights": highlights,
        "by_score_unified": by_unified,
        "by_score": by_score,
        "by_action": by_action,
        "by_market": by_market,
        "by_quadro_consensus": by_quadro,
        "by_unified_and_action": by_unified_action,
        "recent": _recent_rows(pool),
    }


def refresh_analysis_outcomes() -> dict[str, Any]:
    """Ricalcola e salva JSON (chiamato dopo settle / backfill)."""
    report = build_analysis_outcomes_report()
    report["live_summary"] = build_analysis_outcomes_report(live_only=True)
    report["trainable_summary"] = build_analysis_outcomes_report(trainable_only=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        from modules.data_update.history import export_plays_csv

        report["plays_csv"] = export_plays_csv()
    except Exception as exc:
        report["plays_csv_error"] = str(exc)
    return report


def load_analysis_outcomes() -> dict[str, Any] | None:
    if not OUT_PATH.exists():
        return None
    try:
        return json.loads(OUT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def outcome_phrase(*, score_unified: int, live_only: bool = False) -> str | None:
    """Es. '4/5 (80%)' per voto unificato 10."""
    rep = load_analysis_outcomes() or refresh_analysis_outcomes()
    data = rep.get("live_summary") if live_only and rep.get("live_summary") else rep
    data = data or rep
    for row in data.get("by_score_unified") or []:
        if str(row.get("key")) == str(int(score_unified)):
            return f"{row['label']} ({row['hit_rate']:.0%})"
    return None
