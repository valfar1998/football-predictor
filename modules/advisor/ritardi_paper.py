"""Paper ROI ritardi: cohort di alert GIOCA freeze che avevano anomalie pre-match.

Non è un tipster indipendente: misura come sono andati i pick Telegram
quando sul match c'era almeno un ritardo statistico (severity ≥ soglia).
Solo lettura — non tocca EV/Kelly live.
"""

from __future__ import annotations

import json
from datetime import date as date_cls
from pathlib import Path
from typing import Any

from modules.advisor.paper_stats import _kelly_of, _odds_of

_CACHE_KEY: tuple[float, float, int, int] | None = None
_CACHE_VAL: dict[str, Any] | None = None

# Allineato a ritardi_paper_report / Telegram GIOCA
ROI_RITARDI_MIN_SEVERITY = 0.45
ROOT = Path(__file__).resolve().parents[2]
UPCOMING_PATH = ROOT / "data" / "processed" / "upcoming_predictions.json"


def clear_ritardi_paper_cache() -> None:
    global _CACHE_KEY, _CACHE_VAL
    _CACHE_KEY = None
    _CACHE_VAL = None


def _history_mtime() -> float:
    try:
        from modules.data_update.history import DB

        return float(DB.stat().st_mtime) if DB.is_file() else 0.0
    except OSError:
        return 0.0


def _subset_stats(rows: list[dict[str, Any]], *, kelly_frac: float = 0.25) -> dict[str, Any]:
    hits = 0
    odds_pnls: list[float] = []
    kelly_unit: list[tuple[float, float]] = []
    for r in rows:
        hit = int(r.get("hit") or 0) == 1
        if hit:
            hits += 1
        od = _odds_of(r)
        if od is not None and od > 1.01:
            ret = (od - 1.0) if hit else -1.0
            odds_pnls.append(ret)
            stake = _kelly_of(r, kelly_frac=kelly_frac)
            if stake is not None and stake > 0:
                kelly_unit.append((stake, ret))
    n = len(rows)
    stake_sum = sum(s for s, _ in kelly_unit)
    k_pnl = sum(s * ret for s, ret in kelly_unit)
    return {
        "n": n,
        "hits": hits,
        "hit_rate": round(hits / n, 3) if n else None,
        "odds_n": len(odds_pnls),
        "odds_pnl": round(sum(odds_pnls), 4) if odds_pnls else 0.0,
        "odds_roi": round(sum(odds_pnls) / len(odds_pnls), 3) if odds_pnls else None,
        "kelly_n": len(kelly_unit),
        "kelly_stake_sum": round(stake_sum, 4) if kelly_unit else None,
        "kelly_pnl": round(k_pnl, 4) if kelly_unit else None,
        "kelly_roi": round(k_pnl / stake_sum, 3) if stake_sum > 0 else None,
    }


def _score_unified(row: dict[str, Any]) -> int | None:
    from modules.advisor.learn_policy import score_unified_of

    return score_unified_of(row)


def _is_gioca_roi_row(row: dict[str, Any], *, min_score: int) -> bool:
    """Stesso gate del paper ROI: GIOCA Telegram (freeze o voto unificato ≥ soglia)."""
    kind = str(row.get("alert_kind") or "").strip().lower()
    action = str(row.get("action") or "").strip().lower()
    if kind == "watch" or action in {"no_bet", "invalido", "n/d"}:
        return False
    if kind != "gioca" and action != "gioca":
        return False
    locked = int(row.get("score_locked") or 0) == 1
    su = _score_unified(row)
    if locked:
        return True
    return su is not None and su >= int(min_score)


def _anomaly_block(row: dict[str, Any]) -> dict[str, Any] | None:
    from modules.advisor.stat_anomalies import sanitize_anomaly_block

    block = row.get("stat_anomalies")
    if not isinstance(block, dict):
        pred = row.get("prediction") if isinstance(row.get("prediction"), dict) else {}
        block = pred.get("stat_anomalies") if isinstance(pred, dict) else None
    if not isinstance(block, dict):
        return None
    return sanitize_anomaly_block(block)


def _match_key(day: str, home: str, away: str) -> str:
    return f"{str(day)[:10]}|{home}|{away}"


def list_ritardi_roi_playable(
    upcoming: list[dict[str, Any]] | None = None,
    *,
    min_severity: float = ROI_RITARDI_MIN_SEVERITY,
    min_score: int | None = None,
    compute_missing: bool = False,
) -> dict[str, Any]:
    """GIOCA (Telegram / voto≥min) con ritardo ≥ soglia: candidati al ROI ritardi.

    Include pending già freeze in storico e partite upcoming ancora da giocare.
    Non è la tabella settimanale completa: solo quelle che, a settle, entrano nel ROI ritardi.
    """
    from modules.advisor.learn_policy import ROI_MIN_SCORE

    if min_score is None:
        min_score = ROI_MIN_SCORE
    rows_up = upcoming
    if rows_up is None:
        rows_up = []
        if UPCOMING_PATH.is_file():
            try:
                raw = json.loads(UPCOMING_PATH.read_text(encoding="utf-8"))
                rows_up = raw if isinstance(raw, list) else []
            except Exception:
                rows_up = []

    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _push(row: dict[str, Any], *, source: str, block: dict[str, Any]) -> None:
        day = str(row.get("date") or "")[:10]
        home = str(row.get("home") or "")
        away = str(row.get("away") or "")
        if not day or not home or not away:
            return
        key = _match_key(day, home, away)
        if key in seen:
            return
        seen.add(key)
        items = list(block.get("items") or [])
        top = items[0] if items else {}
        locked = int(row.get("score_locked") or 0) == 1
        out.append(
            {
                "date": day,
                "time": row.get("time") or "",
                "league": row.get("league") or "",
                "home": home,
                "away": away,
                "pick": row.get("pick") or row.get("play") or "",
                "score_unified": _score_unified(row),
                "score_locked": 1 if locked else 0,
                "quota_pick": row.get("quota_pick") or row.get("odds") or row.get("quota"),
                "action": row.get("action"),
                "alert_kind": row.get("alert_kind"),
                "severity": float(block.get("max_severity") or 0),
                "n_ritardi": int(block.get("n") or len(items) or 0),
                "top_ritardo": (top or {}).get("title"),
                "kinds": sorted({str(a.get("kind") or "") for a in items if a.get("kind")}),
                "source": source,
                "status": "freeze Telegram" if locked else "GIOCA live (entrerà a freeze)",
            }
        )

    # 1) Pending freeze in storico (già in coda ROI)
    try:
        from modules.data_update.history import load_history

        for r in load_history():
            if r.get("hit") is not None:
                continue
            if not _is_gioca_roi_row(r, min_score=int(min_score)):
                continue
            if int(r.get("score_locked") or 0) != 1:
                continue
            block = _anomaly_block(r)
            if block is None and compute_missing:
                try:
                    from modules.advisor.stat_anomalies import match_anomalies

                    block = match_anomalies(
                        str(r.get("home") or ""),
                        str(r.get("away") or ""),
                        str(r.get("date") or "")[:10],
                        max_n=4,
                        min_severity=float(min_severity),
                    )
                except Exception:
                    block = None
            if not isinstance(block, dict):
                continue
            if float(block.get("max_severity") or 0) < float(min_severity):
                continue
            _push(r, source="history_pending", block=block)
    except Exception:
        pass

    # 2) Calendario upcoming: GIOCA voto≥min (o già freeze) + ritardo (solo future/oggi)
    today = date_cls.today().isoformat()
    for r in rows_up:
        if not isinstance(r, dict):
            continue
        day = str(r.get("date") or "")[:10]
        if day and day < today:
            continue
        if not _is_gioca_roi_row(r, min_score=int(min_score)):
            continue
        block = _anomaly_block(r)
        if block is None and compute_missing:
            try:
                from modules.advisor.stat_anomalies import match_anomalies

                pred = r.get("prediction") if isinstance(r.get("prediction"), dict) else {}
                tac = pred.get("tactical") if isinstance(pred, dict) else None
                block = match_anomalies(
                    str(r.get("home") or ""),
                    str(r.get("away") or ""),
                    str(r.get("date") or "")[:10],
                    tactical=tac if isinstance(tac, dict) else None,
                    max_n=4,
                    min_severity=float(min_severity),
                )
            except Exception:
                block = None
        if not isinstance(block, dict) or not block.get("ready"):
            continue
        if float(block.get("max_severity") or 0) < float(min_severity):
            continue
        src = "freeze" if int(r.get("score_locked") or 0) == 1 else "upcoming_gioca"
        _push(r, source=src, block=block)

    out.sort(
        key=lambda x: (
            -int(x.get("score_locked") or 0),
            -float(x.get("severity") or 0),
            str(x.get("date") or ""),
        )
    )
    n_freeze = sum(1 for r in out if int(r.get("score_locked") or 0) == 1)
    return {
        "ok": True,
        "min_severity": float(min_severity),
        "min_score": int(min_score),
        "n": len(out),
        "n_freeze": n_freeze,
        "n_live_gioca": len(out) - n_freeze,
        "note": (
            "Solo GIOCA Telegram (voto freeze ≥ "
            f"{int(min_score)}) con ritardo severity ≥ {float(min_severity):.0%}. "
            "Queste, a settle, entrano nel ROI ritardi — non tutta la tabella settimanale."
        ),
        "rows": out,
    }


def ritardi_paper_report(
    *,
    min_severity: float = 0.45,
    kelly_frac: float = 0.25,
    min_score: int | None = None,
    max_scan: int = 400,
) -> dict[str, Any]:
    """ROI sui GIOCA freeze settled che avevano ritardi pre-kickoff (ricalcolo storico)."""
    global _CACHE_KEY, _CACHE_VAL
    try:
        from modules.data_update.history import load_history
        from modules.advisor.learn_policy import ROI_MIN_SCORE, meets_roi_score, roi_settled
        from modules.advisor.stat_anomalies import match_anomalies
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    if min_score is None:
        min_score = ROI_MIN_SCORE
    mtime = _history_mtime()
    cache_key = (mtime, round(float(min_severity), 3), int(min_score), int(max_scan))
    if _CACHE_KEY == cache_key and isinstance(_CACHE_VAL, dict):
        return dict(_CACHE_VAL)

    all_rows = load_history()
    settled = [r for r in all_rows if r.get("hit") is not None]
    pending = [
        r
        for r in all_rows
        if r.get("hit") is None
        and int(r.get("score_locked") or 0) == 1
        and str(r.get("alert_kind") or "").lower() != "watch"
        and (
            str(r.get("alert_kind") or "").lower() == "gioca"
            or str(r.get("action") or "").lower() == "gioca"
        )
    ]
    base = roi_settled(settled, min_score=min_score)
    # Ordina recenti prima: scan limitato per non bloccare la UI
    base = sorted(base, key=lambda r: str(r.get("date") or ""), reverse=True)[: int(max_scan)]

    with_rit: list[dict[str, Any]] = []
    scanned = 0
    for r in base:
        home = str(r.get("home") or "")
        away = str(r.get("away") or "")
        day = str(r.get("date") or "")[:10]
        if not home or not away or not day:
            continue
        scanned += 1
        try:
            block = match_anomalies(home, away, day, max_n=4, min_severity=float(min_severity))
        except Exception:
            continue
        if not block.get("ready"):
            continue
        sev = float(block.get("max_severity") or 0)
        if sev < float(min_severity):
            continue
        items = list(block.get("items") or [])
        enriched = dict(r)
        enriched["_ritardi_severity"] = sev
        enriched["_ritardi_n"] = len(items)
        enriched["_ritardi_top"] = (items[0] or {}).get("title") if items else None
        enriched["_ritardi_kinds"] = sorted({str(a.get("kind") or "") for a in items if a.get("kind")})
        with_rit.append(enriched)

    stats = _subset_stats(with_rit, kelly_frac=kelly_frac)
    by_kind: dict[str, dict[str, Any]] = {}
    for kind in ("season_start", "streak", "rate_drought", "absence"):
        subset = [r for r in with_rit if kind in (r.get("_ritardi_kinds") or [])]
        by_kind[kind] = _subset_stats(subset, kelly_frac=kelly_frac)

    # Pending con ritardi già in calendario (se presenti)
    pending_rit = 0
    for r in pending:
        pred = r.get("prediction") if isinstance(r.get("prediction"), dict) else {}
        block = r.get("stat_anomalies") or (pred or {}).get("stat_anomalies")
        if isinstance(block, dict) and float(block.get("max_severity") or 0) >= float(min_severity):
            pending_rit += 1
            continue
        # non ricalcolare tutti i pending (lento): conta solo se già arricchiti

    out = {
        "ok": True,
        "roi_mode": "gioca_freeze_with_ritardi",
        "note": (
            "Cohort: alert Telegram GIOCA settled (voto freeze) su cui, al kickoff, "
            f"c'era almeno un ritardo statistico (severity ≥ {min_severity:.0%}). "
            "Non è un mercato separato: misura i pick analisi in presenza di anomalie."
        ),
        "min_severity": float(min_severity),
        "min_score": int(min_score),
        "n_scanned": scanned,
        "n": stats["n"],
        "n_pending": pending_rit,
        "hits": stats["hits"],
        "hit_rate": stats["hit_rate"],
        "odds_roi": stats["odds_roi"],
        "kelly_roi": stats["kelly_roi"],
        "kelly_pnl": stats["kelly_pnl"],
        "odds_pnl": stats["odds_pnl"],
        "by_kind": by_kind,
        "recent": [
            {
                "date": r.get("date"),
                "home": r.get("home"),
                "away": r.get("away"),
                "pick": r.get("pick"),
                "hit": r.get("hit"),
                "severity": r.get("_ritardi_severity"),
                "top": r.get("_ritardi_top"),
                "kinds": ",".join(r.get("_ritardi_kinds") or []),
            }
            for r in sorted(with_rit, key=lambda x: str(x.get("date") or ""), reverse=True)[:25]
        ],
    }
    _CACHE_KEY = cache_key
    _CACHE_VAL = out
    return dict(out)
