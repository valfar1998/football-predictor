"""Paper trading: solo freeze Telegram (quota+Kelly messaggio), ROI pesato per stake Kelly."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import math


def _odds_of(r: dict[str, Any]) -> float | None:
    """Quota ufficiale: freeze Telegram = quota_pick (mai ricalcolo live)."""
    for k in ("quota_pick", "odds", "quota", "odds_real", "odd"):
        v = r.get(k)
        try:
            x = float(v)
            if 1.01 <= x <= 50:
                return x
        except (TypeError, ValueError):
            pass
    return None


def _kelly_of(r: dict[str, Any], *, kelly_frac: float = 0.25) -> float | None:
    """Stake Kelly congelato al messaggio; fallback ricostruito da p+quota freeze."""
    try:
        k = float(r.get("kelly_quarter"))
        if k > 0:
            return min(0.08, k)
    except (TypeError, ValueError):
        pass
    od = _odds_of(r)
    try:
        p = float(r.get("probability"))
    except (TypeError, ValueError):
        p = None
    if od is None or p is None or p <= 0:
        return None
    return _kelly_fraction(p, od, frac=kelly_frac, risk_scale=1.0) or None


def _clv_of(r: dict[str, Any]) -> float | None:
    from modules.advisor.staking import clv_prob

    try:
        if r.get("clv") is not None:
            return float(r["clv"])
    except (TypeError, ValueError):
        pass
    q = _odds_of(r)
    try:
        qc = float(r["quota_close"]) if r.get("quota_close") is not None else None
    except (TypeError, ValueError):
        qc = None
    if q and qc:
        return clv_prob(q, qc)
    return None


def _odds_band(od: float | None) -> str:
    if od is None:
        return "n/d"
    if od < 1.5:
        return "1.20-1.50"
    if od < 2.0:
        return "1.50-2.00"
    if od < 2.5:
        return "2.00-2.50"
    if od < 3.5:
        return "2.50-3.50"
    return "3.50+"


def _score_bucket_stats(items: list[dict[str, Any]], *, kelly_frac: float = 0.25) -> dict[str, Any]:
    """Flat + ROI unitario @ quote + ROI pesato Kelly su un sottoinsieme."""
    n = len(items)
    hits = sum(1 for x in items if int(x.get("hit") or 0) == 1)
    flat_pnl = float(hits - (n - hits))
    odds_pnl = 0.0
    odds_n = 0
    stake_sum = 0.0
    kelly_pnl = 0.0
    kelly_n = 0
    for x in items:
        od = _odds_of(x)
        if od is None:
            continue
        odds_n += 1
        ret = (od - 1.0) if int(x.get("hit") or 0) == 1 else -1.0
        odds_pnl += ret
        stake = _kelly_of(x, kelly_frac=kelly_frac)
        if stake is None or stake <= 0:
            continue
        kelly_n += 1
        stake_sum += stake
        kelly_pnl += stake * ret
    return {
        "n": n,
        "hits": hits,
        "hit_rate": round(hits / n, 3) if n else None,
        "flat_pnl": round(flat_pnl, 2),
        "flat_roi": round(flat_pnl / n, 3) if n else None,
        "odds_n": odds_n,
        "odds_pnl": round(odds_pnl, 2),
        "odds_roi": round(odds_pnl / odds_n, 3) if odds_n else None,
        "kelly_n": kelly_n,
        "kelly_stake_sum": round(stake_sum, 4) if kelly_n else None,
        "kelly_pnl": round(kelly_pnl, 4) if kelly_n else None,
        "kelly_roi": round(kelly_pnl / stake_sum, 3) if stake_sum > 0 else None,
    }


def _by_unified_vote(
    settled_rows: list[dict[str, Any]],
    pending_rows: list[dict[str, Any]],
    *,
    votes: tuple[int, ...] = (8, 9, 10),
    kelly_frac: float = 0.25,
) -> dict[str, dict[str, Any]]:
    from modules.advisor.learn_policy import score_unified_of

    out: dict[str, dict[str, Any]] = {}
    for v in votes:
        settled_v = [r for r in settled_rows if score_unified_of(r) == v]
        pending_v = [r for r in pending_rows if score_unified_of(r) == v]
        stats = _score_bucket_stats(settled_v, kelly_frac=kelly_frac)
        stats["pending"] = len(pending_v)
        stats["vote"] = v
        out[str(v)] = stats
    return out


def _kelly_fraction(p: float, odds: float, frac: float = 0.25, risk_scale: float = 1.0) -> float:
    if odds <= 1.01 or p <= 0:
        return 0.0
    b = odds - 1.0
    q = 1.0 - p
    f = (b * p - q) / b
    scale = max(0.45, min(1.0, float(risk_scale)))
    return max(0.0, min(0.08 * scale, f * frac * scale))


def _equity_stats(pnls: list[float]) -> dict[str, Any]:
    if not pnls:
        return {"n": 0}
    equity = []
    bank = 0.0
    peak = 0.0
    max_dd = 0.0
    for x in pnls:
        bank += x
        equity.append(bank)
        peak = max(peak, bank)
        max_dd = min(max_dd, bank - peak)
    mean = sum(pnls) / len(pnls)
    var = sum((x - mean) ** 2 for x in pnls) / len(pnls)
    std = math.sqrt(var) if var > 0 else 0.0
    sharpe = (mean / std) if std > 1e-12 else None
    return {
        "n": len(pnls),
        "pnl": round(bank, 4),
        "max_drawdown": round(max_dd, 4),
        "sharpe": round(sharpe, 3) if sharpe is not None else None,
        "mean_pnl": round(mean, 4),
    }


def kelly_equity_snapshot(
    *,
    kelly_frac: float = 0.25,
    trainable_only: bool = True,
    min_score: int | None = None,
) -> dict[str, Any]:
    try:
        from modules.data_update.history import load_history
        from modules.advisor.learn_policy import ROI_MIN_SCORE, roi_settled
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    if min_score is None:
        min_score = ROI_MIN_SCORE
    all_rows = load_history()
    settled = [r for r in all_rows if r.get("hit") is not None]
    rows = roi_settled(settled, min_score=min_score) if trainable_only else [
        r for r in settled if int(r.get("score_locked") or 0) == 1
    ]
    rows = sorted(rows, key=lambda r: str(r.get("date") or ""))
    pnls: list[float] = []
    bank = 100.0
    for r in rows:
        od = _odds_of(r)
        stake_f = _kelly_of(r, kelly_frac=kelly_frac)
        if od is None or stake_f is None or stake_f <= 0:
            continue
        stake = bank * stake_f
        hit = int(r.get("hit") or 0) == 1
        pnl = stake * (od - 1.0) if hit else -stake
        bank += pnl
        pnls.append(pnl)
    stats = _equity_stats(pnls)
    stats["ok"] = True
    stats["bankroll_end"] = round(bank, 2)
    return stats


def kelly_risk_scale_from_history(*, kelly_frac: float = 0.25) -> float:
    from modules.advisor.staking import kelly_risk_scale

    snap = kelly_equity_snapshot(kelly_frac=kelly_frac)
    if not snap.get("ok"):
        return 1.0
    return kelly_risk_scale(
        max_drawdown=snap.get("max_drawdown"),
        sharpe=snap.get("sharpe"),
    )


def paper_trading_report(
    *,
    bankroll: float = 100.0,
    kelly_frac: float = 0.25,
    trainable_only: bool = True,
    min_score: int | None = None,
) -> dict[str, Any]:
    """ROI paper: solo alert Telegram GIOCA congelati; PnL pesato per Kelly freeze."""
    try:
        from modules.data_update.history import load_history
        from modules.model_training.league_clusters import cluster_for
        from modules.advisor.learn_policy import ROI_MIN_SCORE, is_live, meets_roi_score, roi_settled
        from modules.advisor.staking import beat_close, kelly_risk_scale
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    if min_score is None:
        min_score = ROI_MIN_SCORE
    all_rows = load_history()
    all_settled = sorted(
        [r for r in all_rows if r.get("hit") is not None],
        key=lambda r: str(r.get("date") or ""),
    )
    all_pending = [r for r in all_rows if r.get("hit") is None]
    pending_score = [
        r
        for r in all_pending
        if int(r.get("score_locked") or 0) == 1
        and str(r.get("alert_kind") or "").lower() != "watch"
        and str(r.get("action") or "").lower() not in {"no_bet", "invalido", "n/d"}
        and (
            str(r.get("alert_kind") or "").lower() == "gioca"
            or str(r.get("action") or "").lower() == "gioca"
        )
    ]
    n_locked_pending = sum(1 for r in all_pending if int(r.get("score_locked") or 0) == 1)
    n_locked_total = sum(1 for r in all_rows if int(r.get("score_locked") or 0) == 1)
    db_stats = {
        "n_history": len(all_rows),
        "n_settled_total": len(all_settled),
        "n_pending": len(all_pending),
        "n_pending_score": len(pending_score),
        "n_locked_pending": n_locked_pending,
        "n_locked_total": n_locked_total,
        "settled_pct": round(len(all_settled) / len(all_rows), 3) if all_rows else None,
        "roi_mode": "telegram_freeze_kelly",
    }
    rows = roi_settled(all_settled, min_score=min_score) if trainable_only else [
        r
        for r in all_settled
        if int(r.get("score_locked") or 0) == 1
        and meets_roi_score(r, min_score=min_score)
    ]
    live_rows = [r for r in rows if is_live(r)]
    if not rows:
        from modules.advisor.learn_policy import roi_sample_progress

        progress0 = roi_sample_progress(0)
        progress0["pct_level"] = 0.0
        progress0["pct_of_target"] = 0.0
        progress0["pct_of_stable"] = 0.0
        return {
            "ok": True,
            "n": 0,
            "n_locked": 0,
            "hits": 0,
            "hit_rate": None,
            "flat_pnl": 0.0,
            "flat_roi": None,
            "odds_n": 0,
            "odds_pnl": 0.0,
            "odds_roi": None,
            "kelly_roi": None,
            "min_score": min_score,
            **db_stats,
            "by_vote": _by_unified_vote([], pending_score, kelly_frac=kelly_frac),
            "sample_progress": progress0,
            "note": (
                f"nessun esito Telegram GIOCA settled con voto ≥{min_score}"
                if min_score
                else "nessun esito Telegram GIOCA settled"
            ),
        }

    def _bucket(key_fn, pool: list[dict]):
        g: dict[str, list] = defaultdict(list)
        for r in pool:
            g[str(key_fn(r) or "n/d")].append(r)
        out = []
        for k, items in sorted(g.items(), key=lambda t: -len(t[1])):
            stats = _score_bucket_stats(items, kelly_frac=kelly_frac)
            clvs: list[float] = []
            beats = 0
            n_beat = 0
            for x in items:
                cv = _clv_of(x)
                if cv is not None:
                    clvs.append(cv)
                od = _odds_of(x)
                if x.get("beat_close") is not None:
                    n_beat += 1
                    beats += int(x.get("beat_close") or 0)
                elif od and x.get("quota_close"):
                    bc = beat_close(od, float(x["quota_close"]))
                    if bc is not None:
                        n_beat += 1
                        beats += 1 if bc else 0
            out.append(
                {
                    "key": k,
                    **stats,
                    "mean_clv": round(sum(clvs) / len(clvs), 4) if clvs else None,
                    "beat_close_rate": round(beats / n_beat, 4) if n_beat else None,
                }
            )
        return out

    by_league = _bucket(lambda r: r.get("league"), rows)
    by_cluster = _bucket(lambda r: cluster_for(r.get("league")), rows)
    by_action = _bucket(lambda r: r.get("action"), rows)
    by_pick = _bucket(lambda r: r.get("pick"), rows)
    by_market = _bucket(lambda r: r.get("pick_group") or "1x2", rows)
    by_odds_band = _bucket(lambda r: _odds_band(_odds_of(r)), rows)

    def score_band(r):
        s = r.get("score_unified")
        if s is None:
            s = r.get("score")
        try:
            s = int(s)
        except (TypeError, ValueError):
            return "n/d"
        if s >= 8:
            return "8-10"
        if s >= 6:
            return "6-7"
        if s >= 4:
            return "4-5"
        return "1-3"

    by_score = _bucket(score_band, rows)
    by_vote = _by_unified_vote(rows, pending_score, kelly_frac=kelly_frac)

    flat_pnls = [1.0 if int(r.get("hit") or 0) == 1 else -1.0 for r in rows]
    odds_pnls: list[float] = []
    kelly_pnls: list[float] = []
    kelly_unit_rets: list[tuple[float, float]] = []  # (stake, ret)
    clv_all: list[float] = []
    bank = float(bankroll)
    pre_snap = kelly_equity_snapshot(
        kelly_frac=kelly_frac, trainable_only=trainable_only, min_score=min_score
    )
    risk_scale = kelly_risk_scale(
        max_drawdown=pre_snap.get("max_drawdown") if pre_snap.get("ok") else None,
        sharpe=pre_snap.get("sharpe") if pre_snap.get("ok") else None,
    )
    for r in rows:
        hit = int(r.get("hit") or 0) == 1
        od = _odds_of(r)
        cv = _clv_of(r)
        if cv is not None:
            clv_all.append(cv)
        if od is None:
            continue
        ret = (od - 1.0) if hit else -1.0
        odds_pnls.append(ret)
        stake_f = _kelly_of(r, kelly_frac=kelly_frac)
        if stake_f is None or stake_f <= 0:
            kelly_pnls.append(0.0)
            continue
        # Equity: bankroll path con Kelly freeze (no ricalcolo da p live)
        stake = bank * stake_f * risk_scale
        pnl = stake * ret
        bank += pnl
        kelly_pnls.append(pnl)
        kelly_unit_rets.append((stake_f, ret))

    stake_sum = sum(s for s, _ in kelly_unit_rets)
    kelly_weighted_pnl = sum(s * ret for s, ret in kelly_unit_rets)
    kelly_roi = (kelly_weighted_pnl / stake_sum) if stake_sum > 0 else None

    wf = []
    if len(kelly_unit_rets) >= 10:
        step = max(5, len(kelly_unit_rets) // 5)
        for i in range(step, len(kelly_unit_rets) + 1, step):
            chunk = kelly_unit_rets[:i]
            ss = sum(s for s, _ in chunk)
            pp = sum(s * ret for s, ret in chunk)
            wf.append(
                {
                    "n": i,
                    "roi": round(pp / ss, 4) if ss > 0 else None,
                    "pnl": round(pp, 4),
                    "stake_sum": round(ss, 4),
                }
            )

    overall_hits = sum(1 for r in rows if int(r.get("hit") or 0) == 1)
    live_odds_n = sum(1 for r in live_rows if _odds_of(r))
    n_locked = sum(1 for r in rows if int(r.get("score_locked") or 0) == 1)
    from modules.advisor.learn_policy import roi_sample_progress

    progress = roi_sample_progress(len(rows))
    target_n = int(progress.get("target_n") or 15)
    final_target = 40
    progress["pct_level"] = round(100.0 * min(1.0, float(progress.get("progress") or 0.0)), 1)
    progress["pct_of_target"] = round(100.0 * len(rows) / target_n, 1) if target_n else 0.0
    progress["pct_of_stable"] = round(100.0 * len(rows) / final_target, 1)
    return {
        "ok": True,
        "trainable_only": trainable_only,
        "min_score": min_score,
        **db_stats,
        "n": len(rows),
        "n_live": len(live_rows),
        "n_live_odds": live_odds_n,
        "n_locked": n_locked,
        "hit_rate": round(overall_hits / len(rows), 3),
        "hits": overall_hits,
        "flat_pnl": round(sum(flat_pnls), 2),
        "flat_roi": round(sum(flat_pnls) / len(rows), 3),
        "odds_n": len(odds_pnls),
        "odds_pnl": round(sum(odds_pnls), 2) if odds_pnls else 0.0,
        "odds_roi": round(sum(odds_pnls) / len(odds_pnls), 3) if odds_pnls else None,
        # Metriche principali: peso ∝ Kelly congelato del messaggio
        "kelly_n": len(kelly_unit_rets),
        "kelly_stake_sum": round(stake_sum, 4) if kelly_unit_rets else None,
        "kelly_pnl": round(kelly_weighted_pnl, 4) if kelly_unit_rets else None,
        "kelly_roi": round(kelly_roi, 3) if kelly_roi is not None else None,
        "mean_clv": round(sum(clv_all) / len(clv_all), 4) if clv_all else None,
        "flat_equity": _equity_stats(flat_pnls),
        "odds_equity": _equity_stats(odds_pnls),
        "kelly_equity": _equity_stats(kelly_pnls),
        "kelly": {
            "bankroll_start": bankroll,
            "bankroll_end": round(bank, 2),
            "frac": kelly_frac,
            "risk_scale": round(risk_scale, 3),
            "n_staked": sum(1 for x in kelly_pnls if abs(x) > 1e-9),
            "frozen": True,
        },
        "sample_progress": progress,
        "walk_forward_odds_roi": wf,
        "by_league": by_league[:15],
        "by_cluster": by_cluster,
        "by_market": by_market,
        "by_odds_band": by_odds_band,
        "by_action": by_action,
        "by_pick": by_pick,
        "by_score": by_score,
        "by_vote": by_vote,
    }
