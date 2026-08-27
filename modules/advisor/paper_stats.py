"""Paper trading: flat, ROI @ quote, Kelly, equity, drawdown, Sharpe, CLV, walk-forward."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import math


def _odds_of(r: dict[str, Any]) -> float | None:
    for k in ("quota_pick", "odds", "quota", "odds_real", "odd"):
        v = r.get(k)
        try:
            x = float(v)
            if 1.01 <= x <= 50:
                return x
        except (TypeError, ValueError):
            pass
    return None


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
    var = sum((x - mean) ** 2 for x in pnls) / max(1, len(pnls) - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    sharpe = (mean / std) * math.sqrt(len(pnls)) if std > 1e-9 else None
    return {
        "n": len(pnls),
        "final": round(bank, 3),
        "max_drawdown": round(max_dd, 3),
        "sharpe": None if sharpe is None else round(sharpe, 3),
        "equity_tail": [round(e, 3) for e in equity[-30:]],
    }


def kelly_equity_snapshot(*, kelly_frac: float = 0.25, trainable_only: bool = True) -> dict[str, Any]:
    """Snapshot equity Kelly per drawdown guard (solo righe con quota)."""
    try:
        from modules.data_update.history import load_history
        from modules.advisor.learn_policy import trainable_settled
    except Exception:
        return {"ok": False}

    rows = sorted(
        [r for r in load_history() if r.get("hit") is not None],
        key=lambda r: str(r.get("date") or ""),
    )
    if trainable_only:
        rows = trainable_settled(rows)
    bank = 100.0
    pnls: list[float] = []
    for r in rows:
        od = _odds_of(r)
        if od is None:
            continue
        hit = int(r.get("hit") or 0) == 1
        p = None
        try:
            p = float(r.get("probability"))
        except (TypeError, ValueError):
            p = None
        if p is None or p <= 0:
            p = 1.0 / od
        f = _kelly_fraction(p, od, frac=kelly_frac, risk_scale=1.0)
        stake = bank * f
        if stake <= 0:
            pnls.append(0.0)
            continue
        pnl = stake * (od - 1.0) if hit else -stake
        bank += pnl
        pnls.append(pnl)
    if not pnls:
        return {"ok": False, "n": 0}
    eq = _equity_stats(pnls)
    return {"ok": True, **eq}


def paper_trading_report(*, bankroll: float = 100.0, kelly_frac: float = 0.25, trainable_only: bool = True) -> dict[str, Any]:
    try:
        from modules.data_update.history import load_history
        from modules.model_training.league_clusters import cluster_for
        from modules.advisor.learn_policy import trainable_settled, is_live
        from modules.advisor.staking import beat_close, kelly_risk_scale
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    all_settled = sorted(
        [r for r in load_history() if r.get("hit") is not None],
        key=lambda r: str(r.get("date") or ""),
    )
    rows = trainable_settled(all_settled) if trainable_only else all_settled
    live_rows = [r for r in rows if is_live(r)]
    if not rows:
        return {"ok": True, "n": 0, "note": "nessun esito settled"}

    def _bucket(key_fn, pool: list[dict]):
        g: dict[str, list] = defaultdict(list)
        for r in pool:
            g[str(key_fn(r) or "n/d")].append(r)
        out = []
        for k, items in sorted(g.items(), key=lambda t: -len(t[1])):
            hits = sum(1 for x in items if int(x.get("hit") or 0) == 1)
            n = len(items)
            flat_pnl = sum(1.0 if int(x.get("hit") or 0) == 1 else -1.0 for x in items)
            odds_pnl = 0.0
            odds_n = 0
            clvs: list[float] = []
            beats = 0
            n_beat = 0
            for x in items:
                od = _odds_of(x)
                if od is not None:
                    odds_n += 1
                    odds_pnl += (od - 1.0) if int(x.get("hit") or 0) == 1 else -1.0
                cv = _clv_of(x)
                if cv is not None:
                    clvs.append(cv)
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
                    "n": n,
                    "hits": hits,
                    "hit_rate": round(hits / n, 3) if n else 0.0,
                    "flat_pnl": round(flat_pnl, 2),
                    "flat_roi": round(flat_pnl / n, 3) if n else 0.0,
                    "odds_n": odds_n,
                    "odds_pnl": round(odds_pnl, 2),
                    "odds_roi": round(odds_pnl / odds_n, 3) if odds_n else None,
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

    flat_pnls = [1.0 if int(r.get("hit") or 0) == 1 else -1.0 for r in rows]
    odds_pnls = []
    kelly_pnls = []
    clv_all: list[float] = []
    bank = float(bankroll)
    pre_snap = kelly_equity_snapshot(kelly_frac=kelly_frac, trainable_only=trainable_only)
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
        odds_pnls.append((od - 1.0) if hit else -1.0)
        p = None
        try:
            p = float(r.get("probability"))
        except (TypeError, ValueError):
            p = None
        if p is None or p <= 0:
            p = 1.0 / od
        f = _kelly_fraction(p, od, frac=kelly_frac, risk_scale=risk_scale)
        stake = bank * f
        if stake <= 0:
            kelly_pnls.append(0.0)
            continue
        pnl = stake * (od - 1.0) if hit else -stake
        bank += pnl
        kelly_pnls.append(pnl)

    wf = []
    if len(odds_pnls) >= 40:
        step = max(10, len(odds_pnls) // 5)
        for i in range(step, len(odds_pnls) + 1, step):
            chunk = odds_pnls[:i]
            wf.append({"n": i, "roi": round(sum(chunk) / i, 4), "pnl": round(sum(chunk), 2)})

    overall_hits = sum(1 for r in rows if int(r.get("hit") or 0) == 1)
    live_odds_n = sum(1 for r in live_rows if _odds_of(r))
    return {
        "ok": True,
        "trainable_only": trainable_only,
        "n_settled_total": len(all_settled),
        "n": len(rows),
        "n_live": len(live_rows),
        "n_live_odds": live_odds_n,
        "hit_rate": round(overall_hits / len(rows), 3),
        "flat_pnl": round(sum(flat_pnls), 2),
        "flat_roi": round(sum(flat_pnls) / len(rows), 3),
        "odds_n": len(odds_pnls),
        "odds_pnl": round(sum(odds_pnls), 2) if odds_pnls else 0.0,
        "odds_roi": round(sum(odds_pnls) / len(odds_pnls), 3) if odds_pnls else None,
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
        },
        "walk_forward_odds_roi": wf,
        "by_league": by_league[:15],
        "by_cluster": by_cluster,
        "by_market": by_market,
        "by_odds_band": by_odds_band,
        "by_action": by_action,
        "by_pick": by_pick,
        "by_score": by_score,
    }
