"""Paper trading stats da SQLite storico locale."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def paper_trading_report() -> dict[str, Any]:
    try:
        from modules.data_update.history import load_history
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    rows = [r for r in load_history() if r.get("hit") is not None]
    if not rows:
        return {"ok": True, "n": 0, "note": "nessun esito settled"}

    def _bucket(key_fn):
        g: dict[str, list] = defaultdict(list)
        for r in rows:
            g[str(key_fn(r) or "n/d")].append(r)
        out = []
        for k, items in sorted(g.items(), key=lambda t: -len(t[1])):
            hits = sum(1 for x in items if int(x.get("hit") or 0) == 1)
            n = len(items)
            # ROI unit stake @ quoted odds if present else skip
            pnl = 0.0
            staked = 0
            for x in items:
                # probability/odds not always stored; use hit proxy: +1 / -1 flat
                staked += 1
                pnl += 1.0 if int(x.get("hit") or 0) == 1 else -1.0
            out.append(
                {
                    "key": k,
                    "n": n,
                    "hits": hits,
                    "hit_rate": round(hits / n, 3) if n else 0.0,
                    "flat_pnl": round(pnl, 2),
                    "flat_roi": round(pnl / staked, 3) if staked else 0.0,
                }
            )
        return out

    by_league = _bucket(lambda r: r.get("league"))
    by_action = _bucket(lambda r: r.get("action"))
    by_pick = _bucket(lambda r: r.get("pick"))

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

    by_score = _bucket(score_band)

    overall_hits = sum(1 for r in rows if int(r.get("hit") or 0) == 1)
    return {
        "ok": True,
        "n": len(rows),
        "hit_rate": round(overall_hits / len(rows), 3),
        "by_league": by_league[:15],
        "by_action": by_action,
        "by_pick": by_pick,
        "by_score": by_score,
    }
