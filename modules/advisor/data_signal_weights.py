"""Walk-forward pesi data_signal: Brier + ROI, per cluster e mercato."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = ROOT / "data" / "models" / "data_signal_weights.json"

DEFAULT_WEIGHTS = {
    "forma": 0.18,
    "casa/trasferta": 0.16,
    "xG rolling": 0.14,
    "Understat": 0.18,
    "classifica": 0.12,
    "FBref": 0.10,
    "StatsBomb": 0.06,
    "riposo": 0.06,
    "Elo": 0.08,
    "FotMob xG": 0.12,
}

KEYS = ["forma", "casa/trasferta", "xG rolling", "Understat", "classifica", "Elo"]


def load_weights(*, cluster: str | None = None, market: str = "1x2") -> dict[str, float]:
    base = dict(DEFAULT_WEIGHTS)
    if not WEIGHTS_PATH.exists():
        return base
    try:
        data = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return base
    w = data.get("weights") or {}
    base.update({k: float(v) for k, v in w.items()})
    by_m = (data.get("by_market") or {}).get(market) or {}
    if by_m:
        base.update({k: float(v) for k, v in by_m.items()})
    if cluster:
        by_c = (data.get("by_cluster") or {}).get(cluster) or {}
        if by_c:
            base.update({k: float(v) for k, v in by_c.items()})
    return base


def _result_code(rec: dict[str, Any]) -> str | None:
    res = str(rec.get("result") or "")
    if res in {"H", "1"}:
        return "1"
    if res in {"A", "2"}:
        return "2"
    if res in {"D", "X"}:
        return "X"
    hit = rec.get("hit")
    pick = str(rec.get("pick") or "")
    if hit is not None and pick in {"1", "X", "2"}:
        return pick if int(hit) == 1 else None
    return None


def _lean_from_factors(factors: list, w: dict[str, float]) -> tuple[str, float]:
    edge = 0.0
    den = 0.0
    for f in factors:
        name = str(f.get("name") or "")
        ww = float(w.get(name, 0.05)) * float(f.get("weight") or 0.1)
        edge += float(f.get("edge") or 0) * ww
        den += ww
    if den <= 0:
        return "X", 0.33
    e = edge / den
    # probabilità soft del lean casa
    p1 = max(0.05, min(0.90, 0.40 + e * 0.55))
    if e > 0.04:
        return "1", p1
    if e < -0.04:
        return "2", 1.0 - p1
    return "X", 0.28


def _brier_lean(lean: str, p: float, y: str) -> float:
    # one-vs-rest Brier sul lean dominante
    hit = 1.0 if lean == y else 0.0
    return (p - hit) ** 2


def _odds_of(rec: dict[str, Any]) -> float | None:
    for k in ("quota_pick", "odds", "quota", "odds_real", "odd"):
        v = rec.get(k)
        try:
            x = float(v)
            if 1.01 <= x <= 50:
                return x
        except (TypeError, ValueError):
            pass
    return None


def _score_combo(
    rows: list[dict[str, Any]],
    w: dict[str, float],
    *,
    prefer: str = "brier",
) -> tuple[float, dict[str, float]]:
    """Ritorna (score da massimizzare, metriche)."""
    briers: list[float] = []
    hits = 0
    n = 0
    pnl = 0.0
    odds_n = 0
    for rec in rows:
        factors = rec.get("data_factors")
        y = _result_code(rec)
        if not isinstance(factors, list) or y is None:
            continue
        lean, p = _lean_from_factors(factors, w)
        n += 1
        briers.append(_brier_lean(lean, p if lean != "X" else 0.33, y))
        if lean == y:
            hits += 1
        od = _odds_of(rec)
        if od is not None and lean in {"1", "X", "2"}:
            odds_n += 1
            pnl += (od - 1.0) if lean == y else -1.0
    if n < 10:
        return -1e9, {"n": n}
    hit_rate = hits / n
    brier = sum(briers) / len(briers)
    roi = (pnl / odds_n) if odds_n >= 8 else None
    if prefer == "roi" and roi is not None:
        score = roi - 0.15 * brier
    else:
        # massimizza -brier + piccolo bonus hit
        score = -brier + 0.05 * hit_rate
        if roi is not None:
            score += 0.08 * max(-0.5, min(0.5, roi))
    return score, {
        "n": n,
        "hit_rate": round(hit_rate, 4),
        "brier": round(brier, 4),
        "odds_n": odds_n,
        "roi": None if roi is None else round(roi, 4),
    }


def _grid_search(rows: list[dict[str, Any]], *, max_combos: int, prefer: str) -> tuple[dict[str, float], dict]:
    scales = [0.65, 1.0, 1.35]
    best_w = dict(DEFAULT_WEIGHTS)
    best_score = -1e9
    best_m: dict = {}
    tried = 0
    for combo in product(scales, repeat=len(KEYS)):
        tried += 1
        if tried > max_combos:
            break
        w = dict(DEFAULT_WEIGHTS)
        for k, s in zip(KEYS, combo):
            w[k] = DEFAULT_WEIGHTS.get(k, 0.1) * s
        score, metrics = _score_combo(rows, w, prefer=prefer)
        if score > best_score:
            best_score = score
            best_w = w
            best_m = metrics
    best_m["tried"] = tried
    best_m["score"] = round(best_score, 5)
    return best_w, best_m


def _walk_forward(rows: list[dict[str, Any]], *, n_folds: int = 4, max_combos: int = 60) -> dict[str, Any]:
    """Expanding WF: fit su passato, valuta su fold successivo (Brier)."""
    if len(rows) < 40:
        return {"ok": False, "n": len(rows)}
    # ordina per data se presente
    rows = sorted(rows, key=lambda r: str(r.get("date") or ""))
    fold_size = max(15, len(rows) // n_folds)
    folds_out = []
    start = max(25, len(rows) // 3)
    idx = start
    while idx < len(rows):
        train = rows[:idx]
        test = rows[idx : idx + fold_size]
        if len(test) < 10:
            break
        w, _ = _grid_search(train, max_combos=max_combos, prefer="brier")
        score, metrics = _score_combo(test, w, prefer="brier")
        folds_out.append({"n_train": len(train), "n_test": len(test), **metrics, "fold_score": round(score, 5)})
        idx += fold_size
    if not folds_out:
        return {"ok": False, "n": len(rows), "error": "wf insufficiente"}
    avg_brier = sum(f.get("brier") or 0 for f in folds_out) / len(folds_out)
    return {"ok": True, "n_folds": len(folds_out), "avg_brier": round(avg_brier, 4), "folds": folds_out}


def optimize_weights(*, max_combos: int = 90, prefer: str = "brier", aggressive: bool = False) -> dict[str, Any]:
    """Walk-forward + griglia globale; pesi per cluster e mercato se n sufficiente."""
    try:
        from modules.data_update.history import load_history
        from modules.model_training.league_clusters import cluster_for
        from modules.advisor.learn_policy import replicate_for_fit, split_trainable
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    settled = [r for r in load_history() if r.get("hit") is not None]
    live_rich, backfill_rich = split_trainable(settled)
    pool, fit_meta = replicate_for_fit(settled)
    raw = []
    for rec in pool:
        factors = rec.get("data_factors")
        if not isinstance(factors, list) or len(factors) < 3:
            # fallback: sintetizza fattori da campi scalari se presenti
            continue
        if _result_code(rec) is None and rec.get("hit") is None:
            continue
        raw.append(rec)

    # Se pochi data_factors, usa proxy da hit/pick per non fallire del tutto
    if len(raw) < 40:
        proxy = []
        for rec in pool:
            if rec.get("hit") is None:
                continue
            # fattori proxy minimi da edge/score se esistono
            edge = rec.get("data_edge")
            if edge is None and rec.get("ev_cons") is not None:
                edge = float(rec["ev_cons"])
            if edge is None:
                continue
            rec = dict(rec)
            rec["data_factors"] = [
                {"name": "forma", "edge": float(edge) * 0.4, "weight": 0.18},
                {"name": "xG rolling", "edge": float(edge) * 0.35, "weight": 0.14},
                {"name": "Elo", "edge": float(edge) * 0.25, "weight": 0.08},
            ]
            if _result_code(rec) or rec.get("pick"):
                proxy.append(rec)
        raw = proxy

    if len(raw) < 25:
        return {"ok": False, "n": len(raw), "error": "servono >=25 settled con fattori o proxy EV"}

    wf = _walk_forward(raw, n_folds=4, max_combos=min(60, max_combos))
    best_w, metrics = _grid_search(raw, max_combos=max_combos, prefer=prefer)

    by_cluster: dict[str, dict] = {}
    buckets: dict[str, list] = {}
    for rec in raw:
        cid = cluster_for(rec.get("league"))
        buckets.setdefault(cid, []).append(rec)
    for cid, items in buckets.items():
        if len(items) < 30:
            continue
        w, m = _grid_search(items, max_combos=min(50, max_combos), prefer=prefer)
        by_cluster[cid] = {"weights": {k: round(v, 4) for k, v in w.items()}, **m}

    # mercati: stesso schema, filtra per pick group se presente
    by_market: dict[str, dict] = {}
    for mkt, pred in (("1x2", lambda r: str(r.get("pick") or "") in {"1", "X", "2", ""}),
                      ("ou", lambda r: "over" in str(r.get("pick") or "").lower() or "under" in str(r.get("pick") or "").lower()),
                      ("ah", lambda r: "ah" in str(r.get("pick") or "").lower() or "handicap" in str(r.get("pick_group") or "").lower())):
        items = [r for r in raw if pred(r)]
        if len(items) < 25:
            continue
        w, m = _grid_search(items, max_combos=min(50, max_combos), prefer="roi" if mkt != "1x2" else prefer)
        by_market[mkt] = {"weights": {k: round(v, 4) for k, v in w.items()}, **m}

    payload = {
        "ok": True,
        "method": "walk_forward_brier_roi",
        "n": len(raw),
        "n_live_rich": len(live_rich),
        "n_backfill_rich": len(backfill_rich),
        "fit_policy": fit_meta,
        "aggressive": aggressive,
        "prefer": prefer,
        "weights": {k: round(v, 4) for k, v in best_w.items()},
        "metrics": metrics,
        "walk_forward": wf,
        "by_cluster": by_cluster,
        "by_market": {k: v.get("weights") for k, v in by_market.items()},
        "by_market_metrics": by_market,
    }
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
