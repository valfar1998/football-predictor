"""Conformal prediction: 1X2 + O/U 2.5 + AH 0 (margine) su OOF."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "data" / "models"
CONF_PATH = MODELS / "conformal.json"


def _q(scores: np.ndarray, alpha: float) -> float:
    return float(np.quantile(scores, 1.0 - alpha, method="higher"))


def fit_conformal(
    proba: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 0.10,
    leagues: np.ndarray | None = None,
    by_cluster: bool = True,
    home_goals: np.ndarray | None = None,
    away_goals: np.ndarray | None = None,
    lam_home: np.ndarray | None = None,
    lam_away: np.ndarray | None = None,
    p_ou25: np.ndarray | None = None,
    p_ah0: np.ndarray | None = None,
    y_ou25: np.ndarray | None = None,
    y_ah0: np.ndarray | None = None,
) -> dict[str, Any]:
    """Nonconformity 1X2 = 1-p_true; O/U e AH da XGB OOF se presenti, altrimenti Poisson λ."""
    from modules.model_training.league_clusters import cluster_for
    from math import exp, factorial

    valid = np.isfinite(proba).all(axis=1) & np.isfinite(y.astype(float))
    p = np.asarray(proba[valid], dtype=float)
    yy = np.asarray(y[valid], dtype=int)
    if len(yy) < 80:
        return {"ok": False, "error": f"OOF troppo piccolo ({len(yy)})", "alpha": alpha}

    scores = 1.0 - p[np.arange(len(yy)), yy]
    q_global = _q(scores, alpha)

    by: dict[str, float] = {}
    if by_cluster and leagues is not None:
        lg = np.asarray(leagues)[valid]
        buckets: dict[str, list[float]] = {}
        for s, league in zip(scores, lg):
            buckets.setdefault(cluster_for(str(league)), []).append(float(s))
        for cid, vals in buckets.items():
            if len(vals) >= 40:
                by[cid] = _q(np.asarray(vals), alpha)

    covered = float(np.mean(scores <= q_global))

    # --- O/U 2.5 e AH 0: preferisci probabilità OOF dei modelli binari ---
    ou_block: dict[str, Any] = {"ok": False}
    ah_block: dict[str, Any] = {"ok": False}
    if p_ou25 is not None and y_ou25 is not None:
        po = np.asarray(p_ou25, dtype=float)
        yo = np.asarray(y_ou25, dtype=float)
        m = np.isfinite(po) & np.isfinite(yo)
        if int(m.sum()) >= 80:
            ou_scores = []
            for i in np.where(m)[0]:
                p_side = po[i] if yo[i] >= 0.5 else (1.0 - po[i])
                ou_scores.append(1.0 - float(p_side))
            ou_q = _q(np.asarray(ou_scores), alpha)
            ou_block = {
                "ok": True,
                "q": round(ou_q, 5),
                "n": int(len(ou_scores)),
                "coverage_train": round(float(np.mean(np.asarray(ou_scores) <= ou_q)), 4),
                "line": 2.5,
                "source": "xgb_oof",
            }
    if p_ah0 is not None and y_ah0 is not None:
        pa = np.asarray(p_ah0, dtype=float)
        ya = np.asarray(y_ah0, dtype=float)
        m = np.isfinite(pa) & np.isfinite(ya)
        if int(m.sum()) >= 80:
            ah_scores = []
            for i in np.where(m)[0]:
                p_side = pa[i] if ya[i] >= 0.5 else (1.0 - pa[i])
                ah_scores.append(1.0 - float(p_side))
            ah_q = _q(np.asarray(ah_scores), alpha)
            ah_block = {
                "ok": True,
                "q": round(ah_q, 5),
                "n": int(len(ah_scores)),
                "coverage_train": round(float(np.mean(np.asarray(ah_scores) <= ah_q)), 4),
                "line": 0.0,
                "source": "xgb_oof",
            }

    if (
        (not ou_block.get("ok") or not ah_block.get("ok"))
        and home_goals is not None
        and away_goals is not None
        and lam_home is not None
        and lam_away is not None
    ):
        hg = np.asarray(home_goals, dtype=float)[valid]
        ag = np.asarray(away_goals, dtype=float)[valid]
        lh = np.asarray(lam_home, dtype=float)[valid]
        la = np.asarray(lam_away, dtype=float)[valid]
        m = np.isfinite(hg) & np.isfinite(ag) & np.isfinite(lh) & np.isfinite(la) & (lh > 0.05) & (la > 0.05)
        if int(m.sum()) >= 80:

            def pois_ge(k: int, lam: float, max_g: int = 10) -> float:
                # P(goals >= k) approx via CDF
                cdf = 0.0
                for g in range(0, k):
                    cdf += exp(-lam) * lam**g / factorial(g)
                return max(0.0, min(1.0, 1.0 - cdf))

            def p_over25(lam_h: float, lam_a: float) -> float:
                # P(H+A >= 3) via convolution truncated
                tot = 0.0
                for a in range(0, 11):
                    pa = exp(-lam_a) * lam_a**a / factorial(a)
                    # need home >= 3-a
                    need = max(0, 3 - a)
                    ph = pois_ge(need, lam_h) if need > 0 else 1.0
                    if need == 0:
                        ph = 1.0
                    else:
                        cdf = sum(exp(-lam_h) * lam_h**g / factorial(g) for g in range(need))
                        ph = max(0.0, 1.0 - cdf)
                    tot += pa * ph
                return max(0.02, min(0.98, tot))

            def p_ah0_home(lam_h: float, lam_a: float) -> float:
                # P(home wins) ignoring push for conformal binary
                p_h = p_d = p_a = 0.0
                for h in range(0, 9):
                    ph = exp(-lam_h) * lam_h**h / factorial(h)
                    for a in range(0, 9):
                        pa = exp(-lam_a) * lam_a**a / factorial(a)
                        pr = ph * pa
                        if h > a:
                            p_h += pr
                        elif h == a:
                            p_d += pr
                        else:
                            p_a += pr
                s = p_h + p_d + p_a
                return p_h / s if s > 0 else 0.45

            ou_scores = []
            ah_scores = []
            for i in np.where(m)[0]:
                po = p_over25(float(lh[i]), float(la[i]))
                y_over = 1.0 if (hg[i] + ag[i]) > 2.5 else 0.0
                p_side = po if y_over == 1.0 else (1.0 - po)
                ou_scores.append(1.0 - p_side)

                ph = p_ah0_home(float(lh[i]), float(la[i]))
                y_home = 1.0 if hg[i] > ag[i] else 0.0
                p_side_ah = ph if y_home == 1.0 else (1.0 - ph)
                ah_scores.append(1.0 - p_side_ah)

            if not ou_block.get("ok"):
                ou_q = _q(np.asarray(ou_scores), alpha)
                ou_block = {
                    "ok": True,
                    "q": round(ou_q, 5),
                    "n": int(len(ou_scores)),
                    "coverage_train": round(float(np.mean(np.asarray(ou_scores) <= ou_q)), 4),
                    "line": 2.5,
                    "source": "poisson_lambda",
                }
            if not ah_block.get("ok"):
                ah_q = _q(np.asarray(ah_scores), alpha)
                ah_block = {
                    "ok": True,
                    "q": round(ah_q, 5),
                    "n": int(len(ah_scores)),
                    "coverage_train": round(float(np.mean(np.asarray(ah_scores) <= ah_q)), 4),
                    "line": 0.0,
                    "source": "poisson_lambda",
                }

    payload = {
        "ok": True,
        "alpha": alpha,
        "q_global": round(q_global, 5),
        "q_by_cluster": {k: round(v, 5) for k, v in by.items()},
        "n": int(len(yy)),
        "coverage_train": round(covered, 4),
        "method": "aps_like_threshold",
        "ou25": ou_block,
        "ah0": ah_block,
    }
    MODELS.mkdir(parents=True, exist_ok=True)
    CONF_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_conformal() -> dict[str, Any] | None:
    if not CONF_PATH.exists():
        return None
    try:
        data = json.loads(CONF_PATH.read_text(encoding="utf-8"))
        return data if data.get("ok") else None
    except Exception:
        return None


def _band_half(p: float) -> float:
    """Mezza-banda per UI/voto: più larga vicino a 50/50, mai un veto costante."""
    p = min(0.98, max(0.02, float(p)))
    spread = 4.0 * p * (1.0 - p)  # 0..1, max a p=0.5
    return float(min(0.16, max(0.04, 0.04 + 0.12 * spread)))


def predict_interval(
    p_h: float,
    p_d: float,
    p_a: float,
    *,
    league: str | None = None,
    conformal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    conf = conformal if conformal is not None else load_conformal()
    if not conf or not conf.get("ok"):
        return {"ready": False, "method": "none"}

    from modules.model_training.league_clusters import cluster_for

    cid = cluster_for(league)
    q = float((conf.get("q_by_cluster") or {}).get(cid) or conf.get("q_global") or 0.5)
    probs = {"1": float(p_h), "X": float(p_d), "2": float(p_a)}
    thresh = max(0.05, 1.0 - q)
    pred_set = [k for k, v in probs.items() if v >= thresh - 1e-9]
    if not pred_set:
        pred_set = [max(probs, key=probs.get)]
    top = max(probs, key=probs.get)
    bands = {}
    for k, v in probs.items():
        half = _band_half(v)
        bands[k] = {
            "p": round(v, 4),
            "lo": round(max(0.01, v - half), 4),
            "hi": round(min(0.99, v + half), 4),
            "width": round(min(0.40, 2 * half), 4),
        }
    width = bands[top]["width"]
    set_size = len(pred_set)
    return {
        "ready": True,
        "method": "conformal_oof",
        "alpha": conf.get("alpha"),
        "cluster": cid,
        "q": round(q, 5),
        "thresh": round(thresh, 4),
        "set": pred_set,
        "set_size": set_size,
        "top": top,
        "bands": bands,
        "top_width": width,
        # voto/Kelly: set a 1 = stabile; tutti e 3 nel 90% = incerto (NON no_bet)
        "stable": set_size == 1,
        "fragile": set_size >= 3,
        "coverage_train": conf.get("coverage_train"),
        "pick_in_set": True,
    }


def predict_market_interval(
    p_side: float,
    *,
    market: str = "ou25",
    conformal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Set conformal 90% per mercato binario (O/U 2.5 o AH 0 casa)."""
    conf = conformal if conformal is not None else load_conformal()
    block = (conf or {}).get("ou25" if market.startswith("ou") else "ah0") or {}
    if not conf or not block.get("ok"):
        return {"ready": False, "method": "none", "market": market}
    q = float(block.get("q") or 0.5)
    p = float(p_side)
    half = _band_half(p)
    width = min(0.40, 2 * half)
    include = (1.0 - p) <= q + 1e-9
    include_opp = p <= q + 1e-9
    return {
        "ready": True,
        "method": "conformal_oof_market",
        "market": market,
        "q": round(q, 5),
        "p": round(p, 4),
        "lo": round(max(0.01, p - half), 4),
        "hi": round(min(0.99, p + half), 4),
        "width": round(width, 4),
        "include": include,
        "include_opposite": include_opp,
        "stable": include and width <= 0.14,
        "fragile": not include,
        "coverage_train": block.get("coverage_train"),
        "n": block.get("n"),
    }


def attach_market_intervals(
    mc: dict[str, Any] | None,
    *,
    conformal: dict[str, Any] | None = None,
    p_over_25: float | None = None,
    p_ah0_home: float | None = None,
) -> dict[str, Any]:
    """Arricchisce dict Monte Carlo con conformal O/U e AH se p disponibili."""
    out = dict(mc or {})
    conf = conformal if conformal is not None else load_conformal()
    p_over = p_over_25
    if p_over is None:
        p_over = out.get("over_2.5")
    if p_over is None:
        p_over = out.get("over_2_5")
    if p_over is None:
        p_over = out.get("over_25")
    if p_over is not None:
        out["conformal_ou25"] = predict_market_interval(float(p_over), market="ou25", conformal=conf)
    p_ah = p_ah0_home
    if p_ah is None:
        p_ah = out.get("ah_home_0")
    if p_ah is None:
        p_ah = out.get("home_win")
    if p_ah is not None:
        out["conformal_ah0"] = predict_market_interval(float(p_ah), market="ah0", conformal=conf)
    return out
