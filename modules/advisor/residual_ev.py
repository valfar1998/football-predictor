"""Second-stage residual EV: walk-forward + modelli per cluster + produzione full."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "data" / "models" / "residual_ev.json"
MIN_FIT = 80
PRODUCTION_GATE = 80
PRIMARY_NO_BET_RESIDUAL = -0.08  # residual molto negativo → filtro primario


def _f(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        x = float(v)
        if x != x:
            return default
        return x
    except (TypeError, ValueError):
        return default


CLUSTER_N = {
    "big5_eng": 0.0,
    "big5_esp": 0.125,
    "big5_ita": 0.25,
    "big5_ger": 0.375,
    "big5_fra": 0.5,
    "serie_b_like": 0.625,
    "latam": 0.75,
    "mls": 0.875,
    "cups_euro": 0.9,
    "global": 0.5,
}


def _cluster_feat(rec: dict[str, Any]) -> float:
    try:
        from modules.model_training.league_clusters import cluster_for

        cid = cluster_for(rec.get("league"))
        return float(CLUSTER_N.get(cid, 0.5))
    except Exception:
        return 0.5


def _features_from_row(row: dict[str, Any]) -> list[float]:
    return [
        _f(row.get("ev_cons")),
        _f(row.get("probability"), 0.33),
        _f(row.get("score_unified") or row.get("score"), 5) / 10.0,
        _f(row.get("agree_share"), 0.5),
        _f(row.get("data_edge")),
        _f(row.get("move_rank")) / 5.0,
        1.0 if str(row.get("pick") or "") in {"1", "X", "2"} else 0.0,
        _cluster_feat(row),
    ]


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 2.0) -> tuple[np.ndarray, float] | None:
    ones = np.ones((len(X), 1))
    Xb = np.hstack([ones, X])
    xtx = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    xty = Xb.T @ y
    try:
        coef = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        return None
    rmse = float(np.sqrt(np.mean((Xb @ coef - y) ** 2)))
    return coef, rmse


def _collect_xy(recs: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    xs, ys, meta = [], [], []
    for rec in recs:
        if rec.get("hit") is None:
            continue
        if rec.get("ev_cons") is None and rec.get("ev") is None:
            continue
        p = _f(rec.get("probability"), 0.0)
        if p <= 0.01:
            continue
        hit = 1.0 if int(rec.get("hit") or 0) == 1 else 0.0
        y = hit - p
        x = _features_from_row(
            {
                "ev_cons": rec.get("ev_cons") if rec.get("ev_cons") is not None else rec.get("ev"),
                "probability": p,
                "score_unified": rec.get("score_unified") or rec.get("score"),
                "agree_share": rec.get("agree_share"),
                "data_edge": rec.get("data_edge"),
                "move_rank": rec.get("move_rank"),
                "pick": rec.get("pick"),
                "league": rec.get("league"),
            }
        )
        xs.append(x)
        ys.append(y)
        meta.append(rec)
    if len(xs) < 20:
        return None
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), meta


def fit_residual_ev(*, min_n: int = MIN_FIT, aggressive: bool = False) -> dict[str, Any]:
    """Walk-forward globale + fit per cluster; salva JSON produzione."""
    try:
        from modules.data_update.history import load_history
        from modules.model_training.league_clusters import cluster_for
        from modules.advisor.learn_policy import replicate_for_fit, split_trainable
    except Exception as exc:
        return {"ok": False, "n": 0, "error": str(exc)}

    settled = sorted(
        [r for r in load_history() if r.get("hit") is not None],
        key=lambda r: str(r.get("date") or ""),
    )
    live_rich, backfill_rich = split_trainable(settled)
    fit_rows, fit_meta = replicate_for_fit(settled)
    recs = sorted(fit_rows, key=lambda r: str(r.get("date") or ""))
    packed = _collect_xy(recs)
    if packed is None:
        return {"ok": False, "n": 0, "error": f"servono >={min_n} settled con EV"}
    X, y, meta = packed
    gate = max(40, min_n // 2) if aggressive and len(live_rich) >= 25 else min_n
    if len(y) < gate:
        return {"ok": False, "n": int(len(y)), "error": f"servono >={gate} settled con EV"}

    # Walk-forward RMSE
    folds = []
    start = max(min_n // 2, int(len(y) * 0.35))
    step = max(20, (len(y) - start) // 4)
    idx = start
    while idx + 15 < len(y):
        fit = _ridge_fit(X[:idx], y[:idx])
        if fit is None:
            break
        coef, _ = fit
        te_x = X[idx : idx + step]
        te_y = y[idx : idx + step]
        if len(te_y) < 10:
            break
        xb = np.hstack([np.ones((len(te_x), 1)), te_x])
        pred = xb @ coef
        folds.append({"n_train": int(idx), "n_test": int(len(te_y)), "rmse": float(np.sqrt(np.mean((pred - te_y) ** 2)))})
        idx += step

    fit_all = _ridge_fit(X, y)
    if fit_all is None:
        return {"ok": False, "n": int(len(y)), "error": "fit numerico fallito"}
    coef, rmse = fit_all
    wf_rmse = float(np.mean([f["rmse"] for f in folds])) if folds else rmse

    by_cluster: dict[str, Any] = {}
    buckets: dict[str, list] = {}
    for rec in meta:
        buckets.setdefault(cluster_for(rec.get("league")), []).append(rec)
    for cid, items in buckets.items():
        packed_c = _collect_xy(items)
        if packed_c is None or len(packed_c[1]) < 40:
            continue
        Xc, yc, _ = packed_c
        fit_c = _ridge_fit(Xc, yc)
        if fit_c is None:
            continue
        cc, rc = fit_c
        by_cluster[cid] = {
            "coef": [float(c) for c in cc],
            "n": int(len(yc)),
            "rmse": float(rc),
        }

    # baseline: predict 0 residual → brier-like on y
    baseline_rmse = float(np.sqrt(np.mean(y**2)))
    improvement = baseline_rmse - wf_rmse

    payload = {
        "ok": True,
        "n": int(len(y)),
        "n_live_rich": len(live_rich),
        "n_backfill_rich": len(backfill_rich),
        "aggressive": aggressive,
        "fit_policy": fit_meta,
        "coef": [float(c) for c in coef],
        "feature_names": [
            "bias",
            "ev_cons",
            "probability",
            "score_n",
            "agree_share",
            "data_edge",
            "move_rank_n",
            "is_1x2",
            "cluster_n",
        ],
        "rmse": float(rmse),
        "wf_rmse": round(wf_rmse, 5),
        "baseline_rmse": round(baseline_rmse, 5),
        "improvement": round(float(improvement), 5),
        "folds": folds,
        "by_cluster": by_cluster,
        "primary_no_bet_threshold": PRIMARY_NO_BET_RESIDUAL,
        "mode": "full_production" if len(y) >= PRODUCTION_GATE and wf_rmse <= 0.55 else "scaffold",
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_residual_model() -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    try:
        return json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def production_ready(model: dict[str, Any] | None = None) -> bool:
    m = model or load_residual_model()
    if not m or not m.get("ok"):
        return False
    rmse = float(m.get("wf_rmse") or m.get("rmse") or 1)
    return int(m.get("n") or 0) >= PRODUCTION_GATE and rmse <= 0.55


def predict_residual(
    play: dict[str, Any],
    *,
    agree_share: float | None = None,
    data_edge: float | None = None,
    move_rank: float | None = None,
    league: str | None = None,
) -> dict[str, Any]:
    """Stima residual (hit - p). Negativo = modello troppo fiducioso."""
    model = load_residual_model()
    if not model or not model.get("ok") or not model.get("coef"):
        return {
            "ready": False,
            "residual": None,
            "adj_ev": None,
            "primary_block": False,
            "note": "residual EV non addestrato (servono >=80 settled)",
        }
    coef = np.asarray(model["coef"], dtype=float)
    cid = None
    try:
        from modules.model_training.league_clusters import cluster_for

        cid = cluster_for(league or play.get("league"))
        cl = (model.get("by_cluster") or {}).get(cid)
        if cl and cl.get("coef") and int(cl.get("n") or 0) >= 40:
            coef = np.asarray(cl["coef"], dtype=float)
    except Exception:
        pass

    x = _features_from_row(
        {
            "ev_cons": play.get("ev_cons") if play.get("ev_cons") is not None else play.get("ev"),
            "probability": play.get("probability") or play.get("p_cons"),
            "score_unified": play.get("score_unified") or play.get("score"),
            "agree_share": agree_share,
            "data_edge": data_edge,
            "move_rank": move_rank,
            "pick": play.get("code"),
        }
    )
    xb = np.asarray([1.0] + x, dtype=float)
    residual = float(xb @ coef)
    residual = max(-0.25, min(0.25, residual))
    ev = play.get("ev_cons")
    if ev is None:
        ev = play.get("ev")
    prod = production_ready(model)
    thr = float(model.get("primary_no_bet_threshold") or PRIMARY_NO_BET_RESIDUAL)
    primary_block = bool(prod and residual <= thr)
    adj = None if ev is None else round(float(ev) + residual * (0.85 if prod else 0.5), 4)
    delta_vote = max(-1.0, min(1.0, residual * (3.0 if prod else 2.0)))
    return {
        "ready": True,
        "production": prod,
        "full": model.get("mode") == "full_production",
        "cluster": cid,
        "residual": round(residual, 4),
        "adj_ev": adj if prod else None,
        "primary_block": primary_block,
        "delta_unified": round(delta_vote, 3),
        "n_train": model.get("n"),
        "rmse": model.get("wf_rmse") or model.get("rmse"),
        "note": (
            f"residual {residual:+.3f} (n={model.get('n')}"
            + (f", {cid}" if cid else "")
            + (", FULL" if prod else ", scaffold")
            + (", PRIMARY_NO_BET" if primary_block else "")
            + ")"
        ),
    }
