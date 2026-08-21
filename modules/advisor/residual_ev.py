"""Second-stage: residual EV stimato da storico locale.

Predice quanto l'EV cons. è ottimistico/pessimistico (errore del modello di value).
Usato per filtro/voto — NON riscrive p_cons / EV / Kelly finché n_settled è basso.

Fit: regressione ridge su esiti settled (hit → target = realized_edge - ev_cons).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "data" / "models" / "residual_ev.json"
MIN_FIT = 40


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


def _features_from_row(row: dict[str, Any]) -> list[float]:
    return [
        _f(row.get("ev_cons")),
        _f(row.get("probability"), 0.33),
        _f(row.get("score_unified") or row.get("score"), 5) / 10.0,
        _f(row.get("agree_share"), 0.5),
        _f(row.get("data_edge")),
        _f(row.get("move_rank")) / 5.0,
        1.0 if str(row.get("pick") or "") in {"1", "X", "2"} else 0.0,
    ]


def fit_residual_ev(*, min_n: int = MIN_FIT) -> dict[str, Any]:
    """Allena da SQLite history; salva coefficienti JSON."""
    try:
        from modules.data_update.history import load_history
    except Exception as exc:
        return {"ok": False, "n": 0, "error": str(exc)}

    rows = []
    for rec in load_history():
        if rec.get("hit") is None or rec.get("result") in (None, ""):
            continue
        if rec.get("ev_cons") is None:
            continue
        odds = None
        # realized: se hit, edge ≈ 1/odds - p? Usiamo proxy: hit→+ev, miss→-1 (unit loss proxy)
        # Meglio: residual target = (1 if hit else 0) - probability
        p = _f(rec.get("probability"), 0.0)
        if p <= 0.01:
            continue
        hit = 1.0 if int(rec.get("hit") or 0) == 1 else 0.0
        # residual probability error; correlato a EV overconfidence
        y = hit - p
        x = _features_from_row(
            {
                "ev_cons": rec.get("ev_cons"),
                "probability": p,
                "score_unified": rec.get("score_unified") or rec.get("score"),
                "agree_share": rec.get("agree_share"),
                "data_edge": rec.get("data_edge"),
                "move_rank": rec.get("move_rank"),
                "pick": rec.get("pick"),
            }
        )
        rows.append((x, y))

    if len(rows) < min_n:
        return {"ok": False, "n": len(rows), "error": f"servono ≥{min_n} settled con EV"}

    X = np.asarray([r[0] for r in rows], dtype=float)
    y = np.asarray([r[1] for r in rows], dtype=float)
    # ridge closed form
    lam = 2.0
    ones = np.ones((len(X), 1))
    Xb = np.hstack([ones, X])
    xtx = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    xty = Xb.T @ y
    try:
        coef = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        return {"ok": False, "n": len(rows), "error": "fit numerico fallito"}

    payload = {
        "ok": True,
        "n": len(rows),
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
        ],
        "rmse": float(np.sqrt(np.mean((Xb @ coef - y) ** 2))),
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


def predict_residual(
    play: dict[str, Any],
    *,
    agree_share: float | None = None,
    data_edge: float | None = None,
    move_rank: float | None = None,
) -> dict[str, Any]:
    """Stima residual (hit - p). Negativo = modello troppo fiducioso."""
    model = load_residual_model()
    if not model or not model.get("ok") or not model.get("coef"):
        return {
            "ready": False,
            "residual": None,
            "adj_ev": None,
            "note": "residual EV non addestrato (servono ≥40 settled)",
        }
    coef = np.asarray(model["coef"], dtype=float)
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
    # clamp
    residual = max(-0.25, min(0.25, residual))
    ev = play.get("ev_cons")
    if ev is None:
        ev = play.get("ev")
    adj = None if ev is None else round(float(ev) + residual * 0.5, 4)
    return {
        "ready": True,
        "residual": round(residual, 4),
        "adj_ev": adj,
        "n_train": model.get("n"),
        "rmse": model.get("rmse"),
        "note": f"residual {residual:+.3f} (n={model.get('n')})",
    }
