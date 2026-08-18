"""Brier, log-loss, ECE e CLV su probabilità out-of-fold."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def brier_multiclass(y_true: np.ndarray, proba: np.ndarray, n_classes: int = 3) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(proba, dtype=float), 1e-12, 1.0)
    onehot = np.eye(n_classes)[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def multiclass_log_loss(y_true: np.ndarray, proba: np.ndarray, n_classes: int = 3) -> float:
    return float(log_loss(y_true, proba, labels=list(range(n_classes))))


def expected_calibration_error(
    proba: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """ECE sulla classe predetta (confidence vs accuracy) e gap medio dei bin."""
    p = np.asarray(proba, dtype=float)
    y = np.asarray(y_true, dtype=int)
    conf = p.max(axis=1)
    hit = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    gaps: list[float] = []
    weights: list[float] = []
    n = len(conf)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)
        k = int(mask.sum())
        if k == 0:
            continue
        acc = float(hit[mask].mean())
        mean_conf = float(conf[mask].mean())
        gap = abs(acc - mean_conf)
        w = k / n
        ece += w * gap
        gaps.append(gap)
        weights.append(w)
    cal_gap = float(np.average(gaps, weights=weights)) if gaps else 0.0
    return {
        "ece": round(float(ece), 4),
        "calibration_gap": round(cal_gap, 4),
        "n": int(n),
    }


def probability_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    mask = np.isfinite(p).all(axis=1)
    y, p = y[mask], p[mask]
    if len(y) < 20:
        return {"n": int(len(y))}
    cal = expected_calibration_error(p, y)
    fav = p.argmax(axis=1)
    acc = float((fav == y).mean())
    return {
        "n": int(len(y)),
        "accuracy": round(acc, 4),
        "log_loss": round(multiclass_log_loss(y, p), 4),
        "brier": round(brier_multiclass(y, p), 4),
        "ece": cal["ece"],
        "calibration_gap": cal["calibration_gap"],
    }


def clv_summary(bets: pd.DataFrame) -> dict[str, float | int | None]:
    if bets.empty or "clv" not in bets.columns:
        return {"n_with_close": 0, "mean_clv": None, "beat_close_rate": None}
    sub = bets.dropna(subset=["clv"])
    if sub.empty:
        return {"n_with_close": 0, "mean_clv": None, "beat_close_rate": None}
    beat = sub["beat_close"].astype(float).mean() if "beat_close" in sub.columns else None
    return {
        "n_with_close": int(len(sub)),
        "mean_clv": round(float(sub["clv"].mean()), 4),
        "median_clv": round(float(sub["clv"].median()), 4),
        "beat_close_rate": round(float(beat), 4) if beat is not None else None,
    }
