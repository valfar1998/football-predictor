"""Calibrazione probabilità 1X2 (temperature scaling) e bin di affidabilità."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from modules.calibration.metrics import brier_multiclass, expected_calibration_error, probability_metrics
from modules.model_training import ModelTrainer, load_oof

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "data" / "models"


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(proba: np.ndarray, y_true: np.ndarray) -> float:
    eps = 1e-12
    log_p = np.log(np.clip(proba, eps, 1.0))
    best_t, best_loss = 1.0, float("inf")
    for t in np.linspace(0.55, 2.5, 40):
        cal = _softmax(log_p / t)
        loss = log_loss(y_true, cal, labels=[0, 1, 2])
        if loss < best_loss:
            best_loss, best_t = loss, float(t)
    return best_t


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0 or abs(temperature - 1.0) < 1e-6:
        return proba
    eps = 1e-12
    log_p = np.log(np.clip(proba, eps, 1.0))
    return _softmax(log_p / temperature)


def apply_temperature_dict(p_h: float, p_d: float, p_a: float, temperature: float) -> tuple[float, float, float]:
    arr = apply_temperature(np.array([[p_h, p_d, p_a]]), temperature)[0]
    return float(arr[0]), float(arr[1]), float(arr[2])


def _reliability_bins(probs: np.ndarray, hits: np.ndarray, n_bins: int = 10) -> list[dict]:
    df = pd.DataFrame({"p": probs, "hit": hits})
    df = df.dropna()
    if len(df) < n_bins * 5:
        return []
    try:
        df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    except ValueError:
        return []
    rows: list[dict] = []
    for label, grp in df.groupby("bin", observed=True):
        pred = float(grp["p"].mean())
        actual = float(grp["hit"].mean())
        n = int(len(grp))
        factor = actual / pred if pred > 0.02 else 1.0
        rows.append(
            {
                "range": [float(label.left), float(label.right)],
                "predicted": round(pred, 4),
                "actual": round(actual, 4),
                "n": n,
                "factor": round(min(1.35, max(0.65, factor)), 4),
            }
        )
    return rows


def calibrate_from_features(feat: pd.DataFrame | None = None) -> dict:
    if feat is None:
        feat = pd.read_csv(ROOT / "data" / "processed" / "features.csv", parse_dates=["date"])

    oof = load_oof()
    if oof is not None:
        proba_raw = np.asarray(oof["proba"], dtype=float)
        y_test = np.asarray(oof["y"], dtype=int)
        valid = np.isfinite(proba_raw).all(axis=1)
        proba_test = proba_raw[valid]
        y_test = y_test[valid]
        split_kind = "rolling_oof"
    else:
        bundle = joblib.load(MODELS / "best_model.joblib")
        model = bundle["model"]
        cols = bundle["feature_cols"]
        trainer = ModelTrainer()
        _x_train, x_test, _y_train, y_test, _ = trainer.split(feat)
        proba_test = model.predict_proba(x_test)
        split_kind = "holdout_last20"

    temperature = fit_temperature(proba_test, y_test)
    cal_proba = apply_temperature(proba_test, temperature)

    fav_idx = proba_test.argmax(axis=1)
    fav_p_raw = proba_test[np.arange(len(proba_test)), fav_idx]
    fav_p_cal = cal_proba[np.arange(len(cal_proba)), fav_idx]
    fav_hit = (fav_idx == y_test).astype(float)

    from sklearn.metrics import brier_score_loss

    brier_raw = float(brier_score_loss(fav_hit, fav_p_raw))
    brier_cal = float(brier_score_loss(fav_hit, fav_p_cal))
    raw_metrics = probability_metrics(y_test, proba_test)
    cal_metrics = probability_metrics(y_test, cal_proba)
    ece_raw = expected_calibration_error(proba_test, y_test)
    ece_cal = expected_calibration_error(cal_proba, y_test)

    hit_h = (y_test == 0).astype(float)
    hit_d = (y_test == 1).astype(float)
    hit_a = (y_test == 2).astype(float)
    rel_1x2 = _reliability_bins(cal_proba.max(axis=1), fav_hit)

    calibrator = {"temperature": temperature, "labels": ["H", "D", "A"]}
    joblib.dump(calibrator, MODELS / "calibrator.joblib")

    payload = {
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "split": split_kind,
        "temperature": temperature,
        "reliability_1x2": rel_1x2,
        "reliability_ou25": [],
        "brier_favorite_raw": round(brier_raw, 4),
        "brier_favorite_calibrated": round(brier_cal, 4),
        "brier_multiclass_raw": round(brier_multiclass(y_test, proba_test), 4),
        "brier_multiclass_calibrated": round(brier_multiclass(y_test, cal_proba), 4),
        "log_loss_raw": raw_metrics.get("log_loss"),
        "log_loss_calibrated": cal_metrics.get("log_loss"),
        "ece_raw": ece_raw.get("ece"),
        "ece_calibrated": ece_cal.get("ece"),
        "calibration_gap_raw": ece_raw.get("calibration_gap"),
        "calibration_gap_calibrated": ece_cal.get("calibration_gap"),
        "n_test": int(len(y_test)),
        "class_rates": {
            "H": round(float(hit_h.mean()), 4),
            "D": round(float(hit_d.mean()), 4),
            "A": round(float(hit_a.mean()), 4),
        },
    }
    return payload
