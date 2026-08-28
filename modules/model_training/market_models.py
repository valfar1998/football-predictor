"""Classificatori XGB dedicati: Over/Under 2.5 e Asian Handicap 0 (casa copre)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

from modules.feature_engineering import FeatureEngineer
from modules.model_training.train import _parallel_jobs
ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "data" / "models"
MARKET_PATH = MODELS / "market_models.joblib"
MARKET_META = MODELS / "market_metrics.json"


def _xgb_bin(random_state: int = 42, n_estimators: int = 280) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


def _labels(feat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    hg = pd.to_numeric(feat["home_goals"], errors="coerce").to_numpy(dtype=float)
    ag = pd.to_numeric(feat["away_goals"], errors="coerce").to_numpy(dtype=float)
    y_ou = ((hg + ag) > 2.5).astype(int)
    # AH 0 casa: vince se home > away (pareggio = push → label 0 per binary cover)
    y_ah = (hg > ag).astype(int)
    return y_ou, y_ah


def _fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    """Temperature su logit binario (minimizza log-loss)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    y = np.asarray(y, dtype=int)
    logit = np.log(p / (1.0 - p))
    best_t, best_loss = 1.0, float("inf")
    for t in np.linspace(0.55, 2.8, 24):
        p2 = 1.0 / (1.0 + np.exp(-logit / t))
        loss = float(log_loss(y, np.column_stack([1.0 - p2, p2]), labels=[0, 1]))
        if loss < best_loss:
            best_loss, best_t = loss, float(t)
    return best_t


def apply_binary_temperature(p: float, temperature: float) -> float:
    t = float(temperature or 1.0)
    p = float(np.clip(p, 1e-5, 1 - 1e-5))
    if abs(t - 1.0) < 1e-6:
        return p
    logit = np.log(p / (1.0 - p))
    return float(1.0 / (1.0 + np.exp(-logit / t)))


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    y = np.asarray(y, dtype=int)
    return {
        "log_loss": float(log_loss(y, np.column_stack([1.0 - p, p]), labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5,
        "base_rate": float(y.mean()),
        "n": int(len(y)),
    }


def _fit_market_fold(
    k: int,
    train_end: int,
    test_start: int,
    test_end: int,
    x_all: np.ndarray,
    y_ou: np.ndarray,
    y_ah: np.ndarray,
    *,
    random_state: int,
) -> tuple[int, int, int, np.ndarray, np.ndarray]:
    m_ou = _xgb_bin(random_state, n_estimators=220)
    m_ah = _xgb_bin(random_state + 1, n_estimators=220)
    m_ou.set_params(n_jobs=max(1, (os.cpu_count() or 4) // 4))
    m_ah.set_params(n_jobs=max(1, (os.cpu_count() or 4) // 4))
    m_ou.fit(x_all[:train_end], y_ou[:train_end])
    m_ah.fit(x_all[:train_end], y_ah[:train_end])
    p_ou = m_ou.predict_proba(x_all[test_start:test_end])[:, 1]
    p_ah = m_ah.predict_proba(x_all[test_start:test_end])[:, 1]
    return k, test_start, test_end, p_ou, p_ah


def train_market_models(
    feat: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    n_folds: int = 5,
) -> dict[str, Any]:
    """Walk-forward OOF + modello finale O/U 2.5 e AH 0. Salva market_models.joblib."""
    from modules.model_training.train import ModelTrainer

    feat = feat.copy()
    for col in FeatureEngineer.FEATURE_COLS:
        if col not in feat.columns:
            feat[col] = 0.0
    feat = feat.sort_values("date").reset_index(drop=True)
    y_ou, y_ah = _labels(feat)
    x_cols = FeatureEngineer.FEATURE_COLS
    n = len(feat)
    trainer = ModelTrainer(test_size=test_size, random_state=random_state)
    folds = trainer.rolling_folds(n, n_folds=n_folds)

    oof_ou = np.full(n, np.nan, dtype=float)
    oof_ah = np.full(n, np.nan, dtype=float)
    x_all = feat[x_cols].to_numpy(dtype=float)
    fold_results = Parallel(n_jobs=_parallel_jobs(len(folds)), prefer="processes")(
        delayed(_fit_market_fold)(
            k, train_end, test_start, test_end, x_all, y_ou, y_ah, random_state=random_state
        )
        for k, (train_end, test_start, test_end) in enumerate(folds)
    )
    for k, test_start, test_end, p_ou, p_ah in sorted(fold_results, key=lambda r: r[0]):
        oof_ou[test_start:test_end] = p_ou
        oof_ah[test_start:test_end] = p_ah
        print(f"market fold {k + 1}/{len(folds)} n_test={test_end - test_start}")

    valid = np.isfinite(oof_ou) & np.isfinite(oof_ah)
    t_ou = _fit_temperature(oof_ou[valid], y_ou[valid]) if valid.sum() >= 80 else 1.0
    t_ah = _fit_temperature(oof_ah[valid], y_ah[valid]) if valid.sum() >= 80 else 1.0
    oof_ou_cal = np.array([apply_binary_temperature(p, t_ou) if np.isfinite(p) else p for p in oof_ou])
    oof_ah_cal = np.array([apply_binary_temperature(p, t_ah) if np.isfinite(p) else p for p in oof_ah])

    cut = int(n * (1 - test_size))
    if n - cut < 50:
        cut = max(int(n * 0.8), n - 50)
    m_ou_f = _xgb_bin(random_state, n_estimators=320)
    m_ah_f = _xgb_bin(random_state + 1, n_estimators=320)
    m_ou_f.fit(feat.iloc[:cut][x_cols], y_ou[:cut])
    m_ah_f.fit(feat.iloc[:cut][x_cols], y_ah[:cut])
    p_ou_te = np.array(
        [apply_binary_temperature(float(p), t_ou) for p in m_ou_f.predict_proba(feat.iloc[cut:][x_cols])[:, 1]]
    )
    p_ah_te = np.array(
        [apply_binary_temperature(float(p), t_ah) for p in m_ah_f.predict_proba(feat.iloc[cut:][x_cols])[:, 1]]
    )

    meta = {
        "ou25": {
            "oof": _metrics(y_ou[valid], oof_ou_cal[valid]),
            "holdout": _metrics(y_ou[cut:], p_ou_te),
            "temperature": round(t_ou, 4),
        },
        "ah0": {
            "oof": _metrics(y_ah[valid], oof_ah_cal[valid]),
            "holdout": _metrics(y_ah[cut:], p_ah_te),
            "temperature": round(t_ah, 4),
        },
        "n_oof": int(valid.sum()),
        "n_train_final": int(cut),
        "n_test_final": int(n - cut),
    }
    bundle = {
        "ou25": m_ou_f,
        "ah0": m_ah_f,
        "feature_cols": list(x_cols),
        "temperature": {"ou25": t_ou, "ah0": t_ah},
        "metrics": meta,
    }
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MARKET_PATH)
    MARKET_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # OOF arrays for conformal (align to feat index)
    oof_payload = {
        "p_ou25": oof_ou_cal,
        "p_ah0": oof_ah_cal,
        "y_ou25": y_ou,
        "y_ah0": y_ah,
        "valid": valid,
        "date": pd.to_datetime(feat["date"]).to_numpy(),
        "league": feat["league"].astype(str).to_numpy() if "league" in feat.columns else None,
        "home_goals": feat["home_goals"].to_numpy(),
        "away_goals": feat["away_goals"].to_numpy(),
    }
    joblib.dump(oof_payload, MODELS / "oof_market.joblib")
    print(
        f"market models: O/U OOF logloss={meta['ou25']['oof']['log_loss']:.4f} "
        f"AH0 OOF logloss={meta['ah0']['oof']['log_loss']:.4f}"
    )
    return {"ok": True, "path": str(MARKET_PATH), "metrics": meta, "oof_path": str(MODELS / "oof_market.joblib")}


def load_market_models(path: str | Path | None = None) -> dict[str, Any] | None:
    p = Path(path) if path else MARKET_PATH
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None


def predict_markets(bundle: dict[str, Any] | None, x_row: pd.DataFrame) -> dict[str, float | None]:
    """x_row: DataFrame 1×feature_cols. Ritorna p_over_25 e p_ah0_home calibrate."""
    if not bundle:
        return {"p_over_25": None, "p_ah0_home": None}
    cols = bundle.get("feature_cols") or FeatureEngineer.FEATURE_COLS
    x = x_row.copy()
    for c in cols:
        if c not in x.columns:
            x[c] = 0.0
    x = x[cols]
    temps = bundle.get("temperature") or {}
    out: dict[str, float | None] = {"p_over_25": None, "p_ah0_home": None}
    try:
        p = float(bundle["ou25"].predict_proba(x)[0, 1])
        out["p_over_25"] = round(apply_binary_temperature(p, float(temps.get("ou25") or 1.0)), 4)
    except Exception:
        pass
    try:
        p = float(bundle["ah0"].predict_proba(x)[0, 1])
        out["p_ah0_home"] = round(apply_binary_temperature(p, float(temps.get("ah0") or 1.0)), 4)
    except Exception:
        pass
    return out
