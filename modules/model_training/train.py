"""Train/test, Random Forest e XGBoost, metriche, salvataggio del modello migliore."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from modules.feature_engineering import FeatureEngineer

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "data" / "models"
OOF_PATH = MODELS / "oof_predictions.joblib"
LABELS = ["H", "D", "A"]


def load_oof() -> dict | None:
    if not OOF_PATH.exists():
        return None
    return joblib.load(OOF_PATH)


class ModelTrainer:
    def __init__(self, test_size: float = 0.2, random_state: int = 42) -> None:
        self.test_size = test_size
        self.random_state = random_state
        MODELS.mkdir(parents=True, exist_ok=True)

    def split(self, feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder]:
        # split temporale: ultime partite nel test, più realistico di uno shuffle puro
        feat = self._with_feature_cols(feat).sort_values("date")
        cut = int(len(feat) * (1 - self.test_size))
        train_df, test_df = feat.iloc[:cut], feat.iloc[cut:]
        if len(test_df) < 50:
            train_df, test_df = train_test_split(
                feat, test_size=self.test_size, random_state=self.random_state, stratify=feat["result"]
            )
        encoder = LabelEncoder()
        encoder.fit(LABELS)
        x_cols = FeatureEngineer.FEATURE_COLS
        x_train = train_df[x_cols]
        x_test = test_df[x_cols]
        y_train = encoder.transform(train_df["result"])
        y_test = encoder.transform(test_df["result"])
        return x_train, x_test, y_train, y_test, encoder

    @staticmethod
    def _with_feature_cols(feat: pd.DataFrame) -> pd.DataFrame:
        out = feat.copy()
        for col in FeatureEngineer.FEATURE_COLS:
            if col not in out.columns:
                out[col] = 0.0
        return out

    def rolling_folds(
        self,
        n: int,
        n_folds: int = 5,
        min_train_frac: float = 0.45,
    ) -> list[tuple[int, int, int]]:
        """Finestre expanding: (train_end, test_start, test_end)."""
        start = int(n * min_train_frac)
        remain = n - start
        if remain < 80 or n_folds < 2:
            cut = int(n * (1 - self.test_size))
            return [(cut, cut, n)]
        sizes = [remain // n_folds] * n_folds
        for i in range(remain % n_folds):
            sizes[-(i + 1)] += 1
        folds = []
        idx = start
        for sz in sizes:
            if sz < 20:
                idx += sz
                continue
            folds.append((idx, idx, idx + sz))
            idx += sz
        return folds or [(int(n * (1 - self.test_size)), int(n * (1 - self.test_size)), n)]

    def _xgb(self, n_estimators: int = 250) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=self.random_state,
            n_jobs=-1,
        )

    def rolling_evaluate(self, feat: pd.DataFrame, n_folds: int = 5) -> dict:
        """Walk-forward expanding: probabilità OOF per metriche oneste."""
        from modules.calibration.metrics import probability_metrics

        feat = self._with_feature_cols(feat).sort_values("date").reset_index(drop=True)
        encoder = LabelEncoder()
        encoder.fit(LABELS)
        x_cols = FeatureEngineer.FEATURE_COLS
        y_all = encoder.transform(feat["result"])
        n = len(feat)
        proba = np.full((n, 3), np.nan, dtype=float)
        fold_id = np.full(n, -1, dtype=int)
        folds_report: list[dict] = []
        folds = self.rolling_folds(n, n_folds=n_folds)
        for k, (train_end, test_start, test_end) in enumerate(folds):
            train_df = feat.iloc[:train_end]
            test_df = feat.iloc[test_start:test_end]
            y_train = encoder.transform(train_df["result"])
            y_test = encoder.transform(test_df["result"])
            model = self._xgb(n_estimators=220)
            model.fit(train_df[x_cols], y_train)
            p = model.predict_proba(test_df[x_cols])
            # allinea colonne all'ordine H,D,A
            cls = list(model.classes_)
            ordered = np.zeros_like(p)
            for src, lab in enumerate(cls):
                ordered[:, int(lab)] = p[:, src]
            proba[test_start:test_end] = ordered
            fold_id[test_start:test_end] = k
            metrics = probability_metrics(y_test, ordered)
            metrics["fold"] = k
            metrics["train_end"] = str(pd.to_datetime(train_df["date"].max()).date())
            metrics["test_start"] = str(pd.to_datetime(test_df["date"].min()).date())
            metrics["test_end"] = str(pd.to_datetime(test_df["date"].max()).date())
            folds_report.append(metrics)
            print(f"rolling fold {k + 1}/{len(folds)} n_test={len(test_df)} log_loss={metrics.get('log_loss')}")

        valid = np.isfinite(proba).all(axis=1)
        overall = probability_metrics(y_all[valid], proba[valid])
        payload = {
            "proba": proba,
            "y": y_all,
            "fold": fold_id,
            "date": pd.to_datetime(feat["date"]).to_numpy(),
            "home_team": feat["home_team"].astype(str).to_numpy(),
            "away_team": feat["away_team"].astype(str).to_numpy(),
            "league": feat["league"].astype(str).to_numpy() if "league" in feat.columns else np.array(["unknown"] * n),
            "country": feat["country"].astype(str).to_numpy() if "country" in feat.columns else np.array(["unknown"] * n),
            "home_xg_avg": feat["home_xg_avg"].to_numpy() if "home_xg_avg" in feat.columns else None,
            "away_xg_avg": feat["away_xg_avg"].to_numpy() if "away_xg_avg" in feat.columns else None,
            "away_xga_avg": feat["away_xga_avg"].to_numpy() if "away_xga_avg" in feat.columns else None,
            "home_xga_avg": feat["home_xga_avg"].to_numpy() if "home_xga_avg" in feat.columns else None,
            "home_goals": feat["home_goals"].to_numpy(),
            "away_goals": feat["away_goals"].to_numpy(),
            "result": feat["result"].astype(str).to_numpy(),
            "folds": folds_report,
            "overall": overall,
            "n_folds": int(len(folds_report)),
        }
        joblib.dump(payload, OOF_PATH)
        return {"overall": overall, "folds": folds_report, "n_oof": int(valid.sum()), "path": str(OOF_PATH)}

    def _metrics(self, y_true: np.ndarray, proba: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        from modules.calibration.metrics import brier_multiclass, expected_calibration_error

        cal = expected_calibration_error(proba, y_true)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "log_loss": float(log_loss(y_true, proba, labels=[0, 1, 2])),
            "auc_ovr": float(roc_auc_score(y_true, proba, multi_class="ovr")),
            "brier": brier_multiclass(y_true, proba),
            "ece": cal["ece"],
            "calibration_gap": cal["calibration_gap"],
        }

    def train(self, feat: pd.DataFrame) -> dict:
        rolling = {}
        try:
            rolling = self.rolling_evaluate(feat)
        except Exception as exc:
            rolling = {"error": str(exc)}

        x_train, x_test, y_train, y_test, encoder = self.split(feat)

        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=self.random_state,
            n_jobs=-1,
        )
        xgb = XGBClassifier(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=self.random_state,
            n_jobs=-1,
        )

        candidates = {"random_forest": rf, "xgboost": xgb}
        report = {}
        best_name, best_loss, best_model = "", float("inf"), None

        for name, model in candidates.items():
            model.fit(x_train, y_train)
            proba = model.predict_proba(x_test)
            pred = proba.argmax(axis=1)
            metrics = self._metrics(y_test, proba, pred)
            report[name] = metrics
            # log-loss più basso = migliore per probabilità
            if metrics["log_loss"] < best_loss:
                best_loss = metrics["log_loss"]
                best_name = name
                best_model = model

        bundle = {
            "model": best_model,
            "encoder": encoder,
            "feature_cols": FeatureEngineer.FEATURE_COLS,
            "best_name": best_name,
            "metrics": report,
        }
        model_path = MODELS / "best_model.joblib"
        joblib.dump(bundle, model_path)
        meta_path = MODELS / "metrics.json"
        meta_path.write_text(
            json.dumps(
                {
                    "best": best_name,
                    "metrics": report,
                    "n_train": int(len(x_train)),
                    "n_test": int(len(x_test)),
                    "rolling": {k: rolling.get(k) for k in ("overall", "folds", "n_oof", "error") if k in rolling},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"best": best_name, "metrics": report, "path": str(model_path), "rolling": rolling}
