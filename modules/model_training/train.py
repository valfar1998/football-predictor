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
LABELS = ["H", "D", "A"]


class ModelTrainer:
    def __init__(self, test_size: float = 0.2, random_state: int = 42) -> None:
        self.test_size = test_size
        self.random_state = random_state
        MODELS.mkdir(parents=True, exist_ok=True)

    def split(self, feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder]:
        # split temporale: ultime partite nel test, più realistico di uno shuffle puro
        feat = feat.sort_values("date")
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

    def _metrics(self, y_true: np.ndarray, proba: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "log_loss": float(log_loss(y_true, proba, labels=[0, 1, 2])),
            "auc_ovr": float(roc_auc_score(y_true, proba, multi_class="ovr")),
        }

    def train(self, feat: pd.DataFrame) -> dict:
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
            json.dumps({"best": best_name, "metrics": report, "n_train": int(len(x_train)), "n_test": int(len(x_test))}, indent=2),
            encoding="utf-8",
        )
        return {"best": best_name, "metrics": report, "path": str(model_path)}
