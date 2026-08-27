"""Train/test, Random Forest e XGBoost, metriche, salvataggio del modello migliore."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
        from modules.calibration.metrics import probability_metrics, simplex_proba

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
            ordered = np.zeros((len(p), 3), dtype=float)
            for src, lab in enumerate(cls):
                ordered[:, int(lab)] = p[:, src]
            ordered = simplex_proba(ordered, 3)
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
        from modules.calibration.metrics import brier_multiclass, expected_calibration_error, simplex_proba

        p = simplex_proba(proba, 3)
        cal = expected_calibration_error(p, y_true)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "log_loss": float(log_loss(y_true, p, labels=[0, 1, 2])),
            "auc_ovr": float(roc_auc_score(y_true, p, multi_class="ovr")),
            "brier": brier_multiclass(y_true, p),
            "ece": cal["ece"],
            "calibration_gap": cal["calibration_gap"],
        }

    def train(self, feat: pd.DataFrame) -> dict:
        from modules.calibration.conformal import fit_conformal
        from modules.model_training.league_clusters import GLOBAL, MIN_ROWS_CLUSTER, build_stat_profiles, cluster_for

        try:
            build_stat_profiles(feat)
        except Exception:
            pass

        rolling = {}
        try:
            rolling = self.rolling_evaluate(feat)
        except Exception as exc:
            rolling = {"error": str(exc)}

        market_info: dict = {}
        try:
            from modules.model_training.market_models import train_market_models

            market_info = train_market_models(feat)
        except Exception as exc:
            market_info = {"ok": False, "error": str(exc)}
            print(f"skip market models O/U-AH: {exc}")

        # Conformal su OOF (1X2 + O/U/AH da modelli binari se disponibili)
        conf_info: dict = {}
        try:
            oof = load_oof()
            p_ou = p_ah = y_ou = y_ah = None
            try:
                oof_m = joblib.load(MODELS / "oof_market.joblib")
                p_ou = oof_m.get("p_ou25")
                p_ah = oof_m.get("p_ah0")
                y_ou = oof_m.get("y_ou25")
                y_ah = oof_m.get("y_ah0")
            except Exception:
                pass
            if oof and oof.get("proba") is not None:
                conf_info = fit_conformal(
                    np.asarray(oof["proba"]),
                    np.asarray(oof["y"]),
                    leagues=oof.get("league"),
                    by_cluster=True,
                    home_goals=oof.get("home_goals"),
                    away_goals=oof.get("away_goals"),
                    lam_home=oof.get("home_xg_avg"),
                    lam_away=oof.get("away_xg_avg"),
                    p_ou25=p_ou,
                    p_ah0=p_ah,
                    y_ou25=y_ou,
                    y_ah0=y_ah,
                )
        except Exception as exc:
            conf_info = {"ok": False, "error": str(exc)}

        weights_info: dict = {}
        try:
            from modules.advisor.data_signal_weights import optimize_weights

            weights_info = optimize_weights()
        except Exception as exc:
            weights_info = {"ok": False, "error": str(exc)}

        x_train, x_test, y_train, y_test, encoder = self.split(feat)
        x_cols = FeatureEngineer.FEATURE_COLS

        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=self.random_state,
            n_jobs=-1,
        )
        xgb = self._xgb(n_estimators=350)

        candidates = {"random_forest": rf, "xgboost": xgb}
        report = {}
        best_name, best_loss, best_model = "", float("inf"), None

        for name, model in candidates.items():
            model.fit(x_train, y_train)
            proba = model.predict_proba(x_test)
            pred = proba.argmax(axis=1)
            metrics = self._metrics(y_test, proba, pred)
            report[name] = metrics
            if metrics["log_loss"] < best_loss:
                best_loss = metrics["log_loss"]
                best_name = name
                best_model = model

        # Modelli per cluster (solo XGB, se abbastanza righe)
        feat_sorted = self._with_feature_cols(feat).sort_values("date").reset_index(drop=True)
        cluster_models: dict[str, Any] = {}
        cluster_metrics: dict[str, Any] = {}
        if "league" in feat_sorted.columns:
            feat_sorted["_cluster"] = feat_sorted["league"].map(lambda lg: cluster_for(str(lg)))
            for cid, sub in feat_sorted.groupby("_cluster"):
                if cid == GLOBAL or len(sub) < MIN_ROWS_CLUSTER:
                    continue
                cut = int(len(sub) * (1 - self.test_size))
                if cut < 200 or len(sub) - cut < 40:
                    continue
                tr, te = sub.iloc[:cut], sub.iloc[cut:]
                ytr = encoder.transform(tr["result"])
                yte = encoder.transform(te["result"])
                m = self._xgb(n_estimators=280)
                m.fit(tr[x_cols], ytr)
                proba = m.predict_proba(te[x_cols])
                # reorder to encoder classes
                cls = list(m.classes_)
                ordered = np.zeros_like(proba)
                for src, lab in enumerate(cls):
                    ordered[:, int(lab)] = proba[:, src]
                pred = ordered.argmax(axis=1)
                cluster_models[str(cid)] = m
                cluster_metrics[str(cid)] = {
                    **self._metrics(yte, ordered, pred),
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                }
                print(f"cluster model {cid}: n={len(sub)} log_loss={cluster_metrics[cid]['log_loss']:.4f}")

        bundle = {
            "model": best_model,
            "models": cluster_models,
            "encoder": encoder,
            "feature_cols": FeatureEngineer.FEATURE_COLS,
            "best_name": best_name,
            "metrics": report,
            "cluster_metrics": cluster_metrics,
        }
        model_path = MODELS / "best_model.joblib"
        joblib.dump(bundle, model_path)
        meta_path = MODELS / "metrics.json"
        meta_path.write_text(
            json.dumps(
                {
                    "best": best_name,
                    "metrics": report,
                    "cluster_metrics": cluster_metrics,
                    "n_clusters": len(cluster_models),
                    "n_train": int(len(x_train)),
                    "n_test": int(len(x_test)),
                    "conformal": {k: conf_info.get(k) for k in ("ok", "n", "coverage_train", "q_global", "error") if k in conf_info},
                    "rolling": {k: rolling.get(k) for k in ("overall", "folds", "n_oof", "error") if k in rolling},
                    "market_models": {
                        k: market_info.get(k)
                        for k in ("ok", "path", "error")
                        if k in market_info
                    }
                    | {
                        "ou25_oof_ll": ((market_info.get("metrics") or {}).get("ou25") or {}).get("oof", {}).get("log_loss"),
                        "ah0_oof_ll": ((market_info.get("metrics") or {}).get("ah0") or {}).get("oof", {}).get("log_loss"),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "best": best_name,
            "metrics": report,
            "path": str(model_path),
            "rolling": rolling,
            "n_clusters": len(cluster_models),
            "cluster_metrics": cluster_metrics,
            "conformal": conf_info,
            "data_signal_weights": weights_info,
            "market_models": market_info,
        }
