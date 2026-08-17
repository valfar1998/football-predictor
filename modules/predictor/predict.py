"""Carica il modello e stima 1X2 + lambda gol per due squadre."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modules.dataset_loader import TEAM_ALIASES
from modules.feature_engineering import FeatureEngineer

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "data" / "models"
PROCESSED = ROOT / "data" / "processed"


def _norm(name: str) -> str:
    key = " ".join(name.strip().lower().split())
    return TEAM_ALIASES.get(key, name.strip().title())


class MatchPredictor:
    def __init__(self, model_path: str | Path | None = None, features_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path else MODELS / "best_model.joblib"
        self.features_path = Path(features_path) if features_path else PROCESSED / "features.csv"
        bundle = joblib.load(self.model_path)
        self.model = bundle["model"]
        self.encoder = bundle["encoder"]
        self.feature_cols = bundle["feature_cols"]
        self.features = pd.read_csv(self.features_path, parse_dates=["date"])

    def _latest_row(self, home: str, away: str) -> pd.Series:
        """Usa l'ultima riga in cui ciascuna squadra compare, poi ricompone un vettore pre-match."""
        home, away = _norm(home), _norm(away)
        h = self.features[(self.features["home_team"] == home) | (self.features["away_team"] == home)]
        a = self.features[(self.features["home_team"] == away) | (self.features["away_team"] == away)]
        if h.empty or a.empty:
            known = sorted(set(self.features["home_team"]) | set(self.features["away_team"]))
            raise KeyError(f"Squadra non trovata. Disponibili: {known}")

        h_last = h.sort_values("date").iloc[-1]
        a_last = a.sort_values("date").iloc[-1]

        def side(row: pd.Series, team: str, prefix_if_home: str, prefix_if_away: str) -> dict:
            if row["home_team"] == team:
                p = prefix_if_home
                return {
                    "form_pts": row[f"{p}_form_pts"],
                    "form_gd": row[f"{p}_form_gd"],
                    "xg": row[f"{p}_xg_avg"],
                    "xga": row[f"{p}_xga_avg"],
                    "gf": row[f"{p}_gf_avg"],
                    "ga": row[f"{p}_ga_avg"],
                    "elo": row[f"{p}_elo"],
                    "home_wr": row["home_home_wr"] if p == "home" else row["away_away_wr"],
                    "rest": row[f"{p}_rest_days"],
                }
            p = prefix_if_away
            return {
                "form_pts": row[f"{p}_form_pts"],
                "form_gd": row[f"{p}_form_gd"],
                "xg": row[f"{p}_xg_avg"],
                "xga": row[f"{p}_xga_avg"],
                "gf": row[f"{p}_gf_avg"],
                "ga": row[f"{p}_ga_avg"],
                "elo": row[f"{p}_elo"],
                "home_wr": row["home_home_wr"] if p == "home" else row["away_away_wr"],
                "rest": row[f"{p}_rest_days"],
            }

        hs = side(h_last, home, "home", "away")
        aws = side(a_last, away, "home", "away")
        today = pd.Timestamp.now()
        return pd.Series(
            {
                "home_form_pts": hs["form_pts"],
                "away_form_pts": aws["form_pts"],
                "home_form_gd": hs["form_gd"],
                "away_form_gd": aws["form_gd"],
                "home_xg_avg": hs["xg"],
                "away_xg_avg": aws["xg"],
                "home_xga_avg": hs["xga"],
                "away_xga_avg": aws["xga"],
                "xg_diff": hs["xg"] - aws["xg"],
                "xga_diff": hs["xga"] - aws["xga"],
                "home_gf_avg": hs["gf"],
                "away_gf_avg": aws["gf"],
                "home_ga_avg": hs["ga"],
                "away_ga_avg": aws["ga"],
                "home_home_wr": hs["home_wr"] if h_last["home_team"] == home else 0.45,
                "away_away_wr": aws["home_wr"] if a_last["away_team"] == away else 0.30,
                "home_elo": hs["elo"],
                "away_elo": aws["elo"],
                "elo_diff": hs["elo"] - aws["elo"],
                "month": int(today.month),
                "weekday": int(today.weekday()),
                "home_rest_days": hs["rest"],
                "away_rest_days": aws["rest"],
                "rest_diff": hs["rest"] - aws["rest"],
            }
        )

    def predict(self, home_team: str, away_team: str) -> dict:
        row = self._latest_row(home_team, away_team)
        x = pd.DataFrame([row[self.feature_cols]])
        proba = self.model.predict_proba(x)[0]
        # allinea all'ordine encoder (H, D, A)
        mapping = {cls: float(p) for cls, p in zip(self.encoder.classes_, proba)}
        p_h = mapping.get("H", 0.0)
        p_d = mapping.get("D", 0.0)
        p_a = mapping.get("A", 0.0)
        total = p_h + p_d + p_a
        p_h, p_d, p_a = p_h / total, p_d / total, p_a / total
        lam_h = float(max(0.35, row["home_xg_avg"] * 0.7 + (1.35 - row["away_xga_avg"]) * 0.15 + 0.25))
        lam_a = float(max(0.25, row["away_xg_avg"] * 0.7 + (1.15 - row["home_xga_avg"]) * 0.15))
        return {
            "home_team": _norm(home_team),
            "away_team": _norm(away_team),
            "home_win": round(p_h, 4),
            "draw": round(p_d, 4),
            "away_win": round(p_a, 4),
            "lambda_home": round(lam_h, 3),
            "lambda_away": round(lam_a, 3),
            "features": {k: (float(row[k]) if isinstance(row[k], (int, float, np.floating)) else row[k]) for k in self.feature_cols},
        }


def list_known_teams(features_path: str | Path | None = None) -> list[str]:
    path = Path(features_path) if features_path else PROCESSED / "features.csv"
    df = pd.read_csv(path, usecols=["home_team", "away_team"])
    return sorted(set(df["home_team"].dropna()) | set(df["away_team"].dropna()))


def predict_match(home_team: str, away_team: str) -> dict:
    return MatchPredictor().predict(home_team, away_team)
