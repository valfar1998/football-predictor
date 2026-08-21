"""Carica il modello e stima 1X2 + lambda gol per due squadre."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modules.dataset_loader.loader import normalize_team
from modules.feature_engineering import FeatureEngineer

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "data" / "models"
PROCESSED = ROOT / "data" / "processed"


def _norm(name: str) -> str:
    return normalize_team(name)


def context_xg(us_row: dict | None, fb_row: dict | None) -> tuple[float, float] | None:
    """xG Understat (for, against); FBref gls/90 solo fallback attacco (ga_p90 è G+A, non xGA)."""

    def _pos(val) -> float | None:
        try:
            num = float(val)
        except (TypeError, ValueError):
            return None
        if num != num or num <= 0.05:
            return None
        return num

    if us_row:
        xf = _pos(us_row.get("xg_for"))
        if xf:
            return xf, _pos(us_row.get("xg_against")) or 0.0
    if fb_row:
        gls = _pos(fb_row.get("gls_p90"))
        if gls:
            return gls, 0.0
    return None


def _json_num(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, (np.integer, int)) and not isinstance(val, bool):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return float(val)
    return val


class MatchPredictor:
    def __init__(self, model_path: str | Path | None = None, features_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path else MODELS / "best_model.joblib"
        self.features_path = Path(features_path) if features_path else PROCESSED / "features.csv"
        bundle = joblib.load(self.model_path)
        self.model = bundle["model"]
        self.encoder = bundle["encoder"]
        self.feature_cols = bundle["feature_cols"]
        self.features = pd.read_csv(self.features_path, parse_dates=["date"])
        self._index_last_seen()

    def _temperature_for(self, league: str | None) -> float:
        try:
            from modules.calibration.config import load_calibration

            cal = load_calibration()
            by = cal.get("temperature_by_league") or {}
            if league and str(league) in by:
                return float(by[str(league)])
            return float(cal.get("temperature", 1.0))
        except Exception:
            return 1.0

    def _calibrate_probs(self, p_h: float, p_d: float, p_a: float, *, league: str | None = None) -> tuple[float, float, float]:
        temperature = self._temperature_for(league)
        if abs(temperature - 1.0) < 1e-6:
            return p_h, p_d, p_a
        try:
            from modules.calibration.calibrate import apply_temperature_dict

            return apply_temperature_dict(p_h, p_d, p_a, temperature)
        except Exception:
            return p_h, p_d, p_a

    def _index_last_seen(self) -> None:
        df = self.features.reset_index(drop=False)
        home = df[["index", "date", "home_team"]].rename(columns={"home_team": "team"})
        away = df[["index", "date", "away_team"]].rename(columns={"away_team": "team"})
        last = pd.concat([home, away], ignore_index=True).sort_values("date").groupby("team").tail(1)
        self.last_idx = dict(zip(last["team"], last["index"]))

    def _latest_row(self, home: str, away: str, kickoff=None) -> pd.Series:
        """Usa l'ultima riga in cui ciascuna squadra compare, poi ricompone un vettore pre-match."""
        from modules.data_update.team_names import known_team_index, resolve_known_team

        idx = known_team_index(self.last_idx.keys())
        home = resolve_known_team(home, idx) or _norm(home)
        away = resolve_known_team(away, idx) or _norm(away)
        if home not in self.last_idx or away not in self.last_idx:
            missing = [t for t in (home, away) if t not in self.last_idx]
            raise KeyError(f"Squadra non nel dataset: {', '.join(missing)}")
        h_last = self.features.iloc[int(self.last_idx[home])]
        a_last = self.features.iloc[int(self.last_idx[away])]

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
        ko = pd.Timestamp(kickoff) if kickoff is not None else pd.Timestamp.now()
        ko = ko.normalize()
        rest_h = int(max((ko - pd.Timestamp(h_last["date"]).normalize()).days, 1))
        rest_a = int(max((ko - pd.Timestamp(a_last["date"]).normalize()).days, 1))
        m7_h = self._matches_7d(home, ko)
        m7_a = self._matches_7d(away, ko)
        payload = {
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
                "month": int(ko.month),
                "weekday": int(ko.weekday()),
                "home_rest_days": rest_h,
                "away_rest_days": rest_a,
                "rest_diff": rest_h - rest_a,
                "home_matches_7d": m7_h,
                "away_matches_7d": m7_a,
                "congestion_diff": m7_h - m7_a,
                "mkt_p_home": 0.34,
                "mkt_p_draw": 0.28,
                "mkt_p_away": 0.38,
                "mkt_overround": 0.0,
                "mkt_has": 0.0,
            }
        return pd.Series(payload)

    def _matches_7d(self, team: str, kickoff: pd.Timestamp) -> int:
        mask = (self.features["home_team"] == team) | (self.features["away_team"] == team)
        dates = pd.to_datetime(self.features.loc[mask, "date"], errors="coerce")
        return int(((dates >= kickoff - pd.Timedelta(days=7)) & (dates < kickoff)).sum())

    def predict(
        self,
        home_team: str,
        away_team: str,
        kickoff=None,
        *,
        league: str | None = None,
        odds: dict | None = None,
        ext_xg_home: tuple[float, float] | None = None,
        ext_xg_away: tuple[float, float] | None = None,
        weather: dict | None = None,
    ) -> dict:
        from modules.data_update.team_names import known_team_index, resolve_known_team
        from modules.predictor.poisson import blend_1x2, dixon_coles_1x2

        idx = known_team_index(self.last_idx.keys())
        home_team = resolve_known_team(home_team, idx) or _norm(home_team)
        away_team = resolve_known_team(away_team, idx) or _norm(away_team)
        row = self._latest_row(home_team, away_team, kickoff=kickoff)
        if odds:
            p1, px, p2, ov, has = FeatureEngineer.implied_1x2(odds)
            row["mkt_p_home"] = p1
            row["mkt_p_draw"] = px
            row["mkt_p_away"] = p2
            row["mkt_overround"] = ov
            row["mkt_has"] = has
        present = [c for c in self.feature_cols if c in row.index]
        x = pd.DataFrame([row[present]])
        for c in self.feature_cols:
            if c not in x.columns:
                x[c] = 0
        x = x[self.feature_cols]
        proba = self.model.predict_proba(x)[0]
        mapping = {cls: float(p) for cls, p in zip(self.encoder.classes_, proba)}
        p_h = mapping.get("H", 0.0)
        p_d = mapping.get("D", 0.0)
        p_a = mapping.get("A", 0.0)

        lam_h = float(max(0.35, row["home_xg_avg"] * 0.7 + (1.35 - row["away_xga_avg"]) * 0.15 + 0.25))
        lam_a = float(max(0.25, row["away_xg_avg"] * 0.7 + (1.15 - row["home_xga_avg"]) * 0.15))
        if ext_xg_home and ext_xg_home[0]:
            lam_h = 0.62 * lam_h + 0.38 * float(ext_xg_home[0])
        if ext_xg_away and ext_xg_away[0]:
            lam_a = 0.62 * lam_a + 0.38 * float(ext_xg_away[0])
        wx_adj = 1.0
        if weather and weather.get("lambda_adj"):
            try:
                wx_adj = float(weather["lambda_adj"])
            except (TypeError, ValueError):
                wx_adj = 1.0
        lam_h = max(0.25, lam_h * wx_adj)
        lam_a = max(0.20, lam_a * wx_adj)

        dc = dixon_coles_1x2(lam_h, lam_a)
        p_h, p_d, p_a = blend_1x2((p_h, p_d, p_a), dc, ml_weight=0.62)
        p_h, p_d, p_a = self._calibrate_probs(p_h, p_d, p_a, league=league)
        total = p_h + p_d + p_a
        p_h, p_d, p_a = p_h / total, p_d / total, p_a / total
        features = {k: _json_num(row[k]) if k in row.index else None for k in self.feature_cols}
        for key in ("mkt_p_home", "mkt_p_draw", "mkt_p_away", "mkt_overround", "mkt_has"):
            if key in row.index:
                features[key] = _json_num(row[key])
        return {
            "home_team": _norm(home_team),
            "away_team": _norm(away_team),
            "home_win": round(p_h, 4),
            "draw": round(p_d, 4),
            "away_win": round(p_a, 4),
            "lambda_home": round(lam_h, 3),
            "lambda_away": round(lam_a, 3),
            "features": features,
            "ensemble": "xgb+dixon-coles",
        }


def list_known_teams(features_path: str | Path | None = None) -> list[str]:
    path = Path(features_path) if features_path else PROCESSED / "features.csv"
    df = pd.read_csv(path, usecols=["home_team", "away_team"])
    return sorted(set(df["home_team"].dropna()) | set(df["away_team"].dropna()))


def list_team_meta(features_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(features_path) if features_path else PROCESSED / "features.csv"
    cols = ["home_team", "away_team", "league"]
    extra = pd.read_csv(path, nrows=0).columns
    if "country" in extra:
        cols.append("country")
    df = pd.read_csv(path, usecols=cols)
    home = df.rename(columns={"home_team": "team"})[["team"] + [c for c in cols if c not in ("home_team", "away_team")]]
    away = df.rename(columns={"away_team": "team"})[["team"] + [c for c in cols if c not in ("home_team", "away_team")]]
    meta = pd.concat([home, away], ignore_index=True).drop_duplicates("team")
    return meta.sort_values("team").reset_index(drop=True)


def predict_match(home_team: str, away_team: str) -> dict:
    return MatchPredictor().predict(home_team, away_team)
