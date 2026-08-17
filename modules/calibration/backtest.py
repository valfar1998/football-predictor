"""Backtest storico: taratura soglie EV e calibrazione O/U."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from modules.calibration.calibrate import apply_temperature
from modules.calibration.config import DEFAULTS
from modules.model_training import ModelTrainer

ROOT = Path(__file__).resolve().parents[2]


def _lambdas_from_row(row: pd.Series) -> tuple[float, float]:
    lam_h = float(max(0.35, row["home_xg_avg"] * 0.7 + (1.35 - row["away_xga_avg"]) * 0.15 + 0.25))
    lam_a = float(max(0.25, row["away_xg_avg"] * 0.7 + (1.15 - row["home_xga_avg"]) * 0.15))
    return lam_h, lam_a


def _p_over25(lam_h: float, lam_a: float, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    hg = rng.poisson(lam_h, 1200)
    ag = rng.poisson(lam_a, 1200)
    return float((hg + ag > 2.5).mean())


def _reliability_bins(probs: np.ndarray, hits: np.ndarray, n_bins: int = 8) -> list[dict]:
    df = pd.DataFrame({"p": probs, "hit": hits}).dropna()
    if len(df) < n_bins * 8:
        return []
    try:
        df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    except ValueError:
        return []
    out: list[dict] = []
    for label, grp in df.groupby("bin", observed=True):
        pred = float(grp["p"].mean())
        actual = float(grp["hit"].mean())
        factor = actual / pred if pred > 0.02 else 1.0
        out.append(
            {
                "range": [float(label.left), float(label.right)],
                "predicted": round(pred, 4),
                "actual": round(actual, 4),
                "n": int(len(grp)),
                "factor": round(min(1.35, max(0.65, factor)), 4),
            }
        )
    return out


def run_backtest(feat: pd.DataFrame, matches: pd.DataFrame, *, temperature: float = 1.0) -> dict:
    import joblib

    bundle = joblib.load(ROOT / "data" / "models" / "best_model.joblib")
    model = bundle["model"]
    encoder = bundle["encoder"]
    cols = bundle["feature_cols"]

    trainer = ModelTrainer()
    feat = feat.sort_values("date")
    cut = int(len(feat) * (1 - trainer.test_size))
    test_df = feat.iloc[cut:].reset_index(drop=True)

    proba = model.predict_proba(test_df[cols])
    if temperature != 1.0:
        proba = apply_temperature(proba, temperature)

    cls_map = {c: i for i, c in enumerate(encoder.classes_)}
    p_h = proba[:, cls_map["H"]]
    p_d = proba[:, cls_map["D"]]
    p_a = proba[:, cls_map["A"]]

    m = matches.copy()
    m["date"] = pd.to_datetime(m["date"]).dt.normalize()
    test_df["date"] = pd.to_datetime(test_df["date"]).dt.normalize()
    test_df["_idx"] = np.arange(len(test_df))

    joined = test_df.merge(
        m[
            [
                "date",
                "home_team",
                "away_team",
                "odd_home",
                "odd_draw",
                "odd_away",
                "odd_over_25",
                "odd_under_25",
            ]
        ],
        on=["date", "home_team", "away_team"],
        how="inner",
    )

    bets: list[dict] = []
    ou_probs: list[float] = []
    ou_hits: list[float] = []

    for _, row in joined.iterrows():
        idx = int(row["_idx"])
        lam_h, lam_a = _lambdas_from_row(row)
        p_over = _p_over25(lam_h, lam_a, seed=idx)
        total_goals = float(row["home_goals"] + row["away_goals"])
        ou_probs.append(p_over)
        ou_hits.append(1.0 if total_goals > 2.5 else 0.0)

        outcomes = [
            ("1", float(p_h[idx]), row.get("odd_home"), row["result"] == "H", 0.28),
            ("X", float(p_d[idx]), row.get("odd_draw"), row["result"] == "D", 0.28),
            ("2", float(p_a[idx]), row.get("odd_away"), row["result"] == "A", 0.28),
            ("O2.5", p_over, row.get("odd_over_25"), total_goals > 2.5, 0.42),
            ("U2.5", 1 - p_over, row.get("odd_under_25"), total_goals <= 2.5, 0.42),
        ]
        for code, prob, odd, won, min_p in outcomes:
            if odd is None or pd.isna(odd) or float(odd) <= 1.01:
                continue
            prob = float(prob)
            odd = float(odd)
            if prob < min_p:
                continue
            ev = prob * odd - 1.0
            bets.append({"code": code, "prob": prob, "odd": odd, "ev": ev, "won": bool(won)})

    bets_df = pd.DataFrame(bets)
    ev_grid: dict[str, dict] = {}
    if not bets_df.empty:
        for thr in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]:
            sub = bets_df[bets_df["ev"] >= thr]
            if len(sub) < 50:
                continue
            roi = float((sub["won"].astype(float) * sub["odd"] - 1.0).mean())
            hit = float(sub["won"].mean())
            ev_grid[str(thr)] = {"n": int(len(sub)), "roi": round(roi, 4), "hit_rate": round(hit, 4)}

    best_thr = DEFAULTS["min_ev_play"]
    best_roi = -999.0
    for thr, stats in ev_grid.items():
        if stats["n"] >= 100 and stats["roi"] > best_roi:
            best_roi = stats["roi"]
            best_thr = float(thr)

    rel_ou = _reliability_bins(np.array(ou_probs), np.array(ou_hits))

    pos = bets_df[bets_df["ev"] > 0] if not bets_df.empty else bets_df
    summary = {
        "n_test_matches": int(len(test_df)),
        "n_with_odds": int(len(joined)),
        "n_bets_evaluated": int(len(bets_df)),
        "ev_grid": ev_grid,
        "recommended_min_ev": best_thr,
        "recommended_min_ev_roi": round(best_roi, 4) if best_roi > -999 else None,
        "hit_rate_all_positive_ev": round(float(pos["won"].mean()), 4) if len(pos) else None,
    }
    return {"backtest_summary": summary, "reliability_ou25": rel_ou, "min_ev_play": best_thr}
