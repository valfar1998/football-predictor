"""Backtest walk-forward: filtri edge/sharp, Kelly ¼ con cap, report per lega e mercato."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from modules.advisor.staking import MIN_EDGE, KELLY_CAP, beat_close, clv_prob, kelly_full, quarter_kelly
from modules.calibration.calibrate import apply_temperature
from modules.calibration.metrics import clv_summary, probability_metrics
from modules.calibration.config import DEFAULTS
from modules.model_training import load_oof, ModelTrainer

ROOT = Path(__file__).resolve().parents[2]


def _lambdas_from_vals(home_xg: float, away_xg: float, home_xga: float, away_xga: float) -> tuple[float, float]:
    lam_h = float(max(0.35, home_xg * 0.7 + (1.35 - away_xga) * 0.15 + 0.25))
    lam_a = float(max(0.25, away_xg * 0.7 + (1.15 - home_xga) * 0.15))
    return lam_h, lam_a


def _p_over25(lam_h: float, lam_a: float, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    hg = rng.poisson(lam_h, 800)
    ag = rng.poisson(lam_a, 800)
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


def _group_report(bets: pd.DataFrame, key: str) -> list[dict]:
    if bets.empty or key not in bets.columns:
        return []
    rows = []
    for name, grp in bets.groupby(key, dropna=False):
        if len(grp) < 25:
            continue
        clv = clv_summary(grp)
        pnl = float(grp["pnl"].sum()) if "pnl" in grp.columns else None
        mean_ev = round(float(grp["ev"].mean()), 4)
        roi = round(float((grp["won"].astype(float) * grp["odd"] - 1.0).mean()), 4)
        realization = None
        if abs(mean_ev) >= 0.005:
            realization = round(float(max(0.15, min(1.15, roi / mean_ev))), 4)
        rows.append(
            {
                key: str(name),
                "n": int(len(grp)),
                "hit_rate": round(float(grp["won"].mean()), 4),
                "roi": roi,
                "mean_ev": mean_ev,
                "realization": realization,
                "mean_clv": clv.get("mean_clv"),
                "beat_close_rate": clv.get("beat_close_rate"),
                "pnl_kelly": round(pnl, 4) if pnl is not None else None,
            }
        )
    rows.sort(key=lambda r: r["n"], reverse=True)
    return rows


def _simulate_bankroll(bets: pd.DataFrame, start: float = 1.0) -> list[dict]:
    if bets.empty:
        return []
    ordered = bets.sort_values("date")
    bank = start
    path = []
    for i, row in enumerate(ordered.itertuples(index=False), start=1):
        stake_frac = float(row.kelly)
        if stake_frac <= 0:
            continue
        stake = bank * stake_frac
        if bool(row.won):
            bank += stake * (float(row.odd) - 1.0)
        else:
            bank -= stake
        if i == 1 or i % max(1, len(ordered) // 80) == 0 or i == len(ordered):
            path.append({"i": i, "date": str(pd.to_datetime(row.date).date()), "bankroll": round(float(bank), 4)})
    return path


def _oof_frame(oof: dict, temperature: float = 1.0) -> pd.DataFrame:
    proba = np.asarray(oof["proba"], dtype=float)
    valid = np.isfinite(proba).all(axis=1)
    if temperature != 1.0:
        cal = np.full_like(proba, np.nan)
        cal[valid] = apply_temperature(proba[valid], temperature)
        proba = cal
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(oof["date"]),
            "home_team": oof["home_team"],
            "away_team": oof["away_team"],
            "league": oof["league"],
            "country": oof.get("country", ["unknown"] * len(valid)),
            "fold": oof["fold"],
            "p_h": proba[:, 0],
            "p_d": proba[:, 1],
            "p_a": proba[:, 2],
            "y": oof["y"],
            "home_goals": oof["home_goals"],
            "away_goals": oof["away_goals"],
            "result": oof["result"],
        }
    )
    for key in ("home_xg_avg", "away_xg_avg", "home_xga_avg", "away_xga_avg"):
        if oof.get(key) is not None:
            df[key] = oof[key]
    return df.loc[valid].reset_index(drop=True)


def _holdout_frame(feat: pd.DataFrame, matches: pd.DataFrame, temperature: float) -> pd.DataFrame:
    import joblib

    bundle = joblib.load(ROOT / "data" / "models" / "best_model.joblib")
    model = bundle["model"]
    encoder = bundle["encoder"]
    cols = bundle["feature_cols"]
    trainer = ModelTrainer()
    feat = feat.sort_values("date").reset_index(drop=True)
    cut = int(len(feat) * (1 - trainer.test_size))
    test_df = feat.iloc[cut:].reset_index(drop=True)
    proba = model.predict_proba(test_df[cols])
    if temperature != 1.0:
        proba = apply_temperature(proba, temperature)
    cls_map = {c: i for i, c in enumerate(encoder.classes_)}
    test_df = test_df.copy()
    test_df["p_h"] = proba[:, cls_map["H"]]
    test_df["p_d"] = proba[:, cls_map["D"]]
    test_df["p_a"] = proba[:, cls_map["A"]]
    test_df["y"] = encoder.transform(test_df["result"])
    test_df["fold"] = 0
    test_df["date"] = pd.to_datetime(test_df["date"]).dt.normalize()
    return test_df


def run_backtest(
    feat: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    temperature: float = 1.0,
    min_edge: float = MIN_EDGE,
    kelly_cap: float = KELLY_CAP,
) -> dict:
    oof = load_oof()
    if oof is not None:
        base = _oof_frame(oof, temperature)
        split_kind = "rolling_oof"
    else:
        base = _holdout_frame(feat, matches, temperature)
        split_kind = "holdout_last20"

    m = matches.copy()
    m["date"] = pd.to_datetime(m["date"]).dt.normalize()
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()
    odds_cols = [
        c
        for c in (
            "odd_home",
            "odd_draw",
            "odd_away",
            "odd_over_25",
            "odd_under_25",
            "odd_home_close",
            "odd_draw_close",
            "odd_away_close",
            "odd_over_25_close",
            "odd_under_25_close",
            "odd_home_sharp",
            "odd_draw_sharp",
            "odd_away_sharp",
        )
        if c in m.columns
    ]
    joined = base.merge(
        m[["date", "home_team", "away_team"] + odds_cols],
        on=["date", "home_team", "away_team"],
        how="left",
        suffixes=("", "_m"),
    )

    bets: list[dict] = []
    ou_probs: list[float] = []
    ou_hits: list[float] = []
    y_true = joined["y"].to_numpy()
    proba = joined[["p_h", "p_d", "p_a"]].to_numpy(dtype=float)

    for i, row in enumerate(joined.itertuples(index=False)):
        p_over = None
        if hasattr(row, "home_xg_avg"):
            lam_h, lam_a = _lambdas_from_vals(
                float(row.home_xg_avg),
                float(row.away_xg_avg),
                float(row.home_xga_avg),
                float(row.away_xga_avg),
            )
            p_over = _p_over25(lam_h, lam_a, seed=i + 7)
            total_goals = float(row.home_goals + row.away_goals)
            ou_probs.append(p_over)
            ou_hits.append(1.0 if total_goals > 2.5 else 0.0)

        outcomes = [
            ("1", "1x2", float(row.p_h), getattr(row, "odd_home", None), getattr(row, "odd_home_close", None), getattr(row, "odd_home_sharp", None), row.result == "H", 0.28),
            ("X", "1x2", float(row.p_d), getattr(row, "odd_draw", None), getattr(row, "odd_draw_close", None), getattr(row, "odd_draw_sharp", None), row.result == "D", 0.28),
            ("2", "1x2", float(row.p_a), getattr(row, "odd_away", None), getattr(row, "odd_away_close", None), getattr(row, "odd_away_sharp", None), row.result == "A", 0.28),
        ]
        if p_over is not None:
            total_goals = float(row.home_goals + row.away_goals)
            outcomes.extend(
                [
                    ("O2.5", "ou", p_over, getattr(row, "odd_over_25", None), getattr(row, "odd_over_25_close", None), None, total_goals > 2.5, 0.42),
                    ("U2.5", "ou", 1 - p_over, getattr(row, "odd_under_25", None), getattr(row, "odd_under_25_close", None), None, total_goals <= 2.5, 0.42),
                ]
            )
        for code, market, prob, odd, close, sharp, won, min_p in outcomes:
            if odd is None or (isinstance(odd, float) and (pd.isna(odd) or odd <= 1.01)):
                continue
            if prob < min_p:
                continue
            odd = float(odd)
            ev = prob * odd - 1.0
            sharp_ev = None
            if sharp is not None and not (isinstance(sharp, float) and pd.isna(sharp)) and float(sharp) > 1.01:
                sharp_ev = prob * float(sharp) - 1.0
            clv = clv_prob(odd, None if close is None or (isinstance(close, float) and pd.isna(close)) else float(close))
            beat = beat_close(odd, None if close is None or (isinstance(close, float) and pd.isna(close)) else float(close))
            bets.append(
                {
                    "date": row.date,
                    "league": getattr(row, "league", "unknown"),
                    "fold": int(getattr(row, "fold", 0)),
                    "code": code,
                    "market": market,
                    "prob": prob,
                    "odd": odd,
                    "ev": ev,
                    "sharp_ev": sharp_ev,
                    "clv": clv,
                    "beat_close": beat,
                    "won": bool(won),
                }
            )

    bets_df = pd.DataFrame(bets)
    playable = bets_df.copy()
    if not playable.empty:
        playable = playable[playable["ev"] >= min_edge]
        if "sharp_ev" in playable.columns:
            sharp_ok = playable["sharp_ev"].isna() | (playable["sharp_ev"] >= min_edge)
            playable = playable[sharp_ok]
        playable = playable.copy()
        playable["kelly"] = [
            quarter_kelly(p, o, cap=kelly_cap) for p, o in zip(playable["prob"], playable["odd"])
        ]
        playable = playable[playable["kelly"] > 0]
        playable["pnl"] = np.where(
            playable["won"],
            playable["kelly"] * (playable["odd"] - 1.0),
            -playable["kelly"],
        )

    ev_grid: dict[str, dict] = {}
    if not bets_df.empty:
        for thr in [0.02, 0.025, 0.03]:
            sub = bets_df[bets_df["ev"] >= thr]
            if "sharp_ev" in sub.columns:
                sub = sub[sub["sharp_ev"].isna() | (sub["sharp_ev"] >= thr)]
            if len(sub) < 50:
                continue
            roi = float((sub["won"].astype(float) * sub["odd"] - 1.0).mean())
            hit = float(sub["won"].mean())
            ev_grid[str(thr)] = {"n": int(len(sub)), "roi": round(roi, 4), "hit_rate": round(hit, 4)}

    best_thr = min_edge
    best_roi = -999.0
    for thr, stats in ev_grid.items():
        if stats["n"] >= 80 and stats["roi"] > best_roi:
            best_roi = stats["roi"]
            best_thr = float(thr)
    best_thr = float(min(0.03, max(0.02, best_thr)))

    rel_ou = _reliability_bins(np.array(ou_probs), np.array(ou_hits)) if ou_probs else []
    prob_metrics = probability_metrics(y_true, proba) if len(joined) else {}
    clv = clv_summary(playable) if not playable.empty else clv_summary(pd.DataFrame())
    bank_path = _simulate_bankroll(playable) if not playable.empty else []
    final_bank = bank_path[-1]["bankroll"] if bank_path else 1.0

    by_league = _group_report(playable, "league") if not playable.empty else []
    by_market = _group_report(playable, "market") if not playable.empty else []
    by_code = _group_report(playable, "code") if not playable.empty else []
    by_fold = _group_report(playable, "fold") if not playable.empty else []

    summary = {
        "split": split_kind,
        "n_test_matches": int(len(joined)),
        "n_bets_evaluated": int(len(bets_df)),
        "n_bets_played": int(len(playable)),
        "min_edge": min_edge,
        "kelly_cap": kelly_cap,
        "ev_grid": ev_grid,
        "recommended_min_ev": best_thr,
        "recommended_min_ev_roi": round(best_roi, 4) if best_roi > -999 else None,
        "hit_rate_played": round(float(playable["won"].mean()), 4) if len(playable) else None,
        "roi_flat": round(float((playable["won"].astype(float) * playable["odd"] - 1.0).mean()), 4) if len(playable) else None,
        "bankroll_final": round(float(final_bank), 4),
        "mean_kelly": round(float(playable["kelly"].mean()), 4) if len(playable) else None,
        **{f"prob_{k}": v for k, v in prob_metrics.items()},
        **{f"clv_{k}": v for k, v in clv.items()},
    }
    return {
        "backtest_summary": summary,
        "reliability_ou25": rel_ou,
        "min_ev_play": best_thr,
        "by_league": by_league,
        "by_market": by_market,
        "by_code": by_code,
        "by_fold": by_fold,
        "bankroll_path": bank_path,
        "kelly_cap": kelly_cap,
    }
