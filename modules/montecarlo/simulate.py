"""Simulazione Poisson: 1X2, O/U, BTTS, DC, DNB, gol squadra, clean sheet."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


class MonteCarloSimulator:
    def __init__(
        self,
        n_sims: int = 10_000,
        over_lines: tuple[float, ...] = (0.5, 1.5, 2.5, 3.5, 4.5),
        team_lines: tuple[float, ...] = (0.5, 1.5, 2.5),
        seed: int = 42,
    ) -> None:
        self.n_sims = n_sims
        self.over_lines = over_lines
        self.team_lines = team_lines
        self.seed = seed

    def simulate(
        self,
        lambda_home: float,
        lambda_away: float,
        *,
        n_sims: int | None = None,
        model_probs: dict[str, float] | None = None,
        blend: float = 0.35,
        extras: dict[str, float] | None = None,
    ) -> dict:
        rng = np.random.default_rng(self.seed)
        n = n_sims or self.n_sims
        hg = rng.poisson(lam=lambda_home, size=n)
        ag = rng.poisson(lam=lambda_away, size=n)
        tot = hg + ag
        home_w = hg > ag
        draw = hg == ag
        away_w = hg < ag

        p_h_raw = float(home_w.mean())
        p_d_raw = float(draw.mean())
        p_a_raw = float(away_w.mean())
        p_h, p_d, p_a = p_h_raw, p_d_raw, p_a_raw
        if model_probs:
            p_h = (1 - blend) * p_h + blend * model_probs.get("home_win", p_h)
            p_d = (1 - blend) * p_d + blend * model_probs.get("draw", p_d)
            p_a = (1 - blend) * p_a + blend * model_probs.get("away_win", p_a)
            s = p_h + p_d + p_a
            p_h, p_d, p_a = p_h / s, p_d / s, p_a / s

        over = {f"over_{line}": float((tot > line).mean()) for line in self.over_lines}
        under = {f"under_{line}": float((tot < line).mean()) for line in self.over_lines}
        home_ou = {}
        away_ou = {}
        for line in self.team_lines:
            home_ou[f"home_over_{line}"] = float((hg > line).mean())
            home_ou[f"home_under_{line}"] = float((hg < line).mean())
            away_ou[f"away_over_{line}"] = float((ag > line).mean())
            away_ou[f"away_under_{line}"] = float((ag < line).mean())

        btts = float(((hg > 0) & (ag > 0)).mean())
        btts_yes = (hg > 0) & (ag > 0)
        btts_no = ~btts_yes
        o25 = tot > 2.5
        u25 = tot <= 2.5
        o15 = tot > 1.5
        u15 = tot <= 1.5
        dc_1x_mask = home_w | draw
        dc_x2_mask = draw | away_w
        dc_12_mask = home_w | away_w

        # Multigol / fasce gol totali (mercati bookmaker IT tipici)
        multigol = {
            "mg_0_1": float(((tot >= 0) & (tot <= 1)).mean()),
            "mg_1_2": float(((tot >= 1) & (tot <= 2)).mean()),
            "mg_2_3": float(((tot >= 2) & (tot <= 3)).mean()),
            "mg_3_4": float(((tot >= 3) & (tot <= 4)).mean()),
            "mg_2_4": float(((tot >= 2) & (tot <= 4)).mean()),
            "mg_1_3": float(((tot >= 1) & (tot <= 3)).mean()),
            "mg_0_2": float(((tot >= 0) & (tot <= 2)).mean()),
            "mg_3_plus": float((tot >= 3).mean()),
            "mg_4_plus": float((tot >= 4).mean()),
            "goals_odd": float((tot % 2 == 1).mean()),
            "goals_even": float((tot % 2 == 0).mean()),
        }
        for g in range(0, 7):
            multigol[f"exact_total_{g}"] = float((tot == g).mean())
        multigol["exact_total_7plus"] = float((tot >= 7).mean())

        combos = {
            "combo_1_o25": float((home_w & o25).mean()),
            "combo_1_u25": float((home_w & u25).mean()),
            "combo_x_o25": float((draw & o25).mean()),
            "combo_x_u25": float((draw & u25).mean()),
            "combo_2_o25": float((away_w & o25).mean()),
            "combo_2_u25": float((away_w & u25).mean()),
            "combo_1_o15": float((home_w & o15).mean()),
            "combo_1_u15": float((home_w & u15).mean()),
            "combo_2_o15": float((away_w & o15).mean()),
            "combo_2_u15": float((away_w & u15).mean()),
            "combo_1_gol": float((home_w & btts_yes).mean()),
            "combo_1_nogol": float((home_w & btts_no).mean()),
            "combo_2_gol": float((away_w & btts_yes).mean()),
            "combo_2_nogol": float((away_w & btts_no).mean()),
            "combo_x_gol": float((draw & btts_yes).mean()),
            "combo_x_nogol": float((draw & btts_no).mean()),
            "combo_1x_o25": float((dc_1x_mask & o25).mean()),
            "combo_1x_u25": float((dc_1x_mask & u25).mean()),
            "combo_x2_o25": float((dc_x2_mask & o25).mean()),
            "combo_x2_u25": float((dc_x2_mask & u25).mean()),
            "combo_12_o25": float((dc_12_mask & o25).mean()),
            "combo_12_u25": float((dc_12_mask & u25).mean()),
            # Multigol combo leggere
            "combo_1_mg12": float((home_w & (tot >= 1) & (tot <= 2)).mean()),
            "combo_2_mg12": float((away_w & (tot >= 1) & (tot <= 2)).mean()),
            "combo_gol_o25": float((btts_yes & o25).mean()),
            "combo_nogol_u25": float((btts_no & u25).mean()),
        }
        dnb_den = max(p_h + p_a, 1e-9)
        pairs = list(zip(hg.tolist(), ag.tolist()))
        top = Counter(pairs).most_common(8)
        scorelines = [{"score": f"{h}-{a}", "prob": round(c / n, 4)} for (h, a), c in top]

        # Cartellini / calci d'angolo: Poisson indipendente (λ da contesto o proxy)
        extras_out: dict[str, float] = {}
        meta_src: dict[str, Any] = {}
        if extras:
            lam_cards = float(extras.get("lambda_cards") or 0)
            lam_corners = float(extras.get("lambda_corners") or 0)
            if extras.get("cards_source"):
                meta_src["cards_source"] = extras["cards_source"]
            if extras.get("corners_source"):
                meta_src["corners_source"] = extras["corners_source"]
            if lam_cards > 0.5:
                cards = rng.poisson(lam=lam_cards, size=n)
                for line in (2.5, 3.5, 4.5, 5.5):
                    extras_out[f"cards_over_{line}"] = float((cards > line).mean())
                    extras_out[f"cards_under_{line}"] = float((cards < line).mean())
            if lam_corners > 1.0:
                corners = rng.poisson(lam=lam_corners, size=n)
                for line in (7.5, 8.5, 9.5, 10.5, 11.5):
                    extras_out[f"corners_over_{line}"] = float((corners > line).mean())
                    extras_out[f"corners_under_{line}"] = float((corners < line).mean())

        return {
            "n_sims": n,
            "lambda_home": lambda_home,
            "lambda_away": lambda_away,
            "home_win": round(p_h, 4),
            "draw": round(p_d, 4),
            "away_win": round(p_a, 4),
            "dc_1x": round(p_h + p_d, 4),
            "dc_12": round(p_h + p_a, 4),
            "dc_x2": round(p_d + p_a, 4),
            "dnb_1": round(p_h / dnb_den, 4),
            "dnb_2": round(p_a / dnb_den, 4),
            "btts": round(btts, 4),
            "btts_no": round(1.0 - btts, 4),
            "home_cs": round(float((ag == 0).mean()), 4),
            "away_cs": round(float((hg == 0).mean()), 4),
            "home_win_to_nil": round(float((home_w & (ag == 0)).mean()), 4),
            "away_win_to_nil": round(float((away_w & (hg == 0)).mean()), 4),
            "avg_home_goals": round(float(hg.mean()), 3),
            "avg_away_goals": round(float(ag.mean()), 3),
            "avg_total_goals": round(float(tot.mean()), 3),
            "ah_home_0": round(p_h, 4),
            "ah_away_0": round(p_a, 4),
            "mc_raw": {
                "home_win": round(p_h_raw, 4),
                "draw": round(p_d_raw, 4),
                "away_win": round(p_a_raw, 4),
            },
            "mc_std": {
                "home_win": round(float(np.sqrt(p_h_raw * (1.0 - p_h_raw) / n)), 4),
                "draw": round(float(np.sqrt(p_d_raw * (1.0 - p_d_raw) / n)), 4),
                "away_win": round(float(np.sqrt(p_a_raw * (1.0 - p_a_raw) / n)), 4),
            },
            "prob_intervals": _mc_intervals(hg, ag, n_boot=40, seed=self.seed),
            **{k: round(v, 4) for k, v in over.items()},
            **{k: round(v, 4) for k, v in under.items()},
            **{k: round(v, 4) for k, v in home_ou.items()},
            **{k: round(v, 4) for k, v in away_ou.items()},
            **{k: round(v, 4) for k, v in combos.items()},
            **{k: round(v, 4) for k, v in multigol.items()},
            **{k: round(v, 4) for k, v in extras_out.items()},
            **meta_src,
            "most_likely_scores": scorelines,
        }


def _mc_intervals(
    hg: np.ndarray,
    ag: np.ndarray,
    *,
    n_boot: int = 40,
    seed: int = 42,
    alpha: float = 0.10,
) -> dict[str, Any]:
    """Percentili bootstrap 1X2 (90% se alpha=0.10). Non è conformal full-calibrated."""
    rng = np.random.default_rng(seed + 7)
    n = len(hg)
    if n < 100:
        return {"ready": False}
    ph, pd_, pa = [], [], []
    block = max(50, n // n_boot)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=block)
        h, a = hg[idx], ag[idx]
        ph.append(float((h > a).mean()))
        pd_.append(float((h == a).mean()))
        pa.append(float((h < a).mean()))
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)

    def band(vals: list[float], point: float) -> dict[str, float]:
        arr = np.asarray(vals, dtype=float)
        lo, hi = float(np.percentile(arr, lo_q)), float(np.percentile(arr, hi_q))
        return {
            "p": round(point, 4),
            "lo": round(lo, 4),
            "hi": round(hi, 4),
            "width": round(hi - lo, 4),
        }

    p_h = float((hg > ag).mean())
    p_d = float((hg == ag).mean())
    p_a = float((hg < ag).mean())
    bands = {
        "1": band(ph, p_h),
        "X": band(pd_, p_d),
        "2": band(pa, p_a),
    }
    # Stabilità: pick MC più stretto → più solido
    top = max(bands, key=lambda k: bands[k]["p"])
    width = bands[top]["width"]
    stable = width <= 0.06
    fragile = width >= 0.12
    return {
        "ready": True,
        "alpha": alpha,
        "method": "mc_bootstrap",
        "bands": bands,
        "top": top,
        "top_width": round(width, 4),
        "stable": stable,
        "fragile": fragile,
    }
