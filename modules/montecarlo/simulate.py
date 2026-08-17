"""Simulazione Poisson: 1X2, O/U, BTTS, DC, DNB, gol squadra, clean sheet."""

from __future__ import annotations

from collections import Counter

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
    ) -> dict:
        rng = np.random.default_rng(self.seed)
        n = n_sims or self.n_sims
        hg = rng.poisson(lam=lambda_home, size=n)
        ag = rng.poisson(lam=lambda_away, size=n)
        tot = hg + ag
        home_w = hg > ag
        draw = hg == ag
        away_w = hg < ag

        p_h = float(home_w.mean())
        p_d = float(draw.mean())
        p_a = float(away_w.mean())
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
        }
        dnb_den = max(p_h + p_a, 1e-9)
        pairs = list(zip(hg.tolist(), ag.tolist()))
        top = Counter(pairs).most_common(8)
        scorelines = [{"score": f"{h}-{a}", "prob": round(c / n, 4)} for (h, a), c in top]

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
            **{k: round(v, 4) for k, v in over.items()},
            **{k: round(v, 4) for k, v in under.items()},
            **{k: round(v, 4) for k, v in home_ou.items()},
            **{k: round(v, 4) for k, v in away_ou.items()},
            **{k: round(v, 4) for k, v in combos.items()},
            "most_likely_scores": scorelines,
        }
