"""Simulazione Poisson vettorizzata: 1X2 empirico, over/under, BTTS, scoreline."""

from __future__ import annotations

from collections import Counter

import numpy as np


class MonteCarloSimulator:
    def __init__(self, n_sims: int = 10_000, over_lines: tuple[float, ...] = (1.5, 2.5, 3.5), seed: int = 42) -> None:
        self.n_sims = n_sims
        self.over_lines = over_lines
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
        """Simula n partite. `blend` mescola 1X2 empirico Poisson con le prob. del modello ML."""
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

        # top scoreline (max 20 per report)
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
            "btts": round(float(((hg > 0) & (ag > 0)).mean()), 4),
            "avg_home_goals": round(float(hg.mean()), 3),
            "avg_away_goals": round(float(ag.mean()), 3),
            "avg_total_goals": round(float(tot.mean()), 3),
            **{k: round(v, 4) for k, v in over.items()},
            **{k: round(v, 4) for k, v in under.items()},
            "most_likely_scores": scorelines,
        }
