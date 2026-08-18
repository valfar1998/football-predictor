"""Visualizzazioni mplsoccer per confronto squadre (supporto analisi)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from modules.data_update.fbref_context import load_fbref_team_index, lookup_team_context
from modules.data_update.understat_context import load_understat_team_index, lookup_understat_team


def _to_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _scale(v: float | None, lo: float, hi: float, *, invert: bool = False) -> float:
    if v is None:
        return 50.0
    if hi <= lo:
        return 50.0
    x = (v - lo) / (hi - lo)
    x = max(0.0, min(1.0, x))
    if invert:
        x = 1.0 - x
    return round(x * 100.0, 1)


def build_team_profile_values(home: str, away: str) -> tuple[list[str], list[float], list[float], pd.DataFrame]:
    fb_idx = load_fbref_team_index()
    us_idx = load_understat_team_index()
    h_fb = lookup_team_context(home, fb_idx) or {}
    a_fb = lookup_team_context(away, fb_idx) or {}
    h_us = lookup_understat_team(home, us_idx) or {}
    a_us = lookup_understat_team(away, us_idx) or {}

    def values(fb: dict[str, Any], us: dict[str, Any]) -> list[float]:
        xg_for = _to_float(us.get("xg_for"))
        xg_against = _to_float(us.get("xg_against"))
        xg_diff = _to_float(us.get("xg_diff"))
        poss = _to_float(fb.get("poss"))
        ga_p90 = _to_float(fb.get("ga_p90"))
        cards_y = _to_float(fb.get("cards_y"))
        cards_r = _to_float(fb.get("cards_r"))
        discipline = None
        if cards_y is not None or cards_r is not None:
            discipline = (cards_y or 0.0) + 2.0 * (cards_r or 0.0)
        return [
            _scale(xg_for, 0.7, 2.4),  # Attacco xG
            _scale(xg_against, 0.7, 2.2, invert=True),  # Solidita difensiva
            _scale(poss, 35.0, 70.0),  # Possesso
            _scale(ga_p90, 0.7, 2.8),  # Produzione offensiva
            _scale(xg_diff, -0.9, 0.9),  # Bilancio xG
            _scale(discipline, 1.0, 10.0, invert=True),  # Disciplina
        ]

    params = [
        "Attacco xG",
        "Solidita dif.",
        "Possesso",
        "Produzione",
        "Bilancio xG",
        "Disciplina",
    ]
    hv = values(h_fb, h_us)
    av = values(a_fb, a_us)
    table = pd.DataFrame(
        {
            "metrica": params,
            home: hv,
            away: av,
            "vantaggio": [home if h > a else away if a > h else "pari" for h, a in zip(hv, av)],
        }
    )
    return params, hv, av, table


def plot_team_radar(home: str, away: str):
    try:
        from mplsoccer import PyPizza
    except Exception as exc:
        raise RuntimeError(
            "Dipendenze radar mancanti: installa matplotlib e mplsoccer nella .venv del progetto."
        ) from exc

    params, home_values, away_values, table = build_team_profile_values(home, away)
    baker = PyPizza(
        params=params,
        background_color="#ffffff",
        straight_line_color="#e8e8e8",
        straight_line_lw=1,
        last_circle_lw=1,
        last_circle_color="#888888",
        other_circle_lw=0.8,
        other_circle_color="#d9d9d9",
    )
    fig, ax = baker.make_pizza(
        home_values,
        compare_values=away_values,
        figsize=(8, 8),
        kwargs_slices={"facecolor": "#1f77b4", "alpha": 0.45, "edgecolor": "#222222", "linewidth": 1},
        kwargs_compare={"facecolor": "#d62728", "alpha": 0.40, "edgecolor": "#222222", "linewidth": 1},
        kwargs_params={"color": "#111111", "fontsize": 10},
        kwargs_values={"color": "#111111", "fontsize": 9},
    )
    fig.text(0.08, 0.97, home, size=11, color="#1f77b4", weight="bold")
    fig.text(0.78, 0.97, away, size=11, color="#d62728", weight="bold")
    fig.text(0.5, 0.02, "Dati da FBref + Understat (soccerdata) normalizzati 0-100", ha="center", size=9)
    return fig, table

