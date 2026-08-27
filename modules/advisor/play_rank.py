"""Indice gioca: ranking unico per ordinare il calendario (Azione + EV + voto + Kelly)."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def compute_play_rank(play: dict[str, Any]) -> float:
    """0–100: più alto = più sensato giocare. Gioca in cima, no_bet sotto."""
    action = str(play.get("action") or "").strip().lower()
    if action in {"invalido", "n/d"}:
        return 0.0

    try:
        raw_v = play.get("score_unified")
        if raw_v is None:
            raw_v = play.get("score")
        voto = int(raw_v if raw_v is not None else 1)
    except (TypeError, ValueError):
        voto = 1
    voto = max(1, min(10, voto))
    voto_n = (voto - 1) / 9.0

    ev = play.get("ev_cons")
    if ev is None:
        ev = play.get("ev")
    ev_n = _clip((_safe_float(ev, -0.12) + 0.05) / 0.25)
    kelly_n = _clip(_safe_float(play.get("kelly_quarter"), 0.0) / 0.03)

    core = 0.38 * voto_n + 0.37 * ev_n + 0.25 * kelly_n
    score = 100.0 * core

    if action == "gioca":
        score = min(100.0, score + 4.0)
    elif action == "no_bet":
        score = min(28.0, score * 0.32)
    else:
        score = min(12.0, score * 0.15)

    if play.get("odds_real") is False or (ev is None and action == "gioca"):
        score = min(score, 18.0)

    return round(max(0.0, score), 1)


def attach_play_rank(play: dict[str, Any]) -> None:
    play["play_rank"] = compute_play_rank(play)


def ensure_play_rank_df(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola o completa play_rank su un DataFrame calendario."""
    if df.empty:
        return df
    out = df.copy()
    if "play_rank" not in out.columns:
        out["play_rank"] = pd.NA
    missing = out["play_rank"].isna()
    if missing.any():
        out.loc[missing, "play_rank"] = out.loc[missing].apply(
            lambda r: compute_play_rank(r.to_dict()), axis=1
        )
    return out
