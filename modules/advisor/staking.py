"""Kelly frazionato con cap e filtri di giocabilità."""

from __future__ import annotations

from typing import Any

KELLY_FRACTION = 0.25
KELLY_CAP = 0.02
MIN_EDGE = 0.025
LIQUID_AGAINST_RANK = 3  # Forte+
LIQUID_AGAINST_PP = 2.0


def kelly_full(prob: float, odds: float) -> float:
    if odds <= 1.01 or prob <= 0:
        return 0.0
    edge = prob * odds - 1.0
    if edge <= 0:
        return 0.0
    return edge / (odds - 1.0)


def quarter_kelly(
    prob: float,
    odds: float,
    *,
    fraction: float = KELLY_FRACTION,
    cap: float = KELLY_CAP,
) -> float:
    """Frazione di bankroll: ¼ Kelly, tetto sul bankroll."""
    stake = kelly_full(prob, odds) * fraction
    if stake <= 0:
        return 0.0
    return float(min(stake, cap))


def clv_prob(odds_bet: float | None, odds_close: float | None) -> float | None:
    """CLV in probabilità: positivo se la quota presa è migliore della close."""
    if not odds_bet or not odds_close or odds_bet <= 1.01 or odds_close <= 1.01:
        return None
    return round((1.0 / float(odds_close)) - (1.0 / float(odds_bet)), 4)


def beat_close(odds_bet: float | None, odds_close: float | None) -> bool | None:
    if not odds_bet or not odds_close or odds_bet <= 1.01 or odds_close <= 1.01:
        return None
    return float(odds_bet) > float(odds_close) + 0.005


def market_too_liquid_against(
    play: dict[str, Any],
    market_move: dict[str, Any] | None,
    alignment: dict[str, Any] | None,
    *,
    min_rank: int = LIQUID_AGAINST_RANK,
    min_pp: float = LIQUID_AGAINST_PP,
) -> bool:
    """Steam forte contrario e quota del pick allungata: mercato liquido contro di te."""
    if not market_move or not alignment:
        return False
    from modules.data_update.asian_odds import MOVE_RANK

    if (alignment.get("label") or "") != "contrario":
        return False
    lvl = market_move.get("movement_level") or "Stabile"
    if MOVE_RANK.get(lvl, 0) < min_rank:
        return False
    drop = _pick_implied_drop(play, market_move)
    if drop is None:
        return True
    return drop <= -abs(min_pp)


def no_bet_reasons(
    play: dict[str, Any],
    *,
    market_move: dict[str, Any] | None = None,
    alignment: dict[str, Any] | None = None,
    min_edge: float = MIN_EDGE,
    min_rank: int = LIQUID_AGAINST_RANK,
    min_pp: float = LIQUID_AGAINST_PP,
    sharp_ev: float | None = None,
) -> list[str]:
    reasons: list[str] = []
    ev = play.get("ev_cons")
    if ev is None:
        ev = play.get("ev")
    if play.get("odds_real") is False and play.get("odds"):
        reasons.append("quota ipotetica: value non misurabile sul book")
    elif ev is None:
        reasons.append("quota assente: edge non misurabile")
    elif float(ev) < min_edge:
        reasons.append(f"edge stimato {float(ev):+.1%} sotto la soglia {min_edge:.0%}")
    if sharp_ev is not None and sharp_ev < min_edge:
        reasons.append(f"Pinnacle/sharp non offre edge ({sharp_ev:+.1%})")
    if market_too_liquid_against(play, market_move, alignment, min_rank=min_rank, min_pp=min_pp):
        reasons.append("mercato troppo liquido contrario (steam forte, quota pick allungata)")
    return reasons


def _pick_implied_drop(play: dict[str, Any], market_move: dict[str, Any]) -> float | None:
    code = str(play.get("code") or "")
    group = play.get("group") or "1x2"
    if code == "1":
        return market_move.get("drop_1")
    if code == "X":
        return market_move.get("drop_x")
    if code == "2":
        return market_move.get("drop_2")
    if group == "ou" or code.startswith("O"):
        if "U" in code and not code.startswith("O"):
            return market_move.get("drop_under")
        if code.startswith("O") and "GOL" not in code:
            return market_move.get("drop_over")
    if code.startswith("U"):
        return market_move.get("drop_under")
    return None
