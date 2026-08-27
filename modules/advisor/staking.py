"""Kelly frazionato con cap e filtri di giocabilità."""

from __future__ import annotations

from typing import Any

KELLY_FRACTION = 0.25
KELLY_CAP = 0.02
MIN_EDGE = 0.025
MIN_PROB_1X2_PLAY = 0.32  # sotto: EV può esserci, ma il pick non è seguibile
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
    risk_scale: float = 1.0,
) -> float:
    """Frazione di bankroll: ¼ Kelly, tetto sul bankroll, scala opzionale (drawdown guard)."""
    stake = kelly_full(prob, odds) * fraction
    if stake <= 0:
        return 0.0
    scale = max(0.45, min(1.0, float(risk_scale)))
    return float(min(stake * scale, cap * scale))


def kelly_risk_scale(*, max_drawdown: float | None, sharpe: float | None) -> float:
    """Riduce stake se drawdown profondo o Sharpe negativo recente."""
    scale = 1.0
    if max_drawdown is not None and max_drawdown < -0.08:
        scale *= max(0.55, 1.0 + float(max_drawdown) * 1.5)
    if sharpe is not None:
        if sharpe < -0.2:
            scale *= 0.82
        elif sharpe < 0:
            scale *= 0.92
        elif sharpe > 0.8:
            scale = min(1.0, scale * 1.04)
    return max(0.45, min(1.0, scale))


def kelly_risk_scale_from_history(*, kelly_frac: float = KELLY_FRACTION) -> float:
    """Drawdown guard da paper equity su righe trainable con quota."""
    try:
        from modules.advisor.paper_stats import kelly_equity_snapshot
    except Exception:
        return 1.0
    snap = kelly_equity_snapshot(kelly_frac=kelly_frac)
    if not snap.get("ok"):
        return 1.0
    return kelly_risk_scale(
        max_drawdown=snap.get("max_drawdown"),
        sharpe=snap.get("sharpe"),
    )


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


def _side_in_binary_set(miv: dict[str, Any] | None, *, want_primary: bool) -> bool | None:
    """True se il lato è nel set conformal 90%. None se il blocco non è pronto."""
    if not isinstance(miv, dict) or not miv.get("ready"):
        return None
    if want_primary:
        if "include" in miv:
            return bool(miv["include"])
        p = miv.get("p")
        q = miv.get("q")
        if p is None or q is None:
            return None
        return (1.0 - float(p)) <= float(q) + 1e-9
    if "include_opposite" in miv:
        return bool(miv["include_opposite"])
    p = miv.get("p")
    q = miv.get("q")
    if p is None or q is None:
        return None
    return float(p) <= float(q) + 1e-9


def _market_set_block(
    code: str,
    group: str,
    ou_iv: dict[str, Any] | None,
    ah_iv: dict[str, Any] | None,
) -> str | None:
    """Veto conformal solo sul mercato del pick (O2.5/U2.5 o AH0), se il lato è fuori set."""
    g = (group or "").lower()
    c = str(code or "")
    if g == "ou" and c in {"O2.5", "U2.5"}:
        inside = _side_in_binary_set(ou_iv, want_primary=(c == "O2.5"))
        if inside is False:
            return f"{c} fuori dal set conformal O/U 2.5"
        return None
    if g == "ah" and c in {"AH0 1", "AH0 2"}:
        inside = _side_in_binary_set(ah_iv, want_primary=(c == "AH0 1"))
        if inside is False:
            return f"{c} fuori dal set conformal AH 0"
        return None
    return None


def no_bet_reasons(
    play: dict[str, Any],
    *,
    market_move: dict[str, Any] | None = None,
    alignment: dict[str, Any] | None = None,
    min_edge: float = MIN_EDGE,
    min_rank: int = LIQUID_AGAINST_RANK,
    min_pp: float = LIQUID_AGAINST_PP,
    sharp_ev: float | None = None,
    agreement: dict[str, Any] | None = None,
    prob_intervals: dict[str, Any] | None = None,
    residual: dict[str, Any] | None = None,
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
    p_play = play.get("p_cons")
    if p_play is None:
        p_play = play.get("probability")
    if str(play.get("code") or "") in {"1", "X", "2"} and p_play is not None:
        if float(p_play) < MIN_PROB_1X2_PLAY:
            reasons.append(
                f"probabilità 1X2 {float(p_play):.0%} sotto il minimo giocabile {MIN_PROB_1X2_PLAY:.0%}"
            )
    if sharp_ev is not None and sharp_ev < min_edge:
        reasons.append(f"Pinnacle/sharp non offre edge ({sharp_ev:+.1%})")
    if market_too_liquid_against(play, market_move, alignment, min_rank=min_rank, min_pp=min_pp):
        reasons.append("mercato troppo liquido contrario (steam forte, quota pick allungata)")
    if agreement and agreement.get("block_no_bet"):
        reasons.append(
            "fonti in disaccordo sul pick ("
            + (agreement.get("notes") or ["quadro spezzato"])[0]
            + ")"
        )
    iv = prob_intervals or {}
    code = str(play.get("code") or "")
    group = str(play.get("group") or "")
    # 1X2: veto solo se il pick è fuori dal set 90%. IC largo → voto/Kelly, non no_bet.
    if iv.get("ready") and iv.get("set") and code in {"1", "X", "2"}:
        if code not in (iv.get("set") or []):
            reasons.append(f"pick {code} fuori dal set conformal {iv.get('set')}")
    ou_iv = play.get("conformal_ou25") or iv.get("conformal_ou25") or {}
    ah_iv = play.get("conformal_ah0") or iv.get("conformal_ah0") or {}
    blocked = _market_set_block(code, group, ou_iv, ah_iv)
    if blocked:
        reasons.append(blocked)
    if residual and residual.get("primary_block"):
        reasons.append(
            f"residual EV filtro primario ({residual.get('residual'):+.3f} ≤ soglia)"
        )
    elif residual and residual.get("ready") and residual.get("adj_ev") is not None:
        if float(residual["adj_ev"]) < min_edge and float(ev or 0) >= min_edge:
            reasons.append(
                f"residual EV riduce l'edge a {float(residual['adj_ev']):+.1%} "
                f"({residual.get('note') or 'second-stage'})"
            )
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
