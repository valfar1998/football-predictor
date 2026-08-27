"""Consiglio su 1X2 e mercati principali (O/U, gol, DC, DNB) con voto 1-10."""

from __future__ import annotations

from math import floor
from typing import Any

from modules.calibration.config import load_calibration, prob_bin_factor
from modules.advisor.staking import (
    KELLY_CAP,
    MIN_EDGE,
    no_bet_reasons,
    quarter_kelly,
)
from modules.advisor.value import PLAY_VALUE_KEYS, enrich_value, prob_score_cap
from modules.advisor.quadro import build_quadro, validation_source
from modules.advisor.validation import apply_to_play, run_validation

# Senza EV/Kelly/edge misurati il voto non può sembrare una giocata forte.
NO_MODEL_MAX_SCORE = 3
WORKFLOW_INCOMPLETE_LO = 0.20
WORKFLOW_INCOMPLETE_HI = 0.30


def _clamp_score(value: float) -> int:
    return int(max(1, min(10, floor(value + 0.5))))


def _value_metrics_missing(play: dict[str, Any] | None) -> bool:
    """True se EV, Kelly o edge non sono misurabili (serve quota reale)."""
    play = play or {}
    if not play.get("odds_real"):
        return True
    if play.get("edge_pp") is None:
        return True
    ev = play.get("ev_cons")
    if ev is None:
        ev = play.get("ev")
    if ev is None:
        return True
    if play.get("kelly_quarter") is None:
        return True
    return False


def _analysis_play(
    *,
    action: str,
    kind: str,
    name: str,
    code: str = "—",
    tipster: dict[str, Any] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Scheletro senza pick: fonti esterne non generano, quote assenti invalidano."""
    return {
        "code": code,
        "name": name,
        "group": "1x2",
        "kind": kind,
        "action": action,
        "score": None,
        "probability": None,
        "ev": None,
        "ev_cons": None,
        "kelly_quarter": None,
        "odds": None,
        "fair_odds": None,
        "odds_real": False,
        "edge_pp": None,
        "p_market": None,
        "p_cons": None,
        "clv": None,
        "tipster": tipster,
        "no_bet_reasons": [reason] if reason else [],
    }


def _fair_odds(prob: float) -> float | None:
    if prob is None or prob <= 0.02:
        return None
    return round(1.0 / min(prob, 0.98), 2)


def score_probability(p: float, p_second: float, p_model: float, baseline: float = 0.3333) -> int:
    span = max(0.05, 0.92 - baseline)
    dominance = max(0.0, min(1.0, (p - baseline) / span))
    gap = max(0.0, min(1.0, (p - p_second) / max(0.15, 1.0 - baseline)))
    agree = 1.0 - min(1.0, abs(p - p_model) / 0.25)
    raw = 1 + 9 * (0.55 * dominance + 0.30 * gap + 0.15 * agree)
    return _clamp_score(raw)


def score_value(prob: float, odds: float) -> int:
    """Fallback grezzo; il voto reale è in enrich_value."""
    ev = prob * odds - 1.0
    if ev >= 0:
        raw = 4 + 6 * min(1.0, ev / 0.20)
    else:
        raw = 4 - 3 * min(1.0, -ev / 0.25)
    return _clamp_score(raw)


def _kelly_fraction(prob: float, odds: float) -> float:
    from modules.advisor.staking import kelly_full

    return kelly_full(prob, odds)


def _capped_kelly(prob: float, odds: float, cal: dict | None = None) -> float:
    cal = cal or load_calibration()
    from modules.advisor.staking import kelly_risk_scale_from_history, quarter_kelly

    return quarter_kelly(
        prob,
        odds,
        fraction=float(cal.get("kelly_fraction", 0.25)),
        cap=float(cal.get("kelly_cap", KELLY_CAP)),
        risk_scale=kelly_risk_scale_from_history(
            kelly_frac=float(cal.get("kelly_fraction", 0.25)),
        ),
    )


def _ml_mc_divergence(mc_prob: float, ml_prob: float | None) -> float:
    if ml_prob is None:
        return 0.0
    return abs(float(mc_prob) - float(ml_prob))


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _src_ready(sources: list[dict[str, Any]], name: str) -> bool:
    return any(s.get("fonte") == name and not s.get("mancante") for s in sources)


def _workflow_norm(
    play: dict[str, Any],
    quadro: dict[str, Any] | None,
    validation: dict[str, Any] | None,
) -> tuple[float, dict[str, bool]]:
    """Workflow = copertura di ML/MC, mercato, EV, Kelly, forma, tattica. Senza modello: 20–30%."""
    val = validation or {}
    src = (quadro or {}).get("sources") or []
    ml_mc = _src_ready(src, "Modello ML") and _src_ready(src, "Monte Carlo")
    if not ml_mc:
        ml_mc = bool((val.get("stability") or {}).get("ready"))
    market = _src_ready(src, "Book (devig)") or bool(play.get("odds_real") and play.get("p_market") is not None)
    if not market:
        market = bool((val.get("market") or {}).get("ready"))
    ev_ok = bool(play.get("odds_real") and (play.get("ev_cons") is not None or play.get("ev") is not None))
    kelly_ok = bool(play.get("odds_real") and play.get("kelly_quarter") is not None)
    form_ok = bool((val.get("form") or {}).get("ready"))
    tac_ok = bool((val.get("tactical") or {}).get("ready"))
    pillars = {
        "ml_mc": ml_mc,
        "market": market,
        "ev": ev_ok,
        "kelly": kelly_ok,
        "form": form_ok,
        "tactics": tac_ok,
    }
    n_ok = sum(1 for v in pillars.values() if v)
    core_ok = ml_mc and market and ev_ok and kelly_ok
    if not core_ok:
        span = WORKFLOW_INCOMPLETE_HI - WORKFLOW_INCOMPLETE_LO
        return round(WORKFLOW_INCOMPLETE_LO + span * (n_ok / 6.0), 3), pillars
    share = _safe_float((quadro or {}).get("agree_share"), 0.50)
    coverage = n_ok / 6.0
    extra = 1.0 if form_ok and tac_ok else 0.5
    return round(_clip01(0.55 * coverage + 0.25 * share + 0.20 * extra), 3), pillars


def _has_asian_signal(market_move: dict[str, Any] | None) -> bool:
    if not market_move:
        return False
    lvl = market_move.get("movement_level")
    return bool(
        market_move.get("steam_1x2")
        or (lvl not in {None, "", "Stabile"})
        or market_move.get("drop_1") is not None
        or market_move.get("odds_moves")
    )


def _meta_analysis(
    play: dict[str, Any],
    *,
    alignment: dict[str, Any] | None,
    market_move: dict[str, Any] | None,
    quadro: dict[str, Any] | None,
    legs: tuple[str, ...] | None = None,
    validation: dict[str, Any] | None = None,
    history_weight: float | None = None,
) -> dict[str, Any]:
    """Unico indicatore: value+kelly+asian+workflow+storico. Gambe assenti restano a 0 (non si ricalcola)."""
    metrics_nd = _value_metrics_missing(play)
    ev = play.get("ev_cons")
    if ev is None and play.get("odds_real"):
        ev = play.get("ev")
    edge = play.get("edge_pp") if play.get("odds_real") else None
    kelly_q = play.get("kelly_quarter") if play.get("odds_real") else None

    if metrics_nd:
        value_norm = 0.0
        kelly_norm = 0.0
    else:
        ev_norm = _clip01((_safe_float(ev, -0.08) + 0.08) / 0.20)
        edge_norm = _clip01((_safe_float(edge, -0.03) + 0.03) / 0.08)
        value_norm = 0.6 * ev_norm + 0.4 * edge_norm
        kelly_norm = _clip01(_safe_float(kelly_q, 0.0) / 0.03)

    asian_ok = _has_asian_signal(market_move)
    if asian_ok:
        move_lvl = (market_move or {}).get("movement_level") or "Stabile"
        lvl_map = {"Stabile": 0.45, "Leggero": 0.55, "Medio": 0.65, "Forte": 0.75, "Fortissimo": 0.82, "Raro": 0.88}
        align_lbl = (alignment or {}).get("label") or "n/d"
        align_map = {"allineato": 0.78, "misto": 0.55, "n/d": 0.50, "contrario": 0.28, "stabile": 0.50}
        asian_norm = 0.65 * align_map.get(align_lbl, 0.50) + 0.35 * lvl_map.get(move_lvl, 0.50)
    else:
        asian_norm = 0.0

    q = quadro or {}
    src = q.get("sources") or []
    workflow_norm, pillars = _workflow_norm(play, q, validation)

    hist_src = next((s for s in src if s.get("fonte") == "Storico locale" and not s.get("mancante")), None)
    hist_ok = hist_src is not None
    if hist_src:
        top = max(
            _safe_float(hist_src.get("p_1"), 0.34),
            _safe_float(hist_src.get("p_x"), 0.34),
            _safe_float(hist_src.get("p_2"), 0.34),
        )
        history_norm = _clip01((top - 0.34) / 0.28)
        play_code = str(play.get("code") or "")
        if play_code in {"1", "X", "2"} and hist_src.get("pick") == play_code:
            history_norm = _clip01(0.65 * history_norm + 0.35)
        elif play_code in {"1", "X", "2"} and hist_src.get("pick") in {"1", "X", "2"}:
            history_norm = _clip01(0.55 * history_norm)
    else:
        history_norm = 0.0

    combo_src = next((s for s in src if s.get("fonte") == "Combo tattica" and not s.get("mancante")), None)
    combos_ok = combo_src is not None
    if combo_src:
        topc = max(
            _safe_float(combo_src.get("p_1"), 0.34),
            _safe_float(combo_src.get("p_x"), 0.34),
            _safe_float(combo_src.get("p_2"), 0.34),
        )
        c12 = _clip01((topc - 0.34) / 0.28)
        play_code = str(play.get("code") or "")
        if play_code in {"1", "X", "2"} and combo_src.get("pick") == play_code:
            c12 = _clip01(0.62 * c12 + 0.38)
        combos_norm = _clip01(0.50 * c12 + 0.30 * value_norm + 0.20 * asian_norm)
    else:
        combos_norm = 0.0

    norms = {
        "value": value_norm,
        "kelly": kelly_norm,
        "asian": asian_norm,
        "workflow": workflow_norm,
        "history": history_norm,
        "combos": combos_norm,
    }
    extra = 0.0
    if hist_ok and history_weight is not None:
        hw = min(0.18, max(0.10, float(history_weight)))
        extra = hw - 0.10
    hw = 0.10 + extra
    weights = {
        "value": 0.28 - extra * 0.5,
        "kelly": 0.16,
        "asian": 0.18,
        "workflow": 0.10,
        "history": hw,
        "combos": 0.18 - extra * 0.5,
    }
    # Gambe assenti restano 0 sul denominatore pieno: niente ricalcolo che gonfia il workflow.
    measured = {
        "value": not metrics_nd,
        "kelly": not metrics_nd,
        "asian": asian_ok,
        "workflow": True,
        "history": hist_ok,
        "combos": combos_ok,
    }
    if legs:
        allowed = set(legs)
        for k in weights:
            if k not in allowed:
                norms[k] = 0.0
                measured[k] = False
    wsum = sum(weights.values()) or 1.0
    final_norm = sum(weights[k] * norms[k] for k in weights) / wsum
    val = validation or {}
    final_score = _clamp_score(1 + 9 * final_norm)
    delta = float(val.get("delta_unified") or 0)
    if delta:
        final_score = int(max(1, min(10, round(final_score + delta))))
    if play.get("action") == "no_bet":
        final_score = min(final_score, 5)
    if metrics_nd or play.get("action") in {"n/d", "invalido"}:
        final_score = min(final_score, NO_MODEL_MAX_SCORE)

    used = [k for k, ok in measured.items() if ok]
    partial = metrics_nd or not all(measured[k] for k in ("value", "kelly", "asian", "workflow"))
    note_bits: list[str] = []
    for k in ("value", "kelly", "asian", "workflow", "history", "combos"):
        if measured[k]:
            note_bits.append(f"{k} {norms[k]:.0%}")
        elif k in {"value", "kelly", "workflow"}:
            note_bits.append(f"{k} n/d")
    note = ("parziale · " if partial else "") + " · ".join(note_bits)
    if val.get("summary"):
        note = (note + " · " if note else "") + str(val["summary"])
    if play.get("action") == "invalido":
        label = "invalido · senza quote"
    elif play.get("action") == "n/d" or metrics_nd:
        label = "n/d · senza modello"
    else:
        label = "solida" if final_score >= 8 else "interessante" if final_score >= 6 else "debole"
    if partial:
        label = f"parziale · {label}"

    return {
        "score": final_score,
        "value": None if not measured["value"] else round(value_norm, 3),
        "kelly": None if not measured["kelly"] else round(kelly_norm, 3),
        "asian": None if not measured["asian"] else round(asian_norm, 3),
        "workflow": round(workflow_norm, 3),
        "history": None if not measured["history"] else round(history_norm, 3),
        "combos": None if not measured["combos"] else round(combos_norm, 3),
        "pillars": pillars,
        "legs": used,
        "partial": partial,
        "label": label,
        "note": note,
        "validation_delta": round(float(val.get("delta_unified") or 0), 3),
    }


def _grouped_from_odds(odds: dict[str, Any] | None) -> dict[str, list]:
    if not odds:
        return {}
    o1, ox, o2 = odds.get("1"), odds.get("X"), odds.get("2")
    try:
        if not (o1 and ox and o2) or min(float(o1), float(ox), float(o2)) <= 1.0:
            return {}
        i1, ix, i2 = 1.0 / float(o1), 1.0 / float(ox), 1.0 / float(o2)
    except (TypeError, ValueError):
        return {}
    s = i1 + ix + i2
    if s <= 0:
        return {}
    return {
        "1x2": [
            {"code": "1", "p_market": i1 / s},
            {"code": "X", "p_market": ix / s},
            {"code": "2", "p_market": i2 / s},
        ]
    }


def _has_real_1x2_odds(odds: dict[str, Any] | None) -> bool:
    if not odds:
        return False
    try:
        o1 = float(odds.get("1") or odds.get("home") or 0)
        ox = float(odds.get("X") or odds.get("draw") or odds.get("x") or 0)
        o2 = float(odds.get("2") or odds.get("away") or 0)
    except (TypeError, ValueError):
        return False
    return min(o1, ox, o2) > 1.0


def advise_uncovered(
    home: str,
    away: str,
    *,
    odds: dict[str, Any] | None = None,
    market_move: dict[str, Any] | None = None,
    prediction: dict[str, Any] | None = None,
    tipster: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analisi senza modello: quadro di validazione. Nessun pick da fonti esterne, niente EV/Kelly."""
    from modules.tipsters import consensus_for

    prediction = prediction or {}
    if tipster is None:
        try:
            tipster = consensus_for(home, away)
        except Exception:
            tipster = {"n_sources": 0, "label": "n/d"}
    if int((tipster or {}).get("n_sources") or 0) < 2:
        tipster = {"n_sources": 0, "label": "n/d"}

    has_odds = _has_real_1x2_odds(odds)
    grouped = _grouped_from_odds(odds) if has_odds else {}
    if has_odds:
        play = _analysis_play(
            action="n/d",
            kind="nessun_pick",
            name="nessun pick (fonti esterne non generano giocata)",
            tipster=tipster,
            reason="senza modello: le fonti esterne validano, non generano pick",
        )
        play["odds_real"] = True
    else:
        play = _analysis_play(
            action="invalido",
            kind="invalido",
            name="pick invalido (quote assenti)",
            tipster=tipster,
            reason="pick invalido: senza quote non si calcolano edge, EV, Kelly, quota equa, CLV",
        )

    quadro = build_quadro(
        home=home,
        away=away,
        play=play,
        prediction=prediction,
        grouped=grouped,
        alignment=None,
        market_move=market_move,
        tipster=tipster,
    )
    alignment = {"agrees": [], "disagrees": [], "delta": 0, "label": "n/d"}

    pred = dict(prediction)
    pred.setdefault("home", home)
    pred.setdefault("away", away)
    validation = run_validation(prediction=pred, play=play, grouped=grouped)
    play = apply_to_play(play, validation)
    quadro = dict(quadro)
    quadro["validation"] = validation
    if validation:
        quadro["sources"] = list(quadro.get("sources") or []) + [validation_source(validation)]
    meta = _meta_analysis(
        play,
        alignment=alignment,
        market_move=market_move,
        quadro=quadro,
        validation=validation,
        history_weight=(prediction.get("history_context") or {}).get("weight"),
    )
    from modules.advisor.pro_scores import annotate_source_weights, build_fallback_source, build_match_scores

    quadro["sources"] = annotate_source_weights(list(quadro.get("sources") or []))
    fb = build_fallback_source(pred, quadro["sources"])
    if fb:
        quadro["sources"] = list(quadro["sources"]) + [fb]
        quadro["fallback"] = True
    pro = build_match_scores(
        play=play,
        prediction=pred,
        quadro=quadro,
        agreement=None,
        validation=validation,
        intervals=None,
        residual=None,
        meta=meta,
        grouped=grouped,
        league=prediction.get("league"),
    )
    quadro = pro.get("quadro") or quadro
    play["score_unified"] = meta["score"]
    play["meta_analysis"] = meta
    play["score_100"] = pro.get("score_100")
    play["confidence_100"] = pro.get("confidence_100")
    play["risk_100"] = pro.get("risk_100")
    play["priority_100"] = pro.get("priority_100")
    play["score_band"] = (pro.get("band") or {}).get("label")
    play["match_scores"] = {
        "unified": pro.get("unified"),
        "confidence": pro.get("confidence"),
        "risk": pro.get("risk"),
        "priority": pro.get("priority"),
        "overrides": pro.get("overrides"),
        "coverage": pro.get("coverage"),
        "bet_rec": pro.get("bet_rec"),
        "weights_table": pro.get("weights_table"),
    }
    play["bet_rec"] = pro.get("bet_rec")
    from modules.advisor.play_rank import attach_play_rank

    attach_play_rank(play)

    reason1 = (
        "Pick invalido: senza quote non si calcolano edge, EV, Kelly, quota equa, CLV."
        if play["action"] == "invalido"
        else "Senza modello non c'è pick: le fonti esterne (ClubElo, FBref, tipster, …) validano, non generano."
    )
    reason2 = quadro.get("summary") or "Nessuna fonte di validazione."
    reason2 = f"{reason2} · Voto unificato {meta['score']}/10 ({meta['note']})"
    if play.get("score_100") is not None:
        reason2 += f" · Score {play['score_100']:.0f}/100 ({play.get('score_band')})"

    return {
        "match": prediction.get("match") or f"{home} vs {away}",
        "home": home,
        "away": away,
        "play": play,
        "quadro": quadro,
        "meta_analysis": play.get("meta_analysis"),
        "market_move": market_move,
        "market_align": alignment,
        "tipster": tipster,
        "score_reason_1": reason1,
        "score_reason_2": reason2,
        "grouped": grouped,
        "all_markets": [],
        "match_scores": play.get("match_scores"),
        "score_100": play.get("score_100"),
        "confidence_100": play.get("confidence_100"),
        "risk_100": play.get("risk_100"),
        "priority_100": play.get("priority_100"),
        "score_band": play.get("score_band"),
        "bet_rec": play.get("bet_rec"),
    }


def _borderline_penalty(prob: float, group: str) -> float:
    """Penalità proporzionale sotto la soglia “comoda” (non un −1 fisso)."""
    if group == "1x2":
        # ~37% → −0.3 · ~33% → −1.2 · ~28% → −2.0
        soft_hi, soft_lo, max_pen = 0.39, 0.28, 2.0
    elif group in {"ou", "btts", "team"}:
        soft_hi, soft_lo, max_pen = 0.52, 0.42, 1.5
    elif group == "combo":
        soft_hi, soft_lo, max_pen = 0.24, 0.16, 1.5
    else:
        soft_hi, soft_lo, max_pen = 0.55, 0.48, 1.0
    if prob >= soft_hi:
        return 0.0
    span = max(1e-6, soft_hi - soft_lo)
    return float(min(max_pen, max_pen * (soft_hi - prob) / span))


def _ml_mc_adjust(divergence: float, *, has_ml: bool) -> float:
    """+0.5 se ML≈MC, −0.5 se divergono; malus più forti oltre 8–12 pp."""
    if not has_ml:
        return 0.0
    if divergence <= 0.04:
        return 0.5
    if divergence > 0.12:
        return -1.5
    if divergence > 0.08:
        return -0.75
    if divergence > 0.05:
        return -0.5
    return 0.0


def score_composite(market: dict[str, Any]) -> int:
    """Voto giocabilità: probabilità, robustezza ML/MC, value, Kelly e calibrazione storica."""
    cal = load_calibration()
    prob = float(market["probability"])
    group = market.get("group") or "1x2"
    league = market.get("league")
    sp = int(market.get("score_prob") or 1)
    sv = market.get("score_value")
    ev = market.get("ev_cons")
    if ev is None:
        ev = market.get("ev")
    odds = market.get("odds")
    ml_prob = market.get("model_probability")

    mkt = "ou25" if group == "ou" else "1x2"
    _, bin_n = prob_bin_factor(cal, prob, market=mkt)

    min_prob = (
        0.28
        if group == "1x2"
        else 0.42
        if group in {"ou", "btts", "team"}
        else 0.18
        if group == "combo"
        else 0.52
    )
    divergence = _ml_mc_divergence(prob, ml_prob)
    has_ml = ml_prob is not None

    if prob < min_prob:
        if sv is not None and ev is not None and ev > 0:
            raw = min(5, 1 + sp * 0.45 + sv * 0.25)
        else:
            raw = float(sp)
        raw += _ml_mc_adjust(divergence, has_ml=has_ml)
        if bin_n < cal.get("min_bin_samples", 30):
            raw = min(raw, cal.get("low_sample_max_score", 6))
        cap = prob_score_cap(prob, league=league, cal=cal)
        if cap is not None:
            raw = min(raw, cap)
        return _clamp_score(raw)

    if sv is None or ev is None or ev <= 0:
        raw = float(sp)
    else:
        raw = 0.58 * sp + 0.42 * sv

    raw -= _borderline_penalty(prob, group)
    raw += _ml_mc_adjust(divergence, has_ml=has_ml)

    if odds and float(odds) > 1.01:
        stake_p = float(market.get("p_cons") or prob)
        qk = _kelly_fraction(stake_p, float(odds)) * float(cal.get("kelly_fraction", 0.25))
        if qk < 0.005:
            raw = min(raw, 4)
        elif qk < 0.015:
            raw = min(raw, 6)
        elif qk < 0.03:
            raw = min(raw, 7)

    if bin_n < cal.get("min_bin_samples", 30):
        raw = min(raw, cal.get("low_sample_max_score", 6))

    cap = prob_score_cap(prob, league=league, cal=cal)
    if cap is not None:
        # Lega a bassa varianza alza di 1 il tetto; alta varianza lo abbassa.
        raw = min(raw, float(cap))

    return _clamp_score(raw)


def explain_pick(
    play: dict[str, Any],
    *,
    alignment: dict[str, Any] | None,
    market_move: dict[str, Any] | None,
    ml_prob: float | None = None,
) -> tuple[str, str]:
    p = float(play.get("probability") or 0)
    mc_p = p
    if ml_prob is not None:
        div = _ml_mc_divergence(mc_p, ml_prob)
        if div <= 0.05:
            agree = "ML e Monte Carlo allineati"
        elif div <= 0.10:
            agree = "ML e MC simili"
        else:
            agree = "ML e MC divergono"
    else:
        agree = "solo Monte Carlo"

    group = play.get("group") or "1x2"
    min_prob = (
        0.28
        if group == "1x2"
        else 0.42
        if group in {"ou", "btts", "team"}
        else 0.18
        if group == "combo"
        else 0.52
    )
    if p >= min_prob + 0.12:
        band = "sopra soglia"
    elif p >= min_prob:
        band = "nella soglia minima"
    else:
        band = "sotto soglia (outsider)"

    line1 = f"Probabilità stimata: {p:.0%} ({agree}, {band})"

    bits: list[str] = []
    if play.get("odds") and play.get("p_market") is not None and play.get("edge_pp") is not None:
        bits.append(
            f"modello {float(play.get('p_cons') or p):.0%} vs mercato {float(play['p_market']):.0%} "
            f"(edge {play['edge_pp']:+.1%} pp · quota {play['odds']:.2f} vs equa {play.get('fair_odds')})"
        )
        if play.get("ev_cons") is not None:
            sharp = play.get("ev_sharp")
            sharp_bit = f" · EV sharp {sharp:+.0%}" if sharp is not None else ""
            bits.append(f"EV cons. {play['ev_cons']:+.0%}{sharp_bit}")
    elif play.get("odds") and play.get("fair_odds"):
        ev = play.get("ev_cons") if play.get("ev_cons") is not None else play.get("ev") or 0
        bits.append(f"Quota {play['odds']:.2f} vs equa {play['fair_odds']:.2f} (EV {ev:+.0%})")
        if play.get("odds_real") is False:
            bits.append("quota ipotetica")
    bits.append(_pick_vs_market_bit(play, alignment, market_move))

    kelly = 0.0
    if play.get("odds"):
        kelly = _capped_kelly(float(play.get("p_cons") or p), float(play["odds"]))
    if play.get("action") == "no_bet":
        bits.append("No bet: " + "; ".join(play.get("no_bet_reasons") or ["filtro"]))
    elif kelly < 0.01:
        bits.append("Kelly ¼ basso → stake teorico minimo")
    elif kelly >= 0.03:
        bits.append(f"Kelly ¼ {kelly:.1%} (cap {KELLY_CAP:.0%}) → stake moderato")
    else:
        bits.append(f"Kelly ¼ {kelly:.1%} (cap {KELLY_CAP:.0%})")

    tip = play.get("tipster") or {}
    if tip.get("n_sources"):
        cons = tip.get("consensus") or "?"
        agree = tip.get("agree") or tip.get("label")
        names = ", ".join(str(s.get("source")) for s in (tip.get("sources") or []) if s.get("source"))
        bits.append(f"tipster {names or 'n/d'} → {cons} ({agree})")

    line2 = " · ".join(bits)
    return line1, line2


def _pp(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val:+.1f} pp"


def _pick_drop(play: dict[str, Any], market_move: dict[str, Any] | None) -> tuple[float | None, object, object]:
    """Variazione pp / quota apertura→attuale del mercato consigliato."""
    if not market_move:
        return None, None, None
    code = str(play.get("code") or "")
    group = play.get("group") or "1x2"
    if code == "1":
        return market_move.get("drop_1"), market_move.get("open_1"), market_move.get("odd_1")
    if code == "X":
        return market_move.get("drop_x"), market_move.get("open_x"), market_move.get("odd_x")
    if code == "2":
        return market_move.get("drop_2"), market_move.get("open_2"), market_move.get("odd_2")
    if group == "ou" or code.startswith("O"):
        if "U" in code and not code.startswith("O"):
            return market_move.get("drop_under"), market_move.get("open_under"), market_move.get("odd_under")
        if code.startswith("O") and "GOL" not in code:
            return market_move.get("drop_over"), market_move.get("open_over"), market_move.get("odd_over")
    if code.startswith("U"):
        return market_move.get("drop_under"), market_move.get("open_under"), market_move.get("odd_under")
    return None, None, None


def _pick_vs_market_bit(
    play: dict[str, Any],
    alignment: dict[str, Any] | None,
    market_move: dict[str, Any] | None,
) -> str:
    label = (alignment or {}).get("label") or "n/d"
    move_lvl = (market_move or {}).get("movement_level") or "Stabile"
    drop, open_odd, curr_odd = _pick_drop(play, market_move)
    quota_bit = ""
    if drop is not None and open_odd is not None and curr_odd is not None:
        try:
            q_txt = f"{float(open_odd):.2f}->{float(curr_odd):.2f}"
        except (TypeError, ValueError):
            q_txt = ""
        if drop >= 1.5:
            quota_bit = f"quota pick accorciata {q_txt} ({_pp(drop)}): mercato conferma"
        elif drop <= -1.5:
            quota_bit = f"quota pick allungata {q_txt} ({_pp(drop)}): mercato sconta"
        elif abs(drop) >= 0.4:
            verb = "accorciata" if drop > 0 else "allungata"
            quota_bit = f"quota pick {verb} {q_txt} ({_pp(drop)})"

    agrees = (alignment or {}).get("agrees") or []
    disagrees = (alignment or {}).get("disagrees") or []
    if label == "allineato":
        core = f"mercato Asian {move_lvl.lower()} allineato"
        if agrees:
            core += f" su {', '.join(agrees)}"
    elif label == "contrario":
        core = f"mercato Asian {move_lvl.lower()} contrario"
        if disagrees:
            core += f" (flusso su {', '.join(disagrees)})"
    elif label == "misto":
        core = f"mercato {move_lvl.lower()} misto"
        extra = []
        if agrees:
            extra.append("ok " + ", ".join(agrees))
        if disagrees:
            extra.append("no " + ", ".join(disagrees))
        if extra:
            core += f" ({'; '.join(extra)})"
    elif move_lvl != "Stabile":
        core = f"mercato {move_lvl.lower()}, modello neutro"
    else:
        core = "mercato stabile"

    if quota_bit:
        return f"{core} · {quota_bit}"
    return core


def _market(
    code: str,
    name: str,
    group: str,
    prob: float,
    p_second: float,
    p_model: float,
    odds: float | None,
    *,
    baseline: float = 0.5,
    odds_source: str | None = None,
) -> dict[str, Any]:
    ev = None
    ratio = None
    implied = None
    value = None
    source = None
    if odds is not None and odds > 1.0:
        implied = round(1.0 / odds, 4)
        ev = round(prob * odds - 1.0, 4)
        ratio = round(prob * odds, 4)
        value = None
        source = odds_source or "book"
    return {
        "code": code,
        "name": name,
        "group": group,
        "probability": round(float(prob), 4),
        "model_probability": round(float(p_model), 4),
        "fair_odds": _fair_odds(float(prob)),
        "odds": odds,
        "odds_source": source,
        "implied_prob": implied,
        "ev": ev,
        "prob_odds_ratio": ratio,
        "score_prob": score_probability(float(prob), float(p_second), float(p_model), baseline=baseline),
        "score_value": value,
    }


def _with_composite(market: dict[str, Any]) -> dict[str, Any]:
    out = dict(market)
    out["score"] = score_composite(market)
    odd = out.get("odds")
    stake_p = float(out.get("p_cons") or out["probability"])
    if odd and float(odd) > 1.01:
        out["kelly_quarter"] = round(_capped_kelly(stake_p, float(odd)), 4)
    else:
        out["kelly_quarter"] = 0.0
    return out


def _overround_1x2(o1: float | None, ox: float | None, o2: float | None) -> float:
    if o1 and ox and o2 and min(o1, ox, o2) > 1:
        return (1.0 / o1) + (1.0 / ox) + (1.0 / o2)
    return 1.06


def _overround_pair(a: float | None, b: float | None) -> float:
    if a and b and a > 1 and b > 1:
        return (1.0 / a) + (1.0 / b)
    return 1.05


def _apply_margin(fair_p: float, overround: float) -> float | None:
    if fair_p <= 0.03 or fair_p >= 0.97 or overround <= 1:
        return None
    odd = 1.0 / (fair_p * overround)
    if odd <= 1.01 or odd > 50:
        return None
    return round(odd, 2)


def _overround_combo() -> float:
    return 1.12


def _pick_headline(probable: dict, value: dict | None) -> dict[str, Any]:
    probable = _with_composite(probable)
    value = _with_composite(value) if value else None

    def pack(item: dict, kind: str) -> dict[str, Any]:
        p = float(item.get("p_cons") or item["probability"])
        odd = item.get("odds")
        kq = _capped_kelly(p, float(odd)) if odd else 0.0
        packed = {
            "code": item["code"],
            "name": item["name"],
            "group": item.get("group"),
            "kind": kind,
            "score": item["score"],
            "score_prob": item.get("score_prob"),
            "score_value": item.get("score_value"),
            "probability": item["probability"],
            "model_probability": item.get("model_probability"),
            "odds": item["odds"],
            "ev": item.get("ev_cons") if item.get("ev_cons") is not None else item.get("ev"),
            "fair_odds": item["fair_odds"],
            "odds_source": item.get("odds_source"),
            "kelly_quarter": round(kq, 4),
        }
        for key in PLAY_VALUE_KEYS:
            if key in item:
                packed[key] = item[key]
        return packed

    if value is None:
        return pack(probable, "più_probabile")

    cal = load_calibration()
    min_ev_val = float(cal.get("min_ev_strong_value", 0.06))
    min_p_val = float(cal.get("min_prob_1x2_value", 0.35))
    value_ev = value.get("ev_cons") if value.get("ev_cons") is not None else value.get("ev")

    if probable["code"] == value["code"]:
        kind = "probabile_e_valore" if (value_ev or 0) >= 0 and value.get("odds_real") else "più_probabile"
        return pack(probable, kind)

    strong_value = (
        value.get("odds_real")
        and (value_ev or 0) >= min_ev_val
        and value["probability"] >= min_p_val
        and value["score"] >= probable["score"] + 1
        and (value.get("ev_sharp") is None or value["ev_sharp"] >= 0)
    )
    if strong_value:
        return pack(value, "valore")
    return pack(probable, "più_probabile")


def _actionable(m: dict[str, Any]) -> bool:
    p = m["probability"]
    if m.get("group") == "combo":
        return 0.12 <= p <= 0.55
    if m.get("group") in {"dc", "dnb"}:
        return 0.52 <= p <= 0.88
    if m.get("group") in {"ou", "btts", "team", "cards", "corners"}:
        return 0.42 <= p <= 0.80
    if m.get("group") in {"multigol", "parity"}:
        return 0.22 <= p <= 0.72
    if m.get("group") == "exact":
        return 0.06 <= p <= 0.28
    return 0.28 <= p <= 0.85


def advise(
    prediction: dict[str, Any],
    odds: dict[str, float] | None = None,
    market_move: dict | None = None,
    *,
    odds_from_asian: bool = False,
    tipster: dict | None = None,
    league: str | None = None,
) -> dict[str, Any]:
    """Mercati 1X2, DC, DNB, O/U, BTTS, multigol, exact, corner/card proxy, combo."""
    odds = odds or {}
    home, away = _split_match(prediction.get("match", "Casa vs Trasferta"))
    league = league or prediction.get("league")
    mc = prediction["montecarlo"]
    ml = prediction["model_probabilities"]
    cal = load_calibration()
    book_src = "asianbetsoccer" if odds_from_asian else "book"

    o1 = _get_odd(odds, "home", "1")
    ox = _get_odd(odds, "draw", "X", "x")
    o2 = _get_odd(odds, "away", "2")
    o_o25 = _get_odd(odds, "over_2.5")
    o_u25 = _get_odd(odds, "under_2.5")
    rr_1x2 = _overround_1x2(o1, ox, o2)
    rr_ou = _overround_pair(o_o25, o_u25)

    p1, px, p2 = mc["home_win"], mc["draw"], mc["away_win"]
    ranked = sorted([p1, px, p2], reverse=True)
    dc_1x = float(mc.get("dc_1x", p1 + px))
    dc_12 = float(mc.get("dc_12", p1 + p2))
    dc_x2 = float(mc.get("dc_x2", px + p2))
    dnb_1 = float(mc.get("dnb_1", p1 / max(p1 + p2, 1e-9)))
    dnb_2 = float(mc.get("dnb_2", p2 / max(p1 + p2, 1e-9)))

    markets_1x2 = [
        _market("1", f"{home} vince", "1x2", p1, ranked[1], ml["home_win"], o1, baseline=0.333, odds_source=book_src),
        _market("X", "Pareggio", "1x2", px, ranked[1], ml["draw"], ox, baseline=0.333, odds_source=book_src),
        _market("2", f"{away} vince", "1x2", p2, ranked[1], ml["away_win"], o2, baseline=0.333, odds_source=book_src),
    ]

    def derived(code, name, group, prob, complement, model_p, baseline, book_odd=None, source="stimata"):
        odd = book_odd if book_odd and book_odd > 1 else None
        if odd is None:
            margin = rr_1x2 if group in {"dc", "dnb", "1x2", "ah"} else rr_ou
            odd = _apply_margin(prob, margin)
            src = source
        else:
            src = book_src
        return _market(code, name, group, prob, complement, model_p, odd, baseline=baseline, odds_source=src)

    markets_dc = [
        derived("1X", "1X (casa o pari)", "dc", dc_1x, p2, p1 + px, 0.67, _get_odd(odds, "1X", "dc_1x")),
        derived("12", "12 (no pari)", "dc", dc_12, px, p1 + p2, 0.67, _get_odd(odds, "12", "dc_12")),
        derived("X2", "X2 (trasferta o pari)", "dc", dc_x2, p1, px + p2, 0.67, _get_odd(odds, "X2", "dc_x2")),
        derived("1 DNB", f"{home} DNB", "dnb", dnb_1, dnb_2, p1 / max(p1 + p2, 1e-9), 0.5, _get_odd(odds, "dnb_1")),
        derived("2 DNB", f"{away} DNB", "dnb", dnb_2, dnb_1, p2 / max(p1 + p2, 1e-9), 0.5, _get_odd(odds, "dnb_2")),
    ]

    ou_book = {
        "over_2.5": o_o25,
        "under_2.5": o_u25,
        "over_0.5": _get_odd(odds, "over_0.5"),
        "under_0.5": _get_odd(odds, "under_0.5"),
        "over_1.5": _get_odd(odds, "over_1.5"),
        "under_1.5": _get_odd(odds, "under_1.5"),
        "over_3.5": _get_odd(odds, "over_3.5"),
        "under_3.5": _get_odd(odds, "under_3.5"),
        "over_4.5": _get_odd(odds, "over_4.5"),
        "under_4.5": _get_odd(odds, "under_4.5"),
    }
    markets_ou = []
    ml_mkt = prediction.get("market_ml") or {}
    p_ml_o25 = ml_mkt.get("p_over_25")
    if p_ml_o25 is None:
        p_ml_o25 = prediction.get("p_over_25")
    p_ml_ah = ml_mkt.get("p_ah0_home")
    if p_ml_ah is None:
        p_ml_ah = prediction.get("p_ah0_home")
    ml_w = 0.55
    for line in (0.5, 1.5, 2.5, 3.5, 4.5):
        ok, uk = f"over_{line}", f"under_{line}"
        if ok not in mc:
            continue
        po, pu = float(mc.get(ok, 0)), float(mc.get(uk, 1 - mc.get(ok, 0)))
        model_o, model_u = po, pu
        if line == 2.5 and p_ml_o25 is not None:
            po = ml_w * float(p_ml_o25) + (1.0 - ml_w) * po
            pu = 1.0 - po
            model_o, model_u = float(p_ml_o25), 1.0 - float(p_ml_o25)
        src = book_src if ou_book.get(ok) else "stimata da O/U 2.5"
        markets_ou.append(derived(f"O{line}", f"Over {line}", "ou", po, pu, model_o, 0.5, ou_book.get(ok), src))
        src_u = book_src if ou_book.get(uk) else "stimata da O/U 2.5"
        markets_ou.append(derived(f"U{line}", f"Under {line}", "ou", pu, po, model_u, 0.5, ou_book.get(uk), src_u))

    # Asian Handicap 0 (casa copre = vittoria casa; push = pareggio non nel binary)
    markets_ah = []
    p_ah_h = float(mc.get("ah_home_0", p1))
    if p_ml_ah is not None:
        p_ah_h = ml_w * float(p_ml_ah) + (1.0 - ml_w) * p_ah_h
        model_ah = float(p_ml_ah)
    else:
        model_ah = p_ah_h
    p_ah_a = max(0.02, min(0.98, 1.0 - p_ah_h))
    markets_ah.extend(
        [
            derived(
                "AH0 1",
                f"{home} AH 0",
                "ah",
                p_ah_h,
                p_ah_a,
                model_ah,
                0.45,
                _get_odd(odds, "ah_home_0", "ah0_1", "ah_0_1"),
                "xgb+mc" if p_ml_ah is not None else "stimata AH",
            ),
            derived(
                "AH0 2",
                f"{away} AH 0",
                "ah",
                p_ah_a,
                p_ah_h,
                1.0 - model_ah,
                0.45,
                _get_odd(odds, "ah_away_0", "ah0_2", "ah_0_2"),
                "xgb+mc" if p_ml_ah is not None else "stimata AH",
            ),
        ]
    )
    btts = float(mc.get("btts", 0))
    markets_btts = [
        derived("GOL", "Gol (BTTS sì)", "btts", btts, 1 - btts, btts, 0.5, _get_odd(odds, "btts_yes", "gol")),
        derived("NOGOL", "No gol (BTTS no)", "btts", 1 - btts, btts, 1 - btts, 0.5, _get_odd(odds, "btts_no", "nogo")),
    ]

    markets_team = []
    for side, label, prefix in (("home", home, "home"), ("away", away, "away")):
        for line in (0.5, 1.5, 2.5):
            ok, uk = f"{prefix}_over_{line}", f"{prefix}_under_{line}"
            if ok not in mc:
                continue
            po, pu = float(mc.get(ok, 0)), float(mc.get(uk, 1 - mc.get(ok, 0)))
            markets_team.append(
                derived(f"{'C' if side == 'home' else 'T'}O{line}", f"{label} over {line}", "team", po, pu, po, 0.5, _get_odd(odds, ok))
            )
            markets_team.append(
                derived(f"{'C' if side == 'home' else 'T'}U{line}", f"{label} under {line}", "team", pu, po, pu, 0.5, _get_odd(odds, uk))
            )
    markets_team.extend(
        [
            derived("1 0-0", f"{home} vince a 0", "team", float(mc.get("home_win_to_nil", 0)), 1 - float(mc.get("home_win_to_nil", 0)), float(mc.get("home_win_to_nil", 0)), 0.25, _get_odd(odds, "home_win_to_nil")),
            derived("2 0-0", f"{away} vince a 0", "team", float(mc.get("away_win_to_nil", 0)), 1 - float(mc.get("away_win_to_nil", 0)), float(mc.get("away_win_to_nil", 0)), 0.25, _get_odd(odds, "away_win_to_nil")),
        ]
    )

    rr_combo = _overround_combo()

    def combo(code: str, name: str, mc_key: str, complement: float) -> dict[str, Any]:
        prob = float(mc.get(mc_key, 0))
        book = _get_odd(odds, mc_key, code.replace(" ", "_").lower())
        odd = book if book and book > 1 else _apply_margin(prob, rr_combo)
        src = book_src if book and book > 1 else "stimata combo"
        return _market(code, name, "combo", prob, complement, prob, odd, baseline=0.15, odds_source=src)

    markets_combo = [
        combo("1+O2.5", f"{home} vince e Over 2.5", "combo_1_o25", float(mc.get("combo_1_u25", 0))),
        combo("1+U2.5", f"{home} vince e Under 2.5", "combo_1_u25", float(mc.get("combo_1_o25", 0))),
        combo("X+O2.5", "Pareggio e Over 2.5", "combo_x_o25", float(mc.get("combo_x_u25", 0))),
        combo("X+U2.5", "Pareggio e Under 2.5", "combo_x_u25", float(mc.get("combo_x_o25", 0))),
        combo("2+O2.5", f"{away} vince e Over 2.5", "combo_2_o25", float(mc.get("combo_2_u25", 0))),
        combo("2+U2.5", f"{away} vince e Under 2.5", "combo_2_u25", float(mc.get("combo_2_o25", 0))),
        combo("1+O1.5", f"{home} vince e Over 1.5", "combo_1_o15", float(mc.get("combo_1_u15", 0))),
        combo("1+U1.5", f"{home} vince e Under 1.5", "combo_1_u15", float(mc.get("combo_1_o15", 0))),
        combo("2+O1.5", f"{away} vince e Over 1.5", "combo_2_o15", float(mc.get("combo_2_u15", 0))),
        combo("2+U1.5", f"{away} vince e Under 1.5", "combo_2_u15", float(mc.get("combo_2_o15", 0))),
        combo("1+GOL", f"{home} vince e Gol", "combo_1_gol", float(mc.get("combo_1_nogol", 0))),
        combo("1+NOGOL", f"{home} vince e No gol", "combo_1_nogol", float(mc.get("combo_1_gol", 0))),
        combo("2+GOL", f"{away} vince e Gol", "combo_2_gol", float(mc.get("combo_2_nogol", 0))),
        combo("2+NOGOL", f"{away} vince e No gol", "combo_2_nogol", float(mc.get("combo_2_gol", 0))),
        combo("X+GOL", "Pareggio e Gol", "combo_x_gol", float(mc.get("combo_x_nogol", 0))),
        combo("X+NOGOL", "Pareggio e No gol", "combo_x_nogol", float(mc.get("combo_x_gol", 0))),
        combo("1X+O2.5", "1X e Over 2.5", "combo_1x_o25", float(mc.get("combo_1x_u25", 0))),
        combo("1X+U2.5", "1X e Under 2.5", "combo_1x_u25", float(mc.get("combo_1x_o25", 0))),
        combo("X2+O2.5", "X2 e Over 2.5", "combo_x2_o25", float(mc.get("combo_x2_u25", 0))),
        combo("X2+U2.5", "X2 e Under 2.5", "combo_x2_u25", float(mc.get("combo_x2_o25", 0))),
        combo("12+O2.5", "12 e Over 2.5", "combo_12_o25", float(mc.get("combo_12_u25", 0))),
        combo("12+U2.5", "12 e Under 2.5", "combo_12_u25", float(mc.get("combo_12_o25", 0))),
        combo("GOL+O2.5", "Gol e Over 2.5", "combo_gol_o25", float(mc.get("combo_nogol_u25", 0))),
        combo("NOGOL+U2.5", "No gol e Under 2.5", "combo_nogol_u25", float(mc.get("combo_gol_o25", 0))),
    ]

    # Multigol (fasce gol totali)
    mg_specs = [
        ("MG0-1", "Multigol 0-1", "mg_0_1"),
        ("MG1-2", "Multigol 1-2", "mg_1_2"),
        ("MG2-3", "Multigol 2-3", "mg_2_3"),
        ("MG3-4", "Multigol 3-4", "mg_3_4"),
        ("MG1-3", "Multigol 1-3", "mg_1_3"),
        ("MG2-4", "Multigol 2-4", "mg_2_4"),
        ("MG0-2", "Multigol 0-2", "mg_0_2"),
        ("MG3+", "Multigol 3+", "mg_3_plus"),
        ("MG4+", "Multigol 4+", "mg_4_plus"),
    ]
    markets_multigol = []
    for code, name, key in mg_specs:
        if key not in mc:
            continue
        p = float(mc.get(key, 0))
        markets_multigol.append(
            derived(code, name, "multigol", p, 1 - p, p, 0.25, _get_odd(odds, key, code.lower()), "stimata multigol")
        )

    markets_parity = [
        derived(
            "DISPARI",
            "Gol totali dispari",
            "parity",
            float(mc.get("goals_odd", 0.5)),
            float(mc.get("goals_even", 0.5)),
            float(mc.get("goals_odd", 0.5)),
            0.5,
            _get_odd(odds, "goals_odd", "odd"),
            "stimata parity",
        ),
        derived(
            "PARI",
            "Gol totali pari",
            "parity",
            float(mc.get("goals_even", 0.5)),
            float(mc.get("goals_odd", 0.5)),
            float(mc.get("goals_even", 0.5)),
            0.5,
            _get_odd(odds, "goals_even", "even"),
            "stimata parity",
        ),
    ]

    markets_exact = []
    for row in (mc.get("most_likely_scores") or [])[:6]:
        score = str(row.get("score") or "")
        p = float(row.get("prob") or 0)
        if not score or p < 0.04:
            continue
        markets_exact.append(
            derived(
                score,
                f"Risultato esatto {score}",
                "exact",
                p,
                1 - p,
                p,
                0.08,
                _get_odd(odds, f"cs_{score}", score.replace("-", "_")),
                "stimata exact",
            )
        )

    markets_cards = []
    card_src = str(mc.get("cards_source") or "proxy")
    for line in (2.5, 3.5, 4.5, 5.5):
        ok, uk = f"cards_over_{line}", f"cards_under_{line}"
        if ok not in mc:
            continue
        po, pu = float(mc[ok]), float(mc.get(uk, 1 - mc[ok]))
        markets_cards.append(
            derived(f"CARDO{line}", f"Cartellini Over {line}", "cards", po, pu, po, 0.5, _get_odd(odds, ok), f"λ {card_src}")
        )
        markets_cards.append(
            derived(f"CARDU{line}", f"Cartellini Under {line}", "cards", pu, po, pu, 0.5, _get_odd(odds, uk), f"λ {card_src}")
        )

    markets_corners = []
    corner_src = str(mc.get("corners_source") or "proxy")
    for line in (8.5, 9.5, 10.5, 11.5):
        ok, uk = f"corners_over_{line}", f"corners_under_{line}"
        if ok not in mc:
            continue
        po, pu = float(mc[ok]), float(mc.get(uk, 1 - mc[ok]))
        markets_corners.append(
            derived(f"CORNO{line}", f"Corner Over {line}", "corners", po, pu, po, 0.5, _get_odd(odds, ok), f"λ {corner_src}")
        )
        markets_corners.append(
            derived(f"CORNU{line}", f"Corner Under {line}", "corners", pu, po, pu, 0.5, _get_odd(odds, uk), f"λ {corner_src}")
        )

    markets_scorer: list[dict[str, Any]] = []
    try:
        from modules.advisor.scorers import anytime_probs

        fm_det = (
            ((prediction.get("fotmob_context") or {}).get("details"))
            or prediction.get("fotmob_details")
            or {}
        )
        for row in anytime_probs(
            home,
            away,
            lambda_home=float((prediction.get("expected_goals") or {}).get("home") or mc.get("lambda_home") or 1.2),
            lambda_away=float((prediction.get("expected_goals") or {}).get("away") or mc.get("lambda_away") or 1.0),
            top_n=4,
            lineup_home=fm_det.get("lineup_home") if isinstance(fm_det, dict) else None,
            lineup_away=fm_det.get("lineup_away") if isinstance(fm_det, dict) else None,
        ):
            p_any = float(row["p_anytime"])
            p_first = float(row["p_first"])
            src = str(row.get("source") or "xG share")
            if row.get("in_lineup") is True:
                src = f"{src}+XI"
            elif row.get("in_lineup") is False:
                src = f"{src}+bench"
            markets_scorer.append(
                derived(
                    f"AS {row['player'][:18]}",
                    f"{row['player']} anytime",
                    "scorer",
                    p_any,
                    1 - p_any,
                    p_any,
                    0.25,
                    _get_odd(odds, f"anytime_{row['player']}", "anytime"),
                    src,
                )
            )
            if p_first >= 0.06:
                markets_scorer.append(
                    derived(
                        f"FS {row['player'][:18]}",
                        f"{row['player']} first scorer",
                        "scorer",
                        p_first,
                        1 - p_first,
                        p_first,
                        0.12,
                        _get_odd(odds, f"first_{row['player']}", "first"),
                        src,
                    )
                )
    except Exception:
        markets_scorer = []

    def _finish(m: dict[str, Any], overround: float) -> dict[str, Any]:
        g = m.get("group")
        rr = (
            rr_1x2
            if g in {"1x2", "dc", "dnb", "ah"}
            else rr_ou
            if g in {"ou", "btts", "team", "cards", "corners", "multigol", "parity", "scorer"}
            else rr_combo
        )
        return _with_composite(
            enrich_value(
                m,
                odds=odds,
                overround=overround or rr,
                league=league,
                market_move=market_move,
                odds_from_asian=odds_from_asian,
                cal=cal,
            )
        )

    grouped = {
        "1x2": [_finish(m, rr_1x2) for m in markets_1x2],
        "dc": [_finish(m, rr_1x2) for m in markets_dc],
        "ah": [_finish(m, rr_1x2) for m in markets_ah],
        "ou": [_finish(m, rr_ou) for m in markets_ou],
        "btts": [_finish(m, rr_ou) for m in markets_btts],
        "multigol": [_finish(m, rr_ou) for m in markets_multigol],
        "parity": [_finish(m, rr_ou) for m in markets_parity],
        "exact": [_finish(m, rr_combo) for m in markets_exact],
        "team": [_finish(m, rr_ou) for m in markets_team],
        "cards": [_finish(m, rr_ou) for m in markets_cards],
        "corners": [_finish(m, rr_ou) for m in markets_corners],
        "scorer": [_finish(m, rr_ou) for m in markets_scorer],
        "combo": [_finish(m, rr_combo) for m in markets_combo],
    }
    all_markets = [m for g in grouped.values() for m in g]

    probable_1x2 = max(grouped["1x2"], key=lambda m: m["probability"])
    with_odds_1x2 = [m for m in grouped["1x2"] if m.get("odds_real") and m.get("edge_pp") is not None]
    best_value_1x2 = (
        max(with_odds_1x2, key=lambda m: m.get("edge_pp") if m.get("edge_pp") is not None else -99)
        if with_odds_1x2
        else None
    )
    extras = [m for m in all_markets if m["group"] != "1x2" and _actionable(m) and m.get("odds_real")]
    play_alt = None
    invalid_no_odds = not with_odds_1x2

    if invalid_no_odds:
        play_1x2 = _analysis_play(
            action="invalido",
            kind="invalido",
            name="pick invalido (quote assenti)",
            reason="pick invalido: senza quote non si calcolano edge, EV, Kelly, quota equa, CLV",
        )
        play = play_1x2
    else:
        play_1x2 = _pick_headline(probable_1x2, best_value_1x2)
        if extras:
            def extra_key(m):
                edge = m.get("edge_pp")
                evn = m.get("ev_cons") if m.get("ev_cons") is not None else m.get("ev")
                return m["score"] + max(edge if edge is not None else evn or -0.05, -0.05) * 3

            play_alt = max(extras, key=extra_key)
            play_alt = _pick_headline(play_alt, play_alt if play_alt.get("odds_real") else None)

        play = play_1x2
        min_ev_play = float(cal.get("min_ev_play", MIN_EDGE))
        alt_ev = None if not play_alt else (play_alt.get("ev_cons") if play_alt.get("ev_cons") is not None else play_alt.get("ev"))
        if play_alt and (alt_ev or 0) >= min_ev_play and play_alt["score"] >= play_1x2["score"]:
            play = play_alt

    ml_for_play = play.get("model_probability")
    if ml_for_play is None and play.get("group") == "1x2":
        code = play.get("code")
        if code == "1":
            ml_for_play = ml["home_win"]
        elif code == "X":
            ml_for_play = ml["draw"]
        elif code == "2":
            ml_for_play = ml["away_win"]

    alignment = None
    score_cap: int | None = None
    if market_move and not invalid_no_odds:
        from modules.data_update.asian_odds import MOVE_RANK, move_alignment

        alignment = move_alignment(play.get("code"), market_move)
        delta = float(alignment.get("delta") or 0)
        # Asian già nella gamba value: smorza solo mezzo punto di bonus allineamento
        if delta > 0 and odds_from_asian:
            delta = max(0.0, delta - 0.5)
        lvl = market_move.get("movement_level") or "Stabile"
        if alignment.get("label") == "contrario" and MOVE_RANK.get(lvl, 0) >= MOVE_RANK["Forte"]:
            score_cap = 5 if lvl == "Raro" else 6
            delta = min(delta, -1.0)
        if delta:
            play = dict(play)
            play["score"] = int(max(1, min(10, round(float(play.get("score") or 1) + delta))))
        if score_cap is not None and play.get("score") is not None:
            play["score"] = min(int(play["score"]), score_cap)
        play["market_align"] = alignment["label"]
        _, open_odd, curr_odd = _pick_drop(play, market_move)
        from modules.advisor.staking import clv_prob as _clv

        play["clv"] = _clv(
            float(open_odd) if open_odd else None,
            float(curr_odd) if curr_odd else None,
        )

    if tipster is None:
        try:
            from modules.tipsters import consensus_for

            tipster = consensus_for(home, away)
        except Exception:
            tipster = {"n_sources": 0, "label": "n/d"}
    from modules.tipsters import apply_tipster_balance

    play = apply_tipster_balance(play, tipster)

    play = dict(play)
    if invalid_no_odds or not play.get("odds_real"):
        play["action"] = "invalido"
        play["kind"] = "invalido"
        play["code"] = "—"
        play["name"] = "pick invalido (quote assenti)"
        play["kelly_quarter"] = None
        play["ev"] = None
        play["ev_cons"] = None
        play["edge_pp"] = None
        play["fair_odds"] = None
        play["clv"] = None
        play["no_bet_reasons"] = [
            "pick invalido: senza quote non si calcolano edge, EV, Kelly, quota equa, CLV"
        ]
        if play.get("score") is not None:
            play["score"] = min(int(play["score"]), NO_MODEL_MAX_SCORE)
    else:
        reasons = no_bet_reasons(
            play,
            market_move=market_move,
            alignment=alignment,
            min_edge=float(cal.get("min_ev_play", MIN_EDGE)),
            min_rank=int(cal.get("liquid_against_rank", 3)),
            min_pp=float(cal.get("liquid_against_pp", 2.0)),
            sharp_ev=play.get("ev_sharp"),
        )
        if reasons:
            play["action"] = "no_bet"
            play["no_bet_reasons"] = reasons
            play["kelly_quarter"] = 0.0
            play["score"] = min(int(play.get("score") or 1), 5)
        else:
            play["action"] = "gioca"
            play["no_bet_reasons"] = []
        if _value_metrics_missing(play) and play.get("score") is not None:
            play["score"] = min(int(play["score"]), NO_MODEL_MAX_SCORE)

    if play.get("action") in {"invalido", "n/d"}:
        reason1 = play["no_bet_reasons"][0] if play.get("no_bet_reasons") else "nessun pick"
        reason2 = ""
    else:
        reason1, reason2 = explain_pick(play, alignment=alignment, market_move=market_move, ml_prob=ml_for_play)
    pred = dict(prediction)
    pred.setdefault("home", home)
    pred.setdefault("away", away)
    if not (pred.get("sportly_sim") or {}).get("ready"):
        try:
            from modules.sportly_sim import build_sportly_sim

            pred["sportly_sim"] = build_sportly_sim(pred)
        except Exception:
            pred["sportly_sim"] = {"ready": False}
    if not (pred.get("data_signal") or {}).get("ready"):
        try:
            from modules.advisor.data_signal import build_data_signal

            pred["data_signal"] = build_data_signal(pred)
        except Exception:
            pred["data_signal"] = {"ready": False}
    validation = run_validation(prediction=pred, play=play, grouped=grouped)
    play = apply_to_play(play, validation)
    quadro = build_quadro(
        home=home,
        away=away,
        play=play,
        prediction=pred,
        grouped=grouped,
        alignment=alignment,
        market_move=market_move,
        tipster=play.get("tipster") or tipster,
        validation=validation,
    )

    # Pesi fonti + fallback leghe minori prima dell'accordo pesato
    from modules.advisor.pro_scores import annotate_source_weights, build_fallback_source

    _ou = str(play.get("group") or "").lower() in {"ou", "btts", "goal"}
    quadro = dict(quadro)
    quadro["sources"] = annotate_source_weights(list(quadro.get("sources") or []), ou=_ou)
    _fb = build_fallback_source(pred, quadro["sources"])
    if _fb:
        quadro["sources"] = list(quadro["sources"]) + [_fb]
        quadro["fallback"] = True

    # Accordo fonti + intervalli MC + residual EV (dopo quadro)
    from modules.advisor.agreement import source_agreement
    from modules.advisor.residual_ev import predict_residual

    agree = source_agreement(
        quadro,
        play_code=play.get("code"),
        play_group=play.get("group"),
        league=league or prediction.get("league"),
        interval_width=(
            ((pred.get("conformal_intervals") or {}).get("top_width"))
            or ((pred.get("montecarlo") or {}).get("prob_intervals") or {}).get("top_width")
        ),
    )
    play["source_agreement"] = agree
    # preferisci conformal se pronto, altrimenti bootstrap MC
    intervals = pred.get("conformal_intervals") or {}
    if not intervals.get("ready"):
        intervals = (pred.get("montecarlo") or {}).get("prob_intervals") or {}
    play["prob_intervals"] = intervals
    play["conformal_intervals"] = pred.get("conformal_intervals") or {}
    move_rank = None
    try:
        from modules.data_update.asian_odds import MOVE_RANK

        move_rank = MOVE_RANK.get((market_move or {}).get("movement_level") or "Stabile", 0)
    except Exception:
        move_rank = 0
    # soft-cap Kelly se residual in produzione negativo
    residual = predict_residual(
        play,
        agree_share=agree.get("agree_share"),
        data_edge=(pred.get("data_signal") or {}).get("edge"),
        move_rank=move_rank,
        league=league or prediction.get("league"),
    )
    play["residual_ev"] = residual
    # allega conformal mercati dal MC se presenti
    mc = pred.get("montecarlo") or {}
    if mc.get("conformal_ou25"):
        play["conformal_ou25"] = mc["conformal_ou25"]
    if mc.get("conformal_ah0"):
        play["conformal_ah0"] = mc["conformal_ah0"]
    if residual.get("production") and residual.get("residual") is not None and play.get("kelly_quarter"):
        try:
            factor = max(0.45, min(1.20, 1.0 + 2.5 * float(residual["residual"])))
            play["kelly_quarter"] = round(float(play["kelly_quarter"]) * factor, 5)
        except (TypeError, ValueError):
            pass

    validation = dict(validation or {})
    validation["agreement"] = agree
    validation["prob_intervals"] = intervals
    validation["residual_ev"] = residual
    extra_delta = float(agree.get("delta_unified") or 0)
    if intervals.get("ready"):
        if intervals.get("fragile"):
            extra_delta -= 0.25
        elif intervals.get("stable"):
            extra_delta += 0.25
    if residual.get("ready"):
        extra_delta += float(residual.get("delta_unified") or 0)
    validation["delta_unified"] = round(float(validation.get("delta_unified") or 0) + extra_delta, 3)
    bits = list(str(validation.get("summary") or "").split(" · ")) if validation.get("summary") else []
    bits.append(f"accordo {agree.get('status')}")
    if intervals.get("ready"):
        bits.append("IC " + ("stabile" if intervals.get("stable") else "fragile" if intervals.get("fragile") else "ok"))
    validation["summary"] = " · ".join(b for b in bits if b)

    if play.get("action") not in {"invalido", "n/d"} and play.get("odds_real"):
        reasons = no_bet_reasons(
            play,
            market_move=market_move,
            alignment=alignment,
            min_edge=float(cal.get("min_ev_play", MIN_EDGE)),
            min_rank=int(cal.get("liquid_against_rank", 3)),
            min_pp=float(cal.get("liquid_against_pp", 2.0)),
            sharp_ev=play.get("ev_sharp"),
            agreement=agree,
            prob_intervals=intervals,
            residual=residual,
        )
        if reasons:
            play["action"] = "no_bet"
            play["no_bet_reasons"] = reasons
            play["kelly_quarter"] = 0.0
            play["score"] = min(int(play.get("score") or 1), 5)
        else:
            play["action"] = "gioca"
            play["no_bet_reasons"] = []
            kq = play.get("kelly_quarter")
            if kq and intervals.get("ready") and intervals.get("fragile"):
                try:
                    play["kelly_quarter"] = round(float(kq) * 0.70, 5)
                except (TypeError, ValueError):
                    pass

    meta = _meta_analysis(
        play,
        alignment=alignment,
        market_move=market_move,
        quadro=quadro,
        validation=validation,
        history_weight=(pred.get("history_context") or {}).get("weight"),
    )
    # ri-applica delta accordo/IC sul voto unificato 1–10
    if extra_delta:
        meta = dict(meta)
        meta["score"] = int(max(1, min(10, round(meta["score"] + extra_delta))))

    from modules.advisor.pro_scores import build_match_scores

    pro = build_match_scores(
        play=play,
        prediction=pred,
        quadro=quadro,
        agreement=agree,
        validation=validation,
        intervals=intervals,
        residual=residual,
        meta=meta,
        grouped=grouped,
        league=league or prediction.get("league"),
        market_move=market_move,
    )
    quadro = pro.get("quadro") or quadro
    ov = pro.get("overrides") or {}
    if ov.get("delta_unified"):
        meta = dict(meta)
        meta["score"] = int(max(1, min(10, round(meta["score"] + float(ov["delta_unified"])))))
        validation = dict(validation)
        validation["delta_unified"] = round(
            float(validation.get("delta_unified") or 0) + float(ov["delta_unified"]), 3
        )
        bits = list(str(validation.get("summary") or "").split(" · ")) if validation.get("summary") else []
        bits.extend(ov.get("notes") or [])
        validation["summary"] = " · ".join(b for b in bits if b)
        validation["overrides"] = ov

    play["score_unified"] = meta["score"]
    play["meta_analysis"] = meta
    play["validation"] = validation
    play["score_100"] = pro.get("score_100")
    play["confidence_100"] = pro.get("confidence_100")
    play["risk_100"] = pro.get("risk_100")
    play["priority_100"] = pro.get("priority_100")
    play["score_band"] = (pro.get("band") or {}).get("label")
    play["match_scores"] = {
        "unified": pro.get("unified"),
        "confidence": pro.get("confidence"),
        "risk": pro.get("risk"),
        "priority": pro.get("priority"),
        "overrides": ov,
        "coverage": pro.get("coverage"),
        "bet_rec": pro.get("bet_rec"),
        "weights_table": pro.get("weights_table"),
    }
    play["bet_rec"] = pro.get("bet_rec")
    from modules.advisor.play_rank import attach_play_rank

    attach_play_rank(play)

    if play.get("action") not in {"invalido", "n/d"}:
        reason2 = (reason2 + " · " if reason2 else "") + f"Voto unificato {meta['score']}/10 ({meta['note']})"
        if agree.get("ready"):
            reason2 += f" · accordo {agree.get('agree_share')}"
        if pro.get("score_100") is not None:
            reason2 += f" · Score {pro['score_100']:.0f}/100 ({(pro.get('band') or {}).get('label')})"
        if pro.get("priority_100") is not None:
            reason2 += f" · Priorità {pro['priority_100']:.0f}"
        br = pro.get("bet_rec") or {}
        if br.get("ready") and (br.get("primary") or {}).get("code"):
            prim = br["primary"]
            if prim.get("code") != play.get("code"):
                reason2 += f" · Rec mercato {prim.get('label')} {prim.get('code')}"
    else:
        reason1 = play["no_bet_reasons"][0] if play.get("no_bet_reasons") else reason1

    return {
        "match": prediction.get("match"),
        "home": home,
        "away": away,
        "play": play,
        "play_1x2": play_1x2,
        "play_alt": play_alt,
        "most_probable": probable_1x2,
        "best_value": best_value_1x2,
        "markets": markets_1x2,
        "grouped": grouped,
        "all_markets": all_markets,
        "extras": extras,
        "expected_goals": prediction.get("expected_goals"),
        "most_likely_scores": mc.get("most_likely_scores", []),
        "has_odds": any(bool(m.get("odds_real")) for m in grouped["1x2"]),
        "market_move": market_move,
        "market_align": alignment or {"agrees": [], "disagrees": [], "delta": 0, "label": "n/d"},
        "tipster": play.get("tipster") or tipster,
        "score_reason_1": reason1,
        "score_reason_2": reason2,
        "meta_analysis": meta,
        "quadro": quadro,
        "sportly_sim": pred.get("sportly_sim"),
        "data_signal": pred.get("data_signal"),
        "source_agreement": agree,
        "prob_intervals": intervals,
        "residual_ev": residual,
        "validation": validation,
        "match_scores": play.get("match_scores"),
        "score_100": play.get("score_100"),
        "confidence_100": play.get("confidence_100"),
        "risk_100": play.get("risk_100"),
        "priority_100": play.get("priority_100"),
        "score_band": play.get("score_band"),
        "bet_rec": play.get("bet_rec"),
    }


def format_advice(advice: dict[str, Any]) -> str:
    play = advice["play"]
    kind_it = {
        "più_probabile": "più probabile",
        "valore": "miglior rapporto probabilità/quota",
        "probabile_e_valore": "più probabile e miglior value",
        "invalido": "pick invalido",
        "nessun_pick": "nessun pick",
    }.get(play["kind"], play["kind"])
    action = play.get("action")
    head = "GIOCA" if action == "gioca" else "NO BET" if action == "no_bet" else "INVALIDO" if action == "invalido" else "N/D"
    score_txt = "—" if play.get("score") is None else f"{play['score']}/10"
    mix = play.get("score_unified", play.get("score"))
    mix_txt = "—" if mix is None else f"{mix}/10"
    prob = play.get("probability")
    lines = [
        "",
        "=" * 46,
        f"  {advice['match']}",
        f"  {head}  {play['code']}   {play['name']}",
        f"  Voto   {score_txt}   ({kind_it})",
        f"  Mix    {mix_txt}   (value+kelly+workflow)",
    ]
    if prob is not None:
        lines.append(f"  Prob   {float(prob):.1%}")
    if play.get("action") == "no_bet":
        lines.append("  Azione No bet  (" + "; ".join(play.get("no_bet_reasons") or []) + ")")
    elif play.get("action") in {"invalido", "n/d"}:
        lines.append(
            "  Azione "
            + ("Invalido" if play["action"] == "invalido" else "N/D")
            + "  ("
            + "; ".join(play.get("no_bet_reasons") or [])
            + ")"
        )
    if play.get("odds"):
        ev = play.get("ev_cons") if play.get("ev_cons") is not None else play.get("ev") or 0
        sign = "+" if ev >= 0 else ""
        fair = play.get("fair_odds")
        if fair:
            lines.append(f"  Quota  {play['odds']:.2f}   equa {fair:.2f}   EV cons. {sign}{ev:.1%}")
        else:
            lines.append(f"  Quota  {float(play['odds']):.2f}")
        if play.get("edge_pp") is not None:
            lines.append(f"  Edge   {play['edge_pp']:+.1%} pp vs mercato devig")
    move = advice.get("market_move") or {}
    if move.get("movement_comment"):
        lines.append(f"  Quote  {move['movement_comment']}")
    elif advice.get("score_reason_2"):
        lines.append(f"  Nota   {advice['score_reason_2']}")
    if advice.get("quadro"):
        lines.append(f"  Quadro {advice['quadro'].get('summary')}")
    alt = advice.get("play_alt")
    if alt and alt["code"] != play["code"]:
        lines.append(f"  Alt    {alt['code']}  {alt['name']}  {alt['score']}/10")
    lines.append("=" * 46)
    return "\n".join(lines)


def _split_match(match: str) -> tuple[str, str]:
    if " vs " in match:
        home, away = match.split(" vs ", 1)
        return home.strip(), away.strip()
    return "Casa", "Trasferta"


def _get_odd(odds: dict[str, float], *keys: str) -> float | None:
    for key in keys:
        if key in odds and odds[key] is not None:
            try:
                val = float(odds[key])
            except (TypeError, ValueError):
                continue
            if val > 1.0:
                return val
    return None
