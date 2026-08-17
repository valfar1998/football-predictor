"""Consiglio su 1X2 e mercati principali (O/U, gol, DC, DNB) con voto 1-10."""

from __future__ import annotations

from math import floor
from typing import Any

from modules.calibration.config import load_calibration, prob_bin_factor


def _clamp_score(value: float) -> int:
    return int(max(1, min(10, floor(value + 0.5))))


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
    ev = prob * odds - 1.0
    if ev >= 0:
        raw = 4 + 6 * min(1.0, ev / 0.20)
    else:
        raw = 4 - 3 * min(1.0, -ev / 0.25)
    if prob < 0.20:
        raw = min(raw, 4)
    elif prob < 0.25:
        raw = min(raw, 5)
    elif prob < 0.30:
        raw = min(raw, 6)
    elif prob < 0.35:
        raw = min(raw, 7)
    return _clamp_score(raw)


def _kelly_fraction(prob: float, odds: float) -> float:
    if odds <= 1.01 or prob <= 0:
        return 0.0
    edge = prob * odds - 1.0
    if edge <= 0:
        return 0.0
    return edge / (odds - 1.0)


def _ml_mc_divergence(mc_prob: float, ml_prob: float | None) -> float:
    if ml_prob is None:
        return 0.0
    return abs(float(mc_prob) - float(ml_prob))


def score_composite(market: dict[str, Any]) -> int:
    """Voto giocabilità: probabilità, robustezza ML/MC, value, Kelly e calibrazione storica."""
    cal = load_calibration()
    prob = float(market["probability"])
    group = market.get("group") or "1x2"
    sp = int(market.get("score_prob") or 1)
    sv = market.get("score_value")
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

    if prob < min_prob:
        if sv is not None and ev is not None and ev > 0:
            raw = min(5, 1 + sp * 0.45 + sv * 0.25)
        else:
            raw = float(sp)
        if divergence > 0.10:
            raw -= 1.0
        elif divergence > 0.06:
            raw -= 0.5
        if bin_n < cal.get("min_bin_samples", 30):
            raw = min(raw, cal.get("low_sample_max_score", 6))
        return _clamp_score(raw)

    if sv is None or ev is None or ev <= 0:
        raw = float(sp)
    else:
        raw = 0.58 * sp + 0.42 * sv

    if group == "1x2" and prob < 0.38:
        raw -= 1.0
    elif group in {"ou", "btts", "team"} and prob < 0.50:
        raw -= 0.5
    elif group == "combo" and prob < 0.22:
        raw -= 0.75

    if divergence > 0.12:
        raw -= 1.5
    elif divergence > 0.08:
        raw -= 0.75

    if odds and float(odds) > 1.01:
        qk = _kelly_fraction(prob, float(odds)) * 0.25
        if qk < 0.005:
            raw = min(raw, 4)
        elif qk < 0.015:
            raw = min(raw, 6)
        elif qk < 0.03:
            raw = min(raw, 7)

    if bin_n < cal.get("min_bin_samples", 30):
        raw = min(raw, cal.get("low_sample_max_score", 6))

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
    if play.get("odds") and play.get("fair_odds"):
        ev = play.get("ev") or 0
        bits.append(f"Quota {play['odds']:.2f} vs equa {play['fair_odds']:.2f} (EV {ev:+.0%})")
    label = (alignment or {}).get("label") or "n/d"
    move_lvl = (market_move or {}).get("movement_level") or "Stabile"
    if label == "allineato":
        bits.append(f"mercato Asian {move_lvl.lower()} e allineato")
    elif label == "contrario":
        bits.append(f"mercato Asian {move_lvl.lower()} ma contrario")
    elif move_lvl != "Stabile":
        bits.append(f"mercato {move_lvl.lower()}, modello neutro")
    else:
        bits.append("mercato stabile")

    kelly = 0.0
    if play.get("odds"):
        kelly = _kelly_fraction(p, float(play["odds"])) * 0.25
    if kelly < 0.01:
        bits.append("Kelly ¼ basso → stake teorico minimo")
    elif kelly >= 0.03:
        bits.append(f"Kelly ¼ {kelly:.1%} → stake moderato")
    else:
        bits.append(f"Kelly ¼ {kelly:.1%}")

    line2 = " · ".join(bits)
    return line1, line2


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
        value = score_value(prob, odds)
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
    if odd and float(odd) > 1.01:
        out["kelly_quarter"] = round(_kelly_fraction(float(out["probability"]), float(odd)) * 0.25, 4)
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
        p = float(item["probability"])
        odd = item.get("odds")
        kq = _kelly_fraction(p, float(odd)) * 0.25 if odd else 0.0
        return {
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
            "ev": item["ev"],
            "fair_odds": item["fair_odds"],
            "odds_source": item.get("odds_source"),
            "kelly_quarter": round(kq, 4),
        }

    if value is None:
        return pack(probable, "più_probabile")

    cal = load_calibration()
    min_ev_val = float(cal.get("min_ev_strong_value", 0.06))
    min_p_val = float(cal.get("min_prob_1x2_value", 0.35))

    if probable["code"] == value["code"]:
        kind = "probabile_e_valore" if (value["ev"] or 0) >= 0 else "più_probabile"
        return pack(probable, kind)

    strong_value = (
        (value["ev"] or 0) >= min_ev_val
        and value["probability"] >= min_p_val
        and value["score"] >= probable["score"] + 1
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
    if m.get("group") in {"ou", "btts", "team"}:
        return 0.42 <= p <= 0.80
    return 0.28 <= p <= 0.85


def advise(
    prediction: dict[str, Any],
    odds: dict[str, float] | None = None,
    market_move: dict | None = None,
    *,
    odds_from_asian: bool = False,
) -> dict[str, Any]:
    """Mercati 1X2, doppia chance, DNB, O/U 0.5-4.5, BTTS, gol squadra, combo."""
    odds = odds or {}
    home, away = _split_match(prediction.get("match", "Casa vs Trasferta"))
    mc = prediction["montecarlo"]
    ml = prediction["model_probabilities"]

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
        _market("1", f"{home} vince", "1x2", p1, ranked[1], ml["home_win"], o1, baseline=0.333, odds_source="book"),
        _market("X", "Pareggio", "1x2", px, ranked[1], ml["draw"], ox, baseline=0.333, odds_source="book"),
        _market("2", f"{away} vince", "1x2", p2, ranked[1], ml["away_win"], o2, baseline=0.333, odds_source="book"),
    ]

    def derived(code, name, group, prob, complement, model_p, baseline, book_odd=None, source="stimata"):
        odd = book_odd if book_odd and book_odd > 1 else None
        if odd is None:
            margin = rr_1x2 if group in {"dc", "dnb", "1x2"} else rr_ou
            odd = _apply_margin(prob, margin)
            src = source
        else:
            src = "book"
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
    for line in (0.5, 1.5, 2.5, 3.5, 4.5):
        ok, uk = f"over_{line}", f"under_{line}"
        if ok not in mc:
            continue
        po, pu = float(mc.get(ok, 0)), float(mc.get(uk, 1 - mc.get(ok, 0)))
        src = "book" if line == 2.5 and o_o25 else "stimata da O/U 2.5"
        markets_ou.append(derived(f"O{line}", f"Over {line}", "ou", po, pu, po, 0.5, ou_book.get(ok), src))
        markets_ou.append(derived(f"U{line}", f"Under {line}", "ou", pu, po, pu, 0.5, ou_book.get(uk), src))

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
        src = "book" if book and book > 1 else "stimata combo"
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
    ]

    grouped = {
        "1x2": markets_1x2,
        "dc": markets_dc,
        "ou": markets_ou,
        "btts": markets_btts,
        "team": markets_team,
        "combo": markets_combo,
    }
    all_markets = [_with_composite(m) for g in grouped.values() for m in g]
    for key in grouped:
        grouped[key] = [_with_composite(m) for m in grouped[key]]

    probable_1x2 = max(grouped["1x2"], key=lambda m: m["probability"])
    with_odds_1x2 = [m for m in grouped["1x2"] if m["odds"] is not None]
    best_value_1x2 = max(with_odds_1x2, key=lambda m: m["ev"] if m["ev"] is not None else -99) if with_odds_1x2 else None
    play_1x2 = _pick_headline(probable_1x2, best_value_1x2)

    extras = [m for m in all_markets if m["group"] != "1x2" and _actionable(m)]
    play_alt = None
    if extras:
        def extra_key(m):
            return m["score"] + max(m.get("ev") or -0.05, -0.05) * 3

        play_alt = max(extras, key=extra_key)
        play_alt = _pick_headline(play_alt, play_alt if play_alt["odds"] else None)

    play = play_1x2
    min_ev_play = float(load_calibration().get("min_ev_play", 0.04))
    if play_alt and (play_alt.get("ev") or 0) >= min_ev_play and play_alt["score"] >= play_1x2["score"]:
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
    if market_move:
        from modules.data_update.asian_odds import move_alignment

        alignment = move_alignment(play.get("code"), market_move)
        delta = alignment.get("delta") or 0
        if delta > 0 and odds_from_asian:
            delta = max(0, delta - 1)
        if alignment.get("label") == "contrario" and market_move.get("movement_level") == "Forte":
            score_cap = 6
            delta = min(delta, -1)
        if delta:
            play = dict(play)
            play["score"] = _clamp_score(play["score"] + delta)
        if score_cap is not None:
            play["score"] = min(play["score"], score_cap)
        play["market_align"] = alignment["label"]

    reason1, reason2 = explain_pick(play, alignment=alignment, market_move=market_move, ml_prob=ml_for_play)

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
        "has_odds": any(m["odds"] is not None for m in markets_1x2),
        "market_move": market_move,
        "market_align": alignment or {"agrees": [], "disagrees": [], "delta": 0, "label": "n/d"},
        "score_reason_1": reason1,
        "score_reason_2": reason2,
    }


def format_advice(advice: dict[str, Any]) -> str:
    play = advice["play"]
    kind_it = {
        "più_probabile": "più probabile",
        "valore": "miglior rapporto probabilità/quota",
        "probabile_e_valore": "più probabile e miglior value",
    }.get(play["kind"], play["kind"])
    lines = [
        "",
        "=" * 46,
        f"  {advice['match']}",
        f"  GIOCA  {play['code']}   {play['name']}",
        f"  Voto   {play['score']}/10   ({kind_it})",
        f"  Prob   {play['probability']:.1%}",
    ]
    if play["odds"]:
        ev = play["ev"] or 0
        sign = "+" if ev >= 0 else ""
        lines.append(f"  Quota  {play['odds']:.2f}   equa {play['fair_odds']:.2f}   EV {sign}{ev:.1%}")
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
