"""Trasforma una predizione in un consiglio 1/X/2 con voto 1-10.

Due segnali distinti:
- più probabile: esito con probabilità Monte Carlo più alta
- miglior value: miglior rapporto probabilità / quota implicita (EV = p * quota - 1)

Il voto combina dominanza dell'esito, accordo modello vs Monte Carlo, e (se ci sono
le quote) l'edge rispetto al bookmaker.
"""

from __future__ import annotations

from math import floor
from typing import Any


def _clamp_score(value: float) -> int:
    return int(max(1, min(10, floor(value + 0.5))))


def _fair_odds(prob: float) -> float | None:
    if prob <= 0:
        return None
    return round(1.0 / prob, 2)


def score_probability(p: float, p_second: float, p_model: float) -> int:
    """Voto 1-10 sulla convinzione dell'esito (senza quote)."""
    dominance = max(0.0, min(1.0, (p - 0.3333) / 0.4667))
    gap = max(0.0, min(1.0, (p - p_second) / 0.40))
    agree = 1.0 - min(1.0, abs(p - p_model) / 0.25)
    raw = 1 + 9 * (0.55 * dominance + 0.30 * gap + 0.15 * agree)
    return _clamp_score(raw)


def score_value(prob: float, odds: float) -> int:
    """Voto 1-10 sul rapporto probabilità/quota. 4 = quota equa, 10 = edge forte."""
    ev = prob * odds - 1.0
    if ev >= 0:
        raw = 4 + 6 * min(1.0, ev / 0.20)
    else:
        raw = 4 - 3 * min(1.0, -ev / 0.25)
    if prob < 0.15:
        raw -= 2.0
    elif prob < 0.22:
        raw -= 1.0
    return _clamp_score(raw)


def _market(
    code: str,
    name: str,
    prob: float,
    p_second: float,
    p_model: float,
    odds: float | None,
) -> dict[str, Any]:
    ev = None
    ratio = None
    implied = None
    value = None
    if odds is not None and odds > 1.0:
        implied = round(1.0 / odds, 4)
        ev = round(prob * odds - 1.0, 4)
        ratio = round(prob * odds, 4)
        value = score_value(prob, odds)
    return {
        "code": code,
        "name": name,
        "probability": round(prob, 4),
        "model_probability": round(p_model, 4),
        "fair_odds": _fair_odds(prob),
        "odds": odds,
        "implied_prob": implied,
        "ev": ev,
        "prob_odds_ratio": ratio,
        "score_prob": score_probability(prob, p_second, p_model),
        "score_value": value,
    }


def _pick_headline(probable: dict, value: dict | None) -> dict[str, Any]:
    if value is None:
        return {
            "code": probable["code"],
            "name": probable["name"],
            "kind": "più_probabile",
            "score": probable["score_prob"],
            "probability": probable["probability"],
            "odds": probable["odds"],
            "ev": probable["ev"],
            "fair_odds": probable["fair_odds"],
        }

    same = probable["code"] == value["code"]
    if same:
        score = _clamp_score(0.5 * probable["score_prob"] + 0.5 * (value["score_value"] or probable["score_prob"]))
        kind = "probabile_e_valore" if (value["ev"] or 0) >= 0 else "più_probabile"
        return {
            "code": probable["code"],
            "name": probable["name"],
            "kind": kind,
            "score": score,
            "probability": probable["probability"],
            "odds": probable["odds"],
            "ev": probable["ev"],
            "fair_odds": probable["fair_odds"],
        }

    # Value diventa il consiglio solo se l'edge è chiaro e non è un longshot estremo.
    strong_value = (
        (value["ev"] or 0) >= 0.08
        and (value["score_value"] or 0) >= probable["score_prob"]
        and value["probability"] >= 0.22
    )
    chosen = value if strong_value else probable
    score = value["score_value"] if strong_value else probable["score_prob"]
    return {
        "code": chosen["code"],
        "name": chosen["name"],
        "kind": "valore" if strong_value else "più_probabile",
        "score": score,
        "probability": chosen["probability"],
        "odds": chosen["odds"],
        "ev": chosen["ev"],
        "fair_odds": chosen["fair_odds"],
    }


def advise(prediction: dict[str, Any], odds: dict[str, float] | None = None) -> dict[str, Any]:
    """Calcola consiglio 1/X/2 (+ mercati extra se le quote sono presenti).

    `odds` chiavi ammesse: home / draw / away (oppure 1 / X / 2),
    over_2.5, under_2.5, btts_yes, btts_no.
    """
    odds = odds or {}
    home, away = _split_match(prediction.get("match", "Casa vs Trasferta"))
    mc = prediction["montecarlo"]
    ml = prediction["model_probabilities"]

    p1, px, p2 = mc["home_win"], mc["draw"], mc["away_win"]
    ranked = sorted([p1, px, p2], reverse=True)
    p_second = ranked[1]

    o1 = _get_odd(odds, "home", "1")
    ox = _get_odd(odds, "draw", "X", "x")
    o2 = _get_odd(odds, "away", "2")

    markets = [
        _market("1", f"{home} vince", p1, p_second, ml["home_win"], o1),
        _market("X", "Pareggio", px, p_second, ml["draw"], ox),
        _market("2", f"{away} vince", p2, p_second, ml["away_win"], o2),
    ]

    probable = max(markets, key=lambda m: m["probability"])
    with_odds = [m for m in markets if m["odds"] is not None]
    best_value = max(with_odds, key=lambda m: m["ev"] if m["ev"] is not None else -99) if with_odds else None

    extras: list[dict[str, Any]] = []
    extra_specs = [
        ("over_2.5", "Over 2.5", mc.get("over_2.5"), 1 - mc.get("over_2.5", 0), mc.get("over_2.5")),
        ("under_2.5", "Under 2.5", mc.get("under_2.5"), 1 - mc.get("under_2.5", 0), mc.get("under_2.5")),
        ("btts_yes", "Gol (BTTS sì)", mc.get("btts"), 1 - mc.get("btts", 0), mc.get("btts")),
        ("btts_no", "No gol (BTTS no)", 1 - mc.get("btts", 0), mc.get("btts"), 1 - mc.get("btts", 0)),
    ]
    for key, name, prob, p_sec, p_mod in extra_specs:
        odd = _get_odd(odds, key)
        if prob is None:
            continue
        extras.append(_market(key, name, float(prob), float(p_sec or 0), float(p_mod or prob), odd))

    play = _pick_headline(probable, best_value)
    return {
        "match": prediction.get("match"),
        "home": home,
        "away": away,
        "play": play,
        "most_probable": probable,
        "best_value": best_value,
        "markets": markets,
        "extras": extras,
        "expected_goals": prediction.get("expected_goals"),
        "most_likely_scores": mc.get("most_likely_scores", []),
        "has_odds": bool(with_odds),
    }


def format_advice(advice: dict[str, Any]) -> str:
    play = advice["play"]
    kind_it = {
        "più_probabile": "più probabile",
        "valore": "miglior rapporto probabilità/quota",
        "probabile_e_valore": "più probabile e miglior value",
    }[play["kind"]]
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
    else:
        lines.append("  Inserisci le quote per il voto sul value.")
    if advice["best_value"] and advice["best_value"]["code"] != play["code"]:
        v = advice["best_value"]
        ev = v["ev"] or 0
        sign = "+" if ev >= 0 else ""
        lines.append(f"  Value  {v['code']}  {v['name']}  {v['score_value']}/10  EV {sign}{ev:.1%}")
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
