"""Cinque controlli di validazione sul voto, non sul modello.

Non toccano EV/Kelly. Fonti già in casa (FBref, Understat, quote football-data,
Monte Carlo). Niente scraping Bet365/Pinnacle/Soccerway.
"""

from __future__ import annotations

from math import sqrt
from typing import Any

from modules.data_update.venues import classify_venue


def _clamp_score(value: float) -> int:
    return int(max(1, min(10, round(float(value)))))

# Soglie richieste.
MARKET_OK = 0.10
MARKET_WARN = 0.15
STABLE_OK = 0.05
STABLE_WARN = 0.08
FORM_GOOD = 2.0  # punti medi ultime 5 (max 3)
FORM_BAD = 0.6


def _f(v: Any) -> float | None:
    try:
        if v is None:
            return None
        x = float(v)
        if x != x:  # NaN
            return None
        return x
    except (TypeError, ValueError):
        return None


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _scale(x: float | None, lo: float, hi: float, *, invert: bool = False) -> float | None:
    if x is None:
        return None
    if hi == lo:
        return 0.5
    t = _clip01((float(x) - lo) / (hi - lo))
    return 1.0 - t if invert else t


def _model_favorite(ml: dict[str, Any] | None, mc: dict[str, Any] | None) -> str | None:
    probs = ml or {}
    h, d, a = _f(probs.get("home_win")), _f(probs.get("draw")), _f(probs.get("away_win"))
    if h is None or d is None or a is None:
        probs = mc or {}
        h, d, a = _f(probs.get("home_win")), _f(probs.get("draw")), _f(probs.get("away_win"))
    if h is None or d is None or a is None:
        return None
    ranked = sorted((("1", h), ("X", d), ("2", a)), key=lambda t: t[1], reverse=True)
    return ranked[0][0]


def _team_tactical_score(
    fb: dict[str, Any] | None,
    us: dict[str, Any] | None,
    ws: dict[str, Any] | None = None,
) -> tuple[float | None, list[str]]:
    """Score 0–1 da dati non soggettivi. PPDA vero non è in FBref pubblico: recuperi p90."""
    fb, us, ws = fb or {}, us or {}, ws or {}
    notes: list[str] = []
    parts: list[float] = []

    poss = _scale(_f(fb.get("poss")), 38.0, 65.0)
    if poss is not None:
        parts.append(poss)
        notes.append(f"possesso {fb.get('poss')}")

    recov = _scale(_f(fb.get("recov_p90")), 38.0, 68.0)
    if recov is not None:
        parts.append(recov)
        notes.append(f"recuperi/90 {float(fb['recov_p90']):.1f} (proxy pressing)")

    xga = _f(us.get("xg_against"))
    if xga is None:
        xga = _f(fb.get("ga_p90"))
    def_s = _scale(xga, 0.75, 1.85, invert=True)
    if def_s is not None:
        parts.append(def_s)
        notes.append(f"xGA/GA {xga:.2f}")

    shots_conc = _scale(_f(fb.get("crosses_conc_p90")), 8.0, 22.0, invert=True)
    if shots_conc is not None:
        parts.append(0.6 * shots_conc + 0.4 * (def_s if def_s is not None else 0.5))

    if ws:
        notes.append("WhoScored stile in cache")
        if poss is not None and recov is not None:
            coherent = (poss >= 0.6 and recov >= 0.55) or (poss <= 0.4 and recov <= 0.45)
            if coherent:
                parts.append(0.72)
                notes.append("modulo/stile coerente (possesso ↔ pressing)")
            elif abs(poss - recov) >= 0.35:
                parts.append(0.38)
                notes.append("stile incoerente (possesso vs pressing)")

    if not parts:
        return None, ["dati tattici insufficienti (Big 5 FBref/Understat)"]
    return round(sum(parts) / len(parts), 3), notes[:4]


def apply_venue_to_probs(
    p_home: float | None,
    p_draw: float | None,
    p_away: float | None,
    penalty_pct: float,
) -> dict[str, float] | None:
    if p_home is None or p_draw is None or p_away is None or not penalty_pct:
        return None
    h = max(0.02, float(p_home) * (1.0 + float(penalty_pct)))
    d = max(0.08, float(p_draw))
    a = max(0.02, float(p_away))
    s = h + d + a
    return {"home": round(h / s, 4), "draw": round(d / s, 4), "away": round(a / s, 4)}


def check_venue(prediction: dict[str, Any]) -> dict[str, Any]:
    home = str(prediction.get("home") or "").strip()
    if not home and prediction.get("match"):
        home = str(prediction["match"]).split(" vs ")[0].strip()
    info = classify_venue(
        home=home,
        venue=prediction.get("venue"),
        league=prediction.get("league"),
        city=prediction.get("venue_city"),
        explicit_neutral=prediction.get("venue_neutral"),
    )
    delta = 0.0
    if info["flag"] == "campo_neutro":
        delta = -0.4
    elif info["flag"] == "stadio_alternativo":
        delta = -0.2
    info["delta_unified"] = delta
    return info


def check_tactical(prediction: dict[str, Any]) -> dict[str, Any]:
    fb = prediction.get("fbref_context") or {}
    us = prediction.get("understat_context") or {}
    tac = prediction.get("tactical") or {}
    ws = (tac.get("whoscored") or {}) if isinstance(tac, dict) else {}
    h_s, h_notes = _team_tactical_score(fb.get("home"), us.get("home"), ws.get("style_home"))
    a_s, a_notes = _team_tactical_score(fb.get("away"), us.get("away"), ws.get("style_away"))
    fav = _model_favorite(prediction.get("model_probabilities"), prediction.get("montecarlo"))
    delta = 0.0
    status = "n/d"
    notes = []
    if h_s is None or a_s is None:
        notes.append("tactical score incompleto")
        return {
            "ready": False,
            "score_home": h_s,
            "score_away": a_s,
            "favorite": fav,
            "agrees": None,
            "delta_prob": 0.0,
            "delta_unified": 0.0,
            "status": status,
            "notes": notes + h_notes + a_notes,
        }

    edge = h_s - a_s
    favors = "1" if edge >= 0.08 else "2" if edge <= -0.08 else "X"
    notes.append(f"tactical {h_s:.2f} vs {a_s:.2f} → lean {favors}")
    notes.extend(h_notes[:2] + a_notes[:2])
    if fav in {"1", "2"} and favors in {"1", "2"}:
        if favors == fav:
            delta = 0.5
            status = "ok"
            notes.append("tattica allineata al favorito modello")
        else:
            delta = -0.5
            status = "contrario"
            notes.append("tattica contraria al favorito modello")
    elif fav == "X" or favors == "X":
        status = "neutro"
        notes.append("nessun mismatch 1 vs 2")
    return {
        "ready": True,
        "score_home": h_s,
        "score_away": a_s,
        "favorite": fav,
        "favors": favors,
        "agrees": None if delta == 0 else delta > 0,
        "delta_prob": delta,
        "delta_unified": delta,
        "status": status,
        "notes": notes,
    }


def check_market(play: dict[str, Any], grouped: dict[str, list] | None = None) -> dict[str, Any]:
    p_cons = _f(play.get("p_cons") if play else None)
    p_mkt = _f(play.get("p_market") if play else None)
    if p_cons is None or p_mkt is None:
        # fallback sul 1X2 più coperto
        for m in (grouped or {}).get("1x2") or []:
            if m.get("p_cons") is not None and m.get("p_market") is not None:
                p_cons, p_mkt = _f(m.get("p_cons")), _f(m.get("p_market"))
                break
    if p_cons is None or p_mkt is None:
        return {
            "ready": False,
            "gap": None,
            "status": "n/d",
            "delta_value": 0.0,
            "delta_unified": 0.0,
            "notes": ["niente probabilità implicita (quote assenti)"],
        }
    gap = p_cons - p_mkt
    ag = abs(gap)
    if ag < MARKET_OK:
        status, delta, note = "ok", 0.0, f"gap {gap:+.1%} pp (<10)"
    elif ag <= MARKET_WARN:
        status, delta, note = "warning", 0.0, f"gap {gap:+.1%} pp (10–15: warning)"
    else:
        status, delta, note = "riduci", -1.0, f"gap {gap:+.1%} pp (>15: voto value −1)"
    return {
        "ready": True,
        "gap": round(gap, 4),
        "p_cons": round(p_cons, 4),
        "p_market": round(p_mkt, 4),
        "status": status,
        "delta_value": delta,
        "delta_unified": delta,
        "notes": [note],
    }


def check_stability(prediction: dict[str, Any]) -> dict[str, Any]:
    ml = prediction.get("model_probabilities") or {}
    mc = prediction.get("montecarlo") or {}
    raw = mc.get("mc_raw") or {}
    h_ml, d_ml, a_ml = _f(ml.get("home_win")), _f(ml.get("draw")), _f(ml.get("away_win"))
    h_mc = _f(raw.get("home_win")) or _f(mc.get("home_win"))
    d_mc = _f(raw.get("draw")) or _f(mc.get("draw"))
    a_mc = _f(raw.get("away_win")) or _f(mc.get("away_win"))
    if None in (h_ml, d_ml, a_ml, h_mc, d_mc, a_mc):
        return {
            "ready": False,
            "diff": None,
            "status": "n/d",
            "delta_prob": 0.0,
            "delta_unified": 0.0,
            "notes": ["ML o Monte Carlo assenti"],
        }
    diff = max(abs(h_ml - h_mc), abs(d_ml - d_mc), abs(a_ml - a_mc))
    std = mc.get("mc_std") or {}
    std_h = _f(std.get("home_win"))
    if std_h is None and mc.get("n_sims"):
        n = max(1, int(mc["n_sims"]))
        std_h = sqrt(h_mc * (1.0 - h_mc) / n)
    if diff <= 0.04:
        status, delta, note = "ok", 0.5, f"ML vs MC {diff:.1%} (quasi identici: +0.5)"
    elif diff < STABLE_OK:
        status, delta, note = "ok", 0.0, f"ML vs MC {diff:.1%} (<5%)"
    elif diff <= STABLE_WARN:
        status, delta, note = "warning", 0.0, f"ML vs MC {diff:.1%} (5–8%: warning)"
    else:
        status, delta, note = "riduci", -0.5, f"ML vs MC {diff:.1%} (>8%: −0.5)"
    return {
        "ready": True,
        "diff": round(diff, 4),
        "mc_raw": {k: raw.get(k) for k in ("home_win", "draw", "away_win")} if raw else None,
        "mc_std_home": None if std_h is None else round(float(std_h), 4),
        "status": status,
        "delta_prob": delta,
        "delta_unified": delta,
        "notes": [note],
    }


def check_form(prediction: dict[str, Any]) -> dict[str, Any]:
    feat = prediction.get("features") or {}
    us = prediction.get("understat_context") or {}
    h_pts = _f(feat.get("home_form_pts"))
    a_pts = _f(feat.get("away_form_pts"))
    h_xg = _f(feat.get("home_xg_avg"))
    h_xga = _f(feat.get("home_xga_avg"))
    a_xg = _f(feat.get("away_xg_avg"))
    a_xga = _f(feat.get("away_xga_avg"))
    if h_xg is None:
        h_xg = _f((us.get("home") or {}).get("xg_for"))
    if h_xga is None:
        h_xga = _f((us.get("home") or {}).get("xg_against"))
    if a_xg is None:
        a_xg = _f((us.get("away") or {}).get("xg_for"))
    if a_xga is None:
        a_xga = _f((us.get("away") or {}).get("xg_against"))

    notes: list[str] = []
    warnings: list[str] = []
    if h_pts is None or a_pts is None:
        return {
            "ready": False,
            "status": "n/d",
            "delta_unified": 0.0,
            "incoherent": False,
            "notes": ["forma ultime 5 assente"],
        }

    fav = _model_favorite(prediction.get("model_probabilities"), prediction.get("montecarlo"))
    fav_pts = h_pts if fav == "1" else a_pts if fav == "2" else (h_pts + a_pts) / 2.0
    delta = 0.0
    status = "ok"
    if fav in {"1", "2"}:
        if fav_pts >= FORM_GOOD:
            delta = 0.3
            status = "positiva"
            notes.append(f"forma positiva {fav_pts:.1f} pti/partita (ultime 5)")
        elif fav_pts <= FORM_BAD:
            delta = -0.3
            status = "negativa"
            notes.append(f"forma negativa {fav_pts:.1f} pti/partita (ultime 5)")
        else:
            notes.append(f"forma neutra {fav_pts:.1f} pti/partita")
    else:
        notes.append(f"forma casa {h_pts:.1f} vs ospite {a_pts:.1f}")

    def _incoherent(pts: float | None, xg: float | None, xga: float | None, side: str) -> bool:
        if pts is None or xg is None or xga is None:
            return False
        if pts >= 1.8 and (xg - xga) <= -0.20:
            warnings.append(f"{side}: vince di risultati ma xG pessimo ({xg:.2f}/{xga:.2f})")
            return True
        if pts <= 0.8 and (xg - xga) >= 0.25:
            warnings.append(f"{side}: risultati bassi ma xG buono ({xg:.2f}/{xga:.2f})")
            return True
        return False

    inc = _incoherent(h_pts, h_xg, h_xga, "casa") or _incoherent(a_pts, a_xg, a_xga, "trasferta")
    if inc:
        status = "warning" if status == "ok" else status
    return {
        "ready": True,
        "home_pts": round(h_pts, 2),
        "away_pts": round(a_pts, 2),
        "status": status,
        "incoherent": bool(inc),
        "delta_unified": delta,
        "notes": notes + warnings,
    }


def check_sportly_sim(prediction: dict[str, Any]) -> dict[str, Any]:
    """Validazione tattica automatica dal modulo Sportly-sim interno."""
    block = (prediction.get("sportly_sim") or {}).get("tactical_validation") or {}
    if not block.get("ready"):
        return {
            "ready": False,
            "status": "n/d",
            "delta_unified": 0.0,
            "notes": ["Sportly-sim non disponibile"],
        }
    return {
        "ready": True,
        "status": block.get("status") or "n/d",
        "lean": block.get("lean"),
        "agrees": block.get("agrees") or [],
        "disagrees": block.get("disagrees") or [],
        "delta_unified": float(block.get("delta_unified") or 0),
        "notes": list(block.get("notes") or []),
    }


def check_data_signal(prediction: dict[str, Any], play: dict[str, Any] | None = None) -> dict[str, Any]:
    """Algoritmo dati (xG/forma/casa/classifiche) → micro Δ voto."""
    from modules.advisor.data_signal import data_signal_validation

    return data_signal_validation(prediction, play)


def run_validation(
    *,
    prediction: dict[str, Any] | None,
    play: dict[str, Any] | None = None,
    grouped: dict[str, list] | None = None,
) -> dict[str, Any]:
    prediction = prediction or {}
    play = play or {}
    # genera sim al volo se manca (singola partita / advise senza upcoming)
    if not (prediction.get("sportly_sim") or {}).get("ready"):
        try:
            from modules.sportly_sim import build_sportly_sim

            prediction = dict(prediction)
            prediction["sportly_sim"] = build_sportly_sim(prediction)
        except Exception:
            pass
    if not (prediction.get("data_signal") or {}).get("ready"):
        try:
            from modules.advisor.data_signal import build_data_signal

            prediction = dict(prediction)
            prediction["data_signal"] = build_data_signal(prediction)
        except Exception:
            pass
    venue = check_venue(prediction)
    tactical = check_tactical(prediction)
    market = check_market(play, grouped)
    stability = check_stability(prediction)
    form = check_form(prediction)
    sportly = check_sportly_sim(prediction)
    data_sig = check_data_signal(prediction, play)

    mc = prediction.get("montecarlo") or {}
    ml = prediction.get("model_probabilities") or {}
    p_adj = apply_venue_to_probs(
        _f(ml.get("home_win")) or _f(mc.get("home_win")),
        _f(ml.get("draw")) or _f(mc.get("draw")),
        _f(ml.get("away_win")) or _f(mc.get("away_win")),
        float(venue.get("penalty_pct") or 0),
    )

    delta_unified = (
        float(venue.get("delta_unified") or 0)
        + float(tactical.get("delta_unified") or 0)
        + float(market.get("delta_unified") or 0)
        + float(stability.get("delta_unified") or 0)
        + float(form.get("delta_unified") or 0)
        + float(sportly.get("delta_unified") or 0)
        + float(data_sig.get("delta_unified") or 0)
    )
    delta_prob = float(tactical.get("delta_prob") or 0) + float(stability.get("delta_prob") or 0)
    delta_value = float(market.get("delta_value") or 0)

    warnings: list[str] = []
    for block in (venue, tactical, market, stability, form, sportly, data_sig):
        if block.get("status") in {"warning", "contrario", "riduci", "negativa"}:
            warnings.extend(block.get("notes") or [])
        elif block.get("incoherent"):
            warnings.extend(block.get("notes") or [])

    summary_bits = [
        f"stadio {venue.get('flag')}",
        f"tattica {tactical.get('status')}",
        f"mercato {market.get('status')}",
        f"ML/MC {stability.get('status')}",
        f"forma {form.get('status')}",
        f"sim {sportly.get('status')}",
        f"dati {data_sig.get('status')}",
    ]
    return {
        "venue": venue,
        "tactical": tactical,
        "market": market,
        "stability": stability,
        "form": form,
        "sportly_sim": sportly,
        "data_signal": data_sig,
        "p_validated": p_adj,
        "delta_unified": round(delta_unified, 3),
        "delta_prob": round(delta_prob, 3),
        "delta_value": round(delta_value, 3),
        "warnings": warnings,
        "summary": " · ".join(summary_bits),
    }


def apply_to_play(play: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    """Aggiusta voto probabilità / value del pick. Non tocca p_cons, EV, Kelly."""
    out = dict(play)
    out["validation"] = validation
    if validation.get("p_validated"):
        out["p_validated"] = validation["p_validated"]
    sp = out.get("score_prob")
    if sp is not None:
        out["score_prob"] = _clamp_score(float(sp) + float(validation.get("delta_prob") or 0))
    sv = out.get("score_value")
    if sv is not None and validation.get("delta_value"):
        out["score_value"] = _clamp_score(float(sv) + float(validation.get("delta_value") or 0))
    return out
