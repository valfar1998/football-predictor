"""Sportly-like interno: xG, momentum, shot map, pressione, trend — tutto simulato.

Niente scraping, niente sportly/FotMob. Parte da λ gol, feature, Understat/FBref
e matchup tattico già in casa. Solo quadro/validazione: non tocca EV/Kelly.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def _f(v: Any, default: float | None = None) -> float | None:
    try:
        if v is None:
            return default
        x = float(v)
        if x != x:
            return default
        return x
    except (TypeError, ValueError):
        return default


def _seed(*parts: Any) -> int:
    raw = "|".join(str(p) for p in parts)
    return int(hashlib.md5(raw.encode("utf-8")).hexdigest()[:8], 16)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _style_weights(prediction: dict[str, Any]) -> dict[str, float]:
    """Possesso / pressing da FBref o proxy da Elo/xG."""
    fb = prediction.get("fbref_context") or {}
    h_fb, a_fb = fb.get("home") or {}, fb.get("away") or {}
    feat = prediction.get("features") or {}
    poss_h = _f(h_fb.get("poss"))
    poss_a = _f(a_fb.get("poss"))
    if poss_h is None or poss_a is None:
        elo_h = _f(feat.get("home_elo"), 1500.0) or 1500.0
        elo_a = _f(feat.get("away_elo"), 1500.0) or 1500.0
        # proxy: favorito tiene un filo di possesso in più
        base = 0.52
        gap = _clip((elo_h - elo_a) / 800.0, -0.12, 0.12)
        poss_h = (base + gap) * 100.0
        poss_a = 100.0 - poss_h
    tot = max(1.0, poss_h + poss_a)
    share_h = poss_h / tot
    share_a = 1.0 - share_h
    # pressing proxy: recuperi o inverso possesso
    press_h = _f(h_fb.get("recov_p90"))
    press_a = _f(a_fb.get("recov_p90"))
    if press_h is None:
        press_h = 55.0 - (share_h - 0.5) * 40.0
    if press_a is None:
        press_a = 55.0 - (share_a - 0.5) * 40.0
    return {
        "poss_home": share_h,
        "poss_away": share_a,
        "press_home": _clip(press_h / 70.0, 0.35, 1.15),
        "press_away": _clip(press_a / 70.0, 0.35, 1.15),
    }


def _lambdas(prediction: dict[str, Any]) -> tuple[float, float] | None:
    xg = prediction.get("expected_goals") or {}
    lh = _f(xg.get("home"))
    la = _f(xg.get("away"))
    if lh is None or la is None:
        feat = prediction.get("features") or {}
        lh = _f(feat.get("home_xg_avg"))
        la = _f(feat.get("away_xg_avg"))
    us = prediction.get("understat_context") or {}
    uh, ua = us.get("home") or {}, us.get("away") or {}
    if lh is None:
        lh = _f(uh.get("xg_for"))
    if la is None:
        la = _f(ua.get("xg_for"))
    if lh is None or la is None:
        return None
    return max(0.25, float(lh)), max(0.20, float(la))


def _intensity_curve(minutes: np.ndarray, side_boost: float, rng: np.random.Generator) -> np.ndarray:
    """Profilo tipico: più vivo 1–15 e 70–90, buco a metà."""
    early = np.exp(-((minutes - 10) / 12) ** 2)
    late = np.exp(-((minutes - 78) / 14) ** 2)
    mid = 0.55 + 0.15 * np.sin(minutes / 90.0 * np.pi)
    noise = 1.0 + 0.08 * rng.normal(0, 1, size=len(minutes))
    base = 0.45 * early + 0.35 * late + 0.35 * mid
    return np.clip(base * side_boost * noise, 0.15, 2.2)


def _sample_shots(
    rng: np.random.Generator,
    *,
    n: int,
    team: str,
    attacking_right: bool,
) -> list[dict[str, Any]]:
    """Coordinate pitch 0–100 (lunghezza) × 0–100 (larghezza), xG shot tipici."""
    shots: list[dict[str, Any]] = []
    for _ in range(max(0, n)):
        # zona pericolosa: terzo offensivo
        if attacking_right:
            x = float(rng.beta(5, 2) * 45 + 55)  # 55–100
        else:
            x = float(rng.beta(2, 5) * 45)  # 0–45
        y = float(rng.normal(50, 18))
        y = _clip(y, 5, 95)
        # distanza dalla porta
        goal_x = 100.0 if attacking_right else 0.0
        dist = abs(goal_x - x) + abs(50.0 - y) * 0.35
        xg = float(np.clip(np.exp(-dist / 28.0) * rng.uniform(0.55, 1.25), 0.02, 0.55))
        on_target = bool(rng.random() < 0.35 + xg)
        shots.append(
            {
                "team": team,
                "x": round(x, 1),
                "y": round(y, 1),
                "xg": round(xg, 3),
                "on_target": on_target,
            }
        )
    return shots


def _lean_from_xg(lh: float, la: float) -> tuple[str, float, float, float]:
    """1X2 grezzo da differenza xG (Dixon-Coles light via Poisson indipendente)."""
    from modules.predictor.poisson import dixon_coles_1x2

    p1, px, p2 = dixon_coles_1x2(lh, la)
    lean = "1" if p1 >= px and p1 >= p2 else "2" if p2 >= px else "X"
    return lean, p1, px, p2


def _tactical_validation(
    prediction: dict[str, Any],
    *,
    lean: str,
    p1: float,
    px: float,
    p2: float,
) -> dict[str, Any]:
    ml = prediction.get("model_probabilities") or {}
    mc = prediction.get("montecarlo") or {}
    tac = prediction.get("tactical") or {}

    def fav(probs: dict[str, Any], keys=("home_win", "draw", "away_win")) -> str | None:
        vals = [_f(probs.get(k)) for k in keys]
        if any(v is None for v in vals):
            return None
        ranked = sorted(zip(("1", "X", "2"), vals), key=lambda t: t[1], reverse=True)
        return ranked[0][0]

    ml_fav = fav(ml)
    mc_fav = fav(mc)
    tac_edge = _f((tac.get("style") or {}).get("edge_home"))
    if tac_edge is None:
        tac_edge = _f(tac.get("edge_home"))
    tac_lean = None
    if tac_edge is not None:
        if tac_edge >= 0.06:
            tac_lean = "1"
        elif tac_edge <= -0.06:
            tac_lean = "2"
        else:
            tac_lean = "X"

    agrees: list[str] = []
    disagrees: list[str] = []
    for name, other in (("ML", ml_fav), ("MC", mc_fav), ("tattica", tac_lean)):
        if other is None:
            continue
        if other == lean:
            agrees.append(name)
        else:
            disagrees.append(f"{name}→{other}")

    # delta soft sul voto unificato (non EV)
    if len(agrees) >= 2 and not disagrees:
        delta, status = 0.5, "ok"
        note = f"sim lean {lean} confermato da {', '.join(agrees)}"
    elif disagrees and not agrees:
        delta, status = -0.5, "contrario"
        note = f"sim lean {lean} vs {', '.join(disagrees)}"
    elif agrees and disagrees:
        delta, status = 0.0, "misto"
        note = f"sim {lean}: ok {', '.join(agrees)} · no {', '.join(disagrees)}"
    else:
        delta, status = 0.0, "n/d"
        note = f"sim lean {lean} (poche fonti di confronto)"

    return {
        "ready": True,
        "lean": lean,
        "p_1": round(p1, 4),
        "p_x": round(px, 4),
        "p_2": round(p2, 4),
        "ml_fav": ml_fav,
        "mc_fav": mc_fav,
        "tactical_lean": tac_lean,
        "agrees": agrees,
        "disagrees": disagrees,
        "delta_unified": delta,
        "status": status,
        "notes": [note],
    }


def build_sportly_sim(
    prediction: dict[str, Any] | None,
    *,
    step: int = 5,
    detail: bool = False,
) -> dict[str, Any]:
    """Costruisce il pacchetto Sportly-like. `detail=True` → curve al minuto."""
    prediction = prediction or {}
    empty = {
        "ready": False,
        "source": "internal_sim",
        "note": "λ gol assenti: impossibile simulare",
    }
    lams = _lambdas(prediction)
    if not lams:
        return empty
    lh, la = lams
    home = str(prediction.get("home") or "home")
    away = str(prediction.get("away") or "away")
    date = str(prediction.get("date") or prediction.get("kickoff") or "")
    rng = np.random.default_rng(_seed(home, away, date, round(lh, 3), round(la, 3)))

    style = _style_weights(prediction)
    minutes = np.arange(1, 91, dtype=float)
    ih = _intensity_curve(minutes, 0.85 + 0.35 * style["poss_home"], rng)
    ia = _intensity_curve(minutes, 0.85 + 0.35 * style["poss_away"], rng)
    # normalizza intensità così che ∫≈λ
    rate_h = ih / ih.sum() * lh
    rate_a = ia / ia.sum() * la

    cum_h = np.cumsum(rate_h)
    cum_a = np.cumsum(rate_a)
    # momentum: differenza smussata dei rate
    mom = np.convolve(rate_h - rate_a, np.ones(5) / 5.0, mode="same")
    mom = mom / (np.max(np.abs(mom)) + 1e-9)
    # pressione: rate × pressing relativo
    press_h = rate_h * style["press_home"]
    press_a = rate_a * style["press_away"]
    press_h = press_h / (press_h.mean() + 1e-9)
    press_a = press_a / (press_a.mean() + 1e-9)

    # shot counts ~ 7–14 per punto di xG
    n_h = int(rng.integers(max(4, int(lh * 6)), max(5, int(lh * 11)) + 1))
    n_a = int(rng.integers(max(3, int(la * 6)), max(4, int(la * 11)) + 1))
    shots = _sample_shots(rng, n=n_h, team="home", attacking_right=True)
    shots += _sample_shots(rng, n=n_a, team="away", attacking_right=False)
    rng.shuffle(shots)
    if not detail:
        shots = shots[:16]

    lean, p1, px, p2 = _lean_from_xg(lh, la)
    tac_val = _tactical_validation(prediction, lean=lean, p1=p1, px=px, p2=p2)

    # trend live a blocchi 0–30 / 30–60 / 60–90
    phases = []
    for lo, hi, label in ((0, 30, "1° tempo inizio"), (30, 60, "centrale"), (60, 90, "chiusura")):
        sl = slice(lo, hi)
        dom = float(rate_h[sl].sum() - rate_a[sl].sum())
        if dom > 0.08:
            phase_lean = "casa"
        elif dom < -0.08:
            phase_lean = "trasferta"
        else:
            phase_lean = "equilibrio"
        phases.append(
            {
                "from": lo,
                "to": hi,
                "label": label,
                "lean": phase_lean,
                "xg_home": round(float(rate_h[sl].sum()), 3),
                "xg_away": round(float(rate_a[sl].sum()), 3),
            }
        )

    def downsample(arr: np.ndarray) -> list[float]:
        idx = list(range(step - 1, 90, step)) + ([89] if 89 not in range(step - 1, 90, step) else [])
        return [round(float(arr[i]), 4) for i in idx]

    minute_axis = list(range(step, 91, step))
    if minute_axis[-1] != 90:
        minute_axis.append(90)

    out: dict[str, Any] = {
        "ready": True,
        "source": "internal_sim",
        "note": "sintetico da λ/stile interno — non è feed FotMob/Sofascore live",
        "xg": {
            "home": round(lh, 3),
            "away": round(la, 3),
            "minutes": minute_axis,
            "cum_home": downsample(cum_h),
            "cum_away": downsample(cum_a),
        },
        "momentum": {
            "minutes": minute_axis,
            "values": downsample(mom),
            "final": round(float(mom[-1]), 3),
            "avg": round(float(mom.mean()), 3),
        },
        "pressure": {
            "home_avg": round(float(press_h.mean()), 3),
            "away_avg": round(float(press_a.mean()), 3),
            "minutes": minute_axis,
            "home": downsample(press_h),
            "away": downsample(press_a),
        },
        "shots": {
            "home_n": n_h,
            "away_n": n_a,
            "map": shots,
        },
        "live_trend": phases,
        "tactical_validation": tac_val,
        "lean": lean,
        "p_1": round(p1, 4),
        "p_x": round(px, 4),
        "p_2": round(p2, 4),
    }
    if detail:
        out["xg"]["minutes_full"] = list(range(1, 91))
        out["xg"]["cum_home_full"] = [round(float(x), 4) for x in cum_h]
        out["xg"]["cum_away_full"] = [round(float(x), 4) for x in cum_a]
        out["momentum"]["values_full"] = [round(float(x), 4) for x in mom]
    return out
