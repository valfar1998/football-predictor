"""Algoritmo di analisi basato sui dati (xG, forma, casa/trasferta, classifiche).

Fonde segnali già in cache (feature rolling, Understat, FBref, FotMob, Sofascore,
StatsBomb, Elo) in un lean 1X2 + confidenza + breakdown fattori.

Solo quadro / validazione (±0.5 sul voto unificato). Non tocca EV, Kelly, p_cons.
"""

from __future__ import annotations

from typing import Any


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


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _lean_1x2(p1: float | None, px: float | None, p2: float | None) -> str | None:
    vals = [("1", p1), ("X", px), ("2", p2)]
    ranked = [(k, v) for k, v in vals if v is not None]
    if not ranked:
        return None
    ranked.sort(key=lambda t: t[1], reverse=True)
    return ranked[0][0]


def _two_way_to_1x2(p_home_2way: float, p_draw: float) -> tuple[float, float, float]:
    d = max(0.12, min(0.40, float(p_draw)))
    p1 = max(0.04, p_home_2way * (1.0 - d))
    p2 = max(0.04, (1.0 - p_home_2way) * (1.0 - d))
    total = p1 + d + p2
    return p1 / total, d / total, p2 / total


def _factor(name: str, edge: float, weight: float, note: str) -> dict[str, Any]:
    return {
        "name": name,
        "edge": round(float(edge), 4),
        "weight": round(float(weight), 3),
        "note": note,
    }


def build_data_signal(prediction: dict[str, Any]) -> dict[str, Any]:
    """Restituisce lean, probabilità, confidenza e elenco fattori usati."""
    feat = prediction.get("features") or {}
    us = prediction.get("understat_context") or {}
    fb = prediction.get("fbref_context") or {}
    fm = prediction.get("fotmob_context") or {}
    sofa = prediction.get("sofascore_context") or {}
    sb = prediction.get("statsbomb_context") or {}
    us_h, us_a = us.get("home") or {}, us.get("away") or {}
    fb_h, fb_a = fb.get("home") or {}, fb.get("away") or {}
    fm_h, fm_a = fm.get("home") or {}, fm.get("away") or {}
    sofa_h, sofa_a = sofa.get("home") or {}, sofa.get("away") or {}
    sb_h, sb_a = sb.get("home") or {}, sb.get("away") or {}
    fm_m = fm.get("match") or {}

    factors: list[dict[str, Any]] = []

    # 1) Forma rolling (punti ultime N)
    h_pts = _f(feat.get("home_form_pts"))
    a_pts = _f(feat.get("away_form_pts"))
    if h_pts is not None and a_pts is not None:
        edge = _clip((h_pts - a_pts) / 12.0, -0.35, 0.35)
        factors.append(_factor("forma", edge, 0.18, f"pts {h_pts:.1f} vs {a_pts:.1f}"))

    # 2) Casa / trasferta
    h_wr = _f(feat.get("home_home_wr"))
    a_wr = _f(feat.get("away_away_wr"))
    if h_wr is not None and a_wr is not None:
        # gap tipico casa≈0.45, trasferta≈0.30 → baseline ~0.15 a favore casa
        edge = _clip((h_wr - a_wr - 0.12) * 0.9, -0.30, 0.30)
        factors.append(_factor("casa/trasferta", edge, 0.16, f"WR {h_wr:.0%} vs {a_wr:.0%}"))

    # 3) xG rolling dalle feature
    h_xg = _f(feat.get("home_xg_avg"))
    a_xg = _f(feat.get("away_xg_avg"))
    h_xga = _f(feat.get("home_xga_avg"))
    a_xga = _f(feat.get("away_xga_avg"))
    if h_xg is not None and a_xg is not None:
        att = h_xg - a_xg
        deff = 0.0
        if h_xga is not None and a_xga is not None:
            deff = a_xga - h_xga  # casa concede meno → positivo
        edge = _clip((att + 0.55 * deff) / 2.2, -0.35, 0.35)
        factors.append(
            _factor(
                "xG rolling",
                edge,
                0.14,
                f"{h_xg:.2f}/{h_xga if h_xga is not None else 'n/d'} vs "
                f"{a_xg:.2f}/{a_xga if a_xga is not None else 'n/d'}",
            )
        )

    # 4) Understat xG reale
    h_diff = _f(us_h.get("xg_diff"))
    a_diff = _f(us_a.get("xg_diff"))
    if h_diff is not None and a_diff is not None:
        edge = _clip((h_diff - a_diff) / 1.6, -0.40, 0.40)
        factors.append(
            _factor(
                "Understat",
                edge,
                0.18,
                f"xGdiff {h_diff:+.2f} vs {a_diff:+.2f}",
            )
        )
    else:
        h_xf, a_xf = _f(us_h.get("xg_for")), _f(us_a.get("xg_for"))
        h_xa, a_xa = _f(us_h.get("xg_against")), _f(us_a.get("xg_against"))
        if h_xf is not None and a_xf is not None:
            att = h_xf - a_xf
            deff = 0.0
            if h_xa is not None and a_xa is not None:
                deff = a_xa - h_xa
            edge = _clip((att + 0.5 * deff) / 2.0, -0.35, 0.35)
            factors.append(_factor("Understat", edge, 0.14, f"xG {h_xf:.2f} vs {a_xf:.2f}"))

    # 5) Classifica attuale: FotMob se played≥1, altrimenti Sofascore
    table_edge = None
    table_note = ""
    table_w = 0.12
    if (
        _f(fm_h.get("played"), 0) >= 1
        and _f(fm_a.get("played"), 0) >= 1
        and _f(fm_h.get("ppg")) is not None
        and _f(fm_a.get("ppg")) is not None
    ):
        ppg = _f(fm_h.get("ppg"), 0.0) - _f(fm_a.get("ppg"), 0.0)
        gd = (_f(fm_h.get("gd_pg"), 0.0) or 0.0) - (_f(fm_a.get("gd_pg"), 0.0) or 0.0)
        table_edge = _clip(ppg / 2.0 + gd * 0.04, -0.35, 0.35)
        table_note = f"FotMob PPG {fm_h.get('ppg')} vs {fm_a.get('ppg')}"
        table_w = 0.13
    elif _f(sofa_h.get("ppg")) is not None and _f(sofa_a.get("ppg")) is not None:
        ppg = _f(sofa_h.get("ppg"), 0.0) - _f(sofa_a.get("ppg"), 0.0)
        gd = (_f(sofa_h.get("gd_pg"), 0.0) or 0.0) - (_f(sofa_a.get("gd_pg"), 0.0) or 0.0)
        table_edge = _clip(ppg / 2.0 + gd * 0.04, -0.35, 0.35)
        table_note = f"Sofascore PPG {sofa_h.get('ppg')} vs {sofa_a.get('ppg')}"
    if table_edge is not None:
        factors.append(_factor("classifica", table_edge, table_w, table_note))

    # 6) FBref stile / GD
    if _f(fb_h.get("gd_pg")) is not None and _f(fb_a.get("gd_pg")) is not None:
        edge = _clip((_f(fb_h.get("gd_pg"), 0) - _f(fb_a.get("gd_pg"), 0)) * 0.22, -0.30, 0.30)
        factors.append(
            _factor(
                "FBref",
                edge,
                0.10,
                f"GD/pg {fb_h.get('gd_pg')} vs {fb_a.get('gd_pg')}",
            )
        )

    # 7) StatsBomb open data (peso basso: spesso storico)
    if (
        _f(sb_h.get("n"), 0) >= 5
        and _f(sb_a.get("n"), 0) >= 5
        and _f(sb_h.get("gd_pg")) is not None
        and _f(sb_a.get("gd_pg")) is not None
    ):
        edge = _clip((_f(sb_h.get("gd_pg"), 0) - _f(sb_a.get("gd_pg"), 0)) * 0.18, -0.25, 0.25)
        factors.append(
            _factor(
                "StatsBomb",
                edge,
                0.06,
                f"GD/pg open {sb_h.get('gd_pg')} vs {sb_a.get('gd_pg')}",
            )
        )

    # 8) Riposo
    rest_h = _f(feat.get("home_rest_days"))
    rest_a = _f(feat.get("away_rest_days"))
    if rest_h is not None and rest_a is not None:
        edge = _clip((rest_h - rest_a) / 14.0, -0.12, 0.12)
        factors.append(_factor("riposo", edge, 0.06, f"{rest_h:.0f}d vs {rest_a:.0f}d"))

    # 9) Elo (se in feature)
    elo_h = _f(feat.get("home_elo"))
    elo_a = _f(feat.get("away_elo"))
    if elo_h is not None and elo_a is not None and abs(elo_h - elo_a) > 1:
        # HFA ~65 Elo
        p_elo = 1.0 / (1.0 + 10 ** (-((elo_h + 65.0 - elo_a) / 400.0)))
        edge = _clip(p_elo - 0.5, -0.35, 0.35)
        factors.append(_factor("Elo", edge, 0.08, f"{elo_h:.0f} vs {elo_a:.0f}"))

    if not factors:
        note = "nessun fattore dati disponibile"
        if fm_m.get("match_id"):
            note += f" · FotMob id {fm_m['match_id']}"
        return {
            "ready": False,
            "lean": None,
            "p_1": None,
            "p_x": None,
            "p_2": None,
            "edge": None,
            "confidence": 0.0,
            "n_factors": 0,
            "factors": [],
            "note": note,
        }

    w_sum = sum(f["weight"] for f in factors) or 1.0
    edge = sum(f["edge"] * f["weight"] for f in factors) / w_sum
    edge = _clip(edge, -0.42, 0.42)

    mc = prediction.get("montecarlo") or {}
    ml = prediction.get("model_probabilities") or {}
    p_draw = _f(mc.get("draw")) or _f(ml.get("draw")) or 0.26
    p_h2 = _clip(0.5 + edge, 0.18, 0.82)
    p1, px, p2 = _two_way_to_1x2(p_h2, p_draw)
    lean = _lean_1x2(p1, px, p2)

    # Confidenza: copertura fattori + forza edge + accordo interno dei segni
    coverage = min(1.0, len(factors) / 6.0)
    strength = min(1.0, abs(edge) / 0.22)
    signs = [1 if f["edge"] > 0.02 else (-1 if f["edge"] < -0.02 else 0) for f in factors]
    nonzero = [s for s in signs if s != 0]
    if nonzero:
        agree = sum(1 for s in nonzero if s == (1 if edge >= 0 else -1)) / len(nonzero)
    else:
        agree = 0.5
    confidence = _clip(0.35 * coverage + 0.40 * strength + 0.25 * agree, 0.05, 0.95)

    # Accordo con ML/MC (solo informativo)
    ml_lean = _lean_1x2(ml.get("home_win"), ml.get("draw"), ml.get("away_win"))
    mc_lean = _lean_1x2(mc.get("home_win"), mc.get("draw"), mc.get("away_win"))
    agrees = [s for s in (ml_lean, mc_lean) if s and s == lean]
    disagrees = [s for s, lab in ((ml_lean, "ML"), (mc_lean, "MC")) if s and s != lean]

    top = sorted(factors, key=lambda f: abs(f["edge"]) * f["weight"], reverse=True)[:3]
    note = " · ".join(f"{t['name']} {t['edge']:+.2f}" for t in top)
    if fm_m.get("match_id") and not any(f["name"] == "classifica" and "FotMob" in f["note"] for f in factors):
        note = (note + " · " if note else "") + f"FotMob id {fm_m['match_id']}"

    # --- Varianti per mercato (pesi diversi) ---
    markets = _market_variants(factors, feat, us_h, us_a, p_draw)

    return {
        "ready": True,
        "lean": lean,
        "p_1": round(p1, 4),
        "p_x": round(px, 4),
        "p_2": round(p2, 4),
        "edge": round(edge, 4),
        "p_home_2way": round(p_h2, 4),
        "confidence": round(confidence, 3),
        "n_factors": len(factors),
        "factors": factors,
        "markets": markets,
        "agrees_model": agrees,
        "disagrees_model": [x for x in (ml_lean, mc_lean) if x and x != lean],
        "note": note,
    }


def _market_variants(
    factors: list[dict[str, Any]],
    feat: dict[str, Any],
    us_h: dict[str, Any],
    us_a: dict[str, Any],
    p_draw: float,
) -> dict[str, Any]:
    """Edge dedicato 1X2 vs O/U (e proxy AH)."""
    w_1x2 = {
        "forma": 0.9,
        "casa/trasferta": 1.2,
        "Elo": 1.3,
        "classifica": 1.1,
        "Understat": 0.85,
        "xG rolling": 0.8,
        "FBref": 0.7,
        "StatsBomb": 0.5,
        "riposo": 0.9,
    }
    w_ou = {
        "forma": 0.7,
        "casa/trasferta": 0.4,
        "Elo": 0.35,
        "classifica": 0.5,
        "Understat": 1.35,
        "xG rolling": 1.4,
        "FBref": 1.1,
        "StatsBomb": 0.9,
        "riposo": 0.6,
    }
    w_ah = {
        "forma": 0.8,
        "casa/trasferta": 1.0,
        "Elo": 1.4,
        "classifica": 1.0,
        "Understat": 0.9,
        "xG rolling": 0.85,
        "FBref": 0.7,
        "StatsBomb": 0.5,
        "riposo": 0.7,
    }

    def _edge_for(wmap: dict[str, float]) -> float:
        num = den = 0.0
        for f in factors:
            w = float(f["weight"]) * float(wmap.get(f["name"], 0.6))
            num += float(f["edge"]) * w
            den += w
        return _clip(num / den, -0.42, 0.42) if den else 0.0

    e1 = _edge_for(w_1x2)
    e_ou = _edge_for(w_ou)
    e_ah = _edge_for(w_ah)

    # O/U: edge positivo → Over (gol attesi alti)
    h_xg = _f(feat.get("home_xg_avg")) or _f(us_h.get("xg_for"))
    a_xg = _f(feat.get("away_xg_avg")) or _f(us_a.get("xg_for"))
    tot_xg = None
    if h_xg is not None and a_xg is not None:
        tot_xg = h_xg + a_xg
        # scostamento da 2.5
        ou_bias = _clip((tot_xg - 2.55) / 1.8, -0.35, 0.35)
        e_ou = _clip(0.55 * e_ou + 0.45 * ou_bias, -0.42, 0.42)

    p1, px, p2 = _two_way_to_1x2(_clip(0.5 + e1, 0.18, 0.82), p_draw)
    lean_1x2 = _lean_1x2(p1, px, p2)
    lean_ou = "O2.5" if e_ou > 0.04 else ("U2.5" if e_ou < -0.04 else "push")
    lean_ah = "1" if e_ah > 0.05 else ("2" if e_ah < -0.05 else "push")

    return {
        "1x2": {
            "edge": round(e1, 4),
            "lean": lean_1x2,
            "p_1": round(p1, 4),
            "p_x": round(px, 4),
            "p_2": round(p2, 4),
        },
        "ou": {
            "edge": round(e_ou, 4),
            "lean": lean_ou,
            "tot_xg": None if tot_xg is None else round(tot_xg, 3),
        },
        "ah": {
            "edge": round(e_ah, 4),
            "lean": lean_ah,
        },
    }


def data_signal_validation(prediction: dict[str, Any], play: dict[str, Any] | None = None) -> dict[str, Any]:
    """Micro-aggiustamento voto (±0.5) se il segnale è abbastanza forte."""
    sig = prediction.get("data_signal") or {}
    if not sig.get("ready"):
        try:
            sig = build_data_signal(prediction)
        except Exception:
            sig = {"ready": False}
    if not sig.get("ready"):
        return {
            "ready": False,
            "status": "n/d",
            "delta_unified": 0.0,
            "notes": [sig.get("note") or "analisi dati non disponibile"],
        }

    conf = float(sig.get("confidence") or 0)
    lean = sig.get("lean")
    play = play or {}
    play_code = str(play.get("code") or "")
    notes = [
        f"{sig.get('n_factors', 0)} fattori · conf {conf:.0%}",
        sig.get("note") or "",
    ]
    delta = 0.0
    status = "ok"
    if conf >= 0.55 and lean in {"1", "X", "2"}:
        if play_code in {"1", "X", "2"}:
            if play_code == lean:
                delta = 0.5 if conf >= 0.70 else 0.25
                status = "allinea"
                notes.append(f"allinea pick {lean}")
            else:
                delta = -0.5 if conf >= 0.70 else -0.25
                status = "contrario"
                notes.append(f"contrario al pick (dati→{lean}, pick {play_code})")
        elif abs(float(sig.get("edge") or 0)) >= 0.12:
            status = "segnale"
            notes.append(f"lean dati {lean}")
    elif conf < 0.40:
        status = "debole"
        notes.append("segnale debole / pochi dati")

    return {
        "ready": True,
        "status": status,
        "lean": lean,
        "confidence": conf,
        "edge": sig.get("edge"),
        "delta_unified": float(delta),
        "notes": [n for n in notes if n],
        "n_factors": sig.get("n_factors"),
    }
