"""Score pro: pesi fonti, Unified/Confidence/Risk 0–100, priorità, override, market recommender.

Non tocca EV/Kelly/p_cons: solo voto, quadro, no_bet soft e ordinamento calendario.
"""

from __future__ import annotations

from typing import Any

# Pesi relativi sul voto delle fonti (somma ~1.0). Tipster/meteo bassi di proposito.
SOURCE_WEIGHTS: dict[str, dict[str, Any]] = {
    "Monte Carlo": {"weight": 0.35, "why": "Stabilità, bassa varianza"},
    "Modello ML": {"weight": 0.25, "why": "Reattivo, cattura pattern"},
    "Book (devig)": {"weight": 0.20, "why": "True price"},
    "Understat": {"weight": 0.10, "why": "Forma reale xG"},
    "Analisi dati": {"weight": 0.10, "why": "xG/forma/classifica aggregati"},
    "FotMob xG": {"weight": 0.08, "why": "Rolling xG live"},
    "λ Poisson": {"weight": 0.08, "why": "Baseline gol"},
    "ClubElo": {"weight": 0.07, "why": "Forza relativa esterna"},
    "FBref": {"weight": 0.06, "why": "Team stats avanzate"},
    "FotMob": {"weight": 0.05, "why": "Classifica / PPG"},
    "Sofascore": {"weight": 0.05, "why": "Classifica attuale"},
    "Sportly-sim": {"weight": 0.05, "why": "Sim tattica sintetica"},
    "Storico locale": {"weight": 0.05, "why": "Esiti nostri settled"},
    "Combo tattica": {"weight": 0.05, "why": "Situazionale (stile/assenze)"},
    "Assenze / XI": {"weight": 0.08, "why": "Titoli out pesati xG+xA"},
    "Matchup tattico": {"weight": 0.05, "why": "Stile FBref situazionale"},
    "StatsBomb": {"weight": 0.04, "why": "Eventi open data"},
    "Fatica / calendario": {"weight": 0.04, "why": "Riposo e congestione"},
    "Steam Asian": {"weight": 0.04, "why": "Flusso quote"},
    "Tipster": {"weight": 0.02, "why": "Rumore pubblico"},
    "Meteo": {"weight": 0.03, "why": "Solo se estremo"},
    "Validazione": {"weight": 0.0, "why": "Annotazione, non voto"},
    "FotMob details": {"weight": 0.0, "why": "Dettaglio live, non voto 1X2"},
    "Fallback": {"weight": 0.12, "why": "Sintesi Elo+forma+book se fonti forti assenti"},
}

# Mercati O/U: fonti più utili sul totale
SOURCE_WEIGHTS_OU: dict[str, float] = {
    "Understat": 0.18,
    "Analisi dati": 0.16,
    "FBref": 0.14,
    "FotMob": 0.12,
    "FotMob xG": 0.14,
    "Sportly-sim": 0.12,
    "λ Poisson": 0.14,
    "Monte Carlo": 0.20,
    "Modello ML": 0.10,
    "Book (devig)": 0.12,
}

BANDS = (
    (0, 30, "no_bet", "No bet"),
    (30, 60, "lean", "Lean"),
    (60, 75, "playable", "Playable"),
    (75, 90, "strong", "Strong"),
    (90, 101, "premium", "Premium"),
)

CORE_SOURCES = ("Monte Carlo", "Modello ML", "Book (devig)")


def source_weight(name: str, *, ou: bool = False) -> float:
    if ou:
        if name in SOURCE_WEIGHTS_OU:
            return float(SOURCE_WEIGHTS_OU[name])
    info = SOURCE_WEIGHTS.get(name)
    if info:
        return float(info["weight"])
    return 0.04


def source_why(name: str) -> str:
    info = SOURCE_WEIGHTS.get(name) or {}
    return str(info.get("why") or "")


def annotate_source_weights(sources: list[dict[str, Any]], *, ou: bool = False) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in sources:
        row = dict(s)
        name = str(row.get("fonte") or "")
        row["peso"] = round(source_weight(name, ou=ou), 3)
        row["peso_note"] = source_why(name)
        out.append(row)
    return out


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def _safe(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _band(score: float) -> dict[str, Any]:
    s = _clip(score, 0, 100)
    for lo, hi, key, label in BANDS:
        if lo <= s < hi:
            return {"key": key, "label": label, "lo": lo, "hi": hi}
    return {"key": "premium", "label": "Premium", "lo": 90, "hi": 100}


def _lean_1x2(p1: float | None, px: float | None, p2: float | None) -> str | None:
    vals = [("1", p1), ("X", px), ("2", p2)]
    ranked = [(k, v) for k, v in vals if v is not None]
    if not ranked:
        return None
    ranked.sort(key=lambda t: t[1], reverse=True)
    return ranked[0][0]


def coverage_stats(quadro: dict[str, Any] | None, prediction: dict[str, Any] | None) -> dict[str, Any]:
    """Completezza dati + fallback disponibili."""
    sources = (quadro or {}).get("sources") or []
    present = [s for s in sources if not s.get("mancante")]
    missing = [s for s in sources if s.get("mancante")]
    core_ok = sum(1 for n in CORE_SOURCES if any(s.get("fonte") == n and not s.get("mancante") for s in sources))
    n = len(sources) or 1
    coverage = len(present) / n
    pred = prediction or {}
    has_elo = bool((pred.get("features") or {}).get("home_elo") or (pred.get("clubelo") or {}).get("home"))
    has_form = bool((pred.get("features") or {}).get("home_form_pts") is not None)
    has_book = any(s.get("fonte") == "Book (devig)" and not s.get("mancante") for s in sources)
    has_xg = any(
        s.get("fonte") in {"Understat", "FotMob xG", "Analisi dati"} and not s.get("mancante") for s in sources
    )
    fallback_ok = (has_elo or has_form) and (has_book or core_ok >= 1)
    return {
        "coverage": round(coverage, 3),
        "n_present": len(present),
        "n_missing": len(missing),
        "core_ok": core_ok,
        "has_xg": has_xg,
        "fallback_ok": fallback_ok,
        "missing_names": [str(s.get("fonte")) for s in missing[:8]],
    }


def build_fallback_source(prediction: dict[str, Any], sources: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Se mancano ≥2 fonti core/xG, sintetizza un lean da Elo+forma+book."""
    core_missing = sum(
        1 for n in CORE_SOURCES if not any(s.get("fonte") == n and not s.get("mancante") for s in sources)
    )
    xg_ok = any(
        s.get("fonte") in {"Understat", "FotMob xG", "Analisi dati"} and not s.get("mancante") for s in sources
    )
    if core_missing < 2 and xg_ok:
        return None

    feat = prediction.get("features") or {}
    ml = prediction.get("model_probabilities") or {}
    mc = prediction.get("montecarlo") or {}
    p_draw = _safe(mc.get("draw") or ml.get("draw"), 0.26)

    book = next((s for s in sources if s.get("fonte") == "Book (devig)" and not s.get("mancante")), None)
    elo_h = _safe(feat.get("home_elo"), 1500.0)
    elo_a = _safe(feat.get("away_elo"), 1500.0)
    form_h = _safe(feat.get("home_form_pts"), 1.2)
    form_a = _safe(feat.get("away_form_pts"), 1.2)

    # 2-way home strength
    elo_p = 1.0 / (1.0 + 10 ** (-((elo_h + 65.0 - elo_a) / 400.0)))
    form_p = _clip(0.5 + (form_h - form_a) / 8.0, 0.15, 0.85)
    if book and book.get("p_1") is not None:
        book_p = _safe(book.get("p_1"), 0.4) / max(
            1e-6, _safe(book.get("p_1"), 0.4) + _safe(book.get("p_2"), 0.4)
        )
    else:
        book_p = 0.5

    p_h2 = _clip(0.45 * elo_p + 0.30 * form_p + 0.25 * book_p, 0.12, 0.88)
    d = _clip(p_draw, 0.12, 0.40)
    p1 = max(0.04, p_h2 * (1.0 - d))
    p2 = max(0.04, (1.0 - p_h2) * (1.0 - d))
    tot = p1 + d + p2
    p1, px, p2 = p1 / tot, d / tot, p2 / tot
    note_bits = ["sintesi automatica"]
    if core_missing:
        note_bits.append(f"core mancanti {core_missing}/3")
    if not xg_ok:
        note_bits.append("xG assente")
    note_bits.append(f"Elo {elo_h:.0f}/{elo_a:.0f}")
    note_bits.append(f"forma {form_h:.1f}/{form_a:.1f}")
    return {
        "fonte": "Fallback",
        "idea": "Elo + forma + book (leghe minori / dati incompleti)",
        "pick": _lean_1x2(p1, px, p2) or "n/d",
        "p_1": round(p1, 4),
        "p_x": round(px, 4),
        "p_2": round(p2, 4),
        "nota": " · ".join(note_bits),
        "mancante": False,
        "peso": source_weight("Fallback"),
        "peso_note": source_why("Fallback"),
        "fallback": True,
    }


def situational_overrides(
    prediction: dict[str, Any],
    play: dict[str, Any],
    validation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assenze / meteo estremo / derby-like → Δ voto e flag (non EV)."""
    notes: list[str] = []
    delta = 0.0
    risk_bump = 0.0
    lean_bias: str | None = None  # "home" | "away" | "under" | "draw"

    wx = prediction.get("weather") or {}
    flag = str(wx.get("flag") or "")
    precip = _safe(wx.get("precip_mm"), 0)
    wind = _safe(wx.get("wind_kmh"), 0)
    adj = _safe(wx.get("lambda_adj"), 1.0)
    extreme = (
        flag in {"avverso", "pioggia_forte", "neve", "vento_forte"}
        or precip >= 8
        or wind >= 40
        or abs(adj - 1.0) >= 0.04
    )
    if extreme:
        delta -= 0.4
        risk_bump += 0.12
        lean_bias = "under"
        notes.append(f"meteo estremo ({flag or 'precip/vento'}) -> Under lean, -voto")

    injuries = {}
    try:
        t = prediction.get("tactics") or {}
        injuries = t.get("injuries") or t.get("absences") or {}
        if not injuries.get("ready"):
            for key in ("injury_weights", "injuries", "absences"):
                block = (validation or {}).get(key) or {}
                if isinstance(block, dict) and block.get("ready"):
                    injuries = block
                    break
    except Exception:
        injuries = {}
    wh = _safe(injuries.get("weight_home"), 0)
    wa = _safe(injuries.get("weight_away"), 0)
    if injuries.get("ready") and (wh >= 0.12 or wa >= 0.12):
        gap = wa - wh
        if abs(gap) >= 0.08:
            delta += 0.35 if gap > 0 else -0.35
            lean_bias = "home" if gap > 0 else "away"
            notes.append(
                f"assenze pesanti (casa {wh:.0%} / ospite {wa:.0%}) -> bias {lean_bias}, delta voto"
            )
            risk_bump += 0.08
        elif wh >= 0.18 and wa >= 0.18:
            delta -= 0.25
            risk_bump += 0.10
            notes.append("entrambe difese/titolari out -> rischio su")

    # Derby / alta intensità: stesso paese + keyword o gap Elo basso
    home = str(prediction.get("home") or "")
    away = str(prediction.get("away") or "")
    league = str(prediction.get("league") or "").lower()
    derby_kw = any(k in league for k in ("derby", "clásico", "clasico", "rival"))
    feat = prediction.get("features") or {}
    elo_gap = abs(_safe(feat.get("home_elo"), 1500) - _safe(feat.get("away_elo"), 1500))
    if derby_kw or (elo_gap < 40 and "cup" not in league and "champions" not in league):
        # partita tesa / bilanciata → più rischio, piccolo boost draw/under
        if derby_kw:
            delta -= 0.15
            risk_bump += 0.10
            notes.append("derby / rivalita -> Risk su, voto -")
            if lean_bias is None:
                lean_bias = "draw"

    venue = (validation or {}).get("venue") or {}
    if venue.get("flag") == "campo_neutro":
        delta -= 0.2
        notes.append("campo neutro -> -voto casa")
        if lean_bias is None:
            lean_bias = "away"

    code = str(play.get("code") or "")
    # applica bias soft sul voto se il pick combacia / contrastare
    if lean_bias == "under" and play.get("group") == "ou" and str(code).upper().startswith("U"):
        delta += 0.35
        notes.append("override Under allineato al meteo")
    elif lean_bias == "under" and play.get("group") == "ou" and str(code).upper().startswith("O"):
        delta -= 0.45
        notes.append("override: Over contro meteo estremo")
    elif lean_bias == "home" and code in {"1", "1X", "1 DNB"}:
        delta += 0.25
    elif lean_bias == "away" and code in {"2", "X2", "2 DNB"}:
        delta += 0.25
    elif lean_bias == "draw" and code == "X":
        delta += 0.2

    return {
        "ready": bool(notes),
        "delta_unified": round(delta, 3),
        "risk_bump": round(_clip(risk_bump), 3),
        "lean_bias": lean_bias,
        "notes": notes[:6],
    }


def confidence_index(
    *,
    coverage: dict[str, Any],
    agreement: dict[str, Any] | None,
    validation: dict[str, Any] | None,
    intervals: dict[str, Any] | None,
    league: str | None,
) -> dict[str, Any]:
    agree = agreement or {}
    agree_w = _safe(agree.get("agree_w") or agree.get("agree_share"), 0.5)
    cov = _safe(coverage.get("coverage"), 0.4)
    core = _safe(coverage.get("core_ok"), 0) / 3.0
    fragile = bool((intervals or {}).get("fragile"))
    stable = bool((intervals or {}).get("stable"))
    ic = 0.35 if fragile else (0.85 if stable else 0.60)
    try:
        from modules.model_training.league_clusters import cluster_for

        cid = cluster_for(league)
    except Exception:
        cid = "global"
    league_q = {
        "big5_eng": 0.92,
        "big5_esp": 0.90,
        "big5_ita": 0.90,
        "big5_ger": 0.88,
        "big5_fra": 0.88,
        "serie_b_like": 0.62,
        "latam": 0.58,
        "mls": 0.65,
        "cups_euro": 0.70,
        "global": 0.55,
    }.get(cid, 0.55)

    # coerenza modello–mercato da validation
    mkt = (validation or {}).get("market") or {}
    gap = abs(_safe(mkt.get("gap"), 0.08))
    model_mkt = _clip(1.0 - gap / 0.20)

    raw = (
        0.28 * cov
        + 0.22 * core
        + 0.22 * agree_w
        + 0.14 * ic
        + 0.08 * league_q
        + 0.06 * model_mkt
    )
    if coverage.get("fallback_ok") and core < 0.67:
        raw *= 0.92  # fallback ok ma confidence un filo sotto
    score = round(100 * _clip(raw), 1)
    return {
        "score": score,
        "band": _band(score),
        "parts": {
            "coverage": round(cov, 3),
            "core": round(core, 3),
            "agree": round(agree_w, 3),
            "interval": round(ic, 3),
            "league": round(league_q, 3),
            "model_market": round(model_mkt, 3),
        },
        "cluster": cid,
    }


def risk_index(
    *,
    prediction: dict[str, Any],
    intervals: dict[str, Any] | None,
    overrides: dict[str, Any] | None,
    agreement: dict[str, Any] | None,
) -> dict[str, Any]:
    feat = prediction.get("features") or {}
    mc = prediction.get("montecarlo") or {}
    # varianza: draw alto + gol attesi estremi + IC largo
    p_draw = _safe(mc.get("draw"), 0.26)
    xg_h = _safe((prediction.get("expected_goals") or {}).get("home"), 1.2)
    xg_a = _safe((prediction.get("expected_goals") or {}).get("away"), 1.1)
    total_xg = xg_h + xg_a
    width = _safe((intervals or {}).get("top_width"), 0.10)
    agree_w = _safe((agreement or {}).get("agree_w"), 0.55)

    rest_h = _safe(feat.get("home_rest_days"), 6)
    rest_a = _safe(feat.get("away_rest_days"), 6)
    fatigue = 1.0 if min(rest_h, rest_a) <= 3 else 0.35 if min(rest_h, rest_a) <= 4 else 0.1

    # bilanciamento = alta imprevedibilità
    balance = 1.0 - abs(_safe(mc.get("home_win"), 0.4) - _safe(mc.get("away_win"), 0.3))
    high_intensity = 1.0 if total_xg >= 3.2 or total_xg <= 1.6 else 0.4

    raw = (
        0.22 * _clip(width / 0.22)
        + 0.18 * _clip((0.32 - abs(p_draw - 0.28)) / 0.12)  # draw "strano"
        + 0.18 * _clip(1.0 - agree_w)
        + 0.15 * _clip(balance)
        + 0.12 * fatigue
        + 0.10 * high_intensity
        + 0.05 * _safe((overrides or {}).get("risk_bump"), 0)
    )
    score = round(100 * _clip(raw), 1)
    return {
        "score": score,
        "band": _band(score),
        "parts": {
            "interval_width": round(width, 3),
            "disagree": round(1.0 - agree_w, 3),
            "fatigue": round(fatigue, 3),
            "balance": round(_clip(balance), 3),
        },
    }


def unified_score_100(
    *,
    play: dict[str, Any],
    meta: dict[str, Any] | None,
    agreement: dict[str, Any] | None,
    confidence: dict[str, Any],
    risk: dict[str, Any],
    residual: dict[str, Any] | None,
    data_signal: dict[str, Any] | None,
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Unifica modelli/mercato/accordo/confidence in 0–100 con bande."""
    meta = meta or {}
    # voto 1–10 esistente → base
    base10 = _safe(play.get("score_unified") or meta.get("score") or play.get("score"), 3)
    base = (base10 - 1) / 9.0

    agree_w = _safe((agreement or {}).get("agree_w"), 0.5)
    conf = _safe(confidence.get("score"), 50) / 100.0
    risk_n = _safe(risk.get("score"), 40) / 100.0
    ev = play.get("ev_cons")
    if ev is None:
        ev = play.get("ev")
    ev_n = _clip((_safe(ev, -0.05) + 0.05) / 0.18) if play.get("odds_real") else 0.25
    dsig = _clip(0.5 + _safe((data_signal or {}).get("edge"), 0) * 2.0)
    dsig_c = _safe((data_signal or {}).get("confidence"), 0.4)
    res_n = 0.5
    if (residual or {}).get("ready") and residual.get("residual") is not None:
        res_n = _clip(0.5 + _safe(residual.get("residual"), 0) * 3.0)

    raw = (
        0.32 * base
        + 0.18 * agree_w
        + 0.16 * conf
        + 0.14 * ev_n
        + 0.08 * (0.5 * dsig + 0.5 * dsig_c)
        + 0.06 * res_n
        + 0.06 * (1.0 - risk_n)
    )
    score = 100 * _clip(raw)

    # override Δ (scala 10 → ≈11 punti su 100)
    score += _safe((overrides or {}).get("delta_unified"), 0) * 11.0

    action = play.get("action")
    if action == "no_bet":
        score = min(score, 58)
    elif action in {"n/d", "invalido"}:
        score = min(score, 28)
    if (agreement or {}).get("block_no_bet"):
        score = min(score, 35)

    score = round(_clip(score, 0, 100), 1)
    band = _band(score)
    return {
        "score": score,
        "band": band,
        "parts": {
            "meta10": round(base10, 2),
            "agree": round(agree_w, 3),
            "confidence": round(conf, 3),
            "ev": round(ev_n, 3),
            "risk_inv": round(1.0 - risk_n, 3),
        },
    }


def _hours_to_kickoff(prediction: dict[str, Any], play: dict[str, Any] | None = None) -> float | None:
    """Ore al calcio d'inizio (None se data sconosciuta)."""
    raw = (
        (prediction or {}).get("date")
        or (prediction or {}).get("kickoff")
        or (play or {}).get("date")
        or (play or {}).get("kickoff")
    )
    if raw is None:
        return None
    try:
        import pandas as pd

        ko = pd.to_datetime(raw, errors="coerce")
        if ko is None or (hasattr(ko, "isna") and bool(ko.isna())):
            return None
        # se solo data senza ora, assume 15:00 locale
        if getattr(ko, "hour", 0) == 0 and getattr(ko, "minute", 0) == 0:
            ko = ko + pd.Timedelta(hours=15)
        now = pd.Timestamp.now(tz=ko.tz) if getattr(ko, "tz", None) is not None else pd.Timestamp.now()
        if getattr(ko, "tz", None) is not None and getattr(now, "tz", None) is None:
            now = now.tz_localize(ko.tz)
        hours = (ko - now).total_seconds() / 3600.0
        return float(hours)
    except Exception:
        return None


def _pick_odds_drop(play: dict[str, Any], market_move: dict[str, Any] | None) -> float | None:
    """Drop apertura→attuale sulla selezione (pp). Positivo = quota in discesa (steam a favore)."""
    move = market_move or {}
    code = str(play.get("code") or "").strip().upper()
    group = str(play.get("group") or "1x2").lower()
    key = None
    if group in {"1x2", "dc", "dnb"} or code in {"1", "X", "2"}:
        key = {"1": "drop_1", "X": "drop_x", "2": "drop_2"}.get(code)
    elif group == "ou":
        if code.startswith("O") or "OVER" in code:
            key = "drop_over"
        elif code.startswith("U") or "UNDER" in code:
            key = "drop_under"
    elif group == "ah":
        if code in {"1", "AH1", "1 AH"} or "HOME" in code:
            key = "drop_ah_home"
        elif code in {"2", "AH2", "2 AH"} or "AWAY" in code:
            key = "drop_ah_away"
    if not key:
        return None
    v = move.get(key)
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _s_quota(play: dict[str, Any], market_move: dict[str, Any] | None) -> tuple[float, str]:
    """Stabilità/direzione quota 0.8–1.2. Steam a favore → alto; contro → basso."""
    drop = _pick_odds_drop(play, market_move)
    lvl = str((market_move or {}).get("movement_level") or "")
    if drop is None and not market_move:
        return 0.95, "quota n/d"
    if drop is None:
        # movimento generico senza drop sul pick
        if lvl in {"Forte", "Fortissimo", "Raro"}:
            return 1.05, f"move {lvl} (no drop pick)"
        return 1.0, "stabile"
    # drop tipicamente in punti percentuali di probabilità / o frazioni di quota
    # convention asian: drop > 0 = quota scesa sul lato
    if drop >= 4.0:
        return 1.20, f"steam forte drop {drop:.1f}"
    if drop >= 2.0:
        return 1.12, f"quota in discesa {drop:.1f}"
    if drop >= 0.8:
        return 1.06, f"leggero steam {drop:.1f}"
    if drop <= -4.0:
        return 0.80, f"quota contro {drop:.1f}"
    if drop <= -2.0:
        return 0.88, f"drift contrario {drop:.1f}"
    if drop <= -0.8:
        return 0.94, f"leggero drift {drop:.1f}"
    return 1.0, "stabile"


def _c_models(prediction: dict[str, Any], play: dict[str, Any], confidence: dict[str, Any] | None) -> tuple[float, str]:
    """Confidenza modelli 0.7–1.3: ML↔MC allineati + confidence index."""
    ml = prediction.get("model_probabilities") or {}
    mc = prediction.get("montecarlo") or {}
    code = str(play.get("code") or "")
    key = {"1": "home_win", "X": "draw", "2": "away_win"}.get(code)
    align = 1.0
    note = "n/d ML/MC"
    if key and ml.get(key) is not None and mc.get(key) is not None:
        gap = abs(_safe(ml.get(key)) - _safe(mc.get(key)))
        if gap <= 0.03:
            align = 1.25
            note = f"ML=MC (gap {gap:.0%})"
        elif gap <= 0.06:
            align = 1.12
            note = f"ML~MC (gap {gap:.0%})"
        elif gap <= 0.10:
            align = 0.95
            note = f"ML/MC misto (gap {gap:.0%})"
        else:
            align = 0.75
            note = f"ML≠MC (gap {gap:.0%})"
    elif ml.get("home_win") is not None and mc.get("home_win") is not None:
        gap = max(
            abs(_safe(ml.get("home_win")) - _safe(mc.get("home_win"))),
            abs(_safe(ml.get("draw")) - _safe(mc.get("draw"))),
            abs(_safe(ml.get("away_win")) - _safe(mc.get("away_win"))),
        )
        align = 1.15 if gap <= 0.05 else (0.85 if gap >= 0.12 else 1.0)
        note = f"1X2 gap max {gap:.0%}"

    conf_n = _safe((confidence or {}).get("score"), 55) / 100.0
    # mescola allineamento con confidence: centro 1.0
    c = _clip(0.55 * align + 0.45 * (0.7 + 0.6 * conf_n), 0.70, 1.30)
    return round(c, 3), note


def _t_match(hours: float | None) -> tuple[float, str]:
    """Tempo alla partita 0.5–1.5: imminente → urgenza alta."""
    if hours is None:
        return 1.0, "kickoff n/d"
    if hours < 0:
        return 0.55, "già iniziata/passata"
    if hours <= 4:
        return 1.50, f"tra {hours:.0f}h"
    if hours <= 12:
        return 1.35, f"tra {hours:.0f}h"
    if hours <= 24:
        return 1.20, f"entro 24h"
    if hours <= 48:
        return 1.05, f"tra {hours/24:.1f}g"
    if hours <= 72:
        return 0.90, f"tra {hours/24:.1f}g"
    if hours <= 120:
        return 0.70, f"tra {hours/24:.1f}g"
    return 0.50, f"lontana ({hours/24:.0f}g)"


def _l_market(league: str | None, play: dict[str, Any]) -> tuple[float, str]:
    """Liquidità lega/mercato 0.75–1.15 (Big 5 alto, minori basso)."""
    try:
        from modules.model_training.league_clusters import cluster_for

        cid = cluster_for(league)
    except Exception:
        cid = "global"
    base = {
        "big5_eng": 1.15,
        "big5_esp": 1.12,
        "big5_ita": 1.12,
        "big5_ger": 1.10,
        "big5_fra": 1.08,
        "cups_euro": 1.05,
        "serie_b_like": 0.92,
        "mls": 0.90,
        "latam": 0.85,
        "global": 0.82,
    }.get(cid, 0.82)
    # mercati esotici un filo meno urgenti a parità di EV
    g = str(play.get("group") or "1x2").lower()
    if g in {"cards", "corners", "scorer", "exact", "parity"}:
        base *= 0.90
        return round(_clip(base, 0.75, 1.15), 3), f"{cid} · mercato {g}"
    if g in {"1x2", "ah", "ou", "dc", "btts"}:
        return round(_clip(base, 0.75, 1.15), 3), cid
    return round(_clip(base, 0.75, 1.15), 3), cid


def priority_score(
    play: dict[str, Any],
    *,
    prediction: dict[str, Any] | None = None,
    market_move: dict[str, Any] | None = None,
    confidence: dict[str, Any] | None = None,
    league: str | None = None,
) -> dict[str, Any]:
    """Ranking di urgenza: quale giocata buona guardare per prima.

    PRIORITÀ ∝ EV_eff × S_quota × C_modelli × T_match × L_mercato
    Non è una copia del voto unificato.
    """
    pred = prediction or {}
    action = play.get("action")

    ev = play.get("ev_cons")
    if ev is None:
        ev = play.get("ev")
    odds = play.get("odds")
    fair = play.get("fair_odds")
    # distanza da fair: quota reale sopra equa → più urgenza a prendere
    fair_gap = 0.0
    if odds and fair and float(fair) > 1.01:
        fair_gap = max(0.0, float(odds) / float(fair) - 1.0)

    ev_f = _safe(ev, 0.0) if play.get("odds_real") else 0.0
    # EV relativo: EV negativo non genera urgenza; fair_gap rafforza
    ev_eff = max(0.0, ev_f) * (1.0 + 1.5 * min(0.25, fair_gap))
    # scala base: EV 8% ≈ 1.0 sul moltiplicatore
    ev_factor = _clip(ev_eff / 0.08, 0.0, 2.5)

    s_q, s_note = _s_quota(play, market_move)
    c_m, c_note = _c_models(pred, play, confidence)
    hours = _hours_to_kickoff(pred, play)
    t_m, t_note = _t_match(hours)
    l_m, l_note = _l_market(league or pred.get("league"), play)

    product = ev_factor * s_q * c_m * t_m * l_m

    # mappa prodotto → 0–100 (EV 8% × fattori ~1 → ~55; pick forte → 80+)
    score = 100.0 * _clip(product / 2.2, 0.0, 1.0)

    if action == "gioca":
        score = min(100.0, score + 4.0)
        gate = "gioca"
    elif action == "no_bet":
        score *= 0.45  # resta visibile ma sotto le gioca
        gate = "no_bet"
    elif action in {"n/d", "invalido"}:
        score *= 0.20
        gate = action or "n/d"
    else:
        gate = str(action or "—")

    # senza EV reale non c'è ranking di urgenza value
    if not play.get("odds_real") or ev is None:
        score = min(score, 25.0)
        gate = f"{gate} · senza EV"

    score = round(_clip(score, 0, 100), 1)
    label = "Alta" if score >= 70 else "Media" if score >= 40 else "Bassa"
    return {
        "score": score,
        "rank_hint": label,
        "formula": "EV_eff × S_quota × C_modelli × T_match × L_mercato",
        "parts": {
            "ev": None if ev is None else round(float(ev), 4),
            "ev_factor": round(ev_factor, 3),
            "fair_gap": round(fair_gap, 4),
            "s_quota": s_q,
            "c_models": c_m,
            "t_match": t_m,
            "l_market": l_m,
            "hours_to_ko": None if hours is None else round(hours, 1),
            "product": round(product, 3),
        },
        "notes": [s_note, c_note, t_note, l_note, gate],
    }


def bet_type_recommender(
    *,
    play: dict[str, Any],
    grouped: dict[str, list] | None,
    prediction: dict[str, Any],
    overrides: dict[str, Any] | None,
    agreement: dict[str, Any] | None,
) -> dict[str, Any]:
    """Sceglie il tipo di mercato più adatto ai segnali (non ricalcola quote)."""
    grouped = grouped or {}
    candidates: list[dict[str, Any]] = []

    def add(group: str, label: str, m: dict[str, Any], bonus: float = 0.0) -> None:
        if not m or not m.get("odds_real"):
            return
        ev = m.get("ev_cons")
        if ev is None:
            ev = m.get("ev")
        edge = m.get("edge_pp")
        sc = _safe(m.get("score"), 1)
        score = sc + 3.0 * _safe(ev, -0.05) + 2.0 * _safe(edge, -0.02) + bonus
        candidates.append(
            {
                "group": group,
                "label": label,
                "code": m.get("code"),
                "name": m.get("name"),
                "score": round(score, 3),
                "ev": None if ev is None else round(float(ev), 4),
                "edge_pp": None if edge is None else round(float(edge), 4),
            }
        )

    # bonus da override / contesto
    bias = (overrides or {}).get("lean_bias")
    mc = prediction.get("montecarlo") or {}
    p_over = _safe(mc.get("over_2.5"), 0.5)
    p_btts = _safe(mc.get("btts"), 0.5)
    p1 = _safe(mc.get("home_win"), 0.4)
    p2 = _safe(mc.get("away_win"), 0.3)
    px = _safe(mc.get("draw"), 0.26)
    gap = abs(p1 - p2)

    for m in grouped.get("1x2") or []:
        add("1x2", "1X2", m, bonus=0.15 if gap >= 0.18 else 0.0)
    for m in grouped.get("dc") or []:
        add("dc", "Doppia chance", m, bonus=0.25 if gap < 0.12 else 0.05)
    for m in grouped.get("ah") or []:
        add("ah", "Asian Handicap", m, bonus=0.35 if gap >= 0.20 else 0.1)
    for m in grouped.get("ou") or []:
        b = 0.0
        code = str(m.get("code") or "").upper()
        if bias == "under" and code.startswith("U"):
            b = 0.55
        elif p_over >= 0.58 and code.startswith("O"):
            b = 0.35
        elif p_over <= 0.42 and code.startswith("U"):
            b = 0.35
        add("ou", "Over / Under", m, bonus=b)
    for m in grouped.get("btts") or []:
        add("btts", "Gol / No gol", m, bonus=0.3 if abs(p_btts - 0.5) >= 0.08 else 0.05)
    for m in grouped.get("multigol") or []:
        add("multigol", "Multigol", m, bonus=0.15)
    for m in grouped.get("dnb") or []:
        add("dnb", "DNB", m, bonus=0.2 if gap >= 0.14 else 0.0)

    if not candidates:
        # fallback sul play corrente
        return {
            "ready": False,
            "primary": {
                "group": play.get("group"),
                "label": play.get("group"),
                "code": play.get("code"),
                "name": play.get("name"),
            },
            "alternatives": [],
            "note": "pocchi mercati con quota reale",
        }

    candidates.sort(key=lambda x: x["score"], reverse=True)
    primary = candidates[0]
    alts = candidates[1:4]
    note_bits = [f"top {primary['label']} {primary.get('code')}"]
    if bias:
        note_bits.append(f"bias {bias}")
    if (agreement or {}).get("status"):
        note_bits.append(f"accordo {(agreement or {}).get('status')}")
    return {
        "ready": True,
        "primary": primary,
        "alternatives": alts,
        "note": " · ".join(note_bits),
    }


def build_match_scores(
    *,
    play: dict[str, Any],
    prediction: dict[str, Any],
    quadro: dict[str, Any] | None,
    agreement: dict[str, Any] | None,
    validation: dict[str, Any] | None,
    intervals: dict[str, Any] | None,
    residual: dict[str, Any] | None,
    meta: dict[str, Any] | None,
    grouped: dict[str, list] | None = None,
    league: str | None = None,
    market_move: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calcola tutti gli indici pro e arricchisce il quadro con pesi + fallback."""
    q = dict(quadro or {})
    sources = list(q.get("sources") or [])
    ou = str(play.get("group") or "").lower() in {"ou", "btts", "goal"}
    sources = annotate_source_weights(sources, ou=ou)

    fb = build_fallback_source(prediction, sources)
    if fb and not any(s.get("fonte") == "Fallback" for s in sources):
        sources.append(fb)
        q["fallback"] = True
    else:
        q["fallback"] = False

    q["sources"] = sources
    cov = coverage_stats(q, prediction)
    overrides = situational_overrides(prediction, play, validation)
    conf = confidence_index(
        coverage=cov,
        agreement=agreement,
        validation=validation,
        intervals=intervals,
        league=league or prediction.get("league"),
    )
    risk = risk_index(
        prediction=prediction,
        intervals=intervals,
        overrides=overrides,
        agreement=agreement,
    )
    unified = unified_score_100(
        play=play,
        meta=meta,
        agreement=agreement,
        confidence=conf,
        risk=risk,
        residual=residual,
        data_signal=prediction.get("data_signal"),
        overrides=overrides,
    )
    priority = priority_score(
        play,
        prediction=prediction,
        market_move=market_move,
        confidence=conf,
        league=league or prediction.get("league"),
    )
    bets = bet_type_recommender(
        play=play,
        grouped=grouped,
        prediction=prediction,
        overrides=overrides,
        agreement=agreement,
    )

    # aggiorna consenso quadro con share pesata se agreement pronto
    if agreement and agreement.get("agree_w") is not None and play.get("code") in {"1", "X", "2"}:
        aw = float(agreement["agree_w"])
        if aw >= 0.72:
            q["consenso"] = "ampio accordo (pesato)"
        elif aw >= 0.55:
            q["consenso"] = "maggioranza (pesata)"
        elif aw > 0:
            q["consenso"] = "quadro spezzato (pesato)"
        q["agree_share_w"] = agreement.get("agree_w")

    weights_table = [
        {"fonte": k, "peso": v["weight"], "motivazione": v["why"]}
        for k, v in SOURCE_WEIGHTS.items()
        if float(v["weight"]) > 0
    ]
    weights_table.sort(key=lambda r: -r["peso"])

    return {
        "unified": unified,
        "confidence": conf,
        "risk": risk,
        "priority": priority,
        "overrides": overrides,
        "coverage": cov,
        "bet_rec": bets,
        "quadro": q,
        "weights_table": weights_table,
        "band": unified["band"],
        "score_100": unified["score"],
        "confidence_100": conf["score"],
        "risk_100": risk["score"],
        "priority_100": priority["score"],
    }
