"""Quadro analisi: fonti ortogonali, senza mischiarle nell'EV."""

from __future__ import annotations

from typing import Any


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


def _elo_home_win(elo_h: float, elo_a: float, hfa: float = 65.0) -> float:
    return 1.0 / (1.0 + 10 ** (-((elo_h + hfa - elo_a) / 400.0)))


def _clamp01(x: float) -> float:
    return max(0.01, min(0.99, x))


def _src(
    name: str,
    idea: str,
    pick: str | None,
    *,
    p1: float | None = None,
    px: float | None = None,
    p2: float | None = None,
    note: str | None = None,
    missing: bool = False,
) -> dict[str, Any]:
    return {
        "fonte": name,
        "idea": idea,
        "pick": "n/d" if missing else (pick or "n/d"),
        "p_1": None if p1 is None else round(float(p1), 4),
        "p_x": None if px is None else round(float(px), 4),
        "p_2": None if p2 is None else round(float(p2), 4),
        "nota": note,
        "mancante": missing,
    }


def validation_source(validation: dict[str, Any] | None) -> dict[str, Any]:
    val = validation or {}
    notes: list[str] = []
    for key in ("venue", "tactical", "market", "stability", "form", "sportly_sim", "data_signal"):
        block = val.get(key) or {}
        notes.extend((block.get("notes") or [])[:2])
    p_adj = val.get("p_validated") or {}
    delta = val.get("delta_unified")
    head = val.get("summary") or "controlli automatici"
    if delta:
        head = f"{head} · Δ voto {delta:+.1f}"
    return _src(
        "Validazione",
        "stadio · tattica · mercato · ML/MC · forma · sim · dati (non EV)",
        None,
        p1=p_adj.get("home"),
        px=p_adj.get("draw"),
        p2=p_adj.get("away"),
        missing=False,
        note=head + ((" · " + " · ".join(notes[:6])) if notes else ""),
    )


def build_quadro(
    *,
    home: str,
    away: str,
    play: dict[str, Any],
    prediction: dict[str, Any],
    grouped: dict[str, list],
    alignment: dict[str, Any] | None,
    market_move: dict[str, Any] | None,
    tipster: dict[str, Any] | None,
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mc = prediction.get("montecarlo") or {}
    ml = prediction.get("model_probabilities") or {}
    feat = prediction.get("features") or {}
    xg = prediction.get("expected_goals") or {}
    fbref = prediction.get("fbref_context") or {}
    understat = prediction.get("understat_context") or {}
    statsbomb = prediction.get("statsbomb_context") or {}
    sofascore = prediction.get("sofascore_context") or {}
    fotmob = prediction.get("fotmob_context") or {}
    local_hist = prediction.get("history_context") or {}
    if not local_hist:
        try:
            from modules.data_update.history import lookup_history_match

            local_hist = lookup_history_match(home, away, league=prediction.get("league"))
        except Exception:
            local_hist = {}
    p_draw = float(mc.get("draw") or ml.get("draw") or 0.26)

    sources: list[dict[str, Any]] = []

    if mc.get("home_win") is None:
        sources.append(_src("Monte Carlo", "simulazione gol", None, missing=True, note="squadre non nel modello"))
    else:
        sources.append(
            _src(
                "Monte Carlo",
                "simulazione gol",
                _lean_1x2(mc.get("home_win"), mc.get("draw"), mc.get("away_win")),
                p1=mc.get("home_win"),
                px=mc.get("draw"),
                p2=mc.get("away_win"),
                note="distribuzione da λ gol",
            )
        )
    if ml.get("home_win") is None:
        sources.append(_src("Modello ML", "classificatore 1X2", None, missing=True, note="squadre non nel modello"))
    else:
        sources.append(
            _src(
                "Modello ML",
                "classificatore 1X2",
                _lean_1x2(ml.get("home_win"), ml.get("draw"), ml.get("away_win")),
                p1=ml.get("home_win"),
                px=ml.get("draw"),
                p2=ml.get("away_win"),
                note="ensemble XGB + Dixon-Coles (λ da rolling xG, Understat se c'è)",
            )
        )

    m1 = next((m for m in (grouped.get("1x2") or []) if m.get("code") == "1"), None)
    mx = next((m for m in (grouped.get("1x2") or []) if m.get("code") == "X"), None)
    m2 = next((m for m in (grouped.get("1x2") or []) if m.get("code") == "2"), None)
    if m1 and mx and m2 and m1.get("p_market") is not None:
        sources.append(
            _src(
                "Book (devig)",
                "probabilità implicita",
                _lean_1x2(m1.get("p_market"), mx.get("p_market"), m2.get("p_market")),
                p1=m1.get("p_market"),
                px=mx.get("p_market"),
                p2=m2.get("p_market"),
                note="quota reale, margine rimosso",
            )
        )
    else:
        sources.append(_src("Book (devig)", "probabilità implicita", None, missing=True, note="quote 1X2 assenti"))

    eh, ea = feat.get("home_elo"), feat.get("away_elo")
    if eh is not None and ea is not None:
        p_h2 = _elo_home_win(float(eh), float(ea))
        p1, px, p2 = _two_way_to_1x2(p_h2, p_draw)
        sources.append(
            _src(
                "Elo interno",
                "forza da risultati nostri",
                _lean_1x2(p1, px, p2),
                p1=p1,
                px=px,
                p2=p2,
                note=f"{float(eh):.0f} vs {float(ea):.0f} (parte da 1500, coppe poco visibili)",
            )
        )

    club = None
    try:
        from modules.data_update.clubelo import lookup_elo

        h_elo = lookup_elo(home)
        a_elo = lookup_elo(away)
        if h_elo and a_elo:
            p_h2 = _elo_home_win(h_elo["elo"], a_elo["elo"])
            p1, px, p2 = _two_way_to_1x2(p_h2, p_draw)
            club = {"home": h_elo, "away": a_elo}
            sources.append(
                _src(
                    "ClubElo",
                    "forza europea esterna",
                    _lean_1x2(p1, px, p2),
                    p1=p1,
                    px=px,
                    p2=p2,
                    note=f"{h_elo['elo']:.0f} vs {a_elo['elo']:.0f} (include coppe UEFA)",
                )
            )
        else:
            sources.append(_src("ClubElo", "forza europea esterna", None, missing=True, note="cache assente o squadra non trovata"))
    except Exception:
        sources.append(_src("ClubElo", "forza europea esterna", None, missing=True, note="non disponibile"))

    lam_h = xg.get("home")
    lam_a = xg.get("away")
    if lam_h is not None and lam_a is not None:
        tot = float(lam_h) + float(lam_a)
        p_h2 = float(lam_h) / tot if tot else 0.5
        p1, px, p2 = _two_way_to_1x2(p_h2, p_draw)
        sources.append(
            _src(
                "xG / gol attesi",
                "atteso gol (proxy)",
                _lean_1x2(p1, px, p2),
                p1=p1,
                px=px,
                p2=p2,
                note=f"{float(lam_h):.2f} – {float(lam_a):.2f} · non è xG Understat/FBref, è stima da forma/gol",
            )
        )

    fb_h = fbref.get("home") or {}
    fb_a = fbref.get("away") or {}
    if fb_h and fb_a:
        h_ga = fb_h.get("ga_p90")
        a_ga = fb_a.get("ga_p90")
        h_poss = fb_h.get("poss")
        a_poss = fb_a.get("poss")
        try:
            ga_diff = float(h_ga) - float(a_ga)
        except (TypeError, ValueError):
            ga_diff = 0.0
        try:
            poss_diff = (float(h_poss) - float(a_poss)) / 20.0
        except (TypeError, ValueError):
            poss_diff = 0.0
        p_h2 = _clamp01(0.5 + ga_diff * 0.18 + poss_diff * 0.05)
        p1, px, p2 = _two_way_to_1x2(p_h2, p_draw)
        note = (
            f"GA/90 {fb_h.get('ga_p90', 'n/d')} vs {fb_a.get('ga_p90', 'n/d')} · "
            f"Poss {fb_h.get('poss', 'n/d')}% vs {fb_a.get('poss', 'n/d')}%"
        )
        sources.append(
            _src(
                "FBref",
                "forma team avanzata",
                _lean_1x2(p1, px, p2),
                p1=p1,
                px=px,
                p2=p2,
                note=note,
            )
        )
    elif fb_h or fb_a:
        sources.append(
            _src(
                "FBref",
                "forma team avanzata",
                None,
                missing=True,
                note="una sola squadra in cache, nessun lean",
            )
        )
    else:
        sources.append(_src("FBref", "forma team avanzata", None, missing=True, note="contesto non disponibile"))

    tac = prediction.get("tactical") or {}
    if not tac:
        try:
            from modules.advisor.tactics import match_tactics

            tac = match_tactics(
                home,
                away,
                None,
                None,
                fb_h,
                fb_a,
                country=prediction.get("country"),
                league=prediction.get("league"),
            )
        except Exception:
            tac = {}
    style = (tac or {}).get("style") or {}
    if style.get("ready"):
        edge = float(style.get("edge_home") or 0)
        p_h2 = _clamp01(0.5 + edge)
        p1, px, p2 = _two_way_to_1x2(p_h2, p_draw)
        sources.append(
            _src(
                "Matchup tattico",
                "stile FBref (possesso, cross, transizioni)",
                _lean_1x2(p1, px, p2),
                p1=p1,
                px=px,
                p2=p2,
                note=" · ".join(style.get("notes") or [])[:280],
            )
        )
    else:
        sources.append(
            _src(
                "Matchup tattico",
                "stile FBref (possesso, cross, transizioni)",
                None,
                missing=True,
                note=(style.get("notes") or ["copertura stile solo Big 5 FBref"])[0],
            )
        )
    fat = (tac or {}).get("fatigue") or {}
    if fat.get("ready"):
        edge = float(fat.get("edge_home") or 0)
        p_h2 = _clamp01(0.5 + edge)
        p1, px, p2 = _two_way_to_1x2(p_h2, p_draw)
        pick = _lean_1x2(p1, px, p2) if abs(edge) >= 0.03 else None
        sources.append(
            _src(
                "Fatica / calendario",
                "riposo, 3 in 7 giorni, viaggi",
                pick,
                p1=p1 if pick else None,
                px=px if pick else None,
                p2=p2 if pick else None,
                note=" · ".join(fat.get("notes") or [])[:280],
            )
        )
    else:
        sources.append(_src("Fatica / calendario", "riposo, 3 in 7 giorni, viaggi", None, missing=True))
    absn = (tac or {}).get("absences") or {}
    if absn.get("ready"):
        edge = float(absn.get("edge_home") or 0)
        p_h2 = _clamp01(0.5 + edge)
        p1, px, p2 = _two_way_to_1x2(p_h2, p_draw)
        sources.append(
            _src(
                "Assenze / XI",
                "WhoScored confermati × peso FBref xG+xA",
                _lean_1x2(p1, px, p2) if abs(edge) >= 0.03 else None,
                p1=p1 if abs(edge) >= 0.03 else None,
                px=px if abs(edge) >= 0.03 else None,
                p2=p2 if abs(edge) >= 0.03 else None,
                note=(
                    f"peso casa {absn.get('weight_home', 0):.0%} vs ospite {absn.get('weight_away', 0):.0%} · "
                    + " · ".join(absn.get("notes") or [])
                )[:280],
            )
        )
    else:
        sources.append(
            _src(
                "Assenze / XI",
                "WhoScored confermati × peso FBref xG+xA",
                None,
                missing=True,
                note=(absn.get("notes") or ["premi WhoScored a sinistra per i preview"])[0],
            )
        )
    combos = (tac or {}).get("combos") or {}
    if combos.get("ready") and combos.get("blend") is not None:
        b = float(combos["blend"])
        p_h2 = _clamp01(b)
        p1, px, p2 = _two_way_to_1x2(p_h2, p_draw)
        sources.append(
            _src(
                "Combo tattica",
                "1 stile · 2 infortuni · 3 value/Asian",
                _lean_1x2(p1, px, p2),
                p1=p1,
                px=px,
                p2=p2,
                note=(
                    f"blend {b:.0%} · c1 {combos.get('combo1_tactics')} · "
                    f"c2 {combos.get('combo2_injuries')} · c3 {combos.get('combo3_value')}"
                ),
            )
        )
    else:
        sources.append(
            _src(
                "Combo tattica",
                "1 stile · 2 infortuni · 3 value/Asian",
                None,
                missing=True,
                note="servono almeno 2 combo (FBref/Sofascore, WhoScored, oppure EV+Asian)",
            )
        )
    padj = (tac or {}).get("p_tactical")
    if padj:
        sources.append(
            _src(
                "P tattica",
                "P_ML × (1−infortuni) × (1+stile) — non è EV",
                _lean_1x2(padj.get("home"), padj.get("draw"), padj.get("away")),
                p1=padj.get("home"),
                px=padj.get("draw"),
                p2=padj.get("away"),
                note="solo lettura; Kelly e EV restano sul modello grezzo",
            )
        )

    us_h = understat.get("home") or {}
    us_a = understat.get("away") or {}
    if us_h and us_a:
        h_diff = us_h.get("xg_diff")
        a_diff = us_a.get("xg_diff")
        try:
            p_h2 = _clamp01(0.5 + (float(h_diff) - float(a_diff)) * 0.25)
        except (TypeError, ValueError):
            p_h2 = 0.5
        p1, px, p2 = _two_way_to_1x2(p_h2, p_draw)
        sources.append(
            _src(
                "Understat",
                "xG reale (anche in λ Poisson)",
                _lean_1x2(p1, px, p2),
                p1=p1,
                px=px,
                p2=p2,
                note=(
                    f"xGdiff {us_h.get('xg_diff', 'n/d')} vs {us_a.get('xg_diff', 'n/d')} · "
                    f"xG {us_h.get('xg_for', 'n/d')}/{us_h.get('xg_against', 'n/d')} vs "
                    f"{us_a.get('xg_for', 'n/d')}/{us_a.get('xg_against', 'n/d')} · "
                    f"blend 38% nelle λ"
                ),
            )
        )
    elif us_h or us_a:
        sources.append(
            _src(
                "Understat",
                "xG reale (anche in λ Poisson)",
                None,
                missing=True,
                note="una sola squadra in cache, nessun lean",
            )
        )
    else:
        sources.append(_src("Understat", "xG reale storico", None, missing=True, note="contesto non disponibile"))

    sb_h = statsbomb.get("home") or {}
    sb_a = statsbomb.get("away") or {}
    sb_n_ok = False
    try:
        sb_n_ok = float(sb_h.get("n") or 0) >= 5 and float(sb_a.get("n") or 0) >= 5
    except (TypeError, ValueError):
        sb_n_ok = False
    if sb_h and sb_a and sb_n_ok:
        try:
            p_h2 = _clamp01(0.5 + (float(sb_h.get("gd_pg") or 0) - float(sb_a.get("gd_pg") or 0)) * 0.18)
        except (TypeError, ValueError):
            p_h2 = 0.5
        p1, px, p2 = _two_way_to_1x2(p_h2, p_draw)
        sources.append(
            _src(
                "StatsBomb",
                "eventi open data",
                _lean_1x2(p1, px, p2),
                p1=p1,
                px=px,
                p2=p2,
                note=(
                    f"PPG {sb_h.get('ppg', 'n/d')} vs {sb_a.get('ppg', 'n/d')} · "
                    f"GD/pg {sb_h.get('gd_pg', 'n/d')} vs {sb_a.get('gd_pg', 'n/d')} · "
                    f"{sb_h.get('season') or sb_a.get('season') or ''} "
                    f"(storico open data, non la stagione in corso)"
                ).strip(),
            )
        )
    elif sb_h or sb_a:
        note = "una sola squadra in cache, nessun lean"
        if sb_h and sb_a and not sb_n_ok:
            note = "troppo pochi match nell'open data per un lean"
        sources.append(_src("StatsBomb", "eventi open data", None, missing=True, note=note))
    else:
        sources.append(_src("StatsBomb", "eventi open data", None, missing=True, note="open data non copre queste squadre"))

    sofa_h = sofascore.get("home") or {}
    sofa_a = sofascore.get("away") or {}
    if sofa_h and sofa_a:
        try:
            p_h2 = _clamp01(
                0.5
                + (float(sofa_h.get("ppg") or 0) - float(sofa_a.get("ppg") or 0)) * 0.16
                + (float(sofa_h.get("gd_pg") or 0) - float(sofa_a.get("gd_pg") or 0)) * 0.04
            )
        except (TypeError, ValueError):
            p_h2 = 0.5
        p1, px, p2 = _two_way_to_1x2(p_h2, p_draw)
        sources.append(
            _src(
                "Sofascore",
                "classifica attuale",
                _lean_1x2(p1, px, p2),
                p1=p1,
                px=px,
                p2=p2,
                note=(
                    f"PPG {sofa_h.get('ppg', 'n/d')} vs {sofa_a.get('ppg', 'n/d')} · "
                    f"Pts {sofa_h.get('pts', 'n/d')}-{sofa_a.get('pts', 'n/d')} · "
                    f"{sofa_h.get('league') or ''}"
                ).strip(),
            )
        )
    elif sofa_h or sofa_a:
        sources.append(
            _src("Sofascore", "classifica attuale", None, missing=True, note="una sola squadra in cache, nessun lean")
        )
    else:
        sources.append(_src("Sofascore", "classifica attuale", None, missing=True, note="contesto non disponibile"))

    fm_h = fotmob.get("home") or {}
    fm_a = fotmob.get("away") or {}
    fm_m = fotmob.get("match") or {}
    fm_note_bits: list[str] = []
    if fm_m.get("match_id"):
        fm_note_bits.append(f"id {fm_m['match_id']}")
        if fm_m.get("league"):
            fm_note_bits.append(str(fm_m["league"]))
        if fm_m.get("finished"):
            fm_note_bits.append(f"FT {fm_m.get('score') or ''}".strip())
        elif fm_m.get("started"):
            fm_note_bits.append("live")
        else:
            fm_note_bits.append("pre-match")
    if fm_h and fm_a:
        played_ok = float(fm_h.get("played") or 0) >= 1 and float(fm_a.get("played") or 0) >= 1
        if played_ok:
            try:
                p_h2 = _clamp01(
                    0.5
                    + (float(fm_h.get("ppg") or 0) - float(fm_a.get("ppg") or 0)) * 0.16
                    + (float(fm_h.get("gd_pg") or 0) - float(fm_a.get("gd_pg") or 0)) * 0.04
                )
            except (TypeError, ValueError):
                p_h2 = 0.5
            p1, px, p2 = _two_way_to_1x2(p_h2, p_draw)
            sources.append(
                _src(
                    "FotMob",
                    "classifica + calendario live",
                    _lean_1x2(p1, px, p2),
                    p1=p1,
                    px=px,
                    p2=p2,
                    note=(
                        f"PPG {fm_h.get('ppg', 'n/d')} vs {fm_a.get('ppg', 'n/d')} · "
                        f"Pts {fm_h.get('pts', 'n/d')}-{fm_a.get('pts', 'n/d')}"
                        + ((" · " + " · ".join(fm_note_bits)) if fm_note_bits else "")
                    ).strip(),
                )
            )
        else:
            sources.append(
                _src(
                    "FotMob",
                    "classifica + calendario live",
                    None,
                    missing=True,
                    note=(
                        "stagione ancora senza punti in classifica"
                        + ((" · " + " · ".join(fm_note_bits)) if fm_note_bits else "")
                    ),
                )
            )
    elif fm_m.get("match_id"):
        sources.append(
            _src(
                "FotMob",
                "classifica + calendario live",
                None,
                missing=True,
                note="partita in calendario, classifica squadre non matchata · " + " · ".join(fm_note_bits),
            )
        )
    elif fm_h or fm_a:
        sources.append(
            _src("FotMob", "classifica + calendario live", None, missing=True, note="una sola squadra in cache")
        )
    else:
        sources.append(
            _src("FotMob", "classifica + calendario live", None, missing=True, note="contesto non disponibile")
        )

    # FotMob matchDetails (top picks): xG live / momentum / shotmap — solo note quadro
    fm_d = fotmob.get("details") or {}
    if fm_d and (fm_d.get("xg_home") is not None or fm_d.get("has_momentum") or fm_d.get("has_shotmap")):
        bits = []
        if fm_d.get("xg_home") is not None and fm_d.get("xg_away") is not None:
            bits.append(f"xG {fm_d['xg_home']}-{fm_d['xg_away']}")
        if fm_d.get("poss_home") is not None:
            bits.append(f"poss {fm_d.get('poss_home')}-{fm_d.get('poss_away')}")
        if fm_d.get("momentum_avg") is not None:
            bits.append(f"mom {fm_d['momentum_avg']}")
        elif fm_d.get("has_momentum"):
            bits.append("momentum ok")
        if fm_d.get("shotmap_n"):
            bits.append(f"shots map {fm_d['shotmap_n']}")
        elif fm_d.get("has_shotmap"):
            bits.append("shotmap")
        if fm_d.get("has_lineup"):
            bits.append("lineup")
        sources.append(
            _src(
                "FotMob details",
                "xG/momentum/shotmap (on-demand)",
                None,
                missing=False,
                note=" · ".join(bits) or "dettagli disponibili",
            )
        )

    sim = prediction.get("sportly_sim") or {}
    if sim.get("ready"):
        tv = sim.get("tactical_validation") or {}
        sources.append(
            _src(
                "Sportly-sim",
                "xG/momentum/pressione sintetici",
                sim.get("lean") or _lean_1x2(sim.get("p_1"), sim.get("p_x"), sim.get("p_2")),
                p1=sim.get("p_1"),
                px=sim.get("p_x"),
                p2=sim.get("p_2"),
                note=(
                    f"xG {sim.get('xg', {}).get('home', 'n/d')}/{sim.get('xg', {}).get('away', 'n/d')} · "
                    f"tiri {(sim.get('shots') or {}).get('home_n', '?')}-{(sim.get('shots') or {}).get('away_n', '?')} · "
                    f"{tv.get('status') or 'n/d'}"
                    + (f" · {tv['notes'][0]}" if tv.get("notes") else "")
                ),
            )
        )
    else:
        sources.append(
            _src(
                "Sportly-sim",
                "xG/momentum/pressione sintetici",
                None,
                missing=True,
                note=sim.get("note") or "simulazione non disponibile",
            )
        )

    data_sig = prediction.get("data_signal") or {}
    if data_sig.get("ready"):
        sources.append(
            _src(
                "Analisi dati",
                "xG · forma · casa/trasferta · classifiche",
                data_sig.get("lean") or _lean_1x2(data_sig.get("p_1"), data_sig.get("p_x"), data_sig.get("p_2")),
                p1=data_sig.get("p_1"),
                px=data_sig.get("p_x"),
                p2=data_sig.get("p_2"),
                note=(
                    f"edge {data_sig.get('edge', 'n/d')} · conf {float(data_sig.get('confidence') or 0):.0%} · "
                    f"{data_sig.get('n_factors', 0)} fattori"
                    + (f" · {data_sig.get('note')}" if data_sig.get("note") else "")
                ),
            )
        )
    else:
        sources.append(
            _src(
                "Analisi dati",
                "xG · forma · casa/trasferta · classifiche",
                None,
                missing=True,
                note=data_sig.get("note") or "segnale non calcolato",
            )
        )

    hist_h = (local_hist or {}).get("home") or {}
    hist_a = (local_hist or {}).get("away") or {}
    if local_hist.get("ready") and hist_h and hist_a:
        try:
            p_h2 = _clamp01(0.5 + (float(hist_h.get("gd_pg") or 0) - float(hist_a.get("gd_pg") or 0)) * 0.20)
        except (TypeError, ValueError):
            p_h2 = 0.5
        p1, px, p2 = _two_way_to_1x2(p_h2, p_draw)
        w = int(round(float(local_hist.get("weight") or 0.12) * 100))
        sources.append(
            _src(
                "Storico locale",
                f"esiti nostri (≥{local_hist.get('min_team', 6)} match/squadra)",
                _lean_1x2(p1, px, p2),
                p1=p1,
                px=px,
                p2=p2,
                note=(
                    f"PPG {hist_h.get('ppg')} vs {hist_a.get('ppg')} · "
                    f"n {hist_h.get('n')}/{hist_a.get('n')} · "
                    f"peso voto {w}% · chiusi {local_hist.get('n_global')}"
                    + (
                        f" · lega {local_hist.get('n_league')}"
                        if local_hist.get("n_league")
                        else ""
                    )
                ),
            )
        )
    else:
        need_g = local_hist.get("min_global", 30)
        need_t = local_hist.get("min_team", 6)
        have = local_hist.get("n_global", 0)
        sources.append(
            _src(
                "Storico locale",
                "esiti nostri",
                None,
                missing=True,
                note=(
                    f"ancora non entra nel voto ({have}/{need_g} partite chiuse, "
                    f"servono {need_t} per squadra)"
                ),
            )
        )

    tip = tipster or {}
    if tip.get("n_sources"):
        cons = str(tip.get("consensus") or "").upper().replace("X", "X")
        if cons not in {"1", "X", "2"}:
            cons = _lean_1x2(tip.get("p_home"), tip.get("p_draw"), tip.get("p_away"))
        sources.append(
            _src(
                "Tipster",
                "consenso modelli pubblici",
                cons,
                p1=tip.get("p_home"),
                px=tip.get("p_draw"),
                p2=tip.get("p_away"),
                note=f"{tip.get('n_sources')} fonti · {tip.get('agree') or tip.get('label') or ''}".strip(" ·"),
            )
        )
    else:
        sources.append(_src("Tipster", "consenso modelli pubblici", None, missing=True, note="nessun match Forebet/Vitibet"))

    move = market_move or {}
    steam_pick = move.get("steam_1x2") if move.get("steam_1x2") in {"1", "X", "2"} else None
    label = (alignment or {}).get("label")
    if steam_pick:
        sources.append(
            _src(
                "Steam Asian",
                "flusso quote apertura→attuale",
                steam_pick,
                note=move.get("movement_comment") or f"allineamento {label or 'n/d'}",
            )
        )
    else:
        sources.append(
            _src(
                "Steam Asian",
                "flusso quote apertura→attuale",
                None,
                missing=not move,
                note=(move.get("movement_level") or "n/d") if move else "niente Asian su questa partita",
            )
        )

    wx = prediction.get("weather") or {}
    if wx.get("flag") or wx.get("precip_mm") is not None or wx.get("temp_c") is not None:
        bits: list[str] = []
        if wx.get("city"):
            bits.append(str(wx["city"]))
        if wx.get("temp_c") is not None:
            bits.append(f"{wx['temp_c']}°C")
        if wx.get("precip_mm") is not None:
            bits.append(f"{wx['precip_mm']} mm")
        if wx.get("wind_kmh") is not None:
            bits.append(f"vento {wx['wind_kmh']} km/h")
        try:
            adj = float(wx.get("lambda_adj") or 1)
        except (TypeError, ValueError):
            adj = 1.0
        if abs(adj - 1.0) > 1e-6:
            bits.append(f"λ ×{adj}")
        sources.append(
            _src(
                "Meteo",
                "Open-Meteo forecast",
                None,
                note=f"{wx.get('flag') or 'ok'}" + ((" · " + " · ".join(bits)) if bits else ""),
            )
        )
    else:
        sources.append(
            _src("Meteo", "Open-Meteo forecast", None, missing=True, note="città stadio assente o forecast non disponibile")
        )

    if validation:
        sources.append(validation_source(validation))

    play_code = str(play.get("code") or "")
    play_group = play.get("group") or "1x2"
    votes = [s for s in sources if not s.get("mancante") and s.get("pick") in {"1", "X", "2"}]
    # Fonti esterne: validazione, non generazione. Niente maggioranza → pick.
    if play_code not in {"1", "X", "2", "1X", "X2", "12", "1 DNB", "2 DNB"}:
        consenso = "nessun pick"
        agree, other = [], votes
        n_votes = len(votes)
        share = None
        summary = (
            "Nessun pick: ClubElo, FBref, tipster e le altre fonti validano, non generano giocate. "
            "Servono modello e quota reale per edge, EV e Kelly."
        )
        play_code = "—"
        play_group = "1x2"
    elif play_group != "1x2":
        consenso = "quadro sul 1X2 (il pick è un altro mercato)"
        agree, other = [], votes
        n_votes = len(votes)
        share = None
        summary = (
            f"Pick {play_code}: il consiglio non è 1X2. "
            f"Sotto, {n_votes} letture sul risultato. EV/Kelly restano sul mercato scelto."
        )
    else:
        agree = [s for s in votes if s["pick"] == play_code]
        other = [s for s in votes if s["pick"] != play_code]
        n_votes = len(votes)
        share = (len(agree) / n_votes) if n_votes else None

        if share is None:
            consenso = "n/d"
        elif share >= 0.75:
            consenso = "ampio accordo"
        elif share >= 0.5:
            consenso = "maggioranza"
        elif share > 0:
            consenso = "quadro spezzato"
        else:
            consenso = "nessuna fonte sul pick"

        no_model = ml.get("home_win") is None and mc.get("home_win") is None
        summary = (
            f"Pick {play_code}: {consenso}"
            + (f" ({len(agree)}/{n_votes} fonti)" if n_votes else "")
            + (
                ". Senza modello: le fonti esterne restano validazione, niente EV/Kelly."
                if no_model
                else ". EV e Kelly usano solo modello+quota, non questo quadro."
            )
        )
        if other:
            contra = ", ".join(sorted({s["fonte"] for s in other}))
            summary += f" Contrari: {contra}."

    form = None
    if feat:
        def _n(v, d=2):
            if v is None:
                return None
            try:
                return round(float(v), d)
            except (TypeError, ValueError):
                return None

        form = {
            "pts_casa": _n(feat.get("home_form_pts"), 1),
            "pts_trasferta": _n(feat.get("away_form_pts"), 1),
            "xg_casa": _n(feat.get("home_xg_avg")),
            "xg_trasferta": _n(feat.get("away_xg_avg")),
            "riposo_casa": _n(feat.get("home_rest_days"), 0),
            "riposo_trasferta": _n(feat.get("away_rest_days"), 0),
            "wr_casa": _n(feat.get("home_home_wr"), 3),
            "wr_trasferta": _n(feat.get("away_away_wr"), 3),
        }

    gaps = [
        "infortuni e formazioni (XI live / Elo giocatore: nessuna fonte ufficiale gratis)",
        "motivazioni e clima (salvezza, rotazioni da comunicato: il mercato Asian è il proxy)",
        "altri book oltre football-data (Avg OddsPortal/BetBrain) e Bet365 Asian",
        "OddsPortal/BetExplorer in tempo reale (niente libreria ufficiale, solo scraper)",
    ]
    if not (wx.get("flag") or wx.get("precip_mm") is not None or wx.get("temp_c") is not None):
        gaps.insert(0, "meteo (Open-Meteo: manca la città dello stadio)")
    if not (us_h or us_a):
        gaps.insert(0, "xG Understat (λ da rolling gol/forma; FBref gls/90 solo fallback)")
    if not club:
        gaps.insert(0, "ClubElo (download fallito o squadra assente)")
    if not (fb_h or fb_a):
        gaps.insert(0, "FBref team stats (copertura lega/squadra limitata)")
    if not (sb_h or sb_a):
        gaps.insert(0, "StatsBomb open data (poche stagioni club, soprattutto storiche)")
    if not (sofa_h or sofa_a):
        gaps.insert(0, "Sofascore classifica (Big 5, soccerdata)")
    if not (fm_h or fm_a or fm_m.get("match_id")):
        gaps.insert(0, "FotMob classifica/calendario (API /api/data, non ufficiale)")
    if not ((prediction.get("sportly_sim") or {}).get("ready")):
        gaps.insert(0, "Sportly-sim (λ gol assenti o sim non calcolata)")
    if not ((prediction.get("data_signal") or {}).get("ready")):
        gaps.insert(0, "Analisi dati (servono feature e/o Understat/FBref/classifica)")
    if not (local_hist or {}).get("ready"):
        gaps.insert(0, "Storico locale (SQLite): poche partite chiuse per entrare nel voto")

    return {
        "summary": summary,
        "consenso": consenso,
        "agree_n": len(agree),
        "votes_n": n_votes,
        "agree_share": None if share is None else round(share, 3),
        "sources": sources,
        "form": form,
        "clubelo": club,
        "fbref_summary": None
        if not (fb_h or fb_a)
        else (
            f"FBref GA/90 {fb_h.get('ga_p90', 'n/d')} vs {fb_a.get('ga_p90', 'n/d')} · "
            f"Poss {fb_h.get('poss', 'n/d')}% vs {fb_a.get('poss', 'n/d')}%"
        ),
        "gaps": gaps,
        "play_code": play_code,
        "validation": validation,
    }
