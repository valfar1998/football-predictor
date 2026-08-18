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
) -> dict[str, Any]:
    mc = prediction.get("montecarlo") or {}
    ml = prediction.get("model_probabilities") or {}
    feat = prediction.get("features") or {}
    xg = prediction.get("expected_goals") or {}
    p_draw = float(mc.get("draw") or ml.get("draw") or 0.26)

    sources: list[dict[str, Any]] = []

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
    sources.append(
        _src(
            "Modello ML",
            "classificatore 1X2",
            _lean_1x2(ml.get("home_win"), ml.get("draw"), ml.get("away_win")),
            p1=ml.get("home_win"),
            px=ml.get("draw"),
            p2=ml.get("away_win"),
            note="feature forma/Elo interno/xG proxy",
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

    play_code = str(play.get("code") or "")
    play_group = play.get("group") or "1x2"
    votes = [s for s in sources if not s.get("mancante") and s.get("pick") in {"1", "X", "2"}]
    if play_group != "1x2":
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

        summary = (
            f"Pick {play_code}: {consenso}"
            + (f" ({len(agree)}/{n_votes} fonti)" if n_votes else "")
            + ". EV e Kelly usano solo modello+quota, non questo quadro."
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
        "xG vero Understat/FBref (qui è proxy da gol/forma)",
        "infortuni e formazioni",
        "meteo",
        "altri book oltre football-data e Bet365 Asian",
    ]
    if not club:
        gaps.insert(0, "ClubElo (download fallito o squadra assente)")

    return {
        "summary": summary,
        "consenso": consenso,
        "agree_n": len(agree),
        "votes_n": n_votes,
        "agree_share": None if share is None else round(share, 3),
        "sources": sources,
        "form": form,
        "clubelo": club,
        "gaps": gaps,
        "play_code": play_code,
    }
