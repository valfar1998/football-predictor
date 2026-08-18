"""Matchup tattico, fatica, assenze WhoScored×FBref, tre combo nel voto.

EV/Kelly restano sul modello. Transfermarkt non ha API ufficiale:
il peso infortunio è la quota xG+xA FBref del giocatore out.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from modules.data_update.team_names import resolve_known_team

HIGH_TRAVEL = {
    "usa",
    "united states",
    "brazil",
    "brasile",
    "argentina",
    "mexico",
    "messico",
    "australia",
    "japan",
    "giappone",
    "colombia",
    "chile",
    "cile",
    "ecuador",
}

HIGH_TRAVEL_LEAGUE = (
    "mls",
    "liga mx",
    "liga profesional",
    "brasileiro",
    "serie a brazil",
    "campeonato brasileiro",
    "league one",
    "liga 1",
    "j1",
    "a-league",
)


def _f(v: Any, default: float | None = None) -> float | None:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _clip(x: float, lo: float = -0.25, hi: float = 0.25) -> float:
    return max(lo, min(hi, x))


def style_matchup(home: dict[str, Any] | None, away: dict[str, Any] | None) -> dict[str, Any]:
    """Confronta stile FBref: possesso, cross, distanza tiri, recuperi."""
    h, a = home or {}, away or {}
    if not h or not a:
        return {
            "ready": False,
            "notes": ["FBref stile assente su una o entrambe le squadre (copertura Big 5)."],
            "edge_home": 0.0,
        }

    notes: list[str] = []
    edge = 0.0

    poss_h, poss_a = _f(h.get("poss")), _f(a.get("poss"))
    if poss_h is not None and poss_a is not None:
        d = poss_h - poss_a
        if abs(d) >= 8:
            who = "casa" if d > 0 else "trasferta"
            notes.append(f"Possesso: {who} più territoriale ({poss_h:.0f}% vs {poss_a:.0f}%)")
            edge += _clip(d / 80.0)
        if poss_h >= 58 and poss_a <= 44:
            notes.append("Pressing/territorio casa vs blocco basso ospite")
            edge += 0.04
        elif poss_a >= 58 and poss_h <= 44:
            notes.append("Ospite territoriale vs blocco basso casa")
            edge -= 0.04

    crs_h, crs_a = _f(h.get("crosses_p90")), _f(a.get("crosses_p90"))
    conc_h, conc_a = _f(h.get("crosses_conc_p90")), _f(a.get("crosses_conc_p90"))
    if crs_h is not None and conc_a is not None and crs_h >= 16 and conc_a <= 12:
        notes.append(
            f"Casa crossa molto ({crs_h:.1f}/90) vs difesa che concede pochi cross ({conc_a:.1f}/90) → attacco spesso sterile"
        )
        edge -= 0.06
    if crs_a is not None and conc_h is not None and crs_a >= 16 and conc_h <= 12:
        notes.append(
            f"Ospite crossa molto ({crs_a:.1f}/90) vs casa compatta sui cross ({conc_h:.1f}/90)"
        )
        edge += 0.05

    dist_h, dist_a = _f(h.get("shot_dist")), _f(a.get("shot_dist"))
    if dist_h is not None and dist_a is not None and abs(dist_h - dist_a) >= 1.4:
        closer = "casa" if dist_h < dist_a else "trasferta"
        notes.append(
            f"Tiri più vicini per {closer} ({dist_h:.1f}m vs {dist_a:.1f}m): creazione migliore, non solo volume"
        )
        edge += _clip((dist_a - dist_h) / 20.0)

    rec_h, rec_a = _f(h.get("recov_p90")), _f(a.get("recov_p90"))
    if rec_h is not None and rec_a is not None and abs(rec_h - rec_a) >= 4:
        who = "casa" if rec_h > rec_a else "trasferta"
        notes.append(f"Recuperi palla: {who} più aggressiva in transizione ({rec_h:.1f} vs {rec_a:.1f}/90)")
        edge += _clip((rec_h - rec_a) / 40.0)

    if not notes:
        notes.append("Stili simili su possesso/cross/tiri: pochi mismatch evidenti")

    return {
        "ready": True,
        "notes": notes,
        "edge_home": round(_clip(edge), 3),
        "poss_h": poss_h,
        "poss_a": poss_a,
    }


def _is_high_travel(country: str | None, league: str | None) -> bool:
    c = str(country or "").strip().lower()
    lg = str(league or "").strip().lower()
    if c in HIGH_TRAVEL:
        return True
    return any(tok in lg for tok in HIGH_TRAVEL_LEAGUE)


def fatigue_for_team(
    dates: list[pd.Timestamp],
    kickoff: pd.Timestamp,
    *,
    country: str | None = None,
    league: str | None = None,
) -> dict[str, Any]:
    ko = pd.Timestamp(kickoff).normalize()
    past = [pd.Timestamp(d).normalize() for d in dates if pd.notna(d) and pd.Timestamp(d).normalize() < ko]
    past.sort()
    rest = 7 if not past else int(max((ko - past[-1]).days, 1))
    week = [d for d in past if (ko - d).days <= 7]
    in_7 = len(week)
    around = [d for d in dates if pd.notna(d) and abs((pd.Timestamp(d).normalize() - ko).days) <= 7]
    packed = len({pd.Timestamp(d).normalize() for d in around}) >= 3
    travel = _is_high_travel(country, league)
    tired = rest <= 3 or in_7 >= 2 or packed
    return {
        "rest_days": rest,
        "matches_7d": in_7,
        "packed_7d": packed,
        "high_travel": travel,
        "tired": tired,
    }


def fatigue_matchup(
    home_dates: list[pd.Timestamp],
    away_dates: list[pd.Timestamp],
    kickoff: pd.Timestamp,
    *,
    home_country: str | None = None,
    away_country: str | None = None,
    home_league: str | None = None,
    away_league: str | None = None,
) -> dict[str, Any]:
    h = fatigue_for_team(home_dates, kickoff, country=home_country, league=home_league)
    a = fatigue_for_team(away_dates, kickoff, country=away_country, league=away_league)
    notes: list[str] = []
    edge = 0.0
    if h["rest_days"] <= 3 and a["rest_days"] >= 6:
        notes.append(f"Riposo: casa {h['rest_days']} gg vs ospite {a['rest_days']} gg")
        edge -= 0.07
    elif a["rest_days"] <= 3 and h["rest_days"] >= 6:
        notes.append(f"Riposo: ospite {a['rest_days']} gg vs casa {h['rest_days']} gg")
        edge += 0.07
    if h["packed_7d"] and not a["packed_7d"]:
        notes.append("Casa: 3+ partite in 7 giorni (fatica/rotazioni)")
        edge -= 0.05
    elif a["packed_7d"] and not h["packed_7d"]:
        notes.append("Ospite: 3+ partite in 7 giorni")
        edge += 0.05
    if h["high_travel"] or a["high_travel"]:
        notes.append("Lega ad alto chilometraggio (MLS / Brasile / Argentina / analoghe): i viaggi pesano più dello storico")
    if not notes:
        notes.append(
            f"Calendario ordinario · riposo {h['rest_days']} vs {a['rest_days']} gg · "
            f"match ultimi 7gg {h['matches_7d']}/{a['matches_7d']}"
        )
    return {
        "ready": True,
        "home": h,
        "away": a,
        "notes": notes,
        "edge_home": round(_clip(edge), 3),
    }


def absences_gap() -> dict[str, Any]:
    return {
        "ready": False,
        "weight_home": 0.0,
        "weight_away": 0.0,
        "notes": [
            "Assenze: premi WhoScored in colonna sinistra. Peso = quota xG+xA FBref del giocatore out "
            "(proxy del valore in campo; Transfermarkt market value non ha libreria ufficiale)."
        ],
    }


def _row_confirmed(row: dict[str, Any]) -> bool:
    st = str(row.get("status") or row.get("Status") or "").lower()
    return "doubt" not in st and "unconfirmed" not in st


def injury_weights(
    missing_home: list[dict[str, Any]],
    missing_away: list[dict[str, Any]],
    *,
    home: str,
    away: str,
) -> dict[str, Any]:
    try:
        from modules.data_update.fbref_context import load_player_contrib_index, player_share

        idx = load_player_contrib_index()
    except Exception:
        idx = {}

    def side(rows: list[dict[str, Any]], team: str) -> tuple[float, list[str]]:
        w = 0.0
        notes: list[str] = []
        seen: set[str] = set()
        for row in rows or []:
            player = str(row.get("player") or "").strip()
            if not player or player.lower() in seen:
                continue
            seen.add(player.lower())
            share = player_share(team, player, idx) if idx else 0.0
            if share <= 0:
                share = 0.045
            if not _row_confirmed(row):
                share *= 0.45
            w += share
            reason = str(row.get("reason") or row.get("status") or "out")
            notes.append(f"{player} ({reason}, peso {share:.0%})")
        return min(0.38, w), notes

    wh, nh = side(missing_home, home)
    wa, na = side(missing_away, away)
    ready = bool(missing_home or missing_away)
    notes = []
    if nh:
        notes.append("Casa out: " + "; ".join(nh[:4]))
    if na:
        notes.append("Ospite out: " + "; ".join(na[:4]))
    if not ready:
        notes = absences_gap()["notes"]
    return {
        "ready": ready,
        "weight_home": round(wh, 3),
        "weight_away": round(wa, 3),
        "edge_home": round(_clip(wa - wh), 3),
        "notes": notes,
    }


def combo_scores(
    *,
    style: dict[str, Any],
    fatigue: dict[str, Any],
    injuries: dict[str, Any],
    sofa_home: dict[str, Any] | None,
    sofa_away: dict[str, Any] | None,
    value_norm: float | None,
    asian_align: str | None,
    ws_style_home: dict[str, Any] | None = None,
    ws_style_away: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tre combo gratuite → un voto 0–1. Non ricalcola EV."""
    # Combo 1: tattica (FBref stile + WhoScored + Sofascore forma)
    bits1: list[float] = []
    notes1: list[str] = []
    if style.get("ready"):
        bits1.append(_clip(0.5 + float(style.get("edge_home") or 0), 0.05, 0.95))
        notes1.extend((style.get("notes") or [])[:2])
    fat_edge = float((fatigue or {}).get("edge_home") or 0)
    fat_h = (fatigue or {}).get("home") or {}
    fat_a = (fatigue or {}).get("away") or {}
    fat_signal = abs(fat_edge) >= 0.03 or bool(fat_h.get("tired") or fat_a.get("tired") or fat_h.get("high_travel") or fat_a.get("high_travel"))
    if (fatigue or {}).get("ready") and fat_signal:
        bits1.append(_clip(0.5 + fat_edge, 0.05, 0.95))
        notes1.extend((fatigue.get("notes") or [])[:2])
    if ws_style_home or ws_style_away:
        notes1.append("WhoScored stile/forza-debolezza in cache")
        bits1.append(0.55)
    sh, sa = sofa_home or {}, sofa_away or {}
    if sh.get("ppg") is not None and sa.get("ppg") is not None:
        d = float(sh["ppg"]) - float(sa["ppg"])
        bits1.append(_clip(0.5 + d / 4.0, 0.05, 0.95))
        notes1.append(f"Sofascore PPG {sh.get('ppg')} vs {sa.get('ppg')}")
    c1 = sum(bits1) / len(bits1) if bits1 else None

    # Combo 2: infortuni pesati WhoScored × FBref
    c2 = None
    notes2 = injuries.get("notes") or []
    if injuries.get("ready"):
        c2 = _clip(0.5 + float(injuries.get("edge_home") or 0) * 1.2, 0.05, 0.95)

    # Combo 3: value betting (EV già calcolato + allineamento Asian)
    c3 = None
    notes3: list[str] = []
    if value_norm is not None:
        c3 = _clip(float(value_norm), 0.05, 0.95)
        notes3.append(f"value {c3:.0%}")
    if asian_align in {"allineato", "contrario", "misto"}:
        add = {"allineato": 0.12, "misto": 0.0, "contrario": -0.12}[asian_align]
        if c3 is None:
            c3 = 0.5 + add
        else:
            c3 = _clip(c3 + add, 0.05, 0.95)
        notes3.append(f"Asian {asian_align}")

    parts = [x for x in (c1, c2) if x is not None]
    blend = sum(parts) / len(parts) if parts else None
    return {
        "ready": bool(parts),
        "combo1_tactics": None if c1 is None else round(c1, 3),
        "combo2_injuries": None if c2 is None else round(c2, 3),
        "combo3_value": None if c3 is None else round(c3, 3),
        "blend": None if blend is None else round(blend, 3),
        "notes": {
            "combo1": notes1,
            "combo2": notes2,
            "combo3": notes3,
        },
    }


def adjust_ml_probs(
    p_home: float | None,
    p_draw: float | None,
    p_away: float | None,
    *,
    inj_home: float,
    inj_away: float,
    tac_edge: float,
) -> dict[str, Any] | None:
    """P' = P_ML × (1 − infortuni) × (1 + tattica), poi rinormalizza. Solo display, non EV."""
    if p_home is None or p_draw is None or p_away is None:
        return None
    h = max(0.02, float(p_home) * (1.0 - min(0.35, inj_home)) * (1.0 + _clip(tac_edge, -0.12, 0.12)))
    a = max(0.02, float(p_away) * (1.0 - min(0.35, inj_away)) * (1.0 - _clip(tac_edge, -0.12, 0.12)))
    d = max(0.08, float(p_draw))
    s = h + d + a
    return {"home": round(h / s, 4), "draw": round(d / s, 4), "away": round(a / s, 4)}


def build_calendar_index() -> dict[str, list[pd.Timestamp]]:
    """Date partita per squadra (storico FD + calendario scaricato)."""
    idx: dict[str, list[pd.Timestamp]] = defaultdict(list)
    frames: list[pd.DataFrame] = []
    try:
        from modules.data_update.parse import load_historical

        hist = load_historical(min_date="2025-07-01")
        if hist is not None and not hist.empty:
            frames.append(hist[["date", "home_team", "away_team"]])
    except Exception:
        pass
    try:
        from modules.data_update.parse import load_fixtures

        fx = load_fixtures()
        if fx is not None and not fx.empty:
            frames.append(fx[["date", "home_team", "away_team"]])
    except Exception:
        pass
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])
    for _, row in df.iterrows():
        d = pd.Timestamp(row["date"]).normalize()
        for raw in (row["home_team"], row["away_team"]):
            name = resolve_known_team(str(raw)) or str(raw).strip()
            if name:
                idx[name].append(d)
                idx[str(raw).strip()].append(d)
    return {k: sorted(set(v)) for k, v in idx.items()}


def match_tactics(
    home: str,
    away: str,
    kickoff: Any,
    cal: dict[str, list[pd.Timestamp]] | None,
    fb_home: dict[str, Any] | None,
    fb_away: dict[str, Any] | None,
    *,
    country: str | None = None,
    league: str | None = None,
    sofa_home: dict[str, Any] | None = None,
    sofa_away: dict[str, Any] | None = None,
    ml: dict[str, Any] | None = None,
    value_norm: float | None = None,
    asian_align: str | None = None,
) -> dict[str, Any]:
    ko = pd.to_datetime(kickoff, errors="coerce")
    if pd.isna(ko):
        ko = pd.Timestamp.now()
    cal = cal or {}
    h_name = resolve_known_team(home) or home
    a_name = resolve_known_team(away) or away
    style = style_matchup(fb_home, fb_away)
    fat = fatigue_matchup(
        cal.get(h_name) or cal.get(home) or [],
        cal.get(a_name) or cal.get(away) or [],
        ko,
        home_country=country,
        away_country=country,
        home_league=league,
        away_league=league,
    )
    try:
        from modules.data_update.whoscored_context import lookup_whoscored_match

        ws = lookup_whoscored_match(h_name, a_name, ko)
    except Exception:
        ws = {"home": [], "away": [], "style_home": None, "style_away": None}
    injuries = injury_weights(ws.get("home") or [], ws.get("away") or [], home=h_name, away=a_name)
    combos = combo_scores(
        style=style,
        fatigue=fat,
        injuries=injuries,
        sofa_home=sofa_home,
        sofa_away=sofa_away,
        value_norm=value_norm,
        asian_align=asian_align,
        ws_style_home=ws.get("style_home"),
        ws_style_away=ws.get("style_away"),
    )
    ml = ml or {}
    p_adj = adjust_ml_probs(
        ml.get("home_win"),
        ml.get("draw"),
        ml.get("away_win"),
        inj_home=float(injuries.get("weight_home") or 0),
        inj_away=float(injuries.get("weight_away") or 0),
        tac_edge=float(style.get("edge_home") or 0) + float(fat.get("edge_home") or 0),
    )
    return {
        "style": style,
        "fatigue": fat,
        "absences": injuries,
        "whoscored": ws,
        "combos": combos,
        "p_tactical": p_adj,
        "edge_home": round(
            _clip(float(style.get("edge_home") or 0) + float(fat.get("edge_home") or 0) + float(injuries.get("edge_home") or 0)),
            3,
        ),
    }
