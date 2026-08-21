"""Valore rispetto alla quota: de-vig, calibrazione, edge vs sharp, realizzazione storica."""

from __future__ import annotations

from math import floor
from typing import Any

from modules.calibration.config import load_calibration, prob_bin_factor

REAL_SOURCES = {"book", "asianbetsoccer", "pinnacle", "betfair"}


def _clamp_score(value: float) -> int:
    return int(max(1, min(10, floor(value + 0.5))))


def is_real_odds(source: str | None, odds: float | None) -> bool:
    if odds is None or float(odds) <= 1.01:
        return False
    if not source:
        return False
    raw = str(source).strip().lower()
    if raw.startswith("stimata"):
        return False
    return raw in REAL_SOURCES or raw == "book"


def devig_multiplicative(odds_map: dict[str, float | None]) -> dict[str, float]:
    implied: dict[str, float] = {}
    for key, odd in odds_map.items():
        if odd is None:
            continue
        try:
            val = float(odd)
        except (TypeError, ValueError):
            continue
        if val > 1.01:
            implied[key] = 1.0 / val
    total = sum(implied.values())
    if total <= 1.0 or not implied:
        return implied
    return {k: v / total for k, v in implied.items()}


def calibrated_prob(prob: float, *, group: str, cal: dict | None = None) -> tuple[float, float, int]:
    """p × fattore bin, con shrinkage se il campione è piccolo."""
    cal = cal or load_calibration()
    market = "ou25" if group == "ou" else "1x2"
    factor, n = prob_bin_factor(cal, prob, market=market)
    min_n = int(cal.get("min_bin_samples", 30))
    full_n = max(min_n * 2, 80)
    weight = min(1.0, n / full_n) if full_n else 0.0
    adj = 1.0 + weight * (float(factor) - 1.0)
    p_cal = max(0.02, min(0.96, float(prob) * adj))
    return p_cal, float(factor), int(n)


def conservative_prob(
    p_cal: float,
    *,
    mc_prob: float,
    ml_prob: float | None,
    bin_n: int,
    min_bin: int = 30,
) -> float:
    """Banda bassa: sconto per divergenza ML/MC e bin povero."""
    div = 0.0 if ml_prob is None else abs(float(mc_prob) - float(ml_prob))
    haircut = 0.5 * div
    if bin_n < min_bin:
        haircut += 0.02 * (1.0 - bin_n / max(min_bin, 1))
    return max(0.02, min(p_cal, p_cal - haircut))


def realization_factor(league: str | None, group: str | None, cal: dict | None = None) -> float:
    """Quanto dell'EV storico si è realizzato (ROI / EV) su lega e mercato."""
    cal = cal or load_calibration()

    def from_rows(rows: list, key: str, name: str | None) -> float | None:
        if not name or not rows:
            return None
        needle = str(name).strip().lower()
        for row in rows:
            if str(row.get(key) or "").strip().lower() != needle:
                continue
            if row.get("realization") is not None:
                return float(row["realization"])
            mean_ev = row.get("mean_ev")
            roi = row.get("roi")
            if mean_ev is None or roi is None or abs(float(mean_ev)) < 0.005:
                return None
            return max(0.15, min(1.15, float(roi) / float(mean_ev)))
        return None

    mkt_key = "ou" if group == "ou" else "1x2" if group == "1x2" else None
    parts = [
        from_rows(cal.get("by_market") or [], "market", mkt_key),
        from_rows(cal.get("by_league") or [], "league", league),
    ]
    found = [p for p in parts if p is not None]
    if not found:
        return 1.0
    return float(min(found))


# Lega “stretta” (cap più alto) vs alta varianza (cap più basso) sul voto value/composito.
_LOW_VAR_LEAGUES = (
    "brasileiro",
    "brazil",
    "serie a brazil",
    "brasileirão",
    "eredivisie",
    "primeira liga",
    "liga portugal",
    "saudi",
    "allsvenskan",
)
_HIGH_VAR_LEAGUES = (
    "serie b",
    "championship",
    "2. bundesliga",
    "2 bundesliga",
    "ligue 2",
    "segunda",
    "laliga2",
    "eerste divisie",
    "league one",
    "league two",
    "national league",
    "3. liga",
)


def league_cap_adj(league: str | None, cal: dict | None = None) -> int:
    """+1 leghe più strette, −1 alta varianza. Override soft da calibration.by_league."""
    name = str(league or "").strip().lower()
    if not name:
        return 0
    adj = 0
    if any(k in name for k in _HIGH_VAR_LEAGUES):
        adj = -1
    elif any(k in name for k in _LOW_VAR_LEAGUES):
        adj = 1
    cal = cal or load_calibration()
    for row in cal.get("by_league") or []:
        lg = str(row.get("league") or "").strip().lower()
        if not lg or (lg != name and name not in lg and lg not in name):
            continue
        if int(row.get("n") or 0) < 40:
            break
        realiz = row.get("realization")
        if realiz is None:
            break
        r = float(realiz)
        if r < 0.45:
            adj = min(adj, -1)
        elif r > 0.85:
            adj = max(adj, 1)
        break
    return int(adj)


def prob_score_cap(prob: float, *, league: str | None = None, cal: dict | None = None) -> int | None:
    """Cap sul voto da probabilità: <20→4 … <35→7, spostato ±1 per varianza lega."""
    if prob >= 0.35:
        return None
    if prob < 0.20:
        base = 4
    elif prob < 0.25:
        base = 5
    elif prob < 0.30:
        base = 6
    else:
        base = 7
    return max(3, min(9, base + league_cap_adj(league, cal)))


def score_value_from_edge(
    *,
    edge_pp: float | None,
    ev_cons: float | None,
    ev_sharp: float | None,
    realization: float,
    real_odds: bool,
    steam_against: bool,
    prob: float | None = None,
    league: str | None = None,
    cal: dict | None = None,
) -> int | None:
    if not real_odds or edge_pp is None:
        return None
    if edge_pp >= 0:
        raw = 4 + 6 * min(1.0, edge_pp / 0.08)
    else:
        raw = 4 - 3 * min(1.0, -edge_pp / 0.10)
    if ev_cons is not None and ev_cons < 0:
        raw = min(raw, 4)
    if ev_sharp is not None and ev_cons is not None and ev_cons > 0 and ev_sharp < 0.02:
        raw = min(raw, 6)
    if ev_sharp is not None and ev_sharp < 0:
        raw = min(raw, 5)
    if realization < 0.40:
        raw = min(raw, 6)
    elif realization < 0.70:
        raw = min(raw, 8)
    if steam_against:
        raw -= 1.5
    if prob is not None:
        cap = prob_score_cap(float(prob), league=league, cal=cal)
        if cap is not None:
            raw = min(raw, cap)
    return _clamp_score(raw)


def _sharp_odd(code: str, group: str, market_move: dict | None, book_odd: float | None, odds_from_asian: bool) -> float | None:
    if market_move:
        key = None
        if code == "1":
            key = "odd_1"
        elif code == "X":
            key = "odd_x"
        elif code == "2":
            key = "odd_2"
        elif group == "ou" or (code.startswith("O") and "GOL" not in code):
            key = "odd_over"
        elif code.startswith("U"):
            key = "odd_under"
        if key:
            raw = market_move.get(key)
            try:
                if raw is not None and float(raw) > 1.01:
                    return float(raw)
            except (TypeError, ValueError):
                pass
    if odds_from_asian and book_odd and float(book_odd) > 1.01:
        return float(book_odd)
    return None


def _sibling_implied(code: str, group: str, odds: dict[str, float | None], overround: float) -> float | None:
    book = None
    if code in {"1", "X", "2"}:
        fair = devig_multiplicative(
            {
                "1": odds.get("1") or odds.get("home"),
                "X": odds.get("X") or odds.get("draw") or odds.get("x"),
                "2": odds.get("2") or odds.get("away"),
            }
        )
        return fair.get(code)
    if group == "ou" and len(code) >= 2 and code[0] in {"O", "U"}:
        line = code[1:]
        pair = {f"O{line}": odds.get(f"over_{line}"), f"U{line}": odds.get(f"under_{line}")}
        fair = devig_multiplicative(pair)
        return fair.get(code)
    if group == "btts":
        fair = devig_multiplicative({"GOL": odds.get("btts_yes") or odds.get("gol"), "NOGOL": odds.get("btts_no") or odds.get("nogo")})
        return fair.get(code)
    odd = None
    for key in (code, code.lower()):
        if key in odds and odds[key]:
            odd = odds[key]
            break
    if odd and float(odd) > 1.01 and overround > 1:
        return (1.0 / float(odd)) / overround
    return None


def enrich_value(
    market: dict[str, Any],
    *,
    odds: dict[str, float | None],
    overround: float,
    league: str | None = None,
    market_move: dict | None = None,
    odds_from_asian: bool = False,
    cal: dict | None = None,
) -> dict[str, Any]:
    """Aggiunge p di mercato, EV conservativo, edge vs sharp e voto value."""
    cal = cal or load_calibration()
    out = dict(market)
    prob = float(out.get("probability") or 0)
    group = out.get("group") or "1x2"
    code = str(out.get("code") or "")
    book_odd = out.get("odds")
    real = is_real_odds(out.get("odds_source"), book_odd)
    out["odds_real"] = real

    p_cal, factor, bin_n = calibrated_prob(prob, group=group, cal=cal)
    p_cons = conservative_prob(
        p_cal,
        mc_prob=prob,
        ml_prob=out.get("model_probability"),
        bin_n=bin_n,
        min_bin=int(cal.get("min_bin_samples", 30)),
    )
    out["p_cal"] = round(p_cal, 4)
    out["p_cons"] = round(p_cons, 4)
    out["cal_factor"] = round(factor, 4)
    out["cal_n"] = bin_n

    implied_raw = None
    if book_odd and float(book_odd) > 1.01:
        implied_raw = 1.0 / float(book_odd)
    out["implied_prob"] = None if implied_raw is None else round(implied_raw, 4)

    p_market = _sibling_implied(code, group, odds, overround) if real else None
    if p_market is None and real and implied_raw is not None and overround > 1:
        p_market = implied_raw / overround
    out["p_market"] = None if p_market is None else round(float(p_market), 4)

    edge_pp = None
    if p_market is not None:
        edge_pp = p_cons - float(p_market)
    out["edge_pp"] = None if edge_pp is None else round(edge_pp, 4)

    ev_raw = out.get("ev")
    ev_cons = None
    if real and book_odd and float(book_odd) > 1.01:
        ev_cons = p_cons * float(book_odd) - 1.0
    out["ev_raw"] = ev_raw
    out["ev_cons"] = None if ev_cons is None else round(float(ev_cons), 4)

    sharp = None
    ev_sharp = None
    if real:
        sharp = _sharp_odd(code, group, market_move, float(book_odd) if book_odd else None, odds_from_asian)
        if sharp and sharp > 1.01:
            ev_sharp = p_cons * sharp - 1.0
    out["odds_sharp"] = None if not sharp else round(float(sharp), 2)
    out["ev_sharp"] = None if ev_sharp is None else round(float(ev_sharp), 4)

    realiz = realization_factor(league, group, cal)
    out["realization"] = round(realiz, 4)
    ev_adj = None
    if ev_cons is not None:
        ev_adj = float(ev_cons) * realiz
    drop = None
    if market_move:
        if code == "1":
            drop = market_move.get("drop_1")
        elif code == "X":
            drop = market_move.get("drop_x")
        elif code == "2":
            drop = market_move.get("drop_2")
        elif group == "ou" or (code.startswith("O") and "GOL" not in code):
            drop = market_move.get("drop_over")
        elif code.startswith("U"):
            drop = market_move.get("drop_under")
    steam_against = drop is not None and float(drop) <= -2.0
    if steam_against and ev_adj is not None:
        ev_adj *= 0.7
        if edge_pp is not None:
            edge_pp *= 0.7
            out["edge_pp"] = round(edge_pp, 4)
    out["ev_adj"] = None if ev_adj is None else round(float(ev_adj), 4)
    out["steam_against"] = steam_against

    out["score_value"] = score_value_from_edge(
        edge_pp=out.get("edge_pp"),
        ev_cons=out.get("ev_cons"),
        ev_sharp=out.get("ev_sharp"),
        realization=realiz,
        real_odds=real,
        steam_against=steam_against,
        prob=p_cons,
        league=league,
        cal=cal,
    )
    out["league"] = league or out.get("league")
    if not real:
        out["value_note"] = "quota ipotetica: nessun voto value"
    elif edge_pp is not None:
        out["value_note"] = (
            f"modello {p_cons:.0%} vs mercato {float(p_market):.0%} ({edge_pp:+.1%} pp)"
        )
    else:
        out["value_note"] = None
    return out


PLAY_VALUE_KEYS = (
    "p_cal",
    "p_cons",
    "p_market",
    "edge_pp",
    "ev_raw",
    "ev_cons",
    "ev_sharp",
    "ev_adj",
    "odds_sharp",
    "odds_real",
    "realization",
    "cal_factor",
    "cal_n",
    "value_note",
    "steam_against",
    "implied_prob",
)
