"""Pronostici sul calendario in arrivo, con quote scaricate dal sito."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from modules.advisor.advise import advise, advise_uncovered
from modules.data_update.asian_odds import asian_to_advisor_odds, find_asian_odds, summarize_moves
from modules.data_update.cups import known_team_index, resolve_known_team
from modules.data_update.fbref_context import load_fbref_team_index, lookup_team_context
from modules.data_update.understat_context import load_understat_team_index, lookup_understat_team
from modules.data_update.statsbomb_context import load_statsbomb_team_index, lookup_statsbomb_team
from modules.data_update.sofascore_context import load_sofascore_team_index, lookup_sofascore_team
from modules.data_update.fotmob_context import (
    load_fotmob_matches,
    load_fotmob_team_index,
    load_fotmob_xg_index,
    lookup_fotmob_match,
    lookup_fotmob_team,
    lookup_fotmob_xg,
)
from modules.data_update.parse import load_fixtures
from modules.data_update.venues import update_home_venues
from modules.montecarlo import MonteCarloSimulator
from modules.predictor import MatchPredictor
from modules.predictor.predict import context_xg


def _decimal_odds(value: Any) -> float | None:
    """Quota decimale; ignora bool (odds_real è un flag True/False)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if 1.01 <= x <= 100.0:
        return x
    return None


def _pick_quota_from_play(play: dict[str, Any]) -> float | None:
    """Quota book del pick: play['odds'] (non odds_real, che è solo un flag)."""
    return _decimal_odds(play.get("odds")) or _decimal_odds(play.get("fair_odds"))

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "upcoming_predictions.json"


def _odd(fx: pd.Series, col: str) -> float | None:
    val = fx.get(col)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    return num if num > 1.0 else None


def _fx_text(fx: pd.Series, col: str) -> str:
    val = fx.get(col)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def _fx_bool(fx: pd.Series, col: str) -> bool:
    val = fx.get(col) if col in fx.index else None
    if val is True:
        return True
    if val is False or val is None:
        return False
    if isinstance(val, float) and pd.isna(val):
        return False
    return str(val).strip().lower() in {"1", "true", "yes", "y", "t"}


def _venue_fields(fx: pd.Series) -> dict:
    return {
        "venue": _fx_text(fx, "venue"),
        "venue_city": _fx_text(fx, "venue_city"),
        "venue_neutral": _fx_bool(fx, "venue_neutral"),
    }


_BOOK_FILL = (
    ("odd_home", "1"),
    ("odd_draw", "X"),
    ("odd_away", "2"),
    ("odd_over_25", "over_2.5"),
    ("odd_under_25", "under_2.5"),
)


def _fill_book_odds(
    odds: dict,
    odds_source: str,
    match: dict | None,
    source_name: str,
) -> tuple[dict, str]:
    if not match:
        return odds, odds_source
    filled = False
    for src_k, dst_k in _BOOK_FILL:
        if odds.get(dst_k) is None and match.get(src_k) is not None:
            odds[dst_k] = match[src_k]
            filled = True
    if filled and odds_source not in {"asianbetsoccer"}:
        odds_source = source_name
    return odds, odds_source


def _xg_pair(
    us_row: dict | None,
    fb_row: dict | None,
    fm_row: dict | None = None,
) -> tuple[float, float] | None:
    return context_xg(us_row, fb_row, fm_row)


def _collect_match_odds(
    fx: pd.Series,
    home: str,
    away: str,
    pinnacle_events: list[dict],
    betfair_events: list[dict],
) -> tuple[dict, str, dict | None]:
    odds = {
        "1": _odd(fx, "odd_home"),
        "X": _odd(fx, "odd_draw"),
        "2": _odd(fx, "odd_away"),
        "over_2.5": _odd(fx, "odd_over_25"),
        "under_2.5": _odd(fx, "odd_under_25"),
    }
    src = str(fx.get("source") or "")
    if src.startswith("fixtures-cups-asian") or "asian" in src:
        odds_source = "asianbetsoccer"
    elif src.startswith("fd.org") or src.startswith("fixtures-cups"):
        odds_source = "football-data.org"
    else:
        odds_source = src or "football-data.co.uk"
    day = fx["date"].strftime("%Y-%m-%d")
    market_move = None
    asian = find_asian_odds(home, away, day)
    if asian:
        ao = asian_to_advisor_odds(asian)
        for key, val in ao.items():
            if val is not None:
                odds[key] = val
        odds_source = "asianbetsoccer"
        market_move = summarize_moves(asian)
    pinnacle_match = None
    if pinnacle_events:
        try:
            from modules.data_update.odds_api import lookup_pinnacle

            pinnacle_match = lookup_pinnacle(
                home, away, events=pinnacle_events, kickoff_date=day
            )
        except Exception:
            pass
    odds, odds_source = _fill_book_odds(odds, odds_source, pinnacle_match, "pinnacle")
    bf_match = None
    if betfair_events:
        try:
            from modules.data_update.betfair import lookup_betfair

            bf_match = lookup_betfair(home, away, events=betfair_events, kickoff_date=day)
        except Exception:
            pass
    odds, odds_source = _fill_book_odds(odds, odds_source, bf_match, "betfair")
    return odds, odds_source, market_move


def _val_fields(play: dict | None, extra: dict | None = None, weather: dict | None = None) -> dict:
    val = (play or {}).get("validation") or {}
    if not val and extra:
        val = extra.get("validation") or (extra.get("quadro") or {}).get("validation") or {}
    if weather:
        val = dict(val) if val else {}
        val["weather"] = weather
    venue = val.get("venue") or {}
    wx = weather or val.get("weather") or {}
    return {
        "validation": val or None,
        "validation_summary": val.get("summary"),
        "validation_delta": val.get("delta_unified"),
        "venue_flag": venue.get("flag"),
        "venue_name": venue.get("venue") or "",
        "weather_flag": wx.get("flag"),
    }


def _pro_fields(play: dict | None, advice: dict | None = None) -> dict:
    """Campi Score/Confidence/Risk/Priorità/Bet-rec da advise."""
    play = play or {}
    adv = advice or {}
    ms = play.get("match_scores") or adv.get("match_scores") or {}
    br = play.get("bet_rec") or adv.get("bet_rec") or ms.get("bet_rec") or {}
    prim = (br or {}).get("primary") or {}
    return {
        "score_100": play.get("score_100") if play.get("score_100") is not None else adv.get("score_100"),
        "confidence_100": play.get("confidence_100")
        if play.get("confidence_100") is not None
        else adv.get("confidence_100"),
        "risk_100": play.get("risk_100") if play.get("risk_100") is not None else adv.get("risk_100"),
        "priority_100": play.get("priority_100")
        if play.get("priority_100") is not None
        else adv.get("priority_100"),
        "score_band": play.get("score_band") or adv.get("score_band"),
        "play_rank": play.get("play_rank") if play.get("play_rank") is not None else adv.get("play_rank"),
        "bet_rec_label": None
        if not prim
        else f"{prim.get('label') or prim.get('group') or ''} {prim.get('code') or ''}".strip(),
        "bet_rec": br or None,
        "match_scores": ms or None,
        "coverage_n": None if not ms.get("coverage") else ms["coverage"].get("n_present"),
        "fallback_used": bool((ms.get("coverage") or {}).get("fallback_ok") and (adv.get("quadro") or {}).get("fallback")),
    }


def _match_key(date: str, home: str, away: str) -> str:
    return f"{str(date)[:10]}|{home}|{away}"


def _load_prev_upcoming() -> dict[str, dict]:
    if not OUT.exists():
        return {}
    try:
        rows = json.loads(OUT.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, dict] = {}
    for r in rows or []:
        try:
            out[_match_key(str(r.get("date") or ""), str(r.get("home") or ""), str(r.get("away") or ""))] = r
        except Exception:
            continue
    return out


def _model_newer_than_upcoming() -> bool:
    model = ROOT / "data" / "models" / "best_model.joblib"
    try:
        if not model.is_file() or not OUT.is_file():
            return False
        return model.stat().st_mtime > OUT.stat().st_mtime
    except OSError:
        return False


def _load_odds_caches() -> tuple[list[dict], list[dict]]:
    pinn: list[dict] = []
    bf: list[dict] = []
    try:
        from modules.data_update.odds_api import load_pinnacle_cache

        pinn = load_pinnacle_cache()
    except Exception:
        pass
    try:
        from modules.data_update.betfair import load_betfair_cache

        bf = load_betfair_cache()
    except Exception:
        pass
    return pinn, bf


def _fx_proxy_from_row(row: dict) -> pd.Series:
    odds = row.get("odds") if isinstance(row.get("odds"), dict) else {}
    day = str(row.get("date") or "")[:10]
    return pd.Series(
        {
            "date": pd.Timestamp(day),
            "home_team": row.get("home"),
            "away_team": row.get("away"),
            "odd_home": odds.get("1", row.get("odd_1")),
            "odd_draw": odds.get("X", row.get("odd_x")),
            "odd_away": odds.get("2", row.get("odd_2")),
            "odd_over_25": odds.get("over_2.5", row.get("odd_over_25")),
            "odd_under_25": odds.get("under_2.5", row.get("odd_under_25")),
            "source": row.get("odds_source") or "",
            "time": row.get("time") or "",
            "country": row.get("country") or "",
            "league": row.get("league") or "",
        }
    )


def _slim_markets_from_advice(advice: dict) -> list[dict]:
    return [
        {
            "code": m["code"],
            "name": m["name"],
            "group": m["group"],
            "probability": m["probability"],
            "odds": m["odds"],
            "fair_odds": m["fair_odds"],
            "ev": m.get("ev_cons") if m.get("ev_cons") is not None else m.get("ev"),
            "ev_cons": m.get("ev_cons"),
            "ev_sharp": m.get("ev_sharp"),
            "edge_pp": m.get("edge_pp"),
            "p_cons": m.get("p_cons"),
            "p_market": m.get("p_market"),
            "odds_real": m.get("odds_real"),
            "value_note": m.get("value_note"),
            "score_prob": m["score_prob"],
            "score_value": m["score_value"],
            "score": m.get("score"),
            "odds_source": m.get("odds_source"),
        }
        for m in advice.get("all_markets") or []
    ]


def _row_after_covered_advise(
    *,
    base: dict,
    advice: dict,
    odds: dict,
    odds_source: str,
    market_move: dict | None,
    prediction: dict,
    weather: dict | None = None,
) -> dict:
    """Ricalcola EV/Kelly/voto/pick da quote nuove; conserva prediction + MC."""
    play = advice["play"]
    alt = advice.get("play_alt")
    pick_quota = _pick_quota_from_play(play)
    spread_score = None if not market_move else market_move.get("spread_score")
    ah_line = None
    if market_move and market_move.get("ah_open") is not None and market_move.get("ah_curr") is not None:
        ah_line = f"{market_move['ah_open']}->{market_move['ah_curr']}"
    mc = (prediction or {}).get("montecarlo") or {}
    row = dict(base)
    row.update(
        {
            "odd_1": odds.get("1"),
            "odd_x": odds.get("X"),
            "odd_2": odds.get("2"),
            "odd_over_25": odds.get("over_2.5"),
            "odd_under_25": odds.get("under_2.5"),
            "odds_source": odds_source,
            "odds": odds,
            "market_move": market_move,
            "market_align": None if not advice.get("market_align") else advice["market_align"].get("label"),
            "market_note": None
            if not market_move
            else market_move.get("movement_comment") or market_move.get("movement_summary"),
            "movement_level": None if not market_move else market_move.get("movement_level"),
            "movement_summary": None if not market_move else market_move.get("movement_summary"),
            "movement_comment": None if not market_move else market_move.get("movement_comment"),
            "steam_1x2": None if not market_move else market_move.get("steam_1x2"),
            "steam_ah": None if not market_move else market_move.get("steam_ah"),
            "steam_ou": None if not market_move else market_move.get("steam_ou"),
            "drop_1": None if not market_move else market_move.get("drop_1"),
            "drop_x": None if not market_move else market_move.get("drop_x"),
            "drop_2": None if not market_move else market_move.get("drop_2"),
            "drop_over": None if not market_move else market_move.get("drop_over"),
            "drop_under": None if not market_move else market_move.get("drop_under"),
            "spread_score": spread_score,
            "line_move": None if not market_move else market_move.get("line_move"),
            "ah_line": ah_line,
            "value_edge": play.get("edge_pp"),
            "edge_pp": play.get("edge_pp"),
            "p_cons": play.get("p_cons"),
            "p_market": play.get("p_market"),
            "ev_cons": play.get("ev_cons"),
            "ev_sharp": play.get("ev_sharp"),
            "odds_real": play.get("odds_real"),
            "value_note": play.get("value_note"),
            "quadro": advice.get("quadro"),
            "quadro_consenso": None if not advice.get("quadro") else advice["quadro"].get("consenso"),
            "quadro_n": None
            if not advice.get("quadro")
            else f"{advice['quadro'].get('agree_n')}/{advice['quadro'].get('votes_n')}",
            "quadro_summary": None if not advice.get("quadro") else advice["quadro"].get("summary"),
            "fbref_ctx": None if not advice.get("quadro") else advice["quadro"].get("fbref_summary"),
            "pick": play["code"],
            "pick_name": play["name"],
            "pick_group": play.get("group"),
            "action": play.get("action") or "gioca",
            "no_bet_reasons": play.get("no_bet_reasons") or [],
            "score": play["score"],
            "score_unified": play.get("score_unified"),
            "meta_label": None if not play.get("meta_analysis") else play["meta_analysis"].get("label"),
            "meta_note": None if not play.get("meta_analysis") else play["meta_analysis"].get("note"),
            "kelly_quarter": play.get("kelly_quarter"),
            "clv": play.get("clv"),
            "quota_pick": pick_quota,
            "score_reason_1": advice.get("score_reason_1"),
            "score_reason_2": advice.get("score_reason_2"),
            "kind": play["kind"],
            "probability": play["probability"],
            "ev": play["ev"],
            "fair_odds": play["fair_odds"],
            "tipster_consensus": None
            if not (play.get("tipster") or {}).get("consensus")
            else play["tipster"]["consensus"],
            "tipster_label": None if not play.get("tipster") else play["tipster"].get("label"),
            "tipster_agree": None if not play.get("tipster") else play["tipster"].get("agree"),
            "tipster_n": None if not play.get("tipster") else play["tipster"].get("n_sources"),
            "tipster": play.get("tipster"),
            "alt_pick": None if not alt else alt["code"],
            "alt_name": None if not alt else alt["name"],
            "alt_score": None if not alt else alt["score"],
            "p_home": mc.get("home_win", row.get("p_home")),
            "p_draw": mc.get("draw", row.get("p_draw")),
            "p_away": mc.get("away_win", row.get("p_away")),
            "p_over_25": mc.get("over_2.5", row.get("p_over_25")),
            "p_btts": mc.get("btts", row.get("p_btts")),
            "markets": _slim_markets_from_advice(advice),
            "prediction": prediction,
            "data_signal": prediction.get("data_signal"),
            "source_agreement": advice.get("source_agreement") or play.get("source_agreement"),
            "prob_intervals": advice.get("prob_intervals") or play.get("prob_intervals"),
            "residual_ev": advice.get("residual_ev") or play.get("residual_ev"),
            "conformal_intervals": prediction.get("conformal_intervals") or play.get("conformal_intervals"),
            **_pro_fields(play, advice),
            **_val_fields(play, advice, weather=weather),
        }
    )
    return row


def _patch_uncovered_odds(
    *,
    base: dict,
    fx: pd.Series,
    home: str,
    away: str,
    odds: dict,
    odds_source: str,
    market_move: dict | None,
) -> dict:
    """Aggiorna quote su uno stub N/D già in calendario, senza rifare il quadro."""
    spread_score = None if not market_move else market_move.get("spread_score")
    ah_line = None
    if (
        market_move
        and market_move.get("ah_open") is not None
        and market_move.get("ah_curr") is not None
    ):
        ah_line = f"{market_move['ah_open']}->{market_move['ah_curr']}"
    row = dict(base)
    row.update(
        {
            "date": fx["date"].strftime("%Y-%m-%d"),
            "time": str(fx.get("time") or row.get("time") or ""),
            "country": str(fx.get("country") or row.get("country") or ""),
            "league": str(fx.get("league") or row.get("league") or ""),
            "home": home,
            "away": away,
            "odd_1": odds.get("1"),
            "odd_x": odds.get("X"),
            "odd_2": odds.get("2"),
            "odd_over_25": odds.get("over_2.5"),
            "odd_under_25": odds.get("under_2.5"),
            "odds_source": odds_source,
            "odds": odds,
            "market_move": market_move,
            "market_note": None
            if not market_move
            else market_move.get("movement_comment") or market_move.get("movement_summary"),
            "movement_level": None if not market_move else market_move.get("movement_level"),
            "movement_summary": None if not market_move else market_move.get("movement_summary"),
            "movement_comment": None if not market_move else market_move.get("movement_comment"),
            "drop_1": None if not market_move else market_move.get("drop_1"),
            "drop_x": None if not market_move else market_move.get("drop_x"),
            "drop_2": None if not market_move else market_move.get("drop_2"),
            "spread_score": spread_score,
            "line_move": None if not market_move else market_move.get("line_move"),
            "ah_line": ah_line,
            "odds_real": bool(odds.get("1") and odds.get("X") and odds.get("2")),
        }
    )
    return row


def refresh_upcoming_odds(*, on_progress=None, archive: bool = True) -> dict:
    """Ricalcola solo quote→EV/Kelly/voto sulle predizioni già in JSON.

    Non rifà ML né Monte Carlo: stessa analisi probabilità, quote aggiornate
    (Asian/Pinnacle/Betfair cache). Ideale dopo Scarica Betfair/Pinnacle/Asian.
    """
    from modules.progress_report import emit

    if not OUT.exists():
        return {"ok": False, "error": "Nessun calendario: lancia prima Solo quote o Aggiorna dati", "n_upcoming": 0}
    try:
        rows = json.loads(OUT.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "error": str(exc), "n_upcoming": 0}
    if not rows:
        return {"ok": False, "error": "Calendario vuoto", "n_upcoming": 0}

    pinn, bf = _load_odds_caches()
    out: list[dict] = []
    n_ok = 0
    n_skip = 0
    n = len(rows)
    emit(on_progress, 0.02, f"Ricalcolo value su {n} partite (senza ML/MC)…")
    for i, row in enumerate(rows):
        home = str(row.get("home") or "")
        away = str(row.get("away") or "")
        day = str(row.get("date") or "")[:10]
        pred = row.get("prediction") if isinstance(row.get("prediction"), dict) else None
        if not home or not away or not day or not pred:
            out.append(row)
            n_skip += 1
            continue
        fx = _fx_proxy_from_row(row)
        odds, odds_source, market_move = _collect_match_odds(fx, home, away, pinn, bf)
        wx = (pred.get("weather") if isinstance(pred.get("weather"), dict) else None) or (
            (row.get("validation") or {}).get("weather") if isinstance(row.get("validation"), dict) else None
        )
        mc = pred.get("montecarlo") if isinstance(pred.get("montecarlo"), dict) else {}
        covered = mc.get("home_win") is not None or (
            isinstance(pred.get("model_probabilities"), dict) and pred["model_probabilities"].get("home_win") is not None
        )
        try:
            if covered:
                advice = advise(
                    pred,
                    odds,
                    market_move=market_move,
                    odds_from_asian=(odds_source == "asianbetsoccer"),
                    league=str(row.get("league") or "") or None,
                    lazy_secondary=True,
                )
                out.append(
                    _row_after_covered_advise(
                        base=row,
                        advice=advice,
                        odds=odds,
                        odds_source=odds_source,
                        market_move=market_move,
                        prediction=pred,
                        weather=wx,
                    )
                )
            else:
                uncovered = advise_uncovered(
                    home,
                    away,
                    odds=odds,
                    market_move=market_move,
                    prediction=pred,
                )
                play = uncovered["play"]
                quadro = uncovered.get("quadro") or {}
                meta = uncovered.get("meta_analysis") or {}
                spread_score = None if not market_move else market_move.get("spread_score")
                ah_line = None
                if market_move and market_move.get("ah_open") is not None and market_move.get("ah_curr") is not None:
                    ah_line = f"{market_move['ah_open']}->{market_move['ah_curr']}"
                updated = dict(row)
                updated.update(
                    {
                        "odd_1": odds.get("1"),
                        "odd_x": odds.get("X"),
                        "odd_2": odds.get("2"),
                        "odd_over_25": odds.get("over_2.5"),
                        "odd_under_25": odds.get("under_2.5"),
                        "odds_source": odds_source,
                        "odds": odds,
                        "market_move": market_move,
                        "market_align": None
                        if not uncovered.get("market_align")
                        else uncovered["market_align"].get("label"),
                        "market_note": None
                        if not market_move
                        else market_move.get("movement_comment") or market_move.get("movement_summary"),
                        "movement_level": None if not market_move else market_move.get("movement_level"),
                        "movement_summary": None if not market_move else market_move.get("movement_summary"),
                        "movement_comment": None if not market_move else market_move.get("movement_comment"),
                        "drop_1": None if not market_move else market_move.get("drop_1"),
                        "drop_x": None if not market_move else market_move.get("drop_x"),
                        "drop_2": None if not market_move else market_move.get("drop_2"),
                        "spread_score": spread_score,
                        "line_move": None if not market_move else market_move.get("line_move"),
                        "ah_line": ah_line,
                        "quadro": quadro,
                        "quadro_consenso": quadro.get("consenso"),
                        "quadro_n": None
                        if not quadro.get("votes_n")
                        else f"{quadro.get('agree_n')}/{quadro.get('votes_n')}",
                        "quadro_summary": quadro.get("summary"),
                        "pick": play.get("code") or "—",
                        "pick_name": play.get("name") or "nessun pick",
                        "action": play.get("action") or row.get("action") or "n/d",
                        "score_unified": play.get("score_unified"),
                        "meta_label": meta.get("label"),
                        "meta_note": meta.get("note"),
                        "odds_real": bool(odds.get("1") and odds.get("X") and odds.get("2")),
                        "score_reason_1": uncovered.get("score_reason_1"),
                        "score_reason_2": uncovered.get("score_reason_2"),
                        "tipster": play.get("tipster") or uncovered.get("tipster"),
                        "prediction": pred,
                        **_pro_fields(play, uncovered),
                        **_val_fields(play, uncovered, weather=wx),
                    }
                )
                out.append(updated)
            n_ok += 1
        except Exception:
            out.append(row)
            n_skip += 1
        if n and (i % 50 == 0 or i + 1 == n):
            emit(on_progress, 0.05 + 0.9 * ((i + 1) / n), f"Value {i + 1}/{n}…")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    hist_info: dict = {}
    if archive:
        try:
            from modules.data_update.history import archive_upcoming, settle_pending

            hist_info = archive_upcoming(out)
            settled = settle_pending(learn=False)
            hist_info["settled_now"] = settled.get("settled", 0)
            hist_info["n_settled"] = settled.get("n_settled", 0)
        except Exception as exc:
            hist_info = {"archive_error": str(exc)}
    emit(on_progress, 1.0, f"OK · {n_ok} value aggiornati · {n_skip} saltati")
    print(f"refresh odds: {n_ok} aggiornati, {n_skip} saltati, {len(out)} in calendario")
    alert_info: dict = {}
    try:
        from modules.notify import dispatch_alerts

        alert_info = dispatch_alerts(out)
        if alert_info.get("n_sent"):
            print(
                f"telegram avvisi: gioca {alert_info.get('n_new_gioca', 0)} nuovi, "
                f"watch {alert_info.get('n_new_watch', 0)} nuovi, "
                f"spread {alert_info.get('n_new_spread', 0)} nuovi, "
                f"inviati {alert_info.get('n_sent', 0)}"
            )
    except Exception as exc:
        alert_info = {"telegram_error": str(exc)}
        print(f"skip telegram avvisi: {exc}")
    return {
        "ok": True,
        "n_upcoming": len(out),
        "n_refreshed": n_ok,
        "n_skipped": n_skip,
        "light": True,
        **hist_info,
        **{k: v for k, v in alert_info.items() if k != "status"},
    }


def _emit_cal_progress(
    done: int,
    total: int,
    on_progress: Callable[[float, str], None] | None,
    *,
    every: int = 50,
) -> None:
    """Stampa ogni ``every`` partite (e all'inizio/fine) per la barra Streamlit."""
    if total <= 0:
        return
    if done != 0 and done % every != 0 and done != total:
        return
    from modules.progress_report import emit

    msg = f"calendario {done}/{total}"
    frac = done / total
    emit(on_progress, frac, msg)
    print(f"{msg}…", flush=True)


def build_upcoming(
    n_sims: int = 4000,
    *,
    reuse_predictions: bool = True,
    on_progress: Callable[[float, str], None] | None = None,
) -> list[dict]:
    fixtures = load_fixtures()
    if fixtures.empty:
        OUT.write_text("[]", encoding="utf-8")
        return []
    n_fix = len(fixtures)
    try:
        update_home_venues(fixtures)
    except Exception:
        pass

    prev_map = _load_prev_upcoming() if reuse_predictions else {}
    if reuse_predictions and _model_newer_than_upcoming():
        prev_map = {}
        print("build_upcoming: modello più nuovo del calendario → ricalcolo pieno", flush=True)
    _emit_cal_progress(0, n_fix, on_progress, every=1)
    reused = 0

    predictor = MatchPredictor()
    sim = MonteCarloSimulator(n_sims=n_sims)
    team_idx = known_team_index(predictor.last_idx.keys())
    fbref_idx = load_fbref_team_index()
    understat_idx = load_understat_team_index()
    statsbomb_idx = load_statsbomb_team_index()
    sofascore_idx = load_sofascore_team_index()
    fotmob_idx = load_fotmob_team_index()
    fotmob_matches = load_fotmob_matches()
    fotmob_xg = load_fotmob_xg_index()
    fd_side_idx: dict = {}
    fb_match_idx: dict = {}
    lookup_fd_side = None  # type: ignore
    lookup_fbref_match_side = None  # type: ignore
    try:
        from modules.data_update.side_rates import load_fd_side_index, lookup_fd_side as _lookup_fd

        fd_side_idx = load_fd_side_index()
        lookup_fd_side = _lookup_fd
    except Exception:
        pass
    try:
        from modules.data_update.fbref_context import (
            load_fbref_match_side_index,
            lookup_fbref_match_side as _lookup_fb_m,
        )

        fb_match_idx = load_fbref_match_side_index()
        lookup_fbref_match_side = _lookup_fb_m
    except Exception:
        pass
    from modules.advisor.tactics import build_calendar_index, match_tactics

    cal_idx = build_calendar_index()

    # Cache quote esterne: zero chiamate API durante il ciclo
    _pinnacle_events: list[dict] = []
    _betfair_events: list[dict] = []
    try:
        from modules.data_update.odds_api import load_pinnacle_cache
        _pinnacle_events = load_pinnacle_cache()
    except Exception:
        pass
    try:
        from modules.data_update.betfair import load_betfair_cache
        _betfair_events = load_betfair_cache()
    except Exception:
        pass

    from modules.data_update.history import HistoryLookupCache, lookup_history_match
    from modules.data_update.weather import lookup_weather, prefetch_weather

    hist_cache = HistoryLookupCache()
    hist_cache.prefetch()

    wx_items = []
    for _, fx0 in fixtures.iterrows():
        if not resolve_known_team(str(fx0["home_team"]), team_idx) or not resolve_known_team(
            str(fx0["away_team"]), team_idx
        ):
            continue
        city = _fx_text(fx0, "venue_city")
        venue = _fx_text(fx0, "venue")
        try:
            day0 = fx0["date"].strftime("%Y-%m-%d")
        except Exception:
            day0 = str(fx0.get("date") or "")[:10]
        if (city or venue) and day0:
            wx_items.append({"city": city, "date": day0, "venue": venue})
    try:
        wx_idx = prefetch_weather(wx_items)
    except Exception:
        wx_idx = {}

    rows: list[dict] = []
    skipped = 0

    for i, (_, fx) in enumerate(fixtures.iterrows()):
        _emit_cal_progress(i + 1, n_fix, on_progress)
        home_known = resolve_known_team(str(fx["home_team"]), team_idx)
        away_known = resolve_known_team(str(fx["away_team"]), team_idx)
        home = home_known or str(fx["home_team"])
        away = away_known or str(fx["away_team"])
        league = str(fx.get("league") or "") or None
        day = fx["date"].strftime("%Y-%m-%d")
        city = _fx_text(fx, "venue_city")
        venue = _fx_text(fx, "venue")
        wx = (
            lookup_weather(city, day, wx_idx, venue=venue)
            if home_known and away_known and (city or venue)
            else None
        )
        odds, odds_source, market_move = _collect_match_odds(
            fx, home, away, _pinnacle_events, _betfair_events
        )
        prev = prev_map.get(_match_key(day, home, away))
        prev_pred = prev.get("prediction") if isinstance(prev, dict) else None
        if isinstance(prev_pred, dict):
            mc_prev = prev_pred.get("montecarlo") if isinstance(prev_pred.get("montecarlo"), dict) else {}
            can_reuse = mc_prev.get("home_win") is not None or (
                isinstance(prev_pred.get("model_probabilities"), dict)
                and prev_pred["model_probabilities"].get("home_win") is not None
            )
            if can_reuse:
                try:
                    advice = advise(
                        prev_pred,
                        odds,
                        market_move=market_move,
                        odds_from_asian=(odds_source == "asianbetsoccer"),
                        league=league,
                        lazy_secondary=True,
                    )
                    base = dict(prev)
                    base.update(
                        {
                            "date": day,
                            "time": str(fx.get("time") or base.get("time") or ""),
                            "country": str(fx.get("country") or base.get("country") or ""),
                            "league": str(fx.get("league") or base.get("league") or ""),
                            "home": home,
                            "away": away,
                        }
                    )
                    rows.append(
                        _row_after_covered_advise(
                            base=base,
                            advice=advice,
                            odds=odds,
                            odds_source=odds_source,
                            market_move=market_move,
                            prediction=prev_pred,
                            weather=wx,
                        )
                    )
                    reused += 1
                    continue
                except Exception:
                    pass
            # N/D / invalido: riusa lo stub, niente predict/MC/quadro
            elif prev.get("action") in ("n/d", "invalido") or prev.get("skip_reason"):
                try:
                    rows.append(
                        _patch_uncovered_odds(
                            base=dict(prev),
                            fx=fx,
                            home=home,
                            away=away,
                            odds=odds,
                            odds_source=odds_source,
                            market_move=market_move,
                        )
                    )
                    reused += 1
                    continue
                except Exception:
                    pass
        fb_h = lookup_team_context(home, fbref_idx)
        fb_a = lookup_team_context(away, fbref_idx)
        us_h = lookup_understat_team(home, understat_idx)
        us_a = lookup_understat_team(away, understat_idx)
        fm_xg_h = lookup_fotmob_xg(home, fotmob_xg)
        fm_xg_a = lookup_fotmob_xg(away, fotmob_xg)
        hist = lookup_history_match(home, away, league=league, cache=hist_cache)
        try:
            pred = predictor.predict(
                home,
                away,
                kickoff=fx["date"],
                league=league,
                odds=odds,
                ext_xg_home=_xg_pair(us_h, fb_h, fm_xg_h),
                ext_xg_away=_xg_pair(us_a, fb_a, fm_xg_a),
                weather=wx,
            )
        except KeyError as exc:
            skipped += 1
            stub = {
                "match": f"{home} vs {away}",
                "model_probabilities": {},
                "expected_goals": {},
                "features": {},
                "fbref_context": {"home": fb_h, "away": fb_a},
                "understat_context": {"home": us_h, "away": us_a},
                "statsbomb_context": {
                    "home": lookup_statsbomb_team(home, statsbomb_idx),
                    "away": lookup_statsbomb_team(away, statsbomb_idx),
                },
                "sofascore_context": {
                    "home": lookup_sofascore_team(home, sofascore_idx),
                    "away": lookup_sofascore_team(away, sofascore_idx),
                },
                "fotmob_context": {
                    "home": lookup_fotmob_team(home, fotmob_idx),
                    "away": lookup_fotmob_team(away, fotmob_idx),
                    "match": lookup_fotmob_match(home, away, day, fotmob_matches),
                },
                "montecarlo": {},
                "league": str(fx.get("league") or ""),
                "country": str(fx.get("country") or ""),
                "home": home,
                "away": away,
                "weather": wx,
                "history_context": hist,
                **_venue_fields(fx),
            }
            stub["tactical"] = match_tactics(
                home,
                away,
                fx["date"],
                cal_idx,
                stub["fbref_context"]["home"],
                stub["fbref_context"]["away"],
                country=str(fx.get("country") or ""),
                league=str(fx.get("league") or ""),
                sofa_home=stub["sofascore_context"]["home"],
                sofa_away=stub["sofascore_context"]["away"],
            )
            uncovered = advise_uncovered(
                home,
                away,
                odds=odds,
                market_move=market_move,
                prediction=stub,
            )
            play = uncovered["play"]
            quadro = uncovered.get("quadro") or {}
            meta = uncovered.get("meta_analysis") or {}
            spread_score = None if not market_move else market_move.get("spread_score")
            ah_line = None
            if market_move and market_move.get("ah_open") is not None and market_move.get("ah_curr") is not None:
                ah_line = f"{market_move['ah_open']}->{market_move['ah_curr']}"
            rows.append(
                {
                    "date": fx["date"].strftime("%Y-%m-%d"),
                    "time": str(fx.get("time") or ""),
                    "country": str(fx.get("country") or ""),
                    "league": str(fx.get("league") or ""),
                    "home": home,
                    "away": away,
                    "odd_1": odds.get("1"),
                    "odd_x": odds.get("X"),
                    "odd_2": odds.get("2"),
                    "odd_over_25": odds.get("over_2.5"),
                    "odd_under_25": odds.get("under_2.5"),
                    "odds_source": odds_source,
                    "odds": odds,
                    "market_move": market_move,
                    "market_align": None if not uncovered.get("market_align") else uncovered["market_align"].get("label"),
                    "market_note": None if not market_move else market_move.get("movement_comment") or market_move.get("movement_summary"),
                    "movement_level": None if not market_move else market_move.get("movement_level"),
                    "movement_summary": None if not market_move else market_move.get("movement_summary"),
                    "movement_comment": None if not market_move else market_move.get("movement_comment"),
                    "steam_1x2": None if not market_move else market_move.get("steam_1x2"),
                    "steam_ah": None if not market_move else market_move.get("steam_ah"),
                    "steam_ou": None if not market_move else market_move.get("steam_ou"),
                    "drop_1": None if not market_move else market_move.get("drop_1"),
                    "drop_x": None if not market_move else market_move.get("drop_x"),
                    "drop_2": None if not market_move else market_move.get("drop_2"),
                    "spread_score": spread_score,
                    "line_move": None if not market_move else market_move.get("line_move"),
                    "ah_line": ah_line,
                    "quadro": quadro,
                    "quadro_consenso": quadro.get("consenso"),
                    "quadro_n": None if not quadro.get("votes_n") else f"{quadro.get('agree_n')}/{quadro.get('votes_n')}",
                    "quadro_summary": quadro.get("summary"),
                    "fbref_ctx": quadro.get("fbref_summary"),
                    "pick": play.get("code") or "—",
                    "pick_name": play.get("name") or "nessun pick",
                    "pick_group": "1x2",
                    "action": play.get("action") or "n/d",
                    "no_bet_reasons": play.get("no_bet_reasons") or [],
                    "score": None,
                    "score_unified": play.get("score_unified"),
                    "meta_label": meta.get("label"),
                    "meta_note": meta.get("note"),
                    "meta_analysis": meta,
                    "kelly_quarter": None,
                    "clv": None,
                    "probability": None,
                    "ev_cons": None,
                    "odds_real": bool(odds.get("1") and odds.get("X") and odds.get("2")),
                    "skip_reason": str(exc),
                    "score_reason_1": uncovered.get("score_reason_1"),
                    "score_reason_2": uncovered.get("score_reason_2"),
                    "tipster": play.get("tipster") or uncovered.get("tipster"),
                    "tipster_consensus": None
                    if not (play.get("tipster") or {}).get("consensus")
                    else play["tipster"]["consensus"],
                    "tipster_label": None if not play.get("tipster") else play["tipster"].get("label"),
                    "tipster_agree": None if not play.get("tipster") else play["tipster"].get("agree"),
                    "tipster_n": None if not play.get("tipster") else play["tipster"].get("n_sources"),
                    "markets": [],
                    "prediction": stub,
                    **_pro_fields(play, uncovered),
                    **_val_fields(play, uncovered, weather=wx),
                }
            )
            continue
        from modules.montecarlo.extras import match_side_extras

        mc = sim.simulate(
            pred["lambda_home"],
            pred["lambda_away"],
            model_probs={"home_win": pred["home_win"], "draw": pred["draw"], "away_win": pred["away_win"]},
            extras=match_side_extras(
                lambda_home=pred["lambda_home"],
                lambda_away=pred["lambda_away"],
                fb_home=fb_h,
                fb_away=fb_a,
                fd_home=(lookup_fd_side(home, fd_side_idx) if lookup_fd_side and fd_side_idx else None),
                fd_away=(lookup_fd_side(away, fd_side_idx) if lookup_fd_side and fd_side_idx else None),
                fb_match_home=(lookup_fbref_match_side(home, fb_match_idx) if lookup_fbref_match_side and fb_match_idx else None),
                fb_match_away=(lookup_fbref_match_side(away, fb_match_idx) if lookup_fbref_match_side and fb_match_idx else None),
            ),
        )
        try:
            from modules.calibration.conformal import attach_market_intervals

            mc = attach_market_intervals(
                mc,
                p_over_25=pred.get("p_over_25"),
                p_ah0_home=pred.get("p_ah0_home"),
            )
        except Exception:
            pass
        prediction = {
            "match": f"{pred['home_team']} vs {pred['away_team']}",
            "model_probabilities": {
                "home_win": pred["home_win"],
                "draw": pred["draw"],
                "away_win": pred["away_win"],
            },
            "market_ml": pred.get("market_ml") or {
                "p_over_25": pred.get("p_over_25"),
                "p_ah0_home": pred.get("p_ah0_home"),
            },
            "p_over_25": pred.get("p_over_25"),
            "p_ah0_home": pred.get("p_ah0_home"),
            "expected_goals": {"home": pred["lambda_home"], "away": pred["lambda_away"]},
            "features": pred.get("features") or {},
            "fbref_context": {
                "home": lookup_team_context(pred["home_team"], fbref_idx) or fb_h,
                "away": lookup_team_context(pred["away_team"], fbref_idx) or fb_a,
            },
            "understat_context": {
                "home": lookup_understat_team(pred["home_team"], understat_idx) or us_h,
                "away": lookup_understat_team(pred["away_team"], understat_idx) or us_a,
            },
            "statsbomb_context": {
                "home": lookup_statsbomb_team(pred["home_team"], statsbomb_idx),
                "away": lookup_statsbomb_team(pred["away_team"], statsbomb_idx),
            },
            "sofascore_context": {
                "home": lookup_sofascore_team(pred["home_team"], sofascore_idx),
                "away": lookup_sofascore_team(pred["away_team"], sofascore_idx),
            },
            "fotmob_context": {
                "home": lookup_fotmob_team(pred["home_team"], fotmob_idx),
                "away": lookup_fotmob_team(pred["away_team"], fotmob_idx),
                "match": lookup_fotmob_match(
                    pred["home_team"], pred["away_team"], day, fotmob_matches
                ),
            },
            "fotmob_xg": {
                "home": lookup_fotmob_xg(pred["home_team"], fotmob_xg) or fm_xg_h,
                "away": lookup_fotmob_xg(pred["away_team"], fotmob_xg) or fm_xg_a,
            },
            "conformal_intervals": pred.get("conformal_intervals") or {},
            "model_cluster": pred.get("model_cluster"),
            "montecarlo": mc,
            "league": str(fx.get("league") or ""),
            "country": str(fx.get("country") or ""),
            "home": pred["home_team"],
            "away": pred["away_team"],
            "weather": wx,
            "history_context": hist,
            "ensemble": pred.get("ensemble"),
            **_venue_fields(fx),
        }
        prediction["tactical"] = match_tactics(
            pred["home_team"],
            pred["away_team"],
            fx["date"],
            cal_idx,
            prediction["fbref_context"]["home"],
            prediction["fbref_context"]["away"],
            country=str(fx.get("country") or ""),
            league=str(fx.get("league") or ""),
            sofa_home=prediction["sofascore_context"]["home"],
            sofa_away=prediction["sofascore_context"]["away"],
            ml=prediction.get("model_probabilities"),
        )
        try:
            from modules.sportly_sim import build_sportly_sim

            prediction["date"] = day
            prediction["sportly_sim"] = build_sportly_sim(prediction)
        except Exception:
            prediction["sportly_sim"] = {"ready": False, "note": "sim fallita"}
        try:
            from modules.advisor.data_signal import build_data_signal

            prediction["data_signal"] = build_data_signal(prediction)
        except Exception:
            prediction["data_signal"] = {"ready": False, "note": "analisi dati fallita"}
        advice = advise(
            prediction,
            odds,
            market_move=market_move,
            odds_from_asian=(odds_source == "asianbetsoccer"),
            league=str(fx.get("league") or "") or None,
            lazy_secondary=True,
        )
        play = advice["play"]
        alt = advice.get("play_alt")
        # Quota book del pick (play["odds"]); odds_real è solo flag bool.
        pick_quota = _pick_quota_from_play(play)
        pick_fair = play.get("fair_odds")
        spread_score = None
        ah_line = None
        if market_move:
            spread_score = market_move.get("spread_score")
            if market_move.get("ah_open") is not None and market_move.get("ah_curr") is not None:
                ah_line = f"{market_move['ah_open']}->{market_move['ah_curr']}"
        slim_markets = [
            {
                "code": m["code"],
                "name": m["name"],
                "group": m["group"],
                "probability": m["probability"],
                "odds": m["odds"],
                "fair_odds": m["fair_odds"],
                "ev": m.get("ev_cons") if m.get("ev_cons") is not None else m.get("ev"),
                "ev_cons": m.get("ev_cons"),
                "ev_sharp": m.get("ev_sharp"),
                "edge_pp": m.get("edge_pp"),
                "p_cons": m.get("p_cons"),
                "p_market": m.get("p_market"),
                "odds_real": m.get("odds_real"),
                "value_note": m.get("value_note"),
                "score_prob": m["score_prob"],
                "score_value": m["score_value"],
                "score": m.get("score"),
                "odds_source": m.get("odds_source"),
            }
            for m in advice["all_markets"]
        ]
        rows.append(
            {
                "date": fx["date"].strftime("%Y-%m-%d"),
                "time": str(fx.get("time") or ""),
                "country": str(fx.get("country") or ""),
                "league": str(fx.get("league") or ""),
                "home": pred["home_team"],
                "away": pred["away_team"],
                "odd_1": odds["1"],
                "odd_x": odds["X"],
                "odd_2": odds["2"],
                "odd_over_25": odds.get("over_2.5"),
                "odd_under_25": odds.get("under_2.5"),
                "odds_source": odds_source,
                "odds": odds,
                "market_move": market_move,
                "market_align": None if not advice.get("market_align") else advice["market_align"].get("label"),
                "market_note": None if not market_move else market_move.get("movement_comment") or market_move.get("movement_summary"),
                "movement_level": None if not market_move else market_move.get("movement_level"),
                "movement_summary": None if not market_move else market_move.get("movement_summary"),
                "movement_comment": None if not market_move else market_move.get("movement_comment"),
                "steam_1x2": None if not market_move else market_move.get("steam_1x2"),
                "steam_ah": None if not market_move else market_move.get("steam_ah"),
                "steam_ou": None if not market_move else market_move.get("steam_ou"),
                "drop_1": None if not market_move else market_move.get("drop_1"),
                "drop_x": None if not market_move else market_move.get("drop_x"),
                "drop_2": None if not market_move else market_move.get("drop_2"),
                "drop_over": None if not market_move else market_move.get("drop_over"),
                "drop_under": None if not market_move else market_move.get("drop_under"),
                "spread_score": spread_score,
                "line_move": None if not market_move else market_move.get("line_move"),
                "ah_line": ah_line,
                "value_edge": play.get("edge_pp"),
                "edge_pp": play.get("edge_pp"),
                "p_cons": play.get("p_cons"),
                "p_market": play.get("p_market"),
                "ev_cons": play.get("ev_cons"),
                "ev_sharp": play.get("ev_sharp"),
                "odds_real": play.get("odds_real"),
                "value_note": play.get("value_note"),
                "quadro": advice.get("quadro"),
                "quadro_consenso": None if not advice.get("quadro") else advice["quadro"].get("consenso"),
                "quadro_n": None
                if not advice.get("quadro")
                else f"{advice['quadro'].get('agree_n')}/{advice['quadro'].get('votes_n')}",
                "quadro_summary": None if not advice.get("quadro") else advice["quadro"].get("summary"),
                "fbref_ctx": None if not advice.get("quadro") else advice["quadro"].get("fbref_summary"),
                "pick": play["code"],
                "pick_name": play["name"],
                "pick_group": play.get("group"),
                "action": play.get("action") or "gioca",
                "no_bet_reasons": play.get("no_bet_reasons") or [],
                "score": play["score"],
                "score_unified": play.get("score_unified"),
                "meta_label": None if not play.get("meta_analysis") else play["meta_analysis"].get("label"),
                "meta_note": None if not play.get("meta_analysis") else play["meta_analysis"].get("note"),
                "kelly_quarter": play.get("kelly_quarter"),
                "clv": play.get("clv"),
                "quota_pick": pick_quota,
                "score_reason_1": advice.get("score_reason_1"),
                "score_reason_2": advice.get("score_reason_2"),
                "kind": play["kind"],
                "probability": play["probability"],
                "ev": play["ev"],
                "fair_odds": play["fair_odds"],
                "tipster_consensus": None if not (play.get("tipster") or {}).get("consensus") else play["tipster"]["consensus"],
                "tipster_label": None if not play.get("tipster") else play["tipster"].get("label"),
                "tipster_agree": None if not play.get("tipster") else play["tipster"].get("agree"),
                "tipster_n": None if not play.get("tipster") else play["tipster"].get("n_sources"),
                "tipster": play.get("tipster"),
                "alt_pick": None if not alt else alt["code"],
                "alt_name": None if not alt else alt["name"],
                "alt_score": None if not alt else alt["score"],
                "p_home": mc["home_win"],
                "p_draw": mc["draw"],
                "p_away": mc["away_win"],
                "p_over_25": mc.get("over_2.5"),
                "p_btts": mc.get("btts"),
                "markets": slim_markets,
                "prediction": prediction,
                "data_signal": prediction.get("data_signal"),
                "source_agreement": advice.get("source_agreement") or play.get("source_agreement"),
                "prob_intervals": advice.get("prob_intervals") or play.get("prob_intervals"),
                "residual_ev": advice.get("residual_ev") or play.get("residual_ev"),
                "conformal_intervals": prediction.get("conformal_intervals") or play.get("conformal_intervals"),
                **_pro_fields(play, advice),
                **_val_fields(play, advice, weather=wx),
            }
        )

    try:
        from modules.data_update.fotmob_context import enrich_top_picks_fotmob

        fm_info = enrich_top_picks_fotmob(rows, min_score=7, max_n=12)
        if fm_info.get("n_enriched"):
            print(f"FotMob details top picks: {fm_info['n_enriched']}/{fm_info.get('n_candidates', 0)}")
    except Exception as exc:
        print(f"skip FotMob details top-N: {exc}")

    from modules.progress_report import emit

    emit(on_progress, 0.99, f"Salvataggio {len(rows)} partite…")
    print(f"calendario: salvataggio {len(rows)} partite…", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        from modules.data_update.history import archive_upcoming, settle_pending

        hist = archive_upcoming(rows)
        settled = settle_pending(learn=True, learn_only_if_settled=True)
        if settled.get("online_learn"):
            ol = settled["online_learn"]
            print(
                f"apprendimento online: ok={ol.get('ok')} "
                f"trainable={ol.get('n_trainable', '?')} settled={ol.get('n_settled', '?')}"
            )
        print(
            f"storico locale: {hist.get('n_history')} record "
            f"(+{hist.get('added')} nuovi, {settled.get('settled', 0)} chiusi, "
            f"{settled.get('n_settled', 0)} esiti in DB)"
        )
    except Exception as exc:
        print(f"skip storico locale: {exc}")
    print(
        f"upcoming {len(rows)} partite (senza storico modello: {skipped}, "
        f"riuso predizioni: {reused}, tutte restano in tabella)"
    )
    try:
        from modules.notify import dispatch_alerts

        dispatch_alerts(rows)
    except Exception as exc:
        print(f"skip telegram avvisi: {exc}")
    finally:
        hist_cache.close()
    return rows
