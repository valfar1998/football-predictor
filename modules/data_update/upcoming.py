"""Pronostici sul calendario in arrivo, con quote scaricate dal sito."""

from __future__ import annotations

import json
from pathlib import Path

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
    lookup_fotmob_match,
    lookup_fotmob_team,
)
from modules.data_update.parse import load_fixtures
from modules.data_update.venues import update_home_venues
from modules.montecarlo import MonteCarloSimulator
from modules.predictor import MatchPredictor
from modules.predictor.predict import context_xg

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


def _xg_pair(us_row: dict | None, fb_row: dict | None) -> tuple[float, float] | None:
    return context_xg(us_row, fb_row)


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


def build_upcoming(n_sims: int = 4000) -> list[dict]:
    fixtures = load_fixtures()
    if fixtures.empty:
        OUT.write_text("[]", encoding="utf-8")
        return []
    try:
        update_home_venues(fixtures)
    except Exception:
        pass

    predictor = MatchPredictor()
    sim = MonteCarloSimulator(n_sims=n_sims)
    team_idx = known_team_index(predictor.last_idx.keys())
    fbref_idx = load_fbref_team_index()
    understat_idx = load_understat_team_index()
    statsbomb_idx = load_statsbomb_team_index()
    sofascore_idx = load_sofascore_team_index()
    fotmob_idx = load_fotmob_team_index()
    fotmob_matches = load_fotmob_matches()
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

    from modules.data_update.history import lookup_history_match
    from modules.data_update.weather import lookup_weather, prefetch_weather

    wx_items = []
    for _, fx0 in fixtures.iterrows():
        city = _fx_text(fx0, "venue_city")
        try:
            day0 = fx0["date"].strftime("%Y-%m-%d")
        except Exception:
            day0 = str(fx0.get("date") or "")[:10]
        if city and day0:
            wx_items.append({"city": city, "date": day0})
    try:
        wx_idx = prefetch_weather(wx_items)
    except Exception:
        wx_idx = {}

    rows: list[dict] = []
    skipped = 0

    for _, fx in fixtures.iterrows():
        home = resolve_known_team(str(fx["home_team"]), team_idx) or str(fx["home_team"])
        away = resolve_known_team(str(fx["away_team"]), team_idx) or str(fx["away_team"])
        league = str(fx.get("league") or "") or None
        day = fx["date"].strftime("%Y-%m-%d")
        city = _fx_text(fx, "venue_city")
        wx = lookup_weather(city, day, wx_idx) if city else None
        odds, odds_source, market_move = _collect_match_odds(
            fx, home, away, _pinnacle_events, _betfair_events
        )
        fb_h = lookup_team_context(home, fbref_idx)
        fb_a = lookup_team_context(away, fbref_idx)
        us_h = lookup_understat_team(home, understat_idx)
        us_a = lookup_understat_team(away, understat_idx)
        hist = lookup_history_match(home, away, league=league)
        try:
            pred = predictor.predict(
                home,
                away,
                kickoff=fx["date"],
                league=league,
                odds=odds,
                ext_xg_home=_xg_pair(us_h, fb_h),
                ext_xg_away=_xg_pair(us_a, fb_a),
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
                    **_val_fields(play, uncovered, weather=wx),
                }
            )
            continue
        mc = sim.simulate(
            pred["lambda_home"],
            pred["lambda_away"],
            model_probs={"home_win": pred["home_win"], "draw": pred["draw"], "away_win": pred["away_win"]},
        )
        prediction = {
            "match": f"{pred['home_team']} vs {pred['away_team']}",
            "model_probabilities": {
                "home_win": pred["home_win"],
                "draw": pred["draw"],
                "away_win": pred["away_win"],
            },
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
        )
        play = advice["play"]
        alt = advice.get("play_alt")
        pick_quota = play.get("odds")
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
                "source_agreement": advice.get("source_agreement") or play.get("source_agreement"),
                "prob_intervals": advice.get("prob_intervals") or play.get("prob_intervals"),
                "residual_ev": advice.get("residual_ev") or play.get("residual_ev"),
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

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        from modules.data_update.history import archive_upcoming, settle_pending

        hist = archive_upcoming(rows)
        settled = settle_pending()
        print(
            f"storico locale: {hist.get('n_history')} record "
            f"(+{hist.get('added')} nuovi, {settled.get('settled', 0)} chiusi, "
            f"{settled.get('n_settled', 0)} esiti in DB)"
        )
    except Exception as exc:
        print(f"skip storico locale: {exc}")
    print(f"upcoming {len(rows)} partite (senza storico modello: {skipped}, tutte restano in tabella)")
    try:
        from modules.notify import dispatch_alerts

        dispatch_alerts(rows)
    except Exception as exc:
        print(f"skip telegram avvisi: {exc}")
    return rows
