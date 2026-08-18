"""Pronostici sul calendario in arrivo, con quote scaricate dal sito."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from modules.advisor.advise import advise
from modules.data_update.asian_odds import asian_to_advisor_odds, find_asian_odds, summarize_moves
from modules.data_update.cups import known_team_index, resolve_known_team
from modules.data_update.fbref_context import load_fbref_team_index, lookup_team_context
from modules.data_update.understat_context import load_understat_team_index, lookup_understat_team
from modules.data_update.parse import load_fixtures
from modules.montecarlo import MonteCarloSimulator
from modules.predictor import MatchPredictor

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


def build_upcoming(n_sims: int = 4000) -> list[dict]:
    fixtures = load_fixtures()
    if fixtures.empty:
        OUT.write_text("[]", encoding="utf-8")
        return []

    predictor = MatchPredictor()
    sim = MonteCarloSimulator(n_sims=n_sims)
    team_idx = known_team_index(predictor.last_idx.keys())
    fbref_idx = load_fbref_team_index()
    understat_idx = load_understat_team_index()
    rows: list[dict] = []
    skipped = 0

    for _, fx in fixtures.iterrows():
        home = resolve_known_team(str(fx["home_team"]), team_idx) or str(fx["home_team"])
        away = resolve_known_team(str(fx["away_team"]), team_idx) or str(fx["away_team"])
        try:
            pred = predictor.predict(home, away)
        except KeyError:
            skipped += 1
            continue
        mc = sim.simulate(
            pred["lambda_home"],
            pred["lambda_away"],
            model_probs={"home_win": pred["home_win"], "draw": pred["draw"], "away_win": pred["away_win"]},
        )
        odds = {
            "1": _odd(fx, "odd_home"),
            "X": _odd(fx, "odd_draw"),
            "2": _odd(fx, "odd_away"),
            "over_2.5": _odd(fx, "odd_over_25"),
            "under_2.5": _odd(fx, "odd_under_25"),
        }
        odds_source = "football-data.co.uk"
        src = str(fx.get("source") or "")
        if src.startswith("fixtures-cups-asian") or "asian" in src:
            odds_source = "asianbetsoccer"
        elif src.startswith("fd.org") or src.startswith("fixtures-cups"):
            odds_source = "football-data.org"
        asian = find_asian_odds(home, away, fx["date"].strftime("%Y-%m-%d"))
        market_move = None
        if asian:
            ao = asian_to_advisor_odds(asian)
            for k, v in ao.items():
                if v is not None:
                    odds[k] = v
            odds_source = "asianbetsoccer"
            market_move = summarize_moves(asian)
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
                "home": lookup_team_context(pred["home_team"], fbref_idx),
                "away": lookup_team_context(pred["away_team"], fbref_idx),
            },
            "understat_context": {
                "home": lookup_understat_team(pred["home_team"], understat_idx),
                "away": lookup_understat_team(pred["away_team"], understat_idx),
            },
            "montecarlo": mc,
            "league": str(fx.get("league") or ""),
        }
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
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"upcoming {len(rows)} partite (saltate senza storia: {skipped}, coppe incluse se in AsianBetSoccer)")
    return rows
