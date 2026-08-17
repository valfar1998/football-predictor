"""Pronostici sul calendario in arrivo, con quote scaricate dal sito."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from modules.advisor.advise import advise
from modules.data_update.asian_odds import asian_to_advisor_odds, find_asian_odds, summarize_moves
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
    rows: list[dict] = []
    skipped = 0

    for _, fx in fixtures.iterrows():
        home, away = str(fx["home_team"]), str(fx["away_team"])
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
        asian = find_asian_odds(home, away, fx["date"].strftime("%Y-%m-%d"))
        market_move = None
        if asian:
            ao = asian_to_advisor_odds(asian)
            for k, v in ao.items():
                if v is not None:
                    odds[k] = v
            odds_source = "asianbetsoccer"
            market_move = asian.get("market_move") or summarize_moves(asian)
        prediction = {
            "match": f"{pred['home_team']} vs {pred['away_team']}",
            "model_probabilities": {
                "home_win": pred["home_win"],
                "draw": pred["draw"],
                "away_win": pred["away_win"],
            },
            "expected_goals": {"home": pred["lambda_home"], "away": pred["lambda_away"]},
            "montecarlo": mc,
        }
        advice = advise(prediction, odds, market_move=market_move, odds_from_asian=(odds_source == "asianbetsoccer"))
        play = advice["play"]
        alt = advice.get("play_alt")
        pick_quota = play.get("odds")
        pick_fair = play.get("fair_odds")
        value_edge = None
        if pick_quota and pick_fair and pick_fair > 1.01:
            value_edge = round(float(pick_quota) / float(pick_fair) - 1.0, 4)
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
                "ev": m["ev"],
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
                "market_note": None if not market_move else market_move.get("movement_summary") or market_move.get("note"),
                "movement_level": None if not market_move else market_move.get("movement_level"),
                "movement_summary": None if not market_move else market_move.get("movement_summary"),
                "steam_1x2": None if not market_move else market_move.get("steam_1x2"),
                "steam_ah": None if not market_move else market_move.get("steam_ah"),
                "steam_ou": None if not market_move else market_move.get("steam_ou"),
                "drop_1": None if not market_move else market_move.get("drop_1"),
                "drop_x": None if not market_move else market_move.get("drop_x"),
                "drop_2": None if not market_move else market_move.get("drop_2"),
                "spread_score": spread_score,
                "ah_line": ah_line,
                "value_edge": value_edge,
                "pick": play["code"],
                "pick_name": play["name"],
                "pick_group": play.get("group"),
                "score": play["score"],
                "kelly_quarter": play.get("kelly_quarter"),
                "quota_pick": pick_quota,
                "score_reason_1": advice.get("score_reason_1"),
                "score_reason_2": advice.get("score_reason_2"),
                "kind": play["kind"],
                "probability": play["probability"],
                "ev": play["ev"],
                "fair_odds": play["fair_odds"],
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
    print(f"upcoming {len(rows)} partite (saltate senza storia: {skipped})")
    return rows
