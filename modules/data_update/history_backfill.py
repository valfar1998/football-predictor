"""Backfill storico ricco da matches.csv (quote close + esito).

Popola SQLite con righe synthetic già settled per sbloccare gate online_learn
senza attendere settimane di archivi live. Le righe hanno synthetic_backfill=1.
Non sostituisce archivi live pre-match esistenti.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MATCHES = ROOT / "data" / "processed" / "matches.csv"


def _is_rich(rec: dict[str, Any]) -> bool:
    return bool(
        rec.get("quota_pick")
        and (rec.get("ev_cons") is not None or rec.get("ev_sharp") is not None)
        and rec.get("data_factors")
        and rec.get("agree_share") is not None
    )


def _odds_from_row(row: pd.Series) -> dict[str, float | None]:
    def f(*cols: str) -> float | None:
        for c in cols:
            if c in row.index and pd.notna(row[c]):
                try:
                    v = float(row[c])
                    if v >= 1.01:
                        return v
                except (TypeError, ValueError):
                    continue
        return None

    o1 = f("odd_home_close", "odd_home")
    ox = f("odd_draw_close", "odd_draw")
    o2 = f("odd_away_close", "odd_away")
    if not (o1 and ox and o2):
        return {}
    return {
        "1": o1,
        "X": ox,
        "2": o2,
        "over_2.5": f("odd_over_25_close", "odd_over_25"),
        "under_2.5": f("odd_under_25_close", "odd_under_25"),
    }


def backfill_from_matches(
    *,
    max_rows: int = 120,
    since_days: int = 400,
    n_sims: int = 1500,
    on_progress=None,
) -> dict[str, Any]:
    from modules.advisor.advise import advise
    from modules.data_update.history import _hit_for_pick, _key, _upsert, _connect, load_history
    from modules.data_update.team_names import resolve_known_team
    from modules.montecarlo import MonteCarloSimulator
    from modules.predictor import MatchPredictor
    from modules.progress_report import emit

    if not MATCHES.is_file():
        return {"ok": False, "error": "matches.csv assente: esegui --train", "added": 0}

    df = pd.read_csv(MATCHES, parse_dates=["date"], low_memory=False)
    if df.empty:
        return {"ok": False, "error": "matches.csv vuoto", "added": 0}

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=int(since_days))
    df = df[df["date"] >= cutoff].copy()
    df = df[df["home_goals"].notna() & df["away_goals"].notna()]
    df = df.sort_values("date", ascending=False)

    existing = {r["match_key"]: r for r in load_history()}
    rich_live = {
        k
        for k, r in existing.items()
        if _is_rich(r) and not int(r.get("synthetic_backfill") or 0)
    }

    predictor = MatchPredictor()
    team_idx = set(predictor.last_idx.keys())
    sim = MonteCarloSimulator(n_sims=n_sims)
    conn = _connect()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    added = 0
    skipped = 0
    errors = 0
    n_cand = min(len(df), max_rows * 3)

    emit(on_progress, 0.02, f"Backfill storico: scan {n_cand} partite…")

    try:
        for i, (_, row) in enumerate(df.head(n_cand).iterrows()):
            if added >= max_rows:
                break
            if i and i % 25 == 0:
                emit(on_progress, 0.05 + 0.9 * (added / max(1, max_rows)), f"Backfill {added}/{max_rows}…")

            try:
                home = resolve_known_team(str(row["home_team"])) or str(row["home_team"])
                away = resolve_known_team(str(row["away_team"])) or str(row["away_team"])
            except KeyError:
                skipped += 1
                continue
            if home not in team_idx or away not in team_idx:
                skipped += 1
                continue

            day = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
            mk = _key({"date": day, "home": home, "away": away})
            if mk in rich_live:
                skipped += 1
                continue
            prev = existing.get(mk)
            if prev and _is_rich(prev):
                skipped += 1
                continue

            odds = _odds_from_row(row)
            if not odds.get("1"):
                skipped += 1
                continue

            league = str(row.get("league") or "") or None
            try:
                pred = predictor.predict(
                    home,
                    away,
                    kickoff=row["date"],
                    league=league,
                    odds=odds,
                )
            except KeyError:
                skipped += 1
                continue

            mc = sim.simulate(
                pred["lambda_home"],
                pred["lambda_away"],
                model_probs={
                    "home_win": pred["home_win"],
                    "draw": pred["draw"],
                    "away_win": pred["away_win"],
                },
            )
            prediction = {
                "match": f"{home} vs {away}",
                "model_probabilities": {
                    "home_win": pred["home_win"],
                    "draw": pred["draw"],
                    "away_win": pred["away_win"],
                },
                "expected_goals": {"home": pred["lambda_home"], "away": pred["lambda_away"]},
                "montecarlo": mc,
                "features": pred.get("features") or {},
                "model_cluster": pred.get("model_cluster"),
                "league": league or "",
            }
            try:
                from modules.advisor.data_signal import build_data_signal

                prediction["data_signal"] = build_data_signal(prediction)
            except Exception:
                prediction["data_signal"] = {"ready": False, "note": "backfill senza contesto live"}

            try:
                advice = advise(prediction, odds, league=league)
            except Exception:
                errors += 1
                continue

            play = advice.get("play") or {}
            if play.get("action") in ("n/d", "invalido") and not play.get("probability"):
                skipped += 1
                continue

            pick = str(play.get("code") or "—")
            hg, ag = int(row["home_goals"]), int(row["away_goals"])
            res = "1" if hg > ag else ("2" if hg < ag else "X")
            hit = _hit_for_pick(pick, res=res, hg=hg, ag=ag, tot=hg + ag)
            if hit is None:
                skipped += 1
                continue

            from modules.advisor.staking import beat_close, clv_prob

            def _row_odd(*cols):
                for c in cols:
                    if c in row.index and pd.notna(row[c]):
                        try:
                            v = float(row[c])
                            if v >= 1.01:
                                return v
                        except (TypeError, ValueError):
                            continue
                return None

            pk = str(pick).upper()
            if pk == "1":
                open_o, close_o = _row_odd("odd_home"), _row_odd("odd_home_close")
            elif pk == "X":
                open_o, close_o = _row_odd("odd_draw"), _row_odd("odd_draw_close")
            elif pk == "2":
                open_o, close_o = _row_odd("odd_away"), _row_odd("odd_away_close")
            else:
                open_o, close_o = pick_quota, None
            bet_o = open_o or pick_quota
            clv_v = clv_prob(float(bet_o), float(close_o)) if bet_o and close_o else None
            beat = None
            if bet_o and close_o:
                bc = beat_close(float(bet_o), float(close_o))
                beat = 1 if bc else 0 if bc is False else None

            ds = prediction.get("data_signal") or {}
            sa = advice.get("source_agreement") or play.get("source_agreement") or {}
            factors = ds.get("factors") if isinstance(ds, dict) else None
            pick_quota = play.get("odds") or odds.get(pick.replace("O2.5", "over_2.5").replace("U2.5", "under_2.5"))
            if pick in ("1", "X", "2"):
                pick_quota = pick_quota or odds.get(pick)
            try:
                pick_quota = float(pick_quota) if pick_quota is not None else None
            except (TypeError, ValueError):
                pick_quota = None

            rec = {
                "match_key": mk,
                "date": day,
                "time": "",
                "home": home,
                "away": away,
                "league": league,
                "country": str(row.get("country") or ""),
                "pick": pick,
                "action": play.get("action") or "no_bet",
                "score": play.get("score"),
                "score_unified": play.get("score_unified"),
                "ev_cons": play.get("ev_cons"),
                "ev_sharp": play.get("ev_sharp"),
                "probability": play.get("probability"),
                "odds_source": "football-data.co.uk-close",
                "skip_reason": None,
                "covered": 1,
                "home_goals": hg,
                "away_goals": ag,
                "result": res,
                "hit": hit,
                "saved_at": now,
                "settled_at": now,
                "quota_pick": pick_quota,
                "agree_share": sa.get("agree_share") if isinstance(sa, dict) else None,
                "data_edge": ds.get("edge") if isinstance(ds, dict) else None,
                "move_rank": None,
                "residual": None,
                "adj_ev": None,
                "data_factors": factors,
                "no_bet_reasons": play.get("no_bet_reasons"),
                "pick_group": play.get("group"),
                "model_cluster": pred.get("model_cluster"),
                "context_partial": 1,
                "synthetic_backfill": 1,
                "clv": clv_v,
                "quota_close": close_o,
                "beat_close": beat,
            }
            _upsert(conn, rec, now, keep_result=False)
            existing[mk] = rec
            added += 1

        conn.commit()
    finally:
        conn.close()

    info = {
        "ok": True,
        "added": added,
        "skipped": skipped,
        "errors": errors,
        "max_rows": max_rows,
        "note": "righe synthetic_backfill=1; non sovrascrivono archivi live ricchi",
    }
    emit(on_progress, 1.0, f"Backfill OK · +{added} righe synthetic")
    print(f"backfill history: +{added} synthetic, {skipped} skip, {errors} err")
    if added:
        try:
            from modules.advisor.online_learn import learn_from_settled

            learn_from_settled(force=True)
            info["learn"] = True
        except Exception as exc:
            info["learn_error"] = str(exc)
        try:
            from modules.advisor.analysis_outcomes import refresh_analysis_outcomes

            refresh_analysis_outcomes()
            info["analysis_outcomes"] = True
        except Exception as exc:
            info["analysis_outcomes_error"] = str(exc)
    return info
