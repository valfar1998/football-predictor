"""Orchestratore: dati → feature → training → prediction → Monte Carlo → JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from modules.advisor.advise import advise, format_advice
from modules.data_update import build_upcoming, download_all, fetch_asian_odds, save_asian_odds
from modules.data_update.download import download_fixtures, download_season_zip
from modules.data_update.leagues import SEASON_ZIPS
from modules.dataset_loader import DatasetLoader
from modules.feature_engineering import FeatureEngineer
from modules.model_training import ModelTrainer
from modules.montecarlo import MonteCarloSimulator
from modules.calibration import run_full_calibration
from modules.predictor import MatchPredictor
from modules.tipsters import fetch_tipsters
from modules.data_update.fbref_context import lookup_team_context, load_fbref_team_index
from modules.data_update.understat_context import lookup_understat_team, load_understat_team_index

OUT_DIR = ROOT / "data" / "processed"


def ensure_raw_sample() -> None:
    fd = ROOT / "data" / "raw" / "fd" / "main"
    if fd.exists() and any(fd.glob("*/*.csv")):
        return
    raw = ROOT / "data" / "raw"
    if any(p for p in raw.glob("*.csv") if "synthetic" not in p.name.lower()):
        return
    print("Nessun dato reale: scarico football-data.co.uk...")
    download_all()


def update_pipeline(*, retrain: bool = True) -> dict:
    info = download_all()
    train_info: dict = {}
    if retrain or not (ROOT / "data" / "models" / "best_model.joblib").exists():
        train_info = train_pipeline()
    upcoming = build_upcoming()
    return {
        **info,
        **{k: v for k, v in train_info.items() if k != "path"},
        "n_upcoming": len(upcoming),
        "model": train_info.get("path"),
    }


def refresh_odds_pipeline(*, asian: bool = True, on_progress=None) -> dict:
    from modules.progress_report import emit

    def p(frac: float, msg: str) -> None:
        emit(on_progress, frac, msg)

    def span(lo: float, hi: float):
        def cb(frac, msg=""):
            p(lo + (hi - lo) * max(0.0, min(1.0, float(frac))), msg)

        return cb

    p(0.02, "Calendario football-data.co.uk…")
    try:
        download_fixtures()
    except Exception as exc:
        print(f"skip fixtures: {exc}", flush=True)
    p(0.08, "Stagione corrente (zip)…")
    try:
        download_season_zip(SEASON_ZIPS[-1])
    except Exception as exc:
        print(f"skip stagione: {exc}", flush=True)
    asian_info: dict = {}
    if asian:
        p(0.12, "Quote AsianBetSoccer (14 giorni)…")
        rows = fetch_asian_odds(days=14, book="bet365", on_progress=span(0.12, 0.34))
        path = save_asian_odds(rows)
        asian_info = {"n_asian": len(rows), "asian_cache": str(path)}
    # Pinnacle da The Odds API (1 chiamata/giorno, cache 20h)
    try:
        from modules.data_update.odds_api import fetch_pinnacle_odds
        p(0.36, "Quote Pinnacle…")
        pinn = fetch_pinnacle_odds()
        asian_info["pinnacle_events"] = pinn.get("n_events", 0)
        asian_info["pinnacle_remaining"] = pinn.get("remaining")
        asian_info["pinnacle_from_cache"] = pinn.get("from_cache", False)
        if not pinn.get("ok") and pinn.get("error"):
            asian_info["pinnacle_error"] = pinn["error"]
    except Exception as exc:
        asian_info["pinnacle_error"] = str(exc)
        print(f"skip Pinnacle odds: {exc}", flush=True)
    try:
        from modules.data_update.betfair import fetch_betfair_odds
        p(0.40, "Quote Betfair…")
        bf = fetch_betfair_odds()
        asian_info["betfair_events"] = bf.get("n_events", 0)
        asian_info["betfair_from_cache"] = bf.get("from_cache", False)
        asian_info["betfair_ok"] = bool(bf.get("ok"))
        if not bf.get("ok") and bf.get("error"):
            asian_info["betfair_error"] = bf["error"]
            asian_info["betfair_soft_fail"] = True
            print(f"Betfair soft-fail (pipeline continua con Pinnacle/Asian): {bf['error']}", flush=True)
    except Exception as exc:
        asian_info["betfair_error"] = str(exc)
        asian_info["betfair_soft_fail"] = True
        asian_info["betfair_ok"] = False
        print(f"Betfair soft-fail (pipeline continua): {exc}", flush=True)
    try:
        from modules.data_update.clubelo import fetch_clubelo

        p(0.44, "ClubElo…")
        elo = fetch_clubelo()
        asian_info["n_clubelo"] = 0 if elo is None or elo.empty else int(len(elo))
    except Exception as exc:
        asian_info["clubelo_error"] = str(exc)
    try:
        from modules.data_update.cups import download_org_cups

        p(0.48, "Coppe football-data.org…")
        cups_info = download_org_cups(days=14)
    except Exception as exc:
        cups_info = {"n_cup_files": 0, "error": str(exc)}
        print(f"skip coppe org: {exc}", flush=True)
    try:
        from modules.data_update.world_fixtures import download_world_fixtures

        p(0.52, "Calendario mondiale…")
        world_info = download_world_fixtures(days=14)
    except Exception as exc:
        world_info = {"n_world_fixtures": 0, "error": str(exc)}
        print(f"skip calendario mondiale: {exc}", flush=True)
    try:
        from modules.data_update.thesportsdb import download_cup_fixtures

        p(0.56, "Coppe TheSportsDB…")
        tsdb_info = download_cup_fixtures()
    except Exception as exc:
        tsdb_info = {"n_cup_files": 0, "error": str(exc)}
        print(f"skip coppe TheSportsDB: {exc}", flush=True)
    try:
        from modules.data_update.api_football import download_cup_fixtures as download_api_football_cups

        p(0.60, "Coppe API-Football…")
        apif_info = download_api_football_cups(days=14)
    except Exception as exc:
        apif_info = {"n_cup_files": 0, "error": str(exc)}
        print(f"skip coppe API-Football: {exc}", flush=True)

    def _fresh(path: Path, hours: float = 72.0) -> bool:
        try:
            if not path.is_file():
                return False
            import time as _t

            return (_t.time() - path.stat().st_mtime) < hours * 3600
        except OSError:
            return False

    proc = ROOT / "data" / "processed"
    fbref_info: dict = {"skipped_fresh": True} if _fresh(proc / "fbref_team_context.csv") else {}
    if not fbref_info:
        try:
            from modules.data_update.fbref_context import download_fbref_context

            p(0.64, "Contesto FBref (se cache >72h)…")
            fbref_info = download_fbref_context()
        except Exception as exc:
            fbref_info = {"ok": False, "n_teams": 0, "error": str(exc)}
            print(f"skip FBref context: {exc}", flush=True)
    else:
        p(0.64, "FBref: cache fresca, skip")
        print("FBref context: cache fresca (<72h), skip download", flush=True)
    understat_info: dict = {"skipped_fresh": True} if _fresh(proc / "understat_team_context.csv") else {}
    if not understat_info:
        try:
            from modules.data_update.understat_context import download_understat_context

            p(0.70, "Understat…")
            understat_info = download_understat_context()
        except Exception as exc:
            understat_info = {"ok": False, "n_teams": 0, "error": str(exc)}
            print(f"skip Understat context: {exc}", flush=True)
    else:
        p(0.70, "Understat: cache fresca, skip")
        print("Understat context: cache fresca (<72h), skip download", flush=True)
    fd_rates_info: dict = {"skipped_fresh": True} if _fresh(proc / "fd_side_rates.csv") else {}
    if not fd_rates_info:
        try:
            from modules.data_update.side_rates import build_fd_side_rates

            p(0.74, "FD cards/corners…")
            fd_rates_info = build_fd_side_rates()
        except Exception as exc:
            fd_rates_info = {"ok": False, "n_teams": 0, "error": str(exc)}
            print(f"skip FD cards/corners rates: {exc}", flush=True)
    else:
        p(0.74, "FD rates: cache fresca, skip")
        print("FD side rates: cache fresca (<72h), skip", flush=True)
    try:
        from modules.data_update.statsbomb_context import download_statsbomb_context

        p(0.78, "StatsBomb…")
        statsbomb_info = (
            {"skipped_fresh": True}
            if _fresh(proc / "statsbomb_team_context.csv")
            else download_statsbomb_context()
        )
        if statsbomb_info.get("skipped_fresh"):
            print("StatsBomb context: cache fresca (<72h), skip download", flush=True)
    except Exception as exc:
        statsbomb_info = {"ok": False, "n_teams": 0, "error": str(exc)}
        print(f"skip StatsBomb context: {exc}", flush=True)
    fotmob_info: dict = {"skipped_fresh": True} if _fresh(proc / "fotmob_matches.json") else {}
    if not fotmob_info:
        try:
            from modules.data_update.fotmob_context import download_fotmob_context

            p(0.82, "FotMob…")
            fotmob_info = download_fotmob_context(days=7)
        except Exception as exc:
            fotmob_info = {"ok": False, "n_teams": 0, "n_matches": 0, "error": str(exc)}
            print(f"skip FotMob context: {exc}", flush=True)
    else:
        p(0.82, "FotMob: cache fresca, skip")
        print("FotMob context: cache fresca (<72h), skip download", flush=True)
    tips_info: dict = {}
    try:
        p(0.86, "Tipster…")
        tips = fetch_tipsters()
        tips_info = {"n_tipsters": tips.get("n"), "tipster_counts": tips.get("counts"), "tipster_errors": tips.get("errors")}
    except Exception as exc:
        tips_info = {"tipster_error": str(exc)}
    p(0.90, "Ricostruisco il calendario (riuso predizioni)…")
    upcoming = build_upcoming(reuse_predictions=True)
    p(1.0, f"OK · {len(upcoming)} partite")
    return {
        "n_upcoming": len(upcoming),
        "source": "football-data.co.uk + football-data.org + thesportsdb + world + asianbetsoccer",
        "reuse_predictions": True,
        **asian_info,
        **tips_info,
        **{f"cups_{k}": v for k, v in cups_info.items()},
        **{f"tsdb_{k}": v for k, v in tsdb_info.items()},
        **{f"apif_{k}": v for k, v in apif_info.items()},
        **{f"world_{k}": v for k, v in world_info.items()},
        **{f"fbref_{k}": v for k, v in fbref_info.items()},
        **{f"understat_{k}": v for k, v in understat_info.items()},
        **{f"fd_rates_{k}": v for k, v in fd_rates_info.items()},
        **{f"statsbomb_{k}": v for k, v in statsbomb_info.items()},
        **{f"fotmob_{k}": v for k, v in fotmob_info.items()},
    }


def notify_refresh_pipeline(*, days: int = 4, book: str = "bet365") -> dict:
    """Refresh leggero per gli avvisi Telegram: solo Asian + calendario. Non sveglia il PC."""
    import time

    lock = OUT_DIR / "notify_refresh.lock"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if lock.exists() and time.time() - lock.stat().st_mtime < 40 * 60:
        print("notify-refresh skip: già in corso")
        return {"skipped": True, "reason": "già in corso"}
    lock.write_text(str(time.time()), encoding="utf-8")
    try:
        rows = fetch_asian_odds(days=days, book=book)
        asian_info: dict = {"n_asian": len(rows)}
        if rows:
            path = save_asian_odds(rows)
            asian_info["asian_cache"] = str(path)
        else:
            print("asian vuoto: tengo la cache precedente")
            asian_info["kept_previous_cache"] = True
        # Pinnacle: aggiorna solo se cache stantia (>20h), altrimenti usa quella esistente.
        # Non brucia chiamate API nel refresh notturno.
        try:
            from modules.data_update.odds_api import fetch_pinnacle_odds
            pinn = fetch_pinnacle_odds(max_age_hours=20.0)
            asian_info["pinnacle_events"] = pinn.get("n_events", 0)
            asian_info["pinnacle_from_cache"] = pinn.get("from_cache", True)
        except Exception as exc:
            asian_info["pinnacle_error"] = str(exc)
        try:
            from modules.data_update.betfair import fetch_betfair_odds
            bf = fetch_betfair_odds(max_age_hours=6.0)
            asian_info["betfair_events"] = bf.get("n_events", 0)
            asian_info["betfair_from_cache"] = bf.get("from_cache", True)
            asian_info["betfair_ok"] = bool(bf.get("ok"))
            if not bf.get("ok") and bf.get("error"):
                asian_info["betfair_error"] = bf["error"]
                asian_info["betfair_soft_fail"] = True
                print(f"Betfair soft-fail (notify refresh continua): {bf['error']}")
        except Exception as exc:
            asian_info["betfair_error"] = str(exc)
            asian_info["betfair_soft_fail"] = True
            asian_info["betfair_ok"] = False
            print(f"Betfair soft-fail (notify refresh continua): {exc}")
        from modules.data_update.upcoming import OUT as UP_OUT, refresh_upcoming_odds
        from modules.data_update.upcoming import _model_newer_than_upcoming

        if UP_OUT.exists() and not _model_newer_than_upcoming():
            refresh = refresh_upcoming_odds()
            return {
                "n_upcoming": refresh.get("n_upcoming", 0),
                "n_refreshed": refresh.get("n_refreshed", 0),
                "days": days,
                "book": book,
                "light": True,
                **asian_info,
                **{k: v for k, v in refresh.items() if k.startswith(("n_new_", "n_sent", "n_gioca", "n_watch", "n_spread", "telegram"))},
            }
        upcoming = build_upcoming()
        return {"n_upcoming": len(upcoming), "days": days, "book": book, "light": False, **asian_info}
    finally:
        try:
            lock.unlink(missing_ok=True)
        except OSError:
            pass


def asian_odds_pipeline(*, days: int = 14, book: str = "bet365", full_context: bool = False) -> dict:
    """Scarica Asian (+ tipster) e aggiorna EV/voto sul calendario esistente.

    Di default non rifà FBref/FotMob/MC: stesse probabilità, quote nuove.
    ``full_context=True`` ripristina il vecchio comportamento pesante.
    """
    rows = fetch_asian_odds(days=days, book=book)
    path = save_asian_odds(rows)
    tips_info: dict = {}
    try:
        tips = fetch_tipsters()
        tips_info = {"n_tipsters": tips.get("n"), "tipster_counts": tips.get("counts")}
    except Exception as exc:
        tips_info = {"tipster_error": str(exc)}
    extra: dict = {}
    if full_context:
        try:
            from modules.data_update.clubelo import fetch_clubelo

            elo = fetch_clubelo()
            extra["n_clubelo"] = 0 if elo is None or elo.empty else int(len(elo))
        except Exception as exc:
            extra["clubelo_error"] = str(exc)
        for label, loader in (
            ("fbref", "modules.data_update.fbref_context.download_fbref_context"),
            ("understat", "modules.data_update.understat_context.download_understat_context"),
            ("fotmob", "modules.data_update.fotmob_context.download_fotmob_context"),
        ):
            try:
                mod_path, fn_name = loader.rsplit(".", 1)
                import importlib

                mod = importlib.import_module(mod_path)
                fn = getattr(mod, fn_name)
                info = fn(days=7) if label == "fotmob" else fn()
                extra.update({f"{label}_{k}": v for k, v in (info or {}).items()})
            except Exception as exc:
                extra[f"{label}_error"] = str(exc)
        upcoming = build_upcoming()
        return {
            "n_asian": len(rows),
            "asian_cache": str(path),
            "n_upcoming": len(upcoming),
            "book": book,
            "days": days,
            "light": False,
            **tips_info,
            **extra,
        }
    from modules.data_update.upcoming import refresh_upcoming_odds

    refresh = refresh_upcoming_odds()
    return {
        "n_asian": len(rows),
        "asian_cache": str(path),
        "n_upcoming": refresh.get("n_upcoming", 0),
        "n_refreshed": refresh.get("n_refreshed", 0),
        "book": book,
        "days": days,
        "light": True,
        **tips_info,
        **{k: refresh.get(k) for k in ("ok", "error", "n_skipped") if k in refresh},
    }


def tipsters_pipeline() -> dict:
    info = fetch_tipsters()
    from modules.data_update.upcoming import refresh_upcoming_odds

    refresh = refresh_upcoming_odds()
    return (
        {k: v for k, v in info.items() if k != "matches"}
        | {
            "n_upcoming": refresh.get("n_upcoming", 0),
            "n_refreshed": refresh.get("n_refreshed", 0),
            "light": True,
        }
    )


def train_pipeline() -> dict:
    ensure_raw_sample()
    loader = DatasetLoader()
    matches, matches_path = loader.run("matches.csv")
    engineer = FeatureEngineer(window=5)
    features = engineer.transform(matches)
    feat_path = engineer.save(features, "features.csv")
    trainer = ModelTrainer()
    train_info = trainer.train(features)
    cal_info: dict = {}
    try:
        cal_info = run_full_calibration()
    except Exception as exc:
        cal_info = {"calibration_error": str(exc)}
    return {
        "n_matches": int(len(matches)),
        "n_features_rows": int(len(features)),
        "matches_path": str(matches_path),
        "features_path": str(feat_path),
        **train_info,
        "calibration": {
            k: cal_info.get(k)
            for k in (
                "temperature",
                "min_ev_play",
                "brier_favorite_raw",
                "brier_favorite_calibrated",
                "brier_multiclass_calibrated",
                "log_loss_calibrated",
                "ece_calibrated",
                "path",
                "calibration_error",
            )
            if k in cal_info
        },
    }


def predict_pipeline(home: str, away: str, n_sims: int = 10_000, odds: dict | None = None, league: str | None = None) -> dict:
    from modules.data_update.history import lookup_history_match
    from modules.predictor.predict import context_xg

    predictor = MatchPredictor()
    fb_idx = load_fbref_team_index()
    us_idx = load_understat_team_index()
    fb_h = lookup_team_context(home, fb_idx)
    fb_a = lookup_team_context(away, fb_idx)
    us_h = lookup_understat_team(home, us_idx)
    us_a = lookup_understat_team(away, us_idx)
    pred = predictor.predict(
        home,
        away,
        league=league,
        odds=odds,
        ext_xg_home=context_xg(us_h, fb_h),
        ext_xg_away=context_xg(us_a, fb_a),
    )
    sim = MonteCarloSimulator(n_sims=n_sims).simulate(
        pred["lambda_home"],
        pred["lambda_away"],
        model_probs={"home_win": pred["home_win"], "draw": pred["draw"], "away_win": pred["away_win"]},
    )
    try:
        from modules.calibration.conformal import attach_market_intervals

        sim = attach_market_intervals(
            sim,
            p_over_25=pred.get("p_over_25"),
            p_ah0_home=pred.get("p_ah0_home"),
        )
    except Exception:
        pass
    out = {
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
            "home": lookup_team_context(pred["home_team"], fb_idx) or fb_h,
            "away": lookup_team_context(pred["away_team"], fb_idx) or fb_a,
        },
        "understat_context": {
            "home": lookup_understat_team(pred["home_team"], us_idx) or us_h,
            "away": lookup_understat_team(pred["away_team"], us_idx) or us_a,
        },
        "montecarlo": sim,
        "league": league or "",
        "home": pred["home_team"],
        "away": pred["away_team"],
        "history_context": lookup_history_match(pred["home_team"], pred["away_team"], league=league),
        "ensemble": pred.get("ensemble"),
        "model_cluster": pred.get("model_cluster"),
        "conformal_intervals": pred.get("conformal_intervals") or {},
    }
    try:
        from modules.sportly_sim import build_sportly_sim

        out["sportly_sim"] = build_sportly_sim(out)
    except Exception:
        out["sportly_sim"] = {"ready": False}
    dest = OUT_DIR / "last_prediction.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    out["saved_to"] = str(dest)
    return out


def _has_streamlit(python_exe: Path) -> bool:
    import subprocess

    probe = subprocess.run(
        [str(python_exe), "-c", "import streamlit"],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def _ui_python() -> Path:
    candidates = [
        Path(sys.executable),
        ROOT / ".venv" / "Scripts" / "python.exe",
        ROOT / ".venv" / "bin" / "python",
    ]
    seen: set[str] = set()
    for exe in candidates:
        key = str(exe.resolve()) if exe.exists() else str(exe)
        if key in seen or not exe.exists():
            continue
        seen.add(key)
        if _has_streamlit(exe):
            return exe
    print(
        "Streamlit non è in questo Python.\n"
        "Usa l'ambiente del progetto:\n"
        "  .\\.venv\\Scripts\\Activate.ps1\n"
        "  python main.py --ui\n"
        "oppure:\n"
        "  .\\.venv\\Scripts\\python.exe main.py --ui",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _launch_ui() -> int:
    import subprocess

    python_exe = _ui_python()
    print(f"UI con: {python_exe}")
    return subprocess.call([str(python_exe), "-m", "streamlit", "run", str(ROOT / "app.py")])


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Football predictor")
    parser.add_argument("--train", action="store_true", help="ricostruisce dataset, feature e modello")
    parser.add_argument(
        "--train-markets",
        action="store_true",
        help="allena solo i classificatori O/U 2.5 e AH 0 (veloce)",
    )
    parser.add_argument("--update", action="store_true", help="scarica dati mondiali + quote, allena, calendario")
    parser.add_argument("--odds-update", action="store_true", help="aggiorna fixtures/quote (incluso AsianBetSoccer) e pronostici")
    parser.add_argument(
        "--backfill-history",
        action="store_true",
        help="popola SQLite con pick synthetic da matches.csv (quote close, già settled)",
    )
    parser.add_argument("--backfill-max", type=int, default=120, help="max righe synthetic con --backfill-history")
    parser.add_argument("--calibrate", action="store_true", help="calibra probabilità e taratura EV su storico")
    parser.add_argument("--asian-odds", action="store_true", help="scarica quote AsianBetSoccer e ricalcola calendario")
    parser.add_argument("--tipsters", action="store_true", help="scarica pronostici Forebet/PredictZ/Vitibet e ricalcola calendario")
    parser.add_argument("--predict", nargs=2, metavar=("HOME", "AWAY"), help="es. --predict Inter Milan")
    parser.add_argument("--odds", nargs=3, type=float, metavar=("ODD_1", "ODD_X", "ODD_2"), help="quote decimali 1 X 2")
    parser.add_argument("--advise", action="store_true", help="consiglio 1X2 sulla ultima predizione (o su --predict)")
    parser.add_argument("--ui", action="store_true", help="apre l'interfaccia Streamlit")
    parser.add_argument("--sims", type=int, default=10_000)
    parser.add_argument(
        "--notify",
        action="store_true",
        help="invia su Telegram (stesso bot offerte) voto ≥9 e spread Asian Raro ≥1",
    )
    parser.add_argument("--notify-test", action="store_true", help="ping di prova sul bot Telegram")
    parser.add_argument(
        "--notify-resend",
        nargs=2,
        metavar=("HOME", "AWAY"),
        help="reinvia lo spread Raro di una partita (con giocabilità 1–10)",
    )
    parser.add_argument("--notify-dry", action="store_true", help="stampa gli avvisi senza inviarli")
    parser.add_argument(
        "--notify-refresh",
        action="store_true",
        help="ogni ~30 min: scarica AsianBetSoccer (4 giorni) e ricalcola calendario/avvisi",
    )
    args = parser.parse_args()

    if args.ui:
        raise SystemExit(_launch_ui())

    if args.notify_test:
        from modules.notify import ping_bot
        from modules.notify.telegram import telegram_status

        print(telegram_status())
        ok = ping_bot()
        print("ping inviato" if ok else "ping non inviato")
        return

    if args.notify_resend:
        from modules.notify.alerts import resend_spread_match

        info = resend_spread_match(args.notify_resend[0], args.notify_resend[1])
        print(json.dumps({k: v for k, v in info.items() if k != "text"}, indent=2, default=str))
        if info.get("text"):
            print(info["text"])
        if not info.get("ok"):
            raise SystemExit(1)
        return

    if args.notify_refresh:
        info = notify_refresh_pipeline()
        print(json.dumps(info, indent=2, default=str))
        return

    if args.notify or args.notify_dry:
        from modules.notify import dispatch_alerts

        info = dispatch_alerts(dry_run=args.notify_dry)
        print(json.dumps({k: v for k, v in info.items() if k != "status"}, indent=2))
        print(info.get("status"))
        return

    if args.backfill_history:
        from modules.data_update.history_backfill import backfill_from_matches

        info = backfill_from_matches(max_rows=max(1, int(args.backfill_max)))
        print(json.dumps(info, indent=2, default=str))
        return

    if args.update:
        info = update_pipeline(retrain=True)
        print(json.dumps({k: v for k, v in info.items() if k != "model"}, indent=2, default=str))
        if info.get("model"):
            print("modello:", info["model"])
        n_cl = info.get("n_clusters")
        if n_cl is not None:
            print(f"Modelli cluster attivi: {n_cl}. Fallback globale: sì.")
        return

    if args.train_markets:
        import pandas as pd
        from modules.model_training.market_models import train_market_models

        feat_path = ROOT / "data" / "processed" / "features.csv"
        if not feat_path.exists():
            train_pipeline()
        feat = pd.read_csv(feat_path, parse_dates=["date"])
        info = train_market_models(feat)
        print(json.dumps(info, indent=2, default=str))
        return

    if args.train:
        info = train_pipeline()
        print(json.dumps({k: v for k, v in info.items() if k != "path"}, indent=2, default=str))
        n_cl = info.get("n_clusters")
        if n_cl is not None:
            print(f"Modelli cluster attivi: {n_cl}. Fallback globale: sì.")
        mm = info.get("market_models") or {}
        if mm.get("ok"):
            print("Modelli mercato O/U 2.5 + AH 0: ok")
        return

    if args.calibrate:
        if not (ROOT / "data" / "models" / "best_model.joblib").exists():
            train_pipeline()
        else:
            ensure_raw_sample()
        info = run_full_calibration()
        print(json.dumps({k: v for k, v in info.items() if k not in {"reliability_1x2", "reliability_ou25", "bankroll_path", "by_league"}}, indent=2, default=str))
        return

    if args.asian_odds:
        print(json.dumps(asian_odds_pipeline(), indent=2))
        return

    if args.tipsters:
        print(json.dumps(tipsters_pipeline(), indent=2, default=str))
        return

    if args.odds_update:
        print(json.dumps(refresh_odds_pipeline(), indent=2))
        return

    if args.train or not (ROOT / "data" / "models" / "best_model.joblib").exists():
        info = train_pipeline()
        print(json.dumps({k: v for k, v in info.items() if k != "path"}, indent=2))
        print("modello:", info["path"])

    result = None
    if args.predict:
        result = predict_pipeline(args.predict[0], args.predict[1], n_sims=args.sims)
        print(json.dumps(result, indent=2))

    if args.advise or args.odds:
        if result is None:
            dest = OUT_DIR / "last_prediction.json"
            if not dest.exists():
                parser.error("Nessuna predizione: usa --predict HOME AWAY oppure genera last_prediction.json")
            result = json.loads(dest.read_text(encoding="utf-8"))
        odds = None
        if args.odds:
            odds = {"1": args.odds[0], "X": args.odds[1], "2": args.odds[2]}
        advice = advise(result, odds)
        print(format_advice(advice))
        print(json.dumps(advice["play"], indent=2))
    elif not args.train and not args.predict and not args.calibrate:
        parser.print_help()


if __name__ == "__main__":
    main()
