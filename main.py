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


def refresh_odds_pipeline(*, asian: bool = True) -> dict:
    download_fixtures()
    download_season_zip(SEASON_ZIPS[-1])
    asian_info: dict = {}
    if asian:
        rows = fetch_asian_odds(days=14, book="bet365")
        path = save_asian_odds(rows)
        asian_info = {"n_asian": len(rows), "asian_cache": str(path)}
    try:
        from modules.data_update.clubelo import fetch_clubelo

        elo = fetch_clubelo()
        asian_info["n_clubelo"] = 0 if elo is None or elo.empty else int(len(elo))
    except Exception as exc:
        asian_info["clubelo_error"] = str(exc)
    try:
        from modules.data_update.cups import download_org_cups

        cups_info = download_org_cups(days=14)
    except Exception as exc:
        cups_info = {"n_cup_files": 0, "error": str(exc)}
        print(f"skip coppe org: {exc}")
    try:
        from modules.data_update.world_fixtures import download_world_fixtures

        world_info = download_world_fixtures(days=14)
    except Exception as exc:
        world_info = {"n_world_fixtures": 0, "error": str(exc)}
        print(f"skip calendario mondiale: {exc}")
    try:
        from modules.data_update.thesportsdb import download_cup_fixtures

        tsdb_info = download_cup_fixtures()
    except Exception as exc:
        tsdb_info = {"n_cup_files": 0, "error": str(exc)}
        print(f"skip coppe TheSportsDB: {exc}")
    try:
        from modules.data_update.api_football import download_cup_fixtures as download_api_football_cups

        apif_info = download_api_football_cups(days=14)
    except Exception as exc:
        apif_info = {"n_cup_files": 0, "error": str(exc)}
        print(f"skip coppe API-Football: {exc}")
    try:
        from modules.data_update.fbref_context import download_fbref_context

        fbref_info = download_fbref_context()
    except Exception as exc:
        fbref_info = {"ok": False, "n_teams": 0, "error": str(exc)}
        print(f"skip FBref context: {exc}")
    try:
        from modules.data_update.understat_context import download_understat_context

        understat_info = download_understat_context()
    except Exception as exc:
        understat_info = {"ok": False, "n_teams": 0, "error": str(exc)}
        print(f"skip Understat context: {exc}")
    try:
        from modules.data_update.statsbomb_context import download_statsbomb_context

        statsbomb_info = download_statsbomb_context()
    except Exception as exc:
        statsbomb_info = {"ok": False, "n_teams": 0, "error": str(exc)}
        print(f"skip StatsBomb context: {exc}")
    tips_info: dict = {}
    try:
        tips = fetch_tipsters()
        tips_info = {"n_tipsters": tips.get("n"), "tipster_counts": tips.get("counts"), "tipster_errors": tips.get("errors")}
    except Exception as exc:
        tips_info = {"tipster_error": str(exc)}
    upcoming = build_upcoming()
    return {
        "n_upcoming": len(upcoming),
        "source": "football-data.co.uk + football-data.org + thesportsdb + world + asianbetsoccer",
        **asian_info,
        **tips_info,
        **{f"cups_{k}": v for k, v in cups_info.items()},
        **{f"tsdb_{k}": v for k, v in tsdb_info.items()},
        **{f"apif_{k}": v for k, v in apif_info.items()},
        **{f"world_{k}": v for k, v in world_info.items()},
        **{f"fbref_{k}": v for k, v in fbref_info.items()},
        **{f"understat_{k}": v for k, v in understat_info.items()},
        **{f"statsbomb_{k}": v for k, v in statsbomb_info.items()},
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
        upcoming = build_upcoming()
        return {"n_upcoming": len(upcoming), "days": days, "book": book, **asian_info}
    finally:
        try:
            lock.unlink(missing_ok=True)
        except OSError:
            pass


def asian_odds_pipeline(*, days: int = 14, book: str = "bet365") -> dict:
    rows = fetch_asian_odds(days=days, book=book)
    path = save_asian_odds(rows)
    tips_info: dict = {}
    try:
        tips = fetch_tipsters()
        tips_info = {"n_tipsters": tips.get("n"), "tipster_counts": tips.get("counts")}
    except Exception as exc:
        tips_info = {"tipster_error": str(exc)}
    elo_info: dict = {}
    try:
        from modules.data_update.clubelo import fetch_clubelo

        elo = fetch_clubelo()
        elo_info["n_clubelo"] = 0 if elo is None or elo.empty else int(len(elo))
    except Exception as exc:
        elo_info["clubelo_error"] = str(exc)
    tsdb_info: dict = {}
    try:
        from modules.data_update.thesportsdb import download_cup_fixtures

        tsdb_info = download_cup_fixtures()
    except Exception as exc:
        tsdb_info = {"error": str(exc)}
    apif_info: dict = {}
    try:
        from modules.data_update.api_football import download_cup_fixtures as download_api_football_cups

        apif_info = download_api_football_cups(days=14)
    except Exception as exc:
        apif_info = {"error": str(exc)}
    fbref_info: dict = {}
    try:
        from modules.data_update.fbref_context import download_fbref_context

        fbref_info = download_fbref_context()
    except Exception as exc:
        fbref_info = {"error": str(exc)}
    understat_info: dict = {}
    try:
        from modules.data_update.understat_context import download_understat_context

        understat_info = download_understat_context()
    except Exception as exc:
        understat_info = {"error": str(exc)}
    upcoming = build_upcoming()
    return {
        "n_asian": len(rows),
        "asian_cache": str(path),
        "n_upcoming": len(upcoming),
        "book": book,
        "days": days,
        **tips_info,
        **elo_info,
        **{f"tsdb_{k}": v for k, v in tsdb_info.items()},
        **{f"apif_{k}": v for k, v in apif_info.items()},
        **{f"fbref_{k}": v for k, v in fbref_info.items()},
        **{f"understat_{k}": v for k, v in understat_info.items()},
    }


def tipsters_pipeline() -> dict:
    info = fetch_tipsters()
    upcoming = build_upcoming()
    return {k: v for k, v in info.items() if k != "matches"} | {"n_upcoming": len(upcoming)}


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


def predict_pipeline(home: str, away: str, n_sims: int = 10_000) -> dict:
    predictor = MatchPredictor()
    pred = predictor.predict(home, away)
    sim = MonteCarloSimulator(n_sims=n_sims).simulate(
        pred["lambda_home"],
        pred["lambda_away"],
        model_probs={"home_win": pred["home_win"], "draw": pred["draw"], "away_win": pred["away_win"]},
    )
    out = {
        "match": f"{pred['home_team']} vs {pred['away_team']}",
        "model_probabilities": {
            "home_win": pred["home_win"],
            "draw": pred["draw"],
            "away_win": pred["away_win"],
        },
        "expected_goals": {"home": pred["lambda_home"], "away": pred["lambda_away"]},
        "features": pred.get("features") or {},
        "fbref_context": {
            "home": lookup_team_context(pred["home_team"], load_fbref_team_index()),
            "away": lookup_team_context(pred["away_team"], load_fbref_team_index()),
        },
        "understat_context": {
            "home": lookup_understat_team(pred["home_team"], load_understat_team_index()),
            "away": lookup_understat_team(pred["away_team"], load_understat_team_index()),
        },
        "montecarlo": sim,
    }
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
    parser.add_argument("--update", action="store_true", help="scarica dati mondiali + quote, allena, calendario")
    parser.add_argument("--odds-update", action="store_true", help="aggiorna fixtures/quote (incluso AsianBetSoccer) e pronostici")
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

    if args.update:
        info = update_pipeline(retrain=True)
        print(json.dumps({k: v for k, v in info.items() if k != "model"}, indent=2, default=str))
        if info.get("model"):
            print("modello:", info["model"])
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
