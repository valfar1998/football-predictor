"""Orchestratore: dati → feature → training → prediction → Monte Carlo → JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from modules.advisor import advise, format_advice
from modules.dataset_loader import DatasetLoader
from modules.feature_engineering import FeatureEngineer
from modules.model_training import ModelTrainer
from modules.montecarlo import MonteCarloSimulator
from modules.predictor import MatchPredictor

OUT_DIR = ROOT / "data" / "processed"


def ensure_raw_sample() -> None:
    raw = ROOT / "data" / "raw"
    if any(raw.glob("*.csv")):
        return
    from scripts.generate_sample_data import main as gen

    gen()


def train_pipeline() -> dict:
    ensure_raw_sample()
    loader = DatasetLoader()
    matches, matches_path = loader.run("matches.csv")
    engineer = FeatureEngineer(window=5)
    features = engineer.transform(matches)
    feat_path = engineer.save(features, "features.csv")
    trainer = ModelTrainer()
    train_info = trainer.train(features)
    return {
        "n_matches": int(len(matches)),
        "n_features_rows": int(len(features)),
        "matches_path": str(matches_path),
        "features_path": str(feat_path),
        **train_info,
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
        "montecarlo": sim,
    }
    dest = OUT_DIR / "last_prediction.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    out["saved_to"] = str(dest)
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Football predictor")
    parser.add_argument("--train", action="store_true", help="ricostruisce dataset, feature e modello")
    parser.add_argument("--predict", nargs=2, metavar=("HOME", "AWAY"), help="es. --predict Inter Milan")
    parser.add_argument("--odds", nargs=3, type=float, metavar=("ODD_1", "ODD_X", "ODD_2"), help="quote decimali 1 X 2")
    parser.add_argument("--advise", action="store_true", help="consiglio 1X2 sulla ultima predizione (o su --predict)")
    parser.add_argument("--ui", action="store_true", help="apre l'interfaccia Streamlit")
    parser.add_argument("--sims", type=int, default=10_000)
    args = parser.parse_args()

    if args.ui:
        import subprocess

        raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py")]))

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
    elif not args.train and not args.predict:
        parser.print_help()


if __name__ == "__main__":
    main()
