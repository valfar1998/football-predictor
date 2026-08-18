"""Allena modello e features.csv su GitHub Actions (dati football-data.co.uk)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    from modules.data_update.download import download_extra_leagues, download_fixtures, download_season_zip
    from modules.data_update.leagues import SEASON_ZIPS
    from modules.dataset_loader import DatasetLoader
    from modules.feature_engineering import FeatureEngineer
    from modules.model_training import ModelTrainer

    seasons_ok = []
    for season in SEASON_ZIPS:
        if download_season_zip(season):
            seasons_ok.append(season)
    extra = download_extra_leagues()
    download_fixtures()
    loader = DatasetLoader()
    matches, matches_path = loader.run("matches.csv")
    engineer = FeatureEngineer(window=5)
    features = engineer.transform(matches)
    feat_path = engineer.save(features, "features.csv")
    train_info = ModelTrainer().train(features)
    info = {
        "cloud": True,
        "seasons": seasons_ok,
        "extra_files": len(extra),
        "n_matches": int(len(matches)),
        "n_features": int(len(features)),
        "matches_path": str(matches_path),
        "features_path": str(feat_path),
        **{k: v for k, v in train_info.items() if k != "model"},
    }
    print(json.dumps(info, indent=2, default=str))
    model = ROOT / "data" / "models" / "best_model.joblib"
    if not model.is_file():
        raise SystemExit("bootstrap: manca data/models/best_model.joblib")


if __name__ == "__main__":
    main()
