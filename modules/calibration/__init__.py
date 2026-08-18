from .backtest import run_backtest
from .calibrate import apply_temperature_dict, calibrate_from_features
from .config import load_calibration, prob_bin_factor, save_calibration

__all__ = [
    "calibrate_from_features",
    "run_backtest",
    "apply_temperature_dict",
    "load_calibration",
    "save_calibration",
    "prob_bin_factor",
    "run_full_calibration",
]


def run_full_calibration() -> dict:
    """Calibrazione 1X2 + backtest EV + salvataggio config."""
    from modules.dataset_loader import DatasetLoader

    feat_path = __import__("pathlib").Path(__file__).resolve().parents[2] / "data" / "processed" / "features.csv"
    if not feat_path.exists():
        loader = DatasetLoader()
        loader.run("matches.csv")
        from modules.feature_engineering import FeatureEngineer

        matches = __import__("pandas").read_csv(
            __import__("pathlib").Path(__file__).resolve().parents[2] / "data" / "processed" / "matches.csv",
            parse_dates=["date"],
        )
        FeatureEngineer().transform(matches).to_csv(feat_path, index=False)

    feat = __import__("pandas").read_csv(feat_path, parse_dates=["date"])
    matches = __import__("pandas").read_csv(
        __import__("pathlib").Path(__file__).resolve().parents[2] / "data" / "processed" / "matches.csv",
        parse_dates=["date"],
    )

    cal = calibrate_from_features(feat)
    bt = run_backtest(feat, matches, temperature=cal["temperature"])
    cal.update(bt)
    cal.setdefault("min_ev_strong_value", max(0.06, cal.get("min_ev_play", 0.025) + 0.02))
    cal.setdefault("min_prob_1x2_value", 0.35)
    cal.setdefault("min_bin_samples", 30)
    cal.setdefault("low_sample_max_score", 6)
    cal.setdefault("kelly_fraction", 0.25)
    cal.setdefault("kelly_cap", bt.get("kelly_cap", 0.02))
    cal.setdefault("liquid_against_rank", 3)
    cal.setdefault("liquid_against_pp", 2.0)
    path = save_calibration(cal)
    cal["path"] = str(path)
    return cal
