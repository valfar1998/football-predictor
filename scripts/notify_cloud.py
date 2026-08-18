"""Refresh cloud: spread Asian Raro + voto unificato >=9 se il modello e in cache."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL = ROOT / "data" / "models" / "best_model.joblib"
FEATURES = ROOT / "data" / "processed" / "features.csv"


def _has_model() -> bool:
    return MODEL.is_file() and FEATURES.is_file()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    from modules.data_update.asian_odds import fetch_asian_odds, save_asian_odds
    from modules.notify import dispatch_alerts
    from modules.notify.telegram import telegram_status

    print(telegram_status())
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "models").mkdir(parents=True, exist_ok=True)

    rows = fetch_asian_odds(days=4, book="bet365")
    info: dict = {"n_asian": len(rows), "cloud": True, "has_model": _has_model()}
    if rows:
        path = save_asian_odds(rows)
        info["asian_cache"] = str(path)
    else:
        print("asian vuoto: niente cache nuova")
        info["kept_previous_cache"] = True

    upcoming: list[dict] | None = []
    if _has_model():
        try:
            from modules.data_update.download import download_fixtures, download_season_zip
            from modules.data_update.leagues import SEASON_ZIPS
            from modules.data_update.upcoming import build_upcoming

            download_fixtures()
            download_season_zip(SEASON_ZIPS[-1])
            upcoming = build_upcoming(n_sims=400)
            info["n_upcoming"] = len(upcoming)
        except Exception as exc:
            print(f"skip calendario modello: {exc}")
            upcoming = []
            info["upcoming_error"] = str(exc)
    else:
        print("modello assente: solo spread Raro. Esegui il workflow Cloud train.")

    alerts = dispatch_alerts(upcoming=upcoming)
    info.update({k: v for k, v in alerts.items() if k != "status"})
    print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
