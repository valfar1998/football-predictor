"""Refresh cloud: spread Asian Raro + voto ≥9; apprendimento solo nel job giornaliero."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _daily_learn_enabled() -> bool:
    return os.getenv("CLOUD_DAILY_LEARN", "").lower() in {"1", "true", "yes", "on"}


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
    info: dict = {"n_asian": len(rows), "cloud": True, "daily_learn": _daily_learn_enabled()}
    if rows:
        path = save_asian_odds(rows)
        info["asian_cache"] = str(path)
    else:
        print("asian vuoto: niente cache nuova")
        info["kept_previous_cache"] = True

    if _daily_learn_enabled():
        try:
            from modules.data_update.odds_api import fetch_pinnacle_odds

            pinn = fetch_pinnacle_odds(max_age_hours=20.0)
            info["pinnacle_events"] = pinn.get("n_events", 0)
            info["pinnacle_from_cache"] = pinn.get("from_cache")
            info["pinnacle_remaining"] = pinn.get("remaining")
            if not pinn.get("ok") and pinn.get("error"):
                info["pinnacle_error"] = pinn["error"]
                print(f"skip Pinnacle: {pinn['error']}")
        except Exception as exc:
            info["pinnacle_error"] = str(exc)
            print(f"skip Pinnacle: {exc}")

        try:
            from modules.data_update.betfair import fetch_betfair_odds

            bf = fetch_betfair_odds(force=True, days=7)
            info["betfair_events"] = bf.get("n_events", 0)
            info["betfair_from_cache"] = bf.get("from_cache")
            if not bf.get("ok") and bf.get("error"):
                info["betfair_error"] = bf["error"]
                info["betfair_soft_fail"] = True
                print(f"betfair_soft_fail: {bf['error']}")
        except Exception as exc:
            info["betfair_error"] = str(exc)
            info["betfair_soft_fail"] = True
            print(f"betfair_soft_fail: {exc}")

        from scripts.cloud_learn import cloud_learn

        mode = os.getenv("CLOUD_LEARN_MODE", "auto").strip() or "auto"
        if os.getenv("CLOUD_FULL_REBUILD", "").lower() in {"1", "true", "yes", "on"}:
            mode = "full"
        learn_info = cloud_learn(mode=mode)
        info.update(learn_info)

    upcoming: list[dict] = []
    up_path = ROOT / "data" / "processed" / "upcoming_predictions.json"
    if up_path.is_file():
        try:
            raw = json.loads(up_path.read_text(encoding="utf-8"))
            upcoming = raw if isinstance(raw, list) else []
        except (OSError, json.JSONDecodeError):
            upcoming = []
    info["n_upcoming"] = len(upcoming)

    alerts = dispatch_alerts(upcoming=upcoming)
    info.update({k: v for k, v in alerts.items() if k != "status"})
    print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
