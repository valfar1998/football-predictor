"""Refresh cloud (GitHub Actions): solo spread AsianBetSoccer Raro. Nessun modello."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.data_update.asian_odds import fetch_asian_odds, save_asian_odds
from modules.notify import dispatch_alerts
from modules.notify.telegram import telegram_status


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print(telegram_status())
    rows = fetch_asian_odds(days=4, book="bet365")
    info: dict = {"n_asian": len(rows), "cloud": True}
    if rows:
        path = save_asian_odds(rows)
        info["asian_cache"] = str(path)
    else:
        print("asian vuoto: niente cache nuova")
        info["kept_previous_cache"] = True
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    # In cloud non c'è il modello: solo spread Raro, i voti >=9 restano sul PC.
    alerts = dispatch_alerts(upcoming=[])
    info.update({k: v for k, v in alerts.items() if k != "status"})
    print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
