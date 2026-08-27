"""Apprendimento cloud: archive → settle → online_learn (job giornaliero GitHub Actions).

Usato dal workflow «Aggiorna dati e modello» (07:00 IT) con CLOUD_DAILY_LEARN=1.
Persistenza: cache Actions su our_history.sqlite + JSON calibrazione/residual.
Uso:
  python scripts/cloud_learn.py              # auto (leggero se possibile)
  python scripts/cloud_learn.py --settle     # solo chiude esiti + learn
  python scripts/cloud_learn.py --full       # build_upcoming completo (lento)
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS = ROOT / "data" / "models"
PROCESSED = ROOT / "data" / "processed"
HISTORY_DB = PROCESSED / "our_history.sqlite"
UPCOMING = PROCESSED / "upcoming_predictions.json"
MODEL = MODELS / "best_model.joblib"
FEATURES = PROCESSED / "features.csv"

LEARN_JSON = (
    "calibration.json",
    "residual_ev.json",
    "data_signal_weights.json",
    "online_learn_report.json",
)


def _has_model() -> bool:
    return MODEL.is_file() and FEATURES.is_file()


def checkpoint_history_db() -> None:
    """Flush WAL prima di salvare la cache Actions."""
    if not HISTORY_DB.is_file():
        return
    conn = sqlite3.connect(HISTORY_DB)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.commit()
    finally:
        conn.close()


def _settle_only() -> dict:
    from modules.data_update.history import settle_pending

    out = settle_pending()
    out["mode"] = "settle"
    return out


def _light_refresh() -> dict:
    from modules.data_update.upcoming import refresh_upcoming_odds

    ref = refresh_upcoming_odds(archive=True)
    ref["mode"] = "light"
    return ref


def _full_build(*, n_sims: int = 400) -> dict:
    from modules.data_update.download import download_fixtures, download_season_zip
    from modules.data_update.leagues import SEASON_ZIPS
    from modules.data_update.upcoming import build_upcoming

    download_fixtures()
    download_season_zip(SEASON_ZIPS[-1])
    try:
        from modules.data_update.cups import download_org_cups

        cups = download_org_cups()
        if cups.get("error"):
            print(f"skip coppe: {cups['error']}", flush=True)
    except Exception as exc:
        print(f"skip coppe: {exc}", flush=True)
    # Compat: su main remoto build_upcoming può non avere ancora reuse_predictions.
    import inspect

    kwargs: dict = {"n_sims": n_sims}
    if "reuse_predictions" in inspect.signature(build_upcoming).parameters:
        kwargs["reuse_predictions"] = False
    rows = build_upcoming(**kwargs)
    return {"mode": "full", "n_upcoming": len(rows), "ok": True}


def cloud_learn(*, mode: str = "auto") -> dict:
    """Archive (se possibile) + settle + learn_from_settled."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    info: dict = {"cloud_learn": True, "has_model": _has_model(), "has_history": HISTORY_DB.is_file()}
    force_full = os.getenv("CLOUD_FULL_REBUILD", "").lower() in {"1", "true", "yes", "on"}
    m = (mode or "auto").strip().lower()

    try:
        if m == "settle":
            info.update(_settle_only())
        elif m == "light":
            if not UPCOMING.is_file():
                info.update({"ok": False, "error": "manca upcoming_predictions.json", "mode": "light"})
            else:
                info.update(_light_refresh())
        elif m == "full" or force_full:
            if not _has_model():
                info.update(_settle_only())
                info["skipped_full"] = "modello assente"
            else:
                info.update(_full_build())
        else:
            # auto
            if not _has_model():
                info.update(_settle_only())
                info["note"] = "senza modello: solo settle+learn su storico cache"
            else:
                from modules.data_update.upcoming import _model_newer_than_upcoming

                need_full = force_full or not UPCOMING.is_file() or _model_newer_than_upcoming()
                if need_full:
                    info.update(_full_build())
                else:
                    info.update(_light_refresh())
    except Exception as exc:
        info["ok"] = False
        info["error"] = str(exc)
        print(f"cloud_learn errore: {exc}", flush=True)
        try:
            settled = _settle_only()
            info["settle_fallback"] = settled
        except Exception as exc2:
            info["settle_fallback_error"] = str(exc2)

    try:
        from modules.data_update.history import history_summary

        hs = history_summary()
        info["history"] = {
            "n_history": hs.get("n_history"),
            "n_settled": hs.get("n_settled"),
            "n_rich": hs.get("n_rich"),
        }
    except Exception as exc:
        info["history_error"] = str(exc)

    ol = info.get("online_learn") or info.get("settle_fallback", {}).get("online_learn")
    if isinstance(info.get("settle_fallback"), dict):
        ol = ol or info["settle_fallback"].get("online_learn")
    if ol:
        info["online_learn"] = ol

    checkpoint_history_db()
    return info


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Apprendimento cloud (archive/settle/learn)")
    parser.add_argument("--settle", action="store_true", help="solo settle + learn")
    parser.add_argument("--full", action="store_true", help="build_upcoming completo")
    parser.add_argument("--light", action="store_true", help="solo refresh quote sul calendario")
    args = parser.parse_args()
    mode = "auto"
    if args.settle:
        mode = "settle"
    elif args.full:
        mode = "full"
    elif args.light:
        mode = "light"
    info = cloud_learn(mode=mode)
    print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
