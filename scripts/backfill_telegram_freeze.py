"""Backfill freeze journal + SQLite da alert Telegram persi (cache GHA / no persist)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.data_update.history import (  # noqa: E402
    FREEZE_JOURNAL,
    _key,
    _save_freeze_journal,
    apply_freeze_journal,
    _load_freeze_journal,
)


# Snapshot alert Telegram 2026-09-14 (pre-persistenza)
BACKFILL_2026_09_14 = [
    {
        "date": "2026-09-14",
        "time": "18:30",
        "home": "Como",
        "away": "Parma",
        "league": "Serie A",
        "country": "Italia",
        "pick": "CORNO10.5",
        "pick_group": "corners",
        "pick_label": "Corner Over 10.5",
        "action": "gioca",
        "score": 8,
        "score_unified": 8,
        "quota_pick": 2.8,
        "ev_cons": 0.109,
        "probability": 0.36,
        "alert_kind": "gioca",
        "alert_frozen_at": "2026-09-14T10:01:05Z",
        "odds_source": "telegram_backfill",
    },
    {
        "date": "2026-09-14",
        "time": "20:45",
        "home": "Inter",
        "away": "Udinese",
        "league": "Serie A",
        "country": "Italia",
        "pick": "CORNU7.5",
        "pick_group": "corners",
        "pick_label": "Corner Under 7.5",
        "action": "gioca",
        "score": 8,
        "score_unified": 8,
        "quota_pick": 3.05,
        "ev_cons": 0.198,
        "probability": 0.36,
        "alert_kind": "gioca",
        "alert_frozen_at": "2026-09-14T10:01:05Z",
        "odds_source": "telegram_backfill",
    },
    {
        "date": "2026-09-14",
        "time": "18:00",
        "home": "Gaziantep",
        "away": "Fenerbahce",
        "league": "Super Lig",
        "country": "Turchia",
        "pick": "2",
        "pick_group": "1x2",
        "pick_label": "Fenerbahce vince",
        "action": "gioca",
        "score": 8,
        "score_unified": 8,
        "quota_pick": 2.1,
        "ev_cons": 0.086,
        "probability": 0.55,
        "alert_kind": "gioca",
        "alert_frozen_at": "2026-09-14T10:30:00Z",
        "odds_source": "telegram_backfill",
    },
]

# Snapshot alert Telegram 2026-09-19 07:13 Roma (GIOCA voto ≥8) — mancanti da journal/DB
BACKFILL_2026_09_19 = [
    {
        "date": "2026-09-19",
        "time": "15:00",
        "home": "Blackpool",
        "away": "Plymouth",
        "league": "League One",
        "country": "Inghilterra",
        "pick": "1",
        "pick_group": "1x2",
        "pick_label": "Blackpool vince",
        "action": "gioca",
        "score": 8,
        "score_unified": 8,
        "quota_pick": 2.63,
        "ev_cons": 0.113,
        "probability": 0.43,
        "alert_kind": "gioca",
        "alert_frozen_at": "2026-09-19T05:13:00Z",
        "odds_source": "telegram_backfill",
    },
    {
        "date": "2026-09-19",
        "time": "15:00",
        "home": "Leyton Orient",
        "away": "Stevenage",
        "league": "League One",
        "country": "Inghilterra",
        "pick": "1",
        "pick_group": "1x2",
        "pick_label": "Leyton Orient vince",
        "action": "gioca",
        "score": 8,
        "score_unified": 8,
        "quota_pick": 2.4,
        "ev_cons": 0.085,
        "probability": 0.46,
        "alert_kind": "gioca",
        "alert_frozen_at": "2026-09-19T05:13:00Z",
        "odds_source": "telegram_backfill",
    },
    {
        "date": "2026-09-19",
        "time": "15:00",
        "home": "Sheffield Weds",
        "away": "Stockport",
        "league": "League One",
        "country": "Inghilterra",
        "pick": "O2.5",
        "pick_group": "ou",
        "pick_label": "Over 2.5",
        "action": "gioca",
        "score": 8,
        "score_unified": 8,
        "quota_pick": 2.03,
        "ev_cons": 0.243,
        "probability": 0.62,
        "alert_kind": "gioca",
        "alert_frozen_at": "2026-09-19T05:13:00Z",
        "odds_source": "telegram_backfill",
    },
    {
        "date": "2026-09-20",
        "time": "01:30",
        "home": "FC Dallas",
        "away": "Austin FC",
        "league": "MLS",
        "country": "USA",
        "pick": "1",
        "pick_group": "1x2",
        "pick_label": "FC Dallas vince",
        "action": "gioca",
        "score": 8,
        "score_unified": 8,
        "quota_pick": 1.7,
        "ev_cons": 0.07,
        "probability": 0.65,
        "alert_kind": "gioca",
        "alert_frozen_at": "2026-09-19T05:13:00Z",
        "odds_source": "telegram_backfill",
    },
    {
        "date": "2026-09-20",
        "time": "09:00",
        "home": "Machida",
        "away": "Kashiwa Reysol",
        "league": "J1 League",
        "country": "Giappone",
        "pick": "1",
        "pick_group": "1x2",
        "pick_label": "Machida vince",
        "action": "gioca",
        "score": 8,
        "score_unified": 8,
        "quota_pick": 2.55,
        "ev_cons": 0.151,
        "probability": 0.49,
        "alert_kind": "gioca",
        "alert_frozen_at": "2026-09-19T05:13:00Z",
        "odds_source": "telegram_backfill",
    },
    {
        "date": "2026-09-20",
        "time": "01:30",
        "home": "Minnesota United",
        "away": "Los Angeles Galaxy",
        "league": "MLS",
        "country": "USA",
        "pick": "1",
        "pick_group": "1x2",
        "pick_label": "Minnesota United vince",
        "action": "gioca",
        "score": 8,
        "score_unified": 8,
        "quota_pick": 1.67,
        "ev_cons": 0.035,
        "probability": 0.62,
        "alert_kind": "gioca",
        "alert_frozen_at": "2026-09-19T05:13:00Z",
        "odds_source": "telegram_backfill",
    },
    {
        "date": "2026-09-22",
        "time": "01:15",
        "home": "Lanus",
        "away": "Estudiantes L.P.",
        "league": "Liga Profesional",
        "country": "Argentina",
        "pick": "1",
        "pick_group": "1x2",
        "pick_label": "Lanus vince",
        "action": "gioca",
        "score": 8,
        "score_unified": 8,
        "quota_pick": 2.25,
        "ev_cons": 0.035,
        "probability": 0.46,
        "alert_kind": "gioca",
        "alert_frozen_at": "2026-09-19T05:13:00Z",
        "odds_source": "telegram_backfill",
    },
]

BACKFILL = BACKFILL_2026_09_14 + BACKFILL_2026_09_19


def main() -> None:
    data = _load_freeze_journal()
    added = 0
    for row in BACKFILL:
        key = _key(row)
        entry = {"match_key": key, **row}
        if key not in data:
            data[key] = entry
            added += 1
        elif not data[key].get("alert_frozen_at"):
            data[key] = entry
            added += 1
    _save_freeze_journal(data)
    applied = apply_freeze_journal()
    print("journal", FREEZE_JOURNAL)
    print("added_entries", added, "n_journal", len(data))
    print("apply", applied)


if __name__ == "__main__":
    main()
