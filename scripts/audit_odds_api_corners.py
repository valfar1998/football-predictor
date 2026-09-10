"""Audit The Odds API: copertura mercati corner (Pinnacle) su Big 5 + UCL.

Uso:
  python scripts/audit_odds_api_corners.py
  python scripts/audit_odds_api_corners.py --max-events 1

Salva report in data/processed/odds_api_corners_audit.json
Consuma crediti API (events gratis-ish + 1 odds call per evento campionato).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.data_update.odds_api import (  # noqa: E402
    BASE,
    CORNER_MARKETS,
    CORE_SPORT_KEYS,
    _api_key,
    _get,
    _remaining,
)

OUT = ROOT / "data" / "processed" / "odds_api_corners_audit.json"


def _audit_sport(key: str, title: str, *, max_events: int, api_key: str) -> dict:
    row: dict = {"sport_key": key, "title": title, "n_events": 0, "samples": [], "error": None}
    try:
        events, headers = _get(f"{BASE}/sports/{key}/events", api_key)
        row["remaining_after_events"] = _remaining(headers)
    except Exception as exc:
        row["error"] = f"events: {exc}"
        return row
    if not isinstance(events, list):
        row["error"] = "events payload non lista"
        return row
    row["n_events"] = len(events)
    # Preferisci kickoff più vicini
    events_sorted = sorted(events, key=lambda e: str(e.get("commence_time") or ""))
    for ev in events_sorted[: max(0, max_events)]:
        eid = str(ev.get("id") or "")
        if not eid:
            continue
        sample = {
            "event_id": eid,
            "home": ev.get("home_team"),
            "away": ev.get("away_team"),
            "commence_time": ev.get("commence_time"),
            "markets": {},
            "lines_totals_corners": [],
            "error": None,
        }
        markets = ",".join(CORNER_MARKETS)
        url = (
            f"{BASE}/sports/{key}/events/{eid}/odds"
            f"?regions=eu&markets={markets}&bookmakers=pinnacle&oddsFormat=decimal"
        )
        try:
            data, headers = _get(url, api_key)
            sample["remaining"] = _remaining(headers)
        except Exception as exc:
            sample["error"] = str(exc)
            row["samples"].append(sample)
            continue
        for bm in (data.get("bookmakers") or []) if isinstance(data, dict) else []:
            if str(bm.get("key") or "").lower() != "pinnacle":
                continue
            for mkt in bm.get("markets") or []:
                mk = str(mkt.get("key") or "")
                outcomes = mkt.get("outcomes") or []
                sample["markets"][mk] = len(outcomes)
                if mk == "alternate_totals_corners":
                    for o in outcomes:
                        try:
                            pt = float(o.get("point"))
                            name = str(o.get("name") or "")
                            price = float(o.get("price"))
                        except (TypeError, ValueError):
                            continue
                        if name.lower() in {"over", "under"} and 7.0 <= pt <= 13.0:
                            sample["lines_totals_corners"].append(
                                {"name": name, "point": pt, "price": round(price, 3)}
                            )
        row["samples"].append(sample)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit Odds API corner markets")
    ap.add_argument("--max-events", type=int, default=1, help="Eventi per sport (default 1)")
    args = ap.parse_args()
    key = _api_key()
    if not key:
        print("ODDS_API_KEY / data/raw/odds-api.key assente", flush=True)
        return 1

    sports = list(CORE_SPORT_KEYS)
    report = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "markets_requested": list(CORNER_MARKETS),
        "sports": [],
        "summary": {},
    }
    ok_corners = 0
    for sport_key, title in sports:
        print(f"Audit {sport_key}…", flush=True)
        block = _audit_sport(sport_key, title, max_events=args.max_events, api_key=key)
        report["sports"].append(block)
        for s in block.get("samples") or []:
            if s.get("markets", {}).get("alternate_totals_corners"):
                ok_corners += 1
        print(
            f"  events={block.get('n_events')} samples={len(block.get('samples') or [])} "
            f"err={block.get('error')}",
            flush=True,
        )

    report["summary"] = {
        "sports_n": len(sports),
        "samples_with_totals_corners": ok_corners,
        "remaining": report["sports"][-1].get("samples")[-1].get("remaining")
        if report["sports"] and report["sports"][-1].get("samples")
        else None,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK report -> {OUT}", flush=True)
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
