"""Coverage corner Big 5 su upcoming_predictions.json (e opz. cache Kambi)."""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _is_big5(row: dict) -> bool:
    league = str(row.get("league") or "").lower()
    country = str(row.get("country") or "").lower()
    blob = f"{country} {league}"
    if any(
        x in blob
        for x in ("serie b", "ligue 2", "2. bundes", "segunda", "championship", "league one")
    ):
        return False
    # league-only (file senza country) o country+league
    if league in {"premier league", "la liga", "serie a", "bundesliga", "ligue 1"}:
        if "premier" in league and country and not any(
            c in country for c in ("eng", "england", "gb", "uk")
        ):
            return False
        return True
    rules = (
        (("england", "eng"), ("premier league", "premier")),
        (("spain", "esp"), ("la liga", "laliga")),
        (("italy", "ita", "italia"), ("serie a",)),
        (("germany", "ger", "deu", "germania"), ("bundesliga",)),
        (("france", "fra", "francia"), ("ligue 1", "ligue1")),
    )
    for countries, leagues in rules:
        if any(c in country for c in countries) and any(lg in league for lg in leagues):
            return True
    return False


def _corner_keys(odds: dict) -> list[str]:
    return [
        k
        for k in odds
        if str(k).startswith("corners_over_") or str(k).startswith("corners_under_")
    ]


def main() -> None:
    path = ROOT / "data" / "processed" / "upcoming_predictions.json"
    if not path.is_file():
        print("NO upcoming_predictions.json")
        raise SystemExit(1)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        print("upcoming non-list")
        raise SystemExit(1)

    today = date.today().isoformat()
    big5_all = [r for r in rows if isinstance(r, dict) and _is_big5(r)]
    big5 = [r for r in big5_all if str(r.get("date") or "")[:10] >= today]
    n_stale = len(big5_all) - len(big5)
    n = len(big5)
    with_corners = 0
    with_real = 0
    matched_kambi = 0
    samples: list[str] = []
    for r in big5:
        odds = r.get("odds") if isinstance(r.get("odds"), dict) else {}
        ck = _corner_keys(odds)
        play = r.get("play") or {}
        src = str(
            r.get("odds_source") or odds.get("odds_source") or play.get("odds_source") or ""
        )
        if "kambi" in src.lower() or odds.get("side_odds_source") == "kambi_unibet":
            matched_kambi += 1
        if ck:
            with_corners += 1
            ok_real = False
            for k in ck:
                try:
                    if float(odds[k]) > 1.01:
                        ok_real = True
                        break
                except (TypeError, ValueError):
                    pass
            if ok_real:
                with_real += 1
                if len(samples) < 8:
                    samples.append(
                        f"{r.get('date')} {r.get('home')}-{r.get('away')} "
                        f"n_lines={len(ck)} src={src or odds.get('side_odds_source')}"
                    )

    pct = (100.0 * with_real / n) if n else 0.0
    print("=== Big 5 corner coverage (upcoming, future KO) ===")
    print(f"n_big5_future: {n} (excluded_past={n_stale})")
    print(f"with_corners_keys: {with_corners}")
    print(f"with_corners_real: {with_real} ({pct:.1f}%)")
    print(f"odds_source_kambi_hint: {matched_kambi}")
    for s in samples:
        print(" sample:", s)

    kb = ROOT / "data" / "raw" / "kambi_football_odds.json"
    if kb.is_file():
        raw = json.loads(kb.read_text(encoding="utf-8"))
        ev = raw.get("events") or []
        n_corn_cached = sum(
            1
            for e in ev
            if isinstance(e, dict) and any(str(k).startswith("corners_") for k in e)
        )
        print(f"kambi_cache_events: {len(ev)} with_prefetched_corners: {n_corn_cached}")
        print(f"kambi_fetched_at: {raw.get('fetched_at')} from_cache={raw.get('from_cache')}")
        if raw.get("n_big5") is not None:
            print(f"kambi_n_big5: {raw.get('n_big5')} listview={raw.get('n_big5_listview')}")

    print(f"coverage_pct: {pct:.1f}")


if __name__ == "__main__":
    main()
