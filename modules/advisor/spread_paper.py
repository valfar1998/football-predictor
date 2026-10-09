"""Paper ROI sugli alert Telegram AsianBetSoccer · Spread Raro (giocabilità).

Campione = solo alert inviato e congelato (journal). Settle da gol in our_history.
ROI unitario sulle quote freeze del lato consigliato (1/2/O/U; AH senza quota → solo hit).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "data" / "processed" / "telegram_spread_freeze.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _f(v: Any) -> float | None:
    try:
        if v is None or isinstance(v, bool):
            return None
        x = float(v)
        if x != x:
            return None
        return x
    except (TypeError, ValueError):
        return None


def _load_journal() -> dict[str, dict[str, Any]]:
    if not JOURNAL.is_file():
        return {}
    try:
        raw = json.loads(JOURNAL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(raw, dict) and isinstance(raw.get("spreads"), dict):
        raw = raw["spreads"]
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            out[str(k)] = v
    return out


def _save_journal(data: dict[str, dict[str, Any]]) -> None:
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": _now(),
        "n": len(data),
        "spreads": data,
    }
    JOURNAL.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _face_key(date: str, home: str, away: str) -> str:
    return f"{str(date)[:10]}|{str(home).strip().lower()}|{str(away).strip().lower()}"


def _primary_bet(
    *,
    follow: str,
    steam_1x2: str | None,
    steam_ah: str | None,
    steam_ou: str | None,
    odd_1: float | None,
    odd_2: float | None,
    odd_over: float | None,
    odd_under: float | None,
    ah_curr: float | None,
    total_curr: float | None,
) -> dict[str, Any]:
    """Sceglie il mercato primario da settle (allineato al testo Segui: …)."""
    fl = (follow or "").lower()
    s1 = str(steam_1x2 or "").strip()
    sah = str(steam_ah or "").strip().lower()
    sou = str(steam_ou or "").strip().lower()

    # 1X2 + AH allineati → preferisci 1X2 (quota europea chiara)
    if "1 e ah casa" in fl or (s1 == "1" and sah == "home"):
        return {"bet": "1", "odds": odd_1, "line": None, "label": "1 (steam casa)"}
    if "2 e ah trasferta" in fl or (s1 == "2" and sah == "away"):
        return {"bet": "2", "odds": odd_2, "line": None, "label": "2 (steam trasferta)"}

    if fl.strip() == "1" or (s1 == "1" and "ah" not in fl):
        return {"bet": "1", "odds": odd_1, "line": None, "label": "1"}
    if fl.strip() == "2" or (s1 == "2" and "ah" not in fl):
        return {"bet": "2", "odds": odd_2, "line": None, "label": "2"}

    if "over" in fl or sou == "over":
        return {
            "bet": "over",
            "odds": odd_over,
            "line": total_curr if total_curr is not None else 2.5,
            "label": f"Over {total_curr if total_curr is not None else 2.5}",
        }
    if "under" in fl or sou == "under":
        return {
            "bet": "under",
            "odds": odd_under,
            "line": total_curr if total_curr is not None else 2.5,
            "label": f"Under {total_curr if total_curr is not None else 2.5}",
        }

    if "ah casa" in fl or sah == "home":
        return {
            "bet": "ah_home",
            "odds": None,
            "line": ah_curr,
            "label": f"AH casa {ah_curr}",
        }
    if "ah trasferta" in fl or sah == "away":
        # Linea book è handicap casa: away cover ≈ −ah_curr
        line = None if ah_curr is None else -float(ah_curr)
        return {
            "bet": "ah_away",
            "odds": None,
            "line": line,
            "label": f"AH trasferta {line}",
        }
    return {"bet": "unknown", "odds": None, "line": None, "label": follow or "n/d"}


def snapshot_from_alert(alert: dict[str, Any]) -> dict[str, Any] | None:
    """Costruisce entry journal da un item alert spread (con row allegata)."""
    row = alert.get("row") if isinstance(alert.get("row"), dict) else {}
    move = row.get("market_move") if isinstance(row.get("market_move"), dict) else {}
    playab = alert.get("playab") if isinstance(alert.get("playab"), dict) else {}
    if not playab:
        from modules.data_update.asian_odds import spread_playability

        playab = spread_playability(row, move)

    day = str(row.get("date") or "")[:10]
    home = str(row.get("home") or "").strip()
    away = str(row.get("away") or "").strip()
    if not day or not home or not away:
        return None

    odds_d = row.get("odds") if isinstance(row.get("odds"), dict) else {}
    odd_1 = _f(row.get("odd_1")) or _f(odds_d.get("1"))
    odd_2 = _f(row.get("odd_2")) or _f(odds_d.get("2"))
    odd_over = _f(row.get("odd_over")) or _f(odds_d.get("over_2.5")) or _f(odds_d.get("over"))
    odd_under = _f(row.get("odd_under")) or _f(odds_d.get("under_2.5")) or _f(odds_d.get("under"))
    ah_curr = _f(move.get("ah_curr", row.get("ah_curr")))
    total_curr = _f(move.get("total_curr", row.get("total_line") or row.get("total_curr")))
    follow = str(playab.get("follow") or alert.get("follow") or "")
    bet_info = _primary_bet(
        follow=follow,
        steam_1x2=str(move.get("steam_1x2") or row.get("steam_1x2") or "") or None,
        steam_ah=str(move.get("steam_ah") or row.get("steam_ah") or "") or None,
        steam_ou=str(move.get("steam_ou") or row.get("steam_ou") or "") or None,
        odd_1=odd_1,
        odd_2=odd_2,
        odd_over=odd_over,
        odd_under=odd_under,
        ah_curr=ah_curr,
        total_curr=total_curr,
    )
    aid = str(alert.get("id") or f"spread|{_face_key(day, home, away)}")
    try:
        play_score = int(alert.get("score") if alert.get("score") is not None else playab.get("score"))
    except (TypeError, ValueError):
        play_score = int(playab.get("score") or 0)

    return {
        "id": aid,
        "match_key": _face_key(day, home, away),
        "date": day,
        "home": home,
        "away": away,
        "league": row.get("league") or "",
        "playability": play_score,
        "verdict": playab.get("verdict") or alert.get("verdict"),
        "follow": follow,
        "reason": playab.get("reason"),
        "line_move": _f(move.get("line_move", row.get("line_move"))),
        "steam_1x2": move.get("steam_1x2") or row.get("steam_1x2"),
        "steam_ah": move.get("steam_ah") or row.get("steam_ah"),
        "steam_ou": move.get("steam_ou") or row.get("steam_ou"),
        "ah_curr": ah_curr,
        "total_curr": total_curr,
        "bet": bet_info["bet"],
        "bet_label": bet_info["label"],
        "odds": bet_info["odds"],
        "line": bet_info["line"],
        "alert_frozen_at": _now(),
        "hit": None,
        "push": None,
        "pnl": None,
        "settled_at": None,
        "home_goals": None,
        "away_goals": None,
    }


def record_spread_freezes(alerts: list[dict[str, Any]]) -> dict[str, Any]:
    """Prima notifica vince: scrive snapshot nel journal."""
    data = _load_journal()
    added = 0
    skipped = 0
    for alert in alerts:
        snap = snapshot_from_alert(alert)
        if not snap:
            skipped += 1
            continue
        key = str(snap["id"])
        if key in data and data[key].get("alert_frozen_at"):
            skipped += 1
            continue
        data[key] = snap
        added += 1
    if added:
        _save_journal(data)
    return {"ok": True, "added": added, "skipped": skipped, "n_journal": len(data), "path": str(JOURNAL)}


def _settle_entry(entry: dict[str, Any], hg: int, ag: int) -> dict[str, Any]:
    tot = hg + ag
    bet = str(entry.get("bet") or "")
    odds = _f(entry.get("odds"))
    line = _f(entry.get("line"))
    hit: int | None = None
    push = False

    if bet == "1":
        hit = 1 if hg > ag else 0
    elif bet == "2":
        hit = 1 if ag > hg else 0
    elif bet == "over":
        ln = line if line is not None else 2.5
        if tot > ln:
            hit = 1
        elif tot < ln:
            hit = 0
        else:
            push = True
            hit = None
    elif bet == "under":
        ln = line if line is not None else 2.5
        if tot < ln:
            hit = 1
        elif tot > ln:
            hit = 0
        else:
            push = True
            hit = None
    elif bet == "ah_home":
        ln = line if line is not None else 0.0
        margin = (hg - ag) + ln
        if abs(margin) < 1e-9:
            push = True
            hit = None
        else:
            hit = 1 if margin > 0 else 0
    elif bet == "ah_away":
        ln = line if line is not None else 0.0
        # line già espressa come handicap away (es. +0.5)
        margin = (ag - hg) + ln
        if abs(margin) < 1e-9:
            push = True
            hit = None
        else:
            hit = 1 if margin > 0 else 0
    else:
        return {**entry, "hit": None, "push": None, "pnl": None}

    pnl = None
    if push:
        pnl = 0.0
    elif hit == 1 and odds is not None and odds > 1.01:
        pnl = round(odds - 1.0, 4)
    elif hit == 0 and odds is not None and odds > 1.01:
        pnl = -1.0
    elif hit == 1:
        pnl = 1.0  # flat se manca quota
    elif hit == 0:
        pnl = -1.0

    out = dict(entry)
    out["home_goals"] = hg
    out["away_goals"] = ag
    out["hit"] = hit
    out["push"] = 1 if push else 0
    out["pnl"] = pnl
    out["settled_at"] = _now()
    return out


def settle_spread_journal() -> dict[str, Any]:
    """Aggancia gol da our_history e chiude le entry pending."""
    from modules.data_update.history import load_history
    from modules.data_update.team_names import resolve_known_team

    data = _load_journal()
    if not data:
        return {"ok": True, "settled": 0, "pending": 0, "n": 0}

    by_face: dict[str, dict] = {}
    for r in load_history():
        day = str(r.get("date") or "")[:10]
        home = resolve_known_team(r.get("home") or "") or r.get("home")
        away = resolve_known_team(r.get("away") or "") or r.get("away")
        if r.get("home_goals") is None or r.get("away_goals") is None:
            continue
        key = _face_key(day, str(home), str(away))
        by_face[key] = r
        # anche nomi raw
        by_face[_face_key(day, str(r.get("home") or ""), str(r.get("away") or ""))] = r

    settled = 0
    pending = 0
    changed = False
    for key, entry in list(data.items()):
        if entry.get("settled_at"):
            continue
        day = str(entry.get("date") or "")[:10]
        home = str(entry.get("home") or "")
        away = str(entry.get("away") or "")
        face = entry.get("match_key") or _face_key(day, home, away)
        hist = by_face.get(face) or by_face.get(_face_key(day, home, away))
        if not hist:
            pending += 1
            continue
        try:
            hg = int(hist["home_goals"])
            ag = int(hist["away_goals"])
        except (TypeError, ValueError, KeyError):
            pending += 1
            continue
        data[key] = _settle_entry(entry, hg, ag)
        settled += 1
        changed = True

    # recount pending after
    pending = sum(1 for e in data.values() if not e.get("settled_at"))
    if changed:
        _save_journal(data)
    return {
        "ok": True,
        "settled": settled,
        "pending": pending,
        "n": len(data),
        "path": str(JOURNAL),
    }


def _subset_roi(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Hit + ROI @ quote su un sottoinsieme decidibile (hit 0/1)."""
    hits = sum(1 for r in rows if int(r.get("hit") or 0) == 1)
    n = len(rows)
    op: list[float] = []
    for r in rows:
        pnl = _f(r.get("pnl"))
        od = _f(r.get("odds"))
        if pnl is not None and od is not None and od > 1.01:
            op.append(pnl)
    return {
        "n": n,
        "hits": hits,
        "hit_rate": round(hits / n, 3) if n else None,
        "odds_n": len(op),
        "odds_pnl": round(sum(op), 2) if op else 0.0,
        "odds_roi": round(sum(op) / len(op), 3) if op else None,
    }


def spread_outcome_phrase(*, playability: int | None = None, min_n: int = 1) -> str | None:
    """Es. '2/5 (40%) · ROI +8%' per giocabilità freeze (moneyway), senza re-settle."""
    data = _load_journal()
    rows = [r for r in data.values() if r.get("settled_at") and r.get("hit") is not None]
    if playability is not None:
        try:
            pv = int(playability)
        except (TypeError, ValueError):
            return None
        rows = [r for r in rows if int(r.get("playability") or 0) == pv]
    stats = _subset_roi(rows)
    n = int(stats["n"] or 0)
    if n < min_n:
        return None
    hits = int(stats["hits"] or 0)
    hr = stats.get("hit_rate")
    label = f"{hits}/{n}" + (f" ({hr:.0%})" if hr is not None else "")
    roi = stats.get("odds_roi")
    if roi is not None:
        return f"{label} · ROI {roi:+.0%}"
    return label


def spread_paper_report() -> dict[str, Any]:
    """Report ROI: unit stake sulle quote freeze (push esclusi dal ROI @ quote)."""
    settle_info = settle_spread_journal()
    data = _load_journal()
    rows = list(data.values())
    pending = [r for r in rows if not r.get("settled_at")]
    settled = [r for r in rows if r.get("settled_at")]
    # decidibili: hit 0/1 (push fuori dal ROI quote)
    decidable = [r for r in settled if r.get("hit") is not None]
    pushes = [r for r in settled if int(r.get("push") or 0) == 1]

    overall = _subset_roi(decidable)
    hits = int(overall["hits"] or 0)
    n = int(overall["n"] or 0)
    flat_pnl = float(hits - (n - hits)) if n else 0.0

    by_vote: dict[str, dict[str, Any]] = {}
    for v in (8, 9, 10):
        subset = [r for r in decidable if int(r.get("playability") or 0) == v]
        stats = _subset_roi(subset)
        by_vote[str(v)] = {
            "vote": v,
            **stats,
            "pending": sum(1 for r in pending if int(r.get("playability") or 0) == v),
        }

    by_bet: dict[str, list] = {}
    for r in decidable:
        by_bet.setdefault(str(r.get("bet") or "n/d"), []).append(r)
    by_market = []
    for k, items in sorted(by_bet.items(), key=lambda t: -len(t[1])):
        stats = _subset_roi(items)
        by_market.append({"key": k, **stats})

    def _row_view(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "date": r.get("date"),
            "home": r.get("home"),
            "away": r.get("away"),
            "league": r.get("league"),
            "playability": r.get("playability"),
            "bet_label": r.get("bet_label") or r.get("follow"),
            "odds": r.get("odds"),
            "hit": r.get("hit"),
            "push": r.get("push"),
            "pnl": r.get("pnl"),
            "settled_at": r.get("settled_at"),
            "alert_frozen_at": r.get("alert_frozen_at"),
        }

    return {
        "ok": True,
        "roi_mode": "asian_spread_telegram",
        "settle": settle_info,
        "n_journal": len(rows),
        "n": n,
        "n_pending": len(pending),
        "n_push": len(pushes),
        "hits": hits,
        "hit_rate": overall.get("hit_rate"),
        "flat_pnl": round(flat_pnl, 2),
        "flat_roi": round(flat_pnl / n, 3) if n else None,
        "odds_n": overall.get("odds_n"),
        "odds_pnl": overall.get("odds_pnl"),
        "odds_roi": overall.get("odds_roi"),
        "by_vote": by_vote,
        "by_market": by_market,
        "pending_rows": sorted(
            (_row_view(r) for r in pending),
            key=lambda r: str(r.get("date") or ""),
            reverse=True,
        )[:30],
        "recent": sorted(
            (_row_view(r) for r in settled),
            key=lambda r: str(r.get("date") or ""),
            reverse=True,
        )[:15],
        "note": (
            None
            if rows
            else "Nessun alert spread congelato: partono dal prossimo invio Telegram Spread Raro."
        ),
    }
