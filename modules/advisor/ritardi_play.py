"""Ritardi giocabili indipendenti dal freeze GIOCA.

Valuta anomalie (severity) + quota book → pick dedicato, giocabilità 1–10.
Alert Telegram sui più succulenti; journal proprio per ROI (come moneyway).
Non tocca EV/Kelly del modello principale.
"""

from __future__ import annotations

import json
import re
from datetime import date as date_cls
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "data" / "processed" / "telegram_ritardi_freeze.json"
UPCOMING_PATH = ROOT / "data" / "processed" / "upcoming_predictions.json"

# Soglie (conservative: pochi “succulenti”, non spam)
MIN_SEVERITY = 0.62
MIN_PLAY_SCORE = 7  # da giocare in UI
MIN_ALERT_SCORE = 8  # Telegram / ROI forti
MIN_ALERT_SEVERITY = 0.80
MIN_ALERT_EV = 0.06
MIN_ODDS = 1.55
MAX_ODDS = 2.70
MIN_EV = 0.05
MAX_TELEGRAM_PER_RUN = 12


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


def _face_key(day: str, home: str, away: str) -> str:
    return f"{str(day)[:10]}|{str(home).strip().lower()}|{str(away).strip().lower()}"


def _load_journal() -> dict[str, dict[str, Any]]:
    if not JOURNAL.is_file():
        return {}
    try:
        raw = json.loads(JOURNAL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(raw, dict) and isinstance(raw.get("ritardi"), dict):
        raw = raw["ritardi"]
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


def _save_journal(data: dict[str, dict[str, Any]]) -> None:
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": _now(), "n": len(data), "ritardi": data}
    JOURNAL.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _odds_map(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    odds = row.get("odds") if isinstance(row.get("odds"), dict) else {}
    for src in (odds, row):
        if not isinstance(src, dict):
            continue
        for k, v in src.items():
            fv = _f(v)
            if fv is not None and fv > 1.01:
                out[str(k)] = fv
    # alias comuni
    for a, b in (
        ("odd_over_25", "over_2.5"),
        ("odd_under_25", "under_2.5"),
        ("odd_1", "1"),
        ("odd_2", "2"),
        ("odd_x", "X"),
    ):
        if a in out and b not in out:
            out[b] = out[a]
    return out


def _best_corners_over(odds: dict[str, float]) -> tuple[str, float, float] | None:
    """Ritorna (pick_code, line, odds) sul Over corner più EV-friendly."""
    best: tuple[str, float, float, float] | None = None  # code, line, odd, score
    for k, od in odds.items():
        m = re.match(r"corners_over_(\d+(?:\.\d+)?)$", k, re.I)
        if not m:
            continue
        line = float(m.group(1))
        if line < 7.5 or line > 14.5:
            continue
        if od < MIN_ODDS or od > MAX_ODDS:
            continue
        # preferisci quote 1.65–2.20
        sweet = 1.0 - abs(od - 1.90) / 1.5
        score = sweet + (0.05 if 8.5 <= line <= 11.5 else 0)
        if best is None or score > best[3]:
            best = (f"CORNO{line:g}", line, od, score)
    if not best:
        return None
    return best[0], best[1], best[2]


def _anomaly_market(anom: dict[str, Any]) -> str | None:
    """Famiglia mercato: ou25_over | corners_over | cards_over | skip."""
    title = str(anom.get("title") or "").lower()
    kind = str(anom.get("kind") or "")
    if "xg" in title:
        return None
    if "assenze" in title or kind == "absence":
        return None
    if "corner" in title:
        return "corners_over"
    if "cartellin" in title or "falli" in title:
        return "cards_over"
    if "gol in ritardo" in title or "senza segnare" in title or "0 gol" in title:
        return "ou25_over"
    if "under 2.5 di fila" in title:
        return "ou25_over"
    if "tiri in ritardo" in title:
        return "ou25_over"  # proxy debole → gestito con penalty score
    if "senza vittoria" in title or "0 vittorie" in title:
        return None
    return None


def _p_fair(sev: float, *, family: str, baseline: float | None, observed: float | None) -> float:
    """Euristica probabilità fair sul lato Over (non modello ML)."""
    base = 0.50 if family == "corners_over" else 0.48
    # severity alta sposta poco: evitiamo EV gonfiati
    boost = max(0.0, (float(sev) - 0.55)) * 0.18
    gap = 0.0
    if baseline is not None and observed is not None and baseline > 0:
        gap = max(0.0, min(0.06, (float(baseline) - float(observed)) / float(baseline) * 0.10))
    return max(0.42, min(0.58, base + boost + gap))


def _play_score(
    *,
    sev: float,
    odds: float,
    ev: float,
    family: str,
) -> int:
    # severity → 5..9
    s = 5 + int(round(max(0.0, sev - 0.45) / 0.55 * 4))
    if 1.60 <= odds <= 2.30:
        s += 1
    elif odds < 1.50 or odds > 2.80:
        s -= 1
    if ev >= 0.10:
        s += 1
    elif ev >= 0.06:
        s += 0
    elif ev < MIN_EV:
        s -= 2
    if family == "ou25_over":
        pass
    elif family == "corners_over":
        s += 0
    elif family == "cards_over":
        s -= 1  # quote spesso mancanti / noisy
    return max(1, min(10, s))


def evaluate_anomaly_on_row(row: dict[str, Any], anom: dict[str, Any]) -> dict[str, Any] | None:
    """Valuta una singola anomalia sulla riga calendario. None se non giocabile."""
    family = _anomaly_market(anom)
    if not family:
        return None
    sev = _f(anom.get("severity")) or 0.0
    if sev < MIN_SEVERITY:
        return None
    odds_m = _odds_map(row)
    pick = label = None
    odd = line = None
    if family == "ou25_over":
        odd = odds_m.get("over_2.5")
        pick, label, line = "O2.5", "Over 2.5", 2.5
    elif family == "corners_over":
        best = _best_corners_over(odds_m)
        if not best:
            return None
        pick, line, odd = best
        label = f"Corner Over {line:g}"
    elif family == "cards_over":
        # solo se esiste una linea tipica in odds
        for k, od in odds_m.items():
            m = re.match(r"cards_over_(\d+(?:\.\d+)?)$", k, re.I)
            if m and MIN_ODDS <= od <= MAX_ODDS:
                line = float(m.group(1))
                pick, label, odd = f"CARDO{line:g}", f"Card Over {line:g}", od
                break
        if pick is None:
            return None
    else:
        return None
    if odd is None or odd < MIN_ODDS or odd > MAX_ODDS:
        return None
    p_fair = _p_fair(
        sev,
        family=family,
        baseline=_f(anom.get("baseline")),
        observed=_f(anom.get("observed")),
    )
    # tiri = proxy debole
    if "tiri" in str(anom.get("title") or "").lower():
        p_fair = max(0.38, p_fair - 0.04)
    ev = p_fair * odd - 1.0
    score = _play_score(sev=sev, odds=odd, ev=ev, family=family)
    action = "gioca" if score >= MIN_PLAY_SCORE and ev >= MIN_EV else "no_bet"
    day = str(row.get("date") or "")[:10]
    home = str(row.get("home") or "")
    away = str(row.get("away") or "")
    return {
        "date": day,
        "time": row.get("time") or "",
        "league": row.get("league") or "",
        "home": home,
        "away": away,
        "pick": pick,
        "pick_label": label,
        "family": family,
        "odds": round(float(odd), 3),
        "line": line,
        "p_fair": round(p_fair, 3),
        "ev": round(ev, 3),
        "severity": round(sev, 3),
        "playability": score,
        "action": action,
        "title": anom.get("title"),
        "detail": anom.get("detail"),
        "kind": anom.get("kind"),
        "team": anom.get("team"),
        "side": anom.get("side"),
        "match_key": _face_key(day, home, away),
        "id": f"ritardi|{_face_key(day, home, away)}|{pick}|{str(anom.get('title') or '')[:40]}",
    }


def _row_anomalies(row: dict[str, Any], *, compute_missing: bool = False) -> list[dict[str, Any]]:
    from modules.advisor.stat_anomalies import sanitize_anomaly_block

    block = row.get("stat_anomalies")
    if not isinstance(block, dict):
        pred = row.get("prediction") if isinstance(row.get("prediction"), dict) else {}
        block = pred.get("stat_anomalies") if isinstance(pred, dict) else None
    if not isinstance(block, dict) and compute_missing:
        try:
            from modules.advisor.stat_anomalies import match_anomalies

            pred = row.get("prediction") if isinstance(row.get("prediction"), dict) else {}
            tac = pred.get("tactical") if isinstance(pred, dict) else None
            block = match_anomalies(
                str(row.get("home") or ""),
                str(row.get("away") or ""),
                str(row.get("date") or "")[:10],
                tactical=tac if isinstance(tac, dict) else None,
            )
        except Exception:
            block = None
    block = sanitize_anomaly_block(block if isinstance(block, dict) else None)
    return list(block.get("items") or [])


def evaluate_row(row: dict[str, Any], *, compute_missing: bool = False) -> dict[str, Any] | None:
    """Miglior ritardo giocabile sulla partita (max playability, poi EV)."""
    cands: list[dict[str, Any]] = []
    for anom in _row_anomalies(row, compute_missing=compute_missing):
        ev = evaluate_anomaly_on_row(row, anom)
        if ev:
            cands.append(ev)
    if not cands:
        return None
    cands.sort(key=lambda r: (-int(r.get("playability") or 0), -float(r.get("ev") or 0)))
    best = dict(cands[0])
    best["alternatives"] = cands[1:4]
    return best


def list_ritardi_playable(
    upcoming: list[dict[str, Any]] | None = None,
    *,
    min_score: int = MIN_PLAY_SCORE,
    compute_missing: bool = False,
    only_future: bool = True,
    strong_only: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Lista ritardi da giocare (indipendenti dal freeze GIOCA).

    ``strong_only=True`` → stessa barra Telegram/ROI (sev/EV) e di default top N.
    """
    rows = upcoming
    if rows is None:
        rows = []
        if UPCOMING_PATH.is_file():
            try:
                raw = json.loads(UPCOMING_PATH.read_text(encoding="utf-8"))
                rows = raw if isinstance(raw, list) else []
            except Exception:
                rows = []
    today = date_cls.today().isoformat()
    if strong_only:
        min_score = max(int(min_score), MIN_ALERT_SCORE)
        if limit is None:
            limit = MAX_TELEGRAM_PER_RUN
    out: list[dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        day = str(r.get("date") or "")[:10]
        if only_future and day and day < today:
            continue
        best = evaluate_row(r, compute_missing=compute_missing)
        if not best:
            continue
        if int(best.get("playability") or 0) < int(min_score):
            continue
        if str(best.get("action") or "") != "gioca":
            continue
        if strong_only:
            if (_f(best.get("severity")) or 0) < MIN_ALERT_SEVERITY:
                continue
            if (_f(best.get("ev")) or 0) < MIN_ALERT_EV:
                continue
        out.append(best)
    out.sort(key=lambda r: (-int(r.get("playability") or 0), -float(r.get("ev") or 0), str(r.get("date"))))
    if limit is not None:
        out = out[: max(0, int(limit))]
    return {
        "ok": True,
        "n": len(out),
        "min_score": int(min_score),
        "min_severity": MIN_ALERT_SEVERITY if strong_only else MIN_SEVERITY,
        "min_alert_score": MIN_ALERT_SCORE,
        "strong_only": bool(strong_only),
        "note": (
            f"Solo ritardi forti (giocab.≥{MIN_ALERT_SCORE}, sev≥{MIN_ALERT_SEVERITY:.0%}, "
            f"EV≥{MIN_ALERT_EV:.0%}, top {limit}) — ROI/Telegram."
            if strong_only
            else (
                "Ritardi indipendenti dal GIOCA freeze: severity + quota → pick dedicato. "
                f"Lista ≥{min_score}/10."
            )
        ),
        "rows": out,
    }


def attach_ritardi_play(rows: list[dict[str, Any]], *, compute_missing: bool = False) -> int:
    """Attacca ``ritardi_play`` alle righe upcoming (best candidate o None)."""
    n = 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        best = evaluate_row(r, compute_missing=compute_missing)
        r["ritardi_play"] = best
        if best and best.get("action") == "gioca":
            n += 1
    return n


# --- Journal / Telegram freeze / ROI ---


def snapshot_from_play(play: dict[str, Any]) -> dict[str, Any] | None:
    day = str(play.get("date") or "")[:10]
    home = str(play.get("home") or "").strip()
    away = str(play.get("away") or "").strip()
    pick = str(play.get("pick") or "").strip()
    if not day or not home or not away or not pick:
        return None
    return {
        "id": str(play.get("id") or f"ritardi|{_face_key(day, home, away)}|{pick}"),
        "match_key": _face_key(day, home, away),
        "date": day,
        "home": home,
        "away": away,
        "league": play.get("league") or "",
        "pick": pick,
        "pick_label": play.get("pick_label") or pick,
        "family": play.get("family"),
        "odds": _f(play.get("odds")),
        "line": _f(play.get("line")),
        "p_fair": _f(play.get("p_fair")),
        "ev": _f(play.get("ev")),
        "severity": _f(play.get("severity")),
        "playability": int(play.get("playability") or 0),
        "title": play.get("title"),
        "detail": play.get("detail"),
        "kind": play.get("kind"),
        "team": play.get("team"),
        "alert_frozen_at": _now(),
        "hit": None,
        "push": None,
        "pnl": None,
        "settled_at": None,
        "home_goals": None,
        "away_goals": None,
        "corners_home": None,
        "corners_away": None,
    }


def record_ritardi_freezes(alerts: list[dict[str, Any]]) -> dict[str, Any]:
    data = _load_journal()
    added = skipped = 0
    for alert in alerts:
        play = alert.get("play") if isinstance(alert.get("play"), dict) else alert
        snap = snapshot_from_play(play)
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


def _corners_from_hist(hist: dict[str, Any]) -> tuple[int | None, int | None]:
    for hk, ak in (
        ("home_corners", "away_corners"),
        ("HC", "AC"),
    ):
        try:
            if hist.get(hk) is not None and hist.get(ak) is not None:
                return int(hist[hk]), int(hist[ak])
        except (TypeError, ValueError):
            pass
    stats = hist.get("sofascore_stats")
    if isinstance(stats, str) and stats.strip():
        try:
            stats = json.loads(stats)
        except json.JSONDecodeError:
            stats = None
    if isinstance(stats, dict):
        try:
            ch, ca = stats.get("corners_home"), stats.get("corners_away")
            if ch is not None and ca is not None:
                return int(ch), int(ca)
        except (TypeError, ValueError):
            pass
    return None, None


def _settle_entry(entry: dict[str, Any], hist: dict[str, Any]) -> dict[str, Any] | None:
    pick = str(entry.get("pick") or "").upper()
    odds = _f(entry.get("odds"))
    try:
        hg = int(hist["home_goals"])
        ag = int(hist["away_goals"])
    except (TypeError, ValueError, KeyError):
        return None
    tot = hg + ag
    hit: int | None = None
    push = False
    ch = ca = None

    if pick in {"O2.5", "OVER2.5"} or pick.startswith("O2.5"):
        hit = 1 if tot > 2.5 else 0
    elif pick in {"U2.5", "UNDER2.5"} or pick.startswith("U2.5"):
        hit = 1 if tot < 2.5 else 0
    elif pick.startswith("CORNO") or pick.startswith("CORNU"):
        ch, ca = _corners_from_hist(hist)
        if ch is None or ca is None:
            return None  # aspetta stats
        corners = ch + ca
        try:
            line = float(entry.get("line") if entry.get("line") is not None else re.sub(r"[^\d.]", "", pick) or 0)
        except (TypeError, ValueError):
            return None
        if pick.startswith("CORNO"):
            if corners > line:
                hit = 1
            elif corners < line:
                hit = 0
            else:
                push = True
                hit = None
        else:
            if corners < line:
                hit = 1
            elif corners > line:
                hit = 0
            else:
                push = True
                hit = None
    elif pick.startswith("CARDO") or pick.startswith("CARDU"):
        # settle cartellini solo se HY/AY in hist
        try:
            hy = int(hist.get("home_yellow") if hist.get("home_yellow") is not None else hist.get("HY"))
            ay = int(hist.get("away_yellow") if hist.get("away_yellow") is not None else hist.get("AY"))
        except (TypeError, ValueError):
            return None
        cards = hy + ay
        try:
            line = float(entry.get("line") or 0)
        except (TypeError, ValueError):
            return None
        if pick.startswith("CARDO"):
            hit = 1 if cards > line else 0 if cards < line else None
        else:
            hit = 1 if cards < line else 0 if cards > line else None
        if hit is None:
            push = True
    else:
        return None

    pnl = None
    if push:
        pnl = 0.0
    elif hit == 1 and odds is not None and odds > 1.01:
        pnl = round(odds - 1.0, 4)
    elif hit == 0 and odds is not None and odds > 1.01:
        pnl = -1.0
    elif hit == 1:
        pnl = 1.0
    elif hit == 0:
        pnl = -1.0

    out = dict(entry)
    out.update(
        {
            "home_goals": hg,
            "away_goals": ag,
            "corners_home": ch,
            "corners_away": ca,
            "hit": hit,
            "push": 1 if push else 0,
            "pnl": pnl,
            "settled_at": _now(),
        }
    )
    return out


def settle_ritardi_journal() -> dict[str, Any]:
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
        by_face[_face_key(day, str(home), str(away))] = r
        by_face[_face_key(day, str(r.get("home") or ""), str(r.get("away") or ""))] = r

    settled = 0
    changed = False
    for key, entry in list(data.items()):
        if entry.get("settled_at"):
            continue
        face = entry.get("match_key") or _face_key(
            str(entry.get("date") or ""), str(entry.get("home") or ""), str(entry.get("away") or "")
        )
        hist = by_face.get(face)
        if not hist:
            continue
        upd = _settle_entry(entry, hist)
        if not upd:
            continue
        data[key] = upd
        settled += 1
        changed = True
    pending = sum(1 for e in data.values() if not e.get("settled_at"))
    if changed:
        _save_journal(data)
    return {"ok": True, "settled": settled, "pending": pending, "n": len(data), "path": str(JOURNAL)}


def _subset_roi(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hits = sum(1 for r in rows if int(r.get("hit") or 0) == 1)
    n = len(rows)
    op = [float(r["pnl"]) for r in rows if _f(r.get("pnl")) is not None and _f(r.get("odds")) and float(r["odds"]) > 1.01]
    return {
        "n": n,
        "hits": hits,
        "hit_rate": round(hits / n, 3) if n else None,
        "odds_n": len(op),
        "odds_pnl": round(sum(op), 2) if op else 0.0,
        "odds_roi": round(sum(op) / len(op), 3) if op else None,
    }


def _is_strong_reco(row: dict[str, Any]) -> bool:
    """Solo i consigli 'succulenti' (stessa barra del Telegram)."""
    try:
        score = int(row.get("playability") or 0)
    except (TypeError, ValueError):
        score = 0
    sev = _f(row.get("severity")) or 0.0
    ev = _f(row.get("ev")) or 0.0
    if score < MIN_ALERT_SCORE:
        return False
    if sev < MIN_ALERT_SEVERITY:
        return False
    # EV può mancare su freeze vecchi: in quel caso basta score+severity
    if row.get("ev") is not None and ev < MIN_ALERT_EV:
        return False
    return True


def ritardi_alert_paper_report() -> dict[str, Any]:
    """ROI journal: solo ritardi forti consigliati (Telegram / giocab.≥8)."""
    settle_info = settle_ritardi_journal()
    data = _load_journal()
    all_rows = list(data.values())
    rows = [r for r in all_rows if _is_strong_reco(r)]
    pending = [r for r in rows if not r.get("settled_at")]
    settled = [r for r in rows if r.get("settled_at") and r.get("hit") is not None]
    overall = _subset_roi(settled)
    by_score: dict[str, dict[str, Any]] = {}
    for v in (8, 9, 10):
        subset = [r for r in settled if int(r.get("playability") or 0) == v]
        by_score[str(v)] = {
            "vote": v,
            **_subset_roi(subset),
            "pending": sum(1 for r in pending if int(r.get("playability") or 0) == v),
        }
    live = list_ritardi_playable(strong_only=True, compute_missing=False)
    return {
        "ok": True,
        "roi_mode": "ritardi_alert_strong_only",
        "min_playability": MIN_ALERT_SCORE,
        "min_severity": MIN_ALERT_SEVERITY,
        "min_ev": MIN_ALERT_EV,
        "settle": settle_info,
        "n_journal": len(all_rows),
        "n_journal_strong": len(rows),
        "n": overall["n"],
        "n_pending": len(pending),
        "hits": overall["hits"],
        "hit_rate": overall["hit_rate"],
        "odds_roi": overall["odds_roi"],
        "odds_pnl": overall["odds_pnl"],
        "by_vote": by_score,
        "pending_rows": sorted(pending, key=lambda r: str(r.get("date") or ""), reverse=True)[:30],
        "recent": sorted(settled, key=lambda r: str(r.get("settled_at") or ""), reverse=True)[:15],
        "live_playable": live,
        "note": (
            f"ROI solo sui ritardi forti consigliati (giocab.≥{MIN_ALERT_SCORE}, "
            f"severity≥{MIN_ALERT_SEVERITY:.0%}, EV≥{MIN_ALERT_EV:.0%}) — gli stessi del Telegram. "
            "Indipendente dal GIOCA freeze."
            if rows
            else (
                f"Nessun ritardo forte in journal: partono dal prossimo Telegram "
                f"(giocab.≥{MIN_ALERT_SCORE}, max {MAX_TELEGRAM_PER_RUN}/run)."
            )
        ),
    }
