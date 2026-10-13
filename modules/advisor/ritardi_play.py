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


def odds_fair_of(play: dict[str, Any]) -> float | None:
    """Quota equa = 1 / p_fair (soglia sotto cui l'EV euristico diventa negativo)."""
    of = _f(play.get("odds_fair"))
    if of is not None and of > 1.01:
        return round(of, 3)
    pf = _f(play.get("p_fair"))
    if pf is not None and pf > 0:
        return round(1.0 / pf, 3)
    return None


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
    odds_fair = round(1.0 / p_fair, 3) if p_fair and p_fair > 0 else None
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
        "odds_fair": odds_fair,
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

SENT_PATH = ROOT / "data" / "processed" / "telegram_alerts_sent.json"


def _frozen_odds(entry: dict[str, Any]) -> float | None:
    """Quota della notifica: odds_frozen ha priorità, poi odds (legacy)."""
    od = _f(entry.get("odds_frozen"))
    if od is not None and od > 1.01:
        return od
    od = _f(entry.get("odds"))
    if od is not None and od > 1.01:
        return od
    return None


def snapshot_from_play(
    play: dict[str, Any],
    *,
    frozen_at: str | None = None,
    force_id: str | None = None,
) -> dict[str, Any] | None:
    day = str(play.get("date") or "")[:10]
    home = str(play.get("home") or "").strip()
    away = str(play.get("away") or "").strip()
    pick = str(play.get("pick") or "").strip()
    if not day or not home or not away or not pick:
        return None
    odds = _f(play.get("odds"))
    return {
        "id": str(force_id or play.get("id") or f"ritardi|{_face_key(day, home, away)}|{pick}"),
        "match_key": _face_key(day, home, away),
        "date": day,
        "home": home,
        "away": away,
        "league": play.get("league") or "",
        "pick": pick,
        "pick_label": play.get("pick_label") or pick,
        "family": play.get("family"),
        "odds": odds,
        "odds_frozen": odds,
        "odds_fair": odds_fair_of(play),
        "line": _f(play.get("line")),
        "p_fair": _f(play.get("p_fair")),
        "ev": _f(play.get("ev")),
        "severity": _f(play.get("severity")),
        "playability": int(play.get("playability") or 0),
        "title": play.get("title"),
        "detail": play.get("detail"),
        "kind": play.get("kind"),
        "team": play.get("team"),
        "alert_frozen_at": frozen_at or _now(),
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
    """Prima notifica vince: pick + quota congelati nel journal (come moneyway)."""
    data = _load_journal()
    added = skipped = 0
    for alert in alerts:
        play = alert.get("play") if isinstance(alert.get("play"), dict) else alert
        snap = snapshot_from_play(play, force_id=str(alert.get("id") or "") or None)
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


def _parse_sent_ritardi_id(aid: str) -> dict[str, str] | None:
    """ritardi|date|home|away|PICK|title…"""
    parts = str(aid or "").split("|")
    if len(parts) < 5 or parts[0] != "ritardi":
        return None
    return {
        "id": aid,
        "date": parts[1][:10],
        "home_key": parts[2].strip().lower(),
        "away_key": parts[3].strip().lower(),
        "pick": parts[4].strip(),
        "title": "|".join(parts[5:]).strip(),
    }


def _odds_for_pick(row: dict[str, Any], pick: str) -> tuple[float | None, float | None]:
    """(odds, line) per un pick già inviato su Telegram."""
    om = _odds_map(row)
    p = str(pick or "").upper()
    if p in {"O2.5", "OVER2.5"} or p.startswith("O2.5"):
        return om.get("over_2.5"), 2.5
    if p in {"U2.5", "UNDER2.5"} or p.startswith("U2.5"):
        return om.get("under_2.5"), 2.5
    m = re.match(r"CORNO(\d+(?:\.\d+)?)$", p, re.I)
    if m:
        line = float(m.group(1))
        return om.get(f"corners_over_{line:g}") or om.get(f"corners_over_{m.group(1)}"), line
    m = re.match(r"CORNU(\d+(?:\.\d+)?)$", p, re.I)
    if m:
        line = float(m.group(1))
        return om.get(f"corners_under_{line:g}") or om.get(f"corners_under_{m.group(1)}"), line
    m = re.match(r"CARDO(\d+(?:\.\d+)?)$", p, re.I)
    if m:
        line = float(m.group(1))
        return om.get(f"cards_over_{line:g}") or om.get(f"cards_over_{m.group(1)}"), line
    m = re.match(r"CARDU(\d+(?:\.\d+)?)$", p, re.I)
    if m:
        line = float(m.group(1))
        return om.get(f"cards_under_{line:g}") or om.get(f"cards_under_{m.group(1)}"), line
    return None, None


def _play_for_sent_id(aid: str, row: dict[str, Any], meta: dict[str, str]) -> dict[str, Any] | None:
    """Ricostruisce lo snapshot play per un alert già inviato (pick bloccato dall'id)."""
    pick = meta["pick"]
    title = meta.get("title") or ""
    # preferisci la stessa anomalia del Telegram
    for anom in _row_anomalies(row, compute_missing=True):
        cand = evaluate_anomaly_on_row(row, anom)
        if not cand:
            continue
        if str(cand.get("pick") or "").upper() != pick.upper():
            continue
        if title and title.lower() not in str(cand.get("title") or "").lower():
            # titolo diverso: ok se pick matcha e severity alta
            if (_f(cand.get("severity")) or 0) < MIN_ALERT_SEVERITY:
                continue
        cand["id"] = aid
        return cand
    # fallback: quota live sul pick dell'id
    odd, line = _odds_for_pick(row, pick)
    if odd is None:
        return None
    label = pick
    family = "ou25_over"
    if pick.upper().startswith("CORN"):
        family = "corners_over"
        label = f"Corner Over {line:g}" if line is not None else pick
    elif pick.upper().startswith("CARD"):
        family = "cards_over"
        label = f"Card Over {line:g}" if line is not None else pick
    elif pick.upper().startswith("O"):
        label = "Over 2.5"
    elif pick.upper().startswith("U"):
        label = "Under 2.5"
        family = "ou25_under"
    day = str(row.get("date") or meta["date"])[:10]
    return {
        "date": day,
        "home": str(row.get("home") or ""),
        "away": str(row.get("away") or ""),
        "league": row.get("league") or "",
        "pick": pick,
        "pick_label": label,
        "family": family,
        "odds": round(float(odd), 3),
        "line": line,
        "p_fair": None,
        "ev": None,
        "severity": None,
        "playability": MIN_ALERT_SCORE,
        "title": title or None,
        "id": aid,
        "match_key": _face_key(day, str(row.get("home") or ""), str(row.get("away") or "")),
    }


def backfill_ritardi_freezes_from_sent(
    upcoming: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Congela ritardi già in telegram_alerts_sent ma assenti dal journal (quote attuali)."""
    if not SENT_PATH.is_file():
        return {"ok": True, "added": 0, "skipped": 0, "missing": 0, "n_journal": len(_load_journal())}
    try:
        sent_raw = json.loads(SENT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"ok": False, "error": "sent unreadable", "added": 0}
    if isinstance(sent_raw, list):
        sent = {str(x): _now() for x in sent_raw}
    elif isinstance(sent_raw, dict):
        sent = {str(k): str(v) for k, v in sent_raw.items()}
    else:
        return {"ok": False, "error": "sent bad shape", "added": 0}

    data = _load_journal()
    # Patch odds_frozen legacy senza toccare upcoming (88MB).
    patched = 0
    for key, ent in list(data.items()):
        if _f(ent.get("odds_frozen")) is None and _f(ent.get("odds")) is not None:
            ent = dict(ent)
            ent["odds_frozen"] = _f(ent.get("odds"))
            data[key] = ent
            patched += 1

    missing_ids: list[tuple[str, str]] = []
    skipped = 0
    for aid, when in sent.items():
        meta = _parse_sent_ritardi_id(aid)
        if not meta:
            continue
        if aid in data and data[aid].get("alert_frozen_at"):
            skipped += 1
            continue
        missing_ids.append((aid, when))

    if not missing_ids:
        if patched:
            _save_journal(data)
        return {
            "ok": True,
            "added": 0,
            "skipped": skipped,
            "missing": 0,
            "patched": patched,
            "n_journal": len(data),
            "path": str(JOURNAL),
        }

    rows = upcoming
    if rows is None:
        rows = []
        if UPCOMING_PATH.is_file():
            try:
                raw = json.loads(UPCOMING_PATH.read_text(encoding="utf-8"))
                rows = raw if isinstance(raw, list) else []
            except Exception:
                rows = []
    by_face: dict[str, dict[str, Any]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        day = str(r.get("date") or "")[:10]
        home = str(r.get("home") or "").strip().lower()
        away = str(r.get("away") or "").strip().lower()
        if day and home and away:
            by_face[f"{day}|{home}|{away}"] = r

    added = missing = 0
    for aid, when in missing_ids:
        meta = _parse_sent_ritardi_id(aid)
        if not meta:
            continue
        face = f"{meta['date']}|{meta['home_key']}|{meta['away_key']}"
        row = by_face.get(face)
        if not row:
            missing += 1
            continue
        play = _play_for_sent_id(aid, row, meta)
        if not play or _f(play.get("odds")) is None:
            missing += 1
            continue
        frozen_at = when
        if "T" not in str(frozen_at):
            frozen_at = _now()
        else:
            frozen_at = str(frozen_at).replace("+00:00", "Z")
        snap = snapshot_from_play(play, frozen_at=frozen_at, force_id=aid)
        if not snap:
            missing += 1
            continue
        data[aid] = snap
        added += 1
    if added or patched:
        _save_journal(data)
    return {
        "ok": True,
        "added": added,
        "skipped": skipped,
        "missing": missing,
        "patched": patched,
        "n_journal": len(data),
        "path": str(JOURNAL),
    }


def _corners_from_hist(hist: dict[str, Any]) -> tuple[int | None, int | None]:
    for hk, ak in (
        ("home_corners", "away_corners"),
        ("corners_home", "corners_away"),
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
    odds = _frozen_odds(entry)
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
    # non ricalcolare/aggiornare quote: restano quelle freezate alla notifica
    if _f(out.get("odds_frozen")) is None and odds is not None:
        out["odds_frozen"] = odds
    if odds is not None:
        out["odds"] = odds
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


def manual_settle_ritardi(
    entry_id: str,
    *,
    home_goals: int | None = None,
    away_goals: int | None = None,
    corners_home: int | None = None,
    corners_away: int | None = None,
    hit: int | None = None,
) -> dict[str, Any]:
    """Chiude una entry ritardi a mano (FT/corner oppure Presa/Non presa)."""
    data = _load_journal()
    key = str(entry_id)
    entry = data.get(key)
    if entry is None:
        for k, v in data.items():
            if str(v.get("id") or "") == key or str(v.get("match_key") or "") == key:
                key, entry = k, v
                break
    if entry is None:
        return {"ok": False, "error": "entry ritardi non trovata"}
    if entry.get("settled_at") and entry.get("hit") is not None:
        return {
            "ok": False,
            "error": "già settled",
            "id": key,
            "hit": entry.get("hit"),
        }

    if hit is not None:
        try:
            final_hit = 1 if int(hit) == 1 else 0
        except (TypeError, ValueError):
            return {"ok": False, "error": "hit non valido"}
        odds = _frozen_odds(entry)
        pnl = None
        if odds is not None and odds > 1.01:
            pnl = round(odds - 1.0, 4) if final_hit == 1 else -1.0
        else:
            pnl = 1.0 if final_hit == 1 else -1.0
        out = dict(entry)
        out.update(
            {
                "hit": final_hit,
                "push": 0,
                "pnl": pnl,
                "settled_at": _now(),
            }
        )
        if home_goals is not None:
            out["home_goals"] = int(home_goals)
        if away_goals is not None:
            out["away_goals"] = int(away_goals)
        if corners_home is not None:
            out["corners_home"] = int(corners_home)
        if corners_away is not None:
            out["corners_away"] = int(corners_away)
        data[key] = out
        _save_journal(data)
        return {
            "ok": True,
            "id": key,
            "hit": final_hit,
            "pnl": pnl,
            "presa": "Sì" if final_hit == 1 else "No",
            "source": "ritardi",
        }

    if home_goals is None or away_goals is None:
        return {"ok": False, "error": "servono gol FT oppure hit esplicito"}
    try:
        hg_i, ag_i = int(home_goals), int(away_goals)
    except (TypeError, ValueError):
        return {"ok": False, "error": "gol non validi"}

    hist: dict[str, Any] = {"home_goals": hg_i, "away_goals": ag_i}
    if corners_home is not None and corners_away is not None:
        hist["home_corners"] = int(corners_home)
        hist["away_corners"] = int(corners_away)
        hist["corners_home"] = int(corners_home)
        hist["corners_away"] = int(corners_away)

    upd = _settle_entry(entry, hist)
    if not upd:
        pick = str(entry.get("pick") or "").upper()
        if pick.startswith("CORN") and (corners_home is None or corners_away is None):
            return {"ok": False, "error": "pick corner: servono corners_home/away"}
        return {"ok": False, "error": "impossibile calcolare hit (stats o pick)"}
    data[key] = upd
    _save_journal(data)
    return {
        "ok": True,
        "id": key,
        "home_goals": hg_i,
        "away_goals": ag_i,
        "hit": upd.get("hit"),
        "push": upd.get("push"),
        "pnl": upd.get("pnl"),
        "presa": (
            "Push"
            if upd.get("push")
            else ("Sì" if upd.get("hit") == 1 else "No" if upd.get("hit") == 0 else "n/d")
        ),
        "source": "ritardi",
    }


def list_pending_ritardi() -> list[dict[str, Any]]:
    """Tutte le entry ritardi non settled (senza limite)."""
    today = datetime.now(timezone.utc).date()
    out: list[dict[str, Any]] = []
    for key, r in _load_journal().items():
        if r.get("settled_at"):
            continue
        day_s = str(r.get("date") or "")[:10]
        try:
            age = (today - datetime.fromisoformat(day_s).date()).days
        except ValueError:
            age = None
        pick = str(r.get("pick") or "").upper()
        out.append(
            {
                "source": "ritardi",
                "id": str(r.get("id") or key),
                "journal_key": key,
                "match_key": r.get("match_key"),
                "date": day_s,
                "home": r.get("home"),
                "away": r.get("away"),
                "league": r.get("league"),
                "pick": r.get("pick"),
                "pick_label": r.get("pick_label") or r.get("pick"),
                "odds": _frozen_odds(r),
                "playability": r.get("playability"),
                "title": r.get("title"),
                "age_days": age,
                "needs_corners": pick.startswith("CORN"),
                "needs_shots": pick.startswith("SHOT"),
            }
        )
    out.sort(key=lambda x: (int(x.get("age_days") or 0), str(x.get("date") or "")), reverse=True)
    return out


def settle_ritardi_journal() -> dict[str, Any]:
    from modules.data_update.history import load_history
    from modules.data_update.team_names import resolve_known_team

    data = _load_journal()
    if not data:
        return {"ok": True, "settled": 0, "pending": 0, "n": 0}

    pending_items = [(k, e) for k, e in data.items() if not e.get("settled_at")]
    if not pending_items:
        return {
            "ok": True,
            "settled": 0,
            "pending": 0,
            "n": len(data),
            "path": str(JOURNAL),
        }

    needed_days = {
        str(e.get("date") or "")[:10] for _, e in pending_items if str(e.get("date") or "")[:10]
    }
    by_face: dict[str, dict] = {}
    for r in load_history():
        day = str(r.get("date") or "")[:10]
        if needed_days and day not in needed_days:
            continue
        if r.get("home_goals") is None or r.get("away_goals") is None:
            continue
        home = resolve_known_team(r.get("home") or "") or r.get("home")
        away = resolve_known_team(r.get("away") or "") or r.get("away")
        by_face[_face_key(day, str(home), str(away))] = r
        by_face[_face_key(day, str(r.get("home") or ""), str(r.get("away") or ""))] = r

    settled = 0
    changed = False
    for key, entry in pending_items:
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
    """Hit + ROI unitario sulle quote freeze (push esclusi se hit is None)."""
    hits = sum(1 for r in rows if int(r.get("hit") or 0) == 1)
    n = len(rows)
    op: list[float] = []
    for r in rows:
        pnl = _f(r.get("pnl"))
        od = _frozen_odds(r)
        if pnl is not None and od is not None and od > 1.01:
            op.append(float(pnl))
    return {
        "n": n,
        "hits": hits,
        "hit_rate": round(hits / n, 3) if n else None,
        "odds_n": len(op),
        "odds_pnl": round(sum(op), 2) if op else 0.0,
        "odds_roi": round(sum(op) / len(op), 3) if op else None,
    }


def ritardi_alert_paper_report(*, include_live: bool = True) -> dict[str, Any]:
    """ROI journal Telegram ritardi: campione = freeze inviati, PnL su quote freezate.

    ``include_live=False`` salta il scan upcoming (utile in home: stessi n/ROI/pending).
    """
    backfill = backfill_ritardi_freezes_from_sent()
    settle_info = settle_ritardi_journal()
    data = _load_journal()
    rows = list(data.values())
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

    def _row_view(r: dict[str, Any], key: str = "") -> dict[str, Any]:
        return {
            "id": str(r.get("id") or key),
            "journal_key": key,
            "match_key": r.get("match_key"),
            "date": r.get("date"),
            "home": r.get("home"),
            "away": r.get("away"),
            "league": r.get("league"),
            "pick": r.get("pick"),
            "pick_label": r.get("pick_label") or r.get("pick"),
            "odds": _frozen_odds(r),
            "odds_frozen": _frozen_odds(r),
            "playability": r.get("playability"),
            "title": r.get("title"),
            "hit": r.get("hit"),
            "push": r.get("push"),
            "pnl": r.get("pnl"),
            "settled_at": r.get("settled_at"),
            "alert_frozen_at": r.get("alert_frozen_at"),
        }

    live = (
        list_ritardi_playable(strong_only=True, compute_missing=False)
        if include_live
        else {"ok": True, "n": 0, "rows": []}
    )
    return {
        "ok": True,
        "roi_mode": "ritardi_telegram_freeze_odds",
        "min_playability": MIN_ALERT_SCORE,
        "min_severity": MIN_ALERT_SEVERITY,
        "min_ev": MIN_ALERT_EV,
        "backfill": backfill,
        "settle": settle_info,
        "n_journal": len(rows),
        "n_journal_strong": len(rows),
        "n": overall["n"],
        "n_pending": len(pending),
        "hits": overall["hits"],
        "hit_rate": overall["hit_rate"],
        "odds_roi": overall["odds_roi"],
        "odds_pnl": overall["odds_pnl"],
        "by_vote": by_score,
        "pending_rows": sorted(
            (_row_view(v, k) for k, v in data.items() if not v.get("settled_at")),
            key=lambda r: str(r.get("date") or ""),
            reverse=True,
        )[:30],
        "recent": sorted(
            (
                _row_view(v, k)
                for k, v in data.items()
                if v.get("settled_at") and v.get("hit") is not None
            ),
            key=lambda r: str(r.get("settled_at") or ""),
            reverse=True,
        )[:15],
        "live_playable": live,
        "note": (
            "ROI unitario sulle quote freeze Telegram (journal ritardi). "
            "Campione = solo alert inviato e congelato; indipendente dal GIOCA."
            if rows
            else (
                f"Nessun ritardo freeze in journal: partono dal prossimo Telegram "
                f"(giocab.≥{MIN_ALERT_SCORE}, max {MAX_TELEGRAM_PER_RUN}/run)."
            )
        ),
    }
