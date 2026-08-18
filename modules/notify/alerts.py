"""Avvisi: voto unificato >= 9 e spread AsianBetSoccer Raro (linea AH/totale >= 1)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from modules.data_update.asian_odds import load_asian_odds
from modules.notify.telegram import load_credentials, send_message, telegram_status

ROOT = Path(__file__).resolve().parents[2]
UPCOMING = ROOT / "data" / "processed" / "upcoming_predictions.json"
SENT = ROOT / "data" / "processed" / "telegram_alerts_sent.json"
MIN_UNIFIED = 9
RARE_LINE = 1.0
KEEP_DAYS = 21
CHUNK = 10


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _load_sent() -> dict[str, str]:
    if not SENT.is_file():
        return {}
    try:
        raw = json.loads(SENT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(raw, list):
        return {str(k): _now().isoformat() for k in raw}
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    return {}


def _save_sent(ids: dict[str, str]) -> None:
    cutoff = _now() - timedelta(days=KEEP_DAYS)
    kept: dict[str, str] = {}
    for key, ts in ids.items():
        try:
            when = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            when = _now()
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        if when >= cutoff:
            kept[key] = ts
    SENT.parent.mkdir(parents=True, exist_ok=True)
    SENT.write_text(json.dumps(kept, indent=2), encoding="utf-8")


def _fmt_line(val: object) -> str:
    if val is None:
        return "—"
    try:
        num = float(val)
    except (TypeError, ValueError):
        return "—"
    if abs(num - round(num)) < 1e-9:
        return str(int(round(num)))
    return f"{num:.2f}".rstrip("0").rstrip(".")


def _pct(val: object) -> str | None:
    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    if abs(num) <= 1.5:
        num *= 100.0
    return f"{num:+.1f}%"


def _load_upcoming(rows: list[dict] | None) -> list[dict]:
    if rows is not None:
        return rows
    if not UPCOMING.is_file():
        return []
    try:
        data = json.loads(UPCOMING.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _match_key(row: dict) -> str:
    date = str(row.get("date") or "")[:10]
    home = str(row.get("home") or "").strip().lower()
    away = str(row.get("away") or "").strip().lower()
    mid = str(row.get("match_id") or "").strip()
    if mid:
        return f"{date}|{mid}"
    return f"{date}|{home}|{away}"


def _header(row: dict) -> str:
    date = str(row.get("date") or "")[:10]
    time = str(row.get("time") or "").strip()
    league = str(row.get("league") or "").strip()
    home = str(row.get("home") or "?")
    away = str(row.get("away") or "?")
    when = " ".join(p for p in (date, time) if p)
    bits = [b for b in (when, league) if b]
    line = " · ".join(bits)
    title = f"{home} vs {away}"
    return f"{line}\n{title}" if line else title


def _score_alerts(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        try:
            score = int(row.get("score_unified"))
        except (TypeError, ValueError):
            continue
        if score < MIN_UNIFIED:
            continue
        key = f"voto|{_match_key(row)}|{score}"
        pick = str(row.get("pick") or "—")
        name = str(row.get("pick_name") or "")
        action = str(row.get("action") or "")
        ev = _pct(row.get("ev_cons") if row.get("ev_cons") is not None else row.get("ev"))
        body = [_header(row), f"Voto unificato {score}/10"]
        play = pick if pick == name or not name else f"{pick} · {name}"
        body.append(f"Pick: {play}" + (f" · {action}" if action else ""))
        if ev:
            body.append(f"EV {ev}")
        note = str(row.get("score_reason_1") or row.get("meta_note") or "").strip()
        if note:
            body.append(note[:220])
        out.append({"id": key, "kind": "voto", "text": "\n".join(body), "sort": -score})
    return out


def _rare_from_move(row: dict, move: dict | None) -> dict | None:
    move = move or {}
    level = str(move.get("movement_level") or row.get("movement_level") or "")
    try:
        line = float(move.get("line_move") if move.get("line_move") is not None else row.get("line_move") or 0)
    except (TypeError, ValueError):
        line = 0.0
    if level != "Raro" and line < RARE_LINE:
        return None
    ah_open = move.get("ah_open", row.get("ah_open"))
    ah_curr = move.get("ah_curr", row.get("ah_curr"))
    tot_open = move.get("total_open", row.get("total_open"))
    tot_curr = move.get("total_curr", row.get("total_line") or row.get("total_curr"))
    body = [_header(row), f"Spread Raro · linea AH/totale Δ {line:g} (≥1)"]
    if ah_open is not None or ah_curr is not None:
        body.append(f"AH {_fmt_line(ah_open)} → {_fmt_line(ah_curr)}")
    if tot_open is not None or tot_curr is not None:
        body.append(f"Totale {_fmt_line(tot_open)} → {_fmt_line(tot_curr)}")
    steam = " / ".join(
        p
        for p in (
            str(move.get("steam_1x2") or row.get("steam_1x2") or ""),
            str(move.get("steam_ah") or row.get("steam_ah") or ""),
            str(move.get("steam_ou") or row.get("steam_ou") or ""),
        )
        if p
    )
    if steam:
        body.append(f"Steam: {steam}")
    summary = str(move.get("movement_summary") or row.get("movement_summary") or "").strip()
    if summary:
        body.append(summary[:220])
    key = f"spread|{_face_key(row)}"
    return {"id": key, "kind": "spread", "text": "\n".join(body), "sort": -line}


def _face_key(row: dict) -> str:
    date = str(row.get("date") or "")[:10]
    home = str(row.get("home") or "").strip().lower()
    away = str(row.get("away") or "").strip().lower()
    return f"{date}|{home}|{away}"


def _spread_alerts(upcoming: list[dict], asian: list[dict]) -> list[dict]:
    seen_ids: set[str] = set()
    seen_faces: set[str] = set()
    out: list[dict] = []
    for row in list(upcoming) + list(asian):
        move = row.get("market_move") if isinstance(row.get("market_move"), dict) else None
        alert = _rare_from_move(row, move)
        if not alert:
            continue
        face = _face_key(row)
        if alert["id"] in seen_ids or face in seen_faces:
            continue
        seen_ids.add(alert["id"])
        seen_faces.add(face)
        out.append(alert)
    return out


def _pack(title: str, items: list[dict]) -> list[tuple[str, list[str]]]:
    if not items:
        return []
    items = sorted(items, key=lambda r: (r.get("sort") or 0, r["id"]))
    messages: list[tuple[str, list[str]]] = []
    for i in range(0, len(items), CHUNK):
        chunk = items[i : i + CHUNK]
        head = title if i == 0 else f"{title} (cont.)"
        body = "\n\n".join(item["text"] for item in chunk)
        messages.append((f"{head}\n\n{body}", [item["id"] for item in chunk]))
    return messages


def collect_alerts(upcoming: list[dict] | None = None) -> dict:
    rows = _load_upcoming(upcoming)
    try:
        asian = load_asian_odds()
    except Exception:
        asian = []
    scores = _score_alerts(rows)
    rares = _spread_alerts(rows, asian)
    return {
        "voto": scores,
        "spread": rares,
        "n_voto": len(scores),
        "n_spread": len(rares),
    }


def dispatch_alerts(upcoming: list[dict] | None = None, *, dry_run: bool = False) -> dict:
    found = collect_alerts(upcoming)
    sent_ids = _load_sent()
    fresh_voto = [a for a in found["voto"] if a["id"] not in sent_ids]
    fresh_spread = [a for a in found["spread"] if a["id"] not in sent_ids]
    messages = _pack(f"⚽ Voto unificato ≥{MIN_UNIFIED}", fresh_voto)
    messages += _pack("📈 AsianBetSoccer · spread Raro (≥1)", fresh_spread)

    sent_n = 0
    if dry_run:
        for msg, _ids in messages:
            print(msg)
            print("---")
    elif messages:
        if not load_credentials():
            print("telegram skip: credenziali assenti")
        else:
            now = _now().isoformat()
            changed = False
            for msg, ids in messages:
                if send_message(msg):
                    sent_n += 1
                    for key in ids:
                        sent_ids[key] = now
                    changed = True
            if changed:
                _save_sent(sent_ids)

    info = {
        "n_voto": found["n_voto"],
        "n_spread": found["n_spread"],
        "n_new_voto": len(fresh_voto),
        "n_new_spread": len(fresh_spread),
        "n_messages": len(messages),
        "n_sent": sent_n,
        "dry_run": dry_run,
        "status": telegram_status(),
    }
    print(
        f"telegram avvisi: voto {info['n_new_voto']}/{info['n_voto']} nuovi, "
        f"spread {info['n_new_spread']}/{info['n_spread']} nuovi, "
        f"inviati {sent_n}"
    )
    return info


def ping_bot() -> bool:
    return send_message(
        "Football Predictor collegato allo stesso bot delle offerte.\n"
        f"Avvisi: voto unificato ≥{MIN_UNIFIED} e spread AsianBetSoccer Raro (≥1)."
    )
