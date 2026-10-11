"""Promemoria pressione bottoni UI: ultimo click → prossimo consigliato.

Persistenza in data/processed/button_cadence.json. Non tocca EV/Kelly.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "data" / "processed" / "button_cadence.json"

# days = intervallo consigliato; seed_today = già considerato premuto al primo avvio
BUTTONS: dict[str, dict[str, Any]] = {
    "solo_quote": {
        "label": "Solo quote e calendario",
        "days": 1,
        "seed_today": True,
        "note": "routine giornaliera",
    },
    "asian": {
        "label": "Quote AsianBetSoccer",
        "days": 1,
        "seed_today": True,
        "note": "spesso già in Solo quote",
    },
    "tipster": {
        "label": "Pronostici tipster",
        "days": 3,
        "seed_today": True,
        "note": "quadro",
    },
    "cups": {
        "label": "Scarica coppe",
        "days": 7,
        "seed_today": True,
        "note": "finestra UEFA",
    },
    "world": {
        "label": "Calendario mondiale",
        "days": 8,
        "seed_today": True,
        "note": "finestra 14g",
    },
    "fbref": {
        "label": "FBref",
        "days": 7,
        "seed_today": True,
        "note": "stile stagione",
    },
    "fd_rates": {
        "label": "FD cards/corners",
        "days": 3,
        "seed_today": True,
        "note": "CARD/CORN",
    },
    "fbref_logs": {
        "label": "FBref match logs",
        "days": 21,
        "seed_today": True,
        "note": "lento",
    },
    "understat": {
        "label": "Understat xG",
        "days": 3,
        "seed_today": True,
        "note": "λ / marcatori",
    },
    "statsbomb": {
        "label": "StatsBomb open data",
        "days": 30,
        "seed_today": True,
        "note": "raro",
    },
    "sofascore": {
        "label": "Sofascore classifica",
        "days": 7,
        "seed_today": True,
        "note": "quadro",
    },
    "fotmob": {
        "label": "FotMob",
        "days": 3,
        "seed_today": True,
        "note": "classifica + match",
    },
    "geocode": {
        "label": "Geocode stadi",
        "days": 180,
        "seed_today": True,
        "note": "una tantum",
    },
    "whoscored": {
        "label": "WhoScored assenze",
        "days": 2,
        "seed_today": True,
        "note": "on demand / lento",
    },
    "pinnacle": {
        "label": "Quote Pinnacle",
        "days": 1,
        "seed_today": False,
        "note": "~1 fetch/giorno",
    },
    "betfair": {
        "label": "Quote Betfair",
        "days": 1,
        "seed_today": False,
        "note": "se usi exchange",
    },
}


def _today() -> date:
    return date.today()


def _parse_day(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except ValueError:
        return None


def load_store() -> dict[str, Any]:
    if not STORE.exists():
        return {"buttons": {}, "seeded_at": None}
    try:
        data = json.loads(STORE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"buttons": {}, "seeded_at": None}
        data.setdefault("buttons", {})
        return data
    except Exception:
        return {"buttons": {}, "seeded_at": None}


def save_store(data: dict[str, Any]) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    STORE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_seeded(*, force: bool = False) -> dict[str, Any]:
    """Al primo avvio marca come premuti oggi i bottoni con seed_today=True."""
    store = load_store()
    if store.get("seeded_at") and not force:
        return store
    today = _today().isoformat()
    buttons = store.setdefault("buttons", {})
    for key, meta in BUTTONS.items():
        if not meta.get("seed_today"):
            continue
        prev = buttons.get(key) if isinstance(buttons.get(key), dict) else {}
        if prev.get("last_pressed") and not force:
            continue
        buttons[key] = {
            "last_pressed": today,
            "interval_days": int(meta["days"]),
            "seeded": True,
        }
    store["seeded_at"] = datetime.now(timezone.utc).isoformat()
    store["buttons"] = buttons
    save_store(store)
    return store


def mark_pressed(key: str, *, when: date | None = None) -> dict[str, Any]:
    if key not in BUTTONS:
        return {}
    store = ensure_seeded()
    day = (when or _today()).isoformat()
    buttons = store.setdefault("buttons", {})
    buttons[key] = {
        "last_pressed": day,
        "interval_days": int(BUTTONS[key]["days"]),
        "seeded": False,
    }
    store["buttons"] = buttons
    save_store(store)
    return buttons[key]


def button_status(key: str) -> dict[str, Any]:
    meta = BUTTONS.get(key) or {}
    store = ensure_seeded()
    entry = (store.get("buttons") or {}).get(key) or {}
    last = _parse_day(entry.get("last_pressed"))
    interval = int(entry.get("interval_days") or meta.get("days") or 1)
    today = _today()
    if last is None:
        return {
            "key": key,
            "label": meta.get("label") or key,
            "last": None,
            "next": today,
            "interval_days": interval,
            "days_left": 0,
            "state": "never",
            "note": meta.get("note") or "",
        }
    nxt = last + timedelta(days=interval)
    left = (nxt - today).days
    if left <= 0:
        state = "due"
    elif left <= max(1, interval // 3):
        state = "soon"
    else:
        state = "ok"
    return {
        "key": key,
        "label": meta.get("label") or key,
        "last": last,
        "next": nxt,
        "interval_days": interval,
        "days_left": left,
        "state": state,
        "note": meta.get("note") or "",
    }


def status_caption(key: str) -> str:
    stt = button_status(key)
    last = stt["last"]
    nxt = stt["next"]
    left = int(stt["days_left"])
    interval = int(stt["interval_days"])
    last_s = "mai" if last is None else ("oggi" if last == _today() else last.strftime("%d/%m"))
    next_s = nxt.strftime("%d/%m") if isinstance(nxt, date) else "—"
    if stt["state"] == "never":
        return f"⏱ mai premuto · consigliato: oggi (ogni {interval}g)"
    if stt["state"] == "due":
        overdue = abs(left)
        extra = f" · in ritardo di {overdue}g" if overdue else ""
        return f"🔴 premi ora · ultimo {last_s} · ogni {interval}g{extra}"
    if stt["state"] == "soon":
        return f"🟡 tra {left}g ({next_s}) · ultimo {last_s} · ogni {interval}g"
    return f"🟢 ok · prossimo tra {left}g ({next_s}) · ultimo {last_s}"


def due_summary(*, limit: int = 8) -> list[dict[str, Any]]:
    rows = [button_status(k) for k in BUTTONS]
    rows.sort(key=lambda r: (0 if r["state"] in {"never", "due"} else 1 if r["state"] == "soon" else 2, r["days_left"]))
    return [r for r in rows if r["state"] in {"never", "due", "soon"}][:limit]


def all_statuses() -> list[dict[str, Any]]:
    return [button_status(k) for k in BUTTONS]
