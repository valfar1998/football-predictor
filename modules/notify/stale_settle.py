"""Alert Telegram: pending ROI (voto ≥8) ancora aperti dopo N giorni → settle manuale."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from modules.notify.telegram import load_credentials, send_message

ROOT = Path(__file__).resolve().parents[2]
SENT = ROOT / "data" / "processed" / "telegram_stale_settle_sent.json"
STALE_DAYS = 8
REMIND_EVERY_DAYS = 7
CHUNK = 8
BRAND = "FOOTBALL PREDICTOR"
TZ = ZoneInfo("Europe/Rome")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _load_sent() -> dict[str, dict]:
    if not SENT.is_file():
        return {}
    try:
        raw = json.loads(SENT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict] = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            out[str(k)] = v
        elif isinstance(v, str):
            out[str(k)] = {"first_at": v, "last_at": v}
    return out


def _save_sent(data: dict[str, dict]) -> None:
    SENT.parent.mkdir(parents=True, exist_ok=True)
    SENT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _age_days(day: str) -> int | None:
    try:
        d0 = datetime.fromisoformat(str(day)[:10])
        return max(0, (_now().replace(tzinfo=None) - d0).days)
    except ValueError:
        return None


def _should_notify(prev: dict | None, *, age: int) -> bool:
    if age < STALE_DAYS:
        return False
    if not prev or not prev.get("last_at"):
        return True
    try:
        last = datetime.fromisoformat(str(prev["last_at"]).replace("Z", "+00:00"))
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        elapsed = (_now() - last).total_seconds() / 86400.0
        return elapsed >= REMIND_EVERY_DAYS
    except ValueError:
        return True


def notify_stale_pending_settlements(*, dry_run: bool = False) -> dict:
    """Notifica pending voto≥8 / freeze aperti da ≥STALE_DAYS giorni."""
    from modules.data_update.history import list_stale_roi_pending

    stale = list_stale_roi_pending(min_days=STALE_DAYS)
    sent = _load_sent()
    # pulisci chiavi già settled (non più in stale)
    stale_keys = {str(r.get("match_key") or "") for r in stale}
    pruned = {k: v for k, v in sent.items() if k in stale_keys}
    if len(pruned) != len(sent):
        sent = pruned
        if not dry_run:
            _save_sent(sent)

    to_send = []
    for r in stale:
        key = str(r.get("match_key") or "")
        age = int(r.get("age_days") or 0)
        if not key or not _should_notify(sent.get(key), age=age):
            continue
        to_send.append(r)

    if not to_send:
        return {
            "ok": True,
            "n_stale": len(stale),
            "n_new": 0,
            "n_sent": 0,
            "dry_run": dry_run,
            "min_days": STALE_DAYS,
        }

    when = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    header = (
        f"{BRAND} — settle manuale\n{when} Roma\n\n"
        f"⏳ Pending voto ≥8 aperti da ≥{STALE_DAYS} giorni "
        f"({len(to_send)} da inserire).\n"
        "In Streamlit → Valutazione → Settle manuale, inserisci FT "
        "(e corner/tiri se serve) oppure Presa / Non presa.\n"
    )
    blocks = []
    for r in to_send:
        age = r.get("age_days")
        q = r.get("quota_pick")
        q_s = f"{float(q):.2f}" if q is not None else "—"
        blocks.append(
            f"• {r.get('date')} · {r.get('league') or '?'}\n"
            f"  {r.get('home')} vs {r.get('away')}\n"
            f"  Pick {r.get('pick')} · voto {r.get('score_unified')} · q {q_s} · "
            f"+{age}g"
        )

    messages = []
    for i in range(0, len(blocks), CHUNK):
        chunk = blocks[i : i + CHUNK]
        cont = " (cont.)" if i else ""
        messages.append(header.replace("settle manuale", f"settle manuale{cont}") + "\n".join(chunk))

    n_sent = 0
    if dry_run:
        for msg in messages:
            print(msg)
            print("---")
    else:
        if not load_credentials():
            return {
                "ok": False,
                "error": "telegram credenziali assenti",
                "n_stale": len(stale),
                "n_new": len(to_send),
            }
        now_iso = _now().strftime("%Y-%m-%dT%H:%M:%SZ")
        for msg in messages:
            if send_message(msg):
                n_sent += 1
        if n_sent:
            for r in to_send:
                key = str(r.get("match_key") or "")
                prev = sent.get(key) or {}
                sent[key] = {
                    "first_at": prev.get("first_at") or now_iso,
                    "last_at": now_iso,
                    "date": r.get("date"),
                    "home": r.get("home"),
                    "away": r.get("away"),
                    "pick": r.get("pick"),
                }
            _save_sent(sent)

    return {
        "ok": True,
        "n_stale": len(stale),
        "n_new": len(to_send),
        "n_sent": n_sent,
        "dry_run": dry_run,
        "min_days": STALE_DAYS,
        "matches": [
            {
                "match_key": r.get("match_key"),
                "date": r.get("date"),
                "home": r.get("home"),
                "away": r.get("away"),
                "pick": r.get("pick"),
                "age_days": r.get("age_days"),
            }
            for r in to_send
        ],
    }
