"""Stesso bot Telegram di telegram-offerte-sconto: TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIBLING_ENV_DIRS = (
    "telegram-offerte-sconto",
    "offerte_notifications",
    "offerte-notifications",
)


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def _env_candidates() -> list[Path]:
    paths = [ROOT / ".env"]
    parent = ROOT.parent
    extra = os.getenv("OFFERTE_NOTIFICATIONS_DIR", "").strip()
    if extra:
        paths.append(Path(extra) / ".env")
    for name in SIBLING_ENV_DIRS:
        paths.append(parent / name / ".env")
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def load_credentials() -> dict[str, str] | None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if token and chat:
        return {"token": token, "chat_id": chat, "source": "env"}

    for path in _env_candidates():
        data = _parse_env_file(path)
        token = token or data.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat = chat or data.get("TELEGRAM_CHAT_ID", "").strip()
        if token and chat:
            os.environ.setdefault("TELEGRAM_BOT_TOKEN", token)
            os.environ.setdefault("TELEGRAM_CHAT_ID", chat)
            return {"token": token, "chat_id": chat, "source": str(path)}
    return None


def telegram_status() -> str:
    creds = load_credentials()
    if not creds:
        return (
            "Telegram: manca TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID "
            "(stesso .env di telegram-offerte-sconto)."
        )
    src = creds["source"]
    if src == "env":
        where = "variabili d'ambiente"
    else:
        where = Path(src).parent.name
    return f"Telegram: bot offerte pronto ({where}). Voto >=9 e spread Raro >=1."


def send_message(text: str, *, delay: float = 0.5) -> bool:
    creds = load_credentials()
    if not creds:
        print("telegram skip: credenziali assenti")
        return False
    url = f"https://api.telegram.org/bot{creds['token']}/sendMessage"
    payload = urllib.parse.urlencode(
        {
            "chat_id": creds["chat_id"],
            "text": text[:4000],
            "disable_web_page_preview": "true",
        }
    ).encode()
    if delay > 0:
        time.sleep(delay)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=payload), timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if not body.get("ok"):
            print(f"telegram errore: {body.get('description') or body}")
            return False
        return True
    except urllib.error.URLError as exc:
        print(f"telegram errore: {exc}")
        return False
    except json.JSONDecodeError as exc:
        print(f"telegram errore: risposta non JSON ({exc})")
        return False
