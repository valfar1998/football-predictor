"""Quote Exchange Betfair (Delayed App Key).

Login: identitysso.betfair.it
Betting: api.betfair.com Exchange JSON-RPC.

Credenziali:
  .env  BETFAIR_APP_KEY, BETFAIR_USERNAME, BETFAIR_PASSWORD
  oppure data/raw/betfair.appkey (solo Delayed Key)

La Live Key non va usata finché non è attiva (pagamento).
"""

from __future__ import annotations

import json
import os
import time
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
ENV_PATH = ROOT / ".env"
KEY_PATH = RAW / "betfair.appkey"
SESSION_PATH = RAW / "betfair.session.json"
CACHE = RAW / "betfair_odds.json"

LOGIN_URL = "https://identitysso.betfair.it/api/login"
KEEPALIVE_URL = "https://identitysso.betfair.it/api/keepAlive"
BETTING_URL = "https://api.betfair.com/exchange/betting/json-rpc/v1"
UA = "Mozilla/5.0 (compatible; football-predictor/1.0; +local)"

SOCCER_EVENT_TYPE = "1"
MARKET_TYPES = ("MATCH_ODDS", "OVER_UNDER_25")
SESSION_MAX_AGE_H = 6.0
BOOK_BATCH = 40


def _parse_env(path: Path) -> dict[str, str]:
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


def _load_env() -> dict[str, str]:
    data = _parse_env(ENV_PATH)
    for key, val in data.items():
        if key.startswith("BETFAIR_") and val and key not in os.environ:
            os.environ[key] = val
    return data


def upsert_env(updates: dict[str, str]) -> Path:
    """Aggiorna o aggiunge chiavi in .env senza toccare le altre righe."""
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    seen: set[str] = set()
    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                seen.add(key)
                continue
        new_lines.append(line)
    missing = [k for k in updates if k not in seen]
    if missing:
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        new_lines.append("# Betfair Exchange")
        for key in missing:
            new_lines.append(f"{key}={updates[key]}")
    ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    for key, val in updates.items():
        if val:
            os.environ[key] = val
        elif key in os.environ:
            del os.environ[key]
    return ENV_PATH


def save_app_key(key: str, *, live_key: str | None = None) -> Path:
    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    KEY_PATH.write_text(key.strip(), encoding="utf-8")
    payload = {"BETFAIR_APP_KEY": key.strip()}
    if live_key:
        payload["BETFAIR_APP_KEY_LIVE"] = live_key.strip()
    upsert_env(payload)
    return KEY_PATH


def save_credentials(username: str, password: str) -> Path:
    return upsert_env(
        {
            "BETFAIR_USERNAME": username.strip(),
            "BETFAIR_PASSWORD": password.strip(),
        }
    )


def _app_key() -> str | None:
    _load_env()
    val = (os.environ.get("BETFAIR_APP_KEY") or "").strip()
    if val:
        return val
    if KEY_PATH.exists():
        val = KEY_PATH.read_text(encoding="utf-8").strip()
        if val:
            return val
    return None


def _credentials() -> tuple[str | None, str | None]:
    _load_env()
    user = (os.environ.get("BETFAIR_USERNAME") or "").strip()
    pwd = (os.environ.get("BETFAIR_PASSWORD") or "").strip()
    return (user or None, pwd or None)


def app_key_configured() -> bool:
    return bool(_app_key())


def login_configured() -> bool:
    user, pwd = _credentials()
    return bool(_app_key() and user and pwd)


def _post_form(url: str, data: dict[str, str], headers: dict[str, str]) -> dict:
    req = Request(
        url,
        data=urlencode(data).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json", **headers},
    )
    with urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _rpc(method: str, params: dict, token: str, app_key: str) -> object:
    payload = {"jsonrpc": "2.0", "method": f"SportsAPING/v1.0/{method}", "params": params, "id": 1}
    req = Request(
        BETTING_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Application": app_key,
            "X-Authentication": token,
            "User-Agent": UA,
        },
    )
    with urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    if isinstance(body, list):
        body = body[0] if body else {}
    if body.get("error"):
        err = body["error"]
        data = err.get("data") if isinstance(err, dict) else err
        raise RuntimeError(f"{method} fallito: {data or err}")
    return body.get("result")


def _read_session() -> str | None:
    if not SESSION_PATH.exists():
        return None
    try:
        data = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    token = str(data.get("token") or "").strip()
    ts = str(data.get("saved_at") or "")
    if not token or not ts:
        return None
    try:
        saved = datetime.fromisoformat(ts)
        if saved.tzinfo is None:
            saved = saved.replace(tzinfo=timezone.utc)
        age_h = (datetime.now(timezone.utc) - saved).total_seconds() / 3600
    except ValueError:
        return None
    if age_h > SESSION_MAX_AGE_H:
        return None
    return token


def _write_session(token: str) -> None:
    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSION_PATH.write_text(
        json.dumps({"token": token, "saved_at": datetime.now(timezone.utc).isoformat()}, indent=2),
        encoding="utf-8",
    )


def login(*, force: bool = False) -> str:
    app_key = _app_key()
    if not app_key:
        raise RuntimeError("BETFAIR_APP_KEY assente")
    if not force:
        cached = _read_session()
        if cached:
            try:
                alive = _post_form(
                    KEEPALIVE_URL,
                    {},
                    {"X-Application": app_key, "X-Authentication": cached},
                )
                if str(alive.get("status") or "").upper() == "SUCCESS":
                    return cached
            except Exception:
                pass
    user, pwd = _credentials()
    if not user or not pwd:
        raise RuntimeError("Manca BETFAIR_USERNAME o BETFAIR_PASSWORD nel .env")
    result = _post_form(
        LOGIN_URL,
        {"username": user, "password": pwd},
        {"X-Application": app_key},
    )
    if result.get("status") != "SUCCESS":
        raise RuntimeError(f"Login Betfair fallito: {result.get('error') or result.get('errorCode') or result}")
    token = str(result.get("token") or "").strip()
    if not token:
        raise RuntimeError(f"Login Betfair senza token: {result}")
    _write_session(token)
    return token


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _split_event(name: str) -> tuple[str, str]:
    for sep in (" vs ", " v ", " - "):
        if sep in name:
            a, b = name.split(sep, 1)
            return a.strip(), b.strip()
    return name.strip(), ""


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.strip()


def _team_match(a: str, b: str) -> bool:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) >= 4 and len(b) >= 4:
        return a in b or b in a
    return False


def _is_draw(name: str) -> bool:
    n = _norm(name)
    return n in {"the draw", "draw", "pareggio", "x"} or "draw" in n or "pareggio" in n


def _is_over_25(name: str) -> bool:
    n = _norm(name)
    return n.startswith("over") and "2.5" in n


def _is_under_25(name: str) -> bool:
    n = _norm(name)
    return n.startswith("under") and "2.5" in n


def _best_back(runner: dict) -> float | None:
    ex = runner.get("ex") or {}
    backs = ex.get("availableToBack") or []
    if backs:
        try:
            return round(float(backs[0]["price"]), 3)
        except (TypeError, ValueError, KeyError):
            pass
    last = runner.get("lastPriceTraded")
    if last is None:
        return None
    try:
        return round(float(last), 3)
    except (TypeError, ValueError):
        return None


def _catalogue_window(token: str, app_key: str, start: datetime, end: datetime) -> list[dict]:
    return list(
        _rpc(
            "listMarketCatalogue",
            {
                "filter": {
                    "eventTypeIds": [SOCCER_EVENT_TYPE],
                    "marketTypeCodes": list(MARKET_TYPES),
                    "marketStartTime": {"from": _iso(start), "to": _iso(end)},
                },
                "maxResults": 1000,
                "marketProjection": [
                    "COMPETITION",
                    "EVENT",
                    "MARKET_START_TIME",
                    "MARKET_DESCRIPTION",
                    "RUNNER_DESCRIPTION",
                ],
            },
            token,
            app_key,
        )
        or []
    )


def _catalogue_range(token: str, app_key: str, start: datetime, end: datetime, *, depth: int = 0) -> list[dict]:
    try:
        rows = _catalogue_window(token, app_key, start, end)
        time.sleep(0.08)
        return rows
    except RuntimeError as exc:
        if "TOO_MUCH_DATA" not in str(exc) or depth >= 5:
            raise
        mid = start + (end - start) / 2
        left = _catalogue_range(token, app_key, start, mid, depth=depth + 1)
        right = _catalogue_range(token, app_key, mid, end, depth=depth + 1)
        return left + right


def _market_books(token: str, app_key: str, market_ids: list[str]) -> list[dict]:
    out: list[dict] = []
    for i in range(0, len(market_ids), BOOK_BATCH):
        chunk = market_ids[i : i + BOOK_BATCH]
        books = _rpc(
            "listMarketBook",
            {
                "marketIds": chunk,
                "priceProjection": {"priceData": ["EX_BEST_OFFERS"], "virtualise": True},
            },
            token,
            app_key,
        ) or []
        out.extend(books)
        time.sleep(0.12)
    return out


def _group_events(catalogue: list[dict]) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    for mkt in catalogue:
        event = mkt.get("event") or {}
        event_id = str(event.get("id") or "")
        if not event_id:
            continue
        name = str(event.get("name") or "")
        home, away = _split_event(name)
        row = grouped.setdefault(
            event_id,
            {
                "event_id": event_id,
                "event_name": name,
                "home": home,
                "away": away,
                "commence_time": event.get("openDate") or mkt.get("marketStartTime"),
                "competition": (mkt.get("competition") or {}).get("name") or "",
                "markets": {},
                "runners": {},
            },
        )
        mid = str(mkt.get("marketId") or "")
        runners = {
            int(r["selectionId"]): str(r.get("runnerName") or "")
            for r in (mkt.get("runners") or [])
            if r.get("selectionId") is not None
        }
        mtype = str((mkt.get("description") or {}).get("marketType") or "")
        name = str(mkt.get("marketName") or "").lower()
        if mtype == "MATCH_ODDS" or "match odds" in name:
            row["markets"]["MATCH_ODDS"] = mid
            row["runners"]["MATCH_ODDS"] = runners
        elif mtype == "OVER_UNDER_25" or ("2.5" in name and "over" in name):
            row["markets"]["OVER_UNDER_25"] = mid
            row["runners"]["OVER_UNDER_25"] = runners
    return grouped


def _fetch_catalogue_and_books(
    token: str,
    app_key: str,
    *,
    days: int,
) -> list[dict]:
    now = datetime.now(timezone.utc)
    catalogue: list[dict] = []
    # Finestre da 6h: riduce TOO_MUCH_DATA rispetto al giorno intero.
    steps = max(1, int(days) * 4)
    for i in range(steps):
        start = now + timedelta(hours=6 * i)
        end = now + timedelta(hours=6 * (i + 1))
        catalogue.extend(_catalogue_range(token, app_key, start, end))
    grouped = _group_events(catalogue)
    market_ids = []
    for row in grouped.values():
        market_ids.extend([mid for mid in row["markets"].values() if mid])
    books = {str(b.get("marketId")): b for b in _market_books(token, app_key, market_ids)}

    events: list[dict] = []
    for row in grouped.values():
        ev = {
            "event_id": row["event_id"],
            "home": row["home"],
            "away": row["away"],
            "commence_time": row["commence_time"],
            "competition": row["competition"],
            "odd_home": None,
            "odd_draw": None,
            "odd_away": None,
            "odd_over_25": None,
            "odd_under_25": None,
        }
        match_id = row["markets"].get("MATCH_ODDS")
        if match_id and match_id in books:
            names = row["runners"].get("MATCH_ODDS") or {}
            leftover: list[tuple[str, float]] = []
            for runner in books[match_id].get("runners") or []:
                sid = runner.get("selectionId")
                name = names.get(int(sid) if sid is not None else -1, "")
                price = _best_back(runner)
                if price is None:
                    continue
                if _is_draw(name):
                    ev["odd_draw"] = price
                elif _team_match(row["home"], name):
                    ev["odd_home"] = price
                elif _team_match(row["away"], name):
                    ev["odd_away"] = price
                else:
                    leftover.append((name, price))
            if leftover and (ev["odd_home"] is None or ev["odd_away"] is None):
                for name, price in leftover:
                    if ev["odd_home"] is None:
                        ev["odd_home"] = price
                    elif ev["odd_away"] is None:
                        ev["odd_away"] = price
        ou_id = row["markets"].get("OVER_UNDER_25")
        if ou_id and ou_id in books:
            names = row["runners"].get("OVER_UNDER_25") or {}
            for runner in books[ou_id].get("runners") or []:
                sid = runner.get("selectionId")
                name = names.get(int(sid) if sid is not None else -1, "")
                price = _best_back(runner)
                if price is None:
                    continue
                if _is_over_25(name):
                    ev["odd_over_25"] = price
                elif _is_under_25(name):
                    ev["odd_under_25"] = price
        if ev["odd_home"] or ev["odd_away"] or ev["odd_over_25"]:
            events.append(ev)
    return events


def fetch_betfair_odds(*, force: bool = False, days: int = 7, max_age_hours: float = 6.0) -> dict:
    """Scarica 1X2 e Over/Under 2.5 Exchange per il calcio in arrivo."""
    if not force and CACHE.exists():
        fresh = False
        try:
            data = json.loads(CACHE.read_text(encoding="utf-8"))
            ts = str(data.get("fetched_at") or "")
            if ts:
                fetched = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if fetched.tzinfo is None:
                    fetched = fetched.replace(tzinfo=timezone.utc)
                age_s = (datetime.now(timezone.utc) - fetched).total_seconds()
                fresh = age_s < max_age_hours * 3600
            else:
                age_s = datetime.now(timezone.utc).timestamp() - CACHE.stat().st_mtime
                fresh = age_s < max_age_hours * 3600
            if fresh:
                events = data.get("events") or []
                return {
                    "ok": True,
                    "n_events": len(events),
                    "from_cache": True,
                    "events": events,
                }
        except Exception:
            pass

    app_key = _app_key()
    if not app_key:
        return {"ok": False, "error": "BETFAIR_APP_KEY non trovata", "n_events": 0, "events": [], "from_cache": False}

    try:
        token = login(force=force)
        try:
            events = _fetch_catalogue_and_books(token, app_key, days=days)
        except RuntimeError as exc:
            # Sessione “viva” al keepAlive ma rifiutata dall’Exchange → re-login forzato.
            if "INVALID_SESSION" not in str(exc):
                raise
            token = login(force=True)
            events = _fetch_catalogue_and_books(token, app_key, days=days)

        payload = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "n_events": len(events),
            "events": events,
        }
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"ok Betfair odds: {len(events)} eventi")
        return {"ok": True, "n_events": len(events), "from_cache": False, "events": events}
    except HTTPError as exc:
        return {"ok": False, "error": f"HTTP {exc.code}: {exc.reason}", "n_events": 0, "events": [], "from_cache": False}
    except (URLError, TimeoutError, RuntimeError) as exc:
        return {"ok": False, "error": str(exc), "n_events": 0, "events": [], "from_cache": False}


def load_betfair_cache() -> list[dict]:
    if not CACHE.exists():
        return []
    try:
        data = json.loads(CACHE.read_text(encoding="utf-8"))
        return data.get("events") or []
    except Exception:
        return []


def lookup_betfair(
    home: str,
    away: str,
    *,
    events: list[dict] | None = None,
    kickoff_date: str | None = None,
) -> dict | None:
    if events is None:
        events = load_betfair_cache()
    if not events:
        return None
    kd = None
    if kickoff_date:
        try:
            kd = date.fromisoformat(str(kickoff_date)[:10])
        except ValueError:
            pass
    for ev in events:
        if not (_team_match(home, str(ev.get("home") or "")) and _team_match(away, str(ev.get("away") or ""))):
            continue
        if kd:
            ct = str(ev.get("commence_time") or "")[:10]
            try:
                if abs((date.fromisoformat(ct) - kd).days) > 1:
                    continue
            except ValueError:
                pass
        return ev
    return None
