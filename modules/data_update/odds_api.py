"""Quote Pinnacle da The Odds API (the-odds-api.com).

Piano gratuito: 500 chiamate/mese.
Strategia: 1 fetch al giorno (o su richiesta manuale), cache JSON locale.
La cache viene usata da enrich_value come sharp odd di riferimento.

Endpoint usato: /v4/sports/soccer/odds
  - regions=eu
  - markets=h2h,totals
  - bookmakers=pinnacle
  - oddsFormat=decimal

Chiave API: salva in data/raw/odds-api.key oppure env ODDS_API_KEY.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CACHE = RAW / "pinnacle_odds.json"
KEY_PATH = RAW / "odds-api.key"
BASE = "https://api.the-odds-api.com/v4"
UA = "Mozilla/5.0 (compatible; football-predictor/1.0; +local)"

# Numero di chiamate rimanenti lette dall'ultimo header di risposta
_REMAINING_PATH = RAW / "odds-api-remaining.txt"


def _api_key() -> str | None:
    val = (os.environ.get("ODDS_API_KEY") or "").strip()
    if val:
        return val
    if KEY_PATH.exists():
        val = KEY_PATH.read_text(encoding="utf-8").strip()
        if val:
            return val
    return None


def save_api_key(key: str) -> Path:
    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    KEY_PATH.write_text(key.strip(), encoding="utf-8")
    return KEY_PATH


def _get(url: str, key: str) -> tuple[dict | list, dict]:
    """GET con chiave API. Restituisce (dati, headers)."""
    full = f"{url}&apiKey={key}" if "?" in url else f"{url}?apiKey={key}"
    req = Request(full, headers={"User-Agent": UA})
    with urlopen(req, timeout=30) as resp:
        headers = {k.lower(): v for k, v in resp.headers.items()}
        data = json.loads(resp.read().decode("utf-8"))
    return data, headers


def _remaining(headers: dict) -> int | None:
    v = headers.get("x-requests-remaining")
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _cache_is_fresh(max_age_hours: float = 20.0) -> bool:
    if not CACHE.exists():
        return False
    try:
        data = json.loads(CACHE.read_text(encoding="utf-8"))
        ts = str(data.get("fetched_at") or "")
        if ts:
            fetched = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if fetched.tzinfo is None:
                fetched = fetched.replace(tzinfo=timezone.utc)
            age_s = (datetime.now(timezone.utc) - fetched).total_seconds()
            return age_s < max_age_hours * 3600
    except Exception:
        pass
    age_s = datetime.now(timezone.utc).timestamp() - CACHE.stat().st_mtime
    return age_s < max_age_hours * 3600


def fetch_pinnacle_odds(*, force: bool = False, max_age_hours: float = 20.0) -> dict:
    """Scarica le quote Pinnacle per tutte le partite di calcio prossime.

    Restituisce un dizionario:
      {
        "ok": bool,
        "n_events": int,
        "remaining": int | None,
        "from_cache": bool,
        "events": [...],   # lista raw da Odds API
      }

    La cache viene aggiornata solo se stantia (> max_age_hours) o force=True.
    """
    key = _api_key()
    if not key:
        return {"ok": False, "error": "chiave ODDS_API_KEY non trovata", "n_events": 0, "events": [], "from_cache": False}

    if not force and _cache_is_fresh(max_age_hours):
        try:
            data = json.loads(CACHE.read_text(encoding="utf-8"))
            events = data.get("events") or []
            return {"ok": True, "n_events": len(events), "remaining": data.get("remaining"), "from_cache": True, "events": events}
        except Exception:
            pass

    try:
        url = (
            f"{BASE}/sports/soccer/odds"
            f"?regions=eu"
            f"&markets=h2h,totals"
            f"&bookmakers=pinnacle"
            f"&oddsFormat=decimal"
            f"&dateFormat=iso"
        )
        events, headers = _get(url, key)
        remaining = _remaining(headers)
        if remaining is not None:
            _REMAINING_PATH.write_text(str(remaining), encoding="utf-8")
        payload = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "remaining": remaining,
            "events": events if isinstance(events, list) else [],
        }
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        n = len(payload["events"])
        print(f"ok Pinnacle odds: {n} eventi (chiamate rimanenti: {remaining})")
        return {"ok": True, "n_events": n, "remaining": remaining, "from_cache": False, "events": payload["events"]}
    except HTTPError as exc:
        return {"ok": False, "error": f"HTTP {exc.code}: {exc.reason}", "n_events": 0, "events": [], "from_cache": False}
    except (URLError, TimeoutError) as exc:
        return {"ok": False, "error": str(exc), "n_events": 0, "events": [], "from_cache": False}


def load_pinnacle_cache() -> list[dict]:
    """Carica gli eventi dalla cache locale senza fare chiamate API."""
    if not CACHE.exists():
        return []
    try:
        data = json.loads(CACHE.read_text(encoding="utf-8"))
        return data.get("events") or []
    except Exception:
        return []


def _norm(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.strip()


def _team_match(a: str, b: str) -> bool:
    """Match fuzzy semplice: normalizza accenti e ignora case."""
    a, b = _norm(a), _norm(b)
    if a == b:
        return True
    # match parziale: uno contiene l'altro (es. "Inter" vs "Inter Milan")
    if len(a) >= 4 and len(b) >= 4:
        return a in b or b in a
    return False


def lookup_pinnacle(
    home: str,
    away: str,
    *,
    events: list[dict] | None = None,
    kickoff_date: str | None = None,
) -> dict | None:
    """Cerca le quote Pinnacle per una specifica partita.

    Restituisce un dizionario con le quote trovate:
      {
        "odd_home": float,
        "odd_draw": float | None,
        "odd_away": float,
        "odd_over_25": float | None,
        "odd_under_25": float | None,
        "event_id": str,
        "commence_time": str,
      }
    oppure None se non trovata.
    """
    if events is None:
        events = load_pinnacle_cache()
    if not events:
        return None

    kd = None
    if kickoff_date:
        try:
            kd = date.fromisoformat(str(kickoff_date)[:10])
        except ValueError:
            pass

    for ev in events:
        ev_home = str(ev.get("home_team") or "")
        ev_away = str(ev.get("away_team") or "")
        if not (_team_match(home, ev_home) and _team_match(away, ev_away)):
            continue
        # Filtro data opzionale (±1 giorno di tolleranza)
        if kd:
            ct = str(ev.get("commence_time") or "")[:10]
            try:
                ev_date = date.fromisoformat(ct)
                if abs((ev_date - kd).days) > 1:
                    continue
            except ValueError:
                pass

        result: dict = {
            "event_id": ev.get("id"),
            "commence_time": ev.get("commence_time"),
            "odd_home": None,
            "odd_draw": None,
            "odd_away": None,
            "odd_over_25": None,
            "odd_under_25": None,
        }
        for bm in ev.get("bookmakers") or []:
            if str(bm.get("key") or "").lower() != "pinnacle":
                continue
            for mkt in bm.get("markets") or []:
                mkt_key = str(mkt.get("key") or "")
                outcomes = mkt.get("outcomes") or []
                if mkt_key == "h2h":
                    for o in outcomes:
                        name = str(o.get("name") or "").lower()
                        price = o.get("price")
                        if price is None:
                            continue
                        try:
                            price = float(price)
                        except (TypeError, ValueError):
                            continue
                        if _team_match(home, name) or name in {"home", "1"}:
                            result["odd_home"] = round(price, 3)
                        elif _team_match(away, name) or name in {"away", "2"}:
                            result["odd_away"] = round(price, 3)
                        elif name in {"draw", "x", "tie"}:
                            result["odd_draw"] = round(price, 3)
                elif mkt_key == "totals":
                    for o in outcomes:
                        point = o.get("point")
                        name = str(o.get("name") or "").lower()
                        price = o.get("price")
                        if price is None or point is None:
                            continue
                        try:
                            price, point = float(price), float(point)
                        except (TypeError, ValueError):
                            continue
                        if abs(point - 2.5) < 0.01:
                            if name == "over":
                                result["odd_over_25"] = round(price, 3)
                            elif name == "under":
                                result["odd_under_25"] = round(price, 3)
        if result["odd_home"] is not None or result["odd_away"] is not None:
            return result

    return None


def remaining_calls() -> int | None:
    """Legge le chiamate rimanenti dall'ultimo fetch."""
    if _REMAINING_PATH.exists():
        try:
            return int(_REMAINING_PATH.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            pass
    return None
