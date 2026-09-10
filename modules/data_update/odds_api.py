"""Quote Pinnacle da The Odds API (the-odds-api.com).

Piano gratuito: 500 chiamate/mese.
Strategia: fetch per sport-key Big 5 (+ UCL) 1×/giorno, cache JSON locale.
La cache viene usata da enrich_value come sharp odd di riferimento.

Featured: /v4/sports/{sport_key}/odds  markets=h2h,totals
Corner (per evento): /v4/sports/{sport_key}/events/{id}/odds
  markets=alternate_totals_corners,alternate_spreads_corners,corners_1x2

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
CORNERS_CACHE = RAW / "pinnacle_corners.json"
KEY_PATH = RAW / "odds-api.key"
BASE = "https://api.the-odds-api.com/v4"
UA = "Mozilla/5.0 (compatible; football-predictor/1.0; +local)"

# Numero di chiamate rimanenti lette dall'ultimo header di risposta
_REMAINING_PATH = RAW / "odds-api-remaining.txt"

# Sport key reali (il vecchio path /sports/soccer/odds restituiva pochissimi eventi).
CORE_SPORT_KEYS: list[tuple[str, str]] = [
    ("soccer_epl", "Premier League"),
    ("soccer_spain_la_liga", "La Liga"),
    ("soccer_italy_serie_a", "Serie A"),
    ("soccer_germany_bundesliga", "Bundesliga"),
    ("soccer_france_ligue_one", "Ligue 1"),
    ("soccer_uefa_champs_league", "Champions League"),
]

CORNER_MARKETS = (
    "alternate_totals_corners",
    "alternate_spreads_corners",
    "corners_1x2",
    "alternate_team_totals_corners",
)


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
    """Scarica quote Pinnacle h2h+totals per Big 5 + UCL (una call per sport-key)."""
    key = _api_key()
    if not key:
        return {"ok": False, "error": "chiave ODDS_API_KEY non trovata", "n_events": 0, "events": [], "from_cache": False}

    if not force and _cache_is_fresh(max_age_hours):
        try:
            data = json.loads(CACHE.read_text(encoding="utf-8"))
            events = data.get("events") or []
            return {
                "ok": True,
                "n_events": len(events),
                "remaining": data.get("remaining"),
                "from_cache": True,
                "events": events,
                "by_sport": data.get("by_sport") or {},
            }
        except Exception:
            pass

    all_events: list[dict] = []
    by_sport: dict[str, int] = {}
    remaining: int | None = None
    errors: list[str] = []
    try:
        for sport_key, _title in CORE_SPORT_KEYS:
            url = (
                f"{BASE}/sports/{sport_key}/odds"
                f"?regions=eu"
                f"&markets=h2h,totals"
                f"&bookmakers=pinnacle"
                f"&oddsFormat=decimal"
                f"&dateFormat=iso"
            )
            try:
                events, headers = _get(url, key)
                remaining = _remaining(headers)
                if remaining is not None:
                    _REMAINING_PATH.write_text(str(remaining), encoding="utf-8")
            except HTTPError as exc:
                errors.append(f"{sport_key}: HTTP {exc.code}")
                continue
            except (URLError, TimeoutError) as exc:
                errors.append(f"{sport_key}: {exc}")
                continue
            batch = events if isinstance(events, list) else []
            for ev in batch:
                if isinstance(ev, dict):
                    row = dict(ev)
                    row["sport_key"] = sport_key
                    all_events.append(row)
            by_sport[sport_key] = len(batch)
        payload = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "remaining": remaining,
            "events": all_events,
            "by_sport": by_sport,
            "errors": errors,
        }
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        n = len(all_events)
        print(f"ok Pinnacle odds: {n} eventi su {len(by_sport)} sport (chiamate rimanenti: {remaining})")
        return {
            "ok": n > 0,
            "n_events": n,
            "remaining": remaining,
            "from_cache": False,
            "events": all_events,
            "by_sport": by_sport,
            "error": None if n else ("; ".join(errors) or "nessun evento"),
        }
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


def _parse_corner_markets(data: dict) -> dict:
    """Estrae linee corner O/U (e opz. 1X2/spread) da payload event odds."""
    out: dict = {
        "corners_over": {},
        "corners_under": {},
        "corners_1x2": {},
        "markets_present": [],
    }
    for bm in data.get("bookmakers") or []:
        if str(bm.get("key") or "").lower() != "pinnacle":
            continue
        for mkt in bm.get("markets") or []:
            mk = str(mkt.get("key") or "")
            out["markets_present"].append(mk)
            if mk == "alternate_totals_corners":
                for o in mkt.get("outcomes") or []:
                    try:
                        pt = float(o.get("point"))
                        price = float(o.get("price"))
                        name = str(o.get("name") or "").lower()
                    except (TypeError, ValueError):
                        continue
                    key = f"{pt:g}"
                    if name == "over":
                        out["corners_over"][key] = round(price, 3)
                    elif name == "under":
                        out["corners_under"][key] = round(price, 3)
            elif mk == "corners_1x2":
                for o in mkt.get("outcomes") or []:
                    try:
                        price = float(o.get("price"))
                    except (TypeError, ValueError):
                        continue
                    name = str(o.get("name") or "")
                    out["corners_1x2"][name] = round(price, 3)
    out["markets_present"] = sorted(set(out["markets_present"]))
    return out


def fetch_event_corner_odds(
    sport_key: str,
    event_id: str,
    *,
    force: bool = False,
) -> dict:
    """Fetch mercati corner Pinnacle per un singolo evento (1 credito tipico)."""
    key = _api_key()
    if not key:
        return {"ok": False, "error": "chiave ODDS_API_KEY non trovata"}
    cache: dict = {}
    if CORNERS_CACHE.exists() and not force:
        try:
            cache = json.loads(CORNERS_CACHE.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
        hit = (cache.get("events") or {}).get(event_id)
        if isinstance(hit, dict) and hit.get("corners_over"):
            hit = dict(hit)
            hit["from_cache"] = True
            hit["ok"] = True
            return hit

    markets = ",".join(CORNER_MARKETS)
    url = (
        f"{BASE}/sports/{sport_key}/events/{event_id}/odds"
        f"?regions=eu&markets={markets}&bookmakers=pinnacle&oddsFormat=decimal"
    )
    try:
        data, headers = _get(url, key)
        remaining = _remaining(headers)
        if remaining is not None:
            _REMAINING_PATH.write_text(str(remaining), encoding="utf-8")
    except HTTPError as exc:
        return {"ok": False, "error": f"HTTP {exc.code}: {exc.reason}", "event_id": event_id}
    except (URLError, TimeoutError) as exc:
        return {"ok": False, "error": str(exc), "event_id": event_id}

    parsed = _parse_corner_markets(data if isinstance(data, dict) else {})
    row = {
        "ok": bool(parsed.get("corners_over")),
        "event_id": event_id,
        "sport_key": sport_key,
        "home_team": (data or {}).get("home_team") if isinstance(data, dict) else None,
        "away_team": (data or {}).get("away_team") if isinstance(data, dict) else None,
        "commence_time": (data or {}).get("commence_time") if isinstance(data, dict) else None,
        "remaining": remaining,
        "from_cache": False,
        **parsed,
    }
    # advisor keys corners_over_9.5
    for line, price in (parsed.get("corners_over") or {}).items():
        row[f"corners_over_{line}"] = price
    for line, price in (parsed.get("corners_under") or {}).items():
        row[f"corners_under_{line}"] = price

    events_map = dict(cache.get("events") or {}) if isinstance(cache, dict) else {}
    events_map[event_id] = {k: v for k, v in row.items() if k != "remaining"}
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "remaining": remaining,
        "events": events_map,
    }
    CORNERS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CORNERS_CACHE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def lookup_pinnacle_corners(
    home: str,
    away: str,
    *,
    events: list[dict] | None = None,
    kickoff_date: str | None = None,
    fetch_if_missing: bool = False,
) -> dict | None:
    """Trova corner odds: cache corner → opz. fetch se evento noto in pinnacle_odds."""
    if events is None:
        events = load_pinnacle_cache()
    matched = None
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
        if kd:
            ct = str(ev.get("commence_time") or "")[:10]
            try:
                if abs((date.fromisoformat(ct) - kd).days) > 1:
                    continue
            except ValueError:
                pass
        matched = ev
        break
    if not matched:
        return None
    eid = str(matched.get("id") or "")
    sport_key = str(matched.get("sport_key") or matched.get("sport_key") or "")
    if CORNERS_CACHE.exists():
        try:
            cache = json.loads(CORNERS_CACHE.read_text(encoding="utf-8"))
            hit = (cache.get("events") or {}).get(eid)
            if isinstance(hit, dict) and hit.get("corners_over"):
                return hit
        except Exception:
            pass
    if fetch_if_missing and eid and sport_key:
        return fetch_event_corner_odds(sport_key, eid)
    return {
        "ok": False,
        "event_id": eid,
        "sport_key": sport_key,
        "needs_fetch": True,
        "home_team": matched.get("home_team"),
        "away_team": matched.get("away_team"),
    }
