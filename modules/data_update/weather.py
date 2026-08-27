"""Meteo pre-match da Open-Meteo (forecast + geocoding, senza chiave)."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
GEO_CACHE = RAW / "geocode_cache.json"
WX_CACHE = RAW / "weather_cache.json"
UA = "Mozilla/5.0 (compatible; football-predictor/1.0; +local)"


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_json(url: str, timeout: int = 12) -> dict:
    req = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _norm_city(city: str) -> str:
    return " ".join(str(city or "").strip().lower().split())


def geocode_city(city: str, *, venue: str | None = None) -> dict | None:
    """Geocode città; se fallisce prova nome stadio (Open-Meteo)."""
    key = _norm_city(city)
    venue_key = _norm_city(venue or "")
    cache_key = key or venue_key
    if not cache_key or len(cache_key) < 2:
        return None
    cache = _load(GEO_CACHE)
    # prefer exact city hit
    if key and key in cache:
        return cache[key]
    if venue_key and f"venue:{venue_key}" in cache:
        return cache[f"venue:{venue_key}"]

    def _search(name: str) -> dict | None:
        try:
            url = "https://geocoding-api.open-meteo.com/v1/search?" + urlencode(
                {"name": name.strip(), "count": 1, "language": "it"}
            )
            data = _get_json(url)
            results = data.get("results") or []
            if not results:
                return None
            hit = results[0]
            return {
                "lat": float(hit["latitude"]),
                "lon": float(hit["longitude"]),
                "name": hit.get("name") or name,
                "country": hit.get("country") or "",
            }
        except (HTTPError, URLError, TimeoutError, KeyError, TypeError, ValueError):
            return None

    row = _search(city) if city and len(key) >= 2 else None
    if row:
        cache[key] = row
        _save(GEO_CACHE, cache)
        return row
    if venue and len(venue_key) >= 3:
        row = _search(venue)
        if row:
            cache[f"venue:{venue_key}"] = row
            if key:
                cache[key] = row
            _save(GEO_CACHE, cache)
            return row
    if key:
        cache[key] = None
        _save(GEO_CACHE, cache)
    return None


def _wx_key(city: str, day: str) -> str:
    return f"{_norm_city(city)}|{day}"


def forecast_day(city: str, day: str, *, venue: str | None = None) -> dict | None:
    """Precipitazioni, vento e temperatura per una data YYYY-MM-DD."""
    key = _wx_key(city or venue or "", day)
    cache = _load(WX_CACHE)
    hit = cache.get(key)
    if isinstance(hit, dict):
        ts = str(hit.get("fetched_at") or "")
        try:
            fetched = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if fetched.tzinfo is None:
                fetched = fetched.replace(tzinfo=timezone.utc)
            if (datetime.now(timezone.utc) - fetched).total_seconds() < 12 * 3600:
                return hit
        except ValueError:
            return hit
    geo = geocode_city(city, venue=venue)
    if not geo:
        return None
    try:
        url = "https://api.open-meteo.com/v1/forecast?" + urlencode(
            {
                "latitude": f"{geo['lat']:.4f}",
                "longitude": f"{geo['lon']:.4f}",
                "daily": "precipitation_sum,precipitation_probability_max,wind_speed_10m_max,temperature_2m_max",
                "timezone": "auto",
                "start_date": day,
                "end_date": day,
            }
        )
        data = _get_json(url)
        daily = data.get("daily") or {}
        precip = (daily.get("precipitation_sum") or [None])[0]
        pop = (daily.get("precipitation_probability_max") or [None])[0]
        wind = (daily.get("wind_speed_10m_max") or [None])[0]
        temp = (daily.get("temperature_2m_max") or [None])[0]
        row = {
            "city": geo.get("name") or city or venue,
            "venue": venue or "",
            "date": day,
            "precip_mm": None if precip is None else round(float(precip), 1),
            "precip_prob": None if pop is None else int(pop),
            "wind_kmh": None if wind is None else round(float(wind), 1),
            "temp_c": None if temp is None else round(float(temp), 1),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        row["flag"] = _flag(row)
        row["lambda_adj"] = _lambda_adj(row)
        cache[key] = row
        _save(WX_CACHE, cache)
        return row
    except (HTTPError, URLError, TimeoutError, TypeError, ValueError):
        return None


def _flag(row: dict) -> str:
    precip = row.get("precip_mm") or 0
    wind = row.get("wind_kmh") or 0
    if precip >= 8 or wind >= 40:
        return "avverso"
    if precip >= 3 or wind >= 28:
        return "umido/ventoso"
    return "ok"


def _lambda_adj(row: dict) -> float:
    precip = float(row.get("precip_mm") or 0)
    wind = float(row.get("wind_kmh") or 0)
    adj = 1.0
    if precip >= 10:
        adj *= 0.96
    elif precip >= 5:
        adj *= 0.98
    elif precip >= 3:
        adj *= 0.99
    if wind >= 40:
        adj *= 0.97
    elif wind >= 28:
        adj *= 0.985
    elif wind >= 25:
        adj *= 0.992
    return round(adj, 3)


def prefetch_weather(items: list[dict]) -> dict[str, dict]:
    """items: {city, date, venue?}. Ritorna mappa city|date -> meteo."""
    out: dict[str, dict] = {}
    seen: set[str] = set()
    for it in items:
        city = str(it.get("city") or "").strip()
        venue = str(it.get("venue") or "").strip() or None
        day = str(it.get("date") or "")[:10]
        if (not city and not venue) or not day:
            continue
        try:
            date.fromisoformat(day)
        except ValueError:
            continue
        key = _wx_key(city or venue or "", day)
        if key in seen:
            continue
        seen.add(key)
        row = forecast_day(city, day, venue=venue)
        if row:
            out[key] = row
    return out


def lookup_weather(
    city: str,
    day: str,
    index: dict[str, dict] | None = None,
    *,
    venue: str | None = None,
) -> dict | None:
    key = _wx_key(city or venue or "", day)
    if index and key in index:
        return index[key]
    return forecast_day(city, day, venue=venue)


def geocode_batch_venues(
    *,
    max_n: int = 120,
    sleep_s: float = 0.35,
    on_progress: Callable[[float, str], None] | None = None,
) -> dict:
    """Backfill lat/lon per stadi in home_venues.csv (Open-Meteo, rate-limited)."""
    import time

    from modules.data_update.venues import VENUE_CACHE, load_home_venues
    from modules.progress_report import emit

    idx = load_home_venues()
    if not idx:
        return {"ok": False, "n": 0, "error": "home_venues.csv vuoto"}
    cache = _load(GEO_CACHE)
    done = 0
    hits = 0
    skipped = 0
    emit(on_progress, 0.02, "Geocode batch…")
    items = list(idx.items())
    for team, row in items[: max(1, int(max_n) * 2)]:
        if done >= max_n:
            break
        venue = str(row.get("venue") or "").strip()
        city = str(row.get("venue_city") or row.get("city") or "").strip()
        if not venue and not city:
            skipped += 1
            continue
        key_c = _norm_city(city) if city else ""
        key_v = _norm_city(venue) if venue else ""
        if (key_c and cache.get(key_c)) or (key_v and cache.get(f"venue:{key_v}")):
            hits += 1
            continue
        emit(on_progress, min(0.95, 0.05 + 0.9 * (done / max(1, max_n))), f"{team}: {city or venue}")
        geo = geocode_city(city or venue, venue=venue or None)
        done += 1
        if geo:
            hits += 1
        time.sleep(max(0.05, float(sleep_s)))
    # coverage
    covered = 0
    total = 0
    for row in idx.values():
        total += 1
        venue = str(row.get("venue") or "").strip()
        city = str(row.get("venue_city") or row.get("city") or "").strip()
        key_c = _norm_city(city) if city else ""
        key_v = _norm_city(venue) if venue else ""
        if (key_c and _load(GEO_CACHE).get(key_c)) or (key_v and _load(GEO_CACHE).get(f"venue:{key_v}")):
            covered += 1
    emit(on_progress, 1.0, f"Geocode OK · queried={done} coverage={covered}/{total}")
    return {
        "ok": True,
        "queried": done,
        "hits_cache_or_new": hits,
        "skipped": skipped,
        "coverage": round(covered / total, 3) if total else 0.0,
        "n_venues": total,
        "path": str(GEO_CACHE),
        "venue_cache": str(VENUE_CACHE),
    }
