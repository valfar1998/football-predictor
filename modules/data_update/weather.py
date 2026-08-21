"""Meteo pre-match da Open-Meteo (forecast + geocoding, senza chiave)."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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


def geocode_city(city: str) -> dict | None:
    key = _norm_city(city)
    if not key or len(key) < 2:
        return None
    cache = _load(GEO_CACHE)
    if key in cache:
        return cache[key]
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search?" + urlencode(
            {"name": city.strip(), "count": 1, "language": "it"}
        )
        data = _get_json(url)
        results = data.get("results") or []
        if not results:
            cache[key] = None
            _save(GEO_CACHE, cache)
            return None
        hit = results[0]
        row = {
            "lat": float(hit["latitude"]),
            "lon": float(hit["longitude"]),
            "name": hit.get("name") or city,
            "country": hit.get("country") or "",
        }
        cache[key] = row
        _save(GEO_CACHE, cache)
        return row
    except (HTTPError, URLError, TimeoutError, KeyError, TypeError, ValueError):
        return None


def _wx_key(city: str, day: str) -> str:
    return f"{_norm_city(city)}|{day}"


def forecast_day(city: str, day: str) -> dict | None:
    """Precipitazioni, vento e temperatura per una data YYYY-MM-DD."""
    key = _wx_key(city, day)
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
    geo = geocode_city(city)
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
            "city": geo.get("name") or city,
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
    if wind >= 40:
        adj *= 0.98
    return round(adj, 3)


def prefetch_weather(items: list[dict]) -> dict[str, dict]:
    """items: {city, date}. Ritorna mappa city|date -> meteo."""
    out: dict[str, dict] = {}
    seen: set[str] = set()
    for it in items:
        city = str(it.get("city") or "").strip()
        day = str(it.get("date") or "")[:10]
        if not city or not day:
            continue
        try:
            date.fromisoformat(day)
        except ValueError:
            continue
        key = _wx_key(city, day)
        if key in seen:
            continue
        seen.add(key)
        row = forecast_day(city, day)
        if row:
            out[key] = row
    return out


def lookup_weather(city: str, day: str, index: dict[str, dict] | None = None) -> dict | None:
    key = _wx_key(city, day)
    if index and key in index:
        return index[key]
    return forecast_day(city, day)
