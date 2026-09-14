"""Palinsesto football Unibet via Kambi Guest API (1X2, O/U gol, corner/tiri su evento).

Stesso schema del tennis-predictor (DoH + curl_cffi), sport=football.
Corner/tiri: non sono nel listView — si arricchiscono on-demand da
`/betoffer/event/{id}.json` (Total Corners / Total Shots).
"""

from __future__ import annotations

import json
import os
import socket
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from modules.data_update.cache_policy import is_fresh

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "kambi_football_odds.json"
DEFAULT_CLIENT = "ub"
DEFAULT_LANG = "it_IT"
DEFAULT_MARKET = "IT"
DEFAULT_MAX_AGE_MIN = 45.0
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
ENDPOINTS = (
    {
        "url": "https://eu1.offering-api.kambicdn.com/offering/v2018/{client}/listView/football.json",
        "doh": "eu1.offering-api.kambicdn.com",
    },
    {
        "url": "https://eu-offering.kambicdn.org/offering/v2018/{client}/listView/football.json",
        "doh": "eu-offering.kambicdn.org",
    },
)
SKIP_STATES = frozenset({"FINISHED", "CANCELLED", "ABANDONED", "POSTPONED", "SUSPENDED"})
CORNER_LINES = (7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5)
SHOT_LINES = (18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5)


def _enabled() -> bool:
    return (os.environ.get("KAMBI_FOOTBALL_ENABLED") or "1").strip().lower() in ("1", "true", "yes")


def _client() -> str:
    return (os.environ.get("KAMBI_CLIENT") or DEFAULT_CLIENT).strip() or DEFAULT_CLIENT


def _params() -> dict[str, str]:
    return {
        "lang": (os.environ.get("KAMBI_LANG") or DEFAULT_LANG).strip() or DEFAULT_LANG,
        "market": (os.environ.get("KAMBI_MARKET") or DEFAULT_MARKET).strip() or DEFAULT_MARKET,
        "client_id": (os.environ.get("KAMBI_CLIENT_ID") or "2").strip() or "2",
        "channel_id": (os.environ.get("KAMBI_CHANNEL_ID") or "1").strip() or "1",
        "useCombined": "true",
    }


def _headers(host: str) -> dict[str, str]:
    referer = (os.environ.get("KAMBI_REFERER") or "https://www.unibet.it/").strip()
    return {
        "User-Agent": UA,
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
        "Referer": referer,
        "Origin": referer.rstrip("/"),
        "Host": host,
    }


def _system_ips(hostname: str) -> list[str]:
    try:
        return sorted({str(item[4][0]) for item in socket.getaddrinfo(hostname, 443, proto=socket.IPPROTO_TCP)})
    except OSError:
        return []


def _doh_ips(hostname: str) -> list[str]:
    try:
        import requests as http

        resp = http.get(
            "https://cloudflare-dns.com/dns-query",
            params={"name": hostname, "type": "A"},
            headers={"accept": "application/dns-json"},
            timeout=8,
        )
        if resp.status_code != 200:
            return []
        return [
            str(row["data"])
            for row in resp.json().get("Answer") or []
            if row.get("type") == 1 and row.get("data")
        ]
    except Exception:
        return []


def _resolve_ips(hostname: str) -> list[str]:
    sys_ips = _system_ips(hostname)
    if sys_ips and not all(ip.startswith("127.") for ip in sys_ips):
        return sys_ips
    doh = _doh_ips(hostname)
    if doh:
        if sys_ips and any(ip.startswith("127.") for ip in sys_ips):
            print(
                f"  Kambi DNS bloccato per {hostname} ({', '.join(sys_ips)}) — uso DoH ({', '.join(doh[:2])})",
                flush=True,
            )
        return doh
    return sys_ips


def _http_get_json(url: str, *, params: dict[str, str], host: str, ips: list[str]) -> tuple[Any, str | None]:
    last_err: str | None = None
    try:
        from curl_cffi import CurlOpt, requests as http

        for ip in ips or [""]:
            opts = {CurlOpt.RESOLVE: [f"{host}:443:{ip}"]} if ip and not ip.startswith("127.") else None
            try:
                resp = http.get(
                    url,
                    params=params,
                    headers=_headers(host),
                    timeout=25,
                    impersonate="chrome",
                    curl_options=opts,
                )
                if resp.status_code == 200:
                    return resp.json(), None
                last_err = f"HTTP {resp.status_code} ({host})"
            except Exception as exc:
                last_err = str(exc)
    except ImportError:
        pass

    try:
        import requests as http

        resp = http.get(url, params=params, headers=_headers(host), timeout=25)
        if resp.status_code == 200:
            return resp.json(), None
        last_err = f"HTTP {resp.status_code} ({host})"
    except Exception as exc:
        last_err = str(exc)
    return None, last_err


def _fetch_payload() -> tuple[Any, str | None, str | None]:
    client = _client()
    params = _params()
    errors: list[str] = []
    used_host: str | None = None
    for endpoint in ENDPOINTS:
        url = endpoint["url"].format(client=client)
        host = urlparse(url).hostname or endpoint["doh"]
        used_host = host
        ips = _resolve_ips(endpoint["doh"])
        payload, err = _http_get_json(url, params=params, host=host, ips=ips)
        if payload is not None:
            return payload, None, host
        if err:
            errors.append(f"{host}: {err}")
        time.sleep(0.3)
    return None, "; ".join(errors) or "Kambi non raggiungibile", used_host


def _kambi_decimal(odds_raw: Any) -> float | None:
    if odds_raw is None:
        return None
    try:
        val = float(odds_raw)
    except (TypeError, ValueError):
        return None
    if val <= 0:
        return None
    if val > 20:
        val = val / 1000.0
    return round(val, 3) if val > 1.01 else None


def _path_label(path: list[dict] | None) -> str:
    if not isinstance(path, list):
        return ""
    parts: list[str] = []
    for node in path:
        if not isinstance(node, dict):
            continue
        label = str(node.get("englishName") or node.get("name") or "").strip()
        if label and label.lower() not in ("football", "soccer"):
            parts.append(label)
    return " / ".join(parts)


def _is_esportish(name: str, path_label: str) -> bool:
    blob = f"{name} {path_label}".lower()
    if "esport" in blob or "virtual" in blob or "cyber" in blob:
        return True
    # nickname stile "Monaco (Gaga)"
    if "(" in name and ")" in name and " u1" not in blob:
        # Youth leagues keep parentheses sometimes — allow U19/U20
        if "u19" in blob or "u20" in blob or "youth" in blob:
            return False
        return True
    return False


def _pick_1x2(offers: list[dict]) -> tuple[float | None, float | None, float | None]:
    for offer in offers:
        if not isinstance(offer, dict) or offer.get("closed"):
            continue
        crit = str((offer.get("criterion") or {}).get("englishLabel") or "").lower()
        otype = str((offer.get("betOfferType") or {}).get("englishName") or "").lower()
        if crit not in {"full time", "match"} and otype not in {"match", "1x2"}:
            if "full time" not in crit:
                continue
        by_type: dict[str, float] = {}
        for outcome in offer.get("outcomes") or []:
            if not isinstance(outcome, dict):
                continue
            if str(outcome.get("status") or "OPEN").upper() != "OPEN":
                continue
            odd = _kambi_decimal(outcome.get("odds"))
            if odd is None:
                continue
            otype_o = str(outcome.get("type") or "").upper()
            if otype_o:
                by_type[otype_o] = odd
        if "OT_ONE" in by_type and "OT_TWO" in by_type:
            return by_type.get("OT_ONE"), by_type.get("OT_CROSS") or by_type.get("OT_DRAW"), by_type.get("OT_TWO")
    return None, None, None


def _pick_totals(offers: list[dict], *, line: float = 2.5) -> tuple[float | None, float | None]:
    best_over = best_under = None
    best_dist = 99.0
    for offer in offers:
        if not isinstance(offer, dict) or offer.get("closed"):
            continue
        crit = str((offer.get("criterion") or {}).get("englishLabel") or "").lower()
        otype = str((offer.get("betOfferType") or {}).get("englishName") or "").lower()
        if "corner" in crit:
            continue
        if "total" not in crit and "over/under" not in otype:
            continue
        if "half" in crit or "1st" in crit or "2nd" in crit:
            continue
        over = under = None
        point = None
        for outcome in offer.get("outcomes") or []:
            if not isinstance(outcome, dict):
                continue
            if str(outcome.get("status") or "OPEN").upper() != "OPEN":
                continue
            odd = _kambi_decimal(outcome.get("odds"))
            if odd is None:
                continue
            label = str(outcome.get("label") or outcome.get("englishLabel") or "").lower()
            line_raw = outcome.get("line")
            try:
                if line_raw is not None:
                    point = float(line_raw) / (1000.0 if float(line_raw) > 50 else 1.0)
            except (TypeError, ValueError):
                pass
            if "over" in label:
                over = odd
            elif "under" in label:
                under = odd
        if over is None or under is None or point is None:
            continue
        dist = abs(point - line)
        if dist < best_dist:
            best_dist = dist
            best_over, best_under = over, under
    return best_over, best_under


def _normalize_events(payload: Any) -> list[dict]:
    rows: list[dict] = []
    if not isinstance(payload, dict):
        return rows
    for item in payload.get("events") or []:
        if not isinstance(item, dict):
            continue
        event = item.get("event") or {}
        if not isinstance(event, dict):
            continue
        state = str(event.get("state") or "").upper()
        if state in SKIP_STATES:
            continue
        home = str(event.get("homeName") or "").strip()
        away = str(event.get("awayName") or "").strip()
        if not home or not away:
            name = str(event.get("name") or event.get("englishName") or "")
            if " - " in name:
                left, right = name.split(" - ", 1)
                home = home or left.strip()
                away = away or right.strip()
        if not home or not away:
            continue
        path_label = _path_label(event.get("path"))
        name = str(event.get("englishName") or event.get("name") or f"{home} vs {away}")
        if _is_esportish(name, path_label):
            continue
        blob_ya = f"{home} {away} {path_label}".lower()
        if any(x in blob_ya for x in (" u19", " u20", " u21", "youth")):
            continue
        offers = item.get("betOffers") or []
        odd_h, odd_d, odd_a = _pick_1x2(offers)
        odd_o, odd_u = _pick_totals(offers, line=2.5)
        if not odd_h or not odd_a:
            continue
        rows.append(
            {
                "event_id": f"kambi:{event.get('id')}",
                "kambi_id": event.get("id"),
                "home": home,
                "away": away,
                "commence_time": str(event.get("start") or ""),
                "competition": path_label or str(event.get("group") or "Football"),
                "odd_home": odd_h,
                "odd_draw": odd_d,
                "odd_away": odd_a,
                "odd_over_25": odd_o,
                "odd_under_25": odd_u,
                "odds_source": "kambi_unibet",
                "state": state or None,
            }
        )
    return rows


def fetch_kambi_football_odds(
    *,
    force: bool = False,
    max_age_minutes: float = DEFAULT_MAX_AGE_MIN,
) -> dict[str, Any]:
    """Scarica listView football Unibet/Kambi (1X2 + O/U se presenti)."""
    if not _enabled():
        return {"ok": False, "error": "KAMBI_FOOTBALL_ENABLED=0", "n_events": 0, "events": []}

    max_age_hours = max(0.05, float(max_age_minutes) / 60.0)
    if not force and is_fresh(CACHE, max_age_hours=max_age_hours) and CACHE.is_file():
        try:
            cached = json.loads(CACHE.read_text(encoding="utf-8"))
            if cached.get("events") is not None:
                cached["from_cache"] = True
                return cached
        except Exception:
            pass

    payload, err, host = _fetch_payload()
    if payload is None:
        if CACHE.is_file():
            try:
                cached = json.loads(CACHE.read_text(encoding="utf-8"))
                cached["ok"] = bool(cached.get("events"))
                cached["error"] = err
                cached["from_cache"] = True
                cached["stale"] = True
                print(f"  Kambi football: cache stale ({cached.get('n_events', 0)}) — {err}", flush=True)
                return cached
            except Exception:
                pass
        return {"ok": False, "error": err, "n_events": 0, "events": [], "from_cache": False}

    events = _normalize_events(payload)
    info = {
        "ok": bool(events),
        "source": "kambi_unibet",
        "client": _client(),
        "host": host,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_matches_raw": len(payload.get("events") or []) if isinstance(payload, dict) else 0,
        "n_events": len(events),
        "events": events,
        "from_cache": False,
        "error": None if events else "nessun evento football con 1X2",
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Kambi football OK: {info['n_events']} eventi (host {host})", flush=True)
    return info


def load_kambi_football_cache() -> list[dict]:
    if not CACHE.is_file():
        return []
    try:
        return json.loads(CACHE.read_text(encoding="utf-8")).get("events") or []
    except Exception:
        return []


def _norm(s: str) -> str:
    import unicodedata

    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn").strip()


def _team_match(a: str, b: str) -> bool:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) >= 4 and len(b) >= 4:
        return a in b or b in a
    return False


def _fetch_event_betoffers(kambi_id: int | str) -> list[dict]:
    client = _client()
    params = _params()
    url = f"https://eu1.offering-api.kambicdn.com/offering/v2018/{client}/betoffer/event/{kambi_id}.json"
    host = urlparse(url).hostname or "eu1.offering-api.kambicdn.com"
    payload, err = _http_get_json(url, params=params, host=host, ips=_resolve_ips(host))
    if payload is None:
        return []
    return list(payload.get("betOffers") or [])


def _parse_ou_line_offers(
    offers: list[dict],
    *,
    criterion_match,
    key_prefix: str,
    allowed_lines: tuple[float, ...] | None = None,
) -> dict[str, float]:
    """Generic Total X Over/Under → ``{prefix}_over_{line}`` / ``_under_``."""
    out: dict[str, float] = {}
    for offer in offers:
        if not isinstance(offer, dict) or offer.get("closed"):
            continue
        crit = str((offer.get("criterion") or {}).get("englishLabel") or "").lower()
        if not criterion_match(crit):
            continue
        over = under = None
        point = None
        for outcome in offer.get("outcomes") or []:
            if not isinstance(outcome, dict):
                continue
            if str(outcome.get("status") or "OPEN").upper() != "OPEN":
                continue
            odd = _kambi_decimal(outcome.get("odds"))
            if odd is None:
                continue
            label = str(outcome.get("label") or outcome.get("englishLabel") or "").lower()
            line_raw = outcome.get("line")
            try:
                if line_raw is not None:
                    point = float(line_raw)
                    if point > 50:
                        point = point / 1000.0
            except (TypeError, ValueError):
                continue
            if "over" in label:
                over = odd
            elif "under" in label:
                under = odd
        if point is None:
            continue
        if allowed_lines is not None:
            # tollera float noise
            if not any(abs(point - float(x)) < 0.01 for x in allowed_lines):
                # accetta comunque linee .0/.5 ragionevoli fuori allowlist stretta
                if point < min(allowed_lines) - 1 or point > max(allowed_lines) + 1:
                    continue
        if over is not None:
            out[f"{key_prefix}_over_{point:g}"] = over
        if under is not None:
            out[f"{key_prefix}_under_{point:g}"] = under
    return out


def _parse_total_corners(offers: list[dict]) -> dict[str, float]:
    """Ritorna keys corners_over_9.5 / corners_under_9.5 …"""
    return _parse_ou_line_offers(
        offers,
        criterion_match=lambda c: c.strip() == "total corners" or c.strip() == "total corner kicks",
        key_prefix="corners",
        allowed_lines=CORNER_LINES,
    )


def _parse_total_shots(offers: list[dict]) -> dict[str, float]:
    """Ritorna keys shots_over_22.5 / shots_under_22.5 … (total shots, non SOT)."""

    def _is_total_shots(crit: str) -> bool:
        c = crit.strip()
        if "corner" in c:
            return False
        if "on target" in c or "on-target" in c or "sot" in c:
            return False
        if c in {"total shots", "total number of shots", "match shots"}:
            return True
        return "total" in c and "shot" in c and "team" not in c

    return _parse_ou_line_offers(
        offers,
        criterion_match=_is_total_shots,
        key_prefix="shots",
        allowed_lines=SHOT_LINES,
    )


def lookup_kambi_football(
    home: str,
    away: str,
    *,
    events: list[dict] | None = None,
    kickoff_date: str | None = None,
    include_corners: bool = False,
    include_shots: bool = False,
) -> dict | None:
    """Match fuzzy su cache listView; opz. arricchisce Total Corners/Shots via betoffers."""
    if events is None:
        events = load_kambi_football_cache()
    if not events:
        return None
    kd = None
    if kickoff_date:
        try:
            kd = date.fromisoformat(str(kickoff_date)[:10])
        except ValueError:
            pass
    candidates: list[tuple[int, dict]] = []
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
        blob = f"{ev.get('home')} {ev.get('away')} {ev.get('competition')}".lower()
        score = 0
        if any(x in blob for x in ("u19", "u20", "u21", "youth")):
            score -= 5
        if "champions" in blob or "premier" in blob or "serie" in blob:
            score += 1
        candidates.append((score, ev))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    ev = candidates[0][1]
    result = {
        "odd_home": ev.get("odd_home"),
        "odd_draw": ev.get("odd_draw"),
        "odd_away": ev.get("odd_away"),
        "odd_over_25": ev.get("odd_over_25"),
        "odd_under_25": ev.get("odd_under_25"),
        "event_id": ev.get("event_id"),
        "commence_time": ev.get("commence_time"),
        "competition": ev.get("competition"),
        "odds_source": "kambi_unibet",
        "kambi_id": ev.get("kambi_id"),
    }
    if (include_corners or include_shots) and ev.get("kambi_id") is not None:
        try:
            offers = _fetch_event_betoffers(ev["kambi_id"])
            if include_corners:
                result.update(_parse_total_corners(offers))
                result["corners_fetched"] = True
            if include_shots:
                result.update(_parse_total_shots(offers))
                result["shots_fetched"] = True
        except Exception as exc:
            result["side_markets_error"] = str(exc)
    return result
