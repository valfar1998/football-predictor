"""Palinsesto football Unibet via Kambi Guest API (1X2, O/U gol, corner/tiri su evento).

Stesso schema del tennis-predictor (DoH + curl_cffi), sport=football.
ListView generico + path Big 5 (PL/LaLiga/SerieA/Bundesliga/Ligue1).
Corner/tiri: non nel listView — prefetch betoffers su Big 5 e/o on-demand
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
# en_GB: nomi squadre allineati a FD/Sofascore (it_IT → "Bayern Monaco" ecc.)
DEFAULT_LANG = "en_GB"
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
# listView/football.json è una finestra corta (spesso minor/soon) e omette la maggior
# parte dei match Big 5. I path per lega coprono l'intero round corrente.
BIG5_LISTVIEW_PATHS = (
    "football/england/premier_league/all/matches.json",
    "football/spain/la_liga/all/matches.json",
    "football/italy/serie_a/all/matches.json",
    "football/germany/bundesliga/all/matches.json",
    "football/france/ligue_1/all/matches.json",
)
BETOFFER_HOSTS = (
    "eu1.offering-api.kambicdn.com",
    "eu-offering.kambicdn.org",
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


def _fetch_listview_url(url_template: str, *, doh: str) -> tuple[Any, str | None, str | None]:
    client = _client()
    params = _params()
    url = url_template.format(client=client)
    host = urlparse(url).hostname or doh
    ips = _resolve_ips(doh)
    payload, err = _http_get_json(url, params=params, host=host, ips=ips)
    return payload, err, host


def _fetch_payload() -> tuple[Any, str | None, str | None]:
    errors: list[str] = []
    used_host: str | None = None
    for endpoint in ENDPOINTS:
        used_host = endpoint["doh"]
        payload, err, host = _fetch_listview_url(endpoint["url"], doh=endpoint["doh"])
        used_host = host
        if payload is not None:
            return payload, None, host
        if err:
            errors.append(f"{host}: {err}")
        time.sleep(0.3)
    return None, "; ".join(errors) or "Kambi non raggiungibile", used_host


def _is_big5_competition(competition: str) -> bool:
    """Solo top-5 europee (no 'Premier League' di Rwanda/Iraq ecc.)."""
    blob = (competition or "").lower()
    if any(x in blob for x in ("serie b", "ligue 2", "2. bundes", "segunda", "championship")):
        return False
    rules = (
        (("england", "inghilterra"), ("premier league",)),
        (("italy", "italia"), ("serie a",)),
        (("spain", "spagna"), ("la liga", "laliga")),
        (("germany", "germania"), ("bundesliga",)),
        (("france", "francia"), ("ligue 1",)),
    )
    for countries, leagues in rules:
        if any(c in blob for c in countries) and any(lg in blob for lg in leagues):
            return True
    return False


def _merge_events_by_id(*batches: list[dict]) -> list[dict]:
    by_id: dict[Any, dict] = {}
    orphans: list[dict] = []
    for batch in batches:
        for ev in batch:
            kid = ev.get("kambi_id")
            if kid is None:
                orphans.append(ev)
                continue
            prev = by_id.get(kid)
            if prev is None:
                by_id[kid] = dict(ev)
                continue
            merged = dict(prev)
            merged.update({k: v for k, v in ev.items() if v is not None})
            # conserva corner/shot già prefetchati se il nuovo batch non li ha
            for k, v in prev.items():
                if (str(k).startswith("corners_") or str(k).startswith("shots_")) and k not in merged:
                    merged[k] = v
            by_id[kid] = merged
    return list(by_id.values()) + orphans


def _fetch_big5_payloads(primary_host: str | None) -> tuple[list[dict], list[str]]:
    """Scarica listView per ogni Big 5; ritorna eventi normalizzati + errori soft."""
    client = _client()
    errors: list[str] = []
    batches: list[dict] = []
    hosts = []
    if primary_host:
        hosts.append(primary_host)
    for h in BETOFFER_HOSTS:
        if h not in hosts:
            hosts.append(h)
    for path in BIG5_LISTVIEW_PATHS:
        got = False
        last_err: str | None = None
        for host in hosts:
            url = f"https://{host}/offering/v2018/{client}/listView/{path}"
            payload, err = _http_get_json(url, params=_params(), host=host, ips=_resolve_ips(host))
            if payload is None:
                last_err = err
                continue
            rows = _normalize_events(payload)
            for r in rows:
                r["listview_path"] = path
            batches.extend(rows)
            got = True
            break
        if not got:
            errors.append(f"{path}: {last_err or 'fail'}")
        time.sleep(0.15)
    return batches, errors


def _prefetch_side_markets(events: list[dict], *, corners: bool = True, shots: bool = False) -> int:
    """Arricchisce in-place eventi Big 5 con betoffers corner (e opz. tiri)."""
    if not corners and not shots:
        return 0
    try:
        max_n = int((os.environ.get("KAMBI_PREFETCH_MAX") or "90").strip() or "90")
    except ValueError:
        max_n = 90
    n_done = 0
    for ev in events:
        if n_done >= max_n:
            break
        if not _is_big5_competition(str(ev.get("competition") or "")):
            continue
        kid = ev.get("kambi_id")
        if kid is None:
            continue
        already = any(str(k).startswith("corners_") for k in ev) if corners else True
        if already and not shots:
            continue
        if already and shots and any(str(k).startswith("shots_") for k in ev):
            continue
        try:
            offers = _fetch_event_betoffers(kid)
            if not offers:
                continue
            got = False
            if corners:
                corn = _parse_total_corners(offers)
                if corn:
                    ev.update(corn)
                    ev["corners_fetched"] = True
                    got = True
            if shots:
                sh = _parse_total_shots(offers)
                if sh:
                    ev.update(sh)
                    ev["shots_fetched"] = True
                    got = True
            if got:
                n_done += 1
        except Exception:
            continue
        time.sleep(0.12)
    return n_done


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
    """Scarica listView football Unibet/Kambi (1X2 + O/U) + path Big 5 + opz. corners."""
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

    base_events = _normalize_events(payload)
    big5_events, big5_errs = _fetch_big5_payloads(host)
    events = _merge_events_by_id(base_events, big5_events)
    n_big5 = sum(1 for e in events if _is_big5_competition(str(e.get("competition") or "")))

    prefetch_on = (os.environ.get("KAMBI_PREFETCH_CORNERS") or "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    n_pref = 0
    if prefetch_on and n_big5:
        n_pref = _prefetch_side_markets(events, corners=True, shots=False)

    info = {
        "ok": bool(events),
        "source": "kambi_unibet",
        "client": _client(),
        "host": host,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_matches_raw": len(payload.get("events") or []) if isinstance(payload, dict) else 0,
        "n_events": len(events),
        "n_big5": n_big5,
        "n_big5_listview": len(big5_events),
        "n_corners_prefetched": n_pref,
        "big5_errors": big5_errs or None,
        "events": events,
        "from_cache": False,
        "error": None if events else "nessun evento football con 1X2",
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"  Kambi football OK: {info['n_events']} eventi "
        f"(Big5 listView {len(big5_events)}, in cache {n_big5}, "
        f"corners prefetch {n_pref}) host {host}",
        flush=True,
    )
    if big5_errs:
        print(f"  Kambi Big5 soft errors: {'; '.join(big5_errs[:3])}", flush=True)
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


# football-data.co.uk / FD.org abbreviazioni → forma allineata a Kambi en_GB
_TEAM_ALIASES: dict[str, str] = {
    "hsv": "hamburger sv",
    "hamburger": "hamburger sv",
    "fc koln": "1. fc koln",
    "koln": "1. fc koln",
    "cologne": "1. fc koln",
    "m'gladbach": "borussia monchengladbach",
    "mgladbach": "borussia monchengladbach",
    "gladbach": "borussia monchengladbach",
    "ein frankfurt": "eintracht frankfurt",
    "ath bilbao": "athletic bilbao",
    "athletic club": "athletic bilbao",
    "nott'm forest": "nottingham forest",
    "nottm forest": "nottingham forest",
    "nott forest": "nottingham forest",
    "man city": "manchester city",
    "man united": "manchester united",
    "man utd": "manchester united",
    "manchester utd": "manchester united",
    "ath madrid": "atletico madrid",
    "atleti": "atletico madrid",
    "paris sg": "psg",
    "paris saint germain": "psg",
    "paris saint-germain": "psg",
    "stade rennais": "rennes",
    "rennais": "rennes",
    "olympique lyonnais": "lyon",
    "olympique marseille": "marseille",
    "bayern munchen": "bayern munich",
    "bayern monaco": "bayern munich",
    "inter milan": "inter",
    "internazionale": "inter",
    "fc barcelona": "barcelona",
    "barca": "barcelona",
    "spurs": "tottenham",
    "wolves": "wolverhampton",
    "wolverhampton wanderers": "wolverhampton",
}


def _canon_team(s: str) -> str:
    n = _norm(s)
    n = n.replace("'", "'").replace("`", "'")
    if n in _TEAM_ALIASES:
        return _TEAM_ALIASES[n]
    # prova senza punti
    compact = n.replace(".", "").replace("  ", " ").strip()
    return _TEAM_ALIASES.get(compact, n)


def _team_tokens(s: str) -> set[str]:
    stop = {"fc", "cf", "afc", "sc", "ac", "as", "ssc", "calcio", "club", "de", "the", "1", "sv", "ud", "rcd"}
    return {t for t in _canon_team(s).replace("-", " ").replace(".", " ").split() if len(t) >= 3 and t not in stop}


def _team_match(a: str, b: str) -> bool:
    ca, cb = _canon_team(a), _canon_team(b)
    if not ca or not cb:
        return False
    if ca == cb:
        return True
    if len(ca) >= 4 and len(cb) >= 4 and (ca in cb or cb in ca):
        return True
    ta, tb = _team_tokens(a), _team_tokens(b)
    if not ta or not tb:
        return False
    inter = ta & tb
    if len(inter) >= 2:
        return True
    if len(inter) == 1 and (len(ta) == 1 or len(tb) == 1):
        tok = next(iter(inter))
        # evita match deboli su token corti generici
        if len(tok) >= 5:
            return True
    return False


def _fetch_event_betoffers(kambi_id: int | str) -> list[dict]:
    client = _client()
    params = _params()
    last_err: str | None = None
    for host in BETOFFER_HOSTS:
        url = f"https://{host}/offering/v2018/{client}/betoffer/event/{kambi_id}.json"
        payload, err = _http_get_json(url, params=params, host=host, ips=_resolve_ips(host))
        if payload is not None:
            return list(payload.get("betOffers") or [])
        last_err = err
        time.sleep(0.1)
    if last_err:
        return []
    return []


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


def _is_total_corners_criterion(crit: str) -> bool:
    c = (crit or "").strip().lower()
    if c in {"total corners", "total corner kicks", "match corners", "corners"}:
        return True
    if "corner" not in c:
        return False
    if any(x in c for x in ("1st", "2nd", "first half", "second half", "team", "home", "away", "most")):
        return False
    return "total" in c


def _parse_total_corners(offers: list[dict]) -> dict[str, float]:
    """Ritorna keys corners_over_9.5 / corners_under_9.5 …"""
    return _parse_ou_line_offers(
        offers,
        criterion_match=_is_total_corners_criterion,
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
        if _is_big5_competition(str(ev.get("competition") or "")):
            score += 3
        elif "champions" in blob or "premier" in blob or "serie" in blob:
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
            # riusa corner/shot già prefetchati nel listView cache
            reused = False
            if include_corners:
                cached_c = {
                    k: v
                    for k, v in ev.items()
                    if str(k).startswith("corners_") and isinstance(v, (int, float))
                }
                if cached_c:
                    result.update(cached_c)
                    result["corners_fetched"] = True
                    reused = True
            if include_shots:
                cached_s = {
                    k: v
                    for k, v in ev.items()
                    if str(k).startswith("shots_") and isinstance(v, (int, float))
                }
                if cached_s:
                    result.update(cached_s)
                    result["shots_fetched"] = True
                    reused = True
            need_c = include_corners and not any(str(k).startswith("corners_") for k in result)
            need_s = include_shots and not any(str(k).startswith("shots_") for k in result)
            if need_c or need_s:
                offers = _fetch_event_betoffers(ev["kambi_id"])
                if need_c:
                    result.update(_parse_total_corners(offers))
                    result["corners_fetched"] = True
                if need_s:
                    result.update(_parse_total_shots(offers))
                    result["shots_fetched"] = True
            elif reused:
                result["side_markets_from_cache"] = True
        except Exception as exc:
            result["side_markets_error"] = str(exc)
    return result
