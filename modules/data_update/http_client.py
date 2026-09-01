"""Fetch HTTP con impronta TLS browser (curl_cffi) e fallback urllib.

Usato da FotMob, Betfair e altri endpoint JSON soggetti a blocchi anti-bot.
FBref passa da soccerdata (cache propria); qui non lo patchiamo.
"""

from __future__ import annotations

import json
import random
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

USER_AGENTS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0",
)
IMPERSONATE_PROFILES = ("chrome120", "chrome124", "chrome131", "edge101", "safari17_0")
DEFAULT_UA = USER_AGENTS[0]
DEFAULT_IMPERSONATE = IMPERSONATE_PROFILES[0]


def pick_user_agent(seed: str | None = None) -> str:
    if seed:
        return USER_AGENTS[hash(seed) % len(USER_AGENTS)]
    return random.choice(USER_AGENTS)


def fetch_bytes(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 25,
    impersonate: str | None = None,
    retries: int = 3,
    backoff_s: float = 0.45,
) -> bytes:
    hdrs = {"User-Agent": pick_user_agent(url), **(headers or {})}
    profiles = list(IMPERSONATE_PROFILES)
    if impersonate:
        profiles = [impersonate] + [p for p in profiles if p != impersonate]
    last_exc: Exception | None = None
    for attempt in range(max(1, retries)):
        prof = profiles[attempt % len(profiles)]
        hdrs["User-Agent"] = pick_user_agent(f"{url}:{attempt}")
        try:
            from curl_cffi import requests as creq

            resp = creq.get(url, headers=hdrs, timeout=timeout, impersonate=prof)
            if resp.status_code in {403, 429, 503} and attempt + 1 < retries:
                time.sleep(backoff_s * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.content
        except ImportError:
            break
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < retries:
                time.sleep(backoff_s * (attempt + 1))
                continue
    try:
        req = Request(url, headers=hdrs)
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise last_exc or exc


def fetch_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 25,
    impersonate: str | None = None,
    retries: int = 3,
) -> dict[str, Any]:
    if params:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{urlencode({k: v for k, v in params.items() if v is not None})}"
    raw = fetch_bytes(
        url,
        headers=headers,
        timeout=timeout,
        impersonate=impersonate,
        retries=retries,
    )
    return json.loads(raw.decode("utf-8", "replace"))
