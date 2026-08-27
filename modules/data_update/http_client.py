"""Fetch HTTP con impronta TLS browser (curl_cffi) e fallback urllib.

Usato da FotMob e altri endpoint JSON soggetti a blocchi anti-bot.
FBref passa da soccerdata (cache propria); qui non lo patchiamo.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
DEFAULT_IMPERSONATE = "chrome120"


def fetch_bytes(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 25,
    impersonate: str = DEFAULT_IMPERSONATE,
) -> bytes:
    hdrs = {"User-Agent": DEFAULT_UA, **(headers or {})}
    try:
        from curl_cffi import requests as creq

        resp = creq.get(url, headers=hdrs, timeout=timeout, impersonate=impersonate)
        resp.raise_for_status()
        return resp.content
    except ImportError:
        pass
    except Exception:
        pass
    req = Request(url, headers=hdrs)
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 25,
    impersonate: str = DEFAULT_IMPERSONATE,
) -> dict[str, Any]:
    if params:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{urlencode({k: v for k, v in params.items() if v is not None})}"
    raw = fetch_bytes(url, headers=headers, timeout=timeout, impersonate=impersonate)
    return json.loads(raw.decode("utf-8", "replace"))
