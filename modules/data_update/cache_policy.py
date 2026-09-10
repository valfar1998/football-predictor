"""Policy cache locale per download contesto (evita re-scrape inutili)."""

from __future__ import annotations

import time
from pathlib import Path

CONTEXT_CACHE_H = 72.0
LINEUP_CACHE_H = 6.0
ODDS_CACHE_H = 20.0


def cache_fresh(path: Path, *, hours: float = CONTEXT_CACHE_H, min_bytes: int = 80) -> bool:
    try:
        if not path.is_file() or path.stat().st_size < min_bytes:
            return False
        age_h = (time.time() - path.stat().st_mtime) / 3600.0
        return age_h < float(hours)
    except OSError:
        return False


def is_fresh(path: Path, *, max_age_hours: float = CONTEXT_CACHE_H, min_bytes: int = 80) -> bool:
    """Alias usato dai fetch odds (Kambi/Pinnacle style)."""
    return cache_fresh(path, hours=max_age_hours, min_bytes=min_bytes)
