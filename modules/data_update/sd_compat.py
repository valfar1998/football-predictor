"""Compat soccerdata: stagioni non ambigue + warning/log di libreria silenziati.

La dipendenza runtime è il tree locale ``soccerdata-master/`` (editable install).
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Iterator, Sequence

# Root del fork locale (…/football-predictor/soccerdata-master)
_LOCAL_SOCCERDATA_ROOT = Path(__file__).resolve().parents[2] / "soccerdata-master"

# Logger rumorosi durante scrape (retry 403 Sofascore, TLS client, …).
# soccerdata configura un logger di nome "root" (non il RootLogger standard).
_QUIET_LOGGERS = (
    "root",
    "soccerdata",
    "soccerdata._common",
    "soccerdata._config",
    "tls_requests",
    "tls_requests.models",
    "TLSLibrary",
)


def soccerdata_install_hint() -> str:
    """Messaggio se manca soccerdata o non punta al tree locale."""
    return (
        "Installa soccerdata locale: pip install -e ./soccerdata-master "
        "(vedi requirements.txt)."
    )


def assert_soccerdata_available():
    """Import di controllo; preferisce il package sotto soccerdata-master."""
    try:
        import soccerdata as sd
    except ImportError as exc:
        raise ImportError(soccerdata_install_hint()) from exc
    src = Path(getattr(sd, "__file__", "") or "").resolve()
    local = (_LOCAL_SOCCERDATA_ROOT / "soccerdata").resolve()
    if local.is_dir() and local not in src.parents and src != local / "__init__.py":
        # Editable non attivo: il codice PyPI globale funziona comunque.
        warnings.warn(
            f"soccerdata caricato da {src.parent}, non da {_LOCAL_SOCCERDATA_ROOT}. "
            + soccerdata_install_hint(),
            UserWarning,
            stacklevel=2,
        )
    return sd


def season_codes(years: Sequence[int | str] | None = None) -> list[str]:
    """Anni calendario → codici soccerdata tipo ``2526`` (niente warning su ``2021``)."""
    if not years:
        years = [date.today().year - 1, date.today().year]
    out: list[str] = []
    for y in years:
        s = str(y).strip()
        if not re.fullmatch(r"\d{4}", s):
            out.append(s)
            continue
        a, b = int(s[:2]), int(s[2:])
        if a in (19, 20):
            year = int(s)
            out.append(f"{year % 100:02d}{(year % 100) + 1:02d}")
            continue
        if b == (a + 1) % 100:
            out.append(s)
            continue
        year = int(s)
        out.append(f"{year % 100:02d}{(year % 100) + 1:02d}")
    return out


@contextmanager
def quiet_soccerdata() -> Iterator[None]:
    """Nasconde warning + log ERROR/INFO di soccerdata (retry 403 Sofascore, path install, …)."""
    saved_levels: dict[str, int] = {}
    saved_handlers: dict[int, tuple[logging.Handler, int]] = {}

    def mute() -> None:
        for name in _QUIET_LOGGERS:
            log = logging.getLogger(name)
            if name not in saved_levels:
                saved_levels[name] = log.level
            log.setLevel(logging.CRITICAL)
            for h in list(log.handlers):
                hid = id(h)
                if hid not in saved_handlers:
                    saved_handlers[hid] = (h, h.level)
                h.setLevel(logging.CRITICAL)

    prev_loglevel = os.environ.get("SOCCERDATA_LOGLEVEL")
    os.environ["SOCCERDATA_LOGLEVEL"] = "CRITICAL"
    mute()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module=r"soccerdata(\.|$)")
            warnings.filterwarnings("ignore", message=r".*Season id .* is ambiguous.*")
            warnings.filterwarnings(
                "ignore",
                message=r".*DataFrame concatenation with empty or all-NA entries.*",
            )
            warnings.filterwarnings("ignore", message=r".*Different columns found for.*")
            warnings.filterwarnings(
                "ignore",
                message=r".*soccerdata caricato da.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*No custom (team name|league dict).*",
            )
            # Import/dictConfig legge SOCCERDATA_LOGLEVEL; se già importato, ri-muta.
            try:
                import soccerdata  # noqa: F401
            except ImportError:
                pass
            mute()
            yield
    finally:
        if prev_loglevel is None:
            os.environ.pop("SOCCERDATA_LOGLEVEL", None)
        else:
            os.environ["SOCCERDATA_LOGLEVEL"] = prev_loglevel
        for h, level in saved_handlers.values():
            h.setLevel(level)
        for name, level in saved_levels.items():
            logging.getLogger(name).setLevel(level)
