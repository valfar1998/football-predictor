"""Compat soccerdata: stagioni non ambigue + warning di libreria silenziati.

La dipendenza runtime è il tree locale ``soccerdata-master/`` (editable install).
"""

from __future__ import annotations

import re
import warnings
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Iterator, Sequence

# Root del fork locale (…/football-predictor/soccerdata-master)
_LOCAL_SOCCERDATA_ROOT = Path(__file__).resolve().parents[2] / "soccerdata-master"


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
    """Nasconde UserWarning/FutureWarning emessi da soccerdata (pandas concat, stagioni)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module=r"soccerdata(\.|$)")
        warnings.filterwarnings("ignore", message=r".*Season id .* is ambiguous.*")
        warnings.filterwarnings(
            "ignore",
            message=r".*DataFrame concatenation with empty or all-NA entries.*",
        )
        warnings.filterwarnings("ignore", message=r".*Different columns found for.*")
        yield
