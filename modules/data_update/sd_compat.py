"""Compat soccerdata: stagioni non ambigue + warning di libreria silenziati."""

from __future__ import annotations

import re
import warnings
from contextlib import contextmanager
from datetime import date
from typing import Iterator, Sequence


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
