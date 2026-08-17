from .asian_odds import fetch_asian_odds, find_asian_odds, load_asian_odds, save_asian_odds
from .download import download_all
from .parse import load_fixtures, load_historical

__all__ = [
    "download_all",
    "load_historical",
    "load_fixtures",
    "fetch_asian_odds",
    "save_asian_odds",
    "load_asian_odds",
    "find_asian_odds",
]


def build_upcoming(*args, **kwargs):
    from .upcoming import build_upcoming as _build

    return _build(*args, **kwargs)
