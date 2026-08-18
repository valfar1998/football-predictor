"""Allinea nomi API-Football / openfootball / football-data.org al dizionario football-data.co.uk."""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Iterable

# Chiave già passata da _norm_key → spelling football-data.co.uk
SOURCE_TEAM_ALIASES: dict[str, str] = {
    # Inghilterra
    "arsenal": "Arsenal",
    "manchester united": "Man United",
    "manchester utd": "Man United",
    "man utd": "Man United",
    "man united": "Man United",
    "manchester city": "Man City",
    "man city": "Man City",
    "tottenham": "Tottenham",
    "tottenham hotspur": "Tottenham",
    "spurs": "Tottenham",
    "wolverhampton": "Wolves",
    "wolverhampton wanderers": "Wolves",
    "wolves": "Wolves",
    "nottingham forest": "Nott'm Forest",
    "nottm forest": "Nott'm Forest",
    "nott m forest": "Nott'm Forest",
    "newcastle": "Newcastle",
    "newcastle united": "Newcastle",
    "west ham": "West Ham",
    "west ham united": "West Ham",
    "west brom": "West Brom",
    "west bromwich": "West Brom",
    "west bromwich albion": "West Brom",
    "brighton": "Brighton",
    "brighton hove": "Brighton",
    "brighton hove albion": "Brighton",
    "bournemouth": "Bournemouth",
    "crystal palace": "Crystal Palace",
    "leicester": "Leicester",
    "leicester city": "Leicester",
    "leeds": "Leeds",
    "leeds united": "Leeds",
    "ipswich": "Ipswich",
    "ipswich town": "Ipswich",
    "sunderland": "Sunderland",
    "southampton": "Southampton",
    "fulham": "Fulham",
    "everton": "Everton",
    "chelsea": "Chelsea",
    "liverpool": "Liverpool",
    "aston villa": "Aston Villa",
    "brentford": "Brentford",
    "burnley": "Burnley",
    "luton": "Luton",
    "luton town": "Luton",
    "coventry": "Coventry",
    "coventry city": "Coventry",
    "sheffield united": "Sheffield United",
    "sheff utd": "Sheffield United",
    "sheffield wednesday": "Sheffield Weds",
    "sheff wed": "Sheffield Weds",
    "qpr": "QPR",
    "queens park rangers": "QPR",
    "hull": "Hull",
    "hull city": "Hull",
    "middlesbrough": "Middlesbrough",
    "norwich": "Norwich",
    "norwich city": "Norwich",
    "watford": "Watford",
    "cardiff": "Cardiff",
    "swansea": "Swansea",
    "stoke": "Stoke",
    "stoke city": "Stoke",
    "derby": "Derby",
    "derby county": "Derby",
    "birmingham": "Birmingham",
    "preston": "Preston",
    "bristol city": "Bristol City",
    "millwall": "Millwall",
    "blackburn": "Blackburn",
    "oxford": "Oxford",
    "oxford united": "Oxford",
    "portsmouth": "Portsmouth",
    "plymouth": "Plymouth",
    "wrexham": "Wrexham",
    # Italia
    "inter": "Inter",
    "inter milan": "Inter",
    "internazionale": "Inter",
    "internazionale milano": "Inter",
    "milan": "Milan",
    "juventus": "Juventus",
    "roma": "Roma",
    "as roma": "Roma",
    "napoli": "Napoli",
    "lazio": "Lazio",
    "hellas verona": "Hellas Verona",
    "verona": "Hellas Verona",
    "parma": "Parma",
    "como": "Como",
    "venezia": "Venezia",
    "cremonese": "Cremonese",
    "pisa": "Pisa",
    # Spagna
    "atletico madrid": "Ath Madrid",
    "atl madrid": "Ath Madrid",
    "atleti": "Ath Madrid",
    "athletic": "Ath Bilbao",
    "athletic bilbao": "Ath Bilbao",
    "athletic club": "Ath Bilbao",
    "real sociedad": "Sociedad",
    "sociedad": "Sociedad",
    "real betis": "Betis",
    "betis": "Betis",
    "espanol": "Espanol",
    "espanyol": "Espanol",
    "rayo vallecano": "Vallecano",
    "vallecano": "Vallecano",
    "celta": "Celta",
    "celta vigo": "Celta",
    "alaves": "Alaves",
    "deportivo alaves": "Alaves",
    "deportivo la coruna": "La Coruna",
    "dep a coruna": "La Coruna",
    "la coruna": "La Coruna",
    "elche": "Elche",
    "levante": "Levante",
    "girona": "Girona",
    "getafe": "Getafe",
    "osasuna": "Osasuna",
    "mallorca": "Mallorca",
    "las palmas": "Las Palmas",
    "leganes": "Leganes",
    "valladolid": "Valladolid",
    "cadiz": "Cadiz",
    "oviedo": "Oviedo",
    "real oviedo": "Oviedo",
    # Germania
    "bayern": "Bayern Munich",
    "bayern munich": "Bayern Munich",
    "bayern munchen": "Bayern Munich",
    "dortmund": "Dortmund",
    "borussia dortmund": "Dortmund",
    "bayer leverkusen": "Leverkusen",
    "leverkusen": "Leverkusen",
    "rb leipzig": "RB Leipzig",
    "rasenballsport leipzig": "RB Leipzig",
    "leipzig": "RB Leipzig",
    "eintracht frankfurt": "Ein Frankfurt",
    "frankfurt": "Ein Frankfurt",
    "monchengladbach": "M'gladbach",
    "borussia monchengladbach": "M'gladbach",
    "borussia m gladbach": "M'gladbach",
    "gladbach": "M'gladbach",
    "union berlin": "Union Berlin",
    "werder bremen": "Werder Bremen",
    "bremen": "Werder Bremen",
    "mainz": "Mainz",
    "fsv mainz": "Mainz",
    "hoffenheim": "Hoffenheim",
    "tsg hoffenheim": "Hoffenheim",
    "stuttgart": "Stuttgart",
    "vfb stuttgart": "Stuttgart",
    "heidenheim": "Heidenheim",
    "st pauli": "St Pauli",
    "koln": "FC Koln",
    "cologne": "FC Koln",
    "hamburg": "Hamburg",
    "hamburger": "Hamburg",
    "hertha": "Hertha",
    "schalke": "Schalke 04",
    "bochum": "Bochum",
    "augsburg": "Augsburg",
    "freiburg": "Freiburg",
    "wolfsburg": "Wolfsburg",
    # Francia
    "paris saint germain": "Paris SG",
    "paris sg": "Paris SG",
    "psg": "Paris SG",
    "psg": "Paris SG",
    "marseille": "Marseille",
    "olympique marseille": "Marseille",
    "lyon": "Lyon",
    "olympique lyon": "Lyon",
    "olympique lyonnais": "Lyon",
    "lille": "Lille",
    "losc lille": "Lille",
    "monaco": "Monaco",
    "nice": "Nice",
    "ogc nice": "Nice",
    "rennes": "Rennes",
    "lens": "Lens",
    "strasbourg": "Strasbourg",
    "nantes": "Nantes",
    "toulouse": "Toulouse",
    "reims": "Reims",
    "brest": "Brest",
    "le havre": "Le Havre",
    "auxerre": "Auxerre",
    "angers": "Angers",
    "metz": "Metz",
    "lorient": "Lorient",
    "montpellier": "Montpellier",
    # Portogallo / Benelux / Turchia / Grecia
    "porto": "Porto",
    "fc porto": "Porto",
    "benfica": "Benfica",
    "sporting": "Sp Lisbon",
    "sporting lisbon": "Sp Lisbon",
    "sporting lisboa": "Sp Lisbon",
    "sporting cp": "Sp Lisbon",
    "sporting portugal": "Sp Lisbon",
    "sporting clube de portugal": "Sp Lisbon",
    "psv": "PSV Eindhoven",
    "psv eindhoven": "PSV Eindhoven",
    "ajax": "Ajax",
    "feyenoord": "Feyenoord",
    "anderlecht": "Anderlecht",
    "club brugge": "Club Brugge",
    "fenerbahce": "Fenerbahce",
    "galatasaray": "Galatasaray",
    "besiktas": "Besiktas",
    "olympiacos": "Olympiakos",
    "olympiakos": "Olympiakos",
    "salzburg": "Salzburg",
    "red bull salzburg": "Salzburg",
    "rb salzburg": "Salzburg",
    # Extra
    "flamengo": "Flamengo RJ",
    "cr flamengo": "Flamengo RJ",
}

_NOISE = re.compile(
    r"\b(fc|cf|ac|sc|afc|cfc|bk|sk|fk|cd|de|the|club|calcio|ssc|us|sv|as|ss|rc|"
    r"rcd|ud|sd|ogc|losc|vfb|fsv|tsg|rb|bvb|ogc|1|04|05|1899|1909|1846|"
    r"hotspur|wanderers|albion|olympique|olympic|balompie)\b",
    re.I,
)
_SKIP_TOKEN = {"united", "city", "real", "sport", "sporting", "athletic", "inter", "milan"}

_KNOWN_IDX: dict[str, str] | None = None


def _norm_key(name: str) -> str:
    base = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9 ]", " ", base.lower())
    s = _NOISE.sub(" ", s)
    s = re.sub(r"\b\d+\b", " ", s)
    return " ".join(s.split())


def _alias_target(raw: str) -> str | None:
    from modules.dataset_loader.loader import TEAM_ALIASES

    key = _norm_key(raw)
    if key in SOURCE_TEAM_ALIASES:
        return SOURCE_TEAM_ALIASES[key]
    low = " ".join(str(raw).lower().split())
    if low in TEAM_ALIASES:
        return TEAM_ALIASES[low]
    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]
    return None


def known_team_index(names: Iterable[str]) -> dict[str, str]:
    idx: dict[str, str] = {}
    canonical = {str(n).strip() for n in names if str(n).strip()}
    for name in canonical:
        idx[_norm_key(name)] = name
    for key, target in SOURCE_TEAM_ALIASES.items():
        if target in canonical:
            idx[_norm_key(key)] = target
            idx[key] = target
    from modules.dataset_loader.loader import TEAM_ALIASES

    for key, target in TEAM_ALIASES.items():
        if target in canonical:
            idx[_norm_key(key)] = target
    return idx


def _default_known_index() -> dict[str, str]:
    global _KNOWN_IDX
    if _KNOWN_IDX is not None:
        return _KNOWN_IDX
    try:
        from modules.predictor import list_known_teams

        _KNOWN_IDX = known_team_index(list_known_teams())
    except Exception:
        _KNOWN_IDX = {}
    return _KNOWN_IDX


def _token_hit(key: str, index: dict[str, str]) -> str | None:
    q = set(key.split())
    if not q:
        return None
    hits: dict[str, int] = {}
    for k, name in index.items():
        toks = set(k.split())
        if not toks:
            continue
        if toks <= q or q <= toks:
            overlap = len(toks & q)
            if overlap == 0:
                continue
            if len(q) == 1 and (next(iter(q)) in _SKIP_TOKEN or len(next(iter(q))) < 6):
                continue
            hits[name] = max(hits.get(name, 0), overlap)
    if len(hits) == 1:
        return next(iter(hits))
    return None


def resolve_known_team(name: str, known: Iterable[str] | dict[str, str] | None = None) -> str | None:
    """Allinea il nome API/openfootball/org allo spelling football-data.co.uk."""
    from modules.dataset_loader.loader import normalize_team

    raw = str(name or "").strip()
    if not raw:
        return None
    aliased = _alias_target(raw)
    fallback = aliased or normalize_team(raw)

    if known is None:
        index = _default_known_index()
        if not index:
            return fallback
        canonical = set(index.values())
    elif isinstance(known, dict):
        index = known
        canonical = set(known.values())
    else:
        canonical = set(known)
        index = known_team_index(canonical)

    if aliased and aliased in canonical:
        return aliased
    if fallback in canonical:
        return fallback
    key = _norm_key(raw)
    if key in index:
        return index[key]
    key2 = _norm_key(fallback)
    if key2 in index:
        return index[key2]
    tok = _token_hit(key2 or key, index)
    if tok:
        return tok
    probe = key2 or key
    if len(probe) >= 6:
        hit = difflib.get_close_matches(probe, list(index.keys()), n=2, cutoff=0.88)
        names = {index[h] for h in hit if h in index}
        if len(names) == 1:
            return next(iter(names))
    return None if known is not None else fallback


def reset_known_index_cache() -> None:
    global _KNOWN_IDX
    _KNOWN_IDX = None
