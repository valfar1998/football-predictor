"""Campionati coperti da football-data.co.uk (Europa + extra mondiali)."""

from __future__ import annotations

# 22 divisioni europee nei file mmz4281/{stagione}/{codice}.csv
DIV_META: dict[str, tuple[str, str]] = {
    "E0": ("Inghilterra", "Premier League"),
    "E1": ("Inghilterra", "Championship"),
    "E2": ("Inghilterra", "League One"),
    "E3": ("Inghilterra", "League Two"),
    "EC": ("Inghilterra", "National League"),
    "SC0": ("Scozia", "Premiership"),
    "SC1": ("Scozia", "Championship"),
    "SC2": ("Scozia", "League One"),
    "SC3": ("Scozia", "League Two"),
    "D1": ("Germania", "Bundesliga"),
    "D2": ("Germania", "2. Bundesliga"),
    "I1": ("Italia", "Serie A"),
    "I2": ("Italia", "Serie B"),
    "SP1": ("Spagna", "La Liga"),
    "SP2": ("Spagna", "Segunda Division"),
    "F1": ("Francia", "Ligue 1"),
    "F2": ("Francia", "Ligue 2"),
    "N1": ("Olanda", "Eredivisie"),
    "B1": ("Belgio", "Pro League"),
    "P1": ("Portogallo", "Primeira Liga"),
    "T1": ("Turchia", "Super Lig"),
    "G1": ("Grecia", "Super League"),
}

# File extra: https://www.football-data.co.uk/new/{CODE}.csv
EXTRA_LEAGUES: dict[str, str] = {
    "ARG": "Argentina",
    "AUT": "Austria",
    "BRA": "Brasile",
    "CHN": "Cina",
    "DNK": "Danimarca",
    "FIN": "Finlandia",
    "IRL": "Irlanda",
    "JPN": "Giappone",
    "MEX": "Messico",
    "NOR": "Norvegia",
    "POL": "Polonia",
    "ROU": "Romania",
    "RUS": "Russia",
    "SWE": "Svezia",
    "SWZ": "Svizzera",
    "USA": "USA",
}

SEASON_ZIPS = ("2223", "2324", "2425", "2526", "2627")
