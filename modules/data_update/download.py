"""Scarica risultati, quote e fixtures da football-data.co.uk (file CSV pubblici)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from modules.data_update.leagues import EXTRA_LEAGUES, SEASON_ZIPS

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
FD_MAIN = RAW / "fd" / "main"
FD_EXTRA = RAW / "fd" / "extra"
FIXTURES = RAW / "fixtures"
BASE = "https://www.football-data.co.uk"
UA = "Mozilla/5.0 (compatible; football-predictor/1.0; +local)"


def _get(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=60) as resp:
        return resp.read()


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def download_season_zip(season: str) -> Path | None:
    url = f"{BASE}/mmz4281/{season}/data.zip"
    dest_dir = FD_MAIN / season
    try:
        data = _get(url)
    except Exception as exc:
        print(f"skip stagione {season}: {exc}")
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest_dir)
    print(f"ok {season} ({len(list(dest_dir.glob('*.csv')))} csv)")
    return dest_dir


def download_extra_leagues() -> list[Path]:
    saved = []
    for code in EXTRA_LEAGUES:
        url = f"{BASE}/new/{code}.csv"
        try:
            data = _get(url)
        except Exception as exc:
            print(f"skip extra {code}: {exc}")
            continue
        saved.append(_write(FD_EXTRA / f"{code}.csv", data))
        print(f"ok extra {code}")
    return saved


def download_fixtures() -> list[Path]:
    files = []
    for name, url in (
        ("main.csv", f"{BASE}/fixtures.csv"),
        ("extra.csv", f"{BASE}/new_league_fixtures.csv"),
    ):
        data = _get(url)
        files.append(_write(FIXTURES / name, data))
        print(f"ok fixtures {name}")
    return files


def download_all(*, seasons: tuple[str, ...] = SEASON_ZIPS) -> dict:
    """Scarica stagioni europee, campionati extra e calendario con quote."""
    seasons_ok = [s for s in seasons if download_season_zip(s)]
    extra = download_extra_leagues()
    fixtures = download_fixtures()
    try:
        from modules.data_update.cups import download_org_cups

        cups = download_org_cups(days=14)
    except Exception as exc:
        cups = {"n_cup_files": 0, "error": str(exc)}
        print(f"skip coppe org: {exc}")
    try:
        from modules.data_update.world_fixtures import download_world_fixtures

        world = download_world_fixtures(days=14)
    except Exception as exc:
        world = {"n_world_fixtures": 0, "error": str(exc)}
        print(f"skip calendario mondiale: {exc}")
    try:
        from modules.data_update.thesportsdb import download_cup_fixtures

        tsdb = download_cup_fixtures()
    except Exception as exc:
        tsdb = {"n_cup_files": 0, "error": str(exc)}
        print(f"skip coppe TheSportsDB: {exc}")
    try:
        from modules.data_update.api_football import download_cup_fixtures as download_api_football_cups

        apif = download_api_football_cups(days=14)
    except Exception as exc:
        apif = {"n_cup_files": 0, "error": str(exc)}
        print(f"skip coppe API-Football: {exc}")
    try:
        from modules.data_update.fbref_context import download_fbref_context

        fbref = download_fbref_context()
    except Exception as exc:
        fbref = {"ok": False, "n_teams": 0, "error": str(exc)}
        print(f"skip FBref context: {exc}")
    try:
        from modules.data_update.understat_context import download_understat_context

        understat = download_understat_context()
    except Exception as exc:
        understat = {"ok": False, "n_teams": 0, "error": str(exc)}
        print(f"skip Understat context: {exc}")
    try:
        from modules.data_update.statsbomb_context import download_statsbomb_context

        statsbomb = download_statsbomb_context()
    except Exception as exc:
        statsbomb = {"ok": False, "n_teams": 0, "error": str(exc)}
        print(f"skip StatsBomb context: {exc}")
    try:
        from modules.data_update.sofascore_context import download_sofascore_context

        sofascore = download_sofascore_context()
    except Exception as exc:
        sofascore = {"ok": False, "n_teams": 0, "error": str(exc)}
        print(f"skip Sofascore context: {exc}")
    elo_n = 0
    try:
        from modules.data_update.clubelo import fetch_clubelo

        elo = fetch_clubelo()
        elo_n = 0 if elo is None or elo.empty else int(len(elo))
    except Exception as exc:
        print(f"skip ClubElo: {exc}")
    return {
        "seasons": seasons_ok,
        "extra_files": len(extra),
        "fixture_files": [str(p) for p in fixtures],
        "cup_files": int(cups.get("n_cup_files", 0)) + int(tsdb.get("n_cup_files", 0)) + int(apif.get("n_cup_files", 0)),
        "cup_tsdb_fixtures": tsdb.get("n_cup_fixtures", 0),
        "cup_api_football_fixtures": apif.get("n_cup_fixtures", 0),
        "world_fixtures": world.get("n_world_fixtures", 0),
        "fbref_teams": fbref.get("n_teams", 0),
        "understat_teams": understat.get("n_teams", 0),
        "statsbomb_teams": statsbomb.get("n_teams", 0),
        "sofascore_teams": sofascore.get("n_teams", 0),
        "n_clubelo": elo_n,
        "source": BASE,
    }
