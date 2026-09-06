"""Scarica risultati, quote e fixtures da football-data.co.uk (file CSV pubblici)."""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from modules.data_update.leagues import EXTRA_LEAGUES, SEASON_ZIPS

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
FD_MAIN = RAW / "fd" / "main"
FD_EXTRA = RAW / "fd" / "extra"
FIXTURES = RAW / "fixtures"
FD_SEED = RAW / "fd_seed.zip"
BASE = "https://www.football-data.co.uk"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

# Fail-fast globale: un 503 confermato → non martellare FD per il resto del run.
_fd_down = False


def _get(url: str, *, retries: int = 2, backoff_s: float = 1.5) -> bytes:
    """GET con 1 retry su 429/5xx. Dopo un 503, i GET successivi falliscono subito."""
    global _fd_down
    if _fd_down:
        raise HTTPError(url, 503, "Service Temporarily Unavailable (fail-fast)", hdrs=None, fp=None)

    last: BaseException | None = None
    for attempt in range(max(1, retries)):
        req = Request(
            url,
            headers={
                "User-Agent": UA,
                "Accept": "text/csv,text/plain,*/*;q=0.8",
                "Accept-Language": "en-GB,en;q=0.9",
            },
        )
        try:
            with urlopen(req, timeout=60) as resp:
                return resp.read()
        except HTTPError as exc:
            last = exc
            if exc.code == 503:
                _fd_down = True
            retriable = exc.code in {408, 429, 500, 502, 504}  # 503: no retry (sito down)
            if retriable and attempt + 1 < retries:
                sleep_s = min(backoff_s * (attempt + 1), 5.0)
                print(f"retry {attempt + 1}/{retries} HTTP {exc.code}, sleep {sleep_s:.1f}s…", flush=True)
                time.sleep(sleep_s)
                continue
            raise
        except (URLError, TimeoutError, OSError) as exc:
            last = exc
            if attempt + 1 < retries:
                sleep_s = min(backoff_s * (attempt + 1), 5.0)
                print(f"retry {attempt + 1}/{retries} rete ({exc}), sleep {sleep_s:.1f}s…", flush=True)
                time.sleep(sleep_s)
                continue
            raise
    assert last is not None
    raise last


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def has_historical_data() -> bool:
    """True se esistono CSV storici main e/o extra già sul disco."""
    has_main = FD_MAIN.is_dir() and any(FD_MAIN.glob("*/*.csv"))
    has_extra = FD_EXTRA.is_dir() and any(FD_EXTRA.glob("*.csv"))
    return has_main or has_extra


def extract_fd_seed(*, force: bool = False) -> bool:
    """Estrae data/raw/fd_seed.zip se manca lo storico (o force=True)."""
    if not force and has_historical_data():
        return True
    if not FD_SEED.is_file():
        print("seed storico assente (data/raw/fd_seed.zip)", flush=True)
        return False
    print(f"estrai seed storico {FD_SEED.name}…", flush=True)
    dest = RAW / "fd"
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(FD_SEED) as zf:
        zf.extractall(dest)
    ok = has_historical_data()
    print(f"{'ok' if ok else 'fail'} seed storico", flush=True)
    return ok


def download_season_zip(season: str) -> Path | None:
    url = f"{BASE}/mmz4281/{season}/data.zip"
    dest_dir = FD_MAIN / season
    print(f"download stagione {season}…", flush=True)
    try:
        data = _get(url)
    except Exception as exc:
        print(f"skip stagione {season}: {exc}", flush=True)
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest_dir)
    print(f"ok {season} ({len(list(dest_dir.glob('*.csv')))} csv)", flush=True)
    return dest_dir


def download_extra_leagues() -> list[Path]:
    saved = []
    for code in EXTRA_LEAGUES:
        url = f"{BASE}/new/{code}.csv"
        try:
            data = _get(url)
        except Exception as exc:
            print(f"skip extra {code}: {exc}", flush=True)
            continue
        saved.append(_write(FD_EXTRA / f"{code}.csv", data))
        print(f"ok extra {code}", flush=True)
    return saved


def download_fixtures() -> list[Path]:
    """Scarica fixtures; in caso di errore tiene la cache locale se presente."""
    files = []
    for name, url in (
        ("main.csv", f"{BASE}/fixtures.csv"),
        ("extra.csv", f"{BASE}/new_league_fixtures.csv"),
    ):
        print(f"download fixtures {name}…", flush=True)
        dest = FIXTURES / name
        try:
            data = _get(url)
        except Exception as exc:
            if dest.is_file() and dest.stat().st_size > 0:
                print(f"skip fixtures {name}: {exc} (uso cache locale)", flush=True)
                files.append(dest)
            else:
                print(f"skip fixtures {name}: {exc}", flush=True)
            continue
        files.append(_write(dest, data))
        print(f"ok fixtures {name}", flush=True)
    return files


def download_all(*, seasons: tuple[str, ...] = SEASON_ZIPS) -> dict:
    """Scarica stagioni europee, campionati extra e calendario con quote."""
    from modules.data_update.cache_policy import CONTEXT_CACHE_H, cache_fresh

    proc = ROOT / "data" / "processed"
    seasons_ok = [s for s in seasons if download_season_zip(s)]
    extra = download_extra_leagues()
    fixtures = download_fixtures()
    if not has_historical_data():
        extract_fd_seed()
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
        if cache_fresh(proc / "fbref_team_context.csv", hours=CONTEXT_CACHE_H):
            fbref = {"ok": True, "n_teams": 0, "skipped_fresh": True}
            print("FBref context: cache fresca (<72h), skip download", flush=True)
        else:
            from modules.data_update.fbref_context import download_fbref_context

            fbref = download_fbref_context()
    except Exception as exc:
        fbref = {"ok": False, "n_teams": 0, "error": str(exc)}
        print(f"skip FBref context: {exc}")
    try:
        if cache_fresh(proc / "understat_team_context.csv", hours=CONTEXT_CACHE_H):
            understat = {"ok": True, "n_teams": 0, "skipped_fresh": True}
            print("Understat context: cache fresca (<72h), skip download", flush=True)
        else:
            from modules.data_update.understat_context import download_understat_context

            understat = download_understat_context()
    except Exception as exc:
        understat = {"ok": False, "n_teams": 0, "error": str(exc)}
        print(f"skip Understat context: {exc}")
    try:
        if cache_fresh(proc / "statsbomb_team_context.csv", hours=CONTEXT_CACHE_H):
            statsbomb = {"ok": True, "n_teams": 0, "skipped_fresh": True}
            print("StatsBomb context: cache fresca (<72h), skip download", flush=True)
        else:
            from modules.data_update.statsbomb_context import download_statsbomb_context

            statsbomb = download_statsbomb_context(min_season=2015, seasons_per_comp=3)
    except Exception as exc:
        statsbomb = {"ok": False, "n_teams": 0, "error": str(exc)}
        print(f"skip StatsBomb context: {exc}")
    try:
        if cache_fresh(proc / "sofascore_team_context.csv", hours=CONTEXT_CACHE_H):
            sofascore = {"ok": True, "n_teams": 0, "skipped_fresh": True}
            print("Sofascore context: cache fresca (<72h), skip download", flush=True)
        else:
            from modules.data_update.sofascore_context import download_sofascore_context

            sofascore = download_sofascore_context()
    except Exception as exc:
        sofascore = {"ok": False, "n_teams": 0, "error": str(exc)}
        print(f"skip Sofascore context: {exc}")
    try:
        if cache_fresh(proc / "fotmob_matches.json", hours=CONTEXT_CACHE_H):
            fotmob = {"ok": True, "n_teams": 0, "n_matches": 0, "skipped_fresh": True}
            print("FotMob context: cache fresca (<72h), skip download", flush=True)
        else:
            from modules.data_update.fotmob_context import download_fotmob_context

            fotmob = download_fotmob_context(days=7)
    except Exception as exc:
        fotmob = {"ok": False, "n_teams": 0, "n_matches": 0, "error": str(exc)}
        print(f"skip FotMob context: {exc}")
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
        "fotmob_teams": fotmob.get("n_teams", 0),
        "fotmob_matches": fotmob.get("n_matches", 0),
        "n_clubelo": elo_n,
        "source": BASE,
    }
