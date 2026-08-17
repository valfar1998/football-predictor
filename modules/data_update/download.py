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
    return {
        "seasons": seasons_ok,
        "extra_files": len(extra),
        "fixture_files": [str(p) for p in fixtures],
        "source": BASE,
    }
