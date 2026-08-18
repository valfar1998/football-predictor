"""Parser dei CSV football-data.co.uk: risultati storici + fixtures con quote."""

from __future__ import annotations

from pathlib import Path
import io

import numpy as np
import pandas as pd

from modules.data_update.leagues import DIV_META, EXTRA_LEAGUES

COUNTRY_IT = {
    "Argentina": "Argentina",
    "Austria": "Austria",
    "Brazil": "Brasile",
    "China": "Cina",
    "Denmark": "Danimarca",
    "Finland": "Finlandia",
    "Ireland": "Irlanda",
    "Japan": "Giappone",
    "Mexico": "Messico",
    "Norway": "Norvegia",
    "Poland": "Polonia",
    "Romania": "Romania",
    "Russia": "Russia",
    "Sweden": "Svezia",
    "Switzerland": "Svizzera",
    "USA": "USA",
}


def _country_it(name: object) -> str:
    raw = str(name).strip()
    return COUNTRY_IT.get(raw, EXTRA_LEAGUES.get(raw, raw))

ROOT = Path(__file__).resolve().parents[2]
FD_MAIN = ROOT / "data" / "raw" / "fd" / "main"
FD_EXTRA = ROOT / "data" / "raw" / "fd" / "extra"
FIXTURES_DIR = ROOT / "data" / "raw" / "fixtures"

ODDS_HOME_OPEN = ("AvgH", "B365H", "PSH", "MaxH")
ODDS_DRAW_OPEN = ("AvgD", "B365D", "PSD", "MaxD")
ODDS_AWAY_OPEN = ("AvgA", "B365A", "PSA", "MaxA")
ODDS_HOME_CLOSE = ("AvgCH", "B365CH", "PSCH", "MaxCH")
ODDS_DRAW_CLOSE = ("AvgCD", "B365CD", "PSCD", "MaxCD")
ODDS_AWAY_CLOSE = ("AvgCA", "B365CA", "PSCA", "MaxCA")
ODDS_HOME_SHARP = ("PSH",)
ODDS_DRAW_SHARP = ("PSD",)
ODDS_AWAY_SHARP = ("PSA",)
ODDS_O25_OPEN = ("Avg>2.5", "B365>2.5", "Max>2.5")
ODDS_U25_OPEN = ("Avg<2.5", "B365<2.5", "Max<2.5")
ODDS_O25_CLOSE = ("AvgC>2.5", "B365C>2.5", "MaxC>2.5")
ODDS_U25_CLOSE = ("AvgC<2.5", "B365C<2.5", "MaxC<2.5")
# compat: quota "presa" = apertura (venerdì), non la close
ODDS_HOME = ODDS_HOME_OPEN + ODDS_HOME_CLOSE
ODDS_DRAW = ODDS_DRAW_OPEN + ODDS_DRAW_CLOSE
ODDS_AWAY = ODDS_AWAY_OPEN + ODDS_AWAY_CLOSE
ODDS_O25 = ODDS_O25_OPEN + ODDS_O25_CLOSE
ODDS_U25 = ODDS_U25_OPEN + ODDS_U25_CLOSE


def _read_fd_csv(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip")
    df.columns = [str(c).replace("\ufeff", "").replace("ï»¿", "").strip() for c in df.columns]
    return df


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _first_odd(df: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for name in names:
        if name not in df.columns:
            continue
        cand = _num(df[name])
        take = out.isna() & cand.notna() & (cand > 1.0)
        out = out.mask(take, cand)
    return out


def _odds_open_close_sharp(df: pd.DataFrame) -> dict[str, pd.Series]:
    home_open = _first_odd(df, ODDS_HOME_OPEN)
    draw_open = _first_odd(df, ODDS_DRAW_OPEN)
    away_open = _first_odd(df, ODDS_AWAY_OPEN)
    home_close = _first_odd(df, ODDS_HOME_CLOSE).fillna(home_open)
    draw_close = _first_odd(df, ODDS_DRAW_CLOSE).fillna(draw_open)
    away_close = _first_odd(df, ODDS_AWAY_CLOSE).fillna(away_open)
    o25_open = _first_odd(df, ODDS_O25_OPEN)
    u25_open = _first_odd(df, ODDS_U25_OPEN)
    o25_close = _first_odd(df, ODDS_O25_CLOSE).fillna(o25_open)
    u25_close = _first_odd(df, ODDS_U25_CLOSE).fillna(u25_open)
    return {
        "odd_home": home_open.fillna(_first_odd(df, ODDS_HOME_CLOSE)),
        "odd_draw": draw_open.fillna(_first_odd(df, ODDS_DRAW_CLOSE)),
        "odd_away": away_open.fillna(_first_odd(df, ODDS_AWAY_CLOSE)),
        "odd_over_25": o25_open.fillna(o25_close),
        "odd_under_25": u25_open.fillna(u25_close),
        "odd_home_close": home_close,
        "odd_draw_close": draw_close,
        "odd_away_close": away_close,
        "odd_over_25_close": o25_close,
        "odd_under_25_close": u25_close,
        "odd_home_sharp": _first_odd(df, ODDS_HOME_SHARP),
        "odd_draw_sharp": _first_odd(df, ODDS_DRAW_SHARP),
        "odd_away_sharp": _first_odd(df, ODDS_AWAY_SHARP),
    }


def _meta_from_div(div: object) -> tuple[str, str]:
    if div is None or (not isinstance(div, str) and pd.isna(div)):
        return ("Altro", "unknown")
    key = str(div).strip()
    return DIV_META.get(key, ("Altro", key or "unknown"))


def parse_main_results(path: Path) -> pd.DataFrame:
    df = _read_fd_csv(path)
    if "HomeTeam" not in df.columns or "FTHG" not in df.columns:
        return pd.DataFrame()
    country, league = _meta_from_div(df["Div"].iloc[0] if "Div" in df.columns and len(df) else "")
    if "Div" in df.columns:
        mapped = df["Div"].map(lambda d: _meta_from_div(d))
        country_s = mapped.map(lambda t: t[0])
        league_s = mapped.map(lambda t: t[1])
    else:
        country_s = country
        league_s = league
    odds = _odds_open_close_sharp(df)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
            "home_team": df["HomeTeam"].astype(str).str.strip(),
            "away_team": df["AwayTeam"].astype(str).str.strip(),
            "home_goals": _num(df["FTHG"]),
            "away_goals": _num(df["FTAG"]),
            "country": country_s,
            "league": league_s,
            "div": df["Div"] if "Div" in df.columns else league,
            **odds,
            "source": f"fd:{path.stem}",
        }
    )
    return out.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])


def parse_extra_results(path: Path) -> pd.DataFrame:
    df = _read_fd_csv(path)
    home_col = "Home" if "Home" in df.columns else "HomeTeam"
    away_col = "Away" if "Away" in df.columns else "AwayTeam"
    hg = "HG" if "HG" in df.columns else "FTHG"
    ag = "AG" if "AG" in df.columns else "FTAG"
    if home_col not in df.columns or hg not in df.columns:
        return pd.DataFrame()
    country = EXTRA_LEAGUES.get(path.stem, path.stem)
    if "Country" in df.columns:
        country_s = df["Country"].astype(str).str.strip().map(_country_it)
    else:
        country_s = country
    league_s = df["League"].astype(str).str.strip() if "League" in df.columns else path.stem
    odds = _odds_open_close_sharp(df)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
            "home_team": df[home_col].astype(str).str.strip(),
            "away_team": df[away_col].astype(str).str.strip(),
            "home_goals": _num(df[hg]),
            "away_goals": _num(df[ag]),
            "country": country_s,
            "league": league_s,
            "div": path.stem,
            **odds,
            "source": f"fd-extra:{path.stem}",
        }
    )
    return out.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])


def load_historical(min_date: str = "2019-07-01") -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv in sorted(FD_MAIN.glob("*/*.csv")):
        if csv.name.lower().startswith(("sa", "notes")):
            continue
        try:
            part = parse_main_results(csv)
        except Exception as exc:
            print(f"skip {csv.name}: {exc}")
            continue
        if not part.empty:
            frames.append(part)
    for csv in sorted(FD_EXTRA.glob("*.csv")):
        try:
            part = parse_extra_results(csv)
        except Exception as exc:
            print(f"skip extra {csv.name}: {exc}")
            continue
        if not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    try:
        from modules.data_update.cups import load_org_cup_results

        cups = load_org_cup_results()
        if not cups.empty:
            frames_hist = [df, cups]
            df = pd.concat(frames_hist, ignore_index=True)
    except Exception as exc:
        print(f"skip coppe storiche: {exc}")
    df = df[df["date"] >= pd.Timestamp(min_date)]
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    return df.sort_values("date").reset_index(drop=True)


def parse_main_fixtures(path: Path) -> pd.DataFrame:
    df = _read_fd_csv(path)
    if "HomeTeam" not in df.columns:
        return pd.DataFrame()
    mapped = df["Div"].map(lambda d: _meta_from_div(d)) if "Div" in df.columns else None
    odds = _odds_open_close_sharp(df)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
            "time": df["Time"] if "Time" in df.columns else "",
            "home_team": df["HomeTeam"].astype(str).str.strip(),
            "away_team": df["AwayTeam"].astype(str).str.strip(),
            "country": mapped.map(lambda t: t[0]) if mapped is not None else "Europa",
            "league": mapped.map(lambda t: t[1]) if mapped is not None else df.get("Div", "unknown"),
            "div": df["Div"] if "Div" in df.columns else "",
            **odds,
            "source": "fixtures-main",
        }
    )


def parse_extra_fixtures(path: Path) -> pd.DataFrame:
    df = _read_fd_csv(path)
    home_col = "Home" if "Home" in df.columns else "HomeTeam"
    away_col = "Away" if "Away" in df.columns else "AwayTeam"
    if home_col not in df.columns:
        return pd.DataFrame()
    odds = _odds_open_close_sharp(df)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
            "time": df["Time"] if "Time" in df.columns else "",
            "home_team": df[home_col].astype(str).str.strip(),
            "away_team": df[away_col].astype(str).str.strip(),
            "country": df["Country"].astype(str).str.strip().map(_country_it) if "Country" in df.columns else "",
            "league": df["League"].astype(str).str.strip() if "League" in df.columns else "",
            "div": df["Country"] if "Country" in df.columns else "",
            **odds,
            "source": "fixtures-extra",
        }
    )


def load_fixtures() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    main = FIXTURES_DIR / "main.csv"
    extra = FIXTURES_DIR / "extra.csv"
    if main.exists():
        frames.append(parse_main_fixtures(main))
    if extra.exists():
        frames.append(parse_extra_fixtures(extra))
    try:
        from modules.data_update.cups import load_cup_fixtures

        cups = load_cup_fixtures()
        if not cups.empty:
            frames.append(cups)
    except Exception as exc:
        print(f"skip coppe in calendario: {exc}")
    try:
        from modules.data_update.world_fixtures import load_world_fixtures

        world = load_world_fixtures()
        if not world.empty:
            frames.append(world)
    except Exception as exc:
        print(f"skip calendario mondiale: {exc}")
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["date", "home_team", "away_team"])
    today = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    df = df[df["date"] >= today]
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    return df.sort_values(["date", "time"]).reset_index(drop=True)
