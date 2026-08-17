"""Carica CSV (Understat, FBref, football-data.co.uk), unisce, normalizza, pulisce."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

TEAM_ALIASES = {
    "inter": "Inter",
    "inter milan": "Inter",
    "fc internazionale": "Inter",
    "internazionale": "Inter",
    "ac milan": "Milan",
    "milan": "Milan",
    "juventus": "Juventus",
    "juve": "Juventus",
    "as roma": "Roma",
    "roma": "Roma",
    "ssc napoli": "Napoli",
    "napoli": "Napoli",
    "lazio": "Lazio",
    "ss lazio": "Lazio",
    "atalanta": "Atalanta",
    "fiorentina": "Fiorentina",
    "torino": "Torino",
    "bologna": "Bologna",
    "udinese": "Udinese",
    "genoa": "Genoa",
    "cagliari": "Cagliari",
    "empoli": "Empoli",
    "monza": "Monza",
    "lecce": "Lecce",
    "sassuolo": "Sassuolo",
    "salernitana": "Salernitana",
    "verona": "Hellas Verona",
    "hellas verona": "Hellas Verona",
    "frosinone": "Frosinone",
    "man utd": "Manchester United",
    "man united": "Manchester United",
    "manchester utd": "Manchester United",
    "manchester united": "Manchester United",
    "man city": "Manchester City",
    "manchester city": "Manchester City",
    "spurs": "Tottenham",
    "tottenham hotspur": "Tottenham",
    "tottenham": "Tottenham",
    "wolves": "Wolverhampton",
    "wolverhampton wanderers": "Wolverhampton",
}

COLUMN_MAP = {
    "date": ["date", "Date", "match_date"],
    "home_team": ["home_team", "Home", "HomeTeam", "h_team"],
    "away_team": ["away_team", "Away", "AwayTeam", "a_team"],
    "home_goals": ["home_goals", "FTHG", "hg", "home_score"],
    "away_goals": ["away_goals", "FTAG", "ag", "away_score"],
    "home_xg": ["home_xg", "xG_h", "hxG", "Home_xG"],
    "away_xg": ["away_xg", "xG_a", "axG", "Away_xG"],
    "league": ["league", "Div", "competition"],
    "season": ["season", "Season"],
}


def normalize_team(name: str) -> str:
    raw = str(name).strip()
    key = " ".join(raw.lower().split())
    return TEAM_ALIASES.get(key, raw)


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def standardize_frame(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """Rinomina colonne eterogenee nello schema interno."""
    out = pd.DataFrame()
    mapping = {}
    for canonical, aliases in COLUMN_MAP.items():
        col = _first_present(df, aliases)
        if col is not None:
            mapping[canonical] = col
    for canonical, col in mapping.items():
        out[canonical] = df[col]
    if "date" in out:
        out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)
    if "home_team" in out:
        out["home_team"] = out["home_team"].map(normalize_team)
    if "away_team" in out:
        out["away_team"] = out["away_team"].map(normalize_team)
    for goals in ("home_goals", "away_goals"):
        if goals in out:
            out[goals] = pd.to_numeric(out[goals], errors="coerce")
    for xg in ("home_xg", "away_xg"):
        if xg in out:
            out[xg] = pd.to_numeric(out[xg], errors="coerce")
        else:
            out[xg] = pd.NA
    if "league" not in out:
        out["league"] = "unknown"
    if "country" not in out:
        out["country"] = "unknown"
    if "season" not in out:
        out["season"] = out["date"].dt.year if "date" in out else pd.NA
    out["source"] = source
    return out


class DatasetLoader:
    def __init__(self, raw_dir: Path | None = None, processed_dir: Path | None = None) -> None:
        self.raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_csv(self, path: str | Path, source: str | None = None) -> pd.DataFrame:
        path = Path(path)
        df = pd.read_csv(path)
        return standardize_frame(df, source=source or path.stem)

    def load_raw_folder(self, patterns: Iterable[str] = ("*.csv",)) -> pd.DataFrame:
        files: list[Path] = []
        for pattern in patterns:
            files.extend(sorted(self.raw_dir.glob(pattern)))
        if not files:
            raise FileNotFoundError(f"Nessun CSV in {self.raw_dir}")
        frames = [self.load_csv(f) for f in files]
        return self.merge(frames)

    def merge(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        """Unione su data + squadre; media xG se la stessa partita arriva da più fonti."""
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.dropna(subset=["date", "home_team", "away_team"])
        combined = combined.drop_duplicates(
            subset=["date", "home_team", "away_team", "source"],
            keep="last",
        )
        agg = {
            "home_goals": "mean",
            "away_goals": "mean",
            "home_xg": "mean",
            "away_xg": "mean",
            "league": "first",
            "country": "first",
            "season": "first",
            "source": lambda s: "+".join(sorted(set(s.astype(str)))),
            "odd_home": "mean",
            "odd_draw": "mean",
            "odd_away": "mean",
        }
        merged = (
            combined.groupby(["date", "home_team", "away_team"], as_index=False)
            .agg({k: v for k, v in agg.items() if k in combined.columns})
        )
        return merged.sort_values("date").reset_index(drop=True)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = out.dropna(subset=["date", "home_team", "away_team"])
        if "home_xg" not in out:
            out["home_xg"] = pd.NA
        if "away_xg" not in out:
            out["away_xg"] = pd.NA
        if "country" not in out:
            out["country"] = "unknown"
        if "league" not in out:
            out["league"] = "unknown"
        # xG mancante: stima dai gol (football-data.co.uk non pubblica xG)
        out["home_xg"] = pd.to_numeric(out["home_xg"], errors="coerce")
        out["away_xg"] = pd.to_numeric(out["away_xg"], errors="coerce")
        out["home_xg"] = out["home_xg"].fillna(out["home_goals"].clip(lower=0) * 0.95 + 0.15)
        out["away_xg"] = out["away_xg"].fillna(out["away_goals"].clip(lower=0) * 0.95 + 0.15)
        out["home_goals"] = out["home_goals"].round().astype("Int64")
        out["away_goals"] = out["away_goals"].round().astype("Int64")
        out = out.dropna(subset=["home_goals", "away_goals"])
        out["home_goals"] = out["home_goals"].astype(int)
        out["away_goals"] = out["away_goals"].astype(int)
        out["result"] = out.apply(
            lambda r: "H" if r.home_goals > r.away_goals else ("A" if r.away_goals > r.home_goals else "D"),
            axis=1,
        )
        out["home_xg"] = out["home_xg"].astype(float)
        out["away_xg"] = out["away_xg"].astype(float)
        return out.reset_index(drop=True)

    def save_processed(self, df: pd.DataFrame, name: str = "matches.csv") -> Path:
        dest = self.processed_dir / name
        df.to_csv(dest, index=False)
        return dest

    def run(self, output_name: str = "matches.csv") -> tuple[pd.DataFrame, Path]:
        from modules.data_update.parse import load_historical

        fd_main = ROOT / "data" / "raw" / "fd" / "main"
        if fd_main.exists() and any(fd_main.glob("*/*.csv")):
            raw = load_historical()
            if raw.empty:
                raise FileNotFoundError("Nessun risultato in data/raw/fd. Lancia python main.py --update")
        else:
            raw = self.load_raw_folder()
        clean = self.clean(raw)
        path = self.save_processed(clean, output_name)
        return clean, path
