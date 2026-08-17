"""Genera un campionato sintetico realistico (xG + gol) per far girare il pipeline senza dataset esterni."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

TEAMS = {
    "Inter": 86,
    "Milan": 82,
    "Juventus": 81,
    "Napoli": 80,
    "Roma": 76,
    "Atalanta": 78,
    "Lazio": 74,
    "Fiorentina": 72,
    "Bologna": 70,
    "Torino": 68,
    "Udinese": 64,
    "Genoa": 63,
    "Monza": 61,
    "Lecce": 58,
    "Empoli": 57,
    "Cagliari": 56,
    "Hellas Verona": 55,
    "Sassuolo": 54,
    "Frosinone": 52,
    "Salernitana": 50,
}


def generate(seasons: int = 4, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(TEAMS)
    rows = []
    for year in range(2021, 2021 + seasons):
        start = pd.Timestamp(f"{year}-08-20")
        matchday = 0
        # andata e ritorno
        pairs = [(h, a) for h in names for a in names if h != a]
        rng.shuffle(pairs)
        for i, (home, away) in enumerate(pairs):
            matchday = i // 10
            date = start + pd.Timedelta(days=matchday * 7 + int(rng.integers(0, 3)))
            qh, qa = TEAMS[home], TEAMS[away]
            lam_h = np.exp((qh - qa) / 35) * 1.25  # vantaggio casa
            lam_a = np.exp((qa - qh) / 35) * 0.95
            lam_h = float(np.clip(lam_h + rng.normal(0, 0.12), 0.4, 3.2))
            lam_a = float(np.clip(lam_a + rng.normal(0, 0.12), 0.3, 2.8))
            hg = int(rng.poisson(lam_h))
            ag = int(rng.poisson(lam_a))
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "home_team": home,
                    "away_team": away,
                    "home_goals": hg,
                    "away_goals": ag,
                    "home_xg": round(lam_h, 3),
                    "away_xg": round(lam_a, 3),
                    "league": "Serie A",
                    "season": f"{year}/{year+1}",
                }
            )
    return pd.DataFrame(rows).sort_values("date")


def main() -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    df = generate()
    dest = RAW / "serie_a_synthetic.csv"
    df.to_csv(dest, index=False)
    print(f"scritto {dest} ({len(df)} partite)")
    return dest


if __name__ == "__main__":
    main()
