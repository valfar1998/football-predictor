"""Feature pre-match: forma, xG, gol, casa, Elo (qualità rosa), calendario."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


class FeatureEngineer:
    def __init__(self, window: int = 5, elo_k: float = 18.0, elo_start: float = 1500.0) -> None:
        self.window = window
        self.elo_k = elo_k
        self.elo_start = elo_start

    def transform(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Restituisce un DataFrame con feature calcolate solo su partite precedenti (no leakage)."""
        df = matches.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        history: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.window))
        elo: dict[str, float] = defaultdict(lambda: self.elo_start)
        last_played: dict[str, pd.Timestamp] = {}
        rows = []

        for i, (_, m) in enumerate(df.iterrows(), start=1):
            home, away = m["home_team"], m["away_team"]
            h_stats = self._team_stats(history[home])
            a_stats = self._team_stats(history[away])
            rest_h = self._rest_days(last_played.get(home), m["date"])
            rest_a = self._rest_days(last_played.get(away), m["date"])

            rows.append(
                {
                    "date": m["date"],
                    "home_team": home,
                    "away_team": away,
                    "league": m.get("league", "unknown"),
                    "country": m.get("country", "unknown"),
                    "season": m.get("season", ""),
                    "home_goals": m["home_goals"],
                    "away_goals": m["away_goals"],
                    "result": m["result"],
                    # forma recente
                    "home_form_pts": h_stats["pts"],
                    "away_form_pts": a_stats["pts"],
                    "home_form_gd": h_stats["gd"],
                    "away_form_gd": a_stats["gd"],
                    # xG / xGA
                    "home_xg_avg": h_stats["xg"],
                    "away_xg_avg": a_stats["xg"],
                    "home_xga_avg": h_stats["xga"],
                    "away_xga_avg": a_stats["xga"],
                    "xg_diff": h_stats["xg"] - a_stats["xg"],
                    "xga_diff": h_stats["xga"] - a_stats["xga"],
                    # gol fatti/subiti
                    "home_gf_avg": h_stats["gf"],
                    "away_gf_avg": a_stats["gf"],
                    "home_ga_avg": h_stats["ga"],
                    "away_ga_avg": a_stats["ga"],
                    # fattore casa (win rate casa nelle ultime N in casa, fallback 0.45)
                    "home_home_wr": h_stats["home_wr"],
                    "away_away_wr": a_stats["away_wr"],
                    # qualità rosa (Elo pre-match)
                    "home_elo": elo[home],
                    "away_elo": elo[away],
                    "elo_diff": elo[home] - elo[away],
                    # temporali
                    "month": int(m["date"].month),
                    "weekday": int(m["date"].weekday()),
                    "home_rest_days": rest_h,
                    "away_rest_days": rest_a,
                    "rest_diff": rest_h - rest_a,
                    "n_home_hist": h_stats["n"],
                    "n_away_hist": a_stats["n"],
                }
            )

            self._push(history[home], m, is_home=True)
            self._push(history[away], m, is_home=False)
            self._update_elo(elo, home, away, m["home_goals"], m["away_goals"])
            last_played[home] = m["date"]
            last_played[away] = m["date"]
            if i % 15000 == 0:
                print(f"feature {i}/{len(df)}")

        feat = pd.DataFrame(rows)
        print(f"feature rows raw={len(feat)}")
        # scarta le prime giornate senza storia sufficiente
        feat = feat[(feat["n_home_hist"] >= 3) & (feat["n_away_hist"] >= 3)].reset_index(drop=True)
        return feat

    def _team_stats(self, buf: deque) -> dict[str, float]:
        if not buf:
            return {
                "pts": 0.0, "gd": 0.0, "xg": 1.2, "xga": 1.2,
                "gf": 1.2, "ga": 1.2, "home_wr": 0.45, "away_wr": 0.30, "n": 0,
            }
        n = len(buf)
        pts = sum(r["pts"] for r in buf) / n
        gd = sum(r["gf"] - r["ga"] for r in buf) / n
        xg = sum(r["xg"] for r in buf) / n
        xga = sum(r["xga"] for r in buf) / n
        gf = sum(r["gf"] for r in buf) / n
        ga = sum(r["ga"] for r in buf) / n
        home_games = [r for r in buf if r["is_home"]]
        away_games = [r for r in buf if not r["is_home"]]
        home_wr = (sum(1 for r in home_games if r["pts"] == 3) / len(home_games)) if home_games else 0.45
        away_wr = (sum(1 for r in away_games if r["pts"] == 3) / len(away_games)) if away_games else 0.30
        return {
            "pts": pts, "gd": gd, "xg": xg, "xga": xga, "gf": gf, "ga": ga,
            "home_wr": home_wr, "away_wr": away_wr, "n": n,
        }

    def _push(self, buf: deque, m: pd.Series, *, is_home: bool) -> None:
        gf = int(m["home_goals"] if is_home else m["away_goals"])
        ga = int(m["away_goals"] if is_home else m["home_goals"])
        xg = float(m["home_xg"] if is_home else m["away_xg"])
        xga = float(m["away_xg"] if is_home else m["home_xg"])
        if gf > ga:
            pts = 3
        elif gf == ga:
            pts = 1
        else:
            pts = 0
        buf.append({"pts": pts, "gf": gf, "ga": ga, "xg": xg, "xga": xga, "is_home": is_home})

    def _update_elo(self, elo: dict[str, float], home: str, away: str, hg: int, ag: int) -> None:
        exp_h = 1.0 / (1.0 + 10 ** ((elo[away] - elo[home] - 60) / 400))
        if hg > ag:
            score_h = 1.0
        elif hg == ag:
            score_h = 0.5
        else:
            score_h = 0.0
        elo[home] += self.elo_k * (score_h - exp_h)
        elo[away] += self.elo_k * ((1 - score_h) - (1 - exp_h))

    @staticmethod
    def _rest_days(prev: pd.Timestamp | None, current: pd.Timestamp) -> int:
        if prev is None:
            return 7
        return int(max((current - prev).days, 1))

    def save(self, feat: pd.DataFrame, name: str = "features.csv") -> Path:
        dest = PROCESSED / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        feat.to_csv(dest, index=False)
        return dest

    FEATURE_COLS = [
        "home_form_pts", "away_form_pts", "home_form_gd", "away_form_gd",
        "home_xg_avg", "away_xg_avg", "home_xga_avg", "away_xga_avg", "xg_diff", "xga_diff",
        "home_gf_avg", "away_gf_avg", "home_ga_avg", "away_ga_avg",
        "home_home_wr", "away_away_wr",
        "home_elo", "away_elo", "elo_diff",
        "month", "weekday", "home_rest_days", "away_rest_days", "rest_diff",
    ]
