"""Feature pre-match: forma, xG, gol, casa, Elo (qualità rosa), calendario."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
STATE_PATH = PROCESSED / "feature_state.json"
FEATURES_PATH = PROCESSED / "features.csv"


class FeatureEngineer:
    def __init__(self, window: int = 5, elo_k: float = 18.0, elo_start: float = 1500.0) -> None:
        self.window = window
        self.elo_k = elo_k
        self.elo_start = elo_start

    def transform(self, matches: pd.DataFrame, *, incremental: bool = True) -> pd.DataFrame:
        """Restituisce feature pre-match (solo info precedenti alla partita, no leakage)."""
        df = matches.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        if incremental:
            try:
                inc = self._transform_incremental(df)
                if inc is not None:
                    print(f"feature incremental: +{inc.get('added', 0)} righe (tot {inc['n_rows']})")
                    return inc["feat"]
            except Exception as exc:
                print(f"feature incremental skip → full rebuild: {exc}")

        feat = self._transform_full(df)
        print(f"feature rows raw={len(feat)}")
        return feat

    def _transform_full(self, df: pd.DataFrame) -> pd.DataFrame:
        history, elo, last_played, recent = self._empty_runtime()
        rows = self._process_matches(df, history, elo, last_played, recent, start=0)
        feat = pd.DataFrame(rows)
        feat = feat[(feat["n_home_hist"] >= 3) & (feat["n_away_hist"] >= 3)].reset_index(drop=True)
        self._save_state(df, len(df) - 1, history, elo, last_played, recent)
        return feat

    def _transform_incremental(self, df: pd.DataFrame) -> dict | None:
        if not STATE_PATH.is_file() or not FEATURES_PATH.is_file():
            return None
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if int(state.get("matches_rows") or 0) != len(df):
            return None
        start = int(state.get("last_index", -1)) + 1
        existing = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
        if start >= len(df):
            return {"feat": existing, "added": 0, "n_rows": len(existing)}

        history, elo, last_played, recent = self._restore_runtime(state)
        new_rows = self._process_matches(df, history, elo, last_played, recent, start=start)
        if not new_rows:
            return {"feat": existing, "added": 0, "n_rows": len(existing)}

        added_df = pd.DataFrame(new_rows)
        feat = pd.concat([existing, added_df], ignore_index=True)
        feat = feat[(feat["n_home_hist"] >= 3) & (feat["n_away_hist"] >= 3)].reset_index(drop=True)
        self._save_state(df, len(df) - 1, history, elo, last_played, recent)
        return {"feat": feat, "added": len(new_rows), "n_rows": len(feat)}

    def _empty_runtime(self) -> tuple:
        history: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.window))
        elo: dict[str, float] = defaultdict(lambda: 1500.0)
        last_played: dict[str, pd.Timestamp] = {}
        recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=12))
        return history, elo, last_played, recent

    def _restore_runtime(self, state: dict) -> tuple:
        history: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.window))
        for team, buf in (state.get("history") or {}).items():
            history[team] = deque(list(buf)[-self.window :], maxlen=self.window)
        elo = defaultdict(lambda: self.elo_start, {k: float(v) for k, v in (state.get("elo") or {}).items()})
        last_played = {
            k: pd.Timestamp(v) for k, v in (state.get("last_played") or {}).items()
        }
        recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=12))
        for team, dates in (state.get("recent") or {}).items():
            recent[team] = deque([pd.Timestamp(d) for d in dates][-12:], maxlen=12)
        return history, elo, last_played, recent

    def _save_state(
        self,
        df: pd.DataFrame,
        last_index: int,
        history: dict[str, deque],
        elo: dict[str, float],
        last_played: dict[str, pd.Timestamp],
        recent: dict[str, deque],
    ) -> None:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_index": int(last_index),
            "matches_rows": int(len(df)),
            "last_date": str(pd.Timestamp(df.iloc[last_index]["date"]).date()) if len(df) else None,
            "elo": {k: round(float(v), 2) for k, v in elo.items()},
            "history": {k: list(v) for k, v in history.items()},
            "last_played": {k: str(pd.Timestamp(v).date()) for k, v in last_played.items()},
            "recent": {k: [str(pd.Timestamp(d).date()) for d in v] for k, v in recent.items()},
        }
        STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _process_matches(
        self,
        df: pd.DataFrame,
        history: dict[str, deque],
        elo: dict[str, float],
        last_played: dict[str, pd.Timestamp],
        recent: dict[str, deque],
        *,
        start: int,
    ) -> list[dict]:
        rows: list[dict] = []
        n = len(df)
        for i in range(start, n):
            m = df.iloc[i]
            home, away = m["home_team"], m["away_team"]
            h_stats = self._team_stats(history[home])
            a_stats = self._team_stats(history[away])
            rest_h = self._rest_days(last_played.get(home), m["date"])
            rest_a = self._rest_days(last_played.get(away), m["date"])
            m7_h = self._matches_in_days(recent[home], m["date"], 7)
            m7_a = self._matches_in_days(recent[away], m["date"], 7)

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
                    "home_form_pts": h_stats["pts"],
                    "away_form_pts": a_stats["pts"],
                    "home_form_gd": h_stats["gd"],
                    "away_form_gd": a_stats["gd"],
                    "home_xg_avg": h_stats["xg"],
                    "away_xg_avg": a_stats["xg"],
                    "home_xga_avg": h_stats["xga"],
                    "away_xga_avg": a_stats["xga"],
                    "xg_diff": h_stats["xg"] - a_stats["xg"],
                    "xga_diff": h_stats["xga"] - a_stats["xga"],
                    "home_gf_avg": h_stats["gf"],
                    "away_gf_avg": a_stats["gf"],
                    "home_ga_avg": h_stats["ga"],
                    "away_ga_avg": a_stats["ga"],
                    "home_home_wr": h_stats["home_wr"],
                    "away_away_wr": a_stats["away_wr"],
                    "home_elo": elo[home],
                    "away_elo": elo[away],
                    "elo_diff": elo[home] - elo[away],
                    "month": int(m["date"].month),
                    "weekday": int(m["date"].weekday()),
                    "home_rest_days": rest_h,
                    "away_rest_days": rest_a,
                    "rest_diff": rest_h - rest_a,
                    "home_matches_7d": m7_h,
                    "away_matches_7d": m7_a,
                    "congestion_diff": m7_h - m7_a,
                    "days_into_season": self._days_into_season(m["date"], m.get("season")),
                    "n_home_hist": h_stats["n"],
                    "n_away_hist": a_stats["n"],
                    **self._market_features(m),
                }
            )

            self._push(history[home], m, is_home=True)
            self._push(history[away], m, is_home=False)
            self._update_elo(elo, home, away, m["home_goals"], m["away_goals"])
            last_played[home] = m["date"]
            last_played[away] = m["date"]
            recent[home].append(m["date"])
            recent[away].append(m["date"])
            if (i + 1) % 15000 == 0:
                print(f"feature {i + 1}/{n}")

        return rows

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

    @staticmethod
    def _days_into_season(current, season) -> float:
        """Giorni dall'inizio stagione (proxy 1 luglio). Smorza le prime giornate."""
        cur = pd.Timestamp(current)
        year = int(cur.year)
        # stagione europea: se mese < 7, stagione iniziata luglio anno-1
        start_year = year if cur.month >= 7 else year - 1
        s = str(season or "")
        digits = "".join(ch for ch in s if ch.isdigit())
        if len(digits) >= 4:
            try:
                start_year = int(digits[:4])
            except ValueError:
                pass
        start = pd.Timestamp(year=start_year, month=7, day=1)
        return float(max(0, (cur - start).days))

    @staticmethod
    def _matches_in_days(dates: deque, current: pd.Timestamp, window: int) -> int:
        cur = pd.Timestamp(current)
        return sum(1 for d in dates if 0 < (cur - pd.Timestamp(d)).days <= window)

    @staticmethod
    def _odd_val(m, *names: str) -> float | None:
        for name in names:
            if name not in getattr(m, "index", []):
                if not isinstance(m, dict) or name not in m:
                    continue
            try:
                val = float(m[name])
            except (TypeError, ValueError, KeyError):
                continue
            if val > 1.01:
                return val
        return None

    @classmethod
    def implied_1x2(cls, m) -> tuple[float, float, float, float, float]:
        """p_home, p_draw, p_away, overround, has_market (0/1). Close se c'è, altrimenti open."""
        o1 = cls._odd_val(m, "odd_home_close", "odd_home", "1")
        ox = cls._odd_val(m, "odd_draw_close", "odd_draw", "X", "x")
        o2 = cls._odd_val(m, "odd_away_close", "odd_away", "2")
        if not o1 or not ox or not o2:
            return 0.34, 0.28, 0.38, 0.0, 0.0
        i1, ix, i2 = 1.0 / o1, 1.0 / ox, 1.0 / o2
        tot = i1 + ix + i2
        return i1 / tot, ix / tot, i2 / tot, round(tot - 1.0, 4), 1.0

    @classmethod
    def _market_features(cls, m) -> dict[str, float]:
        p1, px, p2, ov, has = cls.implied_1x2(m)
        return {
            "mkt_p_home": p1,
            "mkt_p_draw": px,
            "mkt_p_away": p2,
            "mkt_overround": ov,
            "mkt_has": has,
        }

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
        "home_matches_7d", "away_matches_7d", "congestion_diff",
        "days_into_season",
        "mkt_p_home", "mkt_p_draw", "mkt_p_away", "mkt_overround", "mkt_has",
    ]
