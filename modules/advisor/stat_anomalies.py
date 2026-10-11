"""Ritardi / anomalie statistiche (UI/quadro). Non tocca EV/Kelly/p_cons.

Detector su matches.csv + CSV football-data (falli/tiri) + assenze WhoScored.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MATCHES = PROCESSED / "matches.csv"
FD_MAIN = ROOT / "data" / "raw" / "fd" / "main"
CACHE_PATH = PROCESSED / "stat_anomalies.json"
FD_GAMES_CACHE = PROCESSED / "fd_team_games_long.csv"

MIN_SEVERITY = 0.35
MAX_PER_MATCH = 6
SEASON_START_N = 5
STREAK_K = 5
DROUGHT_M = 2
DROUGHT_SEASON_MIN = 8


def _norm(name: str) -> str:
    from modules.data_update.cups import _norm_key

    return _norm_key(name or "")


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _is_bettable_anomaly(a: dict[str, Any]) -> bool:
    """Esclude ritardi non scommettibili (es. xG)."""
    title = str(a.get("title") or "").lower()
    if "xg in ritardo" in title or title.endswith(": xg in ritardo"):
        return False
    return True


def sanitize_anomaly_block(block: dict[str, Any] | None) -> dict[str, Any]:
    """Ricalcola n / ready / max_severity dopo aver tolto item non bettable."""
    if not isinstance(block, dict):
        return {"ready": False, "items": [], "n": 0, "max_severity": 0.0}
    items = [a for a in (block.get("items") or []) if isinstance(a, dict) and _is_bettable_anomaly(a)]
    max_sev = max((float(a.get("severity") or 0) for a in items), default=0.0)
    out = dict(block)
    out["items"] = items
    out["n"] = len(items)
    out["ready"] = bool(items)
    out["max_severity"] = round(max_sev, 3)
    return out


def _season_label(ts: pd.Timestamp) -> str:
    """Stagione europea tipica (lug→giu)."""
    y = int(ts.year)
    if int(ts.month) >= 7:
        return f"{y}/{y + 1}"
    return f"{y - 1}/{y}"


def _anomaly(
    *,
    kind: str,
    side: str,
    title: str,
    detail: str,
    baseline_kind: str,
    baseline: float | None,
    observed: float | None,
    n: int,
    severity: float,
    source: str,
    team: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "side": side,
        "team": team,
        "title": title,
        "detail": detail,
        "baseline_kind": baseline_kind,
        "baseline": None if baseline is None else round(float(baseline), 4),
        "observed": None if observed is None else round(float(observed), 4),
        "n": int(n),
        "severity": round(_clip01(severity), 3),
        "source": source,
    }


@lru_cache(maxsize=1)
def _load_team_games() -> pd.DataFrame:
    """Storico long: una riga per squadra-partita da matches.csv."""
    if not MATCHES.exists():
        return pd.DataFrame()
    use = [
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "league",
        "result",
        "home_corners",
        "away_corners",
        "home_yellow",
        "away_yellow",
    ]
    df = pd.read_csv(MATCHES, usecols=lambda c: c in use, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        hg = pd.to_numeric(r.get("home_goals"), errors="coerce")
        ag = pd.to_numeric(r.get("away_goals"), errors="coerce")
        if pd.isna(hg) or pd.isna(ag):
            continue
        hg_i, ag_i = int(hg), int(ag)
        league = str(r.get("league") or "")
        day = r["date"]
        for team, opp, gf, ga, is_home, corners, yellow in (
            (
                str(r["home_team"]),
                str(r["away_team"]),
                hg_i,
                ag_i,
                True,
                r.get("home_corners"),
                r.get("home_yellow"),
            ),
            (
                str(r["away_team"]),
                str(r["home_team"]),
                ag_i,
                hg_i,
                False,
                r.get("away_corners"),
                r.get("away_yellow"),
            ),
        ):
            won = gf > ga
            rows.append(
                {
                    "date": day,
                    "team": team,
                    "team_norm": _norm(team),
                    "opp": opp,
                    "league": league,
                    "is_home": is_home,
                    "gf": gf,
                    "ga": ga,
                    "won": won,
                    "drew": gf == ga,
                    "lost": gf < ga,
                    "scored": gf > 0,
                    "total_goals": gf + ga,
                    "corners": pd.to_numeric(corners, errors="coerce"),
                    "yellow": pd.to_numeric(yellow, errors="coerce"),
                    "season": _season_label(pd.Timestamp(day)),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["team_norm", "date"]).reset_index(drop=True)


def _fd_cache_fresh() -> bool:
    if not FD_GAMES_CACHE.exists() or not FD_MAIN.exists():
        return False
    try:
        cache_mtime = FD_GAMES_CACHE.stat().st_mtime
        newest = max((p.stat().st_mtime for p in FD_MAIN.glob("*/*.csv")), default=0.0)
        return cache_mtime >= newest - 1.0
    except OSError:
        return False


@lru_cache(maxsize=1)
def _load_fd_team_games() -> pd.DataFrame:
    """Long FD con falli/tiri/corner/cartellini (raw CSV → cache CSV)."""
    if _fd_cache_fresh():
        try:
            out = pd.read_csv(FD_GAMES_CACHE, parse_dates=["date"])
            if not out.empty:
                return out.sort_values(["team_norm", "date"]).reset_index(drop=True)
        except Exception:
            pass
    if not FD_MAIN.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for csv in sorted(FD_MAIN.glob("*/*.csv")):
        if csv.name.lower().startswith(("sa", "notes")):
            continue
        try:
            raw = csv.read_bytes()
            if raw.startswith(b"\xef\xbb\xbf"):
                raw = raw[3:]
            import io

            df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip")
        except Exception:
            continue
        df.columns = [str(c).replace("\ufeff", "").replace("ï»¿", "").strip() for c in df.columns]
        if not {"HomeTeam", "AwayTeam", "Date"}.issubset(df.columns):
            continue
        if "HF" not in df.columns and "HS" not in df.columns and "HC" not in df.columns:
            continue
        part = pd.DataFrame(
            {
                "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
                "home": df["HomeTeam"].astype(str).str.strip(),
                "away": df["AwayTeam"].astype(str).str.strip(),
                "hf": pd.to_numeric(df["HF"], errors="coerce") if "HF" in df.columns else pd.NA,
                "af": pd.to_numeric(df["AF"], errors="coerce") if "AF" in df.columns else pd.NA,
                "hs": pd.to_numeric(df["HS"], errors="coerce") if "HS" in df.columns else pd.NA,
                "as_": pd.to_numeric(df["AS"], errors="coerce") if "AS" in df.columns else pd.NA,
                "hc": pd.to_numeric(df["HC"], errors="coerce") if "HC" in df.columns else pd.NA,
                "ac": pd.to_numeric(df["AC"], errors="coerce") if "AC" in df.columns else pd.NA,
                "hy": pd.to_numeric(df["HY"], errors="coerce") if "HY" in df.columns else pd.NA,
                "ay": pd.to_numeric(df["AY"], errors="coerce") if "AY" in df.columns else pd.NA,
            }
        )
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    all_m = pd.concat(frames, ignore_index=True).dropna(subset=["date", "home", "away"])
    rows: list[dict[str, Any]] = []
    for _, r in all_m.iterrows():
        for team, fouls, shots, corners, yellow in (
            (r["home"], r["hf"], r["hs"], r["hc"], r["hy"]),
            (r["away"], r["af"], r["as_"], r["ac"], r["ay"]),
        ):
            rows.append(
                {
                    "date": r["date"],
                    "team": str(team),
                    "team_norm": _norm(str(team)),
                    "fouls": pd.to_numeric(fouls, errors="coerce"),
                    "shots": pd.to_numeric(shots, errors="coerce"),
                    "corners": pd.to_numeric(corners, errors="coerce"),
                    "yellow": pd.to_numeric(yellow, errors="coerce"),
                    "season": _season_label(pd.Timestamp(r["date"])),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["team_norm", "date"]).reset_index(drop=True)
    try:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        out.to_csv(FD_GAMES_CACHE, index=False)
    except Exception:
        pass
    return out


def clear_stat_anomaly_caches() -> None:
    _load_team_games.cache_clear()
    _load_fd_team_games.cache_clear()


def _team_slice(games: pd.DataFrame, team: str, before: pd.Timestamp) -> pd.DataFrame:
    key = _norm(team)
    if games.empty or not key:
        return pd.DataFrame()
    sub = games[(games["team_norm"] == key) & (games["date"] < before)]
    return sub.sort_values("date")


def _severity_binom(p: float, n: int, k: int = 0) -> float:
    """Quanto è strano osservare ≤k successi in n prove con p baseline (approx)."""
    if n <= 0 or p <= 0:
        return 0.0
    # P(X=0) ≈ (1-p)^n ; map to severity
    p0 = (1.0 - min(0.95, max(0.05, p))) ** n
    # lower p0 → more anomalous
    return _clip01(1.0 - p0 * 1.15)


def _detect_season_start(team: str, games: pd.DataFrame, before: pd.Timestamp, side: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    sub = _team_slice(games, team, before)
    if sub.empty:
        return out
    season = _season_label(before)
    cur = sub[sub["season"] == season].sort_values("date")
    n = len(cur)
    if n < 2 or n > SEASON_START_N:
        return out
    wins = int(cur["won"].sum())
    scored = int(cur["scored"].sum())
    # baseline: win rate storica squadra (tutte le partite pre-kickoff)
    hist = sub[sub["season"] != season]
    if len(hist) < 20:
        baseline_w = float(sub["won"].mean()) if len(sub) >= 10 else 0.35
    else:
        baseline_w = float(hist["won"].mean())
    if wins == 0 and baseline_w >= 0.28:
        sev = _severity_binom(baseline_w, n, 0)
        if sev >= MIN_SEVERITY:
            out.append(
                _anomaly(
                    kind="season_start",
                    side=side,
                    team=team,
                    title=f"{team}: 0 vittorie nelle prime {n}",
                    detail=(
                        f"0W in {n} all'avvio {season} · baseline win% storica "
                        f"~{baseline_w:.0%}"
                    ),
                    baseline_kind="season_avg",
                    baseline=baseline_w,
                    observed=0.0,
                    n=n,
                    severity=sev + 0.05 * (n - 2),
                    source="matches.csv",
                )
            )
    if scored == 0 and n >= 3:
        base_score = float(hist["scored"].mean()) if len(hist) >= 20 else float(sub["scored"].mean())
        if base_score >= 0.55:
            sev = _severity_binom(base_score, n, 0)
            if sev >= MIN_SEVERITY:
                out.append(
                    _anomaly(
                        kind="season_start",
                        side=side,
                        team=team,
                        title=f"{team}: 0 gol nelle prime {n}",
                        detail=f"a digiuno in {n} · baseline partite con gol ~{base_score:.0%}",
                        baseline_kind="season_avg",
                        baseline=base_score,
                        observed=0.0,
                        n=n,
                        severity=sev,
                        source="matches.csv",
                    )
                )
    return out


def _detect_streaks(team: str, games: pd.DataFrame, before: pd.Timestamp, side: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    sub = _team_slice(games, team, before)
    if len(sub) < STREAK_K + 5:
        return out
    recent = sub.tail(STREAK_K)
    hist = sub.iloc[:-STREAK_K] if len(sub) > STREAK_K else sub
    base_w = float(hist["won"].mean()) if len(hist) >= 15 else float(sub["won"].mean())
    wins = int(recent["won"].sum())
    if wins == 0 and base_w >= 0.28:
        sev = _severity_binom(base_w, STREAK_K, 0)
        if sev >= MIN_SEVERITY:
            out.append(
                _anomaly(
                    kind="streak",
                    side=side,
                    team=team,
                    title=f"{team}: {STREAK_K} senza vittoria",
                    detail=f"0W ultime {STREAK_K} · win% storica ~{base_w:.0%}",
                    baseline_kind="last_n_vs_season",
                    baseline=base_w,
                    observed=0.0,
                    n=STREAK_K,
                    severity=sev,
                    source="matches.csv",
                )
            )
    scored = int(recent["scored"].sum())
    base_s = float(hist["scored"].mean()) if len(hist) >= 15 else 0.7
    if scored == 0 and base_s >= 0.55:
        sev = _severity_binom(base_s, STREAK_K, 0)
        if sev >= MIN_SEVERITY:
            out.append(
                _anomaly(
                    kind="streak",
                    side=side,
                    team=team,
                    title=f"{team}: {STREAK_K} senza segnare",
                    detail=f"0 gol ultime {STREAK_K} · partite con gol ~{base_s:.0%}",
                    baseline_kind="last_n_vs_season",
                    baseline=base_s,
                    observed=0.0,
                    n=STREAK_K,
                    severity=sev,
                    source="matches.csv",
                )
            )
    # no over 2.5 drought
    overs = int((recent["total_goals"] > 2).sum())
    base_o = float((hist["total_goals"] > 2).mean()) if len(hist) >= 15 else 0.5
    if overs == 0 and base_o >= 0.45 and STREAK_K >= 4:
        sev = _severity_binom(base_o, STREAK_K, 0)
        if sev >= MIN_SEVERITY:
            out.append(
                _anomaly(
                    kind="streak",
                    side=side,
                    team=team,
                    title=f"{team}: {STREAK_K} Under 2.5 di fila",
                    detail=f"0 Over ultime {STREAK_K} · baseline Over ~{base_o:.0%}",
                    baseline_kind="last_n_vs_season",
                    baseline=base_o,
                    observed=0.0,
                    n=STREAK_K,
                    severity=sev * 0.9,
                    source="matches.csv",
                )
            )
    return out


def _rate_drought_metric(
    *,
    team: str,
    side: str,
    series: pd.Series,
    dates: pd.Series,
    before: pd.Timestamp,
    label: str,
    unit: str,
    source: str,
    min_season_avg: float,
) -> dict[str, Any] | None:
    mask = dates < before
    vals = pd.to_numeric(series[mask], errors="coerce")
    dts = dates[mask]
    ok = vals.notna()
    vals = vals[ok]
    dts = dts[ok]
    if len(vals) < DROUGHT_SEASON_MIN + DROUGHT_M:
        return None
    # season avg = current season excluding last M
    season = _season_label(before)
    season_mask = dts.map(lambda d: _season_label(pd.Timestamp(d)) == season)
    season_vals = vals[season_mask]
    if len(season_vals) < DROUGHT_SEASON_MIN:
        season_vals = vals  # fallback all history
    recent = vals.tail(DROUGHT_M)
    season_body = season_vals.iloc[:-DROUGHT_M] if len(season_vals) > DROUGHT_M else season_vals
    if len(season_body) < max(5, DROUGHT_SEASON_MIN - 2):
        return None
    base = float(season_body.mean())
    obs = float(recent.mean())
    if base < min_season_avg:
        return None
    # drought if recent << baseline
    if obs > base * 0.45 and obs > 0.15:
        return None
    gap = (base - obs) / max(base, 1e-6)
    sev = _clip01(0.25 + gap * 0.85)
    if sev < MIN_SEVERITY:
        return None
    return _anomaly(
        kind="rate_drought",
        side=side,
        team=team,
        title=f"{team}: {label} in ritardo",
        detail=(
            f"{obs:.1f}{unit}/p ultime {DROUGHT_M} vs {base:.1f}{unit}/p stagione "
            f"(Δ {obs - base:+.1f})"
        ),
        baseline_kind="last_n_vs_season",
        baseline=base,
        observed=obs,
        n=DROUGHT_M,
        severity=sev,
        source=source,
    )


def _detect_rate_droughts(
    team: str,
    games: pd.DataFrame,
    fd: pd.DataFrame,
    before: pd.Timestamp,
    side: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    sub = _team_slice(games, team, before)
    if not sub.empty:
        # Solo metriche scommettibili (no xG: non è un mercato).
        for col, label, unit, amin in (
            ("corners", "corner", "", 3.0),
            ("yellow", "cartellini", "", 1.0),
            ("gf", "gol", "", 0.8),
        ):
            if col not in sub.columns:
                continue
            a = _rate_drought_metric(
                team=team,
                side=side,
                series=sub[col],
                dates=sub["date"],
                before=before,
                label=label,
                unit=unit,
                source="matches.csv",
                min_season_avg=amin,
            )
            if a:
                out.append(a)
    key = _norm(team)
    if not fd.empty and key:
        fsub = fd[(fd["team_norm"] == key) & (fd["date"] < before)].sort_values("date")
        if not fsub.empty:
            for col, label, amin in (
                ("fouls", "falli", 8.0),
                ("shots", "tiri", 8.0),
                ("corners", "corner FD", 3.5),
            ):
                if col not in fsub.columns:
                    continue
                a = _rate_drought_metric(
                    team=team,
                    side=side,
                    series=fsub[col],
                    dates=fsub["date"],
                    before=before,
                    label=label,
                    unit="",
                    source="fd",
                    min_season_avg=amin,
                )
                if a:
                    out.append(a)
    return out


def _detect_absences(team: str, side: str, missing: list[dict[str, Any]], weight: float) -> list[dict[str, Any]]:
    if not missing:
        return []
    notes: list[str] = []
    for row in missing[:6]:
        player = str(row.get("player") or "").strip()
        if not player:
            continue
        reason = str(row.get("reason") or row.get("status") or "out")
        notes.append(f"{player} ({reason})")
    if not notes:
        return []
    sev = _clip01(0.25 + float(weight or 0) * 1.6 + 0.04 * min(len(notes), 5))
    return [
        _anomaly(
            kind="absence",
            side=side,
            team=team,
            title=f"{team}: {len(notes)} assenze",
            detail="; ".join(notes[:4]) + (f" · peso xG+xA ~{weight:.0%}" if weight else ""),
            baseline_kind="whoscored",
            baseline=None,
            observed=float(len(notes)),
            n=len(notes),
            severity=max(MIN_SEVERITY, sev),
            source="whoscored",
        )
    ]


def match_anomalies(
    home: str,
    away: str,
    kickoff: str | pd.Timestamp | None = None,
    *,
    tactical: dict[str, Any] | None = None,
    max_n: int = MAX_PER_MATCH,
    min_severity: float = MIN_SEVERITY,
) -> dict[str, Any]:
    """Calcola ritardi/anomalie per una partita. Soft-fail → lista vuota."""
    try:
        before = pd.Timestamp(kickoff) if kickoff is not None else pd.Timestamp.now()
        if pd.isna(before):
            before = pd.Timestamp.now()
    except Exception:
        before = pd.Timestamp.now()

    items: list[dict[str, Any]] = []
    try:
        games = _load_team_games()
        fd = _load_fd_team_games()
        for team, side in ((home, "home"), (away, "away")):
            if not team:
                continue
            items.extend(_detect_season_start(team, games, before, side))
            items.extend(_detect_streaks(team, games, before, side))
            items.extend(_detect_rate_droughts(team, games, fd, before, side))
    except Exception as exc:
        return {"ready": False, "items": [], "error": str(exc), "n": 0, "max_severity": 0.0}

    # Assenze da tactical già calcolato (WhoScored)
    tac = tactical or {}
    absn = tac.get("absences") or {}
    ws = tac.get("whoscored") or {}
    try:
        if absn.get("ready") or (ws.get("home") or ws.get("away")):
            items.extend(
                _detect_absences(home, "home", list(ws.get("home") or []), float(absn.get("weight_home") or 0))
            )
            items.extend(
                _detect_absences(away, "away", list(ws.get("away") or []), float(absn.get("weight_away") or 0))
            )
        elif not absn.get("ready"):
            # lookup diretto se tactical non ha ancora WS
            try:
                from modules.data_update.whoscored_context import lookup_whoscored_match
                from modules.advisor.tactics import injury_weights

                ws2 = lookup_whoscored_match(home, away, before)
                inj = injury_weights(ws2.get("home") or [], ws2.get("away") or [], home=home, away=away)
                items.extend(
                    _detect_absences(home, "home", list(ws2.get("home") or []), float(inj.get("weight_home") or 0))
                )
                items.extend(
                    _detect_absences(away, "away", list(ws2.get("away") or []), float(inj.get("weight_away") or 0))
                )
            except Exception:
                pass
    except Exception:
        pass

    items = [
        a
        for a in items
        if float(a.get("severity") or 0) >= min_severity and _is_bettable_anomaly(a)
    ]
    items.sort(key=lambda a: -float(a.get("severity") or 0))
    # dedupe by title
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for a in items:
        t = str(a.get("title") or "")
        if t in seen:
            continue
        seen.add(t)
        uniq.append(a)
    top = uniq[: max(1, int(max_n))]
    max_sev = max((float(a.get("severity") or 0) for a in top), default=0.0)
    return {
        "ready": bool(top),
        "items": top,
        "n": len(top),
        "max_severity": round(max_sev, 3),
        "home": home,
        "away": away,
        "date": str(before.date()) if hasattr(before, "date") else str(before)[:10],
    }


def _match_key(date: str, home: str, away: str) -> str:
    return f"{date}|{_norm(home)}|{_norm(away)}"


def load_anomaly_cache() -> dict[str, Any]:
    if not CACHE_PATH.exists():
        return {"ok": True, "matches": {}}
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"ok": True, "matches": {}}
        data.setdefault("matches", {})
        return data
    except Exception:
        return {"ok": True, "matches": {}}


def save_anomaly_cache(data: dict[str, Any]) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload["updated_at"] = pd.Timestamp.utcnow().isoformat()
    CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=0), encoding="utf-8")


def attach_match_anomalies(
    home: str,
    away: str,
    kickoff: str | None = None,
    *,
    tactical: dict[str, Any] | None = None,
    cache: dict[str, Any] | None = None,
    write_cache: bool = False,
) -> dict[str, Any]:
    """Calcola (o riusa cache) anomalie per una partita."""
    day = str(kickoff or "")[:10]
    key = _match_key(day, home, away)
    store = cache if cache is not None else load_anomaly_cache()
    matches = store.setdefault("matches", {})
    prev = matches.get(key)
    if isinstance(prev, dict) and prev.get("items") is not None and prev.get("ready") is not None:
        cleaned = sanitize_anomaly_block(prev)
        if cleaned.get("n") != (prev.get("n") if prev.get("n") is not None else len(prev.get("items") or [])):
            matches[key] = cleaned
            if write_cache:
                save_anomaly_cache(store)
        return cleaned
    result = match_anomalies(home, away, kickoff, tactical=tactical)
    matches[key] = result
    if write_cache:
        save_anomaly_cache(store)
    return result


def enrich_upcoming_anomalies(
    rows: list[dict[str, Any]],
    *,
    write_cache: bool = True,
    days_ahead: int | None = None,
) -> dict[str, Any]:
    """Attacca stat_anomalies a ogni riga upcoming; aggiorna cache JSON."""
    store = load_anomaly_cache()
    n_ok = 0
    today = pd.Timestamp.now().normalize()
    for row in rows:
        home = str(row.get("home") or "")
        away = str(row.get("away") or "")
        day = str(row.get("date") or "")[:10]
        if not home or not away:
            continue
        if days_ahead is not None and day:
            try:
                d = pd.Timestamp(day)
                if d > today + pd.Timedelta(days=int(days_ahead)):
                    continue
            except Exception:
                pass
        pred = row.get("prediction") if isinstance(row.get("prediction"), dict) else {}
        tac = (pred or {}).get("tactical") if isinstance(pred, dict) else None
        try:
            block = attach_match_anomalies(
                home,
                away,
                day,
                tactical=tac if isinstance(tac, dict) else None,
                cache=store,
                write_cache=False,
            )
            block = sanitize_anomaly_block(block)
            matches = store.setdefault("matches", {})
            day_key = _match_key(day, home, away)
            matches[day_key] = block
            row["stat_anomalies"] = block
            if isinstance(pred, dict):
                pred["stat_anomalies"] = block
                row["prediction"] = pred
            if block.get("ready"):
                n_ok += 1
        except Exception as exc:
            row["stat_anomalies"] = {"ready": False, "items": [], "error": str(exc), "n": 0}
    if write_cache:
        save_anomaly_cache(store)
    return {"ok": True, "n_with_anomalies": n_ok, "n_rows": len(rows), "path": str(CACHE_PATH)}


def weekly_anomaly_table(
    rows: list[dict[str, Any]],
    *,
    days: int = 14,
    min_severity: float = MIN_SEVERITY,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Flat list per UI 'Ritardi della settimana'."""
    today = pd.Timestamp.now().normalize()
    flat: list[dict[str, Any]] = []
    for row in rows:
        day = str(row.get("date") or "")[:10]
        try:
            d = pd.Timestamp(day)
            if d < today or d > today + pd.Timedelta(days=int(days)):
                continue
        except Exception:
            continue
        block = row.get("stat_anomalies")
        if not isinstance(block, dict) or not block.get("items"):
            home = str(row.get("home") or "")
            away = str(row.get("away") or "")
            pred = row.get("prediction") if isinstance(row.get("prediction"), dict) else {}
            tac = (pred or {}).get("tactical") if isinstance(pred, dict) else None
            block = match_anomalies(home, away, day, tactical=tac if isinstance(tac, dict) else None)
        block = sanitize_anomaly_block(block if isinstance(block, dict) else None)
        for a in block.get("items") or []:
            if float(a.get("severity") or 0) < min_severity:
                continue
            flat.append(
                {
                    "date": day,
                    "time": row.get("time") or "",
                    "league": row.get("league") or "",
                    "home": row.get("home"),
                    "away": row.get("away"),
                    "kind": a.get("kind"),
                    "side": a.get("side"),
                    "team": a.get("team"),
                    "title": a.get("title"),
                    "detail": a.get("detail"),
                    "severity": a.get("severity"),
                    "source": a.get("source"),
                }
            )
    flat.sort(key=lambda r: (-float(r.get("severity") or 0), str(r.get("date") or "")))
    return flat[: int(limit)]
