"""Cluster di leghe: regole keyword + routing per similarità statistica."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "data" / "models" / "league_stat_profiles.json"

CLUSTER_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("big5_eng", ("premier league", "epl", "england premier")),
    ("big5_esp", ("la liga", "primera", "laliga")),
    ("big5_ita", ("serie a",)),
    ("big5_ger", ("bundesliga",)),
    ("big5_fra", ("ligue 1", "ligue1")),
    ("serie_b_like", ("serie b", "championship", "2. bundesliga", "liga portugal", "eredivisie", "liga mx")),
    ("latam", ("brasileirao", "serie a brazil", "argentina", "liga profesional", "copa libertadores", "sudamericana")),
    ("mls", ("major league soccer", "mls")),
    ("cups_euro", ("champions league", "europa league", "conference league", "uefa")),
]

MIN_ROWS_CLUSTER = 800
GLOBAL = "global"


def _blob(league: str | None, country: str | None = None) -> str:
    return f"{league or ''} {country or ''}".strip().lower()


def cluster_for_keyword(league: str | None, country: str | None = None) -> str:
    blob = _blob(league, country)
    if not blob:
        return GLOBAL
    for cid, keys in CLUSTER_RULES:
        if any(k in blob for k in keys):
            return cid
    return GLOBAL


def build_stat_profiles(feat) -> dict[str, Any]:
    """Profilo per lega: draw rate, home win, goal mean/var, xG mean — centroidi cluster."""
    import numpy as np
    import pandas as pd

    df = feat.copy()
    if "league" not in df.columns:
        return {"ok": False, "error": "no league col"}
    df["result"] = df["result"].astype(str)
    df["tg"] = pd.to_numeric(df.get("home_goals"), errors="coerce").fillna(0) + pd.to_numeric(
        df.get("away_goals"), errors="coerce"
    ).fillna(0)
    rows = []
    for lg, g in df.groupby(df["league"].astype(str)):
        if len(g) < 80:
            continue
        hw = float((g["result"] == "H").mean())
        dr = float((g["result"] == "D").mean())
        gm = float(g["tg"].mean())
        gv = float(g["tg"].var() or 0)
        xg = None
        if "home_xg_avg" in g.columns:
            xg = float(
                (
                    pd.to_numeric(g["home_xg_avg"], errors="coerce").fillna(0)
                    + pd.to_numeric(g["away_xg_avg"], errors="coerce").fillna(0)
                ).mean()
            )
        rows.append(
            {
                "league": lg,
                "n": int(len(g)),
                "home_win": round(hw, 4),
                "draw": round(dr, 4),
                "goals_mean": round(gm, 4),
                "goals_var": round(gv, 4),
                "xg_mean": None if xg is None else round(xg, 4),
                "keyword_cluster": cluster_for_keyword(lg),
            }
        )
    if not rows:
        return {"ok": False, "error": "no profiles"}

    # centroidi keyword cluster
    centroids: dict[str, dict[str, float]] = {}
    for cid in {r["keyword_cluster"] for r in rows if r["keyword_cluster"] != GLOBAL}:
        members = [r for r in rows if r["keyword_cluster"] == cid]
        if len(members) < 1:
            continue
        centroids[cid] = {
            "home_win": float(np.mean([m["home_win"] for m in members])),
            "draw": float(np.mean([m["draw"] for m in members])),
            "goals_mean": float(np.mean([m["goals_mean"] for m in members])),
            "goals_var": float(np.mean([m["goals_var"] for m in members])),
            "xg_mean": float(np.mean([m["xg_mean"] or m["goals_mean"] for m in members])),
        }

    # assegna similarità alle leghe global/unknown
    assignments = {}
    for r in rows:
        if r["keyword_cluster"] != GLOBAL:
            assignments[r["league"]] = r["keyword_cluster"]
            continue
        best, best_d = GLOBAL, 1e9
        vec = np.array(
            [
                r["home_win"],
                r["draw"],
                r["goals_mean"] / 4.0,
                min(r["goals_var"], 8) / 8.0,
                (r["xg_mean"] or r["goals_mean"]) / 4.0,
            ]
        )
        for cid, c in centroids.items():
            cv = np.array(
                [
                    c["home_win"],
                    c["draw"],
                    c["goals_mean"] / 4.0,
                    min(c["goals_var"], 8) / 8.0,
                    c["xg_mean"] / 4.0,
                ]
            )
            d = float(np.linalg.norm(vec - cv))
            if d < best_d:
                best_d, best = d, cid
        assignments[r["league"]] = best if best_d < 0.35 else GLOBAL

    payload = {
        "ok": True,
        "n_leagues": len(rows),
        "profiles": rows,
        "centroids": centroids,
        "assignments": assignments,
    }
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_stat_profiles() -> dict[str, Any] | None:
    if not PROFILE_PATH.exists():
        return None
    try:
        data = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        return data if data.get("ok") else None
    except Exception:
        return None


def cluster_for(league: str | None, country: str | None = None) -> str:
    """Keyword prima; se global, prova assignment da similarità statistica."""
    kw = cluster_for_keyword(league, country)
    if kw != GLOBAL:
        return kw
    prof = load_stat_profiles()
    if not prof:
        return GLOBAL
    lg = str(league or "")
    hit = (prof.get("assignments") or {}).get(lg)
    if hit:
        return str(hit)
    # fuzzy: match case-insensitive
    for k, v in (prof.get("assignments") or {}).items():
        if k.lower() == lg.lower():
            return str(v)
    return GLOBAL


def assign_clusters(leagues: list[str] | Any, countries: list[str] | None = None) -> list[str]:
    countries = countries or [None] * len(leagues)
    return [cluster_for(lg, c) for lg, c in zip(leagues, countries)]


def clusters_with_enough_rows(feat_league_col, *, min_rows: int = MIN_ROWS_CLUSTER) -> dict[str, int]:
    from collections import Counter

    counts: Counter[str] = Counter()
    for lg in feat_league_col.astype(str).tolist():
        counts[cluster_for(lg)] += 1
    out = {GLOBAL: int(sum(counts.values()))}
    for cid, n in counts.items():
        if cid != GLOBAL and n >= min_rows:
            out[cid] = int(n)
    return out
