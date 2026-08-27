"""Marcatori anytime/first: Understat/FBref xG share × λ, filtrati da lineup FotMob se presente."""

from __future__ import annotations

from math import exp
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FBREF_CACHE = ROOT / "data" / "processed" / "fbref_player_contrib.csv"
UNDERSTAT_CACHE = ROOT / "data" / "processed" / "understat_player_xg.csv"

_CACHE: dict[str, list[dict[str, Any]]] | None = None


def _norm(name: str) -> str:
    from modules.data_update.cups import _norm_key

    return _norm_key(name or "")


def _player_key(name: str) -> str:
    return _norm(name).replace(".", "").strip()


def _name_match(a: str, b: str) -> bool:
    ka, kb = _player_key(a), _player_key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    if ka in kb or kb in ka:
        return min(len(ka), len(kb)) >= 5
    # cognome
    ta, tb = ka.split(), kb.split()
    if ta and tb and ta[-1] == tb[-1] and len(ta[-1]) >= 4:
        return True
    return False


def _load_source(path: Path, source: str, *, min_xg: float) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    if not path.exists():
        return out
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        team = _norm(str(row.get("team_norm") or row.get("team") or ""))
        player = str(row.get("player") or "").strip()
        if not team or not player:
            continue
        try:
            xg = float(row.get("xg") or 0)
            share = float(row.get("share") or 0)
            minutes = float(row.get("minutes") or row.get("time") or 0)
        except (TypeError, ValueError):
            continue
        if xg < min_xg and share < 0.05:
            continue
        if share < 0.02 and xg > 0:
            share = min(0.28, xg / 35.0)
        out.setdefault(team, []).append(
            {
                "player": player,
                "xg": xg,
                "minutes": minutes,
                "share": max(0.0, min(0.45, share)),
                "source": source,
            }
        )
    for team, rows in out.items():
        rows.sort(key=lambda r: (-float(r["xg"]), r["player"]))
        out[team] = rows[:15]
    return out


def _team_players() -> dict[str, list[dict[str, Any]]]:
    """Merge Understat (priorità) + FBref per squadra."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    us = _load_source(UNDERSTAT_CACHE, "understat", min_xg=0.8)
    fb = _load_source(FBREF_CACHE, "fbref", min_xg=1.0)
    out: dict[str, list[dict[str, Any]]] = {}
    teams = set(us) | set(fb)
    for team in teams:
        by_name: dict[str, dict[str, Any]] = {}
        for pl in fb.get(team, []):
            by_name[_player_key(pl["player"])] = dict(pl)
        for pl in us.get(team, []):  # understat sovrascrive / arricchisce
            k = _player_key(pl["player"])
            if k in by_name:
                base = by_name[k]
                # media pesata share se entrambi
                base["share"] = 0.55 * float(pl["share"]) + 0.45 * float(base["share"])
                base["xg"] = max(float(pl["xg"]), float(base["xg"]))
                base["source"] = "understat+fbref"
            else:
                by_name[k] = dict(pl)
        rows = list(by_name.values())
        rows.sort(key=lambda r: -float(r["xg"]))
        out[team] = rows[:12]
    _CACHE = out
    return out


def anytime_probs(
    home: str,
    away: str,
    *,
    lambda_home: float,
    lambda_away: float,
    top_n: int = 5,
    lineup_home: list[str] | None = None,
    lineup_away: list[str] | None = None,
) -> list[dict[str, Any]]:
    """P(anytime) ≈ 1-exp(-share×λ). Se lineup FotMob: boost titolari, taglia panchina."""
    idx = _team_players()
    hk, ak = _norm(home), _norm(away)
    rows: list[dict[str, Any]] = []
    for side, team, lam, lineup in (
        ("home", hk, float(lambda_home), lineup_home),
        ("away", ak, float(lambda_away), lineup_away),
    ):
        candidates = list(idx.get(team, [])[: max(top_n + 4, 10)])
        lined = [str(x) for x in (lineup or []) if x]
        for pl in candidates:
            share = float(pl["share"] or 0)
            if share < 0.03 and float(pl["xg"] or 0) < 1.5:
                continue
            in_xi = None
            if lined:
                in_xi = any(_name_match(pl["player"], n) for n in lined)
                if in_xi is False:
                    # fuori XI: probabilità molto ridotta (potrebbe entrare)
                    share *= 0.18
                else:
                    share = min(0.42, share * 1.12)
            lam_p = max(0.02, min(1.6, share * max(0.35, lam)))
            p_any = 1.0 - exp(-lam_p)
            tot = max(0.5, float(lambda_home) + float(lambda_away))
            p_first = max(0.015, min(0.45, p_any * (lam / tot) * (0.78 if in_xi is not False else 0.35)))
            if p_any < 0.04 and in_xi is False:
                continue
            rows.append(
                {
                    "player": pl["player"],
                    "side": side,
                    "team": home if side == "home" else away,
                    "p_anytime": round(p_any, 4),
                    "p_first": round(p_first, 4),
                    "share": round(share, 4),
                    "xg_season": round(float(pl["xg"] or 0), 2),
                    "source": pl.get("source"),
                    "in_lineup": in_xi,
                }
            )
    rows.sort(key=lambda r: (-(1 if r.get("in_lineup") else 0), -float(r["p_anytime"])))
    return rows[: max(top_n * 2, 8)]
