"""Pronostici pubblici di modelli/tipster (Forebet, PredictZ, Vitibet)."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "tipsters.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,it;q=0.8",
    "Referer": "https://www.google.com/",
}

_CACHE: list[dict] | None = None
_CACHE_MTIME: float | None = None


def _norm_team(name: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", str(name).lower())
    s = re.sub(r"\b(fc|sc|ac|cf|cd|deportes|united|city|calcio|cf)\b", " ", s)
    return " ".join(s.split())


def _fetch(url: str, timeout: int = 25) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        enc = resp.headers.get_content_charset()
        if not enc:
            head = raw[:2500].decode("ascii", "ignore")
            found = re.search(r"charset=([a-zA-Z0-9_-]+)", head, re.I)
            enc = found.group(1) if found else "utf-8"
        try:
            return raw.decode(enc, "replace")
        except LookupError:
            return raw.decode("utf-8", "replace")


def _pick_from_probs(p1: float, px: float, p2: float) -> str:
    ranked = sorted((("1", p1), ("X", px), ("2", p2)), key=lambda t: t[1], reverse=True)
    return ranked[0][0]


def _parse_forebet(html: str) -> list[dict]:
    rows: list[dict] = []
    chunks = re.split(r"(?i)class=['\"]rcnt", html)
    for chunk in chunks[1:]:
        home_m = re.search(r'(?is)class="homeTeam"[^>]*>.*?itemprop="name">([^<]+)', chunk)
        away_m = re.search(r'(?is)class="awayTeam"[^>]*>.*?itemprop="name">([^<]+)', chunk)
        if not home_m:
            home_m = re.search(r'(?is)class="homeTeam"[^>]*>\s*<a[^>]*>([^<]+)', chunk)
        if not away_m:
            away_m = re.search(r'(?is)class="awayTeam"[^>]*>\s*<a[^>]*>([^<]+)', chunk)
        fprc = re.search(r"(?is)class=['\"]fprc['\"]>(.*?)</div>", chunk)
        pcts = re.findall(r"(?is)<span[^>]*>(\d{1,3})</span>", fprc.group(1)) if fprc else []
        pred_m = re.search(r'(?is)class="forepr"[^>]*>\s*<span>([1Xx2])</span>', chunk)
        if not home_m or not away_m or len(pcts) < 3:
            continue
        p1, px, p2 = int(pcts[0]) / 100.0, int(pcts[1]) / 100.0, int(pcts[2]) / 100.0
        pick = (pred_m.group(1).upper().replace("X", "X") if pred_m else _pick_from_probs(p1, px, p2))
        if pick not in {"1", "X", "2"}:
            pick = _pick_from_probs(p1, px, p2)
        rows.append(
            {
                "source": "Forebet",
                "home": home_m.group(1).strip(),
                "away": away_m.group(1).strip(),
                "pick": pick,
                "p_home": round(p1, 4),
                "p_draw": round(px, 4),
                "p_away": round(p2, 4),
            }
        )
    return rows[:300]


def _parse_predictz(html: str) -> list[dict]:
    rows: list[dict] = []
    for m in re.finditer(
        r'(?is)<tr[^>]*>\s*<td[^>]*class="[^"]*home[^"]*"[^>]*>\s*<a[^>]*>([^<]+)</a>.*?'
        r'(Home Win|Draw|Away Win|1|X|2)[^<]*</t[dh]>.*?'
        r'<td[^>]*class="[^"]*away[^"]*"[^>]*>\s*<a[^>]*>([^<]+)</a>',
        html,
    ):
        raw = m.group(2).strip()
        pick = {"Home Win": "1", "Draw": "X", "Away Win": "2", "1": "1", "X": "X", "2": "2"}.get(raw)
        if not pick:
            continue
        rows.append(
            {
                "source": "PredictZ",
                "home": m.group(1).strip(),
                "away": m.group(3).strip(),
                "pick": pick,
                "p_home": None,
                "p_draw": None,
                "p_away": None,
            }
        )
    if rows:
        return rows
    for m in re.finditer(
        r'(?is)<a[^>]+href="/predictions/[^"]+"[^>]*>([^<]+)</a>\s*</t[dh]>\s*'
        r'<td[^>]*>\s*(Home Win|Draw|Away Win)\s*</td>.*?'
        r'<a[^>]+href="/predictions/[^"]+"[^>]*>([^<]+)</a>',
        html,
    ):
        pick = {"Home Win": "1", "Draw": "X", "Away Win": "2"}[m.group(2)]
        rows.append(
            {
                "source": "PredictZ",
                "home": m.group(1).strip(),
                "away": m.group(3).strip(),
                "pick": pick,
                "p_home": None,
                "p_draw": None,
                "p_away": None,
            }
        )
    return rows[:250]


def _parse_vitibet(html: str) -> list[dict]:
    rows: list[dict] = []
    pattern = re.compile(
        r'title="Football prediction:\s*([^"]+?)\s+vs\s+([^"]+?)"[^>]*>.*?'
        r'class="standardbunkaprocenta">\s*(\d+)\s*%\s*</td>\s*'
        r'<td class="standardbunkaprocenta">\s*(\d+)\s*%\s*</td>\s*'
        r'<td class="standardbunkaprocenta">\s*(\d+)\s*%\s*</td>\s*'
        r'<td class="barvapodtipek[012]"[^>]*>\s*([012Xx1])',
        re.IGNORECASE | re.DOTALL,
    )
    for m in pattern.finditer(html):
        pick = str(m.group(6)).upper().replace("0", "X")
        if pick not in {"1", "X", "2"}:
            continue
        p1, px, p2 = int(m.group(3)) / 100.0, int(m.group(4)) / 100.0, int(m.group(5)) / 100.0
        rows.append(
            {
                "source": "Vitibet",
                "home": m.group(1).strip(),
                "away": m.group(2).strip(),
                "pick": pick,
                "p_home": round(p1, 4),
                "p_draw": round(px, 4),
                "p_away": round(p2, 4),
            }
        )
    return rows[:300]


def _safe_parse(name: str, url: str, parser) -> tuple[list[dict], str | None]:
    try:
        html = _fetch(url)
        return parser(html), None
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return [], f"{name}: {exc}"


def fetch_tipsters(*, save: bool = True) -> dict[str, Any]:
    sources: list[tuple[str, str, object]] = [
        ("Forebet", "https://www.forebet.com/en/football-tips-and-predictions-for-today", _parse_forebet),
        ("Forebet", "https://www.forebet.com/en/football-tips-and-predictions-for-tomorrow", _parse_forebet),
        ("PredictZ", "https://www.predictz.com/predictions/", _parse_predictz),
        ("Vitibet", "https://www.vitibet.com/index.php?clanek=quicktips&sekce=fotbal&lang=en&design_version=old", _parse_vitibet),
    ]
    matches: list[dict] = []
    errors: list[str] = []
    counts: dict[str, int] = {}
    seen: set[tuple[str, str, str]] = set()
    for name, url, parser in sources:
        rows, err = _safe_parse(name, url, parser)
        kept = 0
        for row in rows:
            key = (str(row.get("source")), _norm_team(row.get("home") or ""), _norm_team(row.get("away") or ""))
            if key in seen:
                continue
            seen.add(key)
            matches.append(row)
            kept += 1
        counts[name] = counts.get(name, 0) + kept
        if err:
            errors.append(err)
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n": len(matches),
        "counts": counts,
        "errors": errors,
        "matches": matches,
    }
    if save:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        global _CACHE, _CACHE_MTIME
        _CACHE, _CACHE_MTIME = matches, CACHE.stat().st_mtime
    return payload


def load_tipsters() -> list[dict]:
    global _CACHE, _CACHE_MTIME
    if not CACHE.exists():
        _CACHE, _CACHE_MTIME = [], None
        return []
    mtime = CACHE.stat().st_mtime
    if _CACHE is not None and _CACHE_MTIME == mtime:
        return _CACHE
    data = json.loads(CACHE.read_text(encoding="utf-8"))
    _CACHE = data.get("matches") or []
    _CACHE_MTIME = mtime
    return _CACHE


def load_tipsters_meta() -> dict[str, Any]:
    if not CACHE.exists():
        return {}
    data = json.loads(CACHE.read_text(encoding="utf-8"))
    data.pop("matches", None)
    return data


def _score_pair(nh: str, na: str, rh: str, ra: str) -> int:
    score = 0
    if nh == rh and na == ra:
        return 100
    if nh and (nh in rh or rh in nh):
        score += 40
    elif nh and any(tok in rh for tok in nh.split() if len(tok) > 3):
        score += 20
    if na and (na in ra or ra in na):
        score += 40
    elif na and any(tok in ra for tok in na.split() if len(tok) > 3):
        score += 20
    return score


def find_tipsters(home: str, away: str) -> list[dict]:
    rows = load_tipsters()
    if not rows:
        return []
    nh, na = _norm_team(home), _norm_team(away)
    best_by_source: dict[str, tuple[int, dict]] = {}
    for row in rows:
        score = _score_pair(nh, na, _norm_team(row.get("home") or ""), _norm_team(row.get("away") or ""))
        if score < 60:
            continue
        src = str(row.get("source") or "?")
        prev = best_by_source.get(src)
        if prev is None or score > prev[0]:
            best_by_source[src] = (score, row)
    return [item[1] for item in best_by_source.values()]


def _same_side(pick: str | None, consensus: str | None) -> bool:
    if not pick or not consensus:
        return False
    pick = str(pick)
    if pick == consensus:
        return True
    if consensus == "1" and pick in {"1", "1X", "1 DNB"}:
        return True
    if consensus == "2" and pick in {"2", "X2", "2 DNB"}:
        return True
    if consensus == "X" and pick in {"X", "1X", "X2"}:
        return True
    return False


def consensus_for(home: str, away: str) -> dict[str, Any]:
    sources = find_tipsters(home, away)
    if not sources:
        return {
            "n_sources": 0,
            "sources": [],
            "consensus": None,
            "strength": 0.0,
            "p_home": None,
            "p_draw": None,
            "p_away": None,
            "label": "n/d",
        }
    votes = Counter(str(s.get("pick")) for s in sources if s.get("pick") in {"1", "X", "2"})
    consensus, n_cons = votes.most_common(1)[0] if votes else (None, 0)
    probs = [s for s in sources if s.get("p_home") is not None]
    avg = None
    if probs:
        avg = {
            "p_home": round(sum(float(s["p_home"]) for s in probs) / len(probs), 4),
            "p_draw": round(sum(float(s["p_draw"]) for s in probs) / len(probs), 4),
            "p_away": round(sum(float(s["p_away"]) for s in probs) / len(probs), 4),
        }
    strength = round(n_cons / max(len(sources), 1), 3) if consensus else 0.0
    if not consensus:
        label = "n/d"
    elif strength >= 0.67 and len(sources) >= 2:
        label = "consenso"
    elif len(sources) == 1:
        label = "singolo"
    else:
        label = "divisi"
    out = {
        "n_sources": len(sources),
        "sources": [
            {
                "source": s.get("source"),
                "pick": s.get("pick"),
                "home": s.get("home"),
                "away": s.get("away"),
                "p_home": s.get("p_home"),
                "p_draw": s.get("p_draw"),
                "p_away": s.get("p_away"),
            }
            for s in sources
        ],
        "consensus": consensus,
        "strength": strength,
        "label": label,
        "p_home": None if not avg else avg["p_home"],
        "p_draw": None if not avg else avg["p_draw"],
        "p_away": None if not avg else avg["p_away"],
    }
    return out


def apply_tipster_balance(play: dict[str, Any], tipster: dict[str, Any] | None) -> dict[str, Any]:
    """I tipster non entrano nell'EV: solo aggiustamento lieve del voto."""
    out = dict(play)
    if not tipster or not tipster.get("n_sources"):
        out["tipster"] = tipster or {"n_sources": 0, "label": "n/d"}
        out["tipster_delta"] = 0
        return out
    pick = str(out.get("code") or "")
    consensus = tipster.get("consensus")
    n = int(tipster.get("n_sources") or 0)
    strength = float(tipster.get("strength") or 0)
    delta = 0
    if consensus and _same_side(pick, consensus):
        if n >= 2 and strength >= 0.67:
            delta = 1
        elif n >= 1:
            delta = 0
        agree = "allineati"
    elif consensus:
        if n >= 2 and strength >= 0.67:
            delta = -1
        agree = "contrari"
    else:
        agree = "divisi"
    score = int(out.get("score") or 1)
    out["score"] = int(max(1, min(10, score + delta)))
    out["tipster"] = {**tipster, "agree": agree}
    out["tipster_delta"] = delta
    return out
