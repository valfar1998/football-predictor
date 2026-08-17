"""Quote da AsianBetSoccer (endpoint botbot3.space usato dal sito)."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "asian_odds.json"

BASE = "https://botbot3.space/tables/v4"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; football-predictor/1.0)",
    "Referer": "https://www.asianbetsoccer.com/it/nextgame.html",
}

BOOKS = {
    "bet365": "12fa2eba2655cdc08a0d92fd601c498da2f49b54",
    "188bet": "6161483bb3095c88f7e154d712b5decd13c888f4",
    "avg": "f742dd97165c680c4b28d22dd56d0567189f8e3d",
    "sbobet": "a812ffd882c5e9447002429bab519de4b78002c6",
}


def _parse_js_args(chunk: str) -> list[str | float | int]:
    """Parser leggero per argomenti stile JS (stringhe, numeri)."""
    args: list[str | float | int] = []
    i = 0
    n = len(chunk)

    def skip_ws() -> None:
        nonlocal i
        while i < n and chunk[i] in " \t\n\r":
            i += 1

    while i < n:
        skip_ws()
        if i >= n:
            break
        ch = chunk[i]
        if ch == ",":
            i += 1
            continue
        if ch == "'":
            i += 1
            start = i
            while i < n and chunk[i] != "'":
                i += 1
            args.append(chunk[start:i])
            i += 1
            continue
        if ch == '"':
            i += 1
            start = i
            while i < n and chunk[i] != '"':
                i += 1
            args.append(chunk[start:i])
            i += 1
            continue
        start = i
        while i < n and chunk[i] not in ",":
            i += 1
        token = chunk[start:i].strip()
        if not token:
            continue
        try:
            if "." in token:
                args.append(float(token))
            else:
                args.append(int(token))
        except ValueError:
            args.append(token)
    return args


def _extract_calls(source: str, fn: str) -> list[list[str | float | int]]:
    calls: list[list[str | float | int]] = []
    needle = fn + "("
    pos = 0
    while True:
        idx = source.find(needle, pos)
        if idx < 0:
            break
        i = idx + len(needle)
        depth = 1
        start = i
        in_str = None
        while i < len(source) and depth > 0:
            c = source[i]
            if in_str:
                if c == in_str and source[i - 1] != "\\":
                    in_str = None
            elif c in "'\"":
                in_str = c
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            i += 1
        if depth == 0:
            inner = source[start : i - 1]
            calls.append(_parse_js_args(inner))
        pos = i
    return calls


def _fetch_js(day_offset: int, book: str = "bet365", stats: str = "Q") -> str:
    book_id = BOOKS.get(book, book)
    day_key = f"tablenext/day{day_offset}"
    url = f"{BASE}/{stats}/{day_key}/{book_id}.js?date={int(time.time() * 1000)}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=45) as resp:
        return resp.read().decode("utf-8", "replace")


def _float(val: object) -> float | None:
    if val is None:
        return None
    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    return num if num > 1.01 else None


def _num(val: object) -> float | None:
    if val is None:
        return None
    if isinstance(val, str):
        token = val.strip().replace("%", "")
        if not token or token in {"N", "U", "D", "R", "X"}:
            return None
        try:
            return float(token)
        except ValueError:
            return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _implied_pp(open_odd: float | None, curr_odd: float | None) -> float | None:
    """Variazione probabilità implicita in punti percentuali. Positivo = quota accorciata."""
    if not open_odd or not curr_odd or open_odd <= 1.01 or curr_odd <= 1.01:
        return None
    return round((1.0 / curr_odd - 1.0 / open_odd) * 100.0, 2)


def parse_payload(js: str, *, day_offset: int, book: str) -> list[dict]:
    next_rows = _extract_calls(js, "getDatanext1")
    stat_rows = _extract_calls(js, "getData2")
    stats_by_id = {str(r[4]): r for r in stat_rows if len(r) > 47}

    out: list[dict] = []
    for row in next_rows:
        if len(row) < 16:
            continue
        match_id = str(row[5])
        league = str(row[6])
        home = str(row[7])
        dt_raw = str(row[8])
        try:
            dt = datetime.fromisoformat(dt_raw.replace("Z", "+00:00"))
        except ValueError:
            dt = None
        item = {
            "match_id": match_id,
            "league": league,
            "home": home,
            "away": str(row[15]),
            "datetime": dt_raw,
            "date": dt.strftime("%Y-%m-%d") if dt else None,
            "time": dt.strftime("%H:%M") if dt else "",
            "book": book,
            "day_offset": day_offset,
            "odd_1": _float(row[9]),
            "odd_x": _float(row[10]),
            "odd_2": _float(row[11]),
            "open_1": _float(row[12]),
            "open_x": _float(row[13]),
            "open_2": _float(row[14]),
            "ah_curr": None,
            "ah_open": None,
            "ah_home_odd": None,
            "ah_home_open": None,
            "ah_away_odd": None,
            "ah_away_open": None,
            "total_line": None,
            "total_open": None,
            "odd_over": None,
            "open_over": None,
            "odd_under": None,
            "open_under": None,
            "source": "asianbetsoccer",
        }
        stats = stats_by_id.get(match_id)
        if stats and len(stats) > 47:
            item["ah_curr"] = _num(stats[5])
            item["ah_open"] = _num(stats[6])
            item["ah_home_odd"] = _float(stats[11])
            item["ah_home_open"] = _float(stats[12])
            item["ah_away_odd"] = _float(stats[16])
            item["ah_away_open"] = _float(stats[17])
            item["total_line"] = _num(stats[21])
            item["total_open"] = _num(stats[22])
            item["odd_over"] = _float(stats[24])
            item["open_over"] = _float(stats[25])
            item["odd_under"] = _float(stats[29])
            item["open_under"] = _float(stats[30])
            item["odd_1"] = _float(stats[42]) or item["odd_1"]
            item["open_1"] = _float(stats[43]) or item["open_1"]
            item["odd_x"] = _float(stats[44]) or item["odd_x"]
            item["open_x"] = _float(stats[45]) or item["open_x"]
            item["odd_2"] = _float(stats[46]) or item["odd_2"]
            item["open_2"] = _float(stats[47]) or item["open_2"]
            line_key = str(item["total_line"] or "2.5")
            item["lines"] = {
                line_key: {"over": item["odd_over"], "under": item["odd_under"]},
            }
        item["market_move"] = summarize_moves(item)
        out.append(item)
    return out


def fetch_asian_odds(*, days: int = 7, book: str = "bet365") -> list[dict]:
    rows: list[dict] = []
    for offset in range(max(0, days)):
        try:
            js = _fetch_js(offset, book=book)
        except urllib.error.URLError as exc:
            print(f"asian skip day{offset}: {exc}")
            continue
        parsed = parse_payload(js, day_offset=offset, book=book)
        print(f"asian day{offset}: {len(parsed)} partite ({book})")
        rows.extend(parsed)
    return rows


def save_asian_odds(rows: list[dict]) -> Path:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "https://www.asianbetsoccer.com/it/nextgame.html",
        "endpoint": BASE,
        "n_matches": len(rows),
        "matches": rows,
    }
    CACHE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return CACHE


def load_asian_odds() -> list[dict]:
    if not CACHE.exists():
        return []
    data = json.loads(CACHE.read_text(encoding="utf-8"))
    return data.get("matches") or []


def _norm_team(name: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", str(name).lower())
    s = re.sub(r"\b(fc|sc|ac|cf|cd|deportes|united|city)\b", " ", s)
    return " ".join(s.split())


def find_asian_odds(home: str, away: str, match_date: str | None = None) -> dict | None:
    rows = load_asian_odds()
    if not rows:
        return None
    nh, na = _norm_team(home), _norm_team(away)
    best = None
    best_score = 0
    for row in rows:
        if match_date and row.get("date") and row["date"] != match_date:
            continue
        rh, ra = _norm_team(row["home"]), _norm_team(row["away"])
        score = 0
        if nh == rh and na == ra:
            score = 100
        elif nh in rh or rh in nh:
            score += 40
        elif any(tok in rh for tok in nh.split() if len(tok) > 3):
            score += 20
        if na in ra or ra in na:
            score += 40
        elif any(tok in ra for tok in na.split() if len(tok) > 3):
            score += 20
        if score > best_score:
            best_score = score
            best = row
    return best if best_score >= 60 else None


def summarize_moves(row: dict) -> dict:
    """Steam apertura→corrente su 1X2, handicap asiatico e totale."""
    d1 = _implied_pp(row.get("open_1"), row.get("odd_1"))
    dx = _implied_pp(row.get("open_x"), row.get("odd_x"))
    d2 = _implied_pp(row.get("open_2"), row.get("odd_2"))
    do = _implied_pp(row.get("open_over"), row.get("odd_over"))
    du = _implied_pp(row.get("open_under"), row.get("odd_under"))

    steam_1x2 = None
    ranked = sorted(
        [(k, v) for k, v in (("1", d1), ("X", dx), ("2", d2)) if v is not None],
        key=lambda kv: kv[1],
        reverse=True,
    )
    if ranked and ranked[0][1] >= 0.8:
        second = ranked[1][1] if len(ranked) > 1 else -99.0
        if ranked[0][1] >= second + 0.4:
            steam_1x2 = ranked[0][0]

    ah_curr, ah_open = row.get("ah_curr"), row.get("ah_open")
    ah_delta = None
    steam_ah = None
    if ah_curr is not None and ah_open is not None:
        ah_delta = round(float(ah_curr) - float(ah_open), 3)
        if ah_delta <= -0.12:
            steam_ah = "home"
        elif ah_delta >= 0.12:
            steam_ah = "away"

    tot_curr, tot_open = row.get("total_line"), row.get("total_open")
    tot_delta = None
    steam_ou = None
    if tot_curr is not None and tot_open is not None:
        tot_delta = round(float(tot_curr) - float(tot_open), 3)
        if tot_delta >= 0.12:
            steam_ou = "over"
        elif tot_delta <= -0.12:
            steam_ou = "under"
    if steam_ou is None and do is not None and du is not None:
        if do - du >= 0.8:
            steam_ou = "over"
        elif du - do >= 0.8:
            steam_ou = "under"

    strength = 0
    if steam_1x2 and ranked and ranked[0][1] >= 2.0:
        strength += 1
    if ah_delta is not None and abs(ah_delta) >= 0.25:
        strength += 1
    if tot_delta is not None and abs(tot_delta) >= 0.25:
        strength += 1
    if (do or 0) >= 2.0 or (du or 0) >= 2.0:
        strength += 1

    note_parts: list[str] = []
    if steam_1x2:
        note_parts.append(f"1X2 verso {steam_1x2}")
    if steam_ah:
        note_parts.append(f"AH verso {'casa' if steam_ah == 'home' else 'trasferta'} ({ah_open}->{ah_curr})")
    if steam_ou:
        note_parts.append(f"totale verso {steam_ou} ({tot_open}->{tot_curr})")

    spread = _spread_score(ah_delta, tot_delta, d1, dx, d2, do, du)

    return {
        "drop_1": d1,
        "drop_x": dx,
        "drop_2": d2,
        "drop_over": do,
        "drop_under": du,
        "steam_1x2": steam_1x2,
        "steam_ah": steam_ah,
        "steam_ou": steam_ou,
        "ah_curr": ah_curr,
        "ah_open": ah_open,
        "ah_delta": ah_delta,
        "total_curr": tot_curr,
        "total_open": tot_open,
        "total_delta": tot_delta,
        "strength": strength,
        "spread_score": spread,
        "movement_level": _movement_level(spread),
        "movement_summary": _movement_summary(ah_open, ah_curr, tot_open, tot_curr, steam_1x2, steam_ou),
        "note": "; ".join(note_parts) if note_parts else "mercato stabile",
    }


def _movement_level(spread_score: float) -> str:
    if spread_score < 1:
        return "Stabile"
    if spread_score < 4:
        return "Leggero"
    if spread_score < 7:
        return "Medio"
    return "Forte"


def _movement_summary(
    ah_open: object,
    ah_curr: object,
    tot_open: object,
    tot_curr: object,
    steam_1x2: str | None,
    steam_ou: str | None,
) -> str:
    parts: list[str] = []
    if ah_open is not None and ah_curr is not None and float(ah_open) != float(ah_curr):
        parts.append(f"AH {ah_open}->{ah_curr}")
    elif steam_1x2:
        parts.append(f"1X2 verso {steam_1x2}")
    if tot_open is not None and tot_curr is not None and float(tot_open) != float(tot_curr):
        parts.append(f"Tot {tot_open}->{tot_curr}")
    elif steam_ou:
        parts.append(f"O/U verso {steam_ou}")
    return " · ".join(parts) if parts else "Quasi nessun movimento"


def _spread_score(
    ah_delta: float | None,
    tot_delta: float | None,
    d1: float | None,
    dx: float | None,
    d2: float | None,
    do: float | None,
    du: float | None,
) -> float:
    parts: list[float] = []
    if ah_delta is not None:
        parts.append(abs(float(ah_delta)) * 10.0)
    if tot_delta is not None:
        parts.append(abs(float(tot_delta)) * 10.0)
    for val in (d1, dx, d2, do, du):
        if val is not None:
            parts.append(abs(float(val)))
    return round(max(parts), 2) if parts else 0.0


def move_alignment(pick: str | None, moves: dict | None) -> dict:
    """Quanto il consiglio del modello è d'accordo con lo steam asiatico."""
    empty = {"agrees": [], "disagrees": [], "delta": 0, "label": "n/d"}
    if not pick or not moves:
        return empty
    agrees: list[str] = []
    disagrees: list[str] = []
    s1 = moves.get("steam_1x2")
    sah = moves.get("steam_ah")
    sou = moves.get("steam_ou")
    pick = str(pick)

    if s1:
        if pick == s1 or (pick in {"1X", "1 DNB"} and s1 == "1") or (pick in {"X2", "2 DNB"} and s1 == "2"):
            agrees.append(f"1X2 {s1}")
        elif pick in {"1", "X", "2", "1X", "X2", "1 DNB", "2 DNB"}:
            disagrees.append(f"1X2 {s1}")
    if sah:
        homeish = pick in {"1", "1X", "1 DNB"}
        awayish = pick in {"2", "X2", "2 DNB"}
        if (sah == "home" and homeish) or (sah == "away" and awayish):
            agrees.append("handicap")
        elif homeish or awayish:
            disagrees.append("handicap")
    if sou:
        overish = pick.startswith("O") and "GOL" not in pick
        underish = pick.startswith("U")
        if (sou == "over" and overish) or (sou == "under" and underish):
            agrees.append("totale")
        elif overish or underish:
            disagrees.append("totale")

    delta = 0
    if agrees and not disagrees:
        delta = 1 + (1 if (moves.get("strength") or 0) >= 2 else 0)
    elif disagrees and not agrees:
        delta = -1 - (1 if (moves.get("strength") or 0) >= 2 else 0)
    elif agrees and disagrees:
        delta = 0

    if delta > 0:
        label = "allineato"
    elif delta < 0:
        label = "contrario"
    elif agrees or disagrees:
        label = "misto"
    else:
        label = "stabile"
    return {"agrees": agrees, "disagrees": disagrees, "delta": delta, "label": label}


def asian_to_advisor_odds(row: dict) -> dict[str, float | None]:
    line = row.get("total_line") or 2.5
    lines = row.get("lines") or {}
    odds = {
        "1": row.get("odd_1"),
        "X": row.get("odd_x"),
        "2": row.get("odd_2"),
        "over_2.5": row.get("odd_over"),
        "under_2.5": row.get("odd_under"),
    }
    for key, spec in lines.items():
        if spec.get("over"):
            odds[f"over_{key}"] = spec["over"]
        if spec.get("under"):
            odds[f"under_{key}"] = spec["under"]
    # se la linea totale non è 2.5, mappa comunque over/under principali
    if line and line != 2.5:
        odds[f"over_{line}"] = row.get("odd_over")
        odds[f"under_{line}"] = row.get("odd_under")
    return odds
