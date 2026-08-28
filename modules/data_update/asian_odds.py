"""Quote da AsianBetSoccer (endpoint botbot3.space usato dal sito)."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from modules.progress_report import emit

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "asian_odds.json"

BASE = "https://botbot3.space/tables/v4"
PAGE = "https://www.asianbetsoccer.com/it/nextgame.html"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.asianbetsoccer.com/it/nextgame.html",
    "Accept": "*/*",
    "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
}

# Hash ruotati dal sito; se 404, fetch_asian_odds li rilege da nextgame.html.
BOOKS = {
    "bet365": "60e3327f197d791424608bd18905b74d09c8247d",
    "188bet": "f85973abdc9aebfd6dab7dd995a4b1a830b6deba",
    "avg": "ed3b0bd5a48a2f9801674f8e7efd7fc8640be069",
    "sbobet": "a232e19a17cec69d03dac9024a5ddc624c411f37",
}
_BOOK_LABELS = {
    "bet365": "bet365",
    "188bet": "188bet",
    "avgodds": "avg",
    "sbobet": "sbobet",
}

MOVE_LEVELS = ("Stabile", "Leggero", "Medio", "Forte", "Fortissimo", "Raro")
MOVE_RANK = {name: i for i, name in enumerate(MOVE_LEVELS)}
MOVE_FILTER_OPTIONS = [
    "Tutti",
    "Leggero+",
    "Medio+",
    "Forte+",
    "Fortissimo+ (0.5/0.75)",
    "Raro (>=1)",
]
MOVE_FILTER_RANK = {
    "Tutti": 0,
    "Leggero+": 1,
    "Medio+": 2,
    "Forte+": 3,
    "Forte": 3,
    "Fortissimo+ (0.5/0.75)": 4,
    "Fortissimo+": 4,
    "Raro (>=1)": 5,
    "Raro": 5,
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


def _http_get(url: str, timeout: int = 45) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def refresh_book_ids() -> dict[str, str]:
    """Rilegge gli hash bookmaker dalla pagina, perché AsianBetSoccer li ruota."""
    html = _http_get(PAGE, timeout=30)
    found: dict[str, str] = {}
    for val, label in re.findall(
        r'<option[^>]*value=["\']([a-f0-9]{40})["\'][^>]*>([^<]+)',
        html,
        flags=re.I,
    ):
        key = _BOOK_LABELS.get(re.sub(r"\s+", "", label).strip().lower())
        if key:
            found[key] = val
    if found:
        BOOKS.update(found)
        print("asian book hash:", ", ".join(f"{k}={v[:8]}…" for k, v in sorted(found.items())))
    return found


def _fetch_js(day_offset: int, book: str = "bet365", stats: str = "Q") -> str:
    book_id = BOOKS.get(book, book)
    day_key = f"tablenext/day{day_offset}"
    url = f"{BASE}/{stats}/{day_key}/{book_id}.js?date={int(time.time() * 1000)}"
    return _http_get(url, timeout=45)


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


def fetch_asian_odds(*, days: int = 14, book: str = "bet365", on_progress=None) -> list[dict]:
    rows: list[dict] = []
    emit(on_progress, 0.02, "AsianBetSoccer: hash book…")
    try:
        refresh_book_ids()
        refreshed = True
    except Exception as exc:
        print(f"asian hash skip: {exc}", flush=True)
        refreshed = False
    n_days = max(1, days)
    for offset in range(max(0, days)):
        emit(on_progress, (offset + 0.3) / n_days, f"Asian giorno {offset + 1}/{days}…")
        try:
            js = _fetch_js(offset, book=book)
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and not refreshed:
                try:
                    refresh_book_ids()
                    refreshed = True
                    js = _fetch_js(offset, book=book)
                except Exception as retry_exc:
                    print(f"asian skip day{offset}: HTTP {exc.code} ({retry_exc})", flush=True)
                    continue
            else:
                print(f"asian skip day{offset}: HTTP {exc.code}", flush=True)
                continue
        except urllib.error.URLError as exc:
            print(f"asian skip day{offset}: {exc}", flush=True)
            continue
        parsed = parse_payload(js, day_offset=offset, book=book)
        print(f"asian day{offset}: {len(parsed)} partite ({book})", flush=True)
        rows.extend(parsed)
    emit(on_progress, 1.0, f"Asian OK · {len(rows)} partite")
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
    global _CACHE_ROWS, _CACHE_MTIME
    _CACHE_ROWS, _CACHE_MTIME = rows, CACHE.stat().st_mtime
    return CACHE


_CACHE_ROWS: list[dict] | None = None
_CACHE_MTIME: float | None = None
_ASIAN_BY_DATE: dict[str, list[dict]] | None = None


def load_asian_odds() -> list[dict]:
    global _CACHE_ROWS, _CACHE_MTIME, _ASIAN_BY_DATE
    if not CACHE.exists():
        _CACHE_ROWS, _CACHE_MTIME, _ASIAN_BY_DATE = [], None, {}
        return []
    mtime = CACHE.stat().st_mtime
    if _CACHE_ROWS is not None and _CACHE_MTIME == mtime:
        return _CACHE_ROWS
    data = json.loads(CACHE.read_text(encoding="utf-8"))
    _CACHE_ROWS = data.get("matches") or []
    _CACHE_MTIME = mtime
    by_date: dict[str, list[dict]] = {}
    for row in _CACHE_ROWS:
        day = str(row.get("date") or "")
        by_date.setdefault(day, []).append(row)
    _ASIAN_BY_DATE = by_date
    return _CACHE_ROWS


def _norm_team(name: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", str(name).lower())
    s = re.sub(r"\b(fc|sc|ac|cf|cd|deportes|united|city)\b", " ", s)
    return " ".join(s.split())


def find_asian_odds(home: str, away: str, match_date: str | None = None) -> dict | None:
    rows = load_asian_odds()
    if not rows:
        return None
    if match_date and _ASIAN_BY_DATE and match_date in _ASIAN_BY_DATE:
        candidates = _ASIAN_BY_DATE[match_date]
    else:
        candidates = rows
    nh, na = _norm_team(home), _norm_team(away)
    best = None
    best_score = 0
    for row in candidates:
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


_SIDE_1X2 = {"1": "casa", "X": "pareggio", "2": "trasferta"}
_SIDE_ART = {"1": "sulla casa", "X": "sul pareggio", "2": "sulla trasferta"}


def _fmt_odd(val: object) -> str:
    if val is None:
        return "—"
    try:
        return f"{float(val):.2f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_line(val: object) -> str:
    if val is None:
        return "—"
    num = float(val)
    if abs(num - round(num)) < 1e-9:
        return str(int(round(num)))
    text = f"{num:.2f}".rstrip("0").rstrip(".")
    return text


def _fmt_pp(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val:+.1f} pp"


def _quota_read(drop: float | None) -> str:
    if drop is None:
        return "n/d"
    if drop >= 1.5:
        return "accorciata (soldi sopra)"
    if drop >= 0.4:
        return "accorciata"
    if drop <= -1.5:
        return "allungata (soldi altrove)"
    if drop <= -0.4:
        return "allungata"
    return "stabile"


def _quota_phrase(label: str, open_odd: object, curr_odd: object, drop: float | None, *, min_pp: float = 0.35) -> str | None:
    if open_odd is None or curr_odd is None or drop is None:
        return None
    moved = abs(float(curr_odd) - float(open_odd)) >= 0.015 or abs(float(drop)) >= min_pp
    if not moved:
        return None
    if drop > 0.15:
        verb = "accorciata"
    elif drop < -0.15:
        verb = "allungata"
    else:
        verb = "quasi ferma"
    return f"{label} {verb} {_fmt_odd(open_odd)}->{_fmt_odd(curr_odd)} ({_fmt_pp(drop)})"


def _odds_move_row(market: str, open_odd: object, curr_odd: object, drop: float | None) -> dict:
    delta_odd = None
    if open_odd is not None and curr_odd is not None:
        delta_odd = round(float(curr_odd) - float(open_odd), 3)
    return {
        "market": market,
        "open": open_odd,
        "current": curr_odd,
        "delta_odd": delta_odd,
        "delta_pp": drop,
        "read": _quota_read(drop),
    }


def summarize_moves(row: dict) -> dict:
    """Steam apertura→corrente su 1X2, handicap asiatico e totale, con commento analitico."""
    d1 = _implied_pp(row.get("open_1"), row.get("odd_1"))
    dx = _implied_pp(row.get("open_x"), row.get("odd_x"))
    d2 = _implied_pp(row.get("open_2"), row.get("odd_2"))
    do = _implied_pp(row.get("open_over"), row.get("odd_over"))
    du = _implied_pp(row.get("open_under"), row.get("odd_under"))
    dah_h = _implied_pp(row.get("ah_home_open"), row.get("ah_home_odd"))
    dah_a = _implied_pp(row.get("ah_away_open"), row.get("ah_away_odd"))

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

    spread = _spread_score(ah_delta, tot_delta, d1, dx, d2, do, du)
    line_move = _line_move(ah_delta, tot_delta)
    level = _movement_level(spread, ah_delta=ah_delta, tot_delta=tot_delta)
    summary = _movement_summary(
        row,
        steam_1x2=steam_1x2,
        steam_ah=steam_ah,
        steam_ou=steam_ou,
        d1=d1,
        dx=dx,
        d2=d2,
        do=do,
        du=du,
        dah_h=dah_h,
        dah_a=dah_a,
    )
    comment = _movement_comment(
        row,
        level=level,
        steam_1x2=steam_1x2,
        steam_ah=steam_ah,
        steam_ou=steam_ou,
        ah_delta=ah_delta,
        tot_delta=tot_delta,
        d1=d1,
        dx=dx,
        d2=d2,
        do=do,
        du=du,
        dah_h=dah_h,
        dah_a=dah_a,
    )
    tot_label = _fmt_line(tot_curr) if tot_curr is not None else "2.5"
    ah_label = _fmt_line(ah_curr) if ah_curr is not None else ""
    odds_moves = [
        _odds_move_row("1", row.get("open_1"), row.get("odd_1"), d1),
        _odds_move_row("X", row.get("open_x"), row.get("odd_x"), dx),
        _odds_move_row("2", row.get("open_2"), row.get("odd_2"), d2),
        _odds_move_row(f"Over {tot_label}", row.get("open_over"), row.get("odd_over"), do),
        _odds_move_row(f"Under {tot_label}", row.get("open_under"), row.get("odd_under"), du),
    ]
    if row.get("ah_home_odd") or row.get("ah_home_open"):
        odds_moves.append(_odds_move_row(f"AH casa {ah_label}", row.get("ah_home_open"), row.get("ah_home_odd"), dah_h))
        odds_moves.append(_odds_move_row(f"AH trasferta {ah_label}", row.get("ah_away_open"), row.get("ah_away_odd"), dah_a))

    return {
        "drop_1": d1,
        "drop_x": dx,
        "drop_2": d2,
        "drop_over": do,
        "drop_under": du,
        "drop_ah_home": dah_h,
        "drop_ah_away": dah_a,
        "steam_1x2": steam_1x2,
        "steam_1x2_label": _SIDE_1X2.get(steam_1x2 or "", steam_1x2),
        "steam_ah": steam_ah,
        "steam_ou": steam_ou,
        "ah_curr": ah_curr,
        "ah_open": ah_open,
        "ah_delta": ah_delta,
        "total_curr": tot_curr,
        "total_open": tot_open,
        "total_delta": tot_delta,
        "open_1": row.get("open_1"),
        "odd_1": row.get("odd_1"),
        "open_x": row.get("open_x"),
        "odd_x": row.get("odd_x"),
        "open_2": row.get("open_2"),
        "odd_2": row.get("odd_2"),
        "open_over": row.get("open_over"),
        "odd_over": row.get("odd_over"),
        "open_under": row.get("open_under"),
        "odd_under": row.get("odd_under"),
        "strength": strength,
        "spread_score": spread,
        "line_move": line_move,
        "movement_level": level,
        "movement_summary": summary,
        "movement_comment": comment,
        "odds_moves": odds_moves,
        "note": comment,
    }


def _line_move(ah_delta: float | None, tot_delta: float | None) -> float:
    parts: list[float] = []
    if ah_delta is not None:
        parts.append(abs(float(ah_delta)))
    if tot_delta is not None:
        parts.append(abs(float(tot_delta)))
    return round(max(parts), 3) if parts else 0.0


def _movement_level(
    spread_score: float,
    ah_delta: float | None = None,
    tot_delta: float | None = None,
) -> str:
    """Livelli su linea AH/totale: Fortissimo = 0.5/0.75, Raro = >=1."""
    line = _line_move(ah_delta, tot_delta)
    if line >= 1.0:
        return "Raro"
    if line >= 0.5:
        return "Fortissimo"
    if spread_score < 1:
        return "Stabile"
    if spread_score < 4:
        return "Leggero"
    if spread_score < 7:
        return "Medio"
    return "Forte"


def _movement_summary(
    row: dict,
    *,
    steam_1x2: str | None,
    steam_ah: str | None,
    steam_ou: str | None,
    d1: float | None,
    dx: float | None,
    d2: float | None,
    do: float | None,
    du: float | None,
    dah_h: float | None,
    dah_a: float | None,
) -> str:
    """Riassunto compatto con quote e punti percentuali, per tabelle."""
    parts: list[str] = []
    side_map = {
        "1": (row.get("open_1"), row.get("odd_1"), d1),
        "X": (row.get("open_x"), row.get("odd_x"), dx),
        "2": (row.get("open_2"), row.get("odd_2"), d2),
    }
    if steam_1x2 and steam_1x2 in side_map:
        open_o, curr_o, drop = side_map[steam_1x2]
        parts.append(
            f"1X2 verso {_SIDE_1X2[steam_1x2]} {_fmt_odd(open_o)}->{_fmt_odd(curr_o)} ({_fmt_pp(drop)})"
        )
    else:
        for code, (open_o, curr_o, drop) in side_map.items():
            phrase = _quota_phrase(code, open_o, curr_o, drop, min_pp=0.8)
            if phrase:
                parts.append(phrase)

    ah_open, ah_curr = row.get("ah_open"), row.get("ah_curr")
    if ah_open is not None and ah_curr is not None and abs(float(ah_curr) - float(ah_open)) >= 0.01:
        dest = "casa" if steam_ah == "home" else "trasferta" if steam_ah == "away" else "linea"
        parts.append(f"AH {_fmt_line(ah_open)}->{_fmt_line(ah_curr)} (verso {dest})")
    else:
        ah_p = _quota_phrase("AH casa", row.get("ah_home_open"), row.get("ah_home_odd"), dah_h, min_pp=0.8)
        if ah_p:
            parts.append(ah_p)

    tot_open, tot_curr = row.get("total_open"), row.get("total_line")
    if tot_open is not None and tot_curr is not None and abs(float(tot_curr) - float(tot_open)) >= 0.01:
        dest = steam_ou or "linea"
        parts.append(f"Tot {_fmt_line(tot_open)}->{_fmt_line(tot_curr)} (verso {dest})")
    elif steam_ou == "over":
        phrase = _quota_phrase("Over", row.get("open_over"), row.get("odd_over"), do, min_pp=0.5)
        parts.append(phrase or "O/U verso over")
    elif steam_ou == "under":
        phrase = _quota_phrase("Under", row.get("open_under"), row.get("odd_under"), du, min_pp=0.5)
        parts.append(phrase or "O/U verso under")
    return " · ".join(parts) if parts else "Quasi nessun movimento"


def _movement_comment(
    row: dict,
    *,
    level: str,
    steam_1x2: str | None,
    steam_ah: str | None,
    steam_ou: str | None,
    ah_delta: float | None,
    tot_delta: float | None,
    d1: float | None,
    dx: float | None,
    d2: float | None,
    do: float | None,
    du: float | None,
    dah_h: float | None,
    dah_a: float | None,
) -> str:
    """Commento esteso per analizzare flusso di denaro e variazioni di quota."""
    line = _line_move(ah_delta, tot_delta)
    if level == "Stabile":
        lead = "Mercato sostanzialmente fermo dall'apertura."
    elif level == "Raro":
        lead = f"Movimento raro dall'apertura: linea AH/totale spostata di {line:g} gol (>=1)."
    elif level == "Fortissimo":
        lead = f"Movimento fortissimo dall'apertura: linea AH/totale spostata di {line:g} (0.5/0.75)."
    else:
        lead = f"Movimento {level.lower()} dall'apertura."

    sentences: list[str] = [lead]
    one_x_two = [
        p
        for p in (
            _quota_phrase("1", row.get("open_1"), row.get("odd_1"), d1),
            _quota_phrase("X", row.get("open_x"), row.get("odd_x"), dx),
            _quota_phrase("2", row.get("open_2"), row.get("odd_2"), d2),
        )
        if p
    ]
    if steam_1x2:
        sentences.append(
            f"1X2: soldi {_SIDE_ART[steam_1x2]}. " + "; ".join(one_x_two) + "."
            if one_x_two
            else f"1X2: flusso verso {_SIDE_1X2[steam_1x2]}."
        )
    elif one_x_two:
        sentences.append("1X2: " + "; ".join(one_x_two) + ".")
    elif row.get("odd_1") and row.get("open_1"):
        sentences.append(
            "1X2: quote 1/X/2 quasi invariate "
            f"({_fmt_odd(row.get('open_1'))}/{_fmt_odd(row.get('open_x'))}/{_fmt_odd(row.get('open_2'))} "
            f"-> {_fmt_odd(row.get('odd_1'))}/{_fmt_odd(row.get('odd_x'))}/{_fmt_odd(row.get('odd_2'))})."
        )

    ah_open, ah_curr = row.get("ah_open"), row.get("ah_curr")
    if ah_open is not None and ah_curr is not None:
        if steam_ah == "home":
            meaning = "linea più a favore della casa (soldi sulla 1)"
        elif steam_ah == "away":
            meaning = "linea più a favore della trasferta (soldi sulla 2)"
        else:
            meaning = "linea invariata"
        ah_bits = [f"AH {_fmt_line(ah_open)}->{_fmt_line(ah_curr)}: {meaning}"]
        ah_odds = [
            p
            for p in (
                _quota_phrase("quota AH casa", row.get("ah_home_open"), row.get("ah_home_odd"), dah_h),
                _quota_phrase("quota AH trasferta", row.get("ah_away_open"), row.get("ah_away_odd"), dah_a),
            )
            if p
        ]
        if ah_odds:
            extra = "; ".join(ah_odds)
            if steam_ah is None:
                if (dah_h or 0) - (dah_a or 0) >= 0.8:
                    extra += " -> soldi sul lato casa a linea ferma"
                elif (dah_a or 0) - (dah_h or 0) >= 0.8:
                    extra += " -> soldi sul lato trasferta a linea ferma"
            ah_bits.append(extra)
        sentences.append(". ".join(ah_bits) + ".")

    tot_open, tot_curr = row.get("total_open"), row.get("total_line")
    ou_odds = [
        p
        for p in (
            _quota_phrase("Over", row.get("open_over"), row.get("odd_over"), do),
            _quota_phrase("Under", row.get("open_under"), row.get("odd_under"), du),
        )
        if p
    ]
    if tot_open is not None and tot_curr is not None:
        if tot_delta and tot_delta > 0 and (du or 0) > (do or 0) + 0.8:
            meaning = "linea alzata, ma sulla nuova linea le quote restano sull'under (transizione)"
        elif tot_delta and tot_delta < 0 and (do or 0) > (du or 0) + 0.8:
            meaning = "linea abbassata, ma sulla nuova linea le quote restano sull'over (transizione)"
        elif steam_ou == "over" and tot_delta and tot_delta > 0:
            meaning = "linea alzata, denaro sull'over"
        elif steam_ou == "under" and tot_delta and tot_delta < 0:
            meaning = "linea abbassata, denaro sull'under"
        elif steam_ou == "over":
            meaning = "linea ferma ma soldi sull'over (quota accorciata)"
        elif steam_ou == "under":
            meaning = "linea ferma ma soldi sull'under (quota accorciata)"
        else:
            meaning = "linea e quote O/U poco mosse"
        tot_txt = f"Totale {_fmt_line(tot_open)}->{_fmt_line(tot_curr)}: {meaning}"
        if ou_odds:
            tot_txt += " | " + "; ".join(ou_odds)
        sentences.append(tot_txt + ".")
    elif ou_odds:
        sentences.append("O/U: " + "; ".join(ou_odds) + ".")

    return " ".join(sentences)


def _pick_one(src: dict, *keys: str) -> object:
    for key in keys:
        if src.get(key) is not None and src.get(key) != "":
            return src[key]
    return None


def spread_playability(row: dict | None, move: dict | None = None) -> dict:
    """1–10: conviene seguire lo steam raro? 8+ gioca, 6–7 valuta, sotto 6 no."""
    row = row or {}
    merged: dict = {}
    nested = row.get("market_move")
    if isinstance(nested, dict):
        merged.update(nested)
    if isinstance(move, dict):
        merged.update(move)
    merged.update({k: v for k, v in row.items() if k != "market_move" and v is not None})

    def g(*keys: str) -> object:
        return _pick_one(merged, *keys)

    line = abs(_num(g("line_move", "line_move")) or 0.0)
    s1 = str(g("steam_1x2", "steam_1x2") or "").strip()
    sah = str(g("steam_ah", "steam_ah") or "").strip().lower()
    sou = str(g("steam_ou", "steam_ou") or "").strip().lower()
    d1, d2 = _num(g("drop_1")), _num(g("drop_2"))
    dx = _num(g("drop_x"))
    do, du = _num(g("drop_over")), _num(g("drop_under"))
    pick = str(g("pick", "code") or "").strip()
    action = str(g("action") or "").strip().lower()
    align_raw = g("market_align", "market_align")
    if isinstance(align_raw, dict):
        align_lbl = str(align_raw.get("label") or "").strip().lower()
    else:
        align_lbl = str(align_raw or "").strip().lower()

    steam_home = sah == "home" or s1 == "1"
    steam_away = sah == "away" or s1 == "2"
    aligned_1x2_ah = (sah == "home" and s1 == "1") or (sah == "away" and s1 == "2")
    conflict_1x2_ah = (sah == "home" and s1 == "2") or (sah == "away" and s1 == "1")

    steam_drop = d2 if s1 == "2" else d1 if s1 == "1" else None
    drop_abs = abs(steam_drop) if steam_drop is not None else None
    if drop_abs is None:
        drops = [abs(v) for v in (d1, d2, dx, do, du) if v is not None]
        drop_abs = max(drops) if drops else 0.0
    # drop è in punti percentuali; se arriva 0.20 (=20 pp) normalizza
    if drop_abs <= 1.5:
        drop_abs *= 100.0

    score = 4.0
    if line >= 2.0:
        score += 2.5
    elif line >= 1.5:
        score += 2.0
    elif line >= 1.25:
        score += 1.5
    elif line >= 1.0:
        score += 1.0

    if aligned_1x2_ah:
        score += 2.5
    elif conflict_1x2_ah:
        score -= 2.0
    elif sah or s1:
        score += 1.0

    if drop_abs >= 15:
        score += 2.5
    elif drop_abs >= 8:
        score += 1.8
    elif drop_abs >= 4:
        score += 1.0
    else:
        score += 0.3

    pick_home = pick in {"1", "1X", "1 DNB"}
    pick_away = pick in {"2", "X2", "2 DNB"}
    model_agrees = (pick_home and steam_home) or (pick_away and steam_away)
    model_disagrees = (pick_home and steam_away) or (pick_away and steam_home)
    if align_lbl == "allineato" or model_agrees:
        score += 1.5
        if action == "gioca":
            score += 0.5
    elif align_lbl == "contrario" or model_disagrees:
        score -= 2.0
        if action == "gioca":
            score -= 1.0
        score = min(score, 6.0)

    if action in {"no_bet", "invalido", "n/d"} and not model_agrees:
        score = min(score, 6.5)
    has_model = bool(pick) and action not in {"", "n/d", "invalido"}
    if not has_model:
        score = min(score, 7.0)

    score_i = int(max(1, min(10, round(score))))

    follow_parts: list[str] = []
    if aligned_1x2_ah:
        follow_parts.append("2 e AH trasferta" if steam_away else "1 e AH casa")
    elif s1 == "2":
        follow_parts.append("2")
    elif s1 == "1":
        follow_parts.append("1")
    elif sah == "away":
        follow_parts.append("AH trasferta")
    elif sah == "home":
        follow_parts.append("AH casa")
    if sou == "over":
        follow_parts.append("Over")
    elif sou == "under":
        follow_parts.append("Under")
    follow = " e ".join(follow_parts) if follow_parts else "nessun lato chiaro"

    why: list[str] = []
    if aligned_1x2_ah:
        why.append("1X2 e handicap allineati")
    elif conflict_1x2_ah:
        why.append("steam misto (AH vs 1X2)")
    if drop_abs >= 8:
        why.append(f"quota steam accorciata di {drop_abs:.0f} pp")
    if model_agrees:
        why.append("modello d'accordo")
    elif model_disagrees:
        why.append("modello contrario")
    if action == "gioca" and model_agrees:
        why.append("azione GIOCA")
    elif action in {"no_bet", "invalido"}:
        why.append("modello no-bet")
    if not has_model:
        why.append("senza copertura modello")

    if score_i >= 8:
        verdict = "GIOCA"
        verdict_long = "GIOCA — segui lo steam"
    elif score_i >= 6:
        verdict = "Valuta"
        verdict_long = "Valuta — segnale ok, non automatico"
    elif score_i >= 4:
        verdict = "Meglio no"
        verdict_long = "Meglio no — raro ma sporco o tardi"
    else:
        verdict = "Non giocare"
        verdict_long = "Non giocare — steam misto o contrario al modello"

    reason = f"Segui: {follow}"
    if why:
        reason += f" ({'; '.join(why)})"
    return {
        "score": score_i,
        "verdict": verdict,
        "verdict_long": verdict_long,
        "follow": follow,
        "reason": reason,
    }


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

    # Scala sul movimento: stabile 0 · leggero ±0.5 · medio ±1 · forte ±2 · fortissimo/raro ±3
    lvl = str(moves.get("movement_level") or "Stabile")
    mag = {
        "Stabile": 0.0,
        "Leggero": 0.5,
        "Medio": 1.0,
        "Forte": 2.0,
        "Fortissimo": 3.0,
        "Raro": 3.0,
    }.get(lvl)
    if mag is None:
        # fallback su strength legacy se il livello non è riconosciuto
        strength = int(moves.get("strength") or 0)
        mag = 0.0 if strength <= 0 else 0.5 if strength == 1 else 1.0 if strength == 2 else 2.0

    delta = 0.0
    if agrees and not disagrees:
        delta = mag
    elif disagrees and not agrees:
        delta = -mag
    elif agrees and disagrees:
        delta = 0.0

    if delta > 0:
        label = "allineato"
    elif delta < 0:
        label = "contrario"
    elif agrees or disagrees:
        label = "misto"
    else:
        label = "stabile"
    return {"agrees": agrees, "disagrees": disagrees, "delta": delta, "label": label, "magnitude": mag}


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
