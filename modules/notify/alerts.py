"""Avvisi Telegram: GIOCA (voto ≥8 + action gioca), da guardare (voto ≥8 + no bet), spread Raro (giocabilità >8)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from modules.data_update.asian_odds import load_asian_odds, spread_playability
from modules.notify.telegram import load_credentials, send_message, telegram_status

ROOT = Path(__file__).resolve().parents[2]
UPCOMING = ROOT / "data" / "processed" / "upcoming_predictions.json"
SENT = ROOT / "data" / "processed" / "telegram_alerts_sent.json"
MIN_UNIFIED = 8
RARE_LINE = 1.0
MIN_SPREAD_PLAYABILITY = 9  # solo giocabilità > 8
KEEP_DAYS = 21
CHUNK = 10
BRAND = "FOOTBALL PREDICTOR"
TZ = ZoneInfo("Europe/Rome")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _brand_header(*, continued: bool = False) -> str:
    when = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    suffix = " (cont.)" if continued else ""
    return f"{BRAND} — alert scommesse{suffix}\n{when} Roma"


def _load_sent() -> dict[str, str]:
    if not SENT.is_file():
        return {}
    try:
        raw = json.loads(SENT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(raw, list):
        return {str(k): _now().isoformat() for k in raw}
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    return {}


def _save_sent(ids: dict[str, str]) -> None:
    cutoff = _now() - timedelta(days=KEEP_DAYS)
    kept: dict[str, str] = {}
    for key, ts in ids.items():
        try:
            when = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            when = _now()
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        if when >= cutoff:
            kept[key] = ts
    SENT.parent.mkdir(parents=True, exist_ok=True)
    SENT.write_text(json.dumps(kept, indent=2), encoding="utf-8")


def _fmt_line(val: object) -> str:
    if val is None:
        return "—"
    try:
        num = float(val)
    except (TypeError, ValueError):
        return "—"
    if abs(num - round(num)) < 1e-9:
        return str(int(round(num)))
    return f"{num:.2f}".rstrip("0").rstrip(".")


def _pct(val: object) -> str | None:
    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    if abs(num) <= 1.5:
        num *= 100.0
    return f"{num:+.1f}%"


def _load_upcoming(rows: list[dict] | None) -> list[dict]:
    if rows is not None:
        return rows
    if not UPCOMING.is_file():
        return []
    try:
        data = json.loads(UPCOMING.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _match_key(row: dict) -> str:
    date = str(row.get("date") or "")[:10]
    home = str(row.get("home") or "").strip().lower()
    away = str(row.get("away") or "").strip().lower()
    mid = str(row.get("match_id") or "").strip()
    if mid:
        return f"{date}|{mid}"
    return f"{date}|{home}|{away}"


def _header(row: dict) -> str:
    date = str(row.get("date") or "")[:10]
    time = str(row.get("time") or "").strip()
    league = str(row.get("league") or "").strip()
    home = str(row.get("home") or "?")
    away = str(row.get("away") or "?")
    when = " ".join(p for p in (date, time) if p)
    bits = [b for b in (when, league) if b]
    line = " · ".join(bits)
    title = f"{home} vs {away}"
    return f"{line}\n{title}" if line else title


def _reasons_text(row: dict) -> str | None:
    raw = row.get("no_bet_reasons")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return raw.strip()[:180] or None
    if isinstance(raw, list) and raw:
        return str(raw[0]).strip()[:180]
    return None


def _pick_line(row: dict) -> str:
    pick = str(row.get("pick") or "—")
    name = str(row.get("pick_name") or "")
    if pick == name or not name:
        return pick
    return f"{pick} · {name}"



def _as_quota(val: object) -> float | None:
    if isinstance(val, bool) or val is None:
        return None
    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    if 1.01 <= num <= 100:
        return num
    return None


def _quota_of(row: dict) -> float | None:
    """Quota book del pick consigliato (mai fair_odds)."""
    num = _as_quota(row.get("quota_pick"))
    if num is not None:
        return num
    if not isinstance(row.get("odds"), bool):
        num = _as_quota(row.get("odds"))
        if num is not None:
            return num
    pick = str(row.get("pick") or "").strip().upper()
    flat = {
        "1": ("odd_1", "1"),
        "X": ("odd_x", "X"),
        "2": ("odd_2", "2"),
        "O2.5": ("odd_over_25", "over_2.5"),
        "U2.5": ("odd_under_25", "under_2.5"),
    }
    odds = row.get("odds")
    if pick in flat:
        flat_key, dict_key = flat[pick]
        num = _as_quota(row.get(flat_key))
        if num is not None:
            return num
        if isinstance(odds, dict):
            num = _as_quota(odds.get(dict_key))
            if num is not None:
                return num
    markets = row.get("markets")
    if isinstance(markets, list):
        for m in markets:
            if isinstance(m, dict) and str(m.get("code") or "").upper() == pick:
                num = _as_quota(m.get("odds"))
                if num is not None:
                    return num
    return None


def _fair_quota_of(row: dict) -> float | None:
    """Quota equa secondo l'analisi (1/p del pick consigliato)."""
    num = _as_quota(row.get("fair_odds"))
    if num is not None:
        return num
    pick = str(row.get("pick") or "").strip().upper()
    markets = row.get("markets")
    if isinstance(markets, list):
        for m in markets:
            if not isinstance(m, dict):
                continue
            if str(m.get("code") or "").upper() != pick:
                continue
            num = _as_quota(m.get("fair_odds"))
            if num is not None:
                return num
            for pk in ("p_cons", "probability", "model_probability"):
                try:
                    p = float(m.get(pk))
                except (TypeError, ValueError):
                    continue
                if p > 0.02:
                    return round(1.0 / min(p, 0.98), 2)
    for pk in ("probability", "p_cons"):
        try:
            p = float(row.get(pk))
        except (TypeError, ValueError):
            continue
        if p > 0.02:
            return round(1.0 / min(p, 0.98), 2)
    return None


def _fmt_quota(val: float | None) -> str | None:
    if val is None:
        return None
    return f"{val:.2f}".rstrip("0").rstrip(".")


def _quota_lines(row: dict, *, book_label: str = "Quota") -> list[str]:
    """Righe Telegram: quota book + quota equa dall'analisi."""
    book = _fmt_quota(_quota_of(row))
    fair = _fmt_quota(_fair_quota_of(row))
    if book and fair and book != fair:
        return [f"{book_label} {book} · equa {fair}"]
    if book:
        return [f"{book_label} {book}"]
    if fair:
        return [f"Equa (analisi) {fair}"]
    return []


def _hist_phrase(score: int) -> str | None:
    try:
        from modules.advisor.analysis_outcomes import outcome_phrase

        return outcome_phrase(score_unified=score, live_only=True) or outcome_phrase(
            score_unified=score, live_only=False
        )
    except Exception:
        return None


def _score_alerts(rows: list[dict]) -> dict[str, list[dict]]:
    gioca: list[dict] = []
    watch: list[dict] = []
    for row in rows:
        try:
            score = int(row.get("score_unified"))
        except (TypeError, ValueError):
            continue
        if score < MIN_UNIFIED:
            continue
        action = str(row.get("action") or "").strip().lower()
        ev = _pct(row.get("ev_cons") if row.get("ev_cons") is not None else row.get("ev"))
        play = _pick_line(row)
        note = str(row.get("score_reason_1") or row.get("meta_note") or "").strip()

        if action == "gioca":
            key = f"gioca|{_match_key(row)}|{score}"
            body = [_header(row), f"🎯 GIOCA · voto {score}/10"]
            body.append(f"Pick: {play}")
            body.extend(_quota_lines(row, book_label="Quota"))
            if ev:
                body.append(f"EV {ev}")
            kq = row.get("kelly_quarter")
            if kq is not None:
                try:
                    body.append(f"Kelly ¼ {float(kq):.1%}")
                except (TypeError, ValueError):
                    pass
            if note:
                body.append(note[:220])
            hist = _hist_phrase(score)
            if hist:
                body.append(f"Storico voto {score}: {hist}")
            gioca.append({"id": key, "kind": "gioca", "text": "\n".join(body), "sort": -score})
            continue

        if action not in {"no_bet", "invalido", "n/d"}:
            continue
        key = f"watch|{_match_key(row)}|{score}"
        body = [_header(row), f"👀 Da guardare · voto {score}/10 · NO BET"]
        body.append(f"Pick: {play} — non giocare")
        body.extend(_quota_lines(row, book_label="Quota rif."))
        if ev:
            body.append(f"EV {ev}")
        reason = _reasons_text(row)
        if reason:
            body.append(f"Motivo: {reason}")
        elif action != "no_bet":
            body.append(f"Stato: {action}")
        if note:
            body.append(note[:220])
        watch.append({"id": key, "kind": "watch", "text": "\n".join(body), "sort": -score})
    return {"gioca": gioca, "watch": watch}


def _rare_from_move(row: dict, move: dict | None) -> dict | None:
    move = move or {}
    level = str(move.get("movement_level") or row.get("movement_level") or "")
    try:
        line = float(move.get("line_move") if move.get("line_move") is not None else row.get("line_move") or 0)
    except (TypeError, ValueError):
        line = 0.0
    if level != "Raro" and line < RARE_LINE:
        return None
    ah_open = move.get("ah_open", row.get("ah_open"))
    ah_curr = move.get("ah_curr", row.get("ah_curr"))
    tot_open = move.get("total_open", row.get("total_open"))
    tot_curr = move.get("total_curr", row.get("total_line") or row.get("total_curr"))
    playab = spread_playability(row, move)
    score = int(playab["score"])
    if score < MIN_SPREAD_PLAYABILITY:
        return None
    verdict = str(playab.get("verdict") or "")
    body = [
        _header(row),
        f"⭐ Giocabilità {score}/10 · {verdict}",
        f"Spread Raro · linea AH/totale Δ {line:g} (≥1)",
    ]
    if ah_open is not None or ah_curr is not None:
        body.append(f"AH {_fmt_line(ah_open)} → {_fmt_line(ah_curr)}")
    if tot_open is not None or tot_curr is not None:
        body.append(f"Totale {_fmt_line(tot_open)} → {_fmt_line(tot_curr)}")
    steam = " / ".join(
        p
        for p in (
            str(move.get("steam_1x2") or row.get("steam_1x2") or ""),
            str(move.get("steam_ah") or row.get("steam_ah") or ""),
            str(move.get("steam_ou") or row.get("steam_ou") or ""),
        )
        if p
    )
    if steam:
        body.append(f"Steam: {steam}")
    reason = str(playab.get("reason") or "").strip()
    if reason:
        body.append(reason[:220])
    summary = str(move.get("movement_summary") or row.get("movement_summary") or "").strip()
    if summary:
        body.append(summary[:220])
    key = f"spread|{_face_key(row)}"
    return {
        "id": key,
        "kind": "spread",
        "text": "\n".join(body),
        "sort": -score * 100 - line,
        "score": score,
        "verdict": verdict,
    }


def _face_key(row: dict) -> str:
    date = str(row.get("date") or "")[:10]
    home = str(row.get("home") or "").strip().lower()
    away = str(row.get("away") or "").strip().lower()
    return f"{date}|{home}|{away}"


def _spread_alerts(upcoming: list[dict], asian: list[dict]) -> list[dict]:
    seen_ids: set[str] = set()
    seen_faces: set[str] = set()
    out: list[dict] = []
    for row in list(upcoming) + list(asian):
        move = row.get("market_move") if isinstance(row.get("market_move"), dict) else None
        alert = _rare_from_move(row, move)
        if not alert:
            continue
        face = _face_key(row)
        if alert["id"] in seen_ids or face in seen_faces:
            continue
        seen_ids.add(alert["id"])
        seen_faces.add(face)
        out.append(alert)
    return out


def _pack(title: str, items: list[dict]) -> list[tuple[str, list[str]]]:
    if not items:
        return []
    items = sorted(items, key=lambda r: (r.get("sort") or 0, r["id"]))
    messages: list[tuple[str, list[str]]] = []
    for i in range(0, len(items), CHUNK):
        chunk = items[i : i + CHUNK]
        head = title if i == 0 else f"{title} (cont.)"
        if len(chunk) == 1 and chunk[0].get("score") is not None:
            sc = int(chunk[0]["score"])
            ver = str(chunk[0].get("verdict") or "").strip()
            extra = f" · giocabilità {sc}/10"
            if ver:
                extra += f" · {ver}"
            if "giocabilità" not in head.lower():
                head = f"{head}{extra}"
        body = "\n\n".join(item["text"] for item in chunk)
        brand = _brand_header(continued=i > 0)
        messages.append((f"{brand}\n\n{head}\n\n{body}", [item["id"] for item in chunk]))
    return messages


def collect_alerts(upcoming: list[dict] | None = None) -> dict:
    rows = _load_upcoming(upcoming)
    try:
        asian = load_asian_odds()
    except Exception:
        asian = []
    scores = _score_alerts(rows)
    rares = _spread_alerts(rows, asian)
    gioca = scores["gioca"]
    watch = scores["watch"]
    return {
        "gioca": gioca,
        "watch": watch,
        "voto": gioca + watch,
        "spread": rares,
        "n_gioca": len(gioca),
        "n_watch": len(watch),
        "n_voto": len(gioca) + len(watch),
        "n_spread": len(rares),
    }


def dispatch_alerts(upcoming: list[dict] | None = None, *, dry_run: bool = False) -> dict:
    found = collect_alerts(upcoming)
    sent_ids = _load_sent()
    fresh_gioca = [a for a in found["gioca"] if a["id"] not in sent_ids]
    fresh_watch = [a for a in found["watch"] if a["id"] not in sent_ids]
    fresh_spread = [a for a in found["spread"] if a["id"] not in sent_ids]
    messages = _pack(f"🎯 GIOCA · voto ≥{MIN_UNIFIED}", fresh_gioca)
    messages += _pack(f"👀 Da guardare · voto ≥{MIN_UNIFIED} · NO BET", fresh_watch)
    messages += _pack(
        f"📈 AsianBetSoccer · spread Raro · giocabilità >8 (min {MIN_SPREAD_PLAYABILITY}/10)",
        fresh_spread,
    )

    sent_n = 0
    if dry_run:
        for msg, _ids in messages:
            print(msg)
            print("---")
    elif messages:
        if not load_credentials():
            print("telegram skip: credenziali assenti")
        else:
            now = _now().isoformat()
            changed = False
            for msg, ids in messages:
                if send_message(msg):
                    sent_n += 1
                    for key in ids:
                        sent_ids[key] = now
                    changed = True
            if changed:
                _save_sent(sent_ids)

    info = {
        "n_gioca": found["n_gioca"],
        "n_watch": found["n_watch"],
        "n_voto": found["n_voto"],
        "n_spread": found["n_spread"],
        "n_new_gioca": len(fresh_gioca),
        "n_new_watch": len(fresh_watch),
        "n_new_voto": len(fresh_gioca) + len(fresh_watch),
        "n_new_spread": len(fresh_spread),
        "n_messages": len(messages),
        "n_sent": sent_n,
        "dry_run": dry_run,
        "status": telegram_status(),
    }
    print(
        f"telegram avvisi: gioca {info['n_new_gioca']}/{info['n_gioca']} nuovi, "
        f"watch {info['n_new_watch']}/{info['n_watch']} nuovi, "
        f"spread {info['n_new_spread']}/{info['n_spread']} nuovi, "
        f"inviati {sent_n}"
    )
    return info


def resend_spread_match(home: str, away: str) -> dict:
    """Reinvia lo spread Raro di una partita (ignora la cache già-inviato)."""
    found = collect_alerts()
    h = home.strip().lower()
    a = away.strip().lower()
    match = None
    for item in found["spread"]:
        blob = f"{item.get('id') or ''} {item.get('text') or ''}".lower()
        if h in blob and a in blob:
            match = item
            break
    if not match:
        return {
            "ok": False,
            "error": f"nessun spread Raro per {home} vs {away}",
            "n_spread": found["n_spread"],
        }
    score = match.get("score")
    verdict = str(match.get("verdict") or "").strip()
    title = "📈 Spread Raro (giocabilità 1–10)"
    if score is not None:
        title = f"📈 Spread Raro · giocabilità {int(score)}/10"
        if verdict:
            title += f" · {verdict}"
    text = f"{_brand_header()}\n\n{title}\n\n{match['text']}"
    if not load_credentials():
        print("telegram skip: credenziali assenti")
        return {"ok": False, "error": "credenziali assenti", "text": text}
    ok = send_message(text)
    return {
        "ok": ok,
        "id": match["id"],
        "score": score,
        "verdict": verdict,
        "text": text,
    }


def ping_bot() -> bool:
    when = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    return send_message(
        f"🧪 PROVA — {BRAND}\n{when} Roma\n\n"
        "Bot collegato.\n"
        f"Avvisi: 🎯 GIOCA (voto ≥{MIN_UNIFIED} + action gioca), "
        f"👀 da guardare (voto ≥{MIN_UNIFIED} + no bet), "
        f"spread Raro con giocabilità >8 (min {MIN_SPREAD_PLAYABILITY}/10)."
    )
