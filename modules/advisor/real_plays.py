"""Giocate reali (quota/stake effettivi) — ROI personale, separato dal paper.

Il paper ROI resta su quote freeze Telegram. Qui logghi solo ciò che hai
davvero puntato, per un PnL reale quando inizi a giocare.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "data" / "processed" / "real_plays.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _f(v: Any) -> float | None:
    try:
        if v is None or isinstance(v, bool):
            return None
        x = float(v)
        if x != x:
            return None
        return x
    except (TypeError, ValueError):
        return None


def _load() -> dict[str, dict[str, Any]]:
    if not JOURNAL.is_file():
        return {}
    try:
        raw = json.loads(JOURNAL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(raw, dict) and isinstance(raw.get("plays"), dict):
        raw = raw["plays"]
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


def _save(data: dict[str, dict[str, Any]]) -> None:
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": _now(), "n": len(data), "plays": data}
    JOURNAL.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def play_key(*, source: str, match_key: str, pick: str = "") -> str:
    src = str(source or "manual").strip().lower()
    mk = str(match_key or "").strip()
    pk = str(pick or "").strip()
    return f"{src}|{mk}|{pk}" if pk else f"{src}|{mk}"


def list_real_plays() -> list[dict[str, Any]]:
    rows = list(_load().values())
    rows.sort(
        key=lambda r: (str(r.get("date") or ""), str(r.get("logged_at") or "")),
        reverse=True,
    )
    return rows


def upsert_real_play(
    *,
    source: str,
    match_key: str,
    date: str | None = None,
    home: str | None = None,
    away: str | None = None,
    league: str | None = None,
    pick: str | None = None,
    pick_label: str | None = None,
    odds_paper: float | None = None,
    odds_played: float,
    stake_units: float = 1.0,
    hit: int | None = None,
    notes: str | None = None,
    entry_id: str | None = None,
) -> dict[str, Any]:
    """Registra / aggiorna una giocata reale. Non tocca il paper ROI."""
    od = _f(odds_played)
    if od is None or od <= 1.01:
        return {"ok": False, "error": "quota giocata non valida (serve > 1.01)"}
    st = _f(stake_units)
    if st is None or st <= 0:
        return {"ok": False, "error": "stake non valido"}

    key = entry_id or play_key(source=source, match_key=match_key, pick=str(pick or ""))
    data = _load()
    prev = data.get(key) or {}

    final_hit = hit if hit is not None else prev.get("hit")
    try:
        if final_hit is not None:
            final_hit = 1 if int(final_hit) == 1 else 0
    except (TypeError, ValueError):
        return {"ok": False, "error": "hit non valido"}

    pnl = None
    if final_hit is not None:
        pnl = round(st * (od - 1.0), 4) if final_hit == 1 else round(-st, 4)

    out = {
        **prev,
        "id": key,
        "source": str(source or "manual").strip().lower(),
        "match_key": str(match_key or "").strip(),
        "date": str(date or prev.get("date") or "")[:10],
        "home": home if home is not None else prev.get("home"),
        "away": away if away is not None else prev.get("away"),
        "league": league if league is not None else prev.get("league"),
        "pick": pick if pick is not None else prev.get("pick"),
        "pick_label": pick_label if pick_label is not None else prev.get("pick_label"),
        "odds_paper": _f(odds_paper) if odds_paper is not None else prev.get("odds_paper"),
        "odds_played": round(od, 3),
        "stake_units": round(st, 4),
        "hit": final_hit,
        "pnl": pnl,
        "notes": (notes if notes is not None else prev.get("notes")) or "",
        "logged_at": prev.get("logged_at") or _now(),
        "updated_at": _now(),
    }
    data[key] = out
    _save(data)
    return {"ok": True, "play": out, "path": str(JOURNAL)}


def delete_real_play(entry_id: str) -> dict[str, Any]:
    data = _load()
    key = str(entry_id)
    if key not in data:
        return {"ok": False, "error": "giocata non trovata"}
    del data[key]
    _save(data)
    return {"ok": True, "deleted": key, "n": len(data)}


def sync_hits_from_paper() -> dict[str, Any]:
    """Aggancia hit/esito dai journal paper / storico se la giocata è ancora aperta."""
    from modules.advisor.ritardi_play import _load_journal as load_ritardi
    from modules.advisor.spread_paper import _load_journal as load_spread
    from modules.data_update.history import load_history

    data = _load()
    if not data:
        return {"ok": True, "updated": 0, "n": 0}

    by_analisi: dict[str, dict] = {}
    for r in load_history():
        if r.get("hit") is None:
            continue
        mk = str(r.get("match_key") or "")
        if mk:
            by_analisi[mk] = r

    by_spread = {str(k): v for k, v in load_spread().items() if v.get("settled_at")}
    by_ritardi = {str(k): v for k, v in load_ritardi().items() if v.get("settled_at")}

    updated = 0
    for key, play in list(data.items()):
        if play.get("hit") is not None:
            continue
        src = str(play.get("source") or "")
        mk = str(play.get("match_key") or "")
        hit = None
        paper = None
        if src == "analisi" and mk in by_analisi:
            paper = by_analisi[mk]
            hit = paper.get("hit")
        elif src == "moneyway":
            paper = by_spread.get(key.split("|", 1)[-1]) or by_spread.get(mk)
            if paper is None:
                for v in by_spread.values():
                    if str(v.get("match_key") or "") == mk:
                        paper = v
                        break
            if paper is not None:
                hit = paper.get("hit")
                if hit is None and int(paper.get("push") or 0) == 1:
                    continue  # push: lascia aperto o pnl 0
        elif src == "ritardi":
            paper = by_ritardi.get(mk)
            if paper is None:
                for v in by_ritardi.values():
                    if str(v.get("match_key") or "") == mk or str(v.get("id") or "") == mk:
                        paper = v
                        break
            if paper is not None:
                hit = paper.get("hit")
                if hit is None and int(paper.get("push") or 0) == 1:
                    continue

        if hit is None:
            continue
        try:
            hit_i = 1 if int(hit) == 1 else 0
        except (TypeError, ValueError):
            continue
        od = _f(play.get("odds_played"))
        st = _f(play.get("stake_units")) or 1.0
        if od is None:
            continue
        play = dict(play)
        play["hit"] = hit_i
        play["pnl"] = round(st * (od - 1.0), 4) if hit_i == 1 else round(-st, 4)
        play["settled_from"] = "paper"
        play["updated_at"] = _now()
        data[key] = play
        updated += 1

    if updated:
        _save(data)
    return {"ok": True, "updated": updated, "n": len(data), "path": str(JOURNAL)}


def real_plays_report() -> dict[str, Any]:
    """ROI reale su stake_units e odds_played (paper escluso)."""
    sync = sync_hits_from_paper()
    rows = list_real_plays()
    settled = [r for r in rows if r.get("hit") is not None and r.get("pnl") is not None]
    pending = [r for r in rows if r.get("hit") is None]
    stake_sum = sum(float(r.get("stake_units") or 0) for r in settled)
    pnl_sum = sum(float(r.get("pnl") or 0) for r in settled)
    hits = sum(1 for r in settled if int(r.get("hit") or 0) == 1)
    n = len(settled)
    by_src: dict[str, dict[str, Any]] = {}
    for src in ("analisi", "moneyway", "ritardi", "manual"):
        subset = [r for r in settled if str(r.get("source") or "") == src]
        if not subset and not any(str(r.get("source") or "") == src for r in pending):
            continue
        st = sum(float(r.get("stake_units") or 0) for r in subset)
        pn = sum(float(r.get("pnl") or 0) for r in subset)
        by_src[src] = {
            "n": len(subset),
            "pending": sum(1 for r in pending if str(r.get("source") or "") == src),
            "hits": sum(1 for r in subset if int(r.get("hit") or 0) == 1),
            "stake": round(st, 3),
            "pnl": round(pn, 3),
            "roi": round(pn / st, 3) if st > 0 else None,
        }

    return {
        "ok": True,
        "sync": sync,
        "n": n,
        "n_pending": len(pending),
        "n_journal": len(rows),
        "hits": hits,
        "hit_rate": round(hits / n, 3) if n else None,
        "stake": round(stake_sum, 3),
        "pnl": round(pnl_sum, 3),
        "roi": round(pnl_sum / stake_sum, 3) if stake_sum > 0 else None,
        "by_source": by_src,
        "rows": rows,
        "recent": settled[:20],
        "pending_rows": pending[:30],
        "note": (
            "ROI reale su quota/stake che hai loggato. "
            "Il paper ROI (home) resta sulle quote freeze e non viene modificato."
            if rows
            else "Nessuna giocata reale ancora: la macchina resta sul paper. "
            "Quando punti, registra qui quota e stake effettivi."
        ),
        "path": str(JOURNAL),
    }


def candidate_plays_from_paper(*, limit: int = 40) -> list[dict[str, Any]]:
    """Pending + recenti settled dai tre paper, per scegliere cosa loggare."""
    from modules.advisor.pending_settle import list_all_roi_pending
    from modules.advisor.ritardi_play import _load_journal as load_ritardi
    from modules.advisor.spread_paper import _load_journal as load_spread
    from modules.advisor.learn_policy import score_unified_of
    from modules.data_update.history import load_history

    existing = {str(p.get("id") or "") for p in list_real_plays()}
    existing |= {
        play_key(
            source=str(p.get("source") or ""),
            match_key=str(p.get("match_key") or ""),
            pick=str(p.get("pick") or ""),
        )
        for p in list_real_plays()
    }
    out: list[dict[str, Any]] = []

    bag = list_all_roi_pending()
    for r in bag.get("rows") or []:
        src = str(r.get("source") or "")
        mk = str(r.get("match_key") or r.get("journal_key") or "")
        pick = str(r.get("pick") or "")
        kid = play_key(source=src, match_key=mk, pick=pick)
        if kid in existing or str(r.get("journal_key") or "") in existing:
            continue
        out.append(
            {
                "source": src,
                "id": str(r.get("journal_key") or kid),
                "match_key": mk,
                "date": r.get("date"),
                "home": r.get("home"),
                "away": r.get("away"),
                "league": r.get("league"),
                "pick": pick,
                "pick_label": r.get("pick_label") or pick,
                "odds_paper": r.get("odds"),
                "hit": None,
                "status": r.get("status") or "pending",
            }
        )

    # recenti settled analisi GIOCA
    settled_an = [
        r
        for r in load_history()
        if r.get("hit") is not None
        and int(r.get("score_locked") or 0) == 1
        and (
            str(r.get("alert_kind") or "").lower() == "gioca"
            or str(r.get("action") or "").lower() == "gioca"
        )
    ]
    settled_an.sort(key=lambda r: str(r.get("settled_at") or r.get("date") or ""), reverse=True)
    for r in settled_an[:25]:
        mk = str(r.get("match_key") or "")
        pick = str(r.get("pick") or "")
        kid = play_key(source="analisi", match_key=mk, pick=pick)
        if kid in existing:
            continue
        out.append(
            {
                "source": "analisi",
                "id": kid,
                "match_key": mk,
                "date": str(r.get("date") or "")[:10],
                "home": r.get("home"),
                "away": r.get("away"),
                "league": r.get("league"),
                "pick": pick,
                "pick_label": r.get("pick_label") or pick,
                "odds_paper": r.get("quota_pick"),
                "hit": int(r.get("hit") or 0),
                "status": "settled",
                "score_unified": score_unified_of(r),
            }
        )

    for src, journal in (("moneyway", load_spread()), ("ritardi", load_ritardi())):
        settled = [v for v in journal.values() if v.get("settled_at") and v.get("hit") is not None]
        settled.sort(key=lambda r: str(r.get("settled_at") or ""), reverse=True)
        for r in settled[:15]:
            mk = str(r.get("match_key") or r.get("id") or "")
            pick = str(r.get("pick") or r.get("bet") or "")
            kid = play_key(source=src, match_key=mk, pick=pick)
            jid = str(r.get("id") or kid)
            if kid in existing or jid in existing:
                continue
            out.append(
                {
                    "source": src,
                    "id": jid,
                    "match_key": mk,
                    "date": str(r.get("date") or "")[:10],
                    "home": r.get("home"),
                    "away": r.get("away"),
                    "league": r.get("league"),
                    "pick": pick,
                    "pick_label": r.get("pick_label") or r.get("bet_label") or pick,
                    "odds_paper": r.get("odds_frozen") or r.get("odds"),
                    "hit": int(r.get("hit") or 0),
                    "status": "settled",
                }
            )

    out.sort(key=lambda r: (str(r.get("date") or ""), str(r.get("source") or "")), reverse=True)
    return out[: int(limit)]
