"""Pending ROI unificati (analisi / moneyway / ritardi) + settle manuale."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _age_days(day_s: str) -> int | None:
    try:
        d0 = datetime.fromisoformat(str(day_s)[:10]).date()
    except ValueError:
        return None
    return (datetime.now(timezone.utc).date() - d0).days


def _with_closable(row: dict[str, Any]) -> dict[str, Any]:
    """Marca se la partita è già giocabile da chiudere (data ≤ oggi)."""
    age = row.get("age_days")
    if age is None:
        age = _age_days(str(row.get("date") or ""))
        row["age_days"] = age
    closable = age is not None and int(age) >= 0
    row["closable"] = closable
    row["status"] = "Chiudibile" if closable else "Futura"
    return row


def list_analisi_roi_pending() -> list[dict[str, Any]]:
    """Stesso campione del `(n)` home ROI analisi: GIOCA freeze senza hit (anche future)."""
    from modules.advisor.learn_policy import score_unified_of
    from modules.data_update.history import load_history

    out: list[dict[str, Any]] = []
    for r in load_history():
        if r.get("hit") is not None:
            continue
        if int(r.get("score_locked") or 0) != 1:
            continue
        if str(r.get("alert_kind") or "").lower() == "watch":
            continue
        if str(r.get("action") or "").lower() in {"no_bet", "invalido", "n/d"}:
            continue
        if not (
            str(r.get("alert_kind") or "").lower() == "gioca"
            or str(r.get("action") or "").lower() == "gioca"
        ):
            continue
        pick = str(r.get("pick") or "").upper()
        day_s = str(r.get("date") or "")[:10]
        out.append(
            _with_closable(
                {
                    "source": "analisi",
                    "id": str(r.get("match_key") or ""),
                    "journal_key": str(r.get("match_key") or ""),
                    "match_key": r.get("match_key"),
                    "date": day_s,
                    "home": r.get("home"),
                    "away": r.get("away"),
                    "league": r.get("league"),
                    "country": r.get("country"),
                    "pick": r.get("pick"),
                    "pick_label": r.get("pick_label") or r.get("pick"),
                    "odds": r.get("quota_pick"),
                    "playability": score_unified_of(r),
                    "score_unified": score_unified_of(r),
                    "age_days": _age_days(day_s),
                    "needs_corners": pick.startswith("CORN"),
                    "needs_shots": pick.startswith("SHOT"),
                }
            )
        )
    out.sort(key=lambda x: (int(x.get("age_days") or 0), str(x.get("date") or "")), reverse=True)
    return out


def list_all_roi_pending() -> dict[str, Any]:
    """Elenca tutti i pending dei tre paper ROI (incluse partite future non ancora chiudibili)."""
    from modules.advisor.ritardi_play import list_pending_ritardi
    from modules.advisor.spread_paper import list_pending_spread

    analisi = list_analisi_roi_pending()
    moneyway = [_with_closable(dict(r)) for r in list_pending_spread()]
    ritardi = [_with_closable(dict(r)) for r in list_pending_ritardi()]
    all_rows = [*analisi, *moneyway, *ritardi]
    all_rows.sort(
        key=lambda x: (int(x.get("age_days") or 0), str(x.get("date") or "")),
        reverse=True,
    )
    n_closable = sum(1 for r in all_rows if r.get("closable"))
    return {
        "ok": True,
        "n": len(all_rows),
        "n_closable": n_closable,
        "n_future": len(all_rows) - n_closable,
        "n_analisi": len(analisi),
        "n_moneyway": len(moneyway),
        "n_ritardi": len(ritardi),
        "rows": all_rows,
        "analisi": analisi,
        "moneyway": moneyway,
        "ritardi": ritardi,
    }


def settle_roi_pending(
    *,
    source: str,
    entry_id: str,
    home_goals: int | None = None,
    away_goals: int | None = None,
    corners_home: int | None = None,
    corners_away: int | None = None,
    shots_home: int | None = None,
    shots_away: int | None = None,
    hit: int | None = None,
    date: str | None = None,
    home: str | None = None,
    away: str | None = None,
) -> dict[str, Any]:
    """Chiude un pending ROI e prova a propagare i gol agli altri journal."""
    src = str(source or "").strip().lower()
    primary: dict[str, Any]

    if src == "analisi":
        from modules.data_update.history import manual_settle

        primary = manual_settle(
            match_key=entry_id or None,
            date=date,
            home=home,
            away=away,
            home_goals=home_goals,
            away_goals=away_goals,
            corners_home=corners_home,
            corners_away=corners_away,
            shots_home=shots_home,
            shots_away=shots_away,
            hit=hit,
        )
    elif src == "moneyway":
        from modules.advisor.spread_paper import manual_settle_spread

        primary = manual_settle_spread(
            entry_id,
            home_goals=home_goals,
            away_goals=away_goals,
            hit=hit,
        )
    elif src == "ritardi":
        from modules.advisor.ritardi_play import manual_settle_ritardi

        primary = manual_settle_ritardi(
            entry_id,
            home_goals=home_goals,
            away_goals=away_goals,
            corners_home=corners_home,
            corners_away=corners_away,
            hit=hit,
        )
    else:
        return {"ok": False, "error": f"source sconosciuta: {source}"}

    if not primary.get("ok"):
        return primary

    cascade: dict[str, Any] = {}
    # Se abbiamo FT, prova a chiudere le altre sorgenti sullo stesso face
    if home_goals is not None and away_goals is not None and date and home and away:
        cascade = _cascade_ft(
            date=str(date)[:10],
            home=str(home),
            away=str(away),
            home_goals=int(home_goals),
            away_goals=int(away_goals),
            corners_home=corners_home,
            corners_away=corners_away,
            skip_source=src,
            skip_id=entry_id,
        )
    elif home_goals is not None and away_goals is not None:
        # prova comunque auto-settle da history (se analisi già aggiornata)
        try:
            from modules.advisor.ritardi_play import settle_ritardi_journal
            from modules.advisor.spread_paper import settle_spread_journal

            cascade = {
                "spread": settle_spread_journal(),
                "ritardi": settle_ritardi_journal(),
            }
        except Exception as exc:
            cascade = {"error": str(exc)}

    return {**primary, "cascade": cascade}


def _cascade_ft(
    *,
    date: str,
    home: str,
    away: str,
    home_goals: int,
    away_goals: int,
    corners_home: int | None,
    corners_away: int | None,
    skip_source: str,
    skip_id: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {"closed": []}

    if skip_source != "analisi":
        try:
            from modules.data_update.history import manual_settle

            r = manual_settle(
                date=date,
                home=home,
                away=away,
                home_goals=home_goals,
                away_goals=away_goals,
                corners_home=corners_home,
                corners_away=corners_away,
            )
            if r.get("ok"):
                out["closed"].append({"source": "analisi", **{k: r.get(k) for k in ("match_key", "hit", "presa")}})
            else:
                out["analisi_skip"] = r.get("error")
        except Exception as exc:
            out["analisi_skip"] = str(exc)

    if skip_source != "moneyway":
        try:
            from modules.advisor.spread_paper import list_pending_spread, manual_settle_spread

            face = f"{date}|{home.strip().lower()}|{away.strip().lower()}"
            for row in list_pending_spread():
                mk = str(row.get("match_key") or "")
                jid = str(row.get("journal_key") or row.get("id") or "")
                if jid == skip_id:
                    continue
                if mk == face or (
                    str(row.get("date") or "")[:10] == date
                    and str(row.get("home") or "").strip().lower() == home.strip().lower()
                    and str(row.get("away") or "").strip().lower() == away.strip().lower()
                ):
                    r = manual_settle_spread(
                        jid,
                        home_goals=home_goals,
                        away_goals=away_goals,
                    )
                    if r.get("ok"):
                        out["closed"].append({"source": "moneyway", "id": jid, "presa": r.get("presa")})
        except Exception as exc:
            out["moneyway_skip"] = str(exc)

    if skip_source != "ritardi":
        try:
            from modules.advisor.ritardi_play import list_pending_ritardi, manual_settle_ritardi

            face = f"{date}|{home.strip().lower()}|{away.strip().lower()}"
            for row in list_pending_ritardi():
                mk = str(row.get("match_key") or "")
                jid = str(row.get("journal_key") or row.get("id") or "")
                if jid == skip_id:
                    continue
                if mk == face or (
                    str(row.get("date") or "")[:10] == date
                    and str(row.get("home") or "").strip().lower() == home.strip().lower()
                    and str(row.get("away") or "").strip().lower() == away.strip().lower()
                ):
                    r = manual_settle_ritardi(
                        jid,
                        home_goals=home_goals,
                        away_goals=away_goals,
                        corners_home=corners_home,
                        corners_away=corners_away,
                    )
                    if r.get("ok"):
                        out["closed"].append({"source": "ritardi", "id": jid, "presa": r.get("presa")})
        except Exception as exc:
            out["ritardi_skip"] = str(exc)

    return out
