"""Interfaccia: calendario, mercati 1X2 / O/U / gol / DC, filtri quote."""

from __future__ import annotations

import json
import subprocess
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import streamlit as st

from main import predict_pipeline
from modules.advisor.advise import advise, format_advice
from modules.calibration.config import load_calibration
from modules.data_update.asian_odds import (
    MOVE_FILTER_OPTIONS,
    MOVE_FILTER_RANK,
    MOVE_RANK,
    find_asian_odds,
    load_asian_odds,
    summarize_moves,
)
from modules.predictor import list_known_teams, list_team_meta
from modules.data_update.cups import download_org_cups, org_token_configured, save_org_token
from modules.data_update.fbref_context import download_fbref_context
from modules.data_update.understat_context import download_understat_context
from modules.data_update.statsbomb_context import download_statsbomb_context
from modules.data_update.sofascore_context import download_sofascore_context
from modules.data_update.parse import load_fixtures

ROOT = Path(__file__).resolve().parent
LAST = ROOT / "data" / "processed" / "last_prediction.json"
UPCOMING = ROOT / "data" / "processed" / "upcoming_predictions.json"

GROUP_LABEL = {
    "1x2": "1X2",
    "dc": "Doppia chance / DNB",
    "ou": "Over / Under",
    "btts": "Gol / No gol",
    "team": "Gol squadra",
    "combo": "Combo (risultato + O/U / Gol)",
}

st.set_page_config(page_title="Football Predictor", page_icon="⚽", layout="wide")
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; max-width: 1280px; }
      .pick-code {
        font-size: 3.2rem; font-weight: 800; line-height: 1;
        letter-spacing: 0.04em; margin: 0;
      }
      .pick-name { font-size: 1.05rem; opacity: 0.85; margin-top: 0.25rem; }
      .score-wrap { display: flex; gap: 6px; margin: 0.8rem 0 0.3rem; }
      .score-cell { width: 20px; height: 9px; border-radius: 2px; background: rgba(250,250,250,0.12); }
      .score-cell.on-hi { background: #3dd68c; }
      .score-cell.on-mid { background: #e6c35c; }
      .score-cell.on-lo { background: #e06c75; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _score_bar(score: int | None) -> str:
    if score is None:
        score = 0
    tone = "on-hi" if score >= 7 else "on-mid" if score >= 4 else "on-lo"
    cells = "".join(
        f'<div class="score-cell {tone if i <= score else ""}"></div>' for i in range(1, 11)
    )
    return f'<div class="score-wrap">{cells}</div>'


def _quota_consiglio(row) -> float | None:
    if row.get("pick") == "1":
        return row.get("odd_1")
    if row.get("pick") == "X":
        return row.get("odd_x")
    if row.get("pick") == "2":
        return row.get("odd_2")
    mk = row.get("markets") or []
    for m in mk:
        if m.get("code") == row.get("pick") and m.get("odds"):
            return m["odds"]
    return None


def _filter_by_date(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if col not in df.columns or df.empty:
        return df
    dates = pd.to_datetime(df[col], errors="coerce").dt.date
    valid = dates.dropna()
    if valid.empty:
        return df
    dmin, dmax = valid.min(), valid.max()
    picked = st.date_input("Intervallo date", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    if isinstance(picked, tuple):
        d1, d2 = picked[0], picked[-1]
    else:
        d1 = d2 = picked
    mask = (dates >= d1) & (dates <= d2)
    return df[mask.fillna(False)]


def _pct(val: float | None) -> str | None:
    if val is None:
        return None
    return f"{float(val):+.0%}"


def _safe_pct(val) -> str:
    if val is None:
        return "—"
    try:
        return f"{float(val):.0%}"
    except (TypeError, ValueError):
        return "—"


def _render_validation(val: dict | None) -> None:
    if not val:
        return
    venue = val.get("venue") or {}
    tac = val.get("tactical") or {}
    mkt = val.get("market") or {}
    stab = val.get("stability") or {}
    form = val.get("form") or {}
    p_adj = val.get("p_validated") or {}
    with st.container(border=True):
        st.markdown("**Validazione (non EV)**")
        with st.container(horizontal=True):
            flag = venue.get("flag") or "n/d"
            pen = venue.get("penalty_pct") or 0
            st.metric(
                "Stadio",
                flag.replace("_", " "),
                delta=f"{pen:.0%}" if pen else None,
                border=True,
            )
            st.metric(
                "Tattica",
                tac.get("status") or "n/d",
                delta=(f"{tac.get('delta_unified'):+.1f}" if tac.get("delta_unified") else None),
                border=True,
            )
            st.metric(
                "Gap mercato",
                "n/d" if mkt.get("gap") is None else f"{mkt['gap']:+.1%}",
                delta=mkt.get("status"),
                border=True,
            )
            st.metric(
                "ML vs MC",
                "n/d" if stab.get("diff") is None else f"{stab['diff']:.1%}",
                delta=stab.get("status"),
                border=True,
            )
            st.metric(
                "Forma",
                form.get("status") or "n/d",
                delta=(f"{form.get('delta_unified'):+.1f}" if form.get("delta_unified") else None),
                border=True,
            )
        bits = []
        if venue.get("venue"):
            bits.append(venue["venue"])
        if p_adj:
            bits.append(f"P' stadio {p_adj.get('home', 0):.0%}/{p_adj.get('draw', 0):.0%}/{p_adj.get('away', 0):.0%}")
        if val.get("delta_unified"):
            bits.append(f"Δ voto unificato {val['delta_unified']:+.1f}")
        if bits:
            st.caption(" · ".join(bits))
        warns = val.get("warnings") or []
        if warns:
            st.caption("Warning: " + " · ".join(warns[:4]))
        if form.get("incoherent"):
            st.caption("Forma incoerente: risultati e xG non coincidono.")


def _pp(val: float | None) -> str | None:
    if val is None:
        return None
    return f"{float(val):+.1f} pp"


def _sort_calendario(view: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = view.copy()
    if mode == "Data (più vicine)":
        if "date" in out.columns:
            return out.sort_values(["date", "time"], ascending=True, na_position="last")
        return out
    if mode == "Voto unificato":
        if "score_unified" in out.columns:
            return out.sort_values(["score_unified", "score", "probability"], ascending=False, na_position="last")
        return out.sort_values(["score", "probability"], ascending=False, na_position="last")
    if mode == "EV cons. %":
        ev = out["ev_cons"] if "ev_cons" in out.columns else out.get("ev")
        out["_sort"] = pd.to_numeric(ev, errors="coerce").fillna(-99)
        return out.sort_values("_sort", ascending=False).drop(columns="_sort")
    if mode == "Movimento mercato (maggiore)":
        out["_sort"] = out["movement_level"].map(MOVE_RANK).fillna(0)
        if "line_move" in out.columns:
            out["_sort"] = out["_sort"] * 100 + out["line_move"].fillna(0)
        elif "spread_score" in out.columns:
            out["_sort"] = out["_sort"] * 100 + out["spread_score"].fillna(0)
        return out.sort_values("_sort", ascending=False).drop(columns="_sort")
    if mode == "Value (edge vs mercato)":
        edge = out["edge_pp"] if "edge_pp" in out.columns else pd.Series(index=out.index, dtype=float)
        cons = out["ev_cons"] if "ev_cons" in out.columns else pd.Series(index=out.index, dtype=float)
        out["_sort"] = edge.fillna(cons).fillna(out["ev"] if "ev" in out.columns else -99).fillna(-99)
        return out.sort_values("_sort", ascending=False).drop(columns="_sort")
    return out.sort_values(["score", "probability"], ascending=False, na_position="last")


def _asian_radar_table(min_rank: int) -> pd.DataFrame:
    rows = load_asian_odds()
    if not rows:
        return pd.DataFrame()
    data = []
    for r in rows:
        move = summarize_moves(r)
        level = move.get("movement_level") or "Stabile"
        if MOVE_RANK.get(level, 0) < min_rank:
            continue
        data.append(
            {
                "Data": r.get("date"),
                "Ora": r.get("time"),
                "Campionato": r.get("league"),
                "Partita": f"{r.get('home')} vs {r.get('away')}",
                "Movimento": level,
                "Var linea": move.get("line_move"),
                "Cosa è cambiato": move.get("movement_summary") or "Quasi nessun movimento",
                "Commento quote": move.get("movement_comment"),
                "Soldi su 1X2": move.get("steam_1x2_label") or move.get("steam_1x2") or "—",
                "Soldi su O/U": move.get("steam_ou") or "—",
                "Δ 1": _pp(move.get("drop_1")),
                "Δ X": _pp(move.get("drop_x")),
                "Δ 2": _pp(move.get("drop_2")),
                "Δ Over": _pp(move.get("drop_over")),
                "Δ Under": _pp(move.get("drop_under")),
                "Quota 1 apertura": r.get("open_1"),
                "Quota 1 attuale": r.get("odd_1"),
                "Quota X apertura": r.get("open_x"),
                "Quota X attuale": r.get("odd_x"),
                "Quota 2 apertura": r.get("open_2"),
                "Quota 2 attuale": r.get("odd_2"),
            }
        )
    if not data:
        return pd.DataFrame()
    out = pd.DataFrame(data)
    out["_ord"] = out["Movimento"].map(MOVE_RANK).fillna(-1)
    out["_line"] = out["Var linea"].fillna(0)
    return out.sort_values(["_ord", "_line"], ascending=False).drop(columns=["_ord", "_line"])


def _enrich_upcoming(rows: list[dict]) -> list[dict]:
    """Ricalcola livelli, commenti e Δ pp dal cache Asian."""
    out: list[dict] = []
    for match in rows:
        item = dict(match)
        asian = find_asian_odds(
            str(item.get("home") or ""),
            str(item.get("away") or ""),
            item.get("date"),
        )
        if asian:
            move = summarize_moves(asian)
            item["market_move"] = move
            item["movement_comment"] = move.get("movement_comment")
            item["movement_summary"] = move.get("movement_summary")
            item["market_note"] = move.get("movement_comment")
            item["movement_level"] = move.get("movement_level")
            item["drop_1"] = move.get("drop_1")
            item["drop_x"] = move.get("drop_x")
            item["drop_2"] = move.get("drop_2")
            item["line_move"] = move.get("line_move")
            item["spread_score"] = move.get("spread_score")
        out.append(item)
    return out


@st.cache_data(show_spinner=False)
def _load_upcoming_enriched(_up_mtime: float, _asian_mtime: float) -> list[dict]:
    rows = _load_json(UPCOMING) or []
    if not rows:
        return []
    return _enrich_upcoming(rows)


def _kind_label(kind: str) -> str:
    return {
        "più_probabile": "Più probabile",
        "valore": "Miglior rapporto probabilità/quota",
        "probabile_e_valore": "Più probabile e miglior value",
        "invalido": "Pick invalido (quote assenti)",
        "nessun_pick": "Nessun pick (fonti non generano)",
        "lean_esterno": "Nessun pick (fonti non generano)",
    }.get(kind, kind)


def _fmt_pair(a, b, *, digits: int = 1, pct: bool = False) -> str:
    def one(v) -> str:
        if v is None:
            return "—"
        try:
            if pd.isna(v):
                return "—"
            n = float(v)
        except (TypeError, ValueError):
            return "—"
        if pct:
            return f"{n:.0%}" if digits == 0 else f"{n:.{digits}%}"
        if digits == 0:
            return f"{n:.0f}"
        return f"{n:.{digits}f}"

    return f"{one(a)} – {one(b)}"


def _display_text(v) -> str:
    if v is None:
        return ""
    s = str(v)
    # fallback visuale: niente caratteri "rotti" in tabella.
    ascii_s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return ascii_s or s


def _prepare_calendario_show(view: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "date", "time", "country", "league", "home", "away",
        "pick", "pick_name", "action", "score", "score_unified", "meta_label", "meta_note", "kelly_quarter", "clv",
        "quadro_consenso", "quadro_n", "tipster_consensus", "tipster_agree",
        "score_reason_1", "score_reason_2", "skip_reason", "probability",
        "quota_pick", "fair_odds", "edge_pp", "ev_cons", "ev_sharp",
        "odds_real", "value_note",
        "venue_flag", "validation_summary", "validation_delta",
        "movement_level", "line_move", "movement_summary", "movement_comment", "market_align",
        "drop_1", "drop_x", "drop_2",
        "odd_1", "odd_x", "odd_2", "odd_over_25", "odd_under_25", "odds_source",
    ]
    show = view[[c for c in wanted if c in view.columns]].copy()
    if "probability" in show.columns:
        show["probability"] = show["probability"].map(lambda x: f"{x:.0%}" if pd.notna(x) else None)
    if "edge_pp" in show.columns:
        show["edge_pp"] = show["edge_pp"].map(_pct)
    if "ev_cons" in show.columns:
        show["ev_cons"] = show["ev_cons"].map(_pct)
    if "ev_sharp" in show.columns:
        show["ev_sharp"] = show["ev_sharp"].map(_pct)
    if "odds_real" in show.columns:
        show["odds_real"] = show["odds_real"].map(lambda x: "Sì" if bool(x) else "No")
    if "kelly_quarter" in show.columns:
        show["kelly_quarter"] = show["kelly_quarter"].map(lambda x: f"{x:.1%}" if pd.notna(x) else None)
    if "clv" in show.columns:
        show["clv"] = show["clv"].map(lambda x: f"{x:+.1%}" if pd.notna(x) else None)
    if "action" in show.columns:
        show["action"] = show["action"].map(
            lambda x: (
                "No bet" if x == "no_bet"
                else "N/D" if x == "n/d"
                else "Invalido" if x == "invalido"
                else "Gioca"
            )
        )
    for drop_col in ("drop_1", "drop_x", "drop_2"):
        if drop_col in show.columns:
            show[drop_col] = show[drop_col].map(_pp)
    show = show.rename(
        columns={
            "date": "Data",
            "time": "Ora",
            "country": "Paese",
            "league": "Campionato",
            "home": "Casa",
            "away": "Trasferta",
            "pick": "Gioca",
            "pick_name": "Mercato",
            "action": "Azione",
            "score": "Voto",
            "score_unified": "Voto unificato",
            "meta_label": "Qualità mix",
            "meta_note": "Dettaglio mix",
            "kelly_quarter": "Kelly ¼",
            "clv": "CLV vs apertura",
            "quadro_consenso": "Quadro",
            "quadro_n": "Fonti vs pick",
            "tipster_consensus": "Tipster",
            "tipster_agree": "Vs tipster",
            "score_reason_1": "Perche questo voto",
            "score_reason_2": "Quote e mercato",
            "skip_reason": "Perche N/D",
            "probability": "Prob.",
            "quota_pick": "Quota book",
            "fair_odds": "Quota equa",
            "edge_pp": "Edge pp",
            "ev_cons": "EV cons.",
            "ev_sharp": "EV sharp",
            "odds_real": "Quota reale",
            "value_note": "Nota value",
            "venue_flag": "Stadio",
            "validation_summary": "Validazione",
            "validation_delta": "Δ validazione",
            "movement_level": "Movimento",
            "line_move": "Var linea",
            "movement_summary": "Cosa è cambiato",
            "movement_comment": "Commento quote",
            "market_align": "Vs mercato",
            "drop_1": "Δ 1",
            "drop_x": "Δ X",
            "drop_2": "Δ 2",
            "odd_1": "1",
            "odd_x": "X",
            "odd_2": "2",
            "odd_over_25": "O2.5",
            "odd_under_25": "U2.5",
            "odds_source": "Fonte quote",
        }
    )
    for col in ("Paese", "Campionato", "Casa", "Trasferta", "Mercato"):
        if col in show.columns:
            show[col] = show[col].map(_display_text)
    return show


def _run_cli(*flags: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "main.py"), *flags],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _table(markets: list[dict]) -> None:
    rows = []
    for m in markets:
        real = m.get("odds_real")
        if real is None:
            src = str(m.get("odds_source") or "")
            real = bool(src) and not src.startswith("stimata")
        rows.append(
            {
                "Mercato": m["name"],
                "Codice": m["code"],
                "Prob.": f"{m['probability']:.0%}",
                "P cons.": f"{m['p_cons']:.0%}" if m.get("p_cons") is not None else "—",
                "P mercato": f"{m['p_market']:.0%}" if real and m.get("p_market") is not None else "—",
                "Quota book": m["odds"],
                "Quota equa": m["fair_odds"],
                "Edge pp": _pct(m.get("edge_pp")) if real else "—",
                "EV cons.": _pct(m.get("ev_cons")) if real else "—",
                "EV sharp": _pct(m.get("ev_sharp")) if real else "—",
                "Voto prob.": m.get("score_prob"),
                "Voto value": m.get("score_value") if real and m.get("score_value") is not None else "—",
                "Voto finale": m.get("score"),
                "Kelly ¼": f"{m['kelly_quarter']:.1%}" if m.get("kelly_quarter") is not None else "—",
                "Fonte": m.get("odds_source") or "—",
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)


def render_advice(
    pred: dict,
    odds: dict,
    market_move: dict | None = None,
    *,
    odds_from_asian: bool = False,
    match_date: str | None = None,
    league: str | None = None,
) -> None:
    if odds_from_asian or market_move:
        match = pred.get("match") or ""
        if " vs " in match:
            home, away = match.split(" vs ", 1)
            asian = find_asian_odds(home.strip(), away.strip(), match_date)
            if asian:
                market_move = summarize_moves(asian)
    advice = advise(
        pred,
        odds,
        market_move=market_move,
        odds_from_asian=odds_from_asian,
        league=league or pred.get("league"),
    )
    play = advice["play"]
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown(f"**{advice['match']}**")
        action = play.get("action")
        if action == "invalido":
            head = f"INVALIDO {play.get('code') or '—'}"
        elif action == "n/d":
            head = "NESSUN PICK"
        elif action == "no_bet":
            head = f"NO BET {play.get('code') or ''}"
        else:
            head = f"GIOCA {play.get('code') or ''}"
        st.markdown(
            f'<p class="pick-code">{head}</p>'
            f'<p class="pick-name">{play["name"]} · {_kind_label(play.get("kind") or "")}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(_score_bar(play.get("score")), unsafe_allow_html=True)
        score_disp = play.get("score")
        st.markdown("**— / 10**" if score_disp is None else f"**{score_disp} / 10**")
        if play.get("score_unified") is not None:
            st.caption(f"Voto unificato (value + Kelly + Asian + workflow + storico): **{play['score_unified']}/10**")
        if play.get("action") == "no_bet":
            st.warning("No bet — " + "; ".join(play.get("no_bet_reasons") or ["filtro edge/mercato"]))
        elif play.get("action") == "invalido":
            st.error("Pick invalido — senza quote non si calcolano edge, EV, Kelly, quota equa, CLV.")
        elif play.get("action") == "n/d":
            st.info("Nessun pick: le fonti esterne validano, non generano giocate.")
        r1 = advice.get("score_reason_1")
        r2 = advice.get("score_reason_2")
        if r1:
            st.caption(r1)
        if r2:
            st.caption(r2)
    with right:
        p_cons = play.get("p_cons")
        p_mkt = play.get("p_market")
        edge = play.get("edge_pp")
        ev_cons = play.get("ev_cons") if play.get("ev_cons") is not None else play.get("ev")
        ev_sharp = play.get("ev_sharp")
        with st.container(border=True):
            st.markdown("**Modello vs mercato**")
            with st.container(horizontal=True):
                st.metric(
                    "Modello (cons.)",
                    (
                        f"{p_cons:.1%}"
                        if p_cons is not None
                        else (f"{play['probability']:.1%}" if play.get("probability") is not None else "—")
                    ),
                    border=True,
                )
                st.metric(
                    "Mercato (devig)",
                    f"{p_mkt:.1%}" if p_mkt is not None else "—",
                    border=True,
                )
                st.metric(
                    "Edge",
                    f"{edge:+.1%} pp" if edge is not None else "—",
                    border=True,
                )
                book_odd = play.get("odds")
                fair = play.get("fair_odds")
                st.metric(
                    "Quota vs equa",
                    f"{book_odd:.2f} vs {fair:.2f}" if book_odd and fair else "—",
                    border=True,
                )
            with st.container(horizontal=True):
                st.metric(
                    "EV cons.",
                    f"{ev_cons:+.1%}" if ev_cons is not None else "—",
                    border=True,
                )
                st.metric(
                    "EV sharp",
                    f"{ev_sharp:+.1%}" if ev_sharp is not None else "—",
                    border=True,
                )
                kq = play.get("kelly_quarter")
                st.metric("Kelly ¼", f"{kq:.1%}" if kq is not None else "—", border=True)
                odds_src = play.get("odds_source") or ("ipotetica" if play.get("odds_real") is False else "—")
                odds_sharp_val = play.get("odds_sharp")
                fonte_label = odds_src
                if odds_sharp_val and str(odds_src).lower() not in {"asianbetsoccer"}:
                    fonte_label = f"{odds_src} · sharp {odds_sharp_val}"
                elif odds_src == "pinnacle":
                    fonte_label = "Pinnacle (sharp)"
                st.metric("Fonte", fonte_label, border=True)
            if play.get("value_note"):
                st.caption(play["value_note"])
        meta = play.get("meta_analysis") or advice.get("meta_analysis") or {}
        if meta:
            with st.container(border=True):
                st.markdown("**Mix unico analisi**")
                with st.container(horizontal=True):
                    st.metric("Voto unificato", f"{meta.get('score', '—')}/10", border=True)
                    st.metric("Value", f"{_safe_pct(meta.get('value'))}", border=True)
                    st.metric("Kelly", f"{_safe_pct(meta.get('kelly'))}", border=True)
                    st.metric("Asian", f"{_safe_pct(meta.get('asian'))}", border=True)
                    st.metric("Workflow", f"{_safe_pct(meta.get('workflow'))}", border=True)
                st.caption(f"Lettura finale: {meta.get('label', 'n/d')} · {meta.get('note', '')}")
        val = play.get("validation") or (advice.get("quadro") or {}).get("validation")
        _render_validation(val)
        alt = advice.get("play_alt")
        if alt and alt["code"] != play["code"]:
            st.caption(f"Alternativa: **{alt['code']}** {alt['name']} · {alt['score']}/10")
        xg = advice.get("expected_goals") or {}
        st.caption(f"xG attesi {xg.get('home', '—')} – {xg.get('away', '—')}")
        move = advice.get("market_move")
        align = advice.get("market_align") or {}
        if move:
            st.markdown("**Mercato asiatico — variazioni di quota dall'apertura**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Intensità", move.get("movement_level") or "Stabile")
            line = move.get("line_move")
            m2.metric("Var linea", f"{line:g}" if line else "0")
            m3.metric("Direzione 1X2", move.get("steam_1x2_label") or move.get("steam_1x2") or "stabile")
            m4.metric("Direzione O/U", move.get("steam_ou") or "stabile")
            comment = move.get("movement_comment") or move.get("movement_summary") or move.get("note")
            if comment:
                st.info(comment)
            odds_moves = move.get("odds_moves") or []
            if odds_moves:
                st.dataframe(
                    [
                        {
                            "Mercato": r["market"],
                            "Apertura": r["open"],
                            "Attuale": r["current"],
                            "Δ quota": r["delta_odd"],
                            "Δ implicita": _pp(r.get("delta_pp")),
                            "Lettura": r.get("read"),
                        }
                        for r in odds_moves
                    ],
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "Δ implicita positiva = quota accorciata (più giocata). "
                    "Negativa = quota allungata (soldi altrove). AH negativo = casa favorita."
                )
            label = align.get("label") or "n/d"
            agrees = ", ".join(align.get("agrees") or []) or "—"
            disagrees = ", ".join(align.get("disagrees") or []) or "—"
            st.caption(f"Modello vs mercato: **{label}** · conferma {agrees} · contrasto {disagrees}")
            if play.get("clv") is not None:
                st.caption(f"CLV vs apertura Asian: **{play['clv']:+.1%}** (positivo = quota accorciata dopo l'open)")

        tip = advice.get("tipster") or play.get("tipster") or {}
        if tip.get("n_sources"):
            st.markdown("**Tipster professionisti**")
            t1, t2, t3 = st.columns(3)
            t1.metric("Consenso", tip.get("consensus") or "—")
            t2.metric("Fonti", str(tip.get("n_sources") or 0))
            t3.metric("Vs modello", tip.get("agree") or tip.get("label") or "—")
            src_rows = []
            for s in tip.get("sources") or []:
                src_rows.append(
                    {
                        "Fonte": s.get("source"),
                        "Pick": s.get("pick"),
                        "1": f"{s['p_home']:.0%}" if s.get("p_home") is not None else "—",
                        "X": f"{s['p_draw']:.0%}" if s.get("p_draw") is not None else "—",
                        "2": f"{s['p_away']:.0%}" if s.get("p_away") is not None else "—",
                    }
                )
            if src_rows:
                st.dataframe(src_rows, width="stretch", hide_index=True)
            st.caption(
                "I tipster pesano poco sul voto e non modificano EV o Kelly. "
                "Se sono contrari e il mercato è liquido contro, scatta il no-bet."
            )

    quadro = advice.get("quadro") or {}
    if quadro.get("sources"):
        with st.container(border=True):
            st.subheader("Quadro analisi")
            st.caption(quadro.get("summary") or "")
            feat = quadro.get("form") or {}
            with st.container(horizontal=True):
                st.metric("Consenso fonti", quadro.get("consenso") or "—", border=True)
                n_a, n_v = quadro.get("agree_n"), quadro.get("votes_n")
                st.metric(
                    "Allineate sul pick",
                    f"{n_a}/{n_v}" if n_v else "—",
                    border=True,
                )
                st.metric(
                    "Forma (pti)",
                    _fmt_pair(feat.get("pts_casa"), feat.get("pts_trasferta"), digits=1),
                    border=True,
                )
                st.metric(
                    "Riposo (gg)",
                    _fmt_pair(feat.get("riposo_casa"), feat.get("riposo_trasferta"), digits=0),
                    border=True,
                )
                st.metric(
                    "xG proxy",
                    _fmt_pair(feat.get("xg_casa"), feat.get("xg_trasferta"), digits=2),
                    border=True,
                )
            wr = _fmt_pair(feat.get("wr_casa"), feat.get("wr_trasferta"), digits=0, pct=True)
            if wr != "— – —":
                st.caption(f"Win rate casa / trasferta: {wr}")
            club = quadro.get("clubelo")
            if club:
                st.caption(
                    f"ClubElo {club['home'].get('club')} {club['home'].get('elo')} "
                    f"(#{club['home'].get('rank') or '—'}) vs "
                    f"{club['away'].get('club')} {club['away'].get('elo')} "
                    f"(#{club['away'].get('rank') or '—'})"
                )
            rows = []
            for s in quadro["sources"]:
                rows.append(
                    {
                        "Fonte": s.get("fonte"),
                        "Idea": s.get("idea"),
                        "Lean": s.get("pick"),
                        "1": f"{s['p_1']:.0%}" if s.get("p_1") is not None else "—",
                        "X": f"{s['p_x']:.0%}" if s.get("p_x") is not None else "—",
                        "2": f"{s['p_2']:.0%}" if s.get("p_2") is not None else "—",
                        "Nota": s.get("nota") or "—",
                    }
                )
            st.dataframe(rows, width="stretch", hide_index=True)
            st.caption("EV e Kelly restano su modello + quota reale. Questo quadro non entra nel conto.")
            gaps = quadro.get("gaps") or []
            if gaps:
                with st.expander("Cosa non è incluso"):
                    for g in gaps:
                        st.markdown(f"- {g}")

    grouped = advice.get("grouped") or {"1x2": advice.get("markets") or []}
    for key, label in GROUP_LABEL.items():
        block = grouped.get(key) or []
        if not block:
            continue
        st.subheader(label)
        _table(block)

    scores = advice.get("most_likely_scores") or []
    if scores:
        st.subheader("Risultati più probabili")
        st.bar_chart({s["score"]: s["prob"] for s in scores}, x_label="Risultato", y_label="Probabilità")
    st.caption(
        "Quote 1X2 e Over/Under da football-data.co.uk e, se disponibili, AsianBetSoccer (Bet365). "
        "Le coppe (Champions, Europa League, Libertadores, AFC Champions, …) arrivano da football-data.org "
        "e, come fallback, TheSportsDB/AsianBetSoccer. "
        "Le variazioni apertura→attuale (1X2, handicap, totale) confermano o scontano il voto: "
        "non sostituiscono il modello. Confronta sempre con il tuo bookmaker."
    )


st.title("Consiglio mercati")
st.caption(
    "Tre livelli: **modello** (soldi: EV/Kelly/Gioca), **voto unificato** (ordine in tabella), "
    "**fonti extra** (quadro, non cambiano EV). A sinistra basta **Aggiorna dati + modello**."
)
with st.expander("Come funziona — cosa fa ogni pezzo", expanded=False):
    st.markdown(
        """
**Cosa guardare in tabella (4 colonne)**

1. **Azione** — *Gioca* = modello + quota reale, con edge. *No bet* = analisi fatta, ma edge basso o mercato contrario. *N/D* = squadre fuori dal modello: le fonti esterne **validano**, non generano pick (voto massimo 3/10). *Invalido* = senza quote non esistono edge/EV/Kelly/quota equa/CLV, quindi **nessun pick**.
2. **Voto unificato** — voto 1–10 per *ordinare* le partite. Mix di value, Kelly, movimento Asian e workflow (ML/MC, mercato, EV, Kelly, forma, tattica). Se EV/Kelly/edge sono N/D il voto è **al massimo 3/10**. Non è una quota da giocare.
3. **EV cons.** — quanto il book paga più (o meno) della probabilità del modello. Solo con quota reale. Vuoto = N/D.
4. **Gioca / Mercato** — il consiglio (1, X, Over 2.5, …). Vuoto o "—" se il pick è invalido o N/D.

Togli la spunta **Nascondi no-bet** se vuoi vedere Premier/Serie A/Liga: quelle partite *hanno* il voto, l'azione è No bet.

---

**Cosa il modello NON vede (e non inventiamo)**

- Assenze / infortuni pesati (un centrale titolare ≠ un esterno): niente XI live ufficiale gratis. Wyscout è a pagamento. Il **movimento Asian** è il proxy di mercato sull'XI.
- Motivazioni (salvezza, coppa, clima).
- Pressing vero tipo PPDA Wyscout: FBref pubblico non lo dà più. Usiamo possesso, cross, distanza tiri, recuperi.

**Cosa abbiamo aggiunto**

- **Matchup tattico** (FBref Big 5): possesso vs blocco, cross vs difesa stretta, transizioni.
- **Fatica / calendario**: riposo reale fino al kickoff, 3 partite in 7 giorni, flag viaggi (MLS, Brasile, Argentina, …). Entra anche nelle feature al prossimo training.
- **Tre combo nel voto unificato (18%)**: 1) FBref+Sofascore+WhoScored stile · 2) assenze WhoScored × peso xG+xA FBref · 3) value+Asian. EV/Kelly restano sul modello, non sulla formula tattica.
- **WhoScored** (bottone, lento): preview con assenze *confermate*. Transfermarkt market value non ha libreria ufficiale: il peso è il contributo FBref in campo.
- **Validazione automatica** (non entra in EV/Kelly): stadio neutro/alternativo (−2%/−1% su P casa, piccolo taglio al voto), tactical score vs favorito (±0.5), gap modello–mercato >15 pp (voto value −1), ML vs Monte Carlo grezzo >8% (voto probabilità −1), forma ultime 5 (±0.3, warning se risultati ≠ xG). Stadio da football-data.org / API-Football / TheSportsDB, non da scraping.

---

**Livello 1 — Modello (unico che muove i soldi)**

Storico e quote **football-data.co.uk** (Premier, Championship, Serie A, Liga, Bundesliga, Ligue 1, extra).  
Allenano il modello e danno probabilità 1X2. Confrontate con la quota book → EV, Kelly, Gioca/No bet.

---

**Livello 2 — Calendario (vedere tutte le partite)**

Altre fonti *aggiungono righe* al calendario, spesso N/D:

| Strumento | A cosa serve | Entra in EV? |
|---|---|---|
| **football-data.org** (token) | Coppe: Champions, Europa, Conference, … | No, solo se le squadre sono già nel modello |
| **TheSportsDB / API-Football / openfootball / OpenLigaDB** | Riempire il calendario (mondo, Germania, Big 5) | No |
| **Quote AsianBetSoccer** | Quote Bet365 + movimento apertura→attuale | No in EV; sì nel voto unificato (gamba Asian) |
| **Tipster** (Forebet, PredictZ, Vitibet) | Consenso siti pubblici | No in EV; sì nel quadro/voto |

---

**Livello 3 — Quadro (validazione, non generazione)**

Non ricalcolano EV/Kelly e **non creano pick**. Se ci sono, pesano al massimo un pezzo piccolo del voto (e solo dopo modello+quota):

| Strumento | Cosa legge |
|---|---|
| **ClubElo** | Forza storica Elo |
| **FBref / Understat** | Stats / xG (Big 5 circa) |
| **StatsBomb / Sofascore** | Open data / classifica; copertura stretta |
| **Storico locale (SQLite)** | Le partite *tue* già viste (anche N/D). Dopo 30 esiti e 6 match/squadra entra al 12% del voto |

---

**Bottoni a sinistra**

- **Aggiorna dati + modello** — uso quotidiano / dopo una pausa. Scarica storico+quote football-data.co.uk, riallena il modello, ricostruisce il calendario e lo storico locale. Minuti, non secondi.
- **Solo quote e calendario** — stesso giorno, più partite o quote cambiate. Aggiorna fixtures/quote (e prova Asian/coppe/ClubElo) **senza** riallenare. Più veloce.
- **Scarica coppe** — solo se vuoi Champions/Europa/Conference/ecc. Serve il token gratis di football-data.org. Senza token quelle coppe non compaiono.
- **Scarica quote AsianBetSoccer** — movimento apertura→attuale (Bet365). Utile prima di fidarti del voto unificato; non cambia EV. Dopo il ricalcolo calendario, i spread **Raro (≥1)** partono sullo stesso bot Telegram delle offerte.
- **Scarica pronostici tipster** — Forebet / PredictZ / Vitibet nel quadro. Opzionale, rumore sulle leghe minori.
- **Calibra probabilità (backtest)** — raro: dopo tante partite chiuse o un riallenamento grosso. Taratura T e soglia EV. Non è un refresh quotidiano.
- **FBref / Understat / StatsBomb / Sofascore** — contesto Big 5 (stile, xG, classifica). Solo quadro/voto. 1–2 volte a settimana basta.
- **WhoScored assenze (lento)** — Selenium, max ~18 preview Big 5. Usalo quando ti servono gli XI confermati, non a ogni refresh.
        """
    )

with st.sidebar:
    st.header("Uso quotidiano")
    st.caption("Basta il bottone in alto: Aggiorna dati + modello. Gli altri sono opzionali.")
    if st.button("Aggiorna dati + modello", type="primary", width="stretch"):
        with st.spinner("Download, training e calendario: può richiedere alcuni minuti…"):
            proc = _run_cli("--update")
        if proc.returncode != 0:
            st.error(proc.stderr[-1500:] or proc.stdout[-1500:] or "Errore aggiornamento")
        else:
            st.success("Dati aggiornati")
            st.rerun()
    if st.button("Solo quote e calendario", width="stretch"):
        with st.spinner("Scarico fixtures e ricalcolo pronostici…"):
            proc = _run_cli("--odds-update")
        if proc.returncode != 0:
            st.error(proc.stderr[-1500:] or proc.stdout[-1500:] or "Errore quote")
        else:
            st.success("Quote aggiornate")
            st.rerun()
    try:
        from modules.data_update.history import history_summary

        hs = history_summary()
        w = int(round(float(hs.get("weight") or 0.12) * 100))
        if hs.get("ready"):
            st.caption(
                f"Storico locale: {hs['n_history']} partite, {hs['n_settled']} chiuse. "
                f"Nel voto unificato al {w}%."
            )
        else:
            st.caption(
                f"Storico locale: {hs['n_settled']}/{hs['min_global']} chiuse "
                f"(poi {w}% del voto). Si riempie da solo ad ogni aggiornamento."
            )
    except Exception:
        st.caption("Storico locale: si crea al primo aggiornamento calendario.")
    try:
        from modules.notify.telegram import telegram_status

        st.caption(telegram_status())
        st.caption(
            "Cloud GitHub ogni 30 min: Asian + Betfair + Pinnacle (~1 fetch Pinnacle/giorno) "
            "e avvisi Telegram (spread Raro; voti ≥9 se c’è il modello dal train). "
            "Secret: TELEGRAM_*, FOOTBALL_DATA_ORG_TOKEN, ODDS_API_KEY, BETFAIR_APP_KEY, "
            "BETFAIR_USERNAME, BETFAIR_PASSWORD."
        )
    except Exception:
        st.caption("Telegram: modulo avvisi non disponibile.")

    with st.expander("Coppe (Champions, Europa, …)"):
        st.caption("Token gratis su football-data.org. Serve solo per le coppe europee/mondiali.")
        org_token = st.text_input(
            "Token football-data.org",
            type="password",
            help="Gratis su football-data.org/client/register.",
        )
        if org_token.strip():
            save_org_token(org_token.strip())
        if org_token_configured():
            st.caption("Token impostato.")
        else:
            st.caption("Senza token le coppe UEFA non entrano in calendario.")
        if st.button("Scarica coppe", width="stretch"):
            if not org_token_configured():
                st.error("Incolla prima il token.")
            else:
                with st.spinner("GET /v4/matches e ricalcolo calendario…"):
                    from modules.data_update.upcoming import build_upcoming

                    info = download_org_cups()
                    upcoming_n = len(build_upcoming())
                if not info.get("token"):
                    st.error("Token assente o non letto.")
                elif info.get("error"):
                    st.error(str(info["error"]))
                else:
                    n = info.get("n_cup_fixtures") or 0
                    comps = ", ".join(info.get("competitions") or []) or "nessuna coppa in finestra"
                    st.success(f"Coppe: {n} match · {comps} · calendario {upcoming_n} partite")
                    st.rerun()

    with st.expander("Quote Pinnacle (The Odds API)"):
        st.caption("Chiave gratis su the-odds-api.com · 500 chiamate/mese · 1 fetch/giorno basta.")
        from modules.data_update.odds_api import _api_key as _pinn_key, remaining_calls, save_api_key as _save_pinn_key
        pinn_configured = bool(_pinn_key())
        pinn_rem = remaining_calls()
        if pinn_configured:
            st.caption(f"Chiave impostata. Chiamate rimanenti: {pinn_rem if pinn_rem is not None else 'n/d'}/mese.")
        odds_api_key = st.text_input(
            "Chiave The Odds API",
            type="password",
            help="Gratis su the-odds-api.com/account. Inserisci e premi Scarica.",
        )
        if odds_api_key.strip():
            _save_pinn_key(odds_api_key.strip())
            st.caption("Chiave salvata.")
        if st.button("Scarica quote Pinnacle", width="stretch"):
            if not bool(_pinn_key()):
                st.error("Incolla prima la chiave API.")
            else:
                with st.spinner("Scarico quote Pinnacle…"):
                    from modules.data_update.odds_api import fetch_pinnacle_odds
                    from modules.data_update.upcoming import build_upcoming
                    pinn = fetch_pinnacle_odds(force=True)
                    upcoming_n = len(build_upcoming())
                if not pinn.get("ok"):
                    st.error(pinn.get("error") or "Errore fetch Pinnacle")
                else:
                    rem = pinn.get("remaining")
                    st.success(
                        f"Pinnacle: {pinn.get('n_events', 0)} partite · "
                        f"chiamate rimanenti {rem if rem is not None else 'n/d'} · "
                        f"calendario {upcoming_n} partite"
                    )
                    st.rerun()

    with st.expander("Quote Betfair Exchange"):
        st.caption(
            "Delayed App Key già salvata. Serve anche username e password Betfair.it "
            "(login API, non vanno in git). Dati ritardati, gratis."
        )
        from modules.data_update.betfair import (
            app_key_configured,
            fetch_betfair_odds,
            login_configured,
            save_credentials,
        )
        if app_key_configured():
            st.caption("App Key impostata (FP3 Delayed).")
        else:
            st.caption("App Key mancante.")
        bf_user = st.text_input("Username Betfair", value="annaabbaaaa@gmail.com")
        bf_pwd = st.text_input(
            "Password Betfair",
            type="password",
            help="Salvata solo in .env locale. Necessaria per scaricare le quote.",
        )
        if bf_user.strip() and bf_pwd.strip():
            save_credentials(bf_user.strip(), bf_pwd.strip())
            st.caption("Credenziali salvate nel .env.")
        if st.button("Scarica quote Betfair", width="stretch"):
            if not login_configured():
                st.error("Inserisci username e password Betfair, poi riprova.")
            else:
                with st.spinner("Login e download quote Exchange…"):
                    from modules.data_update.upcoming import build_upcoming
                    bf = fetch_betfair_odds(force=True)
                    upcoming_n = len(build_upcoming())
                if not bf.get("ok"):
                    st.error(bf.get("error") or "Errore fetch Betfair")
                else:
                    st.success(
                        f"Betfair: {bf.get('n_events', 0)} partite · calendario {upcoming_n} partite"
                    )
                    st.rerun()

    with st.expander("Quote Asian e tipster"):
        st.caption("Opzionali. Asian = movimento quote Bet365. Tipster = consenso siti, non EV.")
        if st.button("Scarica quote AsianBetSoccer", width="stretch"):
            with st.spinner("Scarico quote Bet365 da AsianBetSoccer…"):
                proc = _run_cli("--asian-odds")
            if proc.returncode != 0:
                st.error(proc.stderr[-1500:] or proc.stdout[-1500:] or "Errore quote Asian")
            else:
                st.success("Quote AsianBetSoccer aggiornate")
                st.rerun()
        if st.button("Test notifica Telegram", width="stretch"):
            from modules.notify import ping_bot
            from modules.notify.telegram import telegram_status

            ok = ping_bot()
            if ok:
                st.success("Ping inviato sullo stesso bot delle offerte.")
            else:
                st.error(telegram_status())
        if st.button("Scarica pronostici tipster", width="stretch"):
            with st.spinner("Scarico Forebet, PredictZ e Vitibet…"):
                proc = _run_cli("--tipsters")
            if proc.returncode != 0:
                st.error(proc.stderr[-1500:] or proc.stdout[-1500:] or "Errore tipster")
            else:
                st.success("Tipster aggiornati")
                st.rerun()
        if st.button("Calibra probabilità (backtest)", width="stretch"):
            with st.spinner("Temperature scaling + backtest EV su storico…"):
                proc = _run_cli("--calibrate")
            if proc.returncode != 0:
                st.error(proc.stderr[-1500:] or proc.stdout[-1500:] or "Errore calibrazione")
            else:
                st.success("Calibrazione completata")
                st.rerun()
        cal = load_calibration()
        if cal.get("fitted_at"):
            st.caption(
                f"Calibrazione: T={cal.get('temperature', 1):.2f}, "
                f"EV min={cal.get('min_ev_play', 0.025):.0%}"
            )

    with st.expander("Contesto extra (non entra in EV)"):
        st.caption("Solo quadro/voto unificato. FBref e Understat coprono soprattutto le Big 5.")
        if st.button("FBref", width="stretch"):
            with st.spinner("Scarico statistiche squadra FBref e aggiorno calendario…"):
                from modules.data_update.upcoming import build_upcoming

                info = download_fbref_context()
                upcoming_n = len(build_upcoming())
            if info.get("error"):
                st.error(f"FBref: {info['error']}")
            else:
                st.success(f"FBref: {info.get('n_teams', 0)} squadre · calendario {upcoming_n}")
                st.rerun()
        if st.button("Understat xG", width="stretch"):
            with st.spinner("Scarico xG Understat e aggiorno calendario…"):
                from modules.data_update.upcoming import build_upcoming

                info = download_understat_context()
                upcoming_n = len(build_upcoming())
            if info.get("error"):
                st.error(f"Understat: {info['error']}")
            else:
                st.success(f"Understat: {info.get('n_teams', 0)} squadre · calendario {upcoming_n}")
                st.rerun()
        if st.button("StatsBomb open data", width="stretch"):
            with st.spinner("Scarico partite StatsBomb open data e aggiorno calendario…"):
                from modules.data_update.upcoming import build_upcoming

                info = download_statsbomb_context()
                upcoming_n = len(build_upcoming())
            if info.get("error") and not info.get("n_teams"):
                st.error(f"StatsBomb: {info['error']}")
            else:
                st.success(
                    f"StatsBomb: {info.get('n_teams', 0)} squadre · calendario {upcoming_n}"
                )
                st.rerun()
        if st.button("Sofascore classifica", width="stretch"):
            with st.spinner("Scarico classifiche Sofascore…"):
                from modules.data_update.upcoming import build_upcoming

                info = download_sofascore_context()
                upcoming_n = len(build_upcoming())
            if info.get("error") and not info.get("n_teams"):
                st.error(f"Sofascore: {info['error']}")
            else:
                st.success(f"Sofascore: {info.get('n_teams', 0)} squadre · calendario {upcoming_n}")
                st.rerun()
        if st.button("WhoScored assenze (lento)", width="stretch"):
            with st.spinner("Preview WhoScored: assenze confermate, max 18 partite Big 5…"):
                from modules.data_update.upcoming import build_upcoming
                from modules.data_update.whoscored_context import download_whoscored_context

                info = download_whoscored_context()
                upcoming_n = len(build_upcoming())
            if info.get("error") and not info.get("n_missing"):
                st.error(f"WhoScored: {info['error']}")
            else:
                st.success(
                    f"WhoScored: {info.get('n_missing', 0)} assenze · "
                    f"{info.get('n_games', 0)} preview · calendario {upcoming_n}"
                )
                st.rerun()

upcoming = _load_upcoming_enriched(
    UPCOMING.stat().st_mtime if UPCOMING.exists() else 0.0,
    (ROOT / "data" / "raw" / "asian_odds.json").stat().st_mtime
    if (ROOT / "data" / "raw" / "asian_odds.json").exists()
    else 0.0,
)
tab_cal, tab_mkt, tab_one, tab_eval = st.tabs(
    ["Calendario", "Tutti i mercati", "Singola partita", "Valutazione"]
)

with tab_cal:
    if not upcoming:
        st.info("Nessun calendario. Premi **Aggiorna dati + modello** nella colonna a sinistra.")
    else:
        with st.expander("Dettaglio colonne (opzionale)", expanded=False):
            st.markdown(
                """
                - **Voto** (non unificato): voto interno del consiglio (1–10) sul mercato scelto.
                - **Kelly ¼**: frazione di bankroll suggerita; zero se No bet.
                - **CLV**: quota accorciata dopo l'apertura = mercato andato a favore del pick.
                - **Edge pp / EV cons.**: value vs book (solo partite coperte dal modello).
                - **Movimento / Δ 1 X 2**: steam Asian (apertura→attuale). Non è EV.
                - **Quadro / Tipster**: quante fonti esterne concordano col pick. Non è EV.
                """
            )
        df = pd.DataFrame(upcoming)
        df = _filter_by_date(df)
        n_steam = int(df["movement_level"].notna().sum()) if "movement_level" in df.columns else 0
        st.caption(
            f"Steam Asian agganciato su **{n_steam}** partite di questo calendario "
            f"(su {len(df)}). Il filtro *Leggero+* nasconde le stabili e chi non ha match di nomi."
        )
        countries = sorted(df["country"].dropna().unique())
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            sel_country = st.multiselect("Paese", countries, default=countries)
        with f2:
            league_opts = sorted(df.loc[df["country"].isin(sel_country), "league"].dropna().unique())
            sel_league = st.multiselect("Campionato", league_opts, default=league_opts)
        with f3:
            group_opts = list(GROUP_LABEL.keys())
            sel_groups = st.multiselect("Tipo consiglio", group_opts, default=group_opts, format_func=lambda g: GROUP_LABEL[g])
        with f4:
            min_score = st.slider("Voto minimo", 1, 10, 1)

        q1, q2, q3, q4 = st.columns(4)
        with q1:
            odd_min = st.number_input("Quota min", min_value=1.01, max_value=30.0, value=1.01, step=0.05)
        with q2:
            odd_max = st.number_input("Quota max", min_value=1.01, max_value=30.0, value=15.0, step=0.05)
        with q3:
            min_ev = st.slider("EV cons. minimo", min_value=-0.40, max_value=0.40, value=-0.40, step=0.02, format="%.2f")
        with q4:
            only_value = st.checkbox("Solo EV cons. positivo", value=False)
        aligned_only = st.checkbox("Solo allineati al mercato asiatico", value=False)
        hide_nbet = st.checkbox("Nascondi no-bet (edge basso o mercato contrario)", value=True)
        s1, s2, s3 = st.columns(3)
        with s1:
            sort_mode = st.selectbox(
                "Ordina per",
                ["Voto unificato", "Data (più vicine)", "EV cons. %", "Movimento mercato (maggiore)", "Value (edge vs mercato)", "Consiglio (voto)"],
            )
        with s2:
            only_asian = st.checkbox("Solo partite con quote Asian", value=False)
        with s3:
            min_move = st.selectbox("Movimento minimo", MOVE_FILTER_OPTIONS, index=0)

        view = df[df["country"].isin(sel_country) & df["league"].isin(sel_league)].copy()
        if "pick_group" in view.columns:
            view = view[view["pick_group"].fillna("1x2").isin(sel_groups)]
        view = view[view["score"].isna() | (view["score"] >= min_score)]
        view["quota_pick"] = view.apply(_quota_consiglio, axis=1)
        view = view[view["quota_pick"].isna() | ((view["quota_pick"] >= odd_min) & (view["quota_pick"] <= odd_max))]
        ev_col = view["ev_cons"] if "ev_cons" in view.columns else view["ev"]
        if only_value:
            view = view[ev_col.fillna(-1) > 0]
        else:
            view = view[ev_col.fillna(min_ev) >= min_ev]
        if aligned_only and "market_align" in view.columns:
            view = view[view["market_align"] == "allineato"]
        n_nbet = int((view["action"] == "no_bet").sum()) if "action" in view.columns else 0
        n_nbet_uni = 0
        if "action" in view.columns and "score_unified" in view.columns:
            n_nbet_uni = int(((view["action"] == "no_bet") & view["score_unified"].notna()).sum())
        if hide_nbet and "action" in view.columns:
            view = view[view["action"].fillna("gioca") != "no_bet"]
        if only_asian:
            has_asian = pd.Series(False, index=view.index)
            if "movement_level" in view.columns:
                has_asian = has_asian | view["movement_level"].notna()
            if "odds_source" in view.columns:
                has_asian = has_asian | (view["odds_source"] == "asianbetsoccer")
            if "market_move" in view.columns:
                has_asian = has_asian | view["market_move"].notna()
            view = view[has_asian]
        min_rank = MOVE_FILTER_RANK[min_move]
        if min_rank > 0 and "movement_level" in view.columns:
            view = view[view["movement_level"].map(MOVE_RANK).fillna(0) >= min_rank]
        view = _sort_calendario(view, sort_mode)

        st.write(f"{len(view)} partite dopo i filtri (su {len(df)})")
        if hide_nbet and n_nbet:
            st.caption(
                f"Nascoste **{n_nbet} no-bet** ({n_nbet_uni} con voto unificato già calcolato). "
                "Togli la spunta *Nascondi no-bet* per vederle: Premier, Serie A, Liga, Championship, ecc. "
                "No-bet = edge basso o mercato contrario, non assenza di analisi."
            )
        nd = int((view["action"] == "n/d").sum()) if "action" in view.columns else 0
        n_inv = int((view["action"] == "invalido").sum()) if "action" in view.columns else 0
        if nd:
            st.caption(
                f"Di cui {nd} N/D (squadre senza storico nel modello). "
                "Le fonti esterne (ClubElo, FBref, tipster, …) restano nel quadro come **validazione**, "
                "non generano un pick. Voto unificato massimo 3/10."
            )
        if n_inv:
            st.caption(
                f"Di cui {n_inv} pick **invalidi** (quote assenti): senza book non si calcolano "
                "edge, EV, Kelly, quota equa, CLV."
            )
        st.dataframe(_prepare_calendario_show(view), width="stretch", hide_index=True)

        with st.expander("Tutte le partite scaricate (anche senza pronostico)", expanded=False):
            try:
                all_fx = load_fixtures().copy()
                all_fx["date_s"] = pd.to_datetime(all_fx["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                key_pred = {
                    (str(r.get("date")), str(r.get("home")), str(r.get("away")))
                    for r in upcoming
                }
                all_fx["in_predictor"] = all_fx.apply(
                    lambda r: (str(r.get("date_s")), str(r.get("home_team")), str(r.get("away_team"))) in key_pred,
                    axis=1,
                )
                all_fx["stato"] = all_fx["in_predictor"].map(lambda x: "Pronosticata" if x else "Non coperta dal modello")
                st.caption(
                    f"Scaricate: {len(all_fx)} partite · Pronosticate: {int(all_fx['in_predictor'].sum())} · "
                    f"Non coperte: {int((~all_fx['in_predictor']).sum())}"
                )
                show_cols = [
                    "date_s", "time", "country", "league", "home_team", "away_team", "source", "stato"
                ]
                show = all_fx[[c for c in show_cols if c in all_fx.columns]].rename(
                    columns={
                        "date_s": "Data",
                        "time": "Ora",
                        "country": "Paese",
                        "league": "Campionato",
                        "home_team": "Casa",
                        "away_team": "Trasferta",
                        "source": "Fonte",
                        "stato": "Stato",
                    }
                )
                for col in ("Paese", "Campionato", "Casa", "Trasferta"):
                    if col in show.columns:
                        show[col] = show[col].map(_display_text)
                st.dataframe(show.sort_values(["Data", "Ora"], ascending=True), width="stretch", hide_index=True)
            except Exception as exc:
                st.error(f"Errore nel riepilogo partite scaricate: {exc}")

        with st.expander("Dettagli voto (+)", expanded=False):
            max_rows = st.slider("Massimo righe dettaglio", 5, 60, 20, 5, key="cal_plus_max")
            for _, row in view.head(max_rows).iterrows():
                base = row.get("score")
                uni = row.get("score_unified")
                hdr = f"+ {row.get('date')} {row.get('home')} vs {row.get('away')}"
                if pd.notna(base):
                    hdr += f" · voto {int(base)}/10"
                if pd.notna(uni):
                    hdr += f" · unificato {int(uni)}/10"
                if row.get("action") == "n/d":
                    hdr += " · N/D"
                elif row.get("action") == "invalido":
                    hdr += " · invalido"
                with st.expander(hdr, expanded=False):
                    st.write(f"**Giocata:** {row.get('pick')} — {row.get('pick_name')}")
                    if row.get("action") == "no_bet":
                        st.warning("No bet")
                    if row.get("action") == "n/d":
                        st.info("Nessun pick: senza modello le fonti esterne validano, non generano. Voto max 3/10.")
                    if row.get("action") == "invalido":
                        st.error("Pick invalido: senza quote non si calcolano edge, EV, Kelly, quota equa, CLV.")
                    if row.get("score_reason_1"):
                        st.caption(str(row.get("score_reason_1")))
                    if row.get("score_reason_2"):
                        st.caption(str(row.get("score_reason_2")))
                    if row.get("meta_note"):
                        st.caption(f"Mix: {row.get('meta_note')}")
                    val = row.get("validation") if isinstance(row.get("validation"), dict) else None
                    _render_validation(val)
                    quadro = row.get("quadro") if isinstance(row.get("quadro"), dict) else None
                    sources = None if not quadro else quadro.get("sources")
                    if sources:
                        st.dataframe(
                            [
                                {
                                    "Fonte": s.get("fonte"),
                                    "Lean": s.get("pick"),
                                    "Nota": s.get("nota") or "—",
                                }
                                for s in sources
                            ],
                            width="stretch",
                            hide_index=True,
                        )

        # Vista volutamente tabellare: niente passaggi extra obbligatori.

with tab_mkt:
    if not upcoming:
        st.info("Nessun calendario.")
    else:
        flat_rows = []
        for match in upcoming:
            for m in match.get("markets") or []:
                src = str(m.get("odds_source") or "")
                real = m.get("odds_real")
                if real is None:
                    real = bool(src) and not src.startswith("stimata")
                edge = m.get("edge_pp") if real else None
                ev_cons = m.get("ev_cons") if real else None
                if ev_cons is None and real:
                    ev_cons = m.get("ev")
                voto_value = m.get("score_value") if real else None
                flat_rows.append(
                    {
                        "Data": match["date"],
                        "Paese": match["country"],
                        "Campionato": match["league"],
                        "Partita": f"{match['home']} vs {match['away']}",
                        "Gruppo": GROUP_LABEL.get(m.get("group"), m.get("group")),
                        "group": m.get("group"),
                        "Mercato": m["name"],
                        "Codice": m["code"],
                        "Prob.": f"{m['probability']:.0%}",
                        "prob_num": m["probability"],
                        "Quota book": m.get("odds"),
                        "Quota equa": m.get("fair_odds"),
                        "Edge pp": _pct(edge),
                        "value_num": edge,
                        "EV cons.": _pct(ev_cons),
                        "EV sharp": _pct(m.get("ev_sharp") if real else None),
                        "ev_num": ev_cons,
                        "Voto value": voto_value if voto_value is not None else "—",
                        "Voto": m.get("score") or voto_value or m.get("score_prob"),
                        "Fonte": src or "—",
                        "match_source": match.get("odds_source") or "—",
                        "odds_real": real,
                        "Movimento": match.get("movement_level"),
                        "Var linea": match.get("line_move"),
                        "Cosa è cambiato": match.get("movement_summary"),
                        "Commento quote": match.get("movement_comment") or match.get("market_note"),
                    }
                )
        flat = pd.DataFrame(flat_rows)
        if not flat.empty and "Data" in flat.columns:
            dates = pd.to_datetime(flat["Data"], errors="coerce").dt.date
            valid = dates.dropna()
            if not valid.empty:
                dmin, dmax = valid.min(), valid.max()
                picked = st.date_input("Intervallo date", value=(dmin, dmax), min_value=dmin, max_value=dmax, key="mkt_dates")
                if isinstance(picked, tuple):
                    d1, d2 = picked[0], picked[-1]
                else:
                    d1 = d2 = picked
                flat = flat[((dates >= d1) & (dates <= d2)).fillna(False)]
        g1, g2, g3, g4 = st.columns(4)
        groups = sorted(flat["group"].dropna().unique().tolist())
        sel_g = g1.multiselect("Gruppo", groups, default=groups, format_func=lambda g: GROUP_LABEL.get(g, g))
        min_p = g2.slider("Prob. minima", 0.0, 1.0, 0.45, 0.01)
        qmin = g3.number_input("Quota min", 1.01, 30.0, 1.40, 0.05, key="mkt_qmin")
        qmax = g4.number_input("Quota max", 1.01, 30.0, 3.50, 0.05, key="mkt_qmax")
        e1, e2, e3 = st.columns(3)
        min_ev_m = e1.slider("EV cons. minimo", -0.40, 0.40, 0.0, 0.02, key="mkt_ev")
        min_voto = e2.slider("Voto minimo", 1, 10, 5, key="mkt_voto")
        sort_mkt = e3.selectbox(
            "Ordina per",
            ["Value (edge vs mercato)", "Movimento mercato", "EV cons.", "Voto"],
            key="mkt_sort",
        )
        f1, f2, f3 = st.columns(3)
        src_opts = sorted(flat["Fonte"].dropna().unique().tolist())
        sel_src = f1.multiselect("Fonte quota mercato", src_opts, default=src_opts)
        only_asian_match = f2.checkbox("Solo partite con quote Asian", value=False, key="mkt_asian")
        min_move_m = f3.selectbox("Movimento minimo", MOVE_FILTER_OPTIONS, index=0, key="mkt_move")
        hide_phantom = st.checkbox("Nascondi quote stimate (senza voto value)", value=True, key="mkt_real")
        filt = flat[flat["group"].isin(sel_g)]
        filt = filt[filt["Fonte"].isin(sel_src)]
        if only_asian_match:
            filt = filt[filt["match_source"] == "asianbetsoccer"]
        if hide_phantom and "odds_real" in filt.columns:
            filt = filt[filt["odds_real"].fillna(False)]
        filt = filt[filt["prob_num"] >= min_p]
        filt = filt[filt["Voto"].fillna(0) >= min_voto]
        filt = filt[filt["ev_num"].fillna(-1) >= min_ev_m]
        has_q = filt["Quota book"].notna()
        filt = filt[~has_q | ((filt["Quota book"] >= qmin) & (filt["Quota book"] <= qmax))]
        min_rank_m = MOVE_FILTER_RANK[min_move_m]
        if min_rank_m > 0:
            filt = filt[filt["Movimento"].map(MOVE_RANK).fillna(0) >= min_rank_m]
        if sort_mkt == "Movimento mercato":
            filt["_ord"] = filt["Movimento"].map(MOVE_RANK).fillna(-1)
            filt["_line"] = filt["Var linea"].fillna(0)
            filt = filt.sort_values(["_ord", "_line", "Voto"], ascending=False, na_position="last").drop(columns=["_ord", "_line"])
        elif sort_mkt == "Value (edge vs mercato)":
            filt = filt.sort_values(["value_num", "ev_num"], ascending=False, na_position="last")
        elif sort_mkt == "Voto":
            filt = filt.sort_values("Voto", ascending=False, na_position="last")
        else:
            filt = filt.sort_values(["ev_num", "Voto"], ascending=False, na_position="last")
        st.write(f"{len(filt)} mercati dopo i filtri")
        hide = {"group", "match_source", "prob_num", "value_num", "ev_num", "odds_real"}
        st.dataframe(
            filt.drop(columns=[c for c in hide if c in filt.columns]),
            width="stretch",
            hide_index=True,
        )

with tab_one:
    teams = list_known_teams() if (ROOT / "data" / "processed" / "features.csv").exists() else []
    if not teams:
        st.info("Allena prima il modello con **Aggiorna dati + modello**.")
    else:
        try:
            meta = list_team_meta()
            countries = ["Tutti"] + sorted(meta["country"].dropna().unique().tolist()) if "country" in meta.columns else ["Tutti"]
        except Exception:
            meta = pd.DataFrame({"team": teams})
            countries = ["Tutti"]
        ctry = st.selectbox("Filtra squadre per paese", countries)
        if ctry != "Tutti" and "country" in meta.columns:
            team_opts = sorted(meta.loc[meta["country"] == ctry, "team"].unique().tolist())
        else:
            team_opts = teams
        last = _load_json(LAST) or {}
        default_home, default_away = "Inter", "Milan"
        if last.get("match") and " vs " in last["match"]:
            default_home, default_away = last["match"].split(" vs ", 1)
        hcol, acol = st.columns(2)
        home = hcol.selectbox("Casa", team_opts, index=team_opts.index(default_home) if default_home in team_opts else 0)
        away = acol.selectbox("Trasferta", team_opts, index=team_opts.index(default_away) if default_away in team_opts else min(1, len(team_opts) - 1))
        o1, ox, o2 = st.columns(3)
        odd_1 = o1.number_input("Quota 1", min_value=1.01, max_value=50.0, value=None, placeholder="es. 1.85", step=0.05, format="%.2f")
        odd_x = ox.number_input("Quota X", min_value=1.01, max_value=50.0, value=None, placeholder="es. 3.40", step=0.05, format="%.2f")
        odd_2 = o2.number_input("Quota 2", min_value=1.01, max_value=50.0, value=None, placeholder="es. 4.50", step=0.05, format="%.2f")
        with st.expander("Altre quote (se le hai dal book)"):
            r1 = st.columns(4)
            odd_o15 = r1[0].number_input("Over 1.5", min_value=1.01, max_value=20.0, value=None, step=0.05, format="%.2f")
            odd_u15 = r1[1].number_input("Under 1.5", min_value=1.01, max_value=20.0, value=None, step=0.05, format="%.2f")
            odd_o25 = r1[2].number_input("Over 2.5", min_value=1.01, max_value=20.0, value=None, step=0.05, format="%.2f")
            odd_u25 = r1[3].number_input("Under 2.5", min_value=1.01, max_value=20.0, value=None, step=0.05, format="%.2f")
            r2 = st.columns(4)
            odd_o35 = r2[0].number_input("Over 3.5", min_value=1.01, max_value=20.0, value=None, step=0.05, format="%.2f")
            odd_u35 = r2[1].number_input("Under 3.5", min_value=1.01, max_value=20.0, value=None, step=0.05, format="%.2f")
            odd_o45 = r2[2].number_input("Over 4.5", min_value=1.01, max_value=20.0, value=None, step=0.05, format="%.2f")
            odd_u45 = r2[3].number_input("Under 4.5", min_value=1.01, max_value=20.0, value=None, step=0.05, format="%.2f")
            r3 = st.columns(4)
            odd_btts = r3[0].number_input("Gol sì", min_value=1.01, max_value=20.0, value=None, step=0.05, format="%.2f")
            odd_nbtts = r3[1].number_input("Gol no", min_value=1.01, max_value=20.0, value=None, step=0.05, format="%.2f")
            odd_1x = r3[2].number_input("1X", min_value=1.01, max_value=20.0, value=None, step=0.05, format="%.2f")
            odd_x2 = r3[3].number_input("X2", min_value=1.01, max_value=20.0, value=None, step=0.05, format="%.2f")
            st.caption("Combo — risultato + Over/Under o Gol (se le hai dal book)")
            r4 = st.columns(4)
            c_1o25 = r4[0].number_input("1 + O2.5", min_value=1.01, max_value=50.0, value=None, step=0.05, format="%.2f")
            c_1u25 = r4[1].number_input("1 + U2.5", min_value=1.01, max_value=50.0, value=None, step=0.05, format="%.2f")
            c_2o25 = r4[2].number_input("2 + O2.5", min_value=1.01, max_value=50.0, value=None, step=0.05, format="%.2f")
            c_2u25 = r4[3].number_input("2 + U2.5", min_value=1.01, max_value=50.0, value=None, step=0.05, format="%.2f")
            r5 = st.columns(4)
            c_xo25 = r5[0].number_input("X + O2.5", min_value=1.01, max_value=50.0, value=None, step=0.05, format="%.2f")
            c_xu25 = r5[1].number_input("X + U2.5", min_value=1.01, max_value=50.0, value=None, step=0.05, format="%.2f")
            c_1gol = r5[2].number_input("1 + Gol", min_value=1.01, max_value=50.0, value=None, step=0.05, format="%.2f")
            c_1ng = r5[3].number_input("1 + No gol", min_value=1.01, max_value=50.0, value=None, step=0.05, format="%.2f")
            r6 = st.columns(4)
            c_2gol = r6[0].number_input("2 + Gol", min_value=1.01, max_value=50.0, value=None, step=0.05, format="%.2f")
            c_2ng = r6[1].number_input("2 + No gol", min_value=1.01, max_value=50.0, value=None, step=0.05, format="%.2f")
            c_1xo25 = r6[2].number_input("1X + O2.5", min_value=1.01, max_value=50.0, value=None, step=0.05, format="%.2f")
            c_1xu25 = r6[3].number_input("1X + U2.5", min_value=1.01, max_value=50.0, value=None, step=0.05, format="%.2f")
        n_sims = st.slider("Simulazioni Monte Carlo", 2000, 20000, 8000, 1000)
        extra_odds = {
            "1": odd_1, "X": odd_x, "2": odd_2,
            "over_1.5": odd_o15, "under_1.5": odd_u15,
            "over_2.5": odd_o25, "under_2.5": odd_u25,
            "over_3.5": odd_o35, "under_3.5": odd_u35,
            "over_4.5": odd_o45, "under_4.5": odd_u45,
            "btts_yes": odd_btts, "btts_no": odd_nbtts,
            "1X": odd_1x, "X2": odd_x2,
            "combo_1_o25": c_1o25, "combo_1_u25": c_1u25,
            "combo_2_o25": c_2o25, "combo_2_u25": c_2u25,
            "combo_x_o25": c_xo25, "combo_x_u25": c_xu25,
            "combo_1_gol": c_1gol, "combo_1_nogol": c_1ng,
            "combo_2_gol": c_2gol, "combo_2_nogol": c_2ng,
            "combo_1x_o25": c_1xo25, "combo_1x_u25": c_1xu25,
        }
        if st.button("Calcola predizione", type="primary"):
            if home == away:
                st.error("Scegli due squadre diverse.")
            else:
                with st.spinner("Calcolo modello + Monte Carlo..."):
                    pred = predict_pipeline(home, away, n_sims)
                render_advice(pred, extra_odds)
        elif last:
            st.caption("Ultima predizione salvata")
            render_advice(last, extra_odds)

with tab_eval:
    cal = load_calibration()
    summary = cal.get("backtest_summary") or {}
    if not cal.get("fitted_at") and not summary:
        st.info("Nessuna valutazione. Premi **Calibra probabilità (backtest)** nella colonna a sinistra (meglio dopo **Aggiorna dati + modello** per lo split rolling).")
    else:
        st.caption(
            "Walk-forward temporale (OOF): Brier, log-loss, ECE e CLV. "
            "Le scommesse simulate usano ¼ Kelly con cap, edge minimo 2–3% e scarto se Pinnacle non offre edge."
        )
        split = summary.get("split") or cal.get("split") or "—"
        st.caption(f"Protocollo: **{split}** · T={cal.get('temperature', 1):.2f} · EV min {cal.get('min_ev_play', 0.025):.1%}")
        with st.container(horizontal=True):
            st.metric("Brier", f"{cal.get('brier_multiclass_calibrated') or cal.get('brier_favorite_calibrated') or '—'}", border=True)
            st.metric("Log-loss", f"{cal.get('log_loss_calibrated') or summary.get('prob_log_loss') or '—'}", border=True)
            st.metric("ECE", f"{cal.get('ece_calibrated') or summary.get('prob_ece') or '—'}", border=True)
            st.metric("Calibration gap", f"{cal.get('calibration_gap_calibrated') or summary.get('prob_calibration_gap') or '—'}", border=True)
        with st.container(horizontal=True):
            st.metric("CLV medio", f"{summary.get('clv_mean_clv') or '—'}", border=True)
            beat = summary.get("clv_beat_close_rate")
            st.metric("Beat close", f"{beat:.0%}" if isinstance(beat, (int, float)) else "—", border=True)
            st.metric("Scommesse filtrate", f"{summary.get('n_bets_played') or 0}", border=True)
            bank = summary.get("bankroll_final")
            st.metric("Bankroll ¼ Kelly", f"{bank:.2f}" if isinstance(bank, (int, float)) else "—", border=True)

        path = cal.get("bankroll_path") or []
        if path:
            bank_df = pd.DataFrame(path)
            if "bankroll" in bank_df.columns:
                st.subheader("Bankroll simulato (¼ Kelly con cap)")
                st.line_chart(bank_df, x="i", y="bankroll", x_label="Scommessa", y_label="Bankroll")

        def _show_report(title: str, rows: list, rename: dict | None = None):
            if not rows:
                return
            st.subheader(title)
            df = pd.DataFrame(rows)
            if rename:
                df = df.rename(columns=rename)
            pct_cols = [c for c in df.columns if c in {"hit_rate", "roi", "mean_ev", "realization", "mean_clv", "beat_close_rate", "pnl_kelly"}]
            show = df.copy()
            for c in pct_cols:
                if c in show.columns:
                    show[c] = show[c].map(lambda x: f"{x:+.1%}" if isinstance(x, (int, float)) else x)
            st.dataframe(show, width="stretch", hide_index=True)

        _show_report(
            "Per campionato",
            cal.get("by_league") or [],
            {"league": "Campionato", "n": "N", "hit_rate": "Hit", "roi": "ROI piatto", "mean_ev": "EV medio", "realization": "Realizzazione", "mean_clv": "CLV", "beat_close_rate": "Beat close", "pnl_kelly": "PnL Kelly"},
        )
        _show_report(
            "Per mercato",
            cal.get("by_market") or [],
            {"market": "Mercato", "n": "N", "hit_rate": "Hit", "roi": "ROI piatto", "mean_ev": "EV medio", "realization": "Realizzazione", "mean_clv": "CLV", "beat_close_rate": "Beat close", "pnl_kelly": "PnL Kelly"},
        )
        _show_report(
            "Per esito",
            cal.get("by_code") or [],
            {"code": "Codice", "n": "N", "hit_rate": "Hit", "roi": "ROI piatto", "mean_ev": "EV medio", "realization": "Realizzazione", "mean_clv": "CLV", "beat_close_rate": "Beat close", "pnl_kelly": "PnL Kelly"},
        )
        folds = cal.get("by_fold") or []
        rolling = None
        metrics_path = ROOT / "data" / "models" / "metrics.json"
        if metrics_path.exists():
            try:
                rolling = json.loads(metrics_path.read_text(encoding="utf-8")).get("rolling") or {}
            except Exception:
                rolling = None
        if rolling and rolling.get("folds"):
            st.subheader("Fold rolling")
            fdf = pd.DataFrame(rolling["folds"])
            st.dataframe(fdf, width="stretch", hide_index=True)
        elif folds:
            _show_report("Fold rolling (scommesse)", folds, {"fold": "Fold"})

        rel = cal.get("reliability_1x2") or []
        if rel:
            st.subheader("Calibrazione 1X2 (bin)")
            st.dataframe(
                [
                    {
                        "Range": f"{b['range'][0]:.2f}–{b['range'][1]:.2f}",
                        "Predetta": f"{b['predicted']:.1%}",
                        "Reale": f"{b['actual']:.1%}",
                        "N": b["n"],
                        "Fattore": b.get("factor"),
                    }
                    for b in rel
                ],
                width="stretch",
                hide_index=True,
            )
        st.caption(
            "CLV storico: quota di apertura (venerdì / AvgH) contro la close (AvgCH / B365CH). "
            "Positivo = hai battuto la linea di chiusura. I tipster non entrano in queste metriche."
        )
