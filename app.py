"""Interfaccia: calendario, mercati 1X2 / O/U / gol / DC, filtri quote."""

from __future__ import annotations

import json
import subprocess
import sys
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


def _score_bar(score: int) -> str:
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


def _pp(val: float | None) -> str | None:
    if val is None:
        return None
    return f"{float(val):+.1f} pp"


def _sort_calendario(view: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = view.copy()
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
    }.get(kind, kind)


def _prepare_calendario_show(view: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "date", "time", "country", "league", "home", "away",
        "pick", "pick_name", "action", "score", "kelly_quarter", "clv", "tipster_consensus", "tipster_agree",
        "score_reason_1", "score_reason_2", "probability",
        "quota_pick", "fair_odds", "edge_pp", "ev_cons", "ev_sharp",
        "odds_real", "value_note",
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
        show["action"] = show["action"].map(lambda x: "No bet" if x == "no_bet" else "Gioca")
    for drop_col in ("drop_1", "drop_x", "drop_2"):
        if drop_col in show.columns:
            show[drop_col] = show[drop_col].map(_pp)
    return show.rename(
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
            "kelly_quarter": "Kelly ¼",
            "clv": "CLV vs apertura",
            "tipster_consensus": "Tipster",
            "tipster_agree": "Vs tipster",
            "score_reason_1": "Perché questo voto",
            "score_reason_2": "Quote e mercato",
            "probability": "Prob.",
            "quota_pick": "Quota book",
            "fair_odds": "Quota equa",
            "edge_pp": "Edge pp",
            "ev_cons": "EV cons.",
            "ev_sharp": "EV sharp",
            "odds_real": "Quota reale",
            "value_note": "Nota value",
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
        st.markdown(
            f'<p class="pick-code">GIOCA {play["code"]}</p>'
            f'<p class="pick-name">{play["name"]} · {_kind_label(play["kind"])}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(_score_bar(play["score"]), unsafe_allow_html=True)
        st.markdown(f"**{play['score']} / 10**")
        if play.get("action") == "no_bet":
            st.warning("No bet — " + "; ".join(play.get("no_bet_reasons") or ["filtro edge/mercato"]))
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
                    f"{p_cons:.1%}" if p_cons is not None else f"{play['probability']:.1%}",
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
                st.metric(
                    "Fonte",
                    play.get("odds_source") or ("ipotetica" if play.get("odds_real") is False else "—"),
                    border=True,
                )
            if play.get("value_note"):
                st.caption(play["value_note"])
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
        "Le variazioni apertura→attuale (1X2, handicap, totale) confermano o scontano il voto: "
        "non sostituiscono il modello. Confronta sempre con il tuo bookmaker."
    )


st.title("Consiglio mercati")
st.caption(
    "1X2, doppia chance, DNB, Over/Under 0.5–4.5, Gol/No gol, over squadra, combo. "
    "Dati football-data.co.uk; quote anche da AsianBetSoccer quando disponibili."
)

with st.sidebar:
    st.header("Dati")
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
    if st.button("Quote AsianBetSoccer", width="stretch"):
        with st.spinner("Scarico quote Bet365 da AsianBetSoccer…"):
            proc = _run_cli("--asian-odds")
        if proc.returncode != 0:
            st.error(proc.stderr[-1500:] or proc.stdout[-1500:] or "Errore quote Asian")
        else:
            st.success("Quote AsianBetSoccer aggiornate")
            st.rerun()
    if st.button("Pronostici tipster", width="stretch"):
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
            f"EV min={cal.get('min_ev_play', 0.025):.0%}, "
            f"Kelly cap={cal.get('kelly_cap', 0.02):.0%}, "
            f"Brier {cal.get('brier_multiclass_calibrated') or cal.get('brier_favorite_calibrated', '—')}, "
            f"ECE {cal.get('ece_calibrated', '—')}"
        )
    st.caption("Quote: football-data.co.uk + AsianBetSoccer (Bet365)")

upcoming = _load_upcoming_enriched(
    UPCOMING.stat().st_mtime if UPCOMING.exists() else 0.0,
    (ROOT / "data" / "raw" / "asian_odds.json").stat().st_mtime
    if (ROOT / "data" / "raw" / "asian_odds.json").exists()
    else 0.0,
)
tab_cal, tab_mkt, tab_one, tab_eval = st.tabs(["Calendario", "Tutti i mercati", "Singola partita", "Valutazione"])

with tab_cal:
    if not upcoming:
        st.info("Nessun calendario. Premi **Aggiorna dati + modello** nella colonna a sinistra.")
    else:
        with st.expander("Come leggere la tabella", expanded=False):
            st.markdown(
                """
                - **Kelly ¼**: frazione di bankroll da ¼ Kelly, con tetto (default 2%). Zero se scatta il no-bet.
                - **Azione**: *Gioca* solo se l'edge stimato è almeno 2–3% e il mercato Asian non è fortemente contrario.
                - **CLV vs apertura**: se la quota del pick si è accorciata dopo l'apertura, il mercato si è mosso a tuo favore (CLV positivo).
                - **Tipster**: consenso Forebet / PredictZ / Vitibet. Bilancia il voto, **non** entra nel calcolo dell'EV.
                - **Voto**: mix probabilità + value + robustezza ML/MC + Kelly ¼. Gli outsider non salgono solo per EV.
                - **Perché questo voto**: probabilità, accordo ML/MC, e se il pick è sopra/sotto soglia.
                - **Quote e mercato**: quota book vs equa, allineamento allo steam Asian, e se la quota del pick si è accorciata o allungata.
                - **Movimento**: *Stabile / Leggero / Medio / Forte* (quote), **Fortissimo** (linea AH/tot 0.5 o 0.75), **Raro** (linea ≥1).
                - **Var linea**: massimo spostamento di handicap o totale (0.25, 0.5, 0.75, 1…).
                - **Cosa è cambiato**: riassunto con quote apertura→attuale e punti percentuali impliciti (es. `1X2 verso casa 2.20->2.05 (+3.3 pp)`).
                - **Commento quote**: lettura del flusso: soldi su casa/trasferta/over/under, spostamento linee AH e totale.
                - **Δ 1 / Δ X / Δ 2**: variazione della probabilità implicita. Positivo = quota accorciata (più giocata).
                - **Vs mercato**: il consiglio del modello è *allineato* o *contrario* al movimento delle quote.
                - **Value / Edge pp**: differenza in punti percentuali tra probabilità conservativa del modello e probabilità implicita del book *dopo* de-vig. Non è `quota/equa − 1` (quello coincide con l'EV grezzo).
                - **EV cons.**: value vote sull'EV conservativo (p calibrata e scontata × quota book − 1), non sull'EV grezzo.
                - **EV sharp**: stesso calcolo sulla quota Asian/Pinnacle. Value alto solo se anche lo sharp tiene l'edge.
                - **Quota reale**: sulle quote stimate (combo senza book) non c'è voto value.
                """
            )
        df = pd.DataFrame(upcoming)
        df = _filter_by_date(df)
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
                ["Consiglio (voto)", "Movimento mercato (maggiore)", "Value (edge vs mercato)"],
            )
        with s2:
            only_asian = st.checkbox("Solo partite con quote Asian", value=False)
        with s3:
            min_move = st.selectbox("Movimento minimo", MOVE_FILTER_OPTIONS, index=0)

        view = df[df["country"].isin(sel_country) & df["league"].isin(sel_league)].copy()
        if "pick_group" in view.columns:
            view = view[view["pick_group"].fillna("1x2").isin(sel_groups)]
        view = view[view["score"] >= min_score]
        view["quota_pick"] = view.apply(_quota_consiglio, axis=1)
        view = view[view["quota_pick"].isna() | ((view["quota_pick"] >= odd_min) & (view["quota_pick"] <= odd_max))]
        ev_col = view["ev_cons"] if "ev_cons" in view.columns else view["ev"]
        if only_value:
            view = view[ev_col.fillna(-1) > 0]
        else:
            view = view[ev_col.fillna(min_ev) >= min_ev]
        if aligned_only and "market_align" in view.columns:
            view = view[view["market_align"] == "allineato"]
        if hide_nbet and "action" in view.columns:
            view = view[view["action"].fillna("gioca") != "no_bet"]
        if only_asian and "odds_source" in view.columns:
            view = view[view["odds_source"] == "asianbetsoccer"]
        min_rank = MOVE_FILTER_RANK[min_move]
        if min_rank > 0 and "movement_level" in view.columns:
            view = view[view["movement_level"].map(MOVE_RANK).fillna(0) >= min_rank]
        view = _sort_calendario(view, sort_mode)

        st.write(f"{len(view)} partite dopo i filtri (su {len(df)})")
        st.dataframe(_prepare_calendario_show(view), width="stretch", hide_index=True)

        with st.expander("Radar AsianBetSoccer — partite con movimento quote"):
            r1, r2 = st.columns(2)
            radar_opts = [o for o in MOVE_FILTER_OPTIONS if o != "Tutti"]
            radar_min = r1.selectbox(
                "Movimento minimo radar",
                radar_opts,
                index=radar_opts.index("Fortissimo+ (0.5/0.75)"),
                key="radar_spread",
            )
            radar = _asian_radar_table(MOVE_FILTER_RANK[radar_min])
            if radar.empty:
                r2.caption("Nessun dato Asian. Premi **Quote AsianBetSoccer** nella sidebar.")
            else:
                r2.caption(f"{len(radar)} partite")
                st.dataframe(radar, width="stretch", hide_index=True)

        if not view.empty:
            labels = [f"{r.date} {r.home} vs {r.away}  ({r.league})" for r in view.itertuples()]
            chosen = st.selectbox("Dettaglio partita", labels)
            row = view.iloc[labels.index(chosen)]
            render_advice(
                row["prediction"],
                row.get("odds") or {},
                row.get("market_move"),
                odds_from_asian=(row.get("odds_source") == "asianbetsoccer"),
                match_date=row.get("date"),
                league=row.get("league"),
            )

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
