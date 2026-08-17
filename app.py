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
from modules.data_update.asian_odds import load_asian_odds
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


def _value_edge(quota: float | None, fair: float | None) -> float | None:
    if quota and fair and fair > 1.01:
        return round(float(quota) / float(fair) - 1.0, 4)
    return None


def _pct(val: float | None) -> str | None:
    if val is None:
        return None
    return f"{float(val):+.0%}"


def _sort_calendario(view: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = view.copy()
    if mode == "Movimento mercato (maggiore)":
        order = {"Forte": 4, "Medio": 3, "Leggero": 2, "Stabile": 1}
        out["_sort"] = out["movement_level"].map(order).fillna(0)
        if "spread_score" in out.columns:
            out["_sort"] = out["_sort"] * 100 + out["spread_score"].fillna(0)
        return out.sort_values("_sort", ascending=False).drop(columns="_sort")
    if mode == "Value (equa vs reale)":
        out["_sort"] = out["value_edge"].fillna(out["ev"].fillna(-99))
        return out.sort_values("_sort", ascending=False).drop(columns="_sort")
    return out.sort_values(["score", "probability"], ascending=False, na_position="last")


def _asian_radar_table(min_spread: float) -> pd.DataFrame:
    rows = load_asian_odds()
    if not rows:
        return pd.DataFrame()
    data = []
    for r in rows:
        move = r.get("market_move") or {}
        spread = move.get("spread_score") or 0.0
        if spread < min_spread:
            continue
        data.append(
            {
                "Data": r.get("date"),
                "Ora": r.get("time"),
                "Campionato": r.get("league"),
                "Partita": f"{r.get('home')} vs {r.get('away')}",
                "Movimento": move.get("movement_level") or "Stabile",
                "Cosa è cambiato": move.get("movement_summary") or "Quasi nessun movimento",
                "Soldi su 1X2": move.get("steam_1x2") or "—",
                "Soldi su O/U": move.get("steam_ou") or "—",
                "Quota 1 attuale": r.get("odd_1"),
                "Quota 1 apertura": r.get("open_1"),
                "Quota 2 attuale": r.get("odd_2"),
                "Quota 2 apertura": r.get("open_2"),
            }
        )
    if not data:
        return pd.DataFrame()
    level_order = {"Forte": 0, "Medio": 1, "Leggero": 2, "Stabile": 3}
    out = pd.DataFrame(data)
    out["_ord"] = out["Movimento"].map(level_order).fillna(9)
    return out.sort_values("_ord").drop(columns="_ord")


def _kind_label(kind: str) -> str:
    return {
        "più_probabile": "Più probabile",
        "valore": "Miglior rapporto probabilità/quota",
        "probabile_e_valore": "Più probabile e miglior value",
    }.get(kind, kind)


def _prepare_calendario_show(view: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "date", "time", "country", "league", "home", "away",
        "pick", "pick_name", "score", "kelly_quarter", "score_reason_1", "probability",
        "quota_pick", "fair_odds", "value_edge", "ev",
        "movement_level", "movement_summary", "market_align",
        "odd_1", "odd_x", "odd_2", "odd_over_25", "odd_under_25", "odds_source",
    ]
    show = view[[c for c in wanted if c in view.columns]].copy()
    if "probability" in show.columns:
        show["probability"] = show["probability"].map(lambda x: f"{x:.0%}" if pd.notna(x) else None)
    if "value_edge" in show.columns:
        show["value_edge"] = show["value_edge"].map(_pct)
    if "ev" in show.columns:
        show["ev"] = show["ev"].map(_pct)
    if "kelly_quarter" in show.columns:
        show["kelly_quarter"] = show["kelly_quarter"].map(lambda x: f"{x:.1%}" if pd.notna(x) else None)
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
            "score": "Voto",
            "kelly_quarter": "Kelly ¼",
            "score_reason_1": "Perché questo voto",
            "probability": "Prob.",
            "quota_pick": "Quota book",
            "fair_odds": "Quota equa",
            "value_edge": "Value",
            "ev": "EV atteso",
            "movement_level": "Movimento",
            "movement_summary": "Cosa è cambiato",
            "market_align": "Vs mercato",
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
    rows = [
        {
            "Mercato": m["name"],
            "Codice": m["code"],
            "Prob.": f"{m['probability']:.0%}",
            "Quota book": m["odds"],
            "Quota equa": m["fair_odds"],
            "EV atteso": _pct(m["ev"]),
            "Voto prob.": m.get("score_prob"),
            "Voto value": m.get("score_value"),
            "Voto finale": m.get("score"),
            "Kelly ¼": f"{m['kelly_quarter']:.1%}" if m.get("kelly_quarter") is not None else "—",
            "Fonte": m.get("odds_source") or "—",
        }
        for m in markets
    ]
    st.dataframe(rows, width="stretch", hide_index=True)


def render_advice(pred: dict, odds: dict, market_move: dict | None = None, *, odds_from_asian: bool = False) -> None:
    advice = advise(pred, odds, market_move=market_move, odds_from_asian=odds_from_asian)
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
        r1 = advice.get("score_reason_1")
        r2 = advice.get("score_reason_2")
        if r1:
            st.caption(r1)
        if r2:
            st.caption(r2)
    with right:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Probabilità", f"{play['probability']:.1%}")
        c2.metric("Quota equa", f"{play['fair_odds']:.2f}" if play["fair_odds"] else "—")
        if play["odds"]:
            c3.metric("EV", f"{(play['ev'] or 0):+.1%}")
            kq = play.get("kelly_quarter")
            c4.metric("Kelly ¼", f"{kq:.1%}" if kq is not None else "—")
        else:
            c3.metric("Quota", "—")
            c4.metric("Kelly ¼", "—")
        alt = advice.get("play_alt")
        if alt and alt["code"] != play["code"]:
            st.caption(f"Alternativa: **{alt['code']}** {alt['name']} · {alt['score']}/10")
        xg = advice.get("expected_goals") or {}
        st.caption(f"xG attesi {xg.get('home', '—')} – {xg.get('away', '—')}")
        move = advice.get("market_move")
        align = advice.get("market_align") or {}
        if move:
            st.markdown("**Mercato asiatico — cosa è cambiato dall'apertura**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Intensità", move.get("movement_level") or "Stabile")
            m2.metric("Direzione 1X2", move.get("steam_1x2") or "stabile")
            m3.metric("Direzione O/U", move.get("steam_ou") or "stabile")
            st.caption(move.get("movement_summary") or move.get("note") or "Quasi nessun movimento")
            label = align.get("label") or "n/d"
            st.caption(f"Modello vs mercato: **{label}**")

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
            f"EV min={cal.get('min_ev_play', 0.04):.0%}, "
            f"Brier {cal.get('brier_favorite_raw', '—')}→{cal.get('brier_favorite_calibrated', '—')}"
        )
    st.caption("Quote: football-data.co.uk + AsianBetSoccer (Bet365)")

upcoming = _load_json(UPCOMING) or []
tab_cal, tab_mkt, tab_one = st.tabs(["Calendario", "Tutti i mercati", "Singola partita"])

with tab_cal:
    if not upcoming:
        st.info("Nessun calendario. Premi **Aggiorna dati + modello** nella colonna a sinistra.")
    else:
        with st.expander("Come leggere la tabella", expanded=False):
            st.markdown(
                """
                - **Kelly ¼**: frazione di bankroll suggerita da Kelly (÷4) — stake teorico, non un ordine di giocata.
                - **Voto**: mix probabilità + value + robustezza ML/MC + Kelly ¼. Gli outsider non salgono solo per EV.
                - **Perché questo voto**: due righe che spiegano probabilità, quote e mercato Asian.
                - **Movimento**: quanto si è mosso il mercato asiatico dall'apertura — *Stabile / Leggero / Medio / Forte*.
                - **Cosa è cambiato**: es. `AH -0.25->0` = handicap spostato verso trasferta; `Tot 2.75->2.5` = linea gol abbassata (soldi sull'under).
                - **Vs mercato**: il consiglio del modello è *allineato* o *contrario* al movimento delle quote.
                - **Value**: quanto la quota del book è sopra la quota equa del modello (+15% = 15% più generosa).
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
            min_ev = st.slider("EV minimo", min_value=-0.40, max_value=0.40, value=-0.40, step=0.02, format="%.2f")
        with q4:
            only_value = st.checkbox("Solo EV positivo", value=False)
        aligned_only = st.checkbox("Solo allineati al mercato asiatico", value=False)
        s1, s2, s3 = st.columns(3)
        with s1:
            sort_mode = st.selectbox(
                "Ordina per",
                ["Consiglio (voto)", "Movimento mercato (maggiore)", "Value (equa vs reale)"],
            )
        with s2:
            only_asian = st.checkbox("Solo partite con quote Asian", value=False)
        with s3:
            min_move = st.selectbox("Movimento minimo", ["Tutti", "Leggero+", "Medio+", "Forte"], index=0)

        view = df[df["country"].isin(sel_country) & df["league"].isin(sel_league)].copy()
        if "pick_group" in view.columns:
            view = view[view["pick_group"].fillna("1x2").isin(sel_groups)]
        view = view[view["score"] >= min_score]
        view["quota_pick"] = view.apply(_quota_consiglio, axis=1)
        view = view[view["quota_pick"].isna() | ((view["quota_pick"] >= odd_min) & (view["quota_pick"] <= odd_max))]
        if only_value:
            view = view[view["ev"].fillna(-1) > 0]
        else:
            view = view[view["ev"].fillna(min_ev) >= min_ev]
        if aligned_only and "market_align" in view.columns:
            view = view[view["market_align"] == "allineato"]
        if only_asian and "odds_source" in view.columns:
            view = view[view["odds_source"] == "asianbetsoccer"]
        move_rank = {"Stabile": 0, "Leggero": 1, "Medio": 2, "Forte": 3}
        min_rank = {"Tutti": 0, "Leggero+": 1, "Medio+": 2, "Forte": 3}[min_move]
        if min_rank > 0 and "movement_level" in view.columns:
            view = view[view["movement_level"].map(move_rank).fillna(0) >= min_rank]
        if "value_edge" not in view.columns:
            view["value_edge"] = view.apply(
                lambda r: _value_edge(r.get("quota_pick"), r.get("fair_odds")),
                axis=1,
            )
        view = _sort_calendario(view, sort_mode)

        st.write(f"{len(view)} partite dopo i filtri (su {len(df)})")
        st.dataframe(_prepare_calendario_show(view), width="stretch", hide_index=True)

        with st.expander("Radar AsianBetSoccer — partite con movimento quote"):
            r1, r2 = st.columns(2)
            radar_min = r1.selectbox("Movimento minimo radar", ["Leggero+", "Medio+", "Forte"], index=1, key="radar_spread")
            min_map = {"Leggero+": 1.0, "Medio+": 4.0, "Forte": 7.0}
            radar = _asian_radar_table(min_map[radar_min])
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
            )

with tab_mkt:
    if not upcoming:
        st.info("Nessun calendario.")
    else:
        flat_rows = []
        for match in upcoming:
            for m in match.get("markets") or []:
                quota = m.get("odds")
                fair = m.get("fair_odds")
                ve = _value_edge(quota, fair)
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
                        "Quota book": quota,
                        "Quota equa": fair,
                        "Value": _pct(ve),
                        "value_num": ve,
                        "EV atteso": _pct(m["ev"]),
                        "ev_num": m["ev"],
                        "Movimento": match.get("movement_level"),
                        "Cosa è cambiato": match.get("movement_summary"),
                        "Voto": m.get("score") or m.get("score_value") or m.get("score_prob"),
                        "Fonte": m.get("odds_source") or "—",
                        "match_source": match.get("odds_source") or "—",
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
        min_ev_m = e1.slider("EV minimo", -0.40, 0.40, 0.0, 0.02, key="mkt_ev")
        min_voto = e2.slider("Voto minimo", 1, 10, 5, key="mkt_voto")
        sort_mkt = e3.selectbox(
            "Ordina per",
            ["Value (equa vs reale)", "Movimento mercato", "EV atteso", "Voto"],
            key="mkt_sort",
        )
        f1, f2 = st.columns(2)
        src_opts = sorted(flat["Fonte"].dropna().unique().tolist())
        sel_src = f1.multiselect("Fonte quota mercato", src_opts, default=src_opts)
        only_asian_match = f2.checkbox("Solo partite con quote Asian", value=False, key="mkt_asian")
        filt = flat[flat["group"].isin(sel_g)]
        filt = filt[filt["Fonte"].isin(sel_src)]
        if only_asian_match:
            filt = filt[filt["match_source"] == "asianbetsoccer"]
        filt = filt[filt["prob_num"] >= min_p]
        filt = filt[filt["Voto"].fillna(0) >= min_voto]
        filt = filt[filt["ev_num"].fillna(-1) >= min_ev_m]
        has_q = filt["Quota book"].notna()
        filt = filt[~has_q | ((filt["Quota book"] >= qmin) & (filt["Quota book"] <= qmax))]
        move_rank = {"Forte": 3, "Medio": 2, "Leggero": 1, "Stabile": 0}
        if sort_mkt == "Movimento mercato":
            filt["_ord"] = filt["Movimento"].map(move_rank).fillna(-1)
            filt = filt.sort_values(["_ord", "Voto"], ascending=False, na_position="last").drop(columns="_ord")
        elif sort_mkt == "Value (equa vs reale)":
            filt = filt.sort_values(["value_num", "ev_num"], ascending=False, na_position="last")
        elif sort_mkt == "Voto":
            filt = filt.sort_values("Voto", ascending=False, na_position="last")
        else:
            filt = filt.sort_values(["ev_num", "Voto"], ascending=False, na_position="last")
        st.write(f"{len(filt)} mercati dopo i filtri")
        hide = {"group", "match_source", "prob_num", "value_num", "ev_num"}
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
