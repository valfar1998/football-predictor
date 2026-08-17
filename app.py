"""Interfaccia: consiglio 1/X/2 con voto 1-10 e value sulle quote."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from main import predict_pipeline
from modules.advisor import advise
from modules.predictor import list_known_teams

ROOT = Path(__file__).resolve().parent
LAST = ROOT / "data" / "processed" / "last_prediction.json"

st.set_page_config(page_title="Football Predictor", page_icon="⚽", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.4rem; max-width: 1100px; }
      .pick-code {
        font-size: 4.2rem; font-weight: 800; line-height: 1;
        letter-spacing: 0.04em; margin: 0;
      }
      .pick-name { font-size: 1.15rem; opacity: 0.85; margin-top: 0.25rem; }
      .score-wrap { display: flex; gap: 6px; margin: 0.9rem 0 0.35rem; }
      .score-cell {
        width: 22px; height: 10px; border-radius: 2px;
        background: rgba(250,250,250,0.12);
      }
      .score-cell.on-hi { background: #3dd68c; }
      .score-cell.on-mid { background: #e6c35c; }
      .score-cell.on-lo { background: #e06c75; }
      .muted { opacity: 0.7; font-size: 0.92rem; }
      div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _load_last() -> dict | None:
    if LAST.exists():
        return json.loads(LAST.read_text(encoding="utf-8"))
    return None


@st.cache_data(show_spinner="Calcolo modello + Monte Carlo...")
def _run(home: str, away: str, n_sims: int) -> dict:
    return predict_pipeline(home, away, n_sims=n_sims)


def _score_bar(score: int) -> str:
    tone = "on-hi" if score >= 7 else "on-mid" if score >= 4 else "on-lo"
    cells = []
    for i in range(1, 11):
        cls = f"score-cell {tone}" if i <= score else "score-cell"
        cells.append(f'<div class="{cls}"></div>')
    return f'<div class="score-wrap">{"".join(cells)}</div>'


def _kind_label(kind: str) -> str:
    return {
        "più_probabile": "Più probabile",
        "valore": "Miglior rapporto probabilità/quota",
        "probabile_e_valore": "Più probabile e miglior value",
    }.get(kind, kind)


def _pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.1%}"


def _num(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "—"
    return f"{x:.{digits}f}"


teams = list_known_teams()
last = _load_last()
default_home, default_away = "Inter", "Milan"
if last and " vs " in last.get("match", ""):
    default_home, default_away = last["match"].split(" vs ", 1)

st.title("Consiglio 1X2")
st.caption("Dalla predizione: esito da giocare, value sulle quote, voto da 1 a 10.")

with st.sidebar:
    st.header("Partita")
    home = st.selectbox("Casa", teams, index=teams.index(default_home) if default_home in teams else 0)
    away = st.selectbox("Trasferta", teams, index=teams.index(default_away) if default_away in teams else min(1, len(teams) - 1))
    n_sims = st.slider("Simulazioni Monte Carlo", 2000, 20000, 10000, 1000)
    run_new = st.button("Calcola predizione", type="primary", width="stretch")
    use_last = st.button("Usa ultima predizione", width="stretch")

    st.header("Quote bookmaker")
    st.caption("Decimali. Lascia vuoto se non le hai: resta il voto sulla probabilità.")
    odd_1 = st.number_input("Quota 1", min_value=1.01, max_value=50.0, value=None, placeholder="es. 1.85", step=0.05, format="%.2f")
    odd_x = st.number_input("Quota X", min_value=1.01, max_value=50.0, value=None, placeholder="es. 3.40", step=0.05, format="%.2f")
    odd_2 = st.number_input("Quota 2", min_value=1.01, max_value=50.0, value=None, placeholder="es. 4.50", step=0.05, format="%.2f")
    with st.expander("Mercati extra"):
        odd_o25 = st.number_input("Over 2.5", min_value=1.01, max_value=20.0, value=None, placeholder="es. 1.90", step=0.05, format="%.2f")
        odd_u25 = st.number_input("Under 2.5", min_value=1.01, max_value=20.0, value=None, placeholder="es. 1.90", step=0.05, format="%.2f")
        odd_btts = st.number_input("BTTS sì", min_value=1.01, max_value=20.0, value=None, placeholder="es. 1.80", step=0.05, format="%.2f")
        odd_nbtts = st.number_input("BTTS no", min_value=1.01, max_value=20.0, value=None, placeholder="es. 2.00", step=0.05, format="%.2f")

if "prediction" not in st.session_state:
    st.session_state.prediction = last

if run_new:
    if home == away:
        st.error("Scegli due squadre diverse.")
        st.stop()
    st.session_state.prediction = _run(home, away, n_sims)

if use_last:
    st.session_state.prediction = _load_last()

pred = st.session_state.prediction
if not pred:
    st.info("Calcola una predizione oppure lancia prima `python main.py --predict Inter Milan`.")
    st.stop()

odds = {
    "1": odd_1,
    "X": odd_x,
    "2": odd_2,
    "over_2.5": odd_o25,
    "under_2.5": odd_u25,
    "btts_yes": odd_btts,
    "btts_no": odd_nbtts,
}
advice = advise(pred, odds)
play = advice["play"]

col_pick, col_side = st.columns([1.15, 1])
with col_pick:
    st.markdown(f"**{advice['match']}**")
    st.markdown(
        f'<p class="pick-code">GIOCA {play["code"]}</p>'
        f'<p class="pick-name">{play["name"]} · {_kind_label(play["kind"])}</p>',
        unsafe_allow_html=True,
    )
    st.markdown(_score_bar(play["score"]), unsafe_allow_html=True)
    st.markdown(f"**{play['score']} / 10**")

with col_side:
    m1, m2, m3 = st.columns(3)
    m1.metric("Probabilità", _pct(play["probability"]))
    m2.metric("Quota equa", _num(play["fair_odds"]))
    if play["odds"]:
        ev = play["ev"] or 0
        m3.metric("EV", f"{ev:+.1%}")
    else:
        m3.metric("Quota book", "—")
    xg = advice.get("expected_goals") or {}
    st.caption(
        f"xG attesi {xg.get('home', '—')} – {xg.get('away', '—')} · "
        f"modello ML {pred['model_probabilities']['home_win']:.0%} / "
        f"{pred['model_probabilities']['draw']:.0%} / "
        f"{pred['model_probabilities']['away_win']:.0%}"
    )

c1, c2 = st.columns(2)
with c1:
    p = advice["most_probable"]
    st.subheader("Più probabile")
    st.write(f"**{p['code']}** · {p['name']}")
    st.write(f"Probabilità {p['probability']:.1%} · voto {p['score_prob']}/10")
    st.caption(f"Quota equa {_num(p['fair_odds'])} · modello ML {p['model_probability']:.1%}")
with c2:
    st.subheader("Miglior value")
    v = advice["best_value"]
    if not v:
        st.write("Inserisci le quote 1 / X / 2 nella colonna a sinistra.")
        st.caption("Il value è probabilità × quota. Sopra 1.00 il book paga meno di quanto stima il modello.")
    else:
        ev = v["ev"] or 0
        st.write(f"**{v['code']}** · {v['name']}")
        st.write(f"Rapporto {v['prob_odds_ratio']:.2f} · EV {ev:+.1%} · voto {v['score_value']}/10")
        st.caption(f"Quota {_num(v['odds'])} vs equa {_num(v['fair_odds'])}")

st.subheader("1X2")
rows = []
for m in advice["markets"]:
    rows.append(
        {
            "Esito": m["code"],
            "Mercato": m["name"],
            "Prob.": m["probability"],
            "Quota": m["odds"],
            "Quota equa": m["fair_odds"],
            "EV": m["ev"],
            "p × quota": m["prob_odds_ratio"],
            "Voto prob.": m["score_prob"],
            "Voto value": m["score_value"],
        }
    )
st.dataframe(rows, width="stretch", hide_index=True)

extras_with_odds = [e for e in advice["extras"] if e["odds"] is not None]
if extras_with_odds:
    st.subheader("Mercati extra")
    extra_rows = [
        {
            "Mercato": e["name"],
            "Prob.": e["probability"],
            "Quota": e["odds"],
            "Quota equa": e["fair_odds"],
            "EV": e["ev"],
            "Voto value": e["score_value"],
        }
        for e in extras_with_odds
    ]
    st.dataframe(extra_rows, width="stretch", hide_index=True)

scores = advice.get("most_likely_scores") or []
if scores:
    st.subheader("Risultati più probabili")
    st.bar_chart(
        {s["score"]: s["prob"] for s in scores},
        x_label="Risultato",
        y_label="Probabilità",
    )

st.caption(
    "Il voto 1–10 non è una certezza: il modello ha accuracy circa 55% sul test set. "
    "Senza quote il voto misura solo quanto l'esito è dominante. Con le quote misura anche l'edge."
)
