"""Interfaccia: calendario, mercati 1X2 / O/U / gol / DC, filtri quote."""

from __future__ import annotations

import json
import subprocess
import sys
import unicodedata
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Football Predictor", page_icon="⚽", layout="wide")

from main import predict_pipeline
from modules.advisor.advise import advise, format_advice
from modules.advisor.vote_copy import render_vote_copy
from modules.calibration.config import load_calibration
from modules.data_update.asian_odds import (
    MOVE_FILTER_OPTIONS,
    MOVE_FILTER_RANK,
    MOVE_RANK,
    find_asian_odds,
    load_asian_odds,
    spread_playability,
    summarize_moves,
)
from modules.predictor import list_known_teams, list_team_meta
from modules.data_update.cups import download_org_cups, org_token_configured, save_org_token
from modules.data_update.fbref_context import download_fbref_context
from modules.data_update.understat_context import download_understat_context
from modules.data_update.statsbomb_context import download_statsbomb_context
from modules.data_update.sofascore_context import download_sofascore_context
from modules.data_update.fotmob_context import download_fotmob_context
from modules.data_update.parse import load_fixtures

ROOT = Path(__file__).resolve().parent
LAST = ROOT / "data" / "processed" / "last_prediction.json"
UPCOMING = ROOT / "data" / "processed" / "upcoming_predictions.json"

GROUP_LABEL = {
    "1x2": "1X2",
    "dc": "Doppia chance / DNB",
    "ah": "Asian Handicap 0",
    "ou": "Over / Under",
    "btts": "Gol / No gol",
    "multigol": "Multigol",
    "parity": "Pari / Dispari",
    "exact": "Risultato esatto",
    "team": "Gol squadra",
    "cards": "Cartellini",
    "corners": "Corner",
    "scorer": "Marcatori (xG+XI)",
    "combo": "Combo (risultato + O/U / Gol)",
}

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


def _filter_by_date(df: pd.DataFrame, col: str = "date", *, key: str = "cal_dates") -> pd.DataFrame:
    if col not in df.columns or df.empty:
        return df
    dates = pd.to_datetime(df[col], errors="coerce").dt.date
    valid = dates.dropna()
    if valid.empty:
        return df
    dmin, dmax = valid.min(), valid.max()
    today = date.today()
    # Default: da oggi (o dalla prima data disponibile se tutto è futuro).
    start = min(max(today, dmin), dmax)
    # Chiave con la data odierna: ogni giorno riparte da oggi senza restare sul vecchio range.
    picked = st.date_input(
        "Intervallo date",
        value=(start, dmax),
        min_value=dmin,
        max_value=dmax,
        key=f"{key}_{today.isoformat()}",
    )
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


def _as_frac(val) -> float:
    """Converte EV/edge/prob in frazione float (0.05 = 5%). Accetta anche stringhe '+5%'."""
    if val is None:
        return float("nan")
    try:
        if isinstance(val, (int, float)):
            if pd.isna(val):
                return float("nan")
            return float(val)
    except (TypeError, ValueError):
        pass
    s = str(val).strip().replace(",", ".")
    if not s or s.lower() in {"none", "nan", "—", "-", "n/d"}:
        return float("nan")
    pct = s.endswith("%")
    s = s[:-1].strip() if pct else s
    s = s.replace("+", "").replace(" ", "")
    try:
        x = float(s)
    except ValueError:
        return float("nan")
    # "5%" oppure 5 scritto come punti percentuali → frazione
    if pct or abs(x) > 1.5:
        x = x / 100.0
    return x


def _frac_series(s: pd.Series | None, *, fill: float | None = None) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    out = s.map(_as_frac)
    if fill is not None:
        out = out.fillna(fill)
    return out


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
    wx = val.get("weather") or {}
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
            simv = val.get("sportly_sim") or {}
            if simv.get("ready") or simv.get("status"):
                st.metric(
                    "Sim",
                    simv.get("status") or "n/d",
                    delta=(f"{simv.get('delta_unified'):+.1f}" if simv.get("delta_unified") else None),
                    border=True,
                )
            dsig = val.get("data_signal") or {}
            if dsig.get("ready") or dsig.get("status"):
                st.metric(
                    "Dati",
                    dsig.get("status") or "n/d",
                    delta=(f"{dsig.get('delta_unified'):+.1f}" if dsig.get("delta_unified") else None),
                    border=True,
                )
            agr = val.get("agreement") or {}
            if agr.get("ready") or agr.get("status"):
                st.metric(
                    "Accordo",
                    agr.get("status") or "n/d",
                    delta=(None if agr.get("agree_share") is None else f"{agr['agree_share']:.0%}"),
                    border=True,
                )
            if wx:
                st.metric(
                    "Meteo",
                    wx.get("flag") or "n/d",
                    delta=(None if wx.get("precip_mm") is None else f"{wx['precip_mm']} mm"),
                    border=True,
                )
        bits = []
        if venue.get("venue"):
            bits.append(venue["venue"])
        if wx.get("city") or wx.get("temp_c") is not None:
            wx_bits = [str(wx.get("city") or "").strip()]
            if wx.get("temp_c") is not None:
                wx_bits.append(f"{wx['temp_c']}°C")
            if wx.get("wind_kmh") is not None:
                wx_bits.append(f"vento {wx['wind_kmh']} km/h")
            bits.append(" ".join(b for b in wx_bits if b))
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
        dsig = val.get("data_signal") or {}
        if dsig.get("notes"):
            st.caption("Dati: " + " · ".join(str(n) for n in dsig["notes"][:3] if n))


def _render_data_signal(sig: dict | None) -> None:
    if not sig or not sig.get("ready"):
        return
    with st.container(border=True):
        st.markdown("**Analisi dati** (xG · forma · casa/trasferta · classifiche)")
        with st.container(horizontal=True):
            st.metric("Lean", sig.get("lean") or "—", border=True)
            st.metric(
                "Edge",
                "—" if sig.get("edge") is None else f"{float(sig['edge']):+.2f}",
                border=True,
            )
            st.metric(
                "Confidenza",
                "—" if sig.get("confidence") is None else f"{float(sig['confidence']):.0%}",
                border=True,
            )
            st.metric("Fattori", str(sig.get("n_factors") or 0), border=True)
        factors = sig.get("factors") or []
        if factors:
            bits = [
                f"{f.get('name')}: {float(f.get('edge') or 0):+.2f} (w={f.get('weight')})"
                for f in factors[:6]
            ]
            st.caption(" · ".join(bits))
        if sig.get("note"):
            st.caption(str(sig["note"]))


def _render_sportly_sim(sim: dict | None, *, home: str = "Casa", away: str = "Trasferta") -> None:
    if not sim or not sim.get("ready"):
        return
    xg = sim.get("xg") or {}
    mom = sim.get("momentum") or {}
    press = sim.get("pressure") or {}
    shots = sim.get("shots") or {}
    trend = sim.get("live_trend") or []
    tv = sim.get("tactical_validation") or {}
    with st.container(border=True):
        st.markdown("**Sportly-sim (interno, non live)**")
        st.caption(sim.get("note") or "")
        with st.container(horizontal=True):
            st.metric("Lean sim", sim.get("lean") or "—", border=True)
            st.metric(
                "xG sim",
                f"{xg.get('home', '—')} – {xg.get('away', '—')}",
                border=True,
            )
            st.metric(
                "Tiri",
                f"{shots.get('home_n', '—')}–{shots.get('away_n', '—')}",
                border=True,
            )
            st.metric(
                "Validazione",
                tv.get("status") or "n/d",
                delta=(f"{tv.get('delta_unified'):+.1f}" if tv.get("delta_unified") else None),
                border=True,
            )
        if tv.get("notes"):
            st.caption(tv["notes"][0])
        mins = xg.get("minutes") or mom.get("minutes") or []
        if mins and xg.get("cum_home") and xg.get("cum_away"):
            st.caption("xG cumulato simulato")
            st.line_chart(
                {
                    "minuto": mins,
                    home: xg["cum_home"],
                    away: xg["cum_away"],
                },
                x="minuto",
                y=[home, away],
            )
        if mins and mom.get("values"):
            st.caption("Momentum (positivo = casa)")
            st.line_chart({"minuto": mins, "momentum": mom["values"]}, x="minuto", y="momentum")
        if mins and press.get("home") and press.get("away"):
            st.caption("Pressione sintetica")
            st.line_chart(
                {
                    "minuto": mins,
                    f"press {home}": press["home"],
                    f"press {away}": press["away"],
                },
                x="minuto",
                y=[f"press {home}", f"press {away}"],
            )
        if trend:
            st.caption(
                "Trend live simulato: "
                + " · ".join(f"{p.get('from')}-{p.get('to')}' {p.get('lean')}" for p in trend)
            )
        smap = shots.get("map") or []
        if smap:
            with st.expander("Shot map sintetica", expanded=False):
                st.dataframe(
                    [
                        {
                            "Squadra": home if s.get("team") == "home" else away,
                            "X": s.get("x"),
                            "Y": s.get("y"),
                            "xG": s.get("xg"),
                            "On target": "sì" if s.get("on_target") else "no",
                        }
                        for s in smap
                    ],
                    width="stretch",
                    hide_index=True,
                )


def _pp(val: float | None) -> str | None:
    if val is None:
        return None
    return f"{float(val):+.1f} pp"


def _sort_calendario(view: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = view.copy()
    if mode == "Indice gioca":
        from modules.advisor.play_rank import ensure_play_rank_df

        out = ensure_play_rank_df(out)
        ev = out["ev_cons"] if "ev_cons" in out.columns else out.get("ev")
        out["_ev"] = _frac_series(ev if isinstance(ev, pd.Series) else pd.Series(ev, index=out.index), fill=-99)
        return out.sort_values(
            ["play_rank", "_ev", "score_unified"],
            ascending=False,
            na_position="last",
        ).drop(columns=["_ev"], errors="ignore")
    if mode in {"Consigliato", "Consigliato (lettura)"}:
        # 1) Gioca prima  2) EV  3) Score/voto  4) Confidence  5) Risk basso  6) Priorità
        act_rank = {"gioca": 3, "no_bet": 2, "n/d": 1, "invalido": 0}
        if "action" in out.columns:
            out["_act"] = out["action"].map(lambda x: act_rank.get(str(x), 1)).fillna(1)
        else:
            out["_act"] = 1
        ev = out["ev_cons"] if "ev_cons" in out.columns else out.get("ev")
        out["_ev"] = _frac_series(ev if isinstance(ev, pd.Series) else pd.Series(ev, index=out.index), fill=-99)
        out["_s100"] = pd.to_numeric(out["score_100"], errors="coerce").fillna(-1) if "score_100" in out.columns else -1
        out["_uni"] = pd.to_numeric(out["score_unified"], errors="coerce").fillna(-1) if "score_unified" in out.columns else -1
        out["_conf"] = pd.to_numeric(out["confidence_100"], errors="coerce").fillna(0) if "confidence_100" in out.columns else 0
        out["_risk_inv"] = 100 - pd.to_numeric(out["risk_100"], errors="coerce").fillna(50) if "risk_100" in out.columns else 50
        out["_prio"] = pd.to_numeric(out["priority_100"], errors="coerce").fillna(0) if "priority_100" in out.columns else 0
        return out.sort_values(
            ["_act", "_ev", "_s100", "_uni", "_conf", "_risk_inv", "_prio"],
            ascending=False,
            na_position="last",
        ).drop(columns=["_act", "_ev", "_s100", "_uni", "_conf", "_risk_inv", "_prio"])
    if mode == "Data (più vicine)":
        if "date" in out.columns:
            return out.sort_values(["date", "time"], ascending=True, na_position="last")
        return out
    if mode == "Priorità":
        if "priority_100" in out.columns:
            return out.sort_values(
                ["priority_100", "score_100", "score_unified", "score"],
                ascending=False,
                na_position="last",
            )
        if "score_100" in out.columns:
            return out.sort_values(["score_100", "score_unified"], ascending=False, na_position="last")
    if mode == "Voto unificato":
        if "score_unified" in out.columns:
            return out.sort_values(["score_unified", "score", "probability"], ascending=False, na_position="last")
        return out.sort_values(["score", "probability"], ascending=False, na_position="last")
    if mode == "EV cons. %":
        ev = out["ev_cons"] if "ev_cons" in out.columns else out.get("ev")
        out["_sort"] = _frac_series(ev if isinstance(ev, pd.Series) else pd.Series(ev, index=out.index), fill=-99)
        return out.sort_values("_sort", ascending=False, na_position="last").drop(columns="_sort")
    if mode == "Movimento mercato (maggiore)":
        out["_sort"] = out["movement_level"].map(MOVE_RANK).fillna(0)
        if "line_move" in out.columns:
            out["_sort"] = out["_sort"] * 100 + out["line_move"].fillna(0)
        elif "spread_score" in out.columns:
            out["_sort"] = out["_sort"] * 100 + out["spread_score"].fillna(0)
        return out.sort_values("_sort", ascending=False).drop(columns="_sort")
    if mode == "Value (edge vs mercato)":
        edge = _frac_series(out["edge_pp"]) if "edge_pp" in out.columns else pd.Series(float("nan"), index=out.index)
        cons = _frac_series(out["ev_cons"]) if "ev_cons" in out.columns else pd.Series(float("nan"), index=out.index)
        fallback = _frac_series(out["ev"]) if "ev" in out.columns else pd.Series(-99.0, index=out.index)
        out["_sort"] = edge.fillna(cons).fillna(fallback).fillna(-99)
        return out.sort_values("_sort", ascending=False, na_position="last").drop(columns="_sort")
    if mode == "Consiglio (voto)":
        return out.sort_values(["score", "probability"], ascending=False, na_position="last")
    # fallback = stesso del Consigliato
    return _sort_calendario(out, "Consigliato")


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
    # Dopo partita: Indice gioca → Azione → EV → voto → pick → value/Kelly → resto
    wanted = [
        "date", "time", "country", "league", "home", "away",
        "play_rank", "action", "ev_cons", "score_unified",
        "pick", "pick_name", "kelly_quarter", "edge_pp",
        "quota_pick", "fair_odds", "probability", "ev_sharp", "clv",
        "score_100", "score_band", "confidence_100", "risk_100", "priority_100",
        "bet_rec_label", "score", "meta_label", "meta_note",
        "quadro_consenso", "quadro_n", "tipster_consensus", "tipster_agree",
        "score_reason_1", "score_reason_2", "skip_reason",
        "odds_real", "value_note",
        "venue_flag", "weather_flag", "validation_summary", "validation_delta",
        "movement_level", "line_move", "movement_summary", "movement_comment", "market_align",
        "drop_1", "drop_x", "drop_2",
        "odd_1", "odd_x", "odd_2", "odd_over_25", "odd_under_25", "odds_source",
    ]
    show = view[[c for c in wanted if c in view.columns]].copy()
    # Tieni EV/edge/prob come numeri (frazione): così click-sort e "Ordina per EV" non fanno 5% > 45%
    for col in ("ev_cons", "edge_pp", "ev_sharp", "kelly_quarter", "clv", "probability"):
        if col in show.columns:
            show[col] = _frac_series(show[col])
    if "odds_real" in show.columns:
        show["odds_real"] = show["odds_real"].map(lambda x: "Sì" if bool(x) else "No")
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
            show[drop_col] = pd.to_numeric(show[drop_col], errors="coerce")
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
            "play_rank": "Indice gioca",
            "priority_100": "Priorità",
            "score_100": "Score 0–100",
            "score_band": "Banda",
            "confidence_100": "Confidence",
            "risk_100": "Risk",
            "bet_rec_label": "Mercato consigliato",
            "score_unified": "Voto unificato",
            "score": "Voto mercato",
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
            "weather_flag": "Meteo",
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


def _run_cli(*flags: str, with_progress: bool = True) -> subprocess.CompletedProcess:
    if with_progress:
        try:
            from modules.progress_report import StreamlitProgress, run_cli_with_progress

            prog = StreamlitProgress(" ".join(flags))
            return run_cli_with_progress(*flags, progress=prog, python_exe=sys.executable, main_path=str(ROOT / "main.py"))
        except Exception:
            pass
    return subprocess.run(
        [sys.executable, str(ROOT / "main.py"), *flags],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _batch(title: str):
    """Context manager-like: crea barra progresso Streamlit."""
    from modules.progress_report import StreamlitProgress

    return StreamlitProgress(title)


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
    render_vote_copy(advice, key="singola_predizione")
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
        ms = play.get("match_scores") or advice.get("match_scores") or {}
        if play.get("score_100") is not None or advice.get("score_100") is not None:
            s100 = play.get("score_100") if play.get("score_100") is not None else advice.get("score_100")
            band = play.get("score_band") or advice.get("score_band") or "—"
            with st.container(horizontal=True):
                st.metric("Score", f"{s100:.0f}", band, border=True)
                c100 = play.get("confidence_100") if play.get("confidence_100") is not None else advice.get("confidence_100")
                st.metric("Confidence", "—" if c100 is None else f"{c100:.0f}", border=True)
                r100 = play.get("risk_100") if play.get("risk_100") is not None else advice.get("risk_100")
                st.metric("Risk", "—" if r100 is None else f"{r100:.0f}", border=True)
                p100 = play.get("priority_100") if play.get("priority_100") is not None else advice.get("priority_100")
                prio = (ms.get("priority") or {})
                prio_lbl = prio.get("rank_hint") or ""
                st.metric(
                    "Priorità",
                    "—" if p100 is None else f"{p100:.0f}",
                    prio_lbl or None,
                    border=True,
                )
            br = play.get("bet_rec") or advice.get("bet_rec") or ms.get("bet_rec") or {}
            prim = (br or {}).get("primary") or {}
            if prim.get("code"):
                st.caption(
                    f"Mercato consigliato: **{prim.get('label')} {prim.get('code')}** "
                    f"({prim.get('name') or ''})"
                    + (f" · {br.get('note')}" if br.get("note") else "")
                )
            ov = ms.get("overrides") or {}
            if ov.get("notes"):
                st.caption("Override: " + " · ".join(ov["notes"][:3]))
            prio = ms.get("priority") or {}
            if prio.get("notes") or prio.get("formula"):
                bits = list(prio.get("notes") or [])[:4]
                st.caption(
                    "Priorità = urgenza ranking (non è il voto): "
                    + (prio.get("formula") or "EV × S_quota × C_modelli × T_match × L_mercato")
                    + ((" · " + " · ".join(bits)) if bits else "")
                )
            st.caption("Bande Score: 0–30 no bet · 30–60 lean · 60–75 playable · 75–90 strong · 90–100 premium")
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
        _render_sportly_sim(
            advice.get("sportly_sim") or (pred.get("sportly_sim") if isinstance(pred, dict) else None),
            home=str(advice.get("home") or "Casa"),
            away=str(advice.get("away") or "Trasferta"),
        )
        _render_data_signal(
            advice.get("data_signal")
            or (pred.get("data_signal") if isinstance(pred, dict) else None)
            or ((val or {}).get("data_signal") if isinstance(val, dict) else None)
        )
        alt = advice.get("play_alt")
        if alt and alt["code"] != play["code"]:
            st.caption(f"Alternativa: **{alt['code']}** {alt['name']} · {alt['score']}/10")
        xg = advice.get("expected_goals") or {}
        st.caption(
            f"xG attesi {xg.get('home', '—')} – {xg.get('away', '—')} "
            "· ensemble XGB + Dixon-Coles (Understat/meteo in λ se presenti)"
        )
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
            play_ctx = play if isinstance(play, dict) else {}
            playab = spread_playability(
                {
                    "pick": play_ctx.get("code") or play_ctx.get("pick"),
                    "action": play_ctx.get("action"),
                    "market_align": align.get("label") if isinstance(align, dict) else align,
                },
                move,
            )
            st.metric(
                "Giocabilità spread",
                f"{playab['score']}/10",
                playab["verdict"],
            )
            st.caption(playab.get("reason") or playab.get("verdict_long") or "")
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
                        "Peso": s.get("peso"),
                        "Idea": s.get("idea"),
                        "Lean": s.get("pick"),
                        "1": f"{s['p_1']:.0%}" if s.get("p_1") is not None else "—",
                        "X": f"{s['p_x']:.0%}" if s.get("p_x") is not None else "—",
                        "2": f"{s['p_2']:.0%}" if s.get("p_2") is not None else "—",
                        "Motivazione": s.get("peso_note") or "—",
                        "Nota": s.get("nota") or "—",
                    }
                )
            st.dataframe(rows, width="stretch", hide_index=True)
            st.caption(
                "Pesi sul consenso: MC 0.35 · ML 0.25 · Book 0.20 · xG/dati ~0.10 · "
                "tattica/meteo/tipster bassi. EV/Kelly restano su modello+quota."
            )
            wt = (advice.get("match_scores") or play.get("match_scores") or {}).get("weights_table") or []
            if wt:
                with st.expander("Matrice pesi fonti"):
                    st.dataframe(wt, width="stretch", hide_index=True)
            if quadro.get("fallback"):
                st.info("Fallback attivo: sintesi Elo+forma+book (dati incompleti / lega minore).")
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
    "Tre livelli: **modello** (soldi: EV/Kelly/Gioca; ensemble XGB+Poisson), **voto unificato** (ordine in tabella), "
    "**fonti extra** (quadro). Understat e meteo entrano nelle λ, non nell'EV. A sinistra basta **Aggiorna dati + modello**."
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
- **Ensemble 1X2**: XGBoost + Poisson/Dixon-Coles. Understat xG entra nelle λ (non nel train storico, per evitare leakage). Meteo Open-Meteo su città stadio.
- **Sportly-sim interno**: xG cumulato, momentum, pressione, shot map e trend a blocchi — sintetici da λ/stile, senza API Sportly/FotMob. Solo quadro/validazione (±0.5 sul voto unificato).
- **Quote implicite** come feature di train (open/close football-data). Servono `--train` / Aggiorna dati + modello.
- **Calibrazione T per campionato** (dopo `--calibrate`, se la lega ha ≥120 OOF).
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
| **Storico locale (SQLite)** | Le partite *tue* già viste (anche N/D). Dopo 30 esiti e 6 match/squadra entra al 12% del voto (fino al 18% con abbastanza chiusure globali/di lega) |

---

**Bottoni a sinistra — come usarli al meglio**

*Ogni giorno (o dopo una pausa)*
- **Scarica modello da GitHub** — prende l’ultimo train da Actions (no riallenamento locale). Poi **Solo quote**. Serve `gh auth login`.
- **Aggiorna dati + modello** — train completo in locale (~1h+). Solo se non usi GitHub o vuoi rifare tutto sul PC.
- **Solo quote e calendario** — stesso giorno, quote mosse o nuove partite. **Senza** riallenare. Più veloce; usa questo tra un train e l’altro.

*Quando ti servono le quote “giuste” per il voto*
- **Scarica quote AsianBetSoccer** — steam Bet365. Aggiorna value/voto sul calendario **senza** rifare ML/MC. Per nuove partite: Solo quote.
- **Scarica quote Pinnacle / Betfair** — sharp/exchange; poi solo ricalcolo EV/Kelly (leggero).
- **Scarica pronostici tipster** — quadro + value leggero.

*Coppe e calendario extra*
- **Scarica coppe** — Champions/Europa/… con token football-data.org. Senza token quelle coppe non compaiono. Le squadre fuori storico restano N/D.

*Contesto Big 5 (1–2×/settimana, solo quadro/voto)*
- **FBref** — stile/stats stagione.
- **FD cards/corners** — tassi ammonizioni/calci d’angolo da football-data (veloce; base per extras MC).
- **FBref match logs (lento)** — log partita per partita; più preciso su cards/corners, da fare di rado.
- **Understat xG** — xG squadra + player (marcatore/lineup).
- **StatsBomb / Sofascore / FotMob** — open data / classifica / match+lineup. FotMob anche XI on-demand.
- **Geocode stadi** — una tantum / quando mancano città per il meteo.
- **WhoScored assenze (lento)** — Selenium, ~18 preview. Solo quando ti servono gli XI confermati.

*Raro / manutenzione*
- **Calibra probabilità** — dopo tanti settle o un train grosso. Taratura T e soglie; non è refresh quotidiano.
- Tab **Valutazione**: *Apprendi da partite chiuse*, *Ottimizza pesi Analisi dati*, *Aggiorna report paper* — solo con abbastanza storico settled (vedi TECH_ROADMAP).

**In tabella Calendario:** subito dopo le squadre: **Indice gioca · Azione · EV · Voto unificato · Gioca**. Scorri a destra per il resto.
        """
    )

with st.sidebar:
    st.header("Uso quotidiano")
    st.caption(
        "Giorno per giorno: Solo quote. "
        "Modello: scaricalo da GitHub (veloce) oppure Aggiorna dati + modello (train locale ~1h+)."
    )
    if st.button("Scarica modello da GitHub", width="stretch"):
        with st.spinner("Scarico ultimo artefatto da Actions (serve gh auth login)…"):
            proc = _run_cli("--pull-model", with_progress=False)
        if proc.returncode != 0:
            st.error(proc.stderr[-1500:] or proc.stdout[-1500:] or "Download modello fallito")
        else:
            st.success(
                "Modello installato. Premi «Solo quote e calendario» per usarlo "
                "(ricalcola EV/calendario; se il modello è più nuovo fa Monte Carlo pieno)."
            )
            out = (proc.stdout or "")[-1200:]
            if out.strip():
                st.code(out, language=None)
            st.rerun()
    if st.button("Aggiorna dati + modello", type="primary", width="stretch"):
        proc = _run_cli("--update")
        if proc.returncode != 0:
            st.error(proc.stderr[-1500:] or proc.stdout[-1500:] or "Errore aggiornamento")
        else:
            st.success("Dati aggiornati")
            st.rerun()
    if st.button("Solo quote e calendario", width="stretch"):
        prog = _batch("Solo quote e calendario")
        try:
            from main import refresh_odds_pipeline

            info = refresh_odds_pipeline(on_progress=prog)
            n = int(info.get("n_upcoming") or 0)
            prog.done(f"OK · {n} partite")
            st.success(f"Quote e calendario aggiornati · {n} partite")
            st.rerun()
        except Exception as exc:
            st.error(f"Errore quote: {exc}")
    try:
        from modules.data_update.history import history_summary

        hs = history_summary()
        w = int(round(float(hs.get("weight") or 0.12) * 100))
        if hs.get("ready"):
            st.caption(
                f"Storico locale: {hs['n_history']} partite, {hs['n_settled']} chiuse "
                f"(riche {hs.get('n_rich', '—')}/{hs.get('n_rich_target', 80)}). "
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
        metrics_path = ROOT / "data" / "models" / "metrics.json"
        if metrics_path.is_file():
            meta = json.loads(metrics_path.read_text(encoding="utf-8"))
            n_cl = meta.get("n_clusters")
            if n_cl is None and isinstance(meta.get("cluster_metrics"), dict):
                n_cl = len(meta["cluster_metrics"])
            has_global = (ROOT / "data" / "models" / "best_model.joblib").is_file()
            if n_cl is not None:
                st.caption(
                    f"Modelli cluster attivi: {int(n_cl)}. "
                    f"Fallback globale: {'sì' if has_global else 'no'}."
                )
            mkt = ROOT / "data" / "models" / "market_models.joblib"
            mm = meta.get("market_models") or {}
            if mkt.is_file():
                ou_ll = mm.get("ou25_oof_ll")
                ah_ll = mm.get("ah0_oof_ll")
                extra = ""
                if ou_ll is not None and ah_ll is not None:
                    extra = f" (O/U ll={float(ou_ll):.3f}, AH ll={float(ah_ll):.3f})"
                st.caption(f"Modelli mercato O/U 2.5 + AH 0: attivi{extra}.")
    except Exception:
        pass
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
                prog = _batch("Coppe")
                from modules.data_update.upcoming import build_upcoming

                prog(0.1, "GET /v4/matches…")
                info = download_org_cups()
                if not info.get("token"):
                    st.error("Token assente o non letto.")
                elif info.get("error"):
                    st.error(str(info["error"]))
                else:
                    prog(0.35, "Calendario: riuso predizioni, MC solo sulle nuove…")
                    upcoming_n = len(build_upcoming(reuse_predictions=True))
                    n = info.get("n_cup_fixtures") or 0
                    comps = ", ".join(info.get("competitions") or []) or "nessuna coppa in finestra"
                    prog.done("OK")
                    st.success(
                        f"Coppe: {n} match · {comps} · calendario {upcoming_n} "
                        f"(riuso predizioni dove possibile)"
                    )
                    st.rerun()
        st.caption(
            "Dopo il download non rifà Monte Carlo su tutto: riusa le predizioni già in calendario "
            "e calcola solo le partite nuove."
        )

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
                prog = _batch("Pinnacle")
                from modules.data_update.odds_api import fetch_pinnacle_odds
                from modules.data_update.upcoming import refresh_upcoming_odds

                prog(0.1, "Fetch The Odds API…")
                pinn = fetch_pinnacle_odds(force=True)
                if not pinn.get("ok"):
                    st.error(pinn.get("error") or "Errore fetch Pinnacle")
                else:
                    prog(0.4, "Ricalcolo EV/Kelly (senza ML/MC)…")
                    refresh = refresh_upcoming_odds(on_progress=prog)
                    rem = pinn.get("remaining")
                    prog.done("OK")
                    st.success(
                        f"Pinnacle: {pinn.get('n_events', 0)} partite · "
                        f"chiamate rimanenti {rem if rem is not None else 'n/d'} · "
                        f"value {refresh.get('n_refreshed', 0)}/{refresh.get('n_upcoming', 0)} (leggero)"
                    )
                    st.rerun()

    with st.expander("Quote Betfair Exchange"):
        st.caption(
            "Delayed App Key già salvata. Serve anche username e password Betfair.it "
            "(login API, non vanno in git). Dati ritardati, gratis. "
            "Dopo il download aggiorna solo EV/Kelly sulle predizioni già calcolate."
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
                prog = _batch("Betfair")
                from modules.data_update.upcoming import refresh_upcoming_odds

                prog(0.1, "Login e download Exchange…")
                bf = fetch_betfair_odds(force=True)
                if not bf.get("ok"):
                    st.error(bf.get("error") or "Errore fetch Betfair")
                else:
                    prog(0.35, "Ricalcolo EV/Kelly (senza ML/MC)…")
                    refresh = refresh_upcoming_odds(on_progress=prog)
                    prog.done("OK")
                    st.success(
                        f"Betfair: {bf.get('n_events', 0)} partite · "
                        f"value {refresh.get('n_refreshed', 0)}/{refresh.get('n_upcoming', 0)} (leggero)"
                    )
                    st.rerun()

    with st.expander("Quote Asian e tipster"):
        st.caption(
            "Asian = movimento Bet365. Tipster = consenso siti. "
            "Entrambi aggiornano il value sul calendario esistente senza rifare Monte Carlo "
            "(per nuove partite usa Solo quote / Aggiorna dati)."
        )
        if st.button("Scarica quote AsianBetSoccer", width="stretch"):
            with st.spinner("Scarico Asian + ricalcolo value (leggero)…"):
                proc = _run_cli("--asian-odds")
            if proc.returncode != 0:
                st.error(proc.stderr[-1500:] or proc.stdout[-1500:] or "Errore quote Asian")
            else:
                st.success("Quote AsianBetSoccer aggiornate (value leggero)")
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
            with st.spinner("Scarico tipster + ricalcolo value (leggero)…"):
                proc = _run_cli("--tipsters")
            if proc.returncode != 0:
                st.error(proc.stderr[-1500:] or proc.stdout[-1500:] or "Errore tipster")
            else:
                st.success("Tipster aggiornati (value leggero)")
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
                f"Calibrazione: T={cal.get('temperature', 1):.2f}"
                + (f" · {len(cal.get('temperature_by_league') or {})} T di lega" if cal.get("temperature_by_league") else "")
                + f", EV min={cal.get('min_ev_play', 0.025):.0%}"
            )

    with st.expander("Contesto extra (non entra in EV)"):
        st.caption(
            "Quadro e voto. Barra % + log live: se avanza, sta lavorando; se resta fermo a lungo su un passo, "
            "probabilmente è bloccato dalla rete/scraping."
        )
        if st.button("FBref", width="stretch"):
            prog = _batch("FBref")
            from modules.data_update.upcoming import build_upcoming

            info = download_fbref_context(on_progress=prog)
            if info.get("error") and not info.get("n_teams"):
                st.error(f"FBref: {info['error']}")
            else:
                prog.done(f"{info.get('n_teams', 0)} squadre")
                upcoming_n = len(build_upcoming())
                st.success(f"FBref: {info.get('n_teams', 0)} squadre · calendario {upcoming_n}")
                st.rerun()
        if st.button("FD cards/corners", width="stretch"):
            prog = _batch("FD cards/corners")
            from modules.data_update.side_rates import build_fd_side_rates
            from modules.data_update.upcoming import build_upcoming

            info = build_fd_side_rates(on_progress=prog)
            if not info.get("ok"):
                st.error(f"FD rates: {info.get('error')}")
            else:
                prog.done(f"{info.get('n_teams', 0)} squadre")
                upcoming_n = len(build_upcoming())
                st.success(f"FD rates: {info.get('n_teams', 0)} squadre · calendario {upcoming_n}")
                st.rerun()
        if st.button("FBref match logs (lento)", width="stretch"):
            prog = _batch("FBref match logs")
            from modules.data_update.fbref_context import download_fbref_match_logs
            from modules.data_update.upcoming import build_upcoming

            info = download_fbref_match_logs(on_progress=prog)
            if not info.get("ok"):
                st.error(f"Match logs: {info.get('error')}")
            else:
                prog.done(f"{info.get('n_teams', 0)} squadre")
                upcoming_n = len(build_upcoming())
                st.success(
                    f"Match logs: {info.get('n_teams', 0)} squadre "
                    f"(cards={info.get('has_cards')} corners={info.get('has_corners')}) · cal {upcoming_n}"
                )
                st.rerun()
        if st.button("Understat xG", width="stretch"):
            prog = _batch("Understat")
            from modules.data_update.upcoming import build_upcoming

            info = download_understat_context(on_progress=prog)
            if info.get("error") and not info.get("n_teams"):
                st.error(f"Understat: {info['error']}")
            else:
                pl = (info.get("players") or {}).get("n_players", 0)
                prog.done(f"{info.get('n_teams', 0)} squadre")
                upcoming_n = len(build_upcoming())
                st.success(f"Understat: {info.get('n_teams', 0)} squadre, {pl} player · calendario {upcoming_n}")
                st.rerun()
        if st.button("StatsBomb open data", width="stretch"):
            prog = _batch("StatsBomb")
            from modules.data_update.upcoming import build_upcoming

            info = download_statsbomb_context(on_progress=prog)
            if info.get("error") and not info.get("n_teams"):
                st.error(f"StatsBomb: {info['error']}")
            else:
                prog.done(f"{info.get('n_teams', 0)} squadre")
                upcoming_n = len(build_upcoming())
                st.success(
                    f"StatsBomb: {info.get('n_teams', 0)} squadre · calendario {upcoming_n}"
                )
                st.rerun()
        if st.button("Sofascore classifica", width="stretch"):
            prog = _batch("Sofascore")
            from modules.data_update.upcoming import build_upcoming

            prog(0.2, "Download classifiche…")
            info = download_sofascore_context()
            prog(0.85, "Calendario…")
            upcoming_n = len(build_upcoming())
            if info.get("error") and not info.get("n_teams"):
                st.error(f"Sofascore: {info['error']}")
            else:
                prog.done(f"{info.get('n_teams', 0)} squadre")
                st.success(f"Sofascore: {info.get('n_teams', 0)} squadre · calendario {upcoming_n}")
                st.rerun()
        if st.button("FotMob (classifica + match)", width="stretch"):
            prog = _batch("FotMob")
            from modules.data_update.upcoming import build_upcoming

            info = download_fotmob_context(days=7, on_progress=prog)
            errs = info.get("errors") or ([] if not info.get("error") else [info["error"]])
            if errs and not info.get("n_teams") and not info.get("n_matches"):
                st.error(f"FotMob: {errs[0]}")
            else:
                prog(0.92, "Calendario…")
                upcoming_n = len(build_upcoming())
                xg = info.get("xg") or {}
                prog.done("OK")
                st.success(
                    f"FotMob: {info.get('n_teams', 0)} squadre · "
                    f"{info.get('n_matches', 0)} partite · "
                    f"xG rolling {info.get('n_xg_teams') or xg.get('n_teams') or 0} · "
                    f"calendario {upcoming_n}"
                )
                st.rerun()
        if st.button("Geocode stadi (batch)", width="stretch"):
            prog = _batch("Geocode")
            from modules.data_update.weather import geocode_batch_venues

            info = geocode_batch_venues(max_n=80, on_progress=prog)
            st.write(info)
        if st.button("WhoScored assenze (lento)", width="stretch"):
            prog = _batch("WhoScored")
            from modules.data_update.upcoming import build_upcoming
            from modules.data_update.whoscored_context import download_whoscored_context

            prog(0.1, "Selenium preview (lento)…")
            info = download_whoscored_context()
            prog(0.85, "Calendario…")
            upcoming_n = len(build_upcoming())
            if info.get("error") and not info.get("n_missing"):
                st.error(f"WhoScored: {info['error']}")
            else:
                prog.done(f"{info.get('n_missing', 0)} assenze")
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
                - **Indice gioca** (prima colonna decisione): 0–100, ordina il calendario.
                - **Azione · EV cons. · Voto unificato · Gioca**: cosa guardare per scommettere.
                - **Voto mercato**: voto interno solo sul mercato consigliato (spesso vuoto su no-bet/N/D).
                - **Kelly ¼**: frazione di bankroll; zero se No bet.
                - **CLV**: quota accorciata dopo l'apertura = mercato a favore del pick.
                - **Edge pp / EV cons.**: value vs book (solo partite coperte dal modello).
                - **Movimento / Δ 1 X 2**: steam Asian. Non è EV.
                - **Quadro / Tipster**: fonti esterne vs pick. Non è EV.
                Scorri la tabella in orizzontale: ci sono molte colonne.
                """
            )
        df = pd.DataFrame(upcoming)
        from modules.advisor.play_rank import ensure_play_rank_df

        df = ensure_play_rank_df(df)
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
            min_score = st.slider("Voto unificato minimo", 1, 10, 1)

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
                [
                    "Indice gioca",
                    "Consigliato",
                    "Priorità",
                    "EV cons. %",
                    "Voto unificato",
                    "Value (edge vs mercato)",
                    "Data (più vicine)",
                    "Movimento mercato (maggiore)",
                    "Consiglio (voto)",
                ],
                index=0,
                help="Indice gioca: Azione + EV + voto unificato + Kelly in un unico punteggio 0–100 (default).",
            )
        with s2:
            only_asian = st.checkbox("Solo partite con quote Asian", value=False)
        with s3:
            min_move = st.selectbox("Movimento minimo", MOVE_FILTER_OPTIONS, index=0)

        view = df[df["country"].isin(sel_country) & df["league"].isin(sel_league)].copy()
        if "pick_group" in view.columns:
            view = view[view["pick_group"].fillna("1x2").isin(sel_groups)]
        # Filtro sul voto unificato (quello in tabella); fallback al voto mercato
        if "score_unified" in view.columns:
            view = view[view["score_unified"].isna() | (view["score_unified"] >= min_score)]
        else:
            view = view[view["score"].isna() | (view["score"] >= min_score)]
        view["quota_pick"] = view.apply(_quota_consiglio, axis=1)
        view = view[view["quota_pick"].isna() | ((view["quota_pick"] >= odd_min) & (view["quota_pick"] <= odd_max))]
        ev_col = _frac_series(view["ev_cons"] if "ev_cons" in view.columns else view["ev"])
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
        st.caption(
            "Colonne chiave dopo le squadre: **Indice gioca · Azione · EV cons. · Voto unificato · Gioca · Mercato · Kelly**. "
            "L'**Indice gioca** (0–100) unifica tutto per l'ordinamento."
        )
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
        st.dataframe(
            _prepare_calendario_show(view),
            width="stretch",
            hide_index=True,
            column_config={
                "EV cons.": st.column_config.NumberColumn("EV cons.", format="percent"),
                "Edge pp": st.column_config.NumberColumn("Edge pp", format="percent"),
                "EV sharp": st.column_config.NumberColumn("EV sharp", format="percent"),
                "Kelly ¼": st.column_config.NumberColumn("Kelly ¼", format="percent"),
                "CLV vs apertura": st.column_config.NumberColumn("CLV vs apertura", format="percent"),
                "Prob.": st.column_config.NumberColumn("Prob.", format="percent"),
                "Voto unificato": st.column_config.NumberColumn("Voto unificato", format="%d"),
                "Indice gioca": st.column_config.NumberColumn("Indice gioca", format="%.1f"),
                "Score 0–100": st.column_config.NumberColumn("Score 0–100", format="%.0f"),
                "Risk": st.column_config.NumberColumn("Risk", format="%.0f"),
                "Confidence": st.column_config.NumberColumn("Confidence", format="%.0f"),
                "Priorità": st.column_config.NumberColumn("Priorità", format="%.0f"),
            },
        )
        # CSV: percentuali leggibili (ordinamento numerico resta sul selettore UI)
        csv_show = _prepare_calendario_show(view).copy()
        for col in ("EV cons.", "Edge pp", "EV sharp", "Kelly ¼", "CLV vs apertura", "Prob."):
            if col in csv_show.columns:
                csv_show[col] = csv_show[col].map(lambda x: _pct(x) if pd.notna(x) else None)
        csv_bytes = csv_show.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Esporta CSV (vista filtrata)",
            data=csv_bytes,
            file_name="calendario_filtrato.csv",
            mime="text/csv",
            key="cal_csv_export",
        )
        # Export JSON completo (analisi esterna)
        json_rows = []
        for _, r in view.iterrows():
            raw = None
            for u in upcoming:
                if (
                    str(u.get("date")) == str(r.get("date"))
                    and str(u.get("home")) == str(r.get("home"))
                    and str(u.get("away")) == str(r.get("away"))
                ):
                    raw = u
                    break
            if not raw:
                continue
            pred = raw.get("prediction") if isinstance(raw.get("prediction"), dict) else {}
            json_rows.append(
                {
                    "date": raw.get("date"),
                    "home": raw.get("home"),
                    "away": raw.get("away"),
                    "league": raw.get("league"),
                    "pick": raw.get("pick"),
                    "action": raw.get("action"),
                    "score_unified": raw.get("score_unified"),
                    "ev_cons": raw.get("ev_cons"),
                    "probability": raw.get("probability"),
                    "quota_pick": raw.get("quota_pick"),
                    "agree_share": (raw.get("source_agreement") or {}).get("agree_share"),
                    "no_bet_reasons": raw.get("no_bet_reasons"),
                    "model_probabilities": pred.get("model_probabilities"),
                    "conformal_intervals": raw.get("conformal_intervals") or pred.get("conformal_intervals"),
                    "prob_intervals": raw.get("prob_intervals"),
                    "residual_ev": raw.get("residual_ev"),
                    "data_signal": raw.get("data_signal") or pred.get("data_signal"),
                    "quadro": raw.get("quadro"),
                    "montecarlo": pred.get("montecarlo"),
                }
            )
        if json_rows:
            import json as _json

            st.download_button(
                "Esporta JSON completo (analisi)",
                data=_json.dumps(json_rows, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="calendario_completo.json",
                mime="application/json",
                key="cal_json_export",
            )


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
            by_fx = {
                (str(u.get("date")), str(u.get("home")), str(u.get("away"))): u
                for u in upcoming
            }
            for i, (_, row) in enumerate(view.head(max_rows).iterrows()):
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
                    raw = by_fx.get((str(row.get("date")), str(row.get("home")), str(row.get("away"))))
                    if isinstance(raw, dict):
                        if st.button(
                            "Copia / esporta questo dettaglio",
                            key=f"cal_voto_prep_{i}",
                            icon=":material/content_copy:",
                        ):
                            st.session_state["vote_copy_open"] = i
                        if st.session_state.get("vote_copy_open") == i:
                            render_vote_copy(raw, key=f"cal_voto_{i}")
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
                    pred_row = row.get("prediction") if isinstance(row.get("prediction"), dict) else {}
                    _render_sportly_sim(
                        pred_row.get("sportly_sim") or (row.get("sportly_sim") if isinstance(row.get("sportly_sim"), dict) else None),
                        home=str(row.get("home") or "Casa"),
                        away=str(row.get("away") or "Trasferta"),
                    )
                    _render_data_signal(
                        pred_row.get("data_signal")
                        or (row.get("data_signal") if isinstance(row.get("data_signal"), dict) else None)
                    )
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
                today = date.today()
                start = min(max(today, dmin), dmax)
                picked = st.date_input(
                    "Intervallo date",
                    value=(start, dmax),
                    min_value=dmin,
                    max_value=dmax,
                    key=f"mkt_dates_{today.isoformat()}",
                )
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
        if not filt.empty:
            partite = sorted((filt["Data"].astype(str) + " · " + filt["Partita"]).drop_duplicates().tolist())
            scelta = st.selectbox(
                "Copia dettaglio voto di una partita",
                ["—"] + partite,
                key="mkt_copy_match",
            )
            if scelta != "—":
                data_s, _, partita = scelta.partition(" · ")
                raw_m = next(
                    (
                        u
                        for u in upcoming
                        if str(u.get("date")) == str(data_s)
                        and f"{u.get('home')} vs {u.get('away')}" == partita
                    ),
                    None,
                )
                if raw_m:
                    render_vote_copy(raw_m, key="mkt_voto_copy")

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
                    pred = predict_pipeline(home, away, n_sims, odds=extra_odds)
                render_advice(pred, extra_odds)
        elif last:
            st.caption("Ultima predizione salvata")
            render_advice(last, extra_odds)

with tab_eval:
    cal = load_calibration()
    summary = cal.get("backtest_summary") or {}

    with st.expander("Disaccordi modello (debug)", expanded=False):
        st.caption("Confronta lean ML / mercato / xG / FotMob / MC sulle partite del calendario.")
        if upcoming:
            disc_rows = []
            for u in upcoming:
                pred = u.get("prediction") if isinstance(u.get("prediction"), dict) else {}
                ml = pred.get("model_probabilities") or {}
                mc = pred.get("montecarlo") or {}
                ds = pred.get("data_signal") or u.get("data_signal") or {}
                sa = u.get("source_agreement") or {}
                feat = pred.get("features") or {}
                fm_xg = pred.get("fotmob_xg") or {}
                us = pred.get("understat_context") or {}

                def _lean(p1, px, p2):
                    vals = [("1", p1), ("X", px), ("2", p2)]
                    vals = [(k, v) for k, v in vals if v is not None]
                    if not vals:
                        return None
                    return max(vals, key=lambda t: t[1])[0]

                def _xg_lean(h_diff, a_diff):
                    try:
                        d = float(h_diff) - float(a_diff)
                    except (TypeError, ValueError):
                        return None
                    if d > 0.25:
                        return "1"
                    if d < -0.25:
                        return "2"
                    return "X"

                ml_l = _lean(ml.get("home_win"), ml.get("draw"), ml.get("away_win"))
                mc_l = _lean(mc.get("home_win"), mc.get("draw"), mc.get("away_win"))
                mkt_l = _lean(feat.get("mkt_p_home"), feat.get("mkt_p_draw"), feat.get("mkt_p_away"))
                ds_l = ds.get("lean") if isinstance(ds, dict) else None
                us_l = _xg_lean((us.get("home") or {}).get("xg_diff"), (us.get("away") or {}).get("xg_diff"))
                fm_l = _xg_lean((fm_xg.get("home") or {}).get("xg_diff"), (fm_xg.get("away") or {}).get("xg_diff"))
                pick = u.get("pick")
                flags = []
                if ml_l and mc_l and ml_l != mc_l:
                    flags.append("ML≠MC")
                if ml_l and mkt_l and ml_l != mkt_l:
                    flags.append("ML≠mkt")
                if ml_l and ds_l and ml_l != ds_l:
                    flags.append("ML≠dati")
                if ml_l and us_l and ml_l != us_l:
                    flags.append("ML≠xG")
                if ml_l and fm_l and ml_l != fm_l:
                    flags.append("ML≠FotMob")
                if pick in {"1", "X", "2"} and ml_l and pick != ml_l:
                    flags.append("pick≠ML")
                if sa.get("status") in {"spezzato", "debole"}:
                    flags.append(f"accordo:{sa.get('status')}")
                iv = (
                    u.get("conformal_intervals")
                    or u.get("prob_intervals")
                    or (mc.get("prob_intervals") if isinstance(mc, dict) else {})
                    or {}
                )
                if iv.get("fragile"):
                    flags.append("IC fragile")
                if flags:
                    disc_rows.append(
                        {
                            "Partita": f"{u.get('home')} vs {u.get('away')}",
                            "Pick": pick,
                            "ML": ml_l,
                            "Mkt": mkt_l,
                            "MC": mc_l,
                            "Dati": ds_l,
                            "xG": us_l,
                            "FotMob": fm_l,
                            "Cluster": pred.get("model_cluster"),
                            "Accordo": sa.get("agree_share"),
                            "Flag": ", ".join(flags),
                            "Voto": u.get("score_unified") or u.get("score"),
                        }
                    )
            if disc_rows:
                st.dataframe(pd.DataFrame(disc_rows), width="stretch", hide_index=True)
            else:
                st.caption("Nessun disaccordo evidente sul calendario filtrato.")

            # Timeline disaccordi da storico settled
            st.markdown("**Timeline disaccordi (storico)**")
            try:
                from modules.data_update.history import load_history

                hist = [h for h in load_history() if h.get("hit") is not None]
                tl = []
                for h in hist[-120:]:
                    tl.append(
                        {
                            "Data": h.get("date"),
                            "Partita": f"{h.get('home')} vs {h.get('away')}",
                            "Pick": h.get("pick"),
                            "Hit": h.get("hit"),
                            "EV": h.get("ev_cons"),
                            "Accordo": h.get("agree_share"),
                            "Residual": h.get("residual"),
                            "Quota": h.get("quota_pick"),
                            "Cluster": h.get("model_cluster"),
                        }
                    )
                if tl:
                    st.dataframe(pd.DataFrame(tl), width="stretch", hide_index=True)
                else:
                    st.caption("Nessuna riga settled con metadati.")
            except Exception as exc:
                st.caption(f"Timeline non disponibile: {exc}")
        else:
            st.caption("Calendario vuoto.")

    with st.expander("Storico esiti analisi", expanded=True):
        from modules.advisor.analysis_outcomes import load_analysis_outcomes, refresh_analysis_outcomes

        ao_view = st.radio(
            "Campione",
            ["Tutti settled", "Solo live", "Solo trainable"],
            horizontal=True,
            key="ao_view",
        )
        if st.button("Aggiorna storico esiti", key="ao_refresh"):
            refresh_analysis_outcomes()
            st.rerun()
        try:
            from modules.data_update.history import (
                MONTHLY_SUCCESS_CSV,
                PLAYS_CSV,
                export_plays_csv,
            )

            csv_info = export_plays_csv()
            monthly = csv_info.get("monthly_success") or {}
            st.markdown("**Successo mensile voti 7–10** (`storico_successo_mensile.csv`)")
            if monthly.get("hit_rate") is not None:
                st.caption(
                    "Solo giocate consigliate (Gioca) con voto unificato 7/8/9/10. "
                    f"Chiuse {monthly.get('n_closed', 0)} · hit {monthly['hit_rate']:.0%}."
                )
            else:
                st.caption(
                    "Solo giocate consigliate (Gioca) con voto unificato 7/8/9/10. "
                    "Ancora poche partite chiuse: la % si riempie mese per mese."
                )
            if MONTHLY_SUCCESS_CSV.is_file():
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        "Scarica successo mensile (per voto)",
                        data=MONTHLY_SUCCESS_CSV.read_bytes(),
                        file_name="storico_successo_mensile.csv",
                        mime="text/csv",
                        key="ao_csv_monthly",
                    )
                wide = MONTHLY_SUCCESS_CSV.with_name("storico_successo_mensile_largo.csv")
                with c2:
                    if wide.is_file():
                        st.download_button(
                            "Scarica vista larga (1 riga = 1 mese)",
                            data=wide.read_bytes(),
                            file_name="storico_successo_mensile_largo.csv",
                            mime="text/csv",
                            key="ao_csv_monthly_wide",
                        )
                try:
                    show_m = pd.read_csv(MONTHLY_SUCCESS_CSV, sep=";")
                    st.dataframe(show_m, width="stretch", hide_index=True)
                except Exception:
                    pass
            st.caption(
                f"Dettaglio singole giocate: `{PLAYS_CSV.name}` "
                f"({csv_info.get('n', 0)} righe). Separatore `;` in Excel."
            )
            if PLAYS_CSV.is_file():
                st.download_button(
                    "Scarica dettaglio giocate",
                    data=PLAYS_CSV.read_bytes(),
                    file_name="storico_giocate.csv",
                    mime="text/csv",
                    key="ao_csv_dl",
                )
        except Exception as exc:
            st.caption(f"CSV storico non disponibile: {exc}")
        ao = load_analysis_outcomes()
        if ao is None:
            ao = refresh_analysis_outcomes()
        if ao_view == "Solo live":
            data = ao.get("live_summary") or ao.get("live_summary") or ao
        elif ao_view == "Solo trainable":
            data = ao.get("trainable_summary") or ao.get("trainable_summary") or ao
        else:
            data = ao
        st.caption(
            f"Ultimo aggiornamento: {data.get('updated_at', '—')[:19]} · "
            f"{data.get('n_in_pool', 0)} analisi chiuse con esito · "
            f"{data.get('n_with_score_unified', 0)} con voto unificato"
        )
        if data.get("highlights"):
            for line in data["highlights"][:8]:
                st.markdown(f"- {line}")

        gioca = data.get("gioca") or {}
        if gioca.get("n"):
            st.markdown("**Giocate consigliate (action = gioca)**")
            g1, g2, g3, g4 = st.columns(4)
            rate = gioca.get("hit_rate")
            g1.metric("Hit rate", "—" if rate is None else f"{rate:.0%}", delta=gioca.get("trend"))
            g2.metric("Esiti", gioca.get("label") or "0/0")
            last10 = gioca.get("last_10") or {}
            g3.metric(
                "Ultime 10",
                last10.get("label") or "—",
                delta=None if last10.get("hit_rate") is None else f"{last10['hit_rate']:.0%}",
            )
            hv = gioca.get("high_vote") or {}
            g4.metric("Voto ≥8", hv.get("label") or "—")
            if gioca.get("by_week"):
                st.caption("Andamento settimanale (hit rate cumula le giocate consigliate chiuse)")
                week_df = pd.DataFrame(gioca["by_week"]).rename(
                    columns={
                        "key": "Settimana",
                        "n": "N",
                        "hits": "Prese",
                        "misses": "Sbagliate",
                        "hit_rate": "Hit rate",
                        "label": "Riepilogo",
                    }
                )
                if "Hit rate" in week_df.columns:
                    week_df["Hit rate"] = week_df["Hit rate"].map(
                        lambda x: f"{x:.0%}" if pd.notna(x) else "—"
                    )
                st.dataframe(week_df, width="stretch", hide_index=True)
            if gioca.get("cumulative") and len(gioca["cumulative"]) >= 5:
                cum = pd.DataFrame(gioca["cumulative"])
                st.line_chart(cum.set_index("n")[["hit_rate"]], height=180)

        uni = data.get("by_score_unified") or []
        if uni:
            st.markdown("**Per voto unificato (1–10)**")
            show_uni = pd.DataFrame(uni).rename(
                columns={
                    "key": "Voto unificato",
                    "n": "Analisi",
                    "hits": "Prese",
                    "misses": "Sbagliate",
                    "hit_rate": "Hit rate",
                    "label": "Riepilogo",
                }
            )
            if "hit_rate" in show_uni.columns:
                show_uni["Hit rate"] = show_uni["Hit rate"].map(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
            st.dataframe(show_uni, width="stretch", hide_index=True)
        else:
            st.info(
                "Nessun esito con voto unificato ancora. "
                "Serve archiviare pre-match (Solo quote) con score_unified, poi settle."
            )
        c_a1, c_a2 = st.columns(2)
        with c_a1:
            if data.get("by_action"):
                st.markdown("**Per azione (gioca / no_bet)**")
                st.dataframe(pd.DataFrame(data["by_action"]), width="stretch", hide_index=True)
        with c_a2:
            if data.get("by_market"):
                st.markdown("**Per mercato**")
                st.dataframe(pd.DataFrame(data["by_market"]), width="stretch", hide_index=True)
        if data.get("by_quadro_consensus"):
            st.markdown("**Per accordo quadro (fonti allineate, es. 10/10)**")
            st.caption(
                "Non è il voto unificato 1–10: è quante fonti tattiche/ML/book concordavano sul pick."
            )
            st.dataframe(pd.DataFrame(data["by_quadro_consensus"]), width="stretch", hide_index=True)
        if data.get("recent"):
            st.markdown("**Ultime analisi chiuse**")
            st.dataframe(pd.DataFrame(data["recent"]), width="stretch", hide_index=True)

    with st.expander("Paper trading (SQLite)", expanded=False):
        from modules.advisor.paper_stats import paper_trading_report
        from modules.advisor.residual_ev import fit_residual_ev
        from modules.advisor.data_signal_weights import optimize_weights

        st.caption(
            "Report su righe **trainable** (151+). Kelly con **drawdown guard**. "
            "CLV da quota archiviata vs close Asian/fd."
        )
        cbtn1, cbtn2, cbtn3 = st.columns(3)
        with cbtn1:
            if st.button("Aggiorna report paper + residual EV"):
                fit_info = fit_residual_ev(aggressive=True)
                st.write(fit_info)
        with cbtn2:
            if st.button("Ottimizza pesi Analisi dati"):
                st.write(optimize_weights(aggressive=True))
        with cbtn3:
            if st.button("Apprendi da partite chiuse"):
                from modules.advisor.online_learn import learn_from_settled

                st.write(learn_from_settled(force=True, aggressive=True))
        rep = paper_trading_report()
        if not rep.get("ok"):
            st.warning(rep.get("error") or "report non disponibile")
        elif not rep.get("n"):
            st.caption(rep.get("note") or "nessun esito")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Trainable", rep["n"], delta=f"tot {rep.get('n_settled_total', rep['n'])}")
            m2.metric("Flat ROI", f"{rep.get('flat_roi', 0):.1%}")
            m3.metric(
                "ROI @ quote",
                "n/d" if rep.get("odds_roi") is None else f"{rep.get('odds_roi'):.1%}",
                delta=f"live n={rep.get('n_live_odds', rep.get('odds_n', 0))}",
            )
            m4.metric(
                "CLV medio",
                "n/d" if rep.get("mean_clv") is None else f"{rep.get('mean_clv'):+.2%}",
            )
            ke = rep.get("kelly") or {}
            oe = rep.get("odds_equity") or {}
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Kelly bank end", ke.get("bankroll_end", "n/d"))
            k2.metric("Kelly risk scale", ke.get("risk_scale", "n/d"))
            k3.metric("Max DD (odds)", oe.get("max_drawdown", "n/d"))
            k4.metric("Sharpe (odds)", oe.get("sharpe", "n/d"))
            if rep.get("walk_forward_odds_roi"):
                st.markdown("**Walk-forward ROI @ quote**")
                st.dataframe(pd.DataFrame(rep["walk_forward_odds_roi"]), width="stretch", hide_index=True)
            if rep.get("by_market"):
                st.markdown("**Per mercato (pick_group)**")
                st.dataframe(pd.DataFrame(rep.get("by_market") or []), width="stretch", hide_index=True)
            if rep.get("by_odds_band"):
                st.markdown("**Per fascia quota**")
                st.dataframe(pd.DataFrame(rep.get("by_odds_band") or []), width="stretch", hide_index=True)
            if rep.get("by_cluster"):
                st.markdown("**Per cluster**")
                st.dataframe(pd.DataFrame(rep.get("by_cluster") or []), width="stretch", hide_index=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Per voto**")
                st.dataframe(pd.DataFrame(rep.get("by_score") or []), width="stretch", hide_index=True)
            with c2:
                st.markdown("**Per pick**")
                st.dataframe(pd.DataFrame(rep.get("by_pick") or []), width="stretch", hide_index=True)
            st.markdown("**Per lega (top)**")
            st.dataframe(pd.DataFrame(rep.get("by_league") or []), width="stretch", hide_index=True)

    if not cal.get("fitted_at") and not summary:
        st.info("Nessuna valutazione. Premi **Calibra probabilità (backtest)** nella colonna a sinistra (meglio dopo **Aggiorna dati + modello** per lo split rolling).")
    else:
        st.caption(
            "Walk-forward temporale (OOF): Brier, log-loss, ECE e CLV. "
            "Le scommesse simulate usano ¼ Kelly con cap, edge minimo 2–3% e scarto se Pinnacle non offre edge."
        )
        split = summary.get("split") or cal.get("split") or "—"
        n_lg = len(cal.get("temperature_by_league") or {})
        lg_bit = f" · T per {n_lg} campionati" if n_lg else ""
        st.caption(f"Protocollo: **{split}** · T={cal.get('temperature', 1):.2f}{lg_bit} · EV min {cal.get('min_ev_play', 0.025):.1%}")
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
            online_rel = cal.get("reliability_1x2_online") or []
            if online_rel and any(int(b.get("n") or 0) < int(cal.get("min_bin_samples") or 30) for b in online_rel):
                st.caption(
                    "I bin sopra sono OOF (train). I bin online da settled hanno n troppo piccolo "
                    "e non entrano in EV/Kelly finché ogni bin non ha ≥30 esiti."
                )
        st.caption(
            "CLV storico: quota di apertura (venerdì / AvgH) contro la close (AvgCH / B365CH). "
            "Positivo = hai battuto la linea di chiusura. I tipster non entrano in queste metriche. "
            "No-bet rigido: EV/sharp sotto soglia, steam contrario, pick fuori set conformal, accordo spezzato. "
            "IC largo o set 1X2 a 3 esiti → voto/Kelly ridotti, non veto."
        )
