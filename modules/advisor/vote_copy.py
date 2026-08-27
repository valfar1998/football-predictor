"""Copia/esporta il dettaglio voto: testo di ogni quota + scheda grafica."""

from __future__ import annotations

import math
import re
from typing import Any

import streamlit as st

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

_KIND = {
    "più_probabile": "più probabile",
    "valore": "miglior rapporto probabilità/quota",
    "probabile_e_valore": "più probabile e miglior value",
    "invalido": "pick invalido (quote assenti)",
    "nessun_pick": "nessun pick (fonti non generano)",
    "lean_esterno": "nessun pick (fonti non generano)",
}

_COPY_HTML = """
<div id="vote-copy">
  <div class="bar">
    <button type="button" id="btn-all">Copia scheda</button>
    <button type="button" id="btn-png">Scarica PNG</button>
    <button type="button" id="btn-text">Copia testo lungo</button>
    <button type="button" id="btn-txt">Scarica testo</button>
    <span id="status"></span>
  </div>
  <div class="sheet">
    <canvas id="card" aria-hidden="true"></canvas>
    <img id="preview" alt="Scheda voto unica formato carta d'identità (CIE)" />
  </div>
  <p class="hint">Una sola scheda · formato CIE 85,6 × 54 mm (come la carta d'identità italiana)</p>
</div>
"""

_COPY_CSS = """
#vote-copy { font-family: var(--st-font, sans-serif); }
.bar {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  align-items: center;
  margin-bottom: 0.55rem;
}
.bar button {
  background: var(--st-secondary-background-color, #262730);
  color: var(--st-text-color, #fafafa);
  border: 1px solid var(--st-border-color, #555);
  border-radius: var(--st-button-radius, 0.5rem);
  padding: 0.28rem 0.7rem;
  font-size: 0.85rem;
  cursor: pointer;
}
.bar button#btn-all {
  background: var(--st-primary-color, #ff4b4b);
  color: var(--st-background-color, #fff);
  border-color: transparent;
  font-weight: 600;
}
#status { font-size: 0.8rem; opacity: 0.8; }
.sheet {
  display: flex;
  justify-content: flex-start;
  align-items: flex-start;
}
#card { display: none; }
#preview {
  /* CIE / ID-1: 85.60 × 53.98 mm */
  width: 85.6mm;
  height: 53.98mm;
  max-width: min(85.6mm, 100%);
  object-fit: contain;
  object-position: top left;
  border: 1px solid var(--st-border-color, #444);
  border-radius: 3mm;
  background: var(--st-secondary-background-color, #1e1e1e);
  box-shadow: 0 2px 10px rgba(0,0,0,0.25);
}
.hint {
  margin: 0.45rem 0 0;
  font-size: 0.78rem;
  opacity: 0.75;
}
"""

_COPY_JS = r"""
export default function (component) {
  const { data, parentElement } = component
  const root = parentElement.querySelector("#vote-copy")
  if (!root) return
  const canvas = parentElement.querySelector("#card")
  const preview = parentElement.querySelector("#preview")
  const status = parentElement.querySelector("#status")
  if (!canvas) return
  const card = (data && data.card) || {}
  const text = (data && data.text) || ""
  const filename = (data && data.filename) || "voto"

  const styleHost = resolveStyleHost(root, parentElement)

  drawCard(canvas, card, styleHost)
  if (preview && canvas.width) {
    preview.src = canvas.toDataURL("image/png")
  }

  const setStatus = (msg) => { if (status) status.textContent = msg }

  const on = (sel, fn) => {
    const el = parentElement.querySelector(sel)
    if (el) el.onclick = fn
  }
  on("#btn-text", async () => {
    try {
      await navigator.clipboard.writeText(text)
      setStatus("Testo lungo copiato")
    } catch (err) {
      setStatus("Copia testo non riuscita — usa Scarica testo")
    }
  })
  on("#btn-all", async () => {
    try {
      const blob = await canvasToBlob(canvas)
      const html = '<img src="' + canvas.toDataURL("image/png") + '" width="323" height="204" alt="scheda voto">'
      try {
        await navigator.clipboard.write([
          new ClipboardItem({
            "image/png": blob,
            "text/plain": new Blob([cardFaceText(card)], { type: "text/plain" }),
            "text/html": new Blob([html], { type: "text/html" }),
          }),
        ])
        setStatus("Scheda unica copiata (formato CIE)")
        return
      } catch (err) {
        await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })])
        setStatus("Scheda PNG copiata")
      }
    } catch (err) {
      setStatus("Copia non riuscita — usa Scarica PNG")
    }
  })
  on("#btn-png", () => downloadUrl(canvas.toDataURL("image/png"), filename + ".png"))
  on("#btn-txt", () => {
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" })
    downloadUrl(URL.createObjectURL(blob), filename + ".txt")
  })
}

function cardFaceText(card) {
  const lines = (card && card.face_lines) || []
  return [
    card.title || "",
    (card.action || "") + " " + (card.pick || "") + " " + (card.pick_name || ""),
    ...lines,
  ].filter(Boolean).join("\\n")
}

function resolveStyleHost(root, parent) {
  if (root && root.nodeType === 1) return root
  if (parent && parent.nodeType === 1) return parent
  if (parent && parent.host && parent.host.nodeType === 1) return parent.host
  return document.documentElement
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
}

function canvasToBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("toBlob"))), "image/png")
  })
}

function downloadUrl(url, name) {
  const a = document.createElement("a")
  a.href = url
  a.download = name
  a.click()
}

function css(el, name, fallback) {
  try {
    const host = el && el.nodeType === 1 ? el : document.documentElement
    const v = getComputedStyle(host).getPropertyValue(name).trim()
    return v || fallback
  } catch (err) {
    return fallback
  }
}

function drawCard(canvas, card, host) {
  // CIE / ID-1: 85.60 × 53.98 mm @ 12 px/mm → ~1027×648 (nitido in copia/print)
  const W = 1027
  const H = 648
  const pad = 18
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  canvas.width = Math.round(W * dpr)
  canvas.height = Math.round(H * dpr)
  canvas.style.width = W + "px"
  canvas.style.height = H + "px"
  const ctx = canvas.getContext("2d")
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

  const bg = css(host, "--st-background-color", "#0e1117")
  const panel = css(host, "--st-secondary-background-color", "#1c1e26")
  const fg = css(host, "--st-text-color", "#fafafa")
  const muted = css(host, "--st-secondary-text-color", "#a3a8b8")
  const border = css(host, "--st-border-color", "#3d4250")
  const primary = css(host, "--st-primary-color", "#ff4b4b")
  const hi = "#3dd68c"
  const mid = "#e6c35c"
  const lo = "#e06c75"

  roundRect(ctx, 0, 0, W, H, 22, panel)
  ctx.strokeStyle = border
  ctx.lineWidth = 2
  ctx.stroke()

  const midX = Math.floor(W * 0.52)
  // divisore verticale
  ctx.strokeStyle = border
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(midX, pad)
  ctx.lineTo(midX, H - pad)
  ctx.stroke()

  // --- SINISTRA: grafico compact ---
  let y = pad
  ctx.fillStyle = muted
  ctx.font = "11px sans-serif"
  ctx.fillText(clip(card.subtitle || "", 48), pad, y + 12)
  y += 18
  ctx.fillStyle = fg
  ctx.font = "700 18px sans-serif"
  ctx.fillText(clip(card.title || "Partita", 34), pad, y + 16)
  y += 28

  const action = String(card.action || "—")
  const tone = action === "GIOCA" ? hi : action === "NO BET" ? mid : lo
  ctx.fillStyle = tone
  ctx.font = "800 22px sans-serif"
  ctx.fillText(clip(action + "  " + (card.pick || ""), 28), pad, y + 18)
  y += 26
  ctx.fillStyle = muted
  ctx.font = "12px sans-serif"
  ctx.fillText(clip(card.pick_name || "", 40), pad, y + 6)
  y += 18

  const score = Number(card.score || 0)
  const unified = card.unified
  const barTone = score >= 7 ? hi : score >= 4 ? mid : lo
  const leftW = midX - pad - 12
  for (let i = 1; i <= 10; i++) {
    ctx.fillStyle = i <= score ? barTone : "rgba(250,250,250,0.12)"
    roundRect(ctx, pad + (i - 1) * ((leftW - 8) / 10), y, (leftW - 8) / 10 - 3, 9, 2, ctx.fillStyle)
  }
  y += 16
  ctx.fillStyle = fg
  ctx.font = "600 12px sans-serif"
  const scoreTxt = (card.score == null ? "—" : String(card.score)) + "/10"
  const uniTxt = unified == null ? "" : "  uni " + unified + "/10"
  const s100 = card.score_100 == null ? "" : "  Score " + Math.round(Number(card.score_100))
  const band = card.band ? " (" + card.band + ")" : ""
  ctx.fillText(clip(scoreTxt + uniTxt + s100 + band, 42), pad, y + 10)
  y += 20

  const metrics = card.metrics || []
  const cols = 3
  const boxW = (leftW - 8) / cols
  const boxH = 44
  metrics.slice(0, 6).forEach((m, i) => {
    const col = i % cols
    const row = Math.floor(i / cols)
    const x = pad + col * (boxW + 4)
    const yy = y + row * (boxH + 6)
    roundRect(ctx, x, yy, boxW - 2, boxH, 6, bg)
    ctx.fillStyle = muted
    ctx.font = "10px sans-serif"
    ctx.fillText(clip(m.label || "", 14), x + 6, yy + 14)
    ctx.fillStyle = fg
    ctx.font = "700 13px sans-serif"
    ctx.fillText(clip(String(m.value || "—"), 12), x + 6, yy + 32)
  })
  y += 2 * (boxH + 6) + 8

  const p1x2 = card.p1x2 || []
  if (p1x2.length) {
    ctx.fillStyle = fg
    ctx.font = "600 11px sans-serif"
    ctx.fillText("1X2", pad, y + 10)
    y += 14
    p1x2.forEach((row) => {
      ctx.fillStyle = muted
      ctx.font = "11px sans-serif"
      ctx.fillText(row.label, pad, y + 10)
      const maxW = leftW - 70
      const p = Math.max(0, Math.min(1, Number(row.p || 0)))
      roundRect(ctx, pad + 18, y, maxW, 10, 3, bg)
      roundRect(ctx, pad + 18, y, maxW * p, 10, 3, primary)
      ctx.fillStyle = fg
      ctx.fillText(pct(p), pad + 22 + maxW, y + 9)
      y += 16
    })
  }

  const mix = (card.mix || []).slice(0, 4)
  if (mix.length && y < H - 90) {
    y += 4
    ctx.fillStyle = fg
    ctx.font = "600 11px sans-serif"
    ctx.fillText("Mix", pad, y + 10)
    y += 14
    mix.forEach((row) => {
      ctx.fillStyle = muted
      ctx.font = "10px sans-serif"
      ctx.fillText(clip(row.label, 8), pad, y + 9)
      const maxW = leftW - 90
      const p = Math.max(0, Math.min(1, Number(row.p || 0)))
      roundRect(ctx, pad + 48, y, maxW, 8, 3, bg)
      roundRect(ctx, pad + 48, y, maxW * p, 8, 3, hi)
      y += 14
    })
  }

  // --- DESTRA: testo unificato (stesso foglio) ---
  let ry = pad
  const rx = midX + 14
  const rightW = W - rx - pad
  ctx.fillStyle = fg
  ctx.font = "700 13px sans-serif"
  ctx.fillText("Scheda unica", rx, ry + 12)
  ry += 20
  ctx.fillStyle = muted
  ctx.font = "10px sans-serif"
  ctx.fillText("testo + grafico · CIE 85.6×54 mm", rx, ry + 8)
  ry += 18

  const face = card.face_lines || []
  face.forEach((line) => {
    if (ry > H - pad - 70) return
    ctx.fillStyle = fg
    ctx.font = "12px sans-serif"
    wrapText(ctx, String(line), rx, ry, rightW, 15).forEach((ln) => {
      if (ry > H - pad - 70) return
      ctx.fillText(ln, rx, ry + 11)
      ry += 15
    })
    ry += 2
  })

  const rows = (card.markets || []).slice(0, 4)
  if (rows.length && ry < H - pad - 40) {
    ry += 4
    ctx.fillStyle = muted
    ctx.font = "10px sans-serif"
    ctx.fillText("Top quote", rx, ry + 10)
    ry += 14
    rows.forEach((m) => {
      if (ry > H - pad - 12) return
      ctx.fillStyle = fg
      ctx.font = "11px sans-serif"
      ctx.fillText(
        clip(
          String(m.code || "") + "  " + (m.odds == null ? "—" : m.odds) +
          "  " + (m.ev == null ? "" : signedPct(m.ev)) +
          "  " + clip(m.name || "", 22),
          44
        ),
        rx,
        ry + 10
      )
      ry += 15
    })
  }

  ctx.fillStyle = muted
  ctx.font = "9px sans-serif"
  ctx.fillText("EV/Kelly = modello+quota · quadro non entra nel conto", rx, H - pad)
}

function wrapText(ctx, text, x, y, maxW, lineH) {
  const words = String(text).split(/\\s+/)
  const lines = []
  let cur = ""
  words.forEach((w) => {
    const t = cur ? cur + " " + w : w
    if (ctx.measureText(t).width > maxW && cur) {
      lines.push(cur)
      cur = w
    } else {
      cur = t
    }
  })
  if (cur) lines.push(cur)
  return lines.slice(0, 3)
}

function pct(p) {
  return Math.round(Number(p) * 100) + "%"
}
function signedPct(p) {
  const n = Number(p) * 100
  const s = n >= 0 ? "+" : ""
  return s + Math.round(n) + "%"
}
function clip(s, n) {
  s = String(s)
  return s.length > n ? s.slice(0, n - 1) + "…" : s
}
function roundRect(ctx, x, y, w, h, r, fill) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.arcTo(x + w, y, x + w, y + h, r)
  ctx.arcTo(x + w, y + h, x, y + h, r)
  ctx.arcTo(x, y + h, x, y, r)
  ctx.arcTo(x, y, x + w, y, r)
  ctx.closePath()
  ctx.fillStyle = fill
  ctx.fill()
}
function spark(ctx, xs, ys, x, y, w, h, color) {
  if (!ys || ys.length < 2) return
  let min = Math.min(...ys)
  let max = Math.max(...ys)
  if (max <= min) max = min + 1
  ctx.beginPath()
  ys.forEach((v, i) => {
    const px = x + (w * i) / Math.max(ys.length - 1, 1)
    const py = y + h - ((v - min) / (max - min)) * h
    if (i === 0) ctx.moveTo(px, py)
    else ctx.lineTo(px, py)
  })
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  ctx.stroke()
}
"""

_copy_mount = st.components.v2.component(
    "vote_copy_card_v3",
    html=_COPY_HTML,
    css=_COPY_CSS,
    js=_COPY_JS,
)


def render_vote_copy(src: dict[str, Any], *, key: str) -> None:
    """Barra copia testo+grafico del dettaglio voto (ogni quota inclusa nel testo)."""
    if not isinstance(src, dict) or not src:
        return
    payload = build_vote_payload(src)
    with st.container(border=True):
        st.markdown("**Copia dettaglio voto**")
        st.caption(
            "Copia tutto mette negli appunti la scheda grafica (barre voto, 1X2, mix, risultati, xG) "
            "e, se il browser lo consente, anche il testo con ogni quota. "
            "Altrimenti usa Copia testo / Copia grafico oppure i download."
        )
        _copy_mount(
            data=payload,
            key=key,
            width="stretch",
        )


def build_vote_payload(src: dict[str, Any]) -> dict[str, Any]:
    card = _card(src)
    return {
        "card": _jsonable(card),
        "text": format_vote_text(src),
        "filename": card.get("filename") or "voto",
    }


def format_vote_text(src: dict[str, Any]) -> str:
    play = _play(src)
    pred = _pred(src)
    lines: list[str] = []
    title = src.get("match") or f"{src.get('home') or '?'} vs {src.get('away') or '?'}"
    lines += [
        title,
        " · ".join(
            p
            for p in (
                str(src.get("league") or pred.get("league") or "").strip(),
                str(src.get("date") or pred.get("date") or "").strip(),
                str(src.get("time") or "").strip(),
            )
            if p
        ),
        "",
    ]
    action = _action_label(play.get("action"))
    pick = play.get("code") or play.get("pick") or "—"
    lines.append(f"{action}  {pick}  {play.get('name') or play.get('pick_name') or ''}".rstrip())
    kind = _KIND.get(play.get("kind") or "", play.get("kind") or "")
    score = play.get("score")
    uni = play.get("score_unified")
    lines.append(
        f"Voto {('—' if score is None else str(int(score)) + '/10')}"
        f"  ·  unificato {('—' if uni is None else str(int(uni)) + '/10')}"
        + (f"  ·  {kind}" if kind else "")
    )
    if play.get("action") == "no_bet":
        lines.append("No bet: " + "; ".join(play.get("no_bet_reasons") or ["filtro edge/mercato"]))
    elif play.get("action") in {"invalido", "n/d"}:
        lines.append("; ".join(play.get("no_bet_reasons") or []))
    r1 = src.get("score_reason_1") or play.get("score_reason_1")
    r2 = src.get("score_reason_2") or play.get("score_reason_2")
    if r1:
        lines.append(str(r1))
    if r2:
        lines.append(str(r2))
    meta = src.get("meta_analysis") or play.get("meta_analysis") or {}
    if meta:
        lines.append(
            "Mix: "
            + (meta.get("note") or src.get("meta_note") or "")
            + (f"  ·  lettura {meta.get('label')}" if meta.get("label") else "")
        )
    elif src.get("meta_note"):
        lines.append("Mix: " + str(src.get("meta_note")))
    lines.append("")
    lines.append("Modello vs mercato")
    p_cons = play.get("p_cons") if play.get("p_cons") is not None else play.get("probability")
    lines.append(f"  P cons.     {_pct(p_cons)}")
    lines.append(f"  P mercato   {_pct(play.get('p_market'))}")
    lines.append(f"  Edge        {_signed_pct(play.get('edge_pp'))} pp")
    ev = play.get("ev_cons") if play.get("ev_cons") is not None else play.get("ev")
    lines.append(f"  EV cons.    {_signed_pct(ev)}")
    lines.append(f"  EV sharp    {_signed_pct(play.get('ev_sharp'))}")
    kq = play.get("kelly_quarter")
    lines.append(f"  Kelly ¼     {_pct(kq)}")
    odds, fair = _play_odds(play), _num(play.get("fair_odds"))
    if odds is not None and fair is not None:
        lines.append(f"  Quota       {odds:.2f} vs equa {fair:.2f}")
    elif odds is not None:
        lines.append(f"  Quota       {odds:.2f}")
    src_odds = play.get("odds_source") or src.get("odds_source") or "—"
    lines.append(f"  Fonte       {src_odds}")
    if play.get("value_note"):
        lines.append(f"  Nota value  {play['value_note']}")
    if play.get("clv") is not None:
        lines.append(f"  CLV         {_signed_pct(play.get('clv'))}")

    move = src.get("market_move") or {}
    if isinstance(move, dict) and (move.get("movement_comment") or move.get("movement_level")):
        lines += ["", "Mercato asiatico"]
        lines.append(f"  Intensità   {move.get('movement_level') or '—'}")
        if move.get("movement_comment"):
            lines.append(f"  {move['movement_comment']}")
        elif move.get("movement_summary"):
            lines.append(f"  {move['movement_summary']}")

    val = src.get("validation") or play.get("validation") or {}
    if isinstance(val, dict) and val:
        lines += ["", "Validazione (non EV)"]
        if val.get("summary"):
            lines.append(f"  {val['summary']}")
        warns = val.get("warnings") or []
        if warns:
            lines.append("  Warning: " + " · ".join(str(w) for w in warns[:6]))

    sig = src.get("data_signal") or pred.get("data_signal") or {}
    if isinstance(sig, dict) and sig.get("ready"):
        lines += ["", "Analisi dati"]
        lines.append(
            f"  Lean {sig.get('lean') or '—'}  ·  edge {_num_txt(sig.get('edge'))}"
            f"  ·  conf {_pct(sig.get('confidence'))}"
        )
        if sig.get("note"):
            lines.append(f"  {sig['note']}")

    quadro = src.get("quadro") or {}
    if isinstance(quadro, dict) and quadro:
        lines += ["", "Quadro analisi"]
        if quadro.get("summary"):
            lines.append(f"  {quadro['summary']}")
        lines.append(
            f"  Consenso {quadro.get('consenso') or '—'}  ·  "
            f"allineate {quadro.get('agree_n') or 0}/{quadro.get('votes_n') or 0}"
        )
        for s in quadro.get("sources") or []:
            lines.append(
                f"  - {s.get('fonte')}: {s.get('pick') or '—'}  "
                f"({_pct(s.get('p_1'))}/{_pct(s.get('p_x'))}/{_pct(s.get('p_2'))})  "
                f"{s.get('nota') or ''}".rstrip()
            )

    tip = src.get("tipster") or play.get("tipster") or {}
    if isinstance(tip, dict) and tip.get("n_sources"):
        lines += ["", f"Tipster  consenso {tip.get('consensus') or '—'}  vs modello {tip.get('agree') or '—'}"]
        for s in tip.get("sources") or []:
            lines.append(f"  - {s.get('source')}: {s.get('pick')}")

    sim = src.get("sportly_sim") or pred.get("sportly_sim") or {}
    if isinstance(sim, dict) and sim.get("ready"):
        xg = sim.get("xg") or {}
        lines += ["", "Sportly-sim"]
        lines.append(
            f"  Lean {sim.get('lean') or '—'}  ·  xG {xg.get('home', '—')} – {xg.get('away', '—')}"
        )
        if sim.get("note"):
            lines.append(f"  {sim['note']}")

    scores = src.get("most_likely_scores") or (pred.get("montecarlo") or {}).get("most_likely_scores") or []
    if scores:
        lines += ["", "Risultati più probabili"]
        for s in scores[:10]:
            lines.append(f"  {s.get('score')}  {_pct(s.get('prob'))}")

    xg_exp = src.get("expected_goals") or pred.get("expected_goals") or {}
    if xg_exp:
        lines += ["", f"xG attesi  {xg_exp.get('home', '—')} – {xg_exp.get('away', '—')}"]

    markets = _markets(src)
    if markets:
        lines += ["", "Quote (tutte, con voto)"]
        lines.append(
            f"{'Gruppo':<22} {'Codice':<14} {'Mercato':<28} {'Prob':>6} {'Pcons':>6} "
            f"{'Quota':>6} {'Equa':>6} {'EV':>7} {'Voto':>4}  Fonte"
        )
        for m in markets:
            grp = GROUP_LABEL.get(m.get("group"), m.get("group") or "")
            ev_m = m.get("ev_cons") if m.get("ev_cons") is not None else m.get("ev")
            lines.append(
                f"{str(grp)[:22]:<22} {str(m.get('code') or '')[:14]:<14} "
                f"{str(m.get('name') or '')[:28]:<28} "
                f"{_pct(m.get('probability')):>6} {_pct(m.get('p_cons')):>6} "
                f"{_odd(m.get('odds')):>6} {_odd(m.get('fair_odds')):>6} "
                f"{_signed_pct(ev_m):>7} {_score_txt(m.get('score')):>4}  "
                f"{m.get('odds_source') or '—'}"
            )
    lines.append("")
    lines.append("EV e Kelly restano su modello + quota reale. Quadro e fonti extra non entrano nel conto.")
    return "\n".join(lines).strip() + "\n"


def _card(src: dict[str, Any]) -> dict[str, Any]:
    play = _play(src)
    pred = _pred(src)
    home = src.get("home") or pred.get("home") or "Casa"
    away = src.get("away") or pred.get("away") or "Trasferta"
    title = src.get("match") or f"{home} vs {away}"
    subtitle = " · ".join(
        p
        for p in (
            str(src.get("league") or pred.get("league") or "").strip(),
            str(src.get("date") or pred.get("date") or "").strip(),
            str(src.get("time") or "").strip(),
        )
        if p
    )
    p_cons = play.get("p_cons") if play.get("p_cons") is not None else play.get("probability")
    ev = play.get("ev_cons") if play.get("ev_cons") is not None else play.get("ev")
    odds, fair = _play_odds(play), _num(play.get("fair_odds"))
    quota_txt = "—"
    if odds is not None and fair is not None:
        quota_txt = f"{odds:.2f} vs {fair:.2f}"
    elif odds is not None:
        quota_txt = f"{odds:.2f}"
    metrics = [
        {"label": "P cons.", "value": _pct(p_cons)},
        {"label": "P mercato", "value": _pct(play.get("p_market"))},
        {"label": "Edge", "value": _signed_pct(play.get("edge_pp"))},
        {"label": "EV cons.", "value": _signed_pct(ev)},
        {"label": "Kelly ¼", "value": _pct(play.get("kelly_quarter"))},
        {"label": "Quota vs equa", "value": quota_txt},
    ]
    p1x2 = []
    ml = pred.get("model_probabilities") or {}
    for lab, key, mlk in (("1", "p_home", "home"), ("X", "p_draw", "draw"), ("2", "p_away", "away")):
        p = _num(src.get(key) if src.get(key) is not None else play.get(key) if play.get(key) is not None else ml.get(mlk))
        if p is not None:
            p1x2.append({"label": lab, "p": p})
    mix = _mix_rows(src, play)
    scores_src = src.get("most_likely_scores") or (pred.get("montecarlo") or {}).get("most_likely_scores") or []
    scores = [{"score": s.get("score"), "p": _num(s.get("prob"))} for s in scores_src[:8] if s]
    sim = src.get("sportly_sim") or pred.get("sportly_sim") or {}
    xg = (sim.get("xg") or {}) if isinstance(sim, dict) else {}
    markets = []
    for m in _markets(src):
        ev_m = m.get("ev_cons") if m.get("ev_cons") is not None else m.get("ev")
        markets.append(
            {
                "code": m.get("code"),
                "name": m.get("name"),
                "odds": _odd_num(m.get("odds")),
                "prob": _num(m.get("probability")),
                "ev": _num(ev_m),
                "score": _int(m.get("score")),
            }
        )
    markets.sort(key=lambda r: (-(r["score"] or 0), -(r["ev"] or -9)))
    face_lines: list[str] = []
    if play.get("action") == "no_bet":
        face_lines.append("No bet: " + "; ".join((play.get("no_bet_reasons") or ["filtro"])[:2]))
    r1 = src.get("score_reason_1") or play.get("score_reason_1")
    r2 = src.get("score_reason_2") or play.get("score_reason_2")
    if r1:
        face_lines.append(str(r1)[:120])
    if r2:
        face_lines.append(str(r2)[:140])
    face_lines.append(
        f"P {_pct(p_cons)} · mkt {_pct(play.get('p_market'))} · "
        f"edge {_signed_pct(play.get('edge_pp'))} · EV {_signed_pct(ev)}"
    )
    if odds is not None:
        face_lines.append(f"Quota {quota_txt} · Kelly {_pct(play.get('kelly_quarter'))}")
    s100 = play.get("score_100")
    if s100 is not None:
        face_lines.append(
            f"Score {float(s100):.0f}/100"
            + (f" · {play.get('score_band')}" if play.get("score_band") else "")
            + (
                f" · Priorità {float(play['priority_100']):.0f}"
                if play.get("priority_100") is not None
                else ""
            )
        )
    quadro = src.get("quadro") or {}
    if isinstance(quadro, dict) and quadro.get("consenso"):
        face_lines.append(f"Quadro: {quadro.get('consenso')}")
    return {
        "title": title,
        "subtitle": subtitle,
        "home": home,
        "away": away,
        "action": _action_label(play.get("action")),
        "pick": play.get("code") or play.get("pick") or "—",
        "pick_name": play.get("name") or play.get("pick_name") or "",
        "score": _int(play.get("score")),
        "unified": _int(play.get("score_unified")),
        "score_100": None if play.get("score_100") is None else float(play.get("score_100")),
        "band": play.get("score_band") or "",
        "metrics": metrics,
        "p1x2": p1x2,
        "mix": mix,
        "scores": scores[:4],
        "xg": {
            "mins": list(xg.get("minutes") or []),
            "home": [float(v) for v in (xg.get("cum_home") or [])],
            "away": [float(v) for v in (xg.get("cum_away") or [])],
        },
        "markets": markets[:4],
        "face_lines": face_lines[:8],
        "filename": _slug(title, src.get("date") or pred.get("date")),
    }


def _mix_rows(src: dict, play: dict) -> list[dict]:
    meta = src.get("meta_analysis") or play.get("meta_analysis") or {}
    rows = []
    for key, label in (("value", "Value"), ("kelly", "Kelly"), ("asian", "Asian"), ("workflow", "Workflow")):
        p = _as_unit(meta.get(key))
        if p is None:
            continue
        rows.append({"label": label, "p": p})
    if rows:
        return rows
    note = str(src.get("meta_note") or meta.get("note") or "")
    for key, label in (("value", "Value"), ("kelly", "Kelly"), ("asian", "Asian"), ("workflow", "Workflow")):
        m = re.search(rf"{key}\s+(\d+)\s*%", note, re.I)
        if m:
            rows.append({"label": label, "p": int(m.group(1)) / 100.0})
    return rows


def _markets(src: dict) -> list[dict]:
    if isinstance(src.get("all_markets"), list) and src["all_markets"]:
        return [m for m in src["all_markets"] if isinstance(m, dict)]
    grouped = src.get("grouped")
    if isinstance(grouped, dict):
        out = []
        for block in grouped.values():
            if isinstance(block, list):
                out.extend(m for m in block if isinstance(m, dict))
        if out:
            return out
    mk = src.get("markets")
    if isinstance(mk, list):
        return [m for m in mk if isinstance(m, dict)]
    return []


def _play_odds(play: dict) -> float | None:
    n = _num(play.get("quota_pick"))
    if n is not None:
        return n
    n = _num(play.get("odds"))
    if n is not None:
        return n
    odds = play.get("odds")
    code = play.get("code") or play.get("pick")
    if isinstance(odds, dict) and code:
        n = _num(odds.get(code))
        if n is not None:
            return n
    for m in play.get("markets") or []:
        if isinstance(m, dict) and m.get("code") == code:
            return _num(m.get("odds"))
    return None


def _play(src: dict) -> dict:
    p = src.get("play")
    return p if isinstance(p, dict) else src


def _pred(src: dict) -> dict:
    p = src.get("prediction")
    if isinstance(p, dict):
        return p
    return src if src.get("montecarlo") or src.get("sportly_sim") else {}


def _action_label(action: str | None) -> str:
    return {
        "gioca": "GIOCA",
        "no_bet": "NO BET",
        "invalido": "INVALIDO",
        "n/d": "N/D",
    }.get(str(action or ""), str(action or "—").upper())


def _num(v) -> float | None:
    if v is None or isinstance(v, bool) or v == "" or v == "—":
        return None
    try:
        n = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(n) or math.isinf(n):
        return None
    return n


def _int(v) -> int | None:
    n = _num(v)
    return None if n is None else int(round(n))


def _as_unit(v) -> float | None:
    n = _num(v)
    if n is None:
        return None
    return n / 100.0 if n > 1.0 else n


def _pct(v) -> str:
    n = _num(v)
    return "—" if n is None else f"{n:.0%}"


def _signed_pct(v) -> str:
    n = _num(v)
    return "—" if n is None else f"{n:+.0%}"


def _num_txt(v) -> str:
    n = _num(v)
    return "—" if n is None else f"{n:+.2f}"


def _odd(v) -> str:
    n = _num(v)
    return "—" if n is None else f"{n:.2f}"


def _odd_num(v) -> float | None:
    n = _num(v)
    return None if n is None else round(n, 2)


def _score_txt(v) -> str:
    n = _int(v)
    return "—" if n is None else str(n)


def _slug(*parts) -> str:
    text = "_".join(str(p or "") for p in parts)
    out = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return (out[:70] or "voto")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    n = _num(obj) if isinstance(obj, (float, int)) else None
    if isinstance(obj, float):
        return n
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)
