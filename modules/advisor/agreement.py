"""Accordo tra fonti del quadro → filtro no_bet e Δ voto.

Pesi da `pro_scores.SOURCE_WEIGHTS` (MC/ML/devig alti; tipster/meteo bassi).
Soglie dinamiche per cluster lega, mercato e larghezza intervallo probabilità.
Non tocca EV/Kelly/p_cons.
"""

from __future__ import annotations

from typing import Any

from modules.advisor.pro_scores import source_weight

# soglia agree_w sotto la quale scatta no_bet (più alta = più severo)
CLUSTER_BLOCK = {
    "big5_eng": 0.38,
    "big5_esp": 0.38,
    "big5_ita": 0.40,
    "big5_ger": 0.38,
    "big5_fra": 0.40,
    "serie_b_like": 0.45,
    "latam": 0.42,
    "mls": 0.40,
    "cups_euro": 0.42,
    "global": 0.40,
}

# fonti che non votano 1X2 (solo annotazione / override)
SKIP_VOTE = {"Validazione", "FotMob details"}


def _pick_side(pick: str | None) -> str | None:
    p = str(pick or "").strip().upper()
    if p in {"1", "X", "2"}:
        return p
    return None


def _block_threshold(league: str | None, group: str, interval_width: float | None) -> float:
    try:
        from modules.model_training.league_clusters import cluster_for

        cid = cluster_for(league)
    except Exception:
        cid = "global"
    thr = float(CLUSTER_BLOCK.get(cid, CLUSTER_BLOCK["global"]))
    g = str(group or "1x2").lower()
    if g in {"ou", "ah", "goal", "btts"}:
        thr += 0.03
    if interval_width is not None and float(interval_width) >= 0.12:
        thr += 0.05
    return min(0.55, thr)


def source_agreement(
    quadro: dict[str, Any] | None,
    *,
    play_code: str | None = None,
    play_group: str | None = None,
    league: str | None = None,
    interval_width: float | None = None,
) -> dict[str, Any]:
    sources = (quadro or {}).get("sources") or []
    code = str(play_code or "").strip().upper()
    group = str(play_group or "1x2").lower()
    is_ou = group in {"ou", "btts", "goal"}

    votes: list[tuple[str, str, float]] = []
    for s in sources:
        if s.get("mancante"):
            continue
        fonte = str(s.get("fonte") or "")
        if fonte in SKIP_VOTE:
            continue
        # Meteo vota solo se ha pick 1X2 (di solito no); steam sì
        pick = _pick_side(s.get("pick"))
        if not pick:
            continue
        w = float(s.get("peso") or source_weight(fonte, ou=is_ou))
        if w <= 0:
            continue
        # Tipster/meteo: tetto basso anche se annotati altrimenti
        if fonte == "Tipster":
            w = min(w, 0.02)
        if fonte == "Meteo":
            w = min(w, 0.03)
        votes.append((fonte, pick, w))

    if not votes:
        return {
            "ready": False,
            "n_votes": 0,
            "agree_n": 0,
            "agree_share": None,
            "agree_w": None,
            "status": "n/d",
            "lean_majority": None,
            "split": {},
            "weights_used": {},
            "notes": ["nessuna fonte votante"],
            "delta_unified": 0.0,
            "block_no_bet": False,
            "threshold": None,
        }

    tally: dict[str, float] = {"1": 0.0, "X": 0.0, "2": 0.0}
    w_by_fonte: dict[str, float] = {}
    for fonte, pick, w in votes:
        tally[pick] = tally.get(pick, 0.0) + w
        w_by_fonte[fonte] = w
    total_w = sum(tally.values()) or 1.0
    majority = max(tally, key=tally.get)
    maj_share = tally[majority] / total_w

    if code in {"1", "X", "2"}:
        agree_w = tally.get(code, 0.0) / total_w
        agree_n = sum(1 for _, p, _ in votes if p == code)
    else:
        agree_w = maj_share
        agree_n = sum(1 for _, p, _ in votes if p == majority)

    n_votes = len(votes)
    split = {k: round(v / total_w, 3) for k, v in tally.items() if v > 0}
    thr = _block_threshold(league, group, interval_width)

    block = False
    status = "ok"
    delta = 0.0
    notes: list[str] = []

    if code in {"1", "X", "2"}:
        if n_votes >= 4 and agree_w < thr:
            block = True
            status = "spezzato"
            delta = -0.5
            notes.append(f"accordo pesato {agree_w:.0%} < soglia {thr:.0%} (n={n_votes})")
        elif agree_w >= 0.72 and n_votes >= 5:
            status = "forte"
            delta = 0.5
            notes.append(f"ampio accordo pesato {agree_n}/{n_votes} ({agree_w:.0%})")
        elif agree_w >= 0.58:
            status = "maggioranza"
            delta = 0.25
            notes.append(f"maggioranza pesata {agree_n}/{n_votes} ({agree_w:.0%})")
        elif agree_w < thr + 0.05 and n_votes >= 4:
            status = "debole"
            delta = -0.25
            notes.append(f"accordo basso {agree_w:.0%}")
        else:
            notes.append(f"accordo pesato {agree_n}/{n_votes} ({agree_w:.0%}) · thr {thr:.0%}")
    else:
        if n_votes >= 6 and maj_share < thr:
            status = "instabile"
            delta = -0.25
            notes.append(f"fonti instabili top {majority} {maj_share:.0%}")
        else:
            notes.append(f"maggioranza fonti {majority} {maj_share:.0%}")

    return {
        "ready": True,
        "n_votes": n_votes,
        "agree_n": agree_n,
        "agree_share": round(float(agree_w if code in {"1", "X", "2"} else maj_share), 3),
        "agree_w": round(float(agree_w), 3),
        "status": status,
        "lean_majority": majority,
        "split": split,
        "weights_used": {k: round(v, 3) for k, v in w_by_fonte.items()},
        "notes": notes,
        "delta_unified": float(delta),
        "block_no_bet": block,
        "threshold": round(thr, 3),
    }
