"""Accordo tra fonti del quadro → filtro no_bet e Δ voto.

Non tocca EV/Kelly/p_cons. Usa solo i lean già calcolati nel quadro.
"""

from __future__ import annotations

from typing import Any


# Fonti “forti” per 1X2 (peso doppio nel tally).
STRONG_1X2 = {
    "Modello ML",
    "Monte Carlo",
    "Book (devig)",
    "Understat",
    "Analisi dati",
    "ClubElo",
}

# Per O/U preferiamo segnali gol/xG.
STRONG_OU = {
    "Understat",
    "Analisi dati",
    "FBref",
    "FotMob",
    "Sportly-sim",
    "λ Poisson",
}


def _pick_side(pick: str | None) -> str | None:
    p = str(pick or "").strip().upper()
    if p in {"1", "X", "2"}:
        return p
    return None


def source_agreement(
    quadro: dict[str, Any] | None,
    *,
    play_code: str | None = None,
    play_group: str | None = None,
) -> dict[str, Any]:
    """Conta lean delle fonti sul pick; restituisce share e status."""
    sources = (quadro or {}).get("sources") or []
    code = str(play_code or "").strip().upper()
    group = str(play_group or "1x2").lower()
    is_1x2 = code in {"1", "X", "2"} or group in {"1x2", "dc", "dnb"}

    votes: list[tuple[str, str, float]] = []
    for s in sources:
        if s.get("mancante"):
            continue
        fonte = str(s.get("fonte") or "")
        if fonte in {"Validazione", "Meteo", "Steam Asian", "Tipster"}:
            # Tipster/steam restano informativi ma non nel quorum base
            if fonte != "Tipster":
                continue
        pick = _pick_side(s.get("pick"))
        if not pick:
            continue
        strong = STRONG_1X2 if is_1x2 else STRONG_OU
        w = 2.0 if fonte in strong else 1.0
        if fonte == "Tipster":
            w = 1.25
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
            "notes": ["nessuna fonte votante"],
            "delta_unified": 0.0,
            "block_no_bet": False,
        }

    # tally pesato
    tally: dict[str, float] = {"1": 0.0, "X": 0.0, "2": 0.0}
    for _, pick, w in votes:
        tally[pick] = tally.get(pick, 0.0) + w
    total_w = sum(tally.values()) or 1.0
    majority = max(tally, key=tally.get)
    maj_share = tally[majority] / total_w

    agree_w = 0.0
    agree_n = 0
    if code in {"1", "X", "2"}:
        agree_w = tally.get(code, 0.0) / total_w
        agree_n = sum(1 for _, p, _ in votes if p == code)
    else:
        # mercato non 1X2: usa share della maggioranza come proxy di coerenza
        agree_w = maj_share
        agree_n = sum(1 for _, p, _ in votes if p == majority)

    n_votes = len(votes)
    split = {k: round(v / total_w, 3) for k, v in tally.items() if v > 0}

    # Soglie: spezzato forte → no_bet; ampio accordo → bonus voto
    block = False
    status = "ok"
    delta = 0.0
    notes: list[str] = []

    if code in {"1", "X", "2"}:
        if n_votes >= 5 and agree_w < 0.40:
            block = True
            status = "spezzato"
            delta = -0.5
            notes.append(f"accordo fonti debole {agree_n}/{n_votes} ({agree_w:.0%}) sul pick {code}")
        elif n_votes >= 4 and agree_w < 0.35:
            block = True
            status = "spezzato"
            delta = -0.5
            notes.append(f"quadro spezzato sul pick {code}: {agree_w:.0%}")
        elif agree_w >= 0.72 and n_votes >= 5:
            status = "forte"
            delta = 0.5
            notes.append(f"ampio accordo {agree_n}/{n_votes} ({agree_w:.0%})")
        elif agree_w >= 0.58:
            status = "maggioranza"
            delta = 0.25
            notes.append(f"maggioranza {agree_n}/{n_votes} ({agree_w:.0%})")
        elif agree_w < 0.45 and n_votes >= 4:
            status = "debole"
            delta = -0.25
            notes.append(f"accordo basso {agree_w:.0%} — attenzione")
        else:
            notes.append(f"accordo {agree_n}/{n_votes} ({agree_w:.0%})")
    else:
        # O/U / altri: blocca solo se le fonti 1X2 sono totalmente spezzate (proxy instabilità)
        if n_votes >= 6 and maj_share < 0.42:
            status = "instabile"
            delta = -0.25
            notes.append(f"fonti 1X2 instabili (top {majority} {maj_share:.0%})")
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
        "notes": notes,
        "delta_unified": float(delta),
        "block_no_bet": block,
    }
