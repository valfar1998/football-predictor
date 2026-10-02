"""Policy apprendimento: solo righe ricche + backfill synthetic; esclude storico live incompleto."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

# Repliche nel fit (bins / residual / pesi) — live e backfill pesano uguale, vecchio junk escluso
LIVE_RICH_REPLICATE = 5
BACKFILL_RICH_REPLICATE = 4
# Replicate live ×6 quando abbastanza live ricche (residual fit più reattivo)
LIVE_RICH_REPLICATE_BOOST = 6
LIVE_RICH_BOOST_MIN = 80

# Phasing-out backfill synthetic: fit solo live quando ≥150 archivi ricchi pre-match
LIVE_RICH_PHASEOUT_MIN = 150

# Backfill decay: meno peso synthetic quando cresce il live
BACKFILL_TIER_LOW = 40  # sotto: backfill ×4 (bootstrap)
BACKFILL_TIER_MID = 80  # 40–79: ×2 · ≥80: ×1

# Recency: partite recenti contano di più nel fit online
RECENCY_HALF_LIFE_DAYS = 90.0
RECENCY_FLOOR = 0.25
BACKFILL_RECENCY_SCALE = 0.85

LIVE_1X2_MIN_AGGRESSIVE = 30
TRAINABLE_1X2_MIN = 60
LIVE_ROI_MIN = 8
TRAINABLE_ROI_MIN = 15
# ROI paper: solo freeze Telegram GIOCA; peso ∝ Kelly del messaggio.
# Se score_locked=1, score_unified/quota/kelly restano quelli della notifica.
ROI_MIN_SCORE = 8

# Obiettivi campione paper ROI (roadmap): sblocchi progressivi, non gate codice.
ROI_SAMPLE_LEVELS: list[dict[str, Any]] = [
    {"n": 15, "name": "Campione iniziale", "desc": "Segnale per filtri e min_ev adattivo"},
    {"n": 30, "name": "Validazione economica", "desc": "ROI @ quote credibile (roadmap)"},
    {"n": 40, "name": "ROI stabile", "desc": "Campione robusto n≥40 settled voto≥8"},
]


def roi_sample_progress(n: int) -> dict[str, Any]:
    """Stato livello campione ROI: corrente, prossimo obiettivo, residuo."""
    try:
        n = max(0, int(n or 0))
    except (TypeError, ValueError):
        n = 0
    levels = ROI_SAMPLE_LEVELS
    current = None
    nxt = levels[0]
    for lv in levels:
        if n >= int(lv["n"]):
            current = lv
            nxt = None
        else:
            nxt = lv
            break
    if nxt is None and levels:
        # Oltre l'ultimo livello: mantieni l'ultimo come raggiunto
        current = levels[-1]
    target_n = int(nxt["n"]) if nxt else (int(levels[-1]["n"]) if levels else 40)
    prev_n = int(current["n"]) if current and nxt else 0
    span = max(1, target_n - prev_n)
    done = min(1.0, max(0.0, (n - prev_n) / span)) if nxt else 1.0
    return {
        "n": n,
        "current": current,
        "next": nxt,
        "target_n": target_n,
        "remaining": max(0, target_n - n) if nxt else 0,
        "progress": round(done, 3),
        "label": (
            f"Livello sbloccato: {current['name']}"
            if current and not nxt
            else (
                f"Verso {nxt['name']} ({n}/{target_n})"
                if nxt
                else f"Campione n={n}"
            )
        ),
    }


def score_unified_of(rec: dict[str, Any]) -> int | None:
    for key in ("score_unified", "score"):
        v = rec.get(key)
        if v is None or v == "":
            continue
        try:
            s = int(v)
            if 1 <= s <= 10:
                return s
        except (TypeError, ValueError):
            continue
    return None


def meets_roi_score(rec: dict[str, Any], *, min_score: int = ROI_MIN_SCORE) -> bool:
    s = score_unified_of(rec)
    return s is not None and s >= min_score


def _has_roi_quota(rec: dict[str, Any]) -> bool:
    q = rec.get("quota_pick")
    if isinstance(q, bool) or q is None:
        return False
    try:
        return float(q) >= 1.01
    except (TypeError, ValueError):
        return False


def is_roi_eligible(rec: dict[str, Any], *, min_score: int = ROI_MIN_SCORE) -> bool:
    """Campione paper ROI: solo giocata Telegram congelata (quota+Kelly del messaggio).

    Requisiti:
    - settled (hit noto)
    - score_locked=1 (alert inviato)
    - alert_kind/action = gioca (esclude watch/no-bet)
    - quota_pick reale
    - voto unificato ≥ min_score (di solito già garantito dall'alert)
    """
    if rec.get("hit") is None:
        return False
    if int(rec.get("score_locked") or 0) != 1:
        return False
    kind = str(rec.get("alert_kind") or "").strip().lower()
    action = str(rec.get("action") or "").strip().lower()
    if kind == "watch" or action in {"no_bet", "invalido", "n/d"}:
        return False
    if kind != "gioca" and action != "gioca":
        return False
    if not meets_roi_score(rec, min_score=min_score):
        return False
    return _has_roi_quota(rec)


def roi_settled(rows: list[dict[str, Any]], *, min_score: int = ROI_MIN_SCORE) -> list[dict[str, Any]]:
    return [r for r in rows if is_roi_eligible(r, min_score=min_score)]


BIN_BLEND_MAX_CONSERVATIVE = 0.40
BIN_BLEND_MAX_AGGRESSIVE = 0.72

OPF_ERR_SCALE_AGGRESSIVE = 0.78
OPF_ADJ_CAP_AGGRESSIVE = 0.07


def is_rich(rec: dict[str, Any]) -> bool:
    """Stesso gate roadmap: quota + EV + fattori + accordo."""
    q = rec.get("quota_pick")
    if isinstance(q, bool) or q is None:
        return False
    try:
        if float(q) < 1.01:
            return False
    except (TypeError, ValueError):
        return False
    return bool(
        (rec.get("ev_cons") is not None or rec.get("ev_sharp") is not None)
        and rec.get("data_factors")
        and rec.get("agree_share") is not None
    )


def is_live(rec: dict[str, Any]) -> bool:
    return not int(rec.get("synthetic_backfill") or 0)


def is_backfill(rec: dict[str, Any]) -> bool:
    return bool(int(rec.get("synthetic_backfill") or 0))


def is_trainable(rec: dict[str, Any]) -> bool:
    """Usabile per apprendimento: settled + ricco (live o backfill). Esclude ~679 live vecchie incomplete."""
    return rec.get("hit") is not None and is_rich(rec)


def trainable_settled(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if is_trainable(r)]


def split_trainable(rows: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    pool = trainable_settled(rows)
    live = [r for r in pool if is_live(r)]
    backfill = [r for r in pool if is_backfill(r)]
    return live, backfill


def _parse_row_date(rec: dict[str, Any]) -> datetime | None:
    raw = rec.get("date") or rec.get("settled_at") or rec.get("saved_at")
    if not raw:
        return None
    s = str(raw).strip()
    try:
        if "T" in s:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00")[:19])
        else:
            dt = datetime.strptime(s[:10], "%Y-%m-%d")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def recency_weight(
    rec: dict[str, Any],
    *,
    half_life_days: float = RECENCY_HALF_LIFE_DAYS,
    floor: float = RECENCY_FLOOR,
) -> float:
    """Peso 0.25–1.0: esiti recenti contano di più (exp decay, half-life ~90 gg)."""
    dt = _parse_row_date(rec)
    if dt is None:
        return 1.0
    now = datetime.now(timezone.utc)
    age_days = max(0.0, (now - dt.astimezone(timezone.utc)).total_seconds() / 86400.0)
    if age_days <= 0:
        return 1.0
    w = math.exp(-age_days * math.log(2) / max(half_life_days, 1.0))
    return max(floor, min(1.0, w))


def backfill_excluded_from_fit(n_live: int) -> bool:
    """True quando il live ricco basta: niente righe synthetic_backfill nel fit."""
    return int(n_live) >= LIVE_RICH_PHASEOUT_MIN


def backfill_replicate_for(n_live: int) -> int:
    """Riduce peso backfill synthetic man mano che crescono le live ricche."""
    if n_live >= BACKFILL_TIER_MID:
        return 1
    if n_live >= BACKFILL_TIER_LOW:
        return 2
    return BACKFILL_RICH_REPLICATE


def _effective_replicates(base: int, rec: dict[str, Any], *, live: bool) -> int:
    w = recency_weight(rec)
    if not live:
        w *= BACKFILL_RECENCY_SCALE
    return max(1, round(base * w))


def replicate_for_fit(
    rows: list[dict[str, Any]],
    *,
    live_replicate: int | None = None,
    backfill_replicate: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Live ricche ×5 (×6 se ≥80 live) + backfill ×4/×2/×1; recency sulle repliche."""
    settled = [r for r in rows if r.get("hit") is not None]
    live, backfill = split_trainable(settled)
    if live_replicate is None:
        live_replicate = (
            LIVE_RICH_REPLICATE_BOOST if len(live) >= LIVE_RICH_BOOST_MIN else LIVE_RICH_REPLICATE
        )
    if backfill_replicate is None:
        backfill_replicate = 0 if backfill_excluded_from_fit(len(live)) else backfill_replicate_for(len(live))
    out: list[dict] = []
    for r in live:
        n = _effective_replicates(live_replicate, r, live=True)
        out.extend([r] * n)
    if backfill_replicate > 0:
        for r in backfill:
            n = _effective_replicates(backfill_replicate, r, live=False)
            out.extend([r] * n)
    skipped_old = len(settled) - len(live) - len(backfill)
    meta = {
        "n_settled_total": len(settled),
        "n_trainable_live": len(live),
        "n_trainable_backfill": len(backfill),
        "n_skipped_old_incomplete": skipped_old,
        "fit_rows": len(out),
        "live_replicate": live_replicate,
        "backfill_replicate": backfill_replicate,
        "recency_half_life_days": RECENCY_HALF_LIFE_DAYS,
        "backfill_tier": (
            "excluded"
            if backfill_excluded_from_fit(len(live))
            else "minimal"
            if len(live) >= BACKFILL_TIER_MID
            else "mid"
            if len(live) >= BACKFILL_TIER_LOW
            else "bootstrap"
        ),
        "backfill_excluded": backfill_excluded_from_fit(len(live)),
        "live_rich_phaseout_min": LIVE_RICH_PHASEOUT_MIN,
    }
    return out, meta


def aggressive_enabled(cal: dict[str, Any] | None = None) -> bool:
    if cal is None:
        try:
            from modules.calibration.config import load_calibration

            cal = load_calibration()
        except Exception:
            return True
    return bool(cal.get("aggressive_learn", True))
