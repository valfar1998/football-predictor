"""Policy apprendimento: solo righe ricche + backfill synthetic; esclude storico live incompleto."""

from __future__ import annotations

from typing import Any

# Repliche nel fit (bins / residual / pesi) — live e backfill pesano uguale, vecchio junk escluso
LIVE_RICH_REPLICATE = 5
BACKFILL_RICH_REPLICATE = 4
# Replicate live ×6 quando abbastanza live ricche (residual fit più reattivo)
LIVE_RICH_REPLICATE_BOOST = 6
LIVE_RICH_BOOST_MIN = 80

LIVE_1X2_MIN_AGGRESSIVE = 30
TRAINABLE_1X2_MIN = 60
LIVE_ROI_MIN = 8
TRAINABLE_ROI_MIN = 15

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


def replicate_for_fit(
    rows: list[dict[str, Any]],
    *,
    live_replicate: int | None = None,
    backfill_replicate: int = BACKFILL_RICH_REPLICATE,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Live ricche ×5 (×6 se ≥80 live) + backfill synthetic ricco ×4."""
    settled = [r for r in rows if r.get("hit") is not None]
    live, backfill = split_trainable(settled)
    if live_replicate is None:
        live_replicate = (
            LIVE_RICH_REPLICATE_BOOST if len(live) >= LIVE_RICH_BOOST_MIN else LIVE_RICH_REPLICATE
        )
    out: list[dict] = []
    for r in live:
        out.extend([r] * max(1, live_replicate))
    for r in backfill:
        out.extend([r] * max(1, backfill_replicate))
    skipped_old = len(settled) - len(live) - len(backfill)
    meta = {
        "n_settled_total": len(settled),
        "n_trainable_live": len(live),
        "n_trainable_backfill": len(backfill),
        "n_skipped_old_incomplete": skipped_old,
        "fit_rows": len(out),
        "live_replicate": live_replicate,
        "backfill_replicate": backfill_replicate,
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
