"""Apprendimento continuo da partite chiuse (settle → affina filtri/analisi).

Strategia (aggressive_learn=True in calibration.json):
- Solo righe trainable: live ricche (quota+EV+fattori+accordo) + backfill synthetic ricco.
- Storico live incompleto (~679 righe senza quota/fattori) → escluso dal fit.
- Live ricche replicate ×5 (×6 se ≥80 live) + backfill ×4/×2/×1 con recency (half-life 90 gg).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modules.advisor.learn_policy import (
    BIN_BLEND_MAX_AGGRESSIVE,
    BIN_BLEND_MAX_CONSERVATIVE,
    LIVE_1X2_MIN_AGGRESSIVE,
    LIVE_ROI_MIN,
    OPF_ADJ_CAP_AGGRESSIVE,
    OPF_ERR_SCALE_AGGRESSIVE,
    TRAINABLE_1X2_MIN,
    TRAINABLE_ROI_MIN,
    aggressive_enabled,
    is_live,
    replicate_for_fit,
    split_trainable,
    trainable_settled,
)

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "data" / "models"
REPORT_PATH = MODELS / "online_learn_report.json"


def _f(v: Any, default: float | None = None) -> float | None:
    try:
        if v is None:
            return default
        x = float(v)
        if x != x:
            return default
        return x
    except (TypeError, ValueError):
        return default


def _reliability_from_settled(rows: list[dict[str, Any]], *, n_bins: int = 8) -> list[dict]:
    import pandas as pd

    probs, hits = [], []
    for r in rows:
        p = _f(r.get("probability"))
        if p is None or p < 0.05 or p > 0.95:
            continue
        if r.get("hit") is None:
            continue
        probs.append(p)
        hits.append(float(int(r["hit"])))
    if len(probs) < n_bins * 5:
        return []
    df = pd.DataFrame({"p": probs, "hit": hits})
    try:
        df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    except ValueError:
        return []
    out = []
    for label, grp in df.groupby("bin", observed=True):
        pred = float(grp["p"].mean())
        actual = float(grp["hit"].mean())
        n = int(len(grp))
        raw = actual / pred if pred > 0.02 else 1.0
        w = min(1.0, n / 40.0)
        factor = 1.0 + w * (raw - 1.0)
        factor = min(1.30, max(0.70, factor))
        out.append(
            {
                "range": [float(label.left), float(label.right)],
                "predicted": round(pred, 4),
                "actual": round(actual, 4),
                "n": n,
                "factor": round(factor, 4),
            }
        )
    return out


def _recent_roi(
    rows: list[dict[str, Any]],
    *,
    last_n: int = 80,
    min_n: int = 20,
    live_only: bool = False,
) -> dict[str, Any]:
    pool = trainable_settled(rows)
    if live_only:
        pool = [r for r in pool if is_live(r)]
    chunk = pool[-last_n:]
    pnl = 0.0
    n = 0
    for r in chunk:
        q = _f(r.get("quota_pick"))
        if q is None or q < 1.01:
            continue
        n += 1
        pnl += (q - 1.0) if int(r.get("hit") or 0) == 1 else -1.0
    if n < min_n:
        return {"ok": False, "n": n, "live_only": live_only}
    return {"ok": True, "n": n, "roi": round(pnl / n, 4), "pnl": round(pnl, 2), "live_only": live_only}


def _clv_of_row(r: dict[str, Any]) -> float | None:
    from modules.advisor.staking import beat_close, clv_prob

    try:
        if r.get("clv") is not None:
            return float(r["clv"])
    except (TypeError, ValueError):
        pass
    q = _f(r.get("quota_pick"))
    qc = _f(r.get("quota_close"))
    if q and qc:
        return clv_prob(q, qc)
    return None


def _recent_clv(
    rows: list[dict[str, Any]],
    *,
    last_n: int = 80,
    min_n: int = 8,
    live_only: bool = False,
) -> dict[str, Any]:
    from modules.advisor.staking import beat_close

    pool = trainable_settled(rows)
    if live_only:
        pool = [r for r in pool if is_live(r)]
    chunk = pool[-last_n:]
    clvs: list[float] = []
    beats = 0
    n_beat = 0
    for r in chunk:
        clv = _clv_of_row(r)
        if clv is None:
            continue
        clvs.append(float(clv))
        if r.get("beat_close") is not None:
            n_beat += 1
            beats += int(r.get("beat_close") or 0)
        else:
            q, qc = _f(r.get("quota_pick")), _f(r.get("quota_close"))
            if q and qc:
                bc = beat_close(q, qc)
                if bc is not None:
                    n_beat += 1
                    beats += 1 if bc else 0
    if len(clvs) < min_n:
        return {"ok": False, "n": len(clvs), "live_only": live_only}
    return {
        "ok": True,
        "n": len(clvs),
        "mean_clv": round(sum(clvs) / len(clvs), 4),
        "beat_close_rate": round(beats / n_beat, 4) if n_beat else None,
        "live_only": live_only,
    }


def _online_p_factor(
    rows: list[dict[str, Any]],
    *,
    last_n: int = 100,
    aggressive: bool = False,
) -> dict[str, Any]:
    """Correzione soft p 1X2: solo righe trainable (live ricche + backfill)."""
    live, backfill = split_trainable(rows)
    trainable = trainable_settled(rows)

    def _chunk(pool: list[dict]) -> list[dict]:
        return [
            r
            for r in pool
            if _f(r.get("probability"))
            and str(r.get("pick") or "") in {"1", "X", "2"}
        ][-last_n:]

    live_chunk = _chunk(live)
    all_chunk = _chunk(trainable)

    if aggressive and len(live_chunk) >= LIVE_1X2_MIN_AGGRESSIVE:
        chunk, src = live_chunk, "live_rich"
        min_need = LIVE_1X2_MIN_AGGRESSIVE
    elif len(all_chunk) >= TRAINABLE_1X2_MIN:
        chunk, src = all_chunk, "trainable"
        min_need = TRAINABLE_1X2_MIN
    else:
        return {
            "ok": False,
            "n_live_rich": len(live_chunk),
            "n_trainable": len(all_chunk),
            "n_backfill": len(backfill),
            "factor": 1.0,
            "reason": f"servono >={LIVE_1X2_MIN_AGGRESSIVE} live ricche o >={TRAINABLE_1X2_MIN} trainable 1X2",
        }

    err = sum(float(r["probability"]) - float(int(r["hit"])) for r in chunk) / len(chunk)
    w = min(1.0, len(chunk) / (100.0 if aggressive else 160.0))
    scale = OPF_ERR_SCALE_AGGRESSIVE if aggressive else 0.50
    cap = OPF_ADJ_CAP_AGGRESSIVE if aggressive else 0.04
    adj = max(-cap, min(cap, err * scale * w))
    factor = min(1.06 if aggressive else 1.04, max(0.94 if aggressive else 0.96, 1.0 - adj))
    return {
        "ok": True,
        "n": len(chunk),
        "source": src,
        "min_need": min_need,
        "mean_p_minus_hit": round(err, 4),
        "factor": round(factor, 4),
        "aggressive": aggressive,
    }


def learn_from_settled(*, force: bool = False, aggressive: bool | None = None) -> dict[str, Any]:
    """Aggiorna calibrazione leggera + residual + pesi da SQLite history."""
    from modules.calibration.config import load_calibration, save_calibration
    from modules.data_update.history import load_history

    rows = sorted(load_history(), key=lambda r: str(r.get("date") or ""))
    clv_fill: dict[str, Any] | None = None
    try:
        from modules.data_update.history import enrich_clv_from_matches_csv

        clv_fill = enrich_clv_from_matches_csv()
        if clv_fill.get("updated"):
            rows = sorted(load_history(), key=lambda r: str(r.get("date") or ""))
    except Exception:
        pass

    settled = [r for r in rows if r.get("hit") is not None]
    trainable = trainable_settled(settled)
    live_rich, backfill_rich = split_trainable(settled)

    report: dict[str, Any] = {
        "ok": True,
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "n_settled": len(settled),
        "n_trainable": len(trainable),
        "n_live_rich": len(live_rich),
        "n_backfill_rich": len(backfill_rich),
        "n_skipped_old_incomplete": len(settled) - len(trainable),
        "clv_enriched": clv_fill.get("updated") if isinstance(clv_fill, dict) else 0,
        "steps": {},
    }
    if len(trainable) < 25 and not force:
        report["ok"] = False
        report["error"] = f"servono >=25 righe trainable (ora {len(trainable)})"
        MODELS.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    cal = load_calibration()
    agg = aggressive if aggressive is not None else aggressive_enabled(cal)
    cal["aggressive_learn"] = bool(agg)
    report["aggressive_learn"] = agg

    try:
        from modules.calibration.calibrate import ensure_oof_reliability_1x2

        cal = ensure_oof_reliability_1x2(cal)
    except Exception as exc:
        report["steps"]["ensure_oof"] = {"ok": False, "error": str(exc)}

    fit_rows, fit_meta = replicate_for_fit(rows)
    report["fit_policy"] = fit_meta

    # 1) Reliability bins — live ricche ×5 + backfill ×4 (no junk storico)
    bins = _reliability_from_settled(fit_rows)
    min_bin = int(cal.get("min_bin_samples", 30))
    blend_max = BIN_BLEND_MAX_AGGRESSIVE if agg else BIN_BLEND_MAX_CONSERVATIVE
    if bins and min(int(b.get("n") or 0) for b in bins) >= (min_bin if not agg else max(12, min_bin // 2)):
        oof = cal.get("reliability_1x2_oof") or cal.get("reliability_1x2") or []
        if oof and len(oof) == len(bins):
            blended = []
            for a, b in zip(oof, bins):
                w_new = min(blend_max, int(b.get("n") or 0) / (50.0 if agg else 80.0))
                blended.append(
                    {
                        **b,
                        "factor": round(
                            (1.0 - w_new) * float(a.get("factor", 1.0))
                            + w_new * float(b["factor"]),
                            4,
                        ),
                    }
                )
            cal["reliability_1x2"] = blended
        else:
            cal["reliability_1x2"] = bins
        cal["reliability_1x2_online"] = bins
        report["steps"]["reliability_bins"] = {
            "n_bins": len(bins),
            "updated": True,
            "blend_max": blend_max,
            **fit_meta,
        }
    else:
        cal["reliability_1x2_online"] = bins or []
        report["steps"]["reliability_bins"] = {
            "updated": False,
            "reason": "sample per bin insufficiente",
            "n_online": 0 if not bins else int(sum(int(b.get("n") or 0) for b in bins)),
            **fit_meta,
        }

    # 2) online_p_factor — priorità live
    opf = _online_p_factor(rows, aggressive=agg)
    cal["online_p_factor"] = float(opf.get("factor") or 1.0)
    report["steps"]["online_p_factor"] = opf

    # 3) min_ev — ROI + CLV su trainable (CLV converge prima del ROI)
    roi = _recent_roi(rows, live_only=True, last_n=60, min_n=LIVE_ROI_MIN)
    if not roi.get("ok"):
        roi = _recent_roi(rows, live_only=False, last_n=120, min_n=TRAINABLE_ROI_MIN)
    clv = _recent_clv(rows, live_only=True, last_n=60, min_n=max(6, LIVE_ROI_MIN - 2))
    if not clv.get("ok"):
        clv = _recent_clv(rows, live_only=False, last_n=120, min_n=TRAINABLE_ROI_MIN)
    report["steps"]["recent_roi"] = roi
    report["steps"]["recent_clv"] = clv
    base_ev = float(cal.get("min_ev_play") or 0.025)
    new_ev = base_ev
    if roi.get("ok"):
        r = float(roi["roi"])
        if r < -0.05:
            new_ev = min(0.05, base_ev + (0.012 if agg else 0.008))
        elif r < 0:
            new_ev = min(0.04, base_ev + (0.006 if agg else 0.004))
        elif r > 0.08:
            new_ev = max(0.016, base_ev - (0.006 if agg else 0.004))
    if clv.get("ok"):
        mc = float(clv["mean_clv"])
        if mc < -0.015:
            new_ev = min(0.05, new_ev + (0.008 if agg else 0.005))
        elif mc > 0.02:
            new_ev = max(0.016, new_ev - (0.005 if agg else 0.003))
        bcr = clv.get("beat_close_rate")
        if bcr is not None and bcr < 0.45:
            new_ev = min(0.05, new_ev + 0.004)
        elif bcr is not None and bcr > 0.55:
            new_ev = max(0.016, new_ev - 0.003)
    if roi.get("ok") or clv.get("ok"):
        smooth_old = 0.45 if agg else 0.70
        cal["min_ev_play"] = round(smooth_old * base_ev + (1.0 - smooth_old) * new_ev, 4)
        report["steps"]["min_ev_play"] = {
            "from": base_ev,
            "to": cal["min_ev_play"],
            "aggressive": agg,
            "used_clv": bool(clv.get("ok")),
        }

    cal["online_learn_at"] = report["fitted_at"]
    save_calibration(cal)

    # 4) Residual — fit pesato live (compat firme vecchie senza aggressive)
    try:
        import inspect

        from modules.advisor.residual_ev import fit_residual_ev

        kwargs: dict[str, Any] = {}
        if "aggressive" in inspect.signature(fit_residual_ev).parameters:
            kwargs["aggressive"] = agg
        fit = fit_residual_ev(**kwargs)
        report["steps"]["residual"] = {
            k: fit.get(k)
            for k in ("ok", "n", "n_live_rich", "n_backfill_rich", "rmse", "wf_rmse", "mode", "error")
            if k in fit
        }
    except Exception as exc:
        report["steps"]["residual"] = {"ok": False, "error": str(exc)}

    # 5) Pesi data_signal — preferisci live
    try:
        import inspect

        from modules.advisor.data_signal_weights import optimize_weights

        kwargs = {}
        if "aggressive" in inspect.signature(optimize_weights).parameters:
            kwargs["aggressive"] = agg
        w = optimize_weights(**kwargs)
        report["steps"]["data_signal_weights"] = {
            k: w.get(k)
            for k in ("ok", "n", "n_live_rich", "n_backfill_rich", "method", "error")
            if k in w
        }
        if w.get("metrics"):
            report["steps"]["data_signal_weights"]["metrics"] = w.get("metrics")
    except Exception as exc:
        report["steps"]["data_signal_weights"] = {"ok": False, "error": str(exc)}

    MODELS.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def load_learn_report() -> dict[str, Any] | None:
    if not REPORT_PATH.exists():
        return None
    try:
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
