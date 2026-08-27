"""λ gol: clip xG, smoothing esponenziale, baseline lega."""

from __future__ import annotations

XG_CAP = 3.5
XG_FLOOR = 0.05
LAM_SMOOTH_ALPHA = 0.25
HOME_BASE = 1.35
AWAY_BASE = 1.15


def clip_xg(val: float | None, *, lo: float = XG_FLOOR, hi: float = XG_CAP) -> float | None:
    try:
        x = float(val)
    except (TypeError, ValueError):
        return None
    if x != x or x <= lo:
        return None
    return min(hi, max(lo, x))


def smooth_lambda(lam: float, baseline: float, *, alpha: float = LAM_SMOOTH_ALPHA) -> float:
    """EMA verso baseline lega (riduce outlier singola partita)."""
    a = min(0.45, max(0.05, float(alpha)))
    return (1.0 - a) * float(lam) + a * float(baseline)


def lambdas_from_features(
    row,
    *,
    ext_xg_home: tuple[float, float] | None = None,
    ext_xg_away: tuple[float, float] | None = None,
    ext_blend: float = 0.38,
    hist_blend: float = 0.62,
) -> tuple[float, float]:
    """λ attesi da feature riga + xG contesto (clip + blend)."""
    hxg = clip_xg(row.get("home_xg_avg")) or 1.2
    axg = clip_xg(row.get("away_xg_avg")) or 1.0
    hxga = clip_xg(row.get("home_xga_avg")) or 1.2
    axga = clip_xg(row.get("away_xga_avg")) or 1.0

    lam_h = max(0.35, hxg * 0.7 + (HOME_BASE - axga) * 0.15 + 0.25)
    lam_a = max(0.25, axg * 0.7 + (AWAY_BASE - hxga) * 0.15)

    if ext_xg_home and ext_xg_home[0]:
        xf = clip_xg(ext_xg_home[0])
        if xf:
            lam_h = hist_blend * lam_h + ext_blend * xf
    if ext_xg_away and ext_xg_away[0]:
        xf = clip_xg(ext_xg_away[0])
        if xf:
            lam_a = hist_blend * lam_a + ext_blend * xf

    lam_h = smooth_lambda(lam_h, HOME_BASE)
    lam_a = smooth_lambda(lam_a, AWAY_BASE)
    return max(0.25, lam_h), max(0.20, lam_a)
