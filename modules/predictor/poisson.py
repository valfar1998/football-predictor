"""1X2 da Poisson indipendente + correzione Dixon-Coles (low-score)."""

from __future__ import annotations

from math import exp, factorial


def _tau(hg: int, ag: int, lam_h: float, lam_a: float, rho: float) -> float:
    if hg == 0 and ag == 0:
        return 1.0 - lam_h * lam_a * rho
    if hg == 0 and ag == 1:
        return 1.0 + lam_h * rho
    if hg == 1 and ag == 0:
        return 1.0 + lam_a * rho
    if hg == 1 and ag == 1:
        return 1.0 - rho
    return 1.0


def dixon_coles_1x2(
    lam_h: float,
    lam_a: float,
    *,
    rho: float = -0.08,
    max_goals: int = 8,
) -> tuple[float, float, float]:
    lam_h = max(0.15, min(4.5, float(lam_h)))
    lam_a = max(0.12, min(4.2, float(lam_a)))
    p_h = p_d = p_a = 0.0
    for hg in range(max_goals + 1):
        ph = exp(-lam_h) * lam_h**hg / factorial(hg)
        for ag in range(max_goals + 1):
            pa = exp(-lam_a) * lam_a**ag / factorial(ag)
            p = ph * pa * _tau(hg, ag, lam_h, lam_a, rho)
            if hg > ag:
                p_h += p
            elif hg == ag:
                p_d += p
            else:
                p_a += p
    tot = p_h + p_d + p_a
    if tot <= 0:
        return 0.45, 0.26, 0.29
    return p_h / tot, p_d / tot, p_a / tot


def blend_1x2(
    ml: tuple[float, float, float],
    dc: tuple[float, float, float],
    *,
    ml_weight: float = 0.62,
) -> tuple[float, float, float]:
    w = min(0.85, max(0.40, float(ml_weight)))
    p_h = w * ml[0] + (1 - w) * dc[0]
    p_d = w * ml[1] + (1 - w) * dc[1]
    p_a = w * ml[2] + (1 - w) * dc[2]
    s = p_h + p_d + p_a
    return p_h / s, p_d / s, p_a / s
