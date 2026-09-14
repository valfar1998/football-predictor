"""λ cartellini / corner / tiri: FD match rates → FBref → proxy λ."""

from __future__ import annotations

from typing import Any

import numpy as np

CORNER_OU_LINES = (6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5)
SHOT_OU_LINES = (19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5)
CARD_OU_LINES = (2.5, 3.5, 4.5, 5.5)


def _rate(row: dict[str, Any] | None, key: str, *, n90_key: str = "n90") -> float | None:
    if not row:
        return None
    try:
        tot = float(row.get(key) or 0)
        n90 = float(row.get(n90_key) or 0)
        if n90 >= 3 and tot >= 0:
            return tot / n90
    except (TypeError, ValueError):
        return None
    return None


def _p90(row: dict[str, Any] | None, key: str) -> float | None:
    if not row or row.get(key) is None:
        return None
    try:
        v = float(row[key])
        return v if v == v and v > 0 else None
    except (TypeError, ValueError):
        return None


def _avg(row: dict[str, Any] | None, *keys: str) -> float | None:
    if not row:
        return None
    for k in keys:
        if row.get(k) is None:
            continue
        try:
            v = float(row[k])
            if v == v and v >= 0:
                return v
        except (TypeError, ValueError):
            continue
    return None


def match_side_extras(
    *,
    lambda_home: float,
    lambda_away: float,
    fb_home: dict[str, Any] | None = None,
    fb_away: dict[str, Any] | None = None,
    fd_home: dict[str, Any] | None = None,
    fd_away: dict[str, Any] | None = None,
    fb_match_home: dict[str, Any] | None = None,
    fb_match_away: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ritorna λ cartellini, corner e tiri (somma squadre) con priorità fonti."""
    # --- Cards ---
    cards_src = "proxy_lambda"
    hy = _avg(fd_home, "cards_y_avg")
    ay = _avg(fd_away, "cards_y_avg")
    hr = _avg(fd_home, "cards_r_avg") or 0.0
    ar = _avg(fd_away, "cards_r_avg") or 0.0
    if hy is not None and ay is not None and (fd_home or {}).get("n", 0) >= 4 and (fd_away or {}).get("n", 0) >= 4:
        lam_cards = hy + ay + 0.35 * (hr + ar)
        cards_src = "fd"
    else:
        hy = _avg(fb_match_home, "cards_y_avg") or _avg(fb_home, "match_cards_y_avg")
        ay = _avg(fb_match_away, "cards_y_avg") or _avg(fb_away, "match_cards_y_avg")
        hr = _avg(fb_match_home, "cards_r_avg") or _avg(fb_home, "match_cards_r_avg") or 0.0
        ar = _avg(fb_match_away, "cards_r_avg") or _avg(fb_away, "match_cards_r_avg") or 0.0
        if hy is not None and ay is not None:
            lam_cards = hy + ay + 0.35 * (hr + ar)
            cards_src = "fbref_match"
        else:
            hy = _rate(fb_home, "cards_y")
            ay = _rate(fb_away, "cards_y")
            hr = _rate(fb_home, "cards_r") or 0.0
            ar = _rate(fb_away, "cards_r") or 0.0
            if hy is not None and ay is not None:
                lam_cards = hy + ay + 0.35 * (hr + ar)
                cards_src = "fbref_season"
            else:
                lam_cards = 3.6 + 0.22 * (float(lambda_home) + float(lambda_away))
    lam_cards = max(2.0, min(7.0, float(lam_cards)))

    # --- Corners ---
    corners_src = "proxy_lambda"
    hc = _avg(fd_home, "corners_avg")
    ac = _avg(fd_away, "corners_avg")
    if hc is not None and ac is not None and (fd_home or {}).get("n", 0) >= 4 and (fd_away or {}).get("n", 0) >= 4:
        lam_corners = hc + ac
        corners_src = "fd"
    else:
        hc = _avg(fb_match_home, "corners_avg") or _avg(fb_home, "match_corners_avg")
        ac = _avg(fb_match_away, "corners_avg") or _avg(fb_away, "match_corners_avg")
        if hc is not None and ac is not None:
            lam_corners = hc + ac
            corners_src = "fbref_match"
        else:
            cx_h = _p90(fb_home, "crosses_p90")
            cx_a = _p90(fb_away, "crosses_p90")
            cc_h = _p90(fb_home, "crosses_conc_p90")
            cc_a = _p90(fb_away, "crosses_conc_p90")
            try:
                poss_h = float(fb_home["poss"]) if fb_home and fb_home.get("poss") is not None else None
                poss_a = float(fb_away["poss"]) if fb_away and fb_away.get("poss") is not None else None
            except (TypeError, ValueError, KeyError):
                poss_h = poss_a = None
            if cx_h is not None and cx_a is not None:
                lam_corners = 0.52 * (cx_h + cx_a)
                if cc_h is not None and cc_a is not None:
                    lam_corners = 0.65 * lam_corners + 0.35 * 0.45 * (cc_h + cc_a)
                corners_src = "fbref_crosses"
            else:
                base = 9.2 + 1.15 * (float(lambda_home) + float(lambda_away) - 2.4)
                if poss_h is not None and poss_a is not None:
                    base += 0.03 * ((poss_h + poss_a) / 2.0 - 50.0)
                lam_corners = base
                corners_src = "proxy_lambda"
    lam_corners = max(6.5, min(14.5, float(lam_corners)))

    # --- Shots (totali partita) ---
    shots_src = "proxy_lambda"
    sh = _avg(fb_home, "shots", "sh", "shots_avg")
    sa = _avg(fb_away, "shots", "sh", "shots_avg")
    sh_p90 = _rate(fb_home, "shots") or _p90(fb_home, "shots_p90")
    sa_p90 = _rate(fb_away, "shots") or _p90(fb_away, "shots_p90")
    if sh_p90 is not None and sa_p90 is not None:
        lam_shots = sh_p90 + sa_p90
        shots_src = "fbref_p90"
    elif sh is not None and sa is not None:
        n_h = float((fb_home or {}).get("n90") or (fb_home or {}).get("mp") or 0)
        n_a = float((fb_away or {}).get("n90") or (fb_away or {}).get("mp") or 0)
        if n_h >= 3 and n_a >= 3 and sh > 40 and sa > 40:
            lam_shots = sh / n_h + sa / n_a
            shots_src = "fbref_season"
        else:
            lam_shots = sh + sa
            shots_src = "fbref_avg"
    else:
        lam_shots = 22.0 + 3.2 * (float(lambda_home) + float(lambda_away) - 2.4)
        shots_src = "proxy_lambda"
    # Blend soft con corner (tattica: più corner ↔ più tiri periferici)
    lam_shots = 0.85 * float(lam_shots) + 0.15 * (float(lam_corners) * 1.85)
    lam_shots = max(16.0, min(34.0, float(lam_shots)))

    return {
        "lambda_cards": round(lam_cards, 3),
        "lambda_corners": round(lam_corners, 3),
        "lambda_shots": round(lam_shots, 3),
        "cards_source": cards_src,
        "corners_source": corners_src,
        "shots_source": shots_src,
    }


def poisson_ou_probs(lam: float, lines: tuple[float, ...], *, n: int = 4000, seed: int = 42) -> dict[str, float]:
    """O/U da Poisson (stesso schema MC) → keys ``*_over_{line}`` / ``*_under_`` da passare al chiamante."""
    if lam <= 0:
        return {}
    rng = np.random.default_rng(seed)
    draws = rng.poisson(lam=float(lam), size=int(n))
    out: dict[str, float] = {}
    for line in lines:
        over = float((draws > line).mean())
        out[f"over_{line}"] = round(over, 4)
        out[f"under_{line}"] = round(1.0 - over, 4)
    return out


def side_ou_from_extras(extras: dict[str, Any] | None, *, n: int = 4000, seed: int = 42) -> dict[str, Any]:
    """Prob O/U cards/corners/shots + meta source da un dict extras."""
    if not extras:
        return {}
    out: dict[str, Any] = {}
    lam_c = float(extras.get("lambda_cards") or 0)
    lam_k = float(extras.get("lambda_corners") or 0)
    lam_s = float(extras.get("lambda_shots") or 0)
    if extras.get("cards_source"):
        out["cards_source"] = extras["cards_source"]
    if extras.get("corners_source"):
        out["corners_source"] = extras["corners_source"]
    if extras.get("shots_source"):
        out["shots_source"] = extras["shots_source"]
    if lam_c > 0.5:
        for k, v in poisson_ou_probs(lam_c, CARD_OU_LINES, n=n, seed=seed).items():
            out[f"cards_{k}"] = v
        out["lambda_cards"] = round(lam_c, 3)
    if lam_k > 1.0:
        for k, v in poisson_ou_probs(lam_k, CORNER_OU_LINES, n=n, seed=seed + 1).items():
            out[f"corners_{k}"] = v
        out["lambda_corners"] = round(lam_k, 3)
    if lam_s > 5.0:
        for k, v in poisson_ou_probs(lam_s, SHOT_OU_LINES, n=n, seed=seed + 2).items():
            out[f"shots_{k}"] = v
        out["lambda_shots"] = round(lam_s, 3)
    return out


def ensure_side_markets_on_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    """Se il MC riusato non ha corners/shots, calcola λ + O/U e li merge in montecarlo.

    Usato da Solo quote / reuse: altrimenti le quote Kambi corner non trovano p MC.
    """
    if not isinstance(prediction, dict):
        return prediction
    mc = prediction.get("montecarlo")
    if not isinstance(mc, dict):
        return prediction
    need_corners = not any(str(k).startswith("corners_over_") for k in mc)
    need_shots = not any(str(k).startswith("shots_over_") for k in mc)
    # Completa linee mancanti anche se MC ha già un sottoinsieme (Solo quote riusa MC vecchio)
    missing_corner_lines = [ln for ln in CORNER_OU_LINES if f"corners_over_{ln}" not in mc]
    missing_shot_lines = [ln for ln in SHOT_OU_LINES if f"shots_over_{ln}" not in mc]
    if not need_corners and not need_shots and not missing_corner_lines and not missing_shot_lines:
        return prediction

    xg = prediction.get("expected_goals") if isinstance(prediction.get("expected_goals"), dict) else {}
    try:
        lh = float(xg.get("home") or mc.get("lambda_home") or 1.2)
        la = float(xg.get("away") or mc.get("lambda_away") or 1.0)
    except (TypeError, ValueError):
        lh, la = 1.2, 1.0

    fb = prediction.get("fbref_context") if isinstance(prediction.get("fbref_context"), dict) else {}
    fb_h = fb.get("home") if isinstance(fb.get("home"), dict) else None
    fb_a = fb.get("away") if isinstance(fb.get("away"), dict) else None

    fd_h = fd_a = None
    fb_mh = fb_ma = None
    try:
        from modules.data_update.side_rates import load_fd_side_index, lookup_fd_side

        idx = load_fd_side_index()
        # home/away names from match string
        match = str(prediction.get("match") or "")
        home = away = ""
        if " vs " in match:
            home, away = match.split(" vs ", 1)
        elif " – " in match:
            home, away = match.split(" – ", 1)
        home, away = home.strip(), away.strip()
        if home and away and idx:
            fd_h = lookup_fd_side(home, idx)
            fd_a = lookup_fd_side(away, idx)
    except Exception:
        pass
    try:
        from modules.data_update.fbref_context import load_fbref_match_side_index, lookup_fbref_match_side

        midx = load_fbref_match_side_index()
        match = str(prediction.get("match") or "")
        home = away = ""
        if " vs " in match:
            home, away = match.split(" vs ", 1)
        home, away = home.strip(), away.strip()
        if home and away and midx:
            fb_mh = lookup_fbref_match_side(home, midx)
            fb_ma = lookup_fbref_match_side(away, midx)
    except Exception:
        pass

    extras = match_side_extras(
        lambda_home=lh,
        lambda_away=la,
        fb_home=fb_h,
        fb_away=fb_a,
        fd_home=fd_h,
        fd_away=fd_a,
        fb_match_home=fb_mh,
        fb_match_away=fb_ma,
    )
    side = side_ou_from_extras(extras)
    if not side:
        return prediction
    mc2 = dict(mc)
    for k, v in side.items():
        if str(k).startswith("corners_") and (need_corners or k not in mc2):
            mc2[k] = v
        elif str(k).startswith("shots_") and (need_shots or k not in mc2):
            mc2[k] = v
        elif str(k).startswith("cards_") and k not in mc2:
            mc2[k] = v
        elif k in {"corners_source", "shots_source", "cards_source", "lambda_corners", "lambda_shots", "lambda_cards"}:
            mc2.setdefault(k, v)
    out = dict(prediction)
    out["montecarlo"] = mc2
    return out
