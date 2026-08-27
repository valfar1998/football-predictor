"""λ cartellini / corner: FD match rates → FBref match logs → season/crosses → proxy λ."""

from __future__ import annotations

from typing import Any


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
    """Ritorna λ cartellini e corner (somma delle due squadre) con priorità fonti."""
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
            poss_h = _p90(fb_home, "poss") if fb_home and fb_home.get("poss") is not None else None
            poss_a = _p90(fb_away, "poss") if fb_away and fb_away.get("poss") is not None else None
            # poss is already %, not p90 — fix
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

    return {
        "lambda_cards": round(lam_cards, 3),
        "lambda_corners": round(lam_corners, 3),
        "cards_source": cards_src,
        "corners_source": corners_src,
    }
