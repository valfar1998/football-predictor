"""Config calibrazione e soglie tarate su storico."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAL_PATH = ROOT / "data" / "models" / "calibration.json"

DEFAULTS = {
    "temperature": 1.0,
    "min_ev_play": 0.04,
    "min_ev_strong_value": 0.06,
    "min_prob_1x2_value": 0.35,
    "min_bin_samples": 30,
    "low_sample_max_score": 6,
    "reliability_1x2": [],
    "reliability_ou25": [],
    "backtest_summary": {},
}


def load_calibration() -> dict:
    if not CAL_PATH.exists():
        return dict(DEFAULTS)
    data = json.loads(CAL_PATH.read_text(encoding="utf-8"))
    out = dict(DEFAULTS)
    out.update(data)
    return out


def save_calibration(data: dict) -> Path:
    CAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    CAL_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return CAL_PATH


def prob_bin_factor(cal: dict, prob: float, *, market: str = "1x2") -> tuple[float, int]:
    """Fattore correttivo e dimensione campione del bin di calibrazione."""
    key = "reliability_1x2" if market == "1x2" else "reliability_ou25"
    bins = cal.get(key) or []
    for b in bins:
        lo, hi = b["range"]
        if lo <= prob < hi:
            return float(b.get("factor", 1.0)), int(b.get("n", 0))
    return 1.0, 0
