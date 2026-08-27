"""Rate cartellini/corner per squadra da football-data.co.uk (HY/AY/HC/AC)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FD_MAIN = ROOT / "data" / "raw" / "fd" / "main"
RATES_CACHE = PROCESSED / "fd_side_rates.csv"


def _norm(name: str) -> str:
    from modules.data_update.cups import _norm_key

    return _norm_key(name or "")


def _read_fd_csv(path: Path) -> pd.DataFrame:
    import io

    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip")
    df.columns = [str(c).replace("\ufeff", "").replace("ï»¿", "").strip() for c in df.columns]
    return df


def build_fd_side_rates(
    *,
    min_date: str = "2022-07-01",
    last_n: int = 12,
    on_progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Aggrega HY/AY/HC/AC dalle CSV FD → p90-like (media per partita recente)."""
    from modules.progress_report import emit

    frames: list[pd.DataFrame] = []
    csvs = sorted(FD_MAIN.glob("*/*.csv"))
    n_csv = max(1, len(csvs))
    for i, csv in enumerate(csvs):
        emit(on_progress, 0.05 + 0.75 * (i / n_csv), f"CSV {csv.parent.name}/{csv.name}")
        if csv.name.lower().startswith(("sa", "notes")):
            continue
        try:
            df = _read_fd_csv(csv)
        except Exception:
            continue
        need = {"HomeTeam", "AwayTeam", "Date"}
        if not need.issubset(set(df.columns)):
            continue
        if "HY" not in df.columns and "HC" not in df.columns:
            continue
        part = pd.DataFrame(
            {
                "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
                "home": df["HomeTeam"].astype(str).str.strip(),
                "away": df["AwayTeam"].astype(str).str.strip(),
                "hy": pd.to_numeric(df["HY"], errors="coerce") if "HY" in df.columns else pd.NA,
                "ay": pd.to_numeric(df["AY"], errors="coerce") if "AY" in df.columns else pd.NA,
                "hr": pd.to_numeric(df["HR"], errors="coerce") if "HR" in df.columns else pd.NA,
                "ar": pd.to_numeric(df["AR"], errors="coerce") if "AR" in df.columns else pd.NA,
                "hc": pd.to_numeric(df["HC"], errors="coerce") if "HC" in df.columns else pd.NA,
                "ac": pd.to_numeric(df["AC"], errors="coerce") if "AC" in df.columns else pd.NA,
            }
        )
        frames.append(part)
    emit(on_progress, 0.82, f"Aggregazione {len(frames)} file…")
    if not frames:
        emit(on_progress, 1.0, "Nessuna CSV con HY/HC")
        return {"ok": False, "n_teams": 0, "error": "nessuna CSV con HY/HC"}

    all_m = pd.concat(frames, ignore_index=True)
    all_m = all_m.dropna(subset=["date", "home", "away"])
    all_m = all_m[all_m["date"] >= pd.Timestamp(min_date)]
    rows: list[dict[str, Any]] = []
    for _, r in all_m.iterrows():
        rows.append(
            {
                "date": r["date"],
                "team": r["home"],
                "cards_y": r["hy"],
                "cards_r": r["hr"],
                "corners": r["hc"],
            }
        )
        rows.append(
            {
                "date": r["date"],
                "team": r["away"],
                "cards_y": r["ay"],
                "cards_r": r["ar"],
                "corners": r["ac"],
            }
        )
    emit(on_progress, 0.90, "Medie ultime partite per squadra…")
    long = pd.DataFrame(rows).dropna(subset=["team"])
    long["team_norm"] = long["team"].map(_norm)
    long = long.sort_values("date")
    long["rank"] = long.groupby("team_norm").cumcount(ascending=False)
    recent = long[long["rank"] < last_n]
    agg = (
        recent.groupby(["team_norm", "team"], as_index=False)
        .agg(
            n=("cards_y", "count"),
            cards_y_avg=("cards_y", "mean"),
            cards_r_avg=("cards_r", "mean"),
            corners_avg=("corners", "mean"),
        )
        .sort_values("n", ascending=False)
    )
    agg = agg.drop_duplicates("team_norm", keep="first")
    agg["fetched_at"] = pd.Timestamp.utcnow().isoformat()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    agg.to_csv(RATES_CACHE, index=False)
    emit(on_progress, 1.0, f"OK · {len(agg)} squadre")
    return {"ok": True, "n_teams": int(len(agg)), "path": str(RATES_CACHE), "n_rows": int(len(recent))}


def load_fd_side_index() -> dict[str, dict[str, Any]]:
    if not RATES_CACHE.exists():
        return {}
    df = pd.read_csv(RATES_CACHE)
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        key = _norm(str(row.get("team_norm") or row.get("team") or ""))
        if not key:
            continue
        out[key] = {
            "team": row.get("team"),
            "n": int(row["n"]) if pd.notna(row.get("n")) else 0,
            "cards_y_avg": float(row["cards_y_avg"]) if pd.notna(row.get("cards_y_avg")) else None,
            "cards_r_avg": float(row["cards_r_avg"]) if pd.notna(row.get("cards_r_avg")) else None,
            "corners_avg": float(row["corners_avg"]) if pd.notna(row.get("corners_avg")) else None,
            "source": "fd",
        }
    return out


def lookup_fd_side(team: str, idx: dict[str, dict[str, Any]] | None = None) -> dict[str, Any] | None:
    idx = idx if idx is not None else load_fd_side_index()
    key = _norm(team)
    if key in idx:
        return idx[key]
    # fuzzy light
    for k, v in idx.items():
        if key and (key in k or k in key) and min(len(k), len(key)) >= 5:
            return v
    return None
