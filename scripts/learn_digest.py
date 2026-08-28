"""Scrive un report leggibile sull'apprendimento online.

- docs/APPRENDIMENTO.md          → guida fissa + ultimo aggiornamento
- data/models/learn_digest.md    → snapshot tecnico dell'ultimo learn
- data/models/learn_digest_meta.json → per decidere il digest settimanale

Uso:
  python scripts/learn_digest.py
  python scripts/learn_digest.py --force-weekly
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS = ROOT / "data" / "models"
DOCS = ROOT / "docs"
REPORT_JSON = MODELS / "online_learn_report.json"
DIGEST_MD = MODELS / "learn_digest.md"
DIGEST_META = MODELS / "learn_digest_meta.json"
GUIDE_MD = DOCS / "APPRENDIMENTO.md"
WEEKLY_DAYS = 7


GUIDE_BODY = """# Cosa impara Football Predictor (apprendimento online)

Ogni mattina il job cloud fa due cose diverse:

1. **Riallena il modello base** sullo storico partite (CSV delle leghe).
2. **Apprendimento online** dagli esiti delle giocate archiviate (quando ci sono abbastanza risultati chiusi).

## Cosa aggiorna l’apprendimento online

| Cosa | Effetto pratico |
|---|---|
| **Calibrazione probabilità** | Se il modello è troppo fiducioso/timido, corregge le probabilità. |
| **Soglia EV minima (`min_ev_play`)** | Se il ROI recente è negativo alza il filtro; se va bene lo allenta un po’. |
| **Fattore probabilità online** | Piccolo moltiplicatore sulle p usate nei filtri. |
| **Residual EV** | Impara quanto l’edge stimato è ottimistico/pessimistico. |
| **Pesi data-signal** | Ribilancia forma, xG, casa/trasferta, ecc. in base a cosa ha funzionato. |

Usa solo partite **chiuse** con dati ricchi (quota + EV + fattori). Lo storico incompleto viene escluso.

## Quando vedi miglioramenti

- **Ogni giorno**: settle esiti + eventuale aggiornamento dei filtri (se ci sono abbastanza sample).
- **Ogni ~7 giorni**: arriva su Telegram un **riassunto** di cosa è cambiato (ROI, CLV, soglie, residual).

Se il report dice “servono ≥25 righe trainable”, l’algoritmo sta ancora raccogliendo esiti: il modello base gira, ma i filtri online non si muovono ancora.

## File tecnici

- `data/models/online_learn_report.json` — ultimo fit (macchina)
- `data/models/learn_digest.md` — stesso contenuto in italiano
- `data/models/calibration.json` — soglie attive
- `data/models/residual_ev.json` / `data_signal_weights.json` — modelli secondari
"""


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _pct(x: Any) -> str:
    try:
        return f"{float(x):+.1%}"
    except (TypeError, ValueError):
        return "n/d"


def _num(x: Any, nd: int = 3) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "n/d"


def build_digest_text(report: dict[str, Any] | None, *, cal: dict[str, Any] | None = None) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Digest apprendimento — {now}",
        "",
    ]
    if not report:
        lines += [
            "Nessun `online_learn_report.json` ancora.",
            "Il job giornaliero non ha ancora prodotto un report di apprendimento online.",
            "",
        ]
        return "\n".join(lines)

    ok = bool(report.get("ok"))
    fitted = report.get("fitted_at") or "n/d"
    lines += [
        f"- Esito fit: **{'OK' if ok else 'non aggiornato'}**",
        f"- Momento: `{fitted}`",
        f"- Partite chiuse totali: **{report.get('n_settled', 0)}**",
        f"- Usabili per imparare (ricche): **{report.get('n_trainable', 0)}** "
        f"(live {report.get('n_live_rich', 0)} + backfill {report.get('n_backfill_rich', 0)})",
        f"- Escluse (storico incompleto): **{report.get('n_skipped_old_incomplete', 0)}**",
        "",
    ]
    if report.get("error"):
        lines += [f"**Motivo:** {report['error']}", ""]

    steps = report.get("steps") or {}
    roi = steps.get("recent_roi") or {}
    clv = steps.get("recent_clv") or {}
    ev = steps.get("min_ev_play") or {}
    bins = steps.get("reliability_bins") or {}
    opf = steps.get("online_p_factor") or {}
    residual = steps.get("residual") or {}
    weights = steps.get("data_signal_weights") or {}

    lines.append("## In cosa sta migliorando / correggendo")
    lines.append("")

    if roi.get("ok"):
        lines.append(
            f"- **ROI recente** (ultime {roi.get('n')} giocate): {_pct(roi.get('roi'))} "
            f"(PnL {roi.get('pnl', 'n/d')} u)"
        )
    else:
        lines.append("- **ROI recente:** non ancora calcolabile (pochi esiti live).")

    if clv.get("ok"):
        lines.append(
            f"- **CLV medio:** {_num(clv.get('mean_clv'), 4)} · "
            f"beat close {_pct(clv.get('beat_close_rate'))}"
        )
    else:
        lines.append("- **CLV:** non ancora calcolabile.")

    if ev:
        lines.append(
            f"- **Soglia EV minima:** da `{ev.get('from')}` a `{ev.get('to')}` "
            f"({'più severa' if float(ev.get('to') or 0) > float(ev.get('from') or 0) else 'più permissiva'}"
            f"{', usa anche CLV' if ev.get('used_clv') else ''})"
        )
    elif cal and cal.get("min_ev_play") is not None:
        lines.append(f"- **Soglia EV minima attuale:** `{cal.get('min_ev_play')}`")

    if bins.get("updated"):
        lines.append(
            f"- **Calibrazione probabilità:** aggiornata ({bins.get('n_bins')} bin, "
            f"blend max {bins.get('blend_max')})."
        )
    else:
        reason = bins.get("reason") or "non aggiornata"
        lines.append(f"- **Calibrazione probabilità:** {reason}.")

    if opf.get("ok"):
        lines.append(
            f"- **Fattore p online:** `{opf.get('factor')}` "
            f"(errore medio p−hit {_num(opf.get('mean_p_minus_hit'), 4)})"
        )

    if residual.get("ok"):
        lines.append(
            f"- **Residual EV:** ok su {residual.get('n')} sample "
            f"(RMSE {_num(residual.get('rmse'))}"
            f"{', WF ' + _num(residual.get('wf_rmse')) if residual.get('wf_rmse') is not None else ''})"
        )
    elif residual:
        lines.append(f"- **Residual EV:** non aggiornato ({residual.get('error') or 'n/d'}).")

    if weights.get("ok"):
        m = weights.get("metrics") or {}
        lines.append(
            f"- **Pesi data-signal:** aggiornati (hit rate {_pct(m.get('hit_rate'))}, "
            f"ROI {_pct(m.get('roi'))}, metodo `{weights.get('method')}`)."
        )
    elif weights:
        lines.append(f"- **Pesi data-signal:** non aggiornati ({weights.get('error') or 'n/d'}).")

    lines += [
        "",
        "## Come leggerlo in pratica",
        "",
        "- ROI/CLV **negativi** → filtri più stretti (meno giocate dubbie).",
        "- ROI/CLV **positivi** → filtri un po’ più aperti.",
        "- Residual/pesi → correggono edge e analisi dati, non riscrivono il modello ML base.",
        "",
    ]
    return "\n".join(lines)


def _should_send_weekly(meta: dict[str, Any] | None, *, force: bool) -> bool:
    if force:
        return True
    if os.getenv("LEARN_DIGEST_WEEKLY", "").lower() in {"1", "true", "yes", "on"}:
        return True
    if not meta or not meta.get("last_weekly_at"):
        return True
    try:
        last = datetime.fromisoformat(str(meta["last_weekly_at"]).replace("Z", "+00:00"))
    except ValueError:
        return True
    return datetime.now(timezone.utc) - last >= timedelta(days=WEEKLY_DAYS)


def write_learn_digest(*, force_weekly: bool = False) -> dict[str, Any]:
    MODELS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    report = _load_json(REPORT_JSON)
    cal = None
    try:
        from modules.calibration.config import load_calibration

        cal = load_calibration()
    except Exception:
        cal = _load_json(MODELS / "calibration.json")

    digest = build_digest_text(report, cal=cal)
    DIGEST_MD.write_text(digest, encoding="utf-8")

    guide = GUIDE_BODY + "\n---\n\n## Ultimo aggiornamento automatico\n\n" + digest
    GUIDE_MD.write_text(guide, encoding="utf-8")

    meta = _load_json(DIGEST_META) or {}
    weekly = _should_send_weekly(meta, force=force_weekly)
    out = {
        "ok": True,
        "digest_path": str(DIGEST_MD),
        "guide_path": str(GUIDE_MD),
        "weekly": weekly,
        "learn_ok": bool(report and report.get("ok")),
        "n_trainable": (report or {}).get("n_trainable"),
        "fitted_at": (report or {}).get("fitted_at"),
    }

    if weekly:
        meta["last_weekly_at"] = datetime.now(timezone.utc).isoformat()
        meta["last_weekly_preview"] = digest[:500]
        telegram_ok = _send_telegram_weekly(digest)
        out["telegram_weekly"] = telegram_ok
    DIGEST_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _send_telegram_weekly(digest: str) -> bool:
    import urllib.parse
    import urllib.request

    token = "".join(os.environ.get("TELEGRAM_BOT_TOKEN", "").split())
    chat = "".join(os.environ.get("TELEGRAM_CHAT_ID", "").split())
    if not token or not chat:
        print("skip telegram digest: manca TELEGRAM_BOT_TOKEN/CHAT_ID", flush=True)
        return False
    # Telegram max ~4096; tieni il riassunto corto
    body = f"{BRAND} — digest apprendimento\n\n" + digest
    if len(body) > 3500:
        body = body[:3490] + "\n…"
    data = urllib.parse.urlencode(
        {"chat_id": chat, "text": body, "disable_web_page_preview": "true"}
    ).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=data)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        ok = bool(payload.get("ok"))
        print("telegram digest settimanale inviato" if ok else payload, flush=True)
        return ok
    except Exception as exc:
        print(f"skip telegram digest: {exc}", flush=True)
        return False


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Digest apprendimento online")
    parser.add_argument("--force-weekly", action="store_true", help="forza messaggio Telegram settimanale")
    args = parser.parse_args()
    info = write_learn_digest(force_weekly=args.force_weekly)
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
