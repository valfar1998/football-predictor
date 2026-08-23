"""Progresso batch per UI Streamlit e terminale (percentuale + messaggio)."""

from __future__ import annotations

import re
import sys
import time
from typing import Any, Callable

# calendario 501/1994 · value 501/1994 · feature 15000/63220
_STEP_RE = re.compile(r"(?:calendario|value|feature)\s+(\d+)/(\d+)", re.I)

ProgressFn = Callable[[float, str], None]


def emit(on_progress: ProgressFn | None, frac: float, msg: str = "") -> None:
    """Aggiorna callback UI; se assente stampa su stdout (utile da CLI)."""
    frac = max(0.0, min(1.0, float(frac)))
    msg = str(msg or "").strip()
    if on_progress is not None:
        try:
            on_progress(frac, msg)
            return
        except Exception:
            pass
    pct = int(round(frac * 100))
    print(f"[{pct:3d}%] {msg}".rstrip(), flush=True)


class StreamlitProgress:
    """Barra + testo live in Streamlit. Usabile come callback ``on_progress``."""

    def __init__(self, title: str = "In corso…") -> None:
        import streamlit as st

        self._st = st
        self._title = title
        self.bar = st.progress(0, text=title)
        self.caption = st.empty()
        self.log_box = st.empty()
        self._lines: list[str] = []
        self._t0 = time.monotonic()
        self._frac = 0.0
        self.caption.caption(f"0% — {title}")

    def __call__(self, frac: float, msg: str = "") -> None:
        frac = max(self._frac, max(0.0, min(1.0, float(frac))))  # monotono
        self._frac = frac
        pct = int(round(frac * 100))
        elapsed = time.monotonic() - self._t0
        label = f"{pct}% — {msg}" if msg else f"{pct}% — {self._title}"
        try:
            self.bar.progress(frac, text=label[:120])
        except TypeError:
            self.bar.progress(frac)
        self.caption.caption(f"{label} · {elapsed:.0f}s")
        if msg:
            self._lines.append(f"{pct}% {msg}")
            self._lines = self._lines[-8:]
            self.log_box.code("\n".join(self._lines), language=None)
        # anche su terminale Streamlit / console
        print(f"[{pct:3d}%] {msg}".rstrip(), flush=True)

    def heartbeat(self, msg: str, *, bump: float = 0.012) -> None:
        """Avanza lentamente verso 95% mentre un passo lungo non ha sotto-progresso."""
        nxt = min(0.95, self._frac + bump)
        self(nxt, msg)

    def done(self, msg: str = "Completato") -> None:
        self(1.0, msg)


def run_cli_with_progress(
    *flags: str,
    progress: StreamlitProgress | None = None,
    python_exe: str | None = None,
    main_path: str | None = None,
) -> Any:
    """Esegue ``main.py`` streaming stdout e aggiorna la barra a ogni riga."""
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    exe = python_exe or sys.executable
    main = main_path or str(root / "main.py")
    import os

    prog = progress or StreamlitProgress("CLI: " + " ".join(flags))
    prog(0.02, "Avvio " + " ".join(flags) + "…")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = env.get("PYTHONIOENCODING") or "utf-8"
    proc = subprocess.Popen(
        [exe, "-u", main, *flags],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    def _step_frac(line: str) -> float | None:
        m = _STEP_RE.search(line)
        if not m:
            return None
        done, total = int(m.group(1)), int(m.group(2))
        if total <= 0:
            return None
        low = line.lower()
        if "feature" in low:
            return 0.72 + 0.18 * (done / total)
        return 0.95 + 0.04 * (done / total)

    soft = 0.03
    keywords = (
        ("download", 0.08),
        ("asian", 0.12),
        ("pinnacle", 0.18),
        ("betfair", 0.22),
        ("fbref", 0.35),
        ("understat", 0.42),
        ("fotmob", 0.48),
        ("statsbomb", 0.55),
        ("fd rates", 0.58),
        ("rolling", 0.65),
        ("cluster", 0.72),
        ("market", 0.78),
        ("conformal", 0.82),
        ("upcoming", 0.88),
        ("calendario", 0.90),
        ("storico", 0.93),
        ("modello", 0.95),
    )
    buf: list[str] = []
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        buf.append(line)
        soft = min(0.94, soft + 0.008)
        low = line.lower()
        for key, tgt in keywords:
            if key in low:
                soft = max(soft, float(tgt))
        step = _step_frac(line)
        if step is not None:
            soft = max(soft, step)
        prog(soft, line.strip()[:100])
    code = proc.wait()
    out = "\n".join(buf[-200:])
    if code == 0:
        prog.done("OK")
    else:
        prog(soft, f"Exit code {code}")
    return subprocess.CompletedProcess(
        args=[exe, main, *flags],
        returncode=code,
        stdout=out,
        stderr=out if code != 0 else "",
    )
