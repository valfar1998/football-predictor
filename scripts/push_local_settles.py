"""Esporta i settle locali in un journal e (opzionale) commit+push per il learn cloud.

Il job Actions applica ``data/processed/local_settles.json`` prima di settle/learn,
così le chiusure manuali (es. freeze Telegram) entrano anche nello storico cloud.

Uso:
  python scripts/push_local_settles.py              # solo esporta
  python scripts/push_local_settles.py --commit     # esporta + git commit
  python scripts/push_local_settles.py --commit --push
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Esporta settle locali per apprendimento cloud")
    parser.add_argument(
        "--all-settled",
        action="store_true",
        help="esporta tutti gli settled (default: solo voto≥8 / freeze)",
    )
    parser.add_argument("--commit", action="store_true", help="git add + commit del journal")
    parser.add_argument("--push", action="store_true", help="git push dopo commit (implica --commit)")
    args = parser.parse_args()
    if args.push:
        args.commit = True

    from modules.data_update.history import SETTLE_JOURNAL, export_settle_journal

    info = export_settle_journal(roi_only=not args.all_settled)
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
    if not info.get("ok"):
        raise SystemExit(1)

    rel = str(SETTLE_JOURNAL.relative_to(ROOT))
    if not args.commit:
        print(
            f"Journal pronto: {rel}\n"
            "Per farlo usare al cloud: commit+push, oppure:\n"
            "  python scripts/push_local_settles.py --commit --push",
            flush=True,
        )
        return

    st = _run(["git", "status", "--porcelain", rel])
    if not (st.stdout or "").strip():
        # forza add anche se untracked
        _run(["git", "add", "--", rel])
        st2 = _run(["git", "status", "--porcelain", rel])
        if not (st2.stdout or "").strip():
            print("Nessuna modifica al journal da committare.", flush=True)
            return
    else:
        _run(["git", "add", "--", rel])

    n = int(info.get("n") or 0)
    msg = f"Sync local settles journal ({n} esiti) for cloud learn."
    commit = _run(["git", "commit", "-m", msg])
    if commit.returncode != 0:
        err = (commit.stderr or commit.stdout or "").strip()
        if "nothing to commit" in err.lower():
            print("Nessuna modifica da committare.", flush=True)
        else:
            raise SystemExit(err or "git commit fallito")
    else:
        print(f"commit ok: {msg}", flush=True)

    if args.push:
        push = _run(["git", "push"])
        if push.returncode != 0:
            raise SystemExit((push.stderr or push.stdout or "git push fallito").strip())
        print("push ok — al prossimo job cloud i settle entreranno nello storico.", flush=True)


if __name__ == "__main__":
    main()
