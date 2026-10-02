"""Merge cache Telegram (sent + freeze) senza perdere chiavi già presenti in locale."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fp-telegram-cache")
PROCESSED = ROOT / "data" / "processed"


def _cache_file(name: str) -> Path | None:
    for p in (CACHE_DIR / name, CACHE_DIR / "data" / "processed" / name):
        if p.is_file():
            return p
    return None


def _load_dict(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    # Freeze journal: merge (prima freeze vince)
    src_j = _cache_file("telegram_score_freeze.json")
    if src_j is not None:
        from modules.data_update.history import merge_freeze_journal_file, sync_freeze_state

        info = merge_freeze_journal_file(src_j)
        print(f"freeze merge: {info}", flush=True)
        try:
            sync = sync_freeze_state()
            print(f"freeze sync: n_journal={sync.get('n_journal')}", flush=True)
        except Exception as exc:
            print(f"freeze sync skip: {exc}", flush=True)
    else:
        print("freeze cache: miss", flush=True)

    # Settle journal locale→cloud (se presente in cache o già in repo)
    src_settle = _cache_file("local_settles.json")
    from modules.data_update.history import SETTLE_JOURNAL, apply_settle_journal, merge_settle_journal_file

    if src_settle is not None:
        info_s = merge_settle_journal_file(src_settle)
        print(f"settle journal merge: {info_s}", flush=True)
    elif SETTLE_JOURNAL.is_file():
        print(f"settle journal: using repo file ({SETTLE_JOURNAL.name})", flush=True)
    else:
        print("settle journal: miss", flush=True)
    try:
        applied = apply_settle_journal()
        print(f"settle journal apply: {applied}", flush=True)
    except Exception as exc:
        print(f"settle journal apply skip: {exc}", flush=True)

    # Sent ids: unione (mai perdere un alert già inviato)
    src_s = _cache_file("telegram_alerts_sent.json")
    dst_s = PROCESSED / "telegram_alerts_sent.json"
    incoming = _load_dict(src_s) if src_s else {}
    local = _load_dict(dst_s)
    merged = {**incoming, **local}
    if merged:
        dst_s.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"sent merge: n={len(merged)} (cache={len(incoming)} local={len(local)})", flush=True)
    else:
        print("sent cache: miss/empty", flush=True)

    # Spread Raro journal: merge (prima freeze vince)
    src_sp = _cache_file("telegram_spread_freeze.json")
    dst_sp = PROCESSED / "telegram_spread_freeze.json"
    if src_sp is not None or dst_sp.is_file():
        try:
            from modules.advisor.spread_paper import JOURNAL as SP_JOURNAL
            from modules.advisor.spread_paper import _load_journal, _save_journal

            def _spreads(path: Path | None) -> dict:
                if path is None or not path.is_file():
                    return {}
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return {}
                if isinstance(raw, dict) and isinstance(raw.get("spreads"), dict):
                    return {str(k): v for k, v in raw["spreads"].items() if isinstance(v, dict)}
                if isinstance(raw, dict):
                    return {
                        str(k): v
                        for k, v in raw.items()
                        if isinstance(v, dict) and k not in {"updated_at", "n", "spreads"}
                    }
                return {}

            incoming_sp = _spreads(src_sp)
            local_sp = _load_journal()
            merged_sp = dict(incoming_sp)
            for k, v in local_sp.items():
                if k not in merged_sp:
                    merged_sp[k] = v
            _save_journal(merged_sp)
            print(
                f"spread freeze merge: n={len(merged_sp)} (cache={len(incoming_sp)} local={len(local_sp)})",
                flush=True,
            )
        except Exception as exc:
            print(f"spread freeze merge skip: {exc}", flush=True)
    else:
        print("spread freeze: miss", flush=True)


if __name__ == "__main__":
    main()
