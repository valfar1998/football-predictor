"""Scarica l'ultimo modello allenato su GitHub Actions (artefatto).

Preferisce il workflow «Ritreno settimanale», poi «Aggiorna dati e modello».
Scarica anche l'apprendimento online (residual, pesi, report) dall'ultimo job giornaliero.
Richiede GitHub CLI autenticato: ``gh auth login``.

Uso:
  python scripts/pull_cloud_model.py
  python scripts/pull_cloud_model.py --rebuild-calendar
  python main.py --pull-model
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.progress_report import emit

MODELS = ROOT / "data" / "models"
PROCESSED = ROOT / "data" / "processed"

# Workflow name → prefisso artefatto (upload-artifact name: prefix-${{ github.run_id }})
WORKFLOWS: tuple[tuple[str, str], ...] = (
    ("Ritreno settimanale", "weekly-model-"),
    ("Aggiorna dati e modello", "best-model-"),
)

DAILY_WORKFLOW = "Aggiorna dati e modello"
LEARN_ARTIFACT_PREFIX = "cloud-learn-"

MODEL_FILES = (
    "best_model.joblib",
    "market_models.joblib",
    "metrics.json",
    "market_metrics.json",
    "calibration.json",
    "conformal.json",
    "oof_predictions.joblib",
    "oof_market.joblib",
    "league_stat_profiles.json",
)

LEARN_FILES = (
    "calibration.json",
    "residual_ev.json",
    "data_signal_weights.json",
    "online_learn_report.json",
)


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
    )


def _gh_ok() -> str | None:
    try:
        proc = _run(["gh", "--version"], check=False)
    except FileNotFoundError:
        return (
            "GitHub CLI (gh) non trovato. Installa: https://cli.github.com/ "
            "poi esegui: gh auth login"
        )
    if proc.returncode != 0:
        return "gh non eseguibile. Controlla l'installazione."
    auth = _run(["gh", "auth", "status"], check=False)
    if auth.returncode != 0:
        return "gh non autenticato. Esegui: gh auth login"
    return None


def _repo() -> str:
    proc = _run(["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"], check=False)
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip()
    # fallback da remote
    remote = _run(["git", "remote", "get-url", "origin"], check=False)
    url = (remote.stdout or "").strip()
    if "github.com" in url:
        # git@github.com:user/repo.git  or https://github.com/user/repo.git
        part = url.split("github.com", 1)[-1].lstrip("/:").removesuffix(".git")
        if "/" in part:
            return part
    raise SystemExit("Impossibile risolvere owner/repo (gh repo view / git remote).")


def _list_success_runs(workflow: str, *, limit: int = 8) -> list[dict]:
    proc = _run(
        [
            "gh",
            "run",
            "list",
            "--workflow",
            workflow,
            "--status",
            "success",
            "--limit",
            str(limit),
            "--json",
            "databaseId,createdAt,displayTitle,headBranch,conclusion,url",
        ],
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        print(f"skip workflow «{workflow}»: {err or 'errore gh'}", flush=True)
        return []
    try:
        rows = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return []
    return rows if isinstance(rows, list) else []


def _artifact_names(run_id: int) -> list[str]:
    proc = _run(
        ["gh", "run", "view", str(run_id), "--json", "artifacts"],
        check=False,
    )
    if proc.returncode != 0:
        # fallback API
        proc = _run(
            [
                "gh",
                "api",
                f"repos/{{owner}}/{{repo}}/actions/runs/{run_id}/artifacts",
                "--jq",
                ".artifacts[].name",
            ],
            check=False,
        )
        if proc.returncode != 0:
            return []
        return [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return []
    arts = data.get("artifacts") if isinstance(data, dict) else None
    if not isinstance(arts, list):
        return []
    names: list[str] = []
    for a in arts:
        if isinstance(a, dict) and a.get("name"):
            names.append(str(a["name"]))
        elif isinstance(a, str):
            names.append(a)
    return names


def _pick_run() -> tuple[dict, str, str]:
    """Ritorna (run, workflow_name, artifact_name)."""
    for workflow, prefix in WORKFLOWS:
        for run in _list_success_runs(workflow):
            run_id = int(run["databaseId"])
            names = _artifact_names(run_id)
            match = next((n for n in names if n.startswith(prefix)), None)
            if match is None and names:
                # artefatto singolo senza prefisso atteso
                match = next(
                    (n for n in names if "model" in n.lower() or "weekly" in n.lower()),
                    names[0],
                )
            if match:
                return run, workflow, match
    raise SystemExit(
        "Nessun artefatto modello trovato su run riusciti.\n"
        "Controlla Actions → «Ritreno settimanale» / «Aggiorna dati e modello» "
        "e che lo step Upload artefatti sia andato a buon fine."
    )


def _pick_learn_run(*, prefer_run_id: int | None = None) -> tuple[dict, str] | None:
    """Ritorna (run, artifact_name) per cloud-learn, o None."""
    if prefer_run_id is not None:
        names = _artifact_names(prefer_run_id)
        match = next((n for n in names if n.startswith(LEARN_ARTIFACT_PREFIX)), None)
        if match:
            return {"databaseId": prefer_run_id}, match

    for run in _list_success_runs(DAILY_WORKFLOW):
        run_id = int(run["databaseId"])
        names = _artifact_names(run_id)
        match = next((n for n in names if n.startswith(LEARN_ARTIFACT_PREFIX)), None)
        if match:
            return run, match
    return None


def _collect_files(src_root: Path, *, names: tuple[str, ...]) -> dict[str, Path]:
    """Mappa nome file → path trovato nell'estratto."""
    found: dict[str, Path] = {}
    for path in src_root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name in names:
            # preferisci path sotto data/
            prev = found.get(name)
            if prev is None or "data" in path.parts:
                found[name] = path
    return found


def _download_artifact(run_id: int, artifact: str, tmp_path: Path) -> None:
    dl = _run(
        [
            "gh",
            "run",
            "download",
            str(run_id),
            "-n",
            artifact,
            "-D",
            str(tmp_path),
        ],
        check=False,
    )
    if dl.returncode != 0:
        raise SystemExit(f"Download fallito:\n{(dl.stderr or dl.stdout or '').strip()}")


def pull_cloud_model(*, rebuild_calendar: bool = False, on_progress=None) -> dict:
    def p(frac: float, msg: str) -> None:
        emit(on_progress, frac, msg)

    p(0.02, "Controllo GitHub CLI (gh auth)…")
    err = _gh_ok()
    if err:
        raise SystemExit(err)

    p(0.08, "Risolvo il repo GitHub…")
    repo = _repo()
    p(0.15, "Cerco l'ultimo modello su Actions…")
    run, workflow, artifact = _pick_run()
    run_id = int(run["databaseId"])
    print(f"repo: {repo}", flush=True)
    print(f"workflow: {workflow}", flush=True)
    print(f"run: {run_id} · {run.get('createdAt')} · {run.get('url')}", flush=True)
    print(f"artefatto: {artifact}", flush=True)
    p(0.28, f"Scarico artefatto modello ({artifact})…")

    MODELS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    installed: list[str] = []
    with tempfile.TemporaryDirectory(prefix="fp-model-") as tmp:
        tmp_path = Path(tmp)
        _download_artifact(run_id, artifact, tmp_path)
        p(0.52, "Artefatto modello scaricato · installo i file…")

        found = _collect_files(tmp_path, names=MODEL_FILES + ("features.csv",))
        if "best_model.joblib" not in found:
            listing = "\n".join(
                str(fp.relative_to(tmp_path)) for fp in tmp_path.rglob("*") if fp.is_file()
            )
            raise SystemExit(
                "Nell'artefatto manca best_model.joblib.\n"
                f"Contenuto:\n{listing or '(vuoto)'}"
            )

        n_files = max(1, len(found))
        for i, (name, src) in enumerate(found.items(), start=1):
            if name == "features.csv":
                dest = PROCESSED / "features.csv"
            else:
                dest = MODELS / name
            shutil.copy2(src, dest)
            installed.append(str(dest.relative_to(ROOT)))
            print(f"ok {dest.relative_to(ROOT)}", flush=True)
            p(0.52 + 0.10 * (i / n_files), f"Installato {name} ({i}/{n_files})")

    p(0.66, "Cerco apprendimento cloud (calibrazione/pesi)…")
    learn_installed: list[str] = []
    learn_pick = _pick_learn_run(prefer_run_id=run_id)
    if learn_pick is None:
        print(
            "avviso: nessun artefatto cloud-learn trovato (calibrazione/residual/pesi restano locali o dal modello).",
            flush=True,
        )
        p(0.88, "Nessun artefatto learn su Actions · uso i pesi già in locale")
    else:
        learn_run, learn_artifact = learn_pick
        learn_run_id = int(learn_run["databaseId"])
        same_run = learn_run_id == run_id
        print(
            f"apprendimento: run {learn_run_id}"
            + ("" if same_run else " (job giornaliero più recente)")
            + f" · {learn_artifact}",
            flush=True,
        )
        p(0.72, f"Scarico apprendimento ({learn_artifact})…")
        with tempfile.TemporaryDirectory(prefix="fp-learn-") as tmp:
            learn_tmp = Path(tmp)
            _download_artifact(learn_run_id, learn_artifact, learn_tmp)
            p(0.88, "Apprendimento scaricato · installo i file…")
            learn_found = _collect_files(learn_tmp, names=LEARN_FILES)
            n_learn = max(1, len(learn_found))
            for i, (name, src) in enumerate(learn_found.items(), start=1):
                dest = MODELS / name
                shutil.copy2(src, dest)
                learn_installed.append(str(dest.relative_to(ROOT)))
                print(f"ok learn {dest.relative_to(ROOT)}", flush=True)
                p(0.88 + 0.08 * (i / n_learn), f"Learn {name} ({i}/{n_learn})")
            missing = [n for n in LEARN_FILES if n not in learn_found]
            if missing:
                print(f"avviso: nell'artefatto learn mancano: {', '.join(missing)}", flush=True)

    info: dict = {
        "ok": True,
        "repo": repo,
        "workflow": workflow,
        "run_id": run_id,
        "run_url": run.get("url"),
        "artifact": artifact,
        "created_at": run.get("createdAt"),
        "installed": installed,
        "learn_installed": learn_installed,
        "has_features": (PROCESSED / "features.csv").is_file(),
        "has_model": (MODELS / "best_model.joblib").is_file(),
        "has_learn": bool(learn_installed),
    }

    if not info["has_features"]:
        print(
            "avviso: manca features.csv (serve a MatchPredictor). "
            "Aggiorna gli artefatti GHA oppure lancia un download+feature locale.",
            flush=True,
        )

    if rebuild_calendar:
        if not info["has_model"] or not info["has_features"]:
            raise SystemExit("rebuild calendario: servono best_model.joblib e features.csv")
        sys.path.insert(0, str(ROOT))
        from modules.data_update.upcoming import build_upcoming

        p(0.96, "Ricostruisco il calendario con il modello scaricato…")
        print("ricostruisco calendario con il modello scaricato…", flush=True)
        rows = build_upcoming(reuse_predictions=False)
        info["n_upcoming"] = len(rows)
        print(f"calendario: {len(rows)} partite", flush=True)
    else:
        print(
            "Modello installato. Per usarlo sul calendario: "
            "«Solo quote e calendario» (o --odds-update / --pull-model --rebuild-calendar).",
            flush=True,
        )
    if learn_installed:
        print(
            f"Apprendimento cloud: {len(learn_installed)} file "
            "(calibrazione/residual/pesi/report).",
            flush=True,
        )
    p(1.0, "Completato")

    return info


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Scarica modello da GitHub Actions")
    parser.add_argument(
        "--rebuild-calendar",
        action="store_true",
        help="dopo il download ricostruisce upcoming (Monte Carlo pieno, lento)",
    )
    args = parser.parse_args()
    info = pull_cloud_model(rebuild_calendar=args.rebuild_calendar)
    print(json.dumps({k: v for k, v in info.items() if k != "installed"}, indent=2, default=str))
    if info.get("installed"):
        print("modello:", ", ".join(Path(p).name for p in info["installed"]))
    if info.get("learn_installed"):
        print("learn:", ", ".join(Path(p).name for p in info["learn_installed"]))


if __name__ == "__main__":
    main()
