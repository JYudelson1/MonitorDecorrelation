"""Upload a run's complete ``run.log`` to its wandb run (the run's Files tab, as ``run.log``).

wandb's own console capture (the Logs tab / ``output.log``) only holds what was printed between
``wandb.init`` and ``wandb.finish``: it misses the startup banner (config, reward mode, init weights,
train-against vs held-out split) and the end (early-stop reason, plot paths, "finished OK", or the
traceback of a crash). ``scripts/queue_runs.sh`` calls this after every job, whether it succeeded or
failed. It also works by hand on runs that already finished:

    uv run python scripts/upload_run_log.py data/runs/<run> [data/runs/<run2> ...]

    uv run python scripts/upload_run_log.py --exit-code 1 data/runs/<failed run>

The wandb run is the one named in the "View run at https://…/<entity>/<project>/runs/<id>" line that
wandb itself writes into run.log (the LAST such line, so a resumed run's log goes to the run it
continued). A run that logged offline has no such line and is skipped. Uploading again replaces the
file. Prints one line per run dir; exits 1 if any upload failed.

How: the finished wandb run is briefly resumed (``resume="must"``, console capture / system metrics /
metadata / requirements / code / git all off, so it logs nothing else), the file is saved into it, and it
is finished again with ``--exit-code`` (default 0; queue_runs.sh passes the job's own exit code, so a
failed run stays marked failed). History, summary, config, tags and output.log are untouched (checked
live, 2026-11-10). The public API's ``Run.upload_file`` is NOT used: on wandb 0.27.1 it returns 200 but
the file never shows up in the run.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

_RUN_URL = re.compile(r"View run at https?://\S+?/([^/\s]+)/([^/\s]+)/runs/([A-Za-z0-9]+)")


def wandb_run_path(log_text: str) -> str | None:
    """``<entity>/<project>/<run id>`` from the last "View run at" line wandb wrote; None if offline."""
    hits = _RUN_URL.findall(log_text)
    if not hits:
        return None
    entity, project, run_id = hits[-1]
    return f"{entity}/{project}/{run_id}"


def upload(run_dir: Path, exit_code: int = 0, init=None) -> str | None:
    """Upload ``run_dir/run.log`` to its wandb run; return that run's path (None = offline, skipped).
    ``init`` stands in for ``wandb.init`` (tests)."""
    log = run_dir / "run.log"
    if not log.is_file():
        raise FileNotFoundError(f"{log} does not exist")
    path = wandb_run_path(log.read_text(errors="replace"))
    if path is None:
        return None
    entity, project, run_id = path.split("/")
    settings = None
    if init is None:
        import wandb

        init = wandb.init
        settings = wandb.Settings(console="off", silent=True, x_disable_meta=True, x_disable_stats=True,
                                  x_disable_machine_info=True, x_save_requirements=False,
                                  disable_code=True, disable_git=True)
    tmp = tempfile.mkdtemp(prefix="wandb_log_upload_")  # the resumed session's local files, not ./wandb
    try:
        run = init(entity=entity, project=project, id=run_id, resume="must", dir=tmp, settings=settings)
        try:
            run.save(str(log.resolve()), base_path=str(run_dir.resolve()), policy="now")
        finally:
            run.finish(exit_code=exit_code)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return path


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Upload run dirs' complete run.log to their wandb runs.")
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--exit-code", type=int, default=0,
                    help="exit code the run's job ended with (re-recorded when the wandb run is finished again)")
    args = ap.parse_args(argv)
    load_dotenv()  # WANDB_API_KEY lives in the repo's .env, like for run_experiment.py
    failed = 0
    for d in args.run_dirs:
        try:
            path = upload(d, exit_code=args.exit_code)
        except Exception as e:  # noqa: BLE001 — report and carry on with the other run dirs
            failed += 1
            print(f"{d}: run.log upload to wandb FAILED: {type(e).__name__}: {e}")
            continue
        if path is None:
            print(f"{d}: run.log not uploaded (offline run: no wandb run URL in it)")
        else:
            print(f"{d}: run.log → wandb {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
