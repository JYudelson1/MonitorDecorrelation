"""Offline tests for queue_runs.sh's seed options (-o FIRST_SEED, -O distinct seeds per config) and for
the run.log → wandb upload every queue_runs job ends with (scripts/upload_run_log.py, fake wandb API)."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_QUEUE = _REPO / "scripts/queue_runs.sh"
_needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

_spec = importlib.util.spec_from_file_location("upload_run_log", _REPO / "scripts/upload_run_log.py")
upload_run_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(upload_run_log)


# ---- seeds ------------------------------------------------------------------------------------------

def _plan(tmp_path: Path, *opts: str) -> list[tuple[str, int]]:
    """Dry-run the launcher over two configs (a_s0, b_s0) → [(run_name, seed)] in plan order."""
    cfgs = tmp_path / "cfgs"
    cfgs.mkdir(exist_ok=True)
    for name in ("a", "b"):
        (cfgs / f"{name}.json").write_text(json.dumps({"run_name": f"_pytest_{name}_s0"}))
    out = subprocess.run(["bash", str(_QUEUE), "-c", str(cfgs), "-d", *opts], capture_output=True,
                         text=True, check=True).stdout
    return [(m[0], int(m[1])) for m in re.findall(r"queue (\S+)\s+seed=(\d+)", out)]


@_needs_bash
def test_default_seeds_are_unchanged(tmp_path):
    assert _plan(tmp_path, "-n", "2") == [("_pytest_a_s0", 0), ("_pytest_a_s1", 1),
                                          ("_pytest_b_s0", 0), ("_pytest_b_s1", 1)]


@_needs_bash
def test_first_seed_shifts_every_config(tmp_path):
    assert _plan(tmp_path, "-n", "2", "-o", "5") == [("_pytest_a_s5", 5), ("_pytest_a_s6", 6),
                                                     ("_pytest_b_s5", 5), ("_pytest_b_s6", 6)]


@_needs_bash
def test_distinct_seeds_give_each_config_its_own_block(tmp_path):
    assert _plan(tmp_path, "-n", "2", "-o", "5", "-O") == [("_pytest_a_s5", 5), ("_pytest_a_s6", 6),
                                                           ("_pytest_b_s7", 7), ("_pytest_b_s8", 8)]
    assert _plan(tmp_path, "-n", "1", "-o", "9", "-O", "-R", "ctrl") == [("_pytest_a_s9_ctrl", 9),
                                                                        ("_pytest_b_s10_ctrl", 10)]


@_needs_bash
@pytest.mark.parametrize("opt", [["-o", "-1"], ["-o", "x"], ["-n", "0"]])
def test_bad_seed_options_are_refused(tmp_path, opt):
    (tmp_path / "c.json").write_text(json.dumps({"run_name": "_pytest_c_s0"}))
    r = subprocess.run(["bash", str(_QUEUE), "-c", str(tmp_path / "c.json"), "-d", *opt],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "error" in r.stderr


# ---- run.log → wandb --------------------------------------------------------------------------------

_LOG = ("  wandb: online project=monitor-decorrelation (syncing — logged in)\n"
        "wandb: 🚀 View run at https://wandb.ai/ent/proj/runs/abc123\n"
        "[12:00:00] step 0: …\nterminal_verifier finished OK\n")


class _FakeInit:
    """Stands in for wandb.init: records the resume kwargs, then the run's save / finish calls."""

    def __init__(self, fail_save: bool = False):
        self.calls, self.fail_save = [], fail_save

    def __call__(self, **kw):
        self.calls.append(("init", kw["entity"], kw["project"], kw["id"], kw["resume"]))
        rec = self

        class _Run:
            def save(self, p, base_path, policy):
                if rec.fail_save:
                    raise RuntimeError("boom")
                rec.calls.append(("save", p, base_path, policy, Path(p).read_text()))

            def finish(self, exit_code):
                rec.calls.append(("finish", exit_code))
        return _Run()


def test_run_path_is_the_last_view_run_line():
    assert upload_run_log.wandb_run_path(_LOG) == "ent/proj/abc123"
    resumed = _LOG + "wandb: 🚀 View run at https://wandb.ai/ent/proj/runs/zzz999\n"
    assert upload_run_log.wandb_run_path(resumed) == "ent/proj/zzz999"
    assert upload_run_log.wandb_run_path("  wandb: offline …\nstep 0\n") is None


@pytest.mark.parametrize("exit_code", [0, 3])
def test_upload_resumes_that_run_saves_the_whole_log_and_refinishes_it(tmp_path, exit_code):
    (tmp_path / "run.log").write_text(_LOG)
    init = _FakeInit()
    assert upload_run_log.upload(tmp_path, exit_code=exit_code, init=init) == "ent/proj/abc123"
    assert init.calls == [("init", "ent", "proj", "abc123", "must"),
                          ("save", str((tmp_path / "run.log").resolve()), str(tmp_path.resolve()), "now", _LOG),
                          ("finish", exit_code)]


def test_the_resumed_run_is_finished_even_when_the_save_fails(tmp_path):
    (tmp_path / "run.log").write_text(_LOG)
    init = _FakeInit(fail_save=True)
    with pytest.raises(RuntimeError):
        upload_run_log.upload(tmp_path, init=init)
    assert init.calls[-1] == ("finish", 0)


def test_offline_run_is_skipped_and_a_missing_log_is_an_error(tmp_path, capsys):
    (tmp_path / "run.log").write_text("  wandb: offline\n")
    init = _FakeInit()
    assert upload_run_log.upload(tmp_path, init=init) is None and init.calls == []
    with pytest.raises(FileNotFoundError):
        upload_run_log.upload(tmp_path / "nope", init=init)
    assert upload_run_log.main([str(tmp_path), str(tmp_path / "nope")]) == 1   # one failed → exit 1
    out = capsys.readouterr().out
    assert "offline run" in out and "FAILED" in out


@_needs_bash
@pytest.mark.parametrize("succeeds", [True, False])
def test_worker_uploads_the_log_whether_the_job_succeeds_or_fails(tmp_path, succeeds):
    shim = tmp_path / "bin"
    shim.mkdir()
    calls = tmp_path / "calls"
    # fake `uv`: the run_experiment.py call writes a log line and exits 0/1; the upload call is recorded
    (shim / "uv").write_text(
        "#!/usr/bin/env bash\n"
        f'case "$*" in *run_experiment.py*) echo "a log line"; exit {0 if succeeds else 3} ;; esac\n'
        f'printf "%s\\n" "$*" >> "{calls}"; echo "data/runs/x: run.log → wandb ent/proj/abc123"\n')
    (shim / "uv").chmod(0o755)
    runname = f"_pytest_upload_{uuid.uuid4().hex[:8]}"
    run_dir = _REPO / "data/runs" / runname
    try:
        env = {**os.environ, "PATH": f"{shim}:{os.environ['PATH']}"}
        rec = f"0\t{_REPO}/experiments/configs/terminal_verifier_monitor_only/q3_cot.json\t{runname}" \
              f"\tdata/runs/{runname}/run.log"
        r = subprocess.run(["bash", str(_QUEUE), "__worker", rec], env=env, capture_output=True, text=True)
        assert r.returncode == (0 if succeeds else 3)       # the upload never changes the job's outcome
        rc = 0 if succeeds else 3   # the job's exit code is passed on, so a failed run stays failed on wandb
        assert calls.read_text().split("\n")[:-1] == [
            f"run python scripts/upload_run_log.py --exit-code {rc} data/runs/{runname}"]
        assert f"{runname}" in r.stdout and "run.log → wandb ent/proj/abc123" in r.stdout
        assert (run_dir / "QUEUE_DONE").exists() == succeeds
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
