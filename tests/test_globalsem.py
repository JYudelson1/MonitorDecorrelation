"""Cross-process semaphore (``globalsem``): the cap holds, and a crashed holder can't leak a permit.

Offline: no network, no tinker. The crash tests spawn real subprocesses and SIGKILL them, because
"the permit comes back when the holder dies" is the whole reason this uses flock rather than a
POSIX semaphore or a counter file — mocking it would test nothing.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from monitordecorrelation import globalsem
from monitordecorrelation.globalsem import slot


@pytest.fixture(autouse=True)
def _sem_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MD_GLOBAL_SEM_DIR", str(tmp_path / "sem"))


def test_limits_are_sane():
    """The two knobs exist, in one place, with the documented values."""
    assert globalsem.OPENROUTER_MAX_CONCURRENT == 1024
    assert globalsem.CODE_EXEC_MAX_CONCURRENT == max(1, globalsem._cpu_count() // 2)
    assert globalsem.CODE_EXEC_MAX_CONCURRENT >= 1


def test_zero_disables():
    """n <= 0 is a no-op passthrough — a misconfiguration must not deadlock a run."""
    with slot("off", 0):
        pass
    with slot("off", -1):
        pass


def test_caps_concurrency_within_a_process():
    """Never more than n threads inside the block at once, and all of them get through."""
    n, threads = 3, 12
    live, peak, done = 0, 0, []
    lk = threading.Lock()

    def body(i):
        nonlocal live, peak
        with slot("cap", n):
            with lk:
                live += 1
                peak = max(peak, live)
            time.sleep(0.02)
            with lk:
                live -= 1
            done.append(i)

    ts = [threading.Thread(target=body, args=(i,)) for i in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in ts)
    assert sorted(done) == list(range(threads))
    assert peak <= n
    assert peak > 1  # the semaphore isn't accidentally serialising everything


def test_permit_released_when_block_raises():
    """An exception in the body still releases — otherwise one bad rollout burns a permit."""
    for _ in range(4):  # more iterations than permits: a leak would hang the 4th
        with pytest.raises(ValueError):
            with slot("boom", 2):
                raise ValueError("x")
    with slot("boom", 2):
        pass


# --- the crash properties -------------------------------------------------------------------

_HOLDER = """
import sys, time
sys.path.insert(0, {src!r})
import os
os.environ["MD_GLOBAL_SEM_DIR"] = {d!r}
from monitordecorrelation.globalsem import slot
with slot({name!r}, 1):
    print("held", flush=True)
    time.sleep(600)
"""


def _spawn_holder(name, d):
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    p = subprocess.Popen(
        [sys.executable, "-c", _HOLDER.format(src=src, d=str(d), name=name)],
        stdout=subprocess.PIPE, text=True,
    )
    assert p.stdout.readline().strip() == "held"  # it has the permit before we return
    return p


def _acquired_within(name, n, seconds):
    got = threading.Event()

    def run():
        with slot(name, n):
            got.set()

    threading.Thread(target=run, daemon=True).start()
    return got.wait(timeout=seconds)


def test_permit_is_held_across_processes():
    """Sanity check for the two tests below: a live holder really does block us."""
    d = os.environ["MD_GLOBAL_SEM_DIR"]
    p = _spawn_holder("xproc", d)
    try:
        assert not _acquired_within("xproc", 1, 1.0)
    finally:
        p.kill()
        p.wait(timeout=10)


def test_sigkilled_holder_does_not_leak_its_permit():
    """SIGKILL = no cleanup handler runs. The kernel must return the permit anyway.

    This is the failure mode that rules out POSIX named semaphores and counter files: there, a
    crashed run leaks a permit and the global cap ratchets down to zero.
    """
    d = os.environ["MD_GLOBAL_SEM_DIR"]
    p = _spawn_holder("killed", d)
    assert not _acquired_within("killed", 1, 0.5)
    os.kill(p.pid, signal.SIGKILL)
    p.wait(timeout=10)
    assert _acquired_within("killed", 1, 10.0)


def test_stale_slot_files_are_free(tmp_path):
    """What survives a power cut is the files, not the locks — so the semaphore starts empty."""
    d = tmp_path / "sem" / "stale"
    d.mkdir(parents=True)
    for i in range(4):
        (d / str(i)).write_text("")  # left over from a machine that went down mid-run
    assert _acquired_within("stale", 4, 5.0)


def test_openrouter_backoff_sleep_does_not_hold_a_permit(monkeypatch):
    """`chat` retries forever with backoff; the permit must cover ONE attempt, not the loop.

    If it wrapped the whole call, a single judge stuck in its retry loop (a model that's down,
    say) would sit on a permit indefinitely — with 1024 of them and enough parallel runs, that
    starves every other run on the box. Pinning this because the `with` is one indentation level
    away from being wrong.
    """
    from monitordecorrelation.monitors import openrouter

    monkeypatch.setattr(globalsem, "OPENROUTER_MAX_CONCURRENT", 1)  # one permit: unambiguous
    monkeypatch.setattr(openrouter, "_OPENROUTER_URL", "http://127.0.0.1:1/dead")  # instant failure

    threading.Thread(
        target=lambda: openrouter.chat({}, api_key="k", timeout=0.2, name="probe"), daemon=True
    ).start()
    time.sleep(1.5)  # attempt 1 has failed; we are inside a backoff sleep

    got = threading.Event()

    def other():
        with globalsem.openrouter_slot():
            got.set()

    threading.Thread(target=other, daemon=True).start()
    assert got.wait(timeout=5.0), "backoff sleep held the only OpenRouter permit"
