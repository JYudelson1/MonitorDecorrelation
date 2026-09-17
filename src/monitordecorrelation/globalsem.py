"""Cross-process semaphores: caps that hold across *every* run on this box, not just one process.

Why
---
Two resources are shared by every process the repo starts — the machine's CPUs (untrusted model
code runs in subprocesses) and our OpenRouter account (judge calls). Both are already capped
*inside* a run (``episodes.step_workers``, the ``MonitorScorer`` thread pool), but those caps are
per-process, so launching k runs in parallel (separate terminals / tmux windows) multiplies them by
k: the box thrashes, and OpenRouter starts answering the burst with the in-flight-budget 402 that
``monitors.openrouter`` has to retry around. Dividing each run's worker count by k fixes that but
wastes the elasticity — one run grading while another samples should be allowed the whole budget.

So: a semaphore whose permits live outside any one process.

How (and why it survives crashes)
---------------------------------
A permit is an exclusive ``flock`` on one of ``n`` slot files under ``sem_dir()``. Acquire = scan
the slots for one that locks; release = close the fd.

That choice is the whole point. ``flock`` state lives in the kernel, attached to the open file
description, so the kernel drops it when the holder's fds close — which happens on *every* exit
path, including ``SIGKILL``, an OOM kill, a closed tmux window, or a panic. There is no cleanup
handler to miss and no count on disk that can drift out of sync with reality: a crashed run cannot
leak a permit. Power-cycling the machine is equally safe — the zero-byte slot files may survive, but
lock state does not, so every slot comes back free. Contrast the obvious alternatives, which are
exactly the trap here: a POSIX named semaphore (``multiprocessing.Semaphore``, ``posix_ipc``) or a
counter in a file leaks a permit per crash, ratcheting the global cap down to zero and silently
stalling every future run.

The one thing ``flock`` cannot protect against is a holder that is *hung but alive* (SIGSTOP, an
uninterruptible syscall) — that permit is held until the process dies. The invariant that closes
that gap: **never hold a permit across an unbounded wait.** Every call site obeys it — the code-exec
sites hold it only around a ``subprocess`` call that has a hard timeout, and ``openrouter.chat``
takes a permit per HTTP attempt (bounded by the request timeout) rather than around its
retry-forever loop. Keep it that way when adding sites.

Non-goals: this is not fair (no FIFO) and it polls. Both are fine at this scale — tens of waiters,
holds measured in seconds. It is also *only* a resource cap: it delays work, never reorders anything
semantically meaningful (per-call seeds are position-addressed), so it cannot change a run's
results, only its timing.

Deadlock: the two semaphores must never be nested. Today they can't be — env steps and monitor
scoring run in different thread pools — and no call site should start.
"""

from __future__ import annotations

import fcntl
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------------------------
# The two limits. Change them HERE and nowhere else — every training and evaluation entry point
# in the repo reaches the same semaphore through the helpers below.
# ---------------------------------------------------------------------------------------------


def _cpu_count() -> int:
    """Cores this process may actually run on (``sched_getaffinity`` respects taskset/cpuset)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # non-Linux
        return os.cpu_count() or 2


#: Concurrent executions of model-generated code, across all runs on this box. Half the cores:
#: the other half is left for the runs' own Python (sampling threads, grading, tokenization).
CODE_EXEC_MAX_CONCURRENT = max(1, _cpu_count() // 2)

#: Concurrent in-flight OpenRouter requests, across all runs on this box.
OPENROUTER_MAX_CONCURRENT = 256

#: Where the slot files live. Must be a LOCAL filesystem — ``flock`` over NFS or a shared host
#: volume is not something to rely on, and a volume would also silently share the cap with other
#: instances mounting it. Overridable for tests / for deliberately separating two sets of runs.
_SEM_DIR_ENV = "MD_GLOBAL_SEM_DIR"
_DEFAULT_SEM_DIR = "/tmp/monitordecorrelation-sem"


def sem_dir() -> Path:
    return Path(os.environ.get(_SEM_DIR_ENV) or _DEFAULT_SEM_DIR)


@contextmanager
def slot(name: str, n: int, *, poll: float = 0.005, max_poll: float = 0.25) -> Iterator[None]:
    """Hold one of ``n`` permits of the cross-process semaphore ``name`` for the block's duration.

    Blocks until a permit is free. ``n <= 0`` disables the cap (the block runs immediately), which
    is what makes this safe to add to a hot path: a misconfiguration degrades to the old,
    per-process-only behaviour instead of deadlocking.

    The scan order is shuffled per acquisition so waiters don't all pile onto slot 0, and the poll
    interval backs off to ``max_poll`` so a saturated semaphore doesn't spin.
    """
    if n <= 0:
        yield
        return

    d = sem_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    delay = poll
    while True:
        for i in random.sample(range(n), n):
            fd = os.open(d / str(i), os.O_CREAT | os.O_RDWR, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:  # held by someone else — try the next slot
                os.close(fd)
                continue
            try:
                yield
            finally:
                # Closing the fd releases the flock. This is belt-and-braces only: if we never got
                # here (SIGKILL, power loss) the kernel does exactly the same thing for us.
                os.close(fd)
            return
        time.sleep(delay * (1.0 + random.random()))
        delay = min(delay * 1.5, max_poll)


def code_exec_slot():
    """Permit to run one model-generated-code subprocess. Hold it ONLY around a timed-out spawn."""
    return slot("code_exec", CODE_EXEC_MAX_CONCURRENT)


def openrouter_slot():
    """Permit for one in-flight OpenRouter request. Hold it around ONE attempt, never a retry loop."""
    return slot("openrouter", OPENROUTER_MAX_CONCURRENT)
