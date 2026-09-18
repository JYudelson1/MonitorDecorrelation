"""Capture backend (tinker SDK) warnings into the run's own artifacts.

The tinker client reports a stalled job by logging a WARNING through the stdlib ``logging`` module::

    Training for <id>:train:0 is paused. Reason: Tinker backend is running short on capacity, please wait.

Nothing in this repo configures ``logging``, so that line goes to stderr and is only preserved when a
launcher happens to redirect it (``queue_runs.sh`` → ``data/runs/<run>/run.log``, which is *not*
committed). It matters: runs that hit a capacity pause were observed to change behaviour immediately
afterwards (3 of 6 in the terminal-env batch, 2026-09-16), so "did this run stall?" has to be
answerable from the committed artifacts alone, without shell access to the box that ran it.

``SdkWatch`` is a logging handler that
  - appends every WARNING+ record (from any logger, not just tinker) to ``<run>/sdk_warnings.log``
    with a wall-clock stamp and the training step it happened on, and
  - counts the queue-pause warnings, so ``rl/train.py`` can log a **cumulative** counter into
    ``metrics.jsonl`` — which IS committed. A step where the counter jumps is a stalled step.

Deliberately additive: it only *adds* a handler to the root logger, so existing stderr output is
unchanged, and every method swallows its own errors — a logging problem must never kill a run.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

# The queue-pause warnings we count. tinker phrases them as "<X> is paused. Reason: <reason>" for
# both training (lib/queue_state_logger.py) and sampling (lib/public_interfaces/sampling_client.py);
# the reasons include capacity, a concurrent-client rate limit, and billing holds. Matching the
# shared "is paused" shape catches all of them, including reasons added in later SDK versions.
_PAUSE_RE = re.compile(r"\bis paused\b", re.I)


class SdkWatch(logging.Handler):
    """Root-logger handler: persists WARNING+ records and counts backend queue pauses."""

    def __init__(self, run_dir: str | Path, level: int = logging.WARNING) -> None:
        super().__init__(level=level)
        self.path = Path(run_dir) / "sdk_warnings.log"
        self.n_pause = 0        # cumulative queue-pause warnings (the number logged per step)
        self.n_warnings = 0     # cumulative WARNING+ records of any kind
        self.step: int | None = None  # set by rl/train.py so each line names the step it fell on
        self._fh = None
        try:
            self._fh = self.path.open("a")
        except OSError:  # unwritable dir -> still count, just don't persist
            pass

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 — a bad log record must not raise into the run
            return
        self.n_warnings += 1
        if _PAUSE_RE.search(msg):
            self.n_pause += 1
        if self._fh is None:
            return
        try:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            where = "step=?" if self.step is None else f"step={self.step}"
            self._fh.write(f"[{stamp}] [{where}] {record.levelname} {record.name}: {msg}\n")
            self._fh.flush()
        except Exception:  # noqa: BLE001
            pass

    def metrics(self) -> dict[str, int]:
        """The counters to merge into a metrics row (cumulative, so a jump localises the stall)."""
        return {"backend/queue_pause_warnings": self.n_pause,
                "backend/sdk_warnings": self.n_warnings}

    def close(self) -> None:
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._fh = None
            super().close()


def install(run_dir: str | Path) -> SdkWatch:
    """Attach an ``SdkWatch`` to the root logger and return it.

    The root logger's level is lowered to WARNING only if it is currently higher, so a caller that
    configured its own logging keeps its settings.
    """
    watch = SdkWatch(run_dir)
    root = logging.getLogger()
    if root.level > logging.WARNING or root.level == logging.NOTSET:
        root.setLevel(logging.WARNING)
    root.addHandler(watch)
    return watch


def uninstall(watch: SdkWatch | None) -> None:
    """Detach and close a handler installed by :func:`install` (no-op if ``None``)."""
    if watch is None:
        return
    try:
        logging.getLogger().removeHandler(watch)
    finally:
        watch.close()
