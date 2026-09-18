"""Backend-warning capture: queue pauses must be visible in the committed artifacts."""
from __future__ import annotations

import json
import logging

from monitordecorrelation.rl import sdk_watch as sw


def test_counts_queue_pauses_and_persists_them(tmp_path):
    watch = sw.install(tmp_path)
    try:
        tinker_log = logging.getLogger("tinker.lib.queue_state_logger")
        # the exact line the SDK emits (lib/queue_state_logger.py)
        tinker_log.warning(
            "Training for fe5b3384-eb9b-57a5-8601-5e119f42a137:train:0 is paused. "
            "Reason: Tinker backend is running short on capacity, please wait."
        )
        watch.step = 7
        tinker_log.warning("Sampling is paused for sampler abc. Reason: concurrent training clients "
                           "rate limit hit.")
        logging.getLogger("something.else").warning("unrelated warning")
        logging.getLogger("noisy").info("info is below the handler level")
    finally:
        sw.uninstall(watch)

    assert watch.n_pause == 2          # both pause shapes matched
    assert watch.n_warnings == 3       # the unrelated warning counts, the info record does not
    m = watch.metrics()
    assert m["backend/queue_pause_warnings"] == 2 and m["backend/sdk_warnings"] == 3
    json.dumps(m)  # must be metrics-row serialisable

    text = (tmp_path / "sdk_warnings.log").read_text()
    assert "running short on capacity" in text and "rate limit hit" in text
    assert "unrelated warning" in text
    assert "step=?" in text and "step=7" in text   # step is stamped once train.py sets it
    print("queue pauses counted + persisted OK")


def test_uninstall_detaches_and_is_idempotent(tmp_path):
    before = len(logging.getLogger().handlers)
    watch = sw.install(tmp_path)
    assert len(logging.getLogger().handlers) == before + 1
    sw.uninstall(watch)
    assert len(logging.getLogger().handlers) == before
    logging.getLogger("tinker").warning("Training for x is paused. Reason: capacity.")
    assert watch.n_pause == 0          # nothing recorded after detach
    sw.uninstall(watch)                # second call must not raise
    sw.uninstall(None)
    print("uninstall detaches cleanly OK")


def test_never_raises_into_the_run(tmp_path):
    """A logging failure must not kill a multi-hour run."""
    watch = sw.install(tmp_path)
    try:
        watch._fh.close()  # simulate the file handle dying mid-run
        logging.getLogger("tinker").warning("Training for x is paused. Reason: capacity.")
        assert watch.n_pause == 1      # still counted even though the write failed
        # a record whose message cannot be rendered: emit() is called directly, because routing it
        # through logging would also hand it to every OTHER handler (whose behaviour is not ours).
        bad = logging.LogRecord("tinker", logging.WARNING, __file__, 1, "%d", ("not-an-int",), None)
        watch.emit(bad)                # must swallow the TypeError from getMessage()
        assert watch.n_pause == 1      # and must not have counted the unrenderable record
    finally:
        sw.uninstall(watch)
    print("logging failures are swallowed OK")
