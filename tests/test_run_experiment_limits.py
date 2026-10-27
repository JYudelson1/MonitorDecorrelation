"""experiments/run_experiment.py raises its soft open-files limit at start: at 1024, a TerminalEnv eval's judge calls
(256 episodes × 4 judges at once) failed with EMFILE and aborted three runs of the probe-row batch (2026-10-27)."""

from __future__ import annotations

import resource
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROBE = """
import importlib.util, resource, sys
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(256, hard), hard))
spec = importlib.util.spec_from_file_location("run_experiment", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print(*m._raise_open_files_limit(), *m._raise_open_files_limit(128))
"""


def test_a_run_raises_its_soft_open_files_limit_but_never_past_the_hard_one():
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    out = subprocess.run([sys.executable, "-c", PROBE, str(ROOT / "experiments" / "run_experiment.py")],
                         capture_output=True, text=True, check=True, cwd=ROOT).stdout.split()
    soft1, hard1, soft2, hard2 = (int(x) for x in out)
    want = 65536 if hard == resource.RLIM_INFINITY else min(65536, hard)
    assert (soft1, hard1) == (want, hard)
    assert (soft2, hard2) == (want, hard)  # a smaller want never lowers it
