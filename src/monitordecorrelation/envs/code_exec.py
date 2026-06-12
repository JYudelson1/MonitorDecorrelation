"""Minimal code-test executor for the coding envs.

Runs candidate code against a list of ``assert`` tests in a SUBPROCESS with a timeout and returns
per-test pass/fail.

⚠️ SECURITY: this executes untrusted, model-generated code with ONLY a timeout for protection — it is
NOT a sandbox. The code can read/write files, open sockets, spawn processes, etc. Fine for local
MBPP-scale validation on a trusted machine; **harden before any large or shared run** (docker / nsjail
/ firejail, no network, read-only FS, CPU/mem rlimits). Tracked as the "sandboxed code execution
harness" in STATUS.md.
"""

from __future__ import annotations

import subprocess
import sys


def run_tests(code: str, tests: list[str], setup: str = "", timeout: float = 6.0) -> list[bool]:
    """-> per-test pass booleans (one per ``tests`` entry). A crash / syntax error / timeout in
    ``code`` fails ALL tests. Each test is an ``assert`` string ``exec``'d in the module globals where
    ``code``'s top-level defs live."""
    tests = list(tests)
    if not tests:
        return []
    # Build the harness by CONCATENATION, not str.format — code/tests routinely contain { } braces.
    script = (
        (setup or "")
        + "\n"
        + (code or "")
        + "\n"
        + "__tests = "
        + repr(tests)
        + "\n"
        + "__r = []\n"
        + "for __t in __tests:\n"
        + "    try:\n"
        + "        exec(__t, globals())\n"
        + "        __r.append('1')\n"
        + "    except Exception:\n"
        + "        __r.append('0')\n"
        + "import sys as _s; _s.stdout.write('RESULTS:' + ''.join(__r))\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return [False] * len(tests)
    out = proc.stdout
    if "RESULTS:" not in out:  # code crashed before the test loop ran (e.g. a syntax error)
        return [False] * len(tests)
    bits = out.split("RESULTS:")[-1]
    bits = "".join(ch for ch in bits if ch in "01")[: len(tests)].ljust(len(tests), "0")
    return [ch == "1" for ch in bits]
