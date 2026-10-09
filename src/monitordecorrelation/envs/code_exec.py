"""Minimal code-test executor for the coding envs.

Runs candidate code against a list of ``assert`` tests in a SUBPROCESS with a timeout and returns
per-test pass/fail.

⚠️ SECURITY: this executes untrusted, model-generated code with ONLY a timeout for protection — it is
NOT a sandbox. The code can read/write files, open sockets, spawn processes, etc. Fine for local
MBPP-scale validation on a trusted machine; **harden before any large or shared run** (docker / nsjail
/ firejail, no network, read-only FS, CPU/mem rlimits). Tracked as the "sandboxed code execution
harness" in STATUS.md.

Concurrency is capped ACROSS PROCESSES by ``globalsem.code_exec_slot`` (half the box's cores),
so k runs launched in parallel share one budget instead of each claiming a full one.
"""

from __future__ import annotations

import subprocess
import sys

from monitordecorrelation.globalsem import code_exec_slot


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
        # One permit for the duration of the spawn — the cap is shared with every other run on the
        # box (globalsem). Held only around a call with a hard timeout, so it cannot be stuck.
        with code_exec_slot():
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


# --- stdin/stdout programs (Codeforces-style tests) ----------------------------------------------

# Runs the candidate as ``__main__`` (so ``if __name__ == "__main__":`` guards fire) under an
# address-space cap. A launcher rather than ``preexec_fn``: the RL loop grades from many threads, and
# ``preexec_fn`` is not safe there.
_IO_LAUNCHER = (
    "import resource, runpy, sys\n"
    "_lim = int(sys.argv[2]) * 1024 * 1024\n"
    "resource.setrlimit(resource.RLIMIT_AS, (_lim, _lim))\n"
    "sys.argv = sys.argv[1:2]\n"
    "runpy.run_path(sys.argv[0], run_name='__main__')\n"
)


def _token_eq(got: str, want: str) -> bool:
    if got == want:
        return True
    if want.lower() in ("yes", "no"):  # Codeforces' yes/no checkers ignore case
        return got.lower() == want.lower()
    if "." in want:  # a real-valued answer: absolute/relative 1e-6, the usual Codeforces tolerance
        try:
            g, w = float(got), float(want)
        except ValueError:
            return False
        return abs(g - w) <= 1e-6 * max(1.0, abs(w))
    return False


def outputs_match(got: str, want: str) -> bool:
    """Codeforces-style comparison: whitespace-separated tokens equal (yes/no case-insensitively,
    decimals to 1e-6). Only used on problems without a custom checker (the loader drops those)."""
    g, w = got.split(), want.split()
    return len(g) == len(w) and all(_token_eq(a, b) for a, b in zip(g, w))


def run_io_test(code: str, stdin: str, expected: str, *, timeout: float,
                mem_mb: int = 2048) -> tuple[str, str | None]:
    """Run ``code`` as a program on ``stdin`` → ``(status, stdout)``. ``status`` is ``"ok"`` (exit 0
    and the output matches ``expected``), ``"wrong_answer"``, ``"runtime_error"`` or ``"timeout"``
    (stdout None). Same trust model as :func:`run_tests` — a timeout + memory cap, not a sandbox."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory(prefix="io_test_") as d:
        path = os.path.join(d, "solution.py")
        with open(path, "w") as f:
            f.write(code or "")
        try:
            with code_exec_slot():
                proc = subprocess.run(
                    [sys.executable, "-c", _IO_LAUNCHER, path, str(mem_mb)], input=stdin,
                    capture_output=True, text=True, timeout=timeout, cwd=d, errors="replace",
                )
        except subprocess.TimeoutExpired:
            return "timeout", None
    if proc.returncode != 0:
        return "runtime_error", proc.stdout
    return ("ok" if outputs_match(proc.stdout, expected) else "wrong_answer"), proc.stdout
