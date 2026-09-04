"""Execution grader for LiveCodeBench-format problems (call-based and stdin/stdout).

Every problem is graded in a fresh subprocess per (solution, problem) — call-based problems run all
their tests in ONE child (a harness script that calls the function per test under its own alarm),
stdin problems run ONE child per test (the program is executed with the test's stdin). Comparison
mirrors LiveCodeBench's ``testing_util``: call-based = ``==`` on json-decoded values (tuple→list),
stdin = line-wise strip + exact-or-Decimal comparison.

Not a sandbox (same caveat as the repo's ``code_exec.py``): timeouts + an address-space rlimit only.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

_CODE_RE = re.compile(r"```[a-zA-Z0-9_+-]*[ \t]*\n(.*?)```", re.S)

IMPORT_STRING = (
    "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\n"
    "from heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\n"
    "from random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\n"
    "from operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\n"
    "from builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\n"
    "import collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\n"
    "import statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\n"
    "import json\nsys.setrecursionlimit(50000)\n"
)


def extract_code(text: str) -> str | None:
    """Last fenced code block (LiveCodeBench's extractor also takes the last block)."""
    m = _CODE_RE.findall(text or "")
    return m[-1] if m else None


@dataclass
class Problem:
    benchmark: str
    task_id: str
    question: str
    starter_code: str  # "" for stdin problems
    fn_name: str | None  # call-based entry point (method on class Solution, or a module fn)
    inputs: list  # call-based: list of JSON-encoded arg lines (LCB style) ; stdin: list of str
    outputs: list  # call-based: list of JSON-encoded expected ; stdin: list of str
    difficulty: str | None = None
    difficulty_rank: float | None = None  # larger = harder (benchmark-specific scale)
    meta: dict = field(default_factory=dict)
    # call-based variant where the test is `assert`-style instead of (input, output) pairs
    assert_tests: list[str] | None = None
    prelude: str = ""  # code exec'd before the model's code (helper classes, imports)
    time_limit: float | None = None  # the benchmark's own per-test limit in seconds (C++-calibrated)
    # stdin problems with a special judge: {"kind": "testlib", "bin": path} (argv: input output answer;
    # exit 0 = OK) | {"kind": "spj2", "bin": path} (argv: answer output; exit 0 = OK)
    # | {"kind": "python", "src": code} (argv: input answer output; prints 1 = OK)
    # | {"kind": "floats", "precision": n} (token-wise compare with abs/rel tolerance)
    checker: dict | None = None
    # per-test wall clock override (seconds); None -> the grader's default
    timeout: float | None = None

    @property
    def is_call_based(self) -> bool:
        return self.fn_name is not None or self.assert_tests is not None

    @property
    def n_tests(self) -> int:
        return len(self.assert_tests) if self.assert_tests is not None else len(self.inputs)


@dataclass
class GradeResult:
    passed: bool
    n_tests: int
    n_passed: int
    status: str  # "ok" | "no_code" | "compile_error" | "wrong_answer" | "runtime_error" | "timeout"
    detail: str = ""


_CALL_HARNESS = r'''
import json, signal, sys, builtins, traceback
NONCE = __NONCE__
CODE = __CODE__
FN = __FN__
INPUTS = __INPUTS__
OUTPUTS = __OUTPUTS__
ASSERTS = __ASSERTS__
PER = __PER__
def emit(**e):
    sys.stdout.write(NONCE + json.dumps(e) + "\n"); sys.stdout.flush()
class _TO(Exception): pass
def _alarm(s, f): raise _TO()
signal.signal(signal.SIGVTALRM, _alarm)
emit(kind="start")
ns = {"__name__": "__sol__"}
try:
    signal.setitimer(signal.ITIMER_VIRTUAL, PER)
    exec(compile(__IMPORTS__ + "\n" + CODE, "<sol>", "exec"), ns)
    signal.setitimer(signal.ITIMER_VIRTUAL, 0)
except BaseException as e:
    signal.setitimer(signal.ITIMER_VIRTUAL, 0)
    emit(kind="compile_error", err=(type(e).__name__ + ": " + str(e))[:400]); sys.exit(0)
if ASSERTS is None:
    if FN in ns and callable(ns[FN]):
        method = ns[FN]
    elif "Solution" in ns:
        try:
            method = getattr(ns["Solution"](), FN)
        except BaseException as e:
            emit(kind="compile_error", err="no method " + FN); sys.exit(0)
    else:
        emit(kind="compile_error", err="entry point not found: " + FN); sys.exit(0)
    for i, (inp, out) in enumerate(zip(INPUTS, OUTPUTS)):
        try:
            args = [json.loads(l) for l in inp.split("\n")]
            exp = json.loads(out)
        except Exception as e:
            emit(kind="unit", i=i, ok=False, why="bad_test"); continue
        try:
            signal.setitimer(signal.ITIMER_VIRTUAL, PER)
            pred = method(*args)
            signal.setitimer(signal.ITIMER_VIRTUAL, 0)
            if isinstance(pred, tuple): pred = list(pred)
            ok = (pred == exp) or ([pred] == exp)
            emit(kind="unit", i=i, ok=bool(ok), why=None if ok else "wa")
            if not ok: break
        except _TO:
            emit(kind="unit", i=i, ok=False, why="timeout"); break
        except BaseException as e:
            signal.setitimer(signal.ITIMER_VIRTUAL, 0)
            emit(kind="unit", i=i, ok=False, why="re:" + (type(e).__name__ + ": " + str(e))[:200]); break
else:
    for i, stmt in enumerate(ASSERTS):
        try:
            signal.setitimer(signal.ITIMER_VIRTUAL, PER)
            exec(compile(stmt, "<assert>", "exec"), ns)
            signal.setitimer(signal.ITIMER_VIRTUAL, 0)
            emit(kind="unit", i=i, ok=True, why=None)
        except _TO:
            emit(kind="unit", i=i, ok=False, why="timeout"); break
        except BaseException as e:
            signal.setitimer(signal.ITIMER_VIRTUAL, 0)
            emit(kind="unit", i=i, ok=False, why="re:" + (type(e).__name__ + ": " + str(e))[:200]); break
emit(kind="done")
'''


# --- global execution throttle -------------------------------------------------------------------
# This box advertises 256 CPUs but its cgroup quota is ~31 cores (/sys/fs/cgroup/cpu.max). Several
# eval processes grade at once, so the number of solution subprocesses running at any moment is
# capped ACROSS processes with N flock'd slot files; and the time limit is enforced on CPU time
# (RLIMIT_CPU) rather than wall-clock, so contention can never turn a fast solution into a TLE.
_SLOT_DIR = os.environ.get("HB_SLOT_DIR", os.path.join(tempfile.gettempdir(), "hb_slots"))
N_SLOTS = int(os.environ.get("HB_EXEC_SLOTS", "24"))
_WALL_FACTOR = 4.0  # wall-clock backstop = CPU limit × this


class _Slot:
    def __init__(self):
        import fcntl
        self._fcntl = fcntl
        os.makedirs(_SLOT_DIR, exist_ok=True)
        self.fh = None

    def __enter__(self):
        import time as _t
        while True:
            for i in range(N_SLOTS):
                fh = open(os.path.join(_SLOT_DIR, f"slot{i}"), "w")
                try:
                    self._fcntl.flock(fh, self._fcntl.LOCK_EX | self._fcntl.LOCK_NB)
                    self.fh = fh
                    return self
                except OSError:
                    fh.close()
            _t.sleep(0.05)

    def __exit__(self, *a):
        self._fcntl.flock(self.fh, self._fcntl.LOCK_UN)
        self.fh.close()


def _run_child(script: str, *, stdin: str | None, timeout: float, mem_mb: int, cwd: str,
               argv: list[str] | None = None):
    """Run ``script`` (or ``argv``) with a CPU-time limit of ``timeout`` seconds.

    -> (stdout, stderr, returncode, timed_out). ``timed_out`` covers both the CPU limit (the child
    dies of SIGXCPU/SIGKILL) and the wall-clock backstop.
    """
    import math
    cpu_s = int(math.ceil(timeout))

    def _limits():
        try:
            import resource
            lim = mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
        except Exception:
            pass
    path = None
    if argv is None:
        path = os.path.join(cwd, f"sol_{uuid.uuid4().hex}.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(script)
        argv = [sys.executable, "-I", path]
    try:
        with _Slot():
            p = subprocess.run(
                argv, input=stdin, capture_output=True, text=True,
                timeout=timeout * _WALL_FACTOR, cwd=cwd, preexec_fn=_limits, errors="replace",
            )
        timed_out = p.returncode in (-24, -9)  # SIGXCPU / SIGKILL (hard CPU limit)
        return p.stdout, p.stderr, p.returncode, timed_out
    except subprocess.TimeoutExpired as e:
        so = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", "replace")
        se = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode("utf-8", "replace")
        return so, se, None, True
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def _lines(s: str) -> list[str]:
    s = s.strip()
    return [l.strip() for l in s.split("\n")]


def _dec(line: str):
    try:
        return [Decimal(x) for x in line.split()]
    except (InvalidOperation, ValueError):
        return None


def stdout_matches(pred: str, exp: str) -> bool:
    pl, el = _lines(pred), _lines(exp)
    if len(pl) != len(el):
        return False
    for a, b in zip(pl, el):
        if a == b:
            continue
        da, db = _dec(a), _dec(b)
        if da is None or db is None or da != db:
            return False
    return True


_CHECKER_BIN: dict[str, str] = {}
_CHECKER_DIR = os.path.join(tempfile.gettempdir(), "hb_checkers")
_TESTLIB_DIR = os.path.dirname(os.path.abspath(__file__))


def _testlib_binary(src: str) -> str:
    """Compile a testlib.h checker once (keyed by source hash); raises on compile failure."""
    import hashlib
    key = hashlib.sha1(src.encode()).hexdigest()
    if key in _CHECKER_BIN:
        return _CHECKER_BIN[key]
    os.makedirs(_CHECKER_DIR, exist_ok=True)
    exe = os.path.join(_CHECKER_DIR, key)
    if not os.path.exists(exe):
        cpp = exe + ".cpp"
        with open(cpp, "w") as fh:
            fh.write(src.replace("\r\n", "\n"))
        r = subprocess.run(["g++", "-O2", "-std=c++17", "-I", _TESTLIB_DIR, "-o", exe + ".tmp", cpp],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError("checker compile failed: " + r.stderr[-800:])
        os.replace(exe + ".tmp", exe)
    _CHECKER_BIN[key] = exe
    return exe


def run_checker(checker: dict, inp: str, exp: str, out: str, cwd: str, timeout: float = 30.0) -> bool:
    """-> accepted? Files are written to ``cwd``; the checker decides.

    Kinds: ``floats`` (DMOJ tolerance compare); ``testlib`` (argv: input output answer, exit 0 = OK;
    ``bin`` = precompiled path, else ``src`` compiled on demand); ``spj2`` (ICPC-Eval spj: argv
    answer output, exit 0 = OK); ``python`` / ``open_r1_python`` (open-r1 checker script: argv input
    answer output, prints 1 = OK).
    """
    kind = checker["kind"]
    if kind == "floats":
        return floats_match(out, exp, checker.get("precision", 6))
    fin, fans, fout = (os.path.join(cwd, n) for n in ("input.txt", "answer.txt", "output.txt"))
    for path, txt in ((fin, inp), (fans, exp), (fout, out)):
        with open(path, "w", encoding="utf-8", errors="replace") as fh:
            fh.write(txt)
    if kind in ("open_r1_python", "python"):
        script = os.path.join(cwd, "checker.py")
        with open(script, "w") as fh:
            fh.write(checker["src"])
        r = subprocess.run([sys.executable, "-I", script, fin, fans, fout], capture_output=True, text=True,
                           timeout=timeout, cwd=cwd, errors="replace")
        return r.stdout.strip().split()[-1:] == ["1"]
    if kind == "testlib":
        exe = checker.get("bin") or _testlib_binary(checker["src"])
        r = subprocess.run([exe, fin, fout, fans], capture_output=True, text=True, timeout=timeout, cwd=cwd,
                           errors="replace")
        return r.returncode == 0
    if kind == "spj2":
        r = subprocess.run([checker["bin"], fans, fout], capture_output=True, text=True, timeout=timeout, cwd=cwd,
                           errors="replace")
        return r.returncode == 0
    raise ValueError(f"unknown checker kind {kind!r}")


def floats_match(pred: str, exp: str, precision: int) -> bool:
    """DMOJ 'floats' checker: token-wise, numeric tokens compared with abs/rel tolerance 10^-precision."""
    a, b = pred.split(), exp.split()
    if len(a) != len(b):
        return False
    tol = 10.0 ** (-precision)
    for x, y in zip(a, b):
        if x == y:
            continue
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            return False
        if abs(fx - fy) > tol and abs(fx - fy) > tol * abs(fy):
            return False
    return True


def effective_timeout(prob: Problem, per_test_timeout: float, mult: float = 5.0, cap: float = 60.0) -> float:
    """Python gets ``mult``× the benchmark's C++-calibrated limit, never below ``per_test_timeout``."""
    if prob.time_limit:
        return min(cap, max(per_test_timeout, mult * float(prob.time_limit)))
    return per_test_timeout


def grade(code: str | None, prob: Problem, *, per_test_timeout: float = 6.0,
          mem_mb: int = 4096, workdir: str | None = None) -> GradeResult:
    if code is None or not code.strip():
        return GradeResult(False, prob.n_tests, 0, "no_code")
    n = prob.n_tests
    per_test_timeout = effective_timeout(prob, per_test_timeout)
    if prob.prelude:
        code = prob.prelude + "\n" + code
    if prob.timeout is not None:
        per_test_timeout = max(per_test_timeout, prob.timeout)
    with tempfile.TemporaryDirectory(prefix="hb_", dir=workdir) as tmp:
        if prob.is_call_based:
            nonce = "HB" + uuid.uuid4().hex + ":"
            script = _CALL_HARNESS
            for k, v in (("__NONCE__", nonce), ("__FN__", prob.fn_name or ""), ("__INPUTS__", prob.inputs),
                         ("__OUTPUTS__", prob.outputs), ("__ASSERTS__", prob.assert_tests),
                         ("__PER__", float(per_test_timeout)), ("__IMPORTS__", IMPORT_STRING),
                         ("__CODE__", code)):
                script = script.replace(k, repr(v))
            total = per_test_timeout * max(1, n) + 20.0
            so, se, rc, to = _run_child(script, stdin=None, timeout=total, mem_mb=mem_mb, cwd=tmp)
            if to and not any(True for line in so.splitlines() if line.startswith(nonce)):
                return GradeResult(False, n, 0, "timeout", "whole harness exceeded its CPU budget")
            events = []
            for line in so.splitlines():
                if line.startswith(nonce):
                    try:
                        events.append(json.loads(line[len(nonce):]))
                    except json.JSONDecodeError:
                        pass
            if not any(e["kind"] == "start" for e in events):
                return GradeResult(False, n, 0, "runtime_error", f"harness never started rc={rc} {se[-300:]}")
            ce = next((e for e in events if e["kind"] == "compile_error"), None)
            if ce:
                return GradeResult(False, n, 0, "compile_error", ce["err"])
            units = [e for e in events if e["kind"] == "unit"]
            n_ok = sum(1 for u in units if u["ok"])
            if n_ok == n and any(e["kind"] == "done" for e in events):
                return GradeResult(True, n, n_ok, "ok")
            bad = next((u for u in units if not u["ok"]), None)
            if bad is None:
                return GradeResult(False, n, n_ok, "timeout" if to else "runtime_error",
                                   "wall-clock" if to else f"died rc={rc}")
            why = bad.get("why") or ""
            status = ("timeout" if why == "timeout"
                      else "wrong_answer" if why.startswith("re:AssertionError")  # assert-style tests
                      else "runtime_error" if why.startswith("re:") else "wrong_answer")
            return GradeResult(False, n, n_ok, status, why[:300])
        # ---- stdin/stdout: one child per test ------------------------------------------------------
        n_ok = 0
        for i, (inp, exp) in enumerate(zip(prob.inputs, prob.outputs)):
            so, se, rc, to = _run_child(code, stdin=inp, timeout=per_test_timeout, mem_mb=mem_mb, cwd=tmp)
            if to:
                return GradeResult(False, n, n_ok, "timeout", f"test {i}")
            if rc != 0:
                if i == 0 and ("SyntaxError" in se or "IndentationError" in se):
                    return GradeResult(False, n, n_ok, "compile_error", se[-300:])
                return GradeResult(False, n, n_ok, "runtime_error", f"test {i} rc={rc} {se[-300:]}")
            if prob.checker is not None:
                try:
                    ok = run_checker(prob.checker, inp, exp, so, tmp)
                except Exception as e:  # a broken checker must not silently pass or fail the model
                    return GradeResult(False, n, n_ok, "checker_error", f"test {i}: {e}"[:300])
            else:
                ok = stdout_matches(so, exp)
            if not ok:
                return GradeResult(False, n, n_ok, "wrong_answer", f"test {i}")
            n_ok += 1
        return GradeResult(True, n, n_ok, "ok")
