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
signal.signal(signal.SIGALRM, _alarm)
emit(kind="start")
ns = {"__name__": "__sol__"}
try:
    signal.setitimer(signal.ITIMER_REAL, PER)
    exec(compile(__IMPORTS__ + "\n" + CODE, "<sol>", "exec"), ns)
    signal.setitimer(signal.ITIMER_REAL, 0)
except BaseException as e:
    signal.setitimer(signal.ITIMER_REAL, 0)
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
            signal.setitimer(signal.ITIMER_REAL, PER)
            pred = method(*args)
            signal.setitimer(signal.ITIMER_REAL, 0)
            if isinstance(pred, tuple): pred = list(pred)
            ok = (pred == exp)
            emit(kind="unit", i=i, ok=bool(ok), why=None if ok else "wa")
            if not ok: break
        except _TO:
            emit(kind="unit", i=i, ok=False, why="timeout"); break
        except BaseException as e:
            signal.setitimer(signal.ITIMER_REAL, 0)
            emit(kind="unit", i=i, ok=False, why="re:" + (type(e).__name__ + ": " + str(e))[:200]); break
else:
    for i, stmt in enumerate(ASSERTS):
        try:
            signal.setitimer(signal.ITIMER_REAL, PER)
            exec(compile(stmt, "<assert>", "exec"), ns)
            signal.setitimer(signal.ITIMER_REAL, 0)
            emit(kind="unit", i=i, ok=True, why=None)
        except _TO:
            emit(kind="unit", i=i, ok=False, why="timeout"); break
        except BaseException as e:
            signal.setitimer(signal.ITIMER_REAL, 0)
            emit(kind="unit", i=i, ok=False, why="re:" + (type(e).__name__ + ": " + str(e))[:200]); break
emit(kind="done")
'''


def _run_child(script: str, *, stdin: str | None, timeout: float, mem_mb: int, cwd: str):
    def _limits():
        try:
            import resource
            lim = mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        except Exception:
            pass
    path = os.path.join(cwd, f"sol_{uuid.uuid4().hex}.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(script)
    try:
        p = subprocess.run(
            [sys.executable, "-I", path], input=stdin, capture_output=True, text=True,
            timeout=timeout, cwd=cwd, preexec_fn=_limits, errors="replace",
        )
        return p.stdout, p.stderr, p.returncode, False
    except subprocess.TimeoutExpired as e:
        so = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", "replace")
        se = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode("utf-8", "replace")
        return so, se, None, True
    finally:
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


def grade(code: str | None, prob: Problem, *, per_test_timeout: float = 6.0,
          mem_mb: int = 4096, workdir: str | None = None) -> GradeResult:
    if code is None or not code.strip():
        return GradeResult(False, prob.n_tests, 0, "no_code")
    n = prob.n_tests
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
            status = ("timeout" if why == "timeout" else "runtime_error" if why.startswith("re:")
                      else "wrong_answer")
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
            if not stdout_matches(so, exp):
                return GradeResult(False, n, n_ok, "wrong_answer", f"test {i}")
            n_ok += 1
        return GradeResult(True, n, n_ok, "ok")
