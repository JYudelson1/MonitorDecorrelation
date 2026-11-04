"""Agentic coding benchmarks: a bash-only agent in a docker sandbox, graded on HIDDEN tests.

Scaffold (the "bash-only" style of mini-swe-agent, identical for every model)
-------------------------------------------------------------------------------
The model works in ``/work`` of a network-less container. Every reply must contain exactly ONE
```` ```bash ```` block; its command runs (``bash -c``, ``cmd_timeout`` s) and the exit code + output
(head/tail-shortened past ``obs_chars``) come back as the next user message. A reply with no block gets
a format-error message and costs a turn; a reply with several blocks (a whole plan of commands — base
Qwen3.5 does this a lot) has its FIRST block run and is told so. A block containing only ``submit`` ends
the episode — unless a graded file does not exist yet, in which case the model is told to create it
(submitting it would only ever score 0). After ``max_turns`` replies — or when the context window is
full — the episode ends and the files are graded AS THEY ARE, so "ran out of turns / context" costs the
unfinished work, never an otherwise-working solution. Turn-limit, context-exhaustion, no-block,
multi-block, premature-submit, command-timeout, forced-thinking and truncation counts are all logged per
episode and reported per model, because each of them can move a score without any change in coding skill.

Grading never trusts the agent: only the declared solution file(s) are copied out of its container
into a FRESH one together with the hidden tests, which were never inside the agent's container. Resource
isolation (per-container and box-wide caps, no disk writes, no leftover background processes, reaping of
dead runs' containers) is described in ``sandbox.py``.

Benchmarks
----------
- ``lcb``: LiveCodeBench ``code_generation_lite`` release-v6 slice (problems from 2025-01..2025-04),
  easy+medium by default. The statement's examples are given as files; graded on public+private tests
  (all must pass, ``lcb_test_timeout`` s each) by a grader that runs as root and executes the solution as
  an unprivileged user that cannot read the test file.
- ``exercism``: the Exercism Python track's practice exercises (the set behind Aider's original
  code-editing benchmark), stub + instructions only — the unit tests are hidden and run with pytest.
- ``canitedit``: CanItEdit (instructed edits of a Python file, descriptive and lazy instructions as
  separate items, clustered per problem), graded with the dataset's hidden tests.
Exercism and CanItEdit items are validated at load time: the reference solution must pass the grader
and the starting code must fail it; items that do not are dropped and listed in the output.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import pickle
import re
import subprocess
import zlib

import pandas as pd

from monitordecorrelation.capeval.bench import Benchmark, Item, RunCtx, budget_dict, clip_middle, gen_diag
from monitordecorrelation.capeval.policy import Budget, Generation, Policy
from monitordecorrelation.capeval.sandbox import MAX_CAPTURE_BYTES

# ---------------------------------------------------------------------------------------------
# The scaffold
# ---------------------------------------------------------------------------------------------
AGENT_INSTRUCTIONS = """You are an expert software engineer working autonomously in a Linux sandbox: Python 3.11 with pytest, numpy and the standard library, and no internet access. Your working directory is /work.

# Task
{task}

# How to work
Every reply must contain exactly ONE bash code block with the next command to run, like this:
```bash
ls -la
```
You will then see the command's exit code and its output (stdout and stderr). Each command runs in a fresh `bash -c` in /work: shell state (current directory, variables, background processes) does not carry over between commands, but files do. A command is killed after {timeout} seconds, and long output is shortened.

Create and edit files with shell commands, for example with a heredoc:
```bash
cat > example.py <<'EOF'
print("hello")
EOF
```
Test your work before you submit. When you are done, reply with a bash block that contains only the word `submit`; your work is then graded. {graded_note} You can send at most {max_turns} replies (the `submit` one included); if you run out, your files are graded as they are."""

FORMAT_ERROR = ("Format error: your reply must contain exactly one ```bash code block (it contained none). "
                "Reply with a single ```bash block holding the next command, or holding only `submit` "
                "when you are done.{cut}")
MULTI_BLOCK_NOTE = ("<note>Your reply contained {n} bash blocks; only the FIRST one was executed (its result "
                    "is below). Send exactly one command per reply.</note>\n")
PREMATURE_SUBMIT = ("Cannot submit yet: {missing} does not exist. Create it first, then reply with a block "
                    "containing only `submit`.")
LAST_TURNS_NOTE = ("\nNote: you have {n} repl{ies} left (the `submit` one included); after that your files are "
                   "graded as they are.")

_BASH_RE = re.compile(r"```(?:bash|sh|shell)[ \t]*\r?\n(.*?)```", re.S)


def extract_commands(text: str) -> list[str]:
    return [b.strip() for b in _BASH_RE.findall(text or "")]


def format_observation(res, timeout: float, obs_chars: int) -> str:
    out = res.output
    if res.n_bytes > MAX_CAPTURE_BYTES:
        out += f"\n[... the command printed {res.n_bytes} bytes; only the first {MAX_CAPTURE_BYTES} were kept ...]"
    if len(out) > obs_chars:
        half = obs_chars // 2
        out = (f"{out[:half]}\n[... {len(out) - 2 * half} characters elided; the output was too long, "
               f"use head/tail/grep or redirect it to a file ...]\n{out[-half:]}")
    warn = ""
    if res.timed_out:
        warn = f"<warning>The command did not finish within {timeout:g} seconds and was killed.</warning>\n"
    return f"<returncode>{res.returncode}</returncode>\n{warn}<output>\n{out}\n</output>"


async def run_agent(ctx: RunCtx, policy: Policy, *, seed_parts: tuple, task: str, graded_note: str,
                    files: dict[str, str], graded_files: list[str], budget: Budget, max_turns: int,
                    cmd_timeout: float, obs_chars: int, priority: int = 0) -> dict:
    """One episode. Returns ``{"files": {graded name: content|None}, "turns": [...], "gens": [...],
    "end_reason": ...}``."""
    intro = AGENT_INSTRUCTIONS.format(task=task, timeout=f"{cmd_timeout:g}", graded_note=graded_note,
                                      max_turns=max_turns)
    messages = [{"role": "user", "content": intro}]
    turns: list[dict] = []
    gens: list[Generation] = []
    end_reason = "turn_limit"
    async with ctx.sandbox.container(files) as box:
        for t in range(max_turns):
            g = await policy.generate(messages, budget=budget, seed_parts=(*seed_parts, t),
                                      priority=priority)
            gens.append(g)
            if g.context_exhausted:
                end_reason = "context_exhausted"
                break
            turn = {"answer": g.answer, "cot": clip_middle(g.cot, 600, 600), "cot_chars": len(g.cot),
                    "forced": g.forced, "truncated": g.truncated, "gen_tokens": g.gen_tokens}
            cmds = extract_commands(g.answer)
            left = max_turns - t - 1
            # Several blocks in one reply (a plan of commands) is a common habit of some models: the
            # first one runs and the model is told so, rather than the whole turn being wasted.
            note = MULTI_BLOCK_NOTE.format(n=len(cmds)) if len(cmds) > 1 else ""
            if len(cmds) > 1:
                turn["multi_block"] = len(cmds)
            if not cmds:
                turn["format_error"] = True
                obs = FORMAT_ERROR.format(cut=" (Your reply was cut off at the token limit.)" if g.truncated else "")
            elif cmds[0] == "submit":
                missing = [f"/work/{f}" for f in graded_files if await box.read(f) is None]
                if not missing:
                    turn["submit"] = True
                    turns.append(turn)
                    end_reason = "submitted"
                    break
                # Submitting before the graded file exists would only ever score 0 — say so instead.
                turn["premature_submit"] = True
                obs = note + PREMATURE_SUBMIT.format(missing=", ".join(missing))
            else:
                res = await box.exec(cmds[0], cmd_timeout)
                obs = note + format_observation(res, cmd_timeout, obs_chars)
                turn.update(command=cmds[0], returncode=res.returncode, timed_out=res.timed_out,
                            observation=clip_middle(obs, 1500, 1500))
            if 0 < left <= 3:
                obs += LAST_TURNS_NOTE.format(n=left, ies="y" if left == 1 else "ies")
            turns.append(turn)
            messages += [{"role": "assistant", "content": g.answer}, {"role": "user", "content": obs}]
        graded = {name: await box.read(name) for name in graded_files}
    return {"files": graded, "turns": turns, "gens": gens, "end_reason": end_reason}


def agent_diag(ep: dict) -> dict:
    turns = ep["turns"]
    return {
        **gen_diag(ep["gens"]),
        "turns": len(turns),
        "end_reason": ep["end_reason"],
        "submitted": ep["end_reason"] == "submitted",
        "turn_limit": ep["end_reason"] == "turn_limit",
        "format_errors": sum(1 for t in turns if "format_error" in t),
        "multi_blocks": sum(1 for t in turns if "multi_block" in t),
        "premature_submits": sum(1 for t in turns if t.get("premature_submit")),
        "cmd_timeouts": sum(1 for t in turns if t.get("timed_out")),
        "no_solution_file": any(v is None for v in ep["files"].values()),
    }


class _Agentic(Benchmark):
    family = "agentic"
    metrics = ("pass",)
    priority = 0

    def __init__(self, *, n_samples: int, budget: Budget, max_turns: int, cmd_timeout: float,
                 obs_chars: int) -> None:
        self.n_samples = n_samples
        self.budget = budget
        self.max_turns = max_turns
        self.cmd_timeout = cmd_timeout
        self.obs_chars = obs_chars
        self.dropped: dict[str, str] = {}  # item id -> why it failed validation

    def _scaffold_settings(self) -> dict:
        return {"n_samples": self.n_samples, **budget_dict(self.budget), "max_turns": self.max_turns,
                "cmd_timeout": self.cmd_timeout, "obs_chars": self.obs_chars, "scaffold_version": 3}

    async def _episode(self, ctx, policy, item, sample, *, task, graded_note, files, graded_files):
        return await run_agent(ctx, policy, seed_parts=(self.name, item.id, sample), task=task,
                               graded_note=graded_note, files=files, graded_files=graded_files,
                               budget=self.budget, max_turns=self.max_turns, cmd_timeout=self.cmd_timeout,
                               obs_chars=self.obs_chars, priority=self.priority)


class GraderError(RuntimeError):
    """The harness's grader failed (not the model's code). Recorded as an error, never as a score."""


# ---------------------------------------------------------------------------------------------
# LiveCodeBench
# ---------------------------------------------------------------------------------------------
LCB_REPO = "livecodebench/code_generation_lite"
LCB_FILE = "test6.jsonl"  # release_v6 increment: problems dated 2025-01-04 .. 2025-04-06

# The official lcb_runner import prelude for call-based (LeetCode) problems.
LCB_PRELUDE = (
    "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\n"
    "from heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\n"
    "from random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\n"
    "from operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\n"
    "from builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\n"
    "import collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\n"
    "import statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\n"
    "import json\nsys.setrecursionlimit(50000)\n"
)

# Runs one call-based test: the args arrive as JSON on stdin, the result leaves as JSON on stdout after
# a marker line. Executed as the unprivileged user.
LCB_FN_RUNNER = r'''
import json, sys
spec = json.loads(sys.stdin.read())
marker = spec["marker"]
g = {"__name__": "__capeval_solution__"}
exec(compile(spec["prelude"] + "\n" + open("/grade/solution.py").read(), "solution.py", "exec"), g)
args = [json.loads(line) for line in spec["input"].split("\n")]
out = getattr(g["Solution"](), spec["func_name"])(*args)
sys.stdout.flush()
sys.stdout.write("\n" + marker + json.dumps(out) + "\n")
'''

# The grader: reads the (root-only) test file, runs every test in order as uid 1000, stops at the first
# failure, prints one CAPEVAL_RESULT line.
LCB_GRADER = r'''
import json, os, subprocess, sys, uuid, math
mode, func_name, tl = sys.argv[1], sys.argv[2], float(sys.argv[3])
tests = json.load(open("/tests/tests.json"))
prelude = open("/grade/_prelude.py").read()

def demote():
    os.setgid(1000); os.setgroups([]); os.setuid(1000)

def num(s):
    try:
        return float(s)
    except ValueError:
        return None

def close(a, b):
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)

def tokens_match(got, exp):
    g, e = got.split(), exp.split()
    if len(g) != len(e):
        return False
    for x, y in zip(g, e):
        if x == y:
            continue
        fx, fy = num(x), num(y)
        if fx is None or fy is None or not close(fx, fy):
            return False
    return True

def norm(v):
    if isinstance(v, tuple):
        v = list(v)
    if isinstance(v, list):
        return [norm(x) for x in v]
    if isinstance(v, dict):
        return {k: norm(x) for k, x in v.items()}
    return v

def values_match(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b or close(float(a), float(b))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(values_match(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(values_match(a[k], b[k]) for k in a)
    return a == b

passed, fail = 0, None
for i, t in enumerate(tests):
    if mode == "stdin":
        argv, stdin = [sys.executable, "/grade/solution.py"], t["input"]
    else:
        marker = "CAPEVAL_OUT_" + uuid.uuid4().hex + ":"
        argv = [sys.executable, "/grade/_fn_runner.py"]
        stdin = json.dumps({"marker": marker, "prelude": prelude, "input": t["input"], "func_name": func_name})
    try:
        p = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=tl,
                           preexec_fn=demote, cwd="/tmp")
    except subprocess.TimeoutExpired:
        fail = {"i": i, "reason": "timeout"}
        break
    if p.returncode != 0:
        fail = {"i": i, "reason": "runtime_error", "detail": p.stderr[-800:]}
        break
    if mode == "stdin":
        ok = tokens_match(p.stdout, t["output"])
    else:
        idx = p.stdout.rfind(marker)
        if idx < 0:
            fail = {"i": i, "reason": "no_result"}
            break
        try:
            got = norm(json.loads(p.stdout[idx + len(marker):].strip().splitlines()[0]))
        except Exception as e:
            fail = {"i": i, "reason": "unserializable_result", "detail": repr(e)[:300]}
            break
        ok = values_match(got, norm(json.loads(t["output"])))
    if not ok:
        fail = {"i": i, "reason": "wrong_answer"}
        break
    passed += 1
print("CAPEVAL_RESULT " + json.dumps({"passed": passed, "total": len(tests), "fail": fail}))
'''


class _NoGlobalsUnpickler(pickle.Unpickler):
    """LCB ships private tests as base64(zlib(pickle(json str))). The payload is a plain str, so refuse
    every global — unpickling cannot run code."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing global {module}.{name}")


def _decode_lcb_tests(s: str) -> list[dict]:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        raw = _NoGlobalsUnpickler(io.BytesIO(zlib.decompress(base64.b64decode(s.encode())))).load()
        return json.loads(raw)


class LiveCodeBench(_Agentic):
    name = "lcb"

    def __init__(self, *, difficulties: tuple[str, ...], test_timeout: float, **kw) -> None:
        super().__init__(**kw)
        self.difficulties = tuple(difficulties)
        self.test_timeout = test_timeout

    def settings(self) -> dict:
        return {**self._scaffold_settings(), "file": LCB_FILE, "difficulties": list(self.difficulties),
                "test_timeout": self.test_timeout, "grader_version": 2}

    def load(self, cache_dir: str) -> list[Item]:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(LCB_REPO, LCB_FILE, repo_type="dataset")
        tests_root = os.path.join(cache_dir, "lcb_tests")
        items = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r["difficulty"] not in self.difficulties:
                    continue
                qid = r["question_id"]
                public = json.loads(r["public_test_cases"])
                tdir = os.path.join(tests_root, qid)
                tpath = os.path.join(tdir, "tests.json")
                if not os.path.exists(tpath):
                    private = _decode_lcb_tests(r["private_test_cases"])
                    os.makedirs(tdir, exist_ok=True)
                    tmp = tpath + ".tmp"
                    with open(tmp, "w") as tf:
                        json.dump([{"input": t["input"], "output": t["output"]} for t in public + private], tf)
                    os.chmod(tmp, 0o600)  # root-only inside the grading container
                    os.replace(tmp, tpath)
                meta = json.loads(r["metadata"] or "{}")
                items.append(Item(
                    id=qid, cluster=qid, group=f"{r['platform']}/{r['difficulty']}",
                    data={"title": r["question_title"], "content": r["question_content"],
                          "starter": r["starter_code"], "func_name": meta.get("func_name", ""),
                          "public": public, "tests_dir": tdir, "n_public": len(public)}))
        return items

    def _task(self, item: Item) -> tuple[str, dict[str, str]]:
        d = item.data
        files: dict[str, str] = {}
        if d["starter"]:
            files["examples/examples.json"] = json.dumps(
                [{"input": t["input"], "output": t["output"]} for t in d["public"]], indent=1)
            task = (
                f"Solve this programming problem.\n\n## {d['title']}\n\n{d['content']}\n\n"
                f"Write your solution to `/work/solution.py` as a class `Solution` with this method "
                f"(starter code):\n```python\n{d['starter'].rstrip()}\n```\n"
                f"It will be called as `Solution().{d['func_name']}(...)` on hidden test cases; before "
                "your code the grader prepends `from typing import *` and imports of common "
                "standard-library modules. The examples from the statement are in "
                "`/work/examples/examples.json` (each `input` holds the arguments, one JSON value per line; "
                "`output` is the expected return value as JSON)."
            )
        else:
            for i, t in enumerate(d["public"], 1):
                files[f"examples/{i}.in"] = t["input"]
                files[f"examples/{i}.out"] = t["output"]
            task = (
                f"Solve this competitive programming problem.\n\n## {d['title']}\n\n{d['content']}\n\n"
                "Write your solution as a complete Python 3 program in `/work/solution.py`: it must read "
                "the input from standard input and print the answer to standard output. The examples from "
                "the statement are in `/work/examples/` (`1.in` / `1.out`, ...)."
            )
        task += (f"\n\nYour program is judged on hidden tests with a {self.test_timeout:g}-second time "
                 "limit per test, so it must be correct and efficient.")
        return task, files

    async def grade(self, ctx: RunCtx, item: Item, solution: str | None) -> dict:
        if solution is None:
            return {"pass": 0.0, "passed_public": False, "fail": {"reason": "no_solution_file"}}
        d = item.data
        mode = "functional" if d["starter"] else "stdin"
        n_tests = d.get("n_tests")
        if n_tests is None:
            with open(os.path.join(d["tests_dir"], "tests.json")) as f:
                n_tests = d["n_tests"] = len(json.load(f))
        timeout = min(1800.0, n_tests * (self.test_timeout + 1.5) + 60)
        # As root: only root can read the test file; the grader demotes every solution run to uid 1000.
        res = await ctx.sandbox.run_once(
            {"solution.py": solution, "_grader.py": LCB_GRADER, "_fn_runner.py": LCB_FN_RUNNER,
             "_prelude.py": LCB_PRELUDE},
            ["python3", "/grade/_grader.py", mode, d["func_name"] or "-", f"{self.test_timeout:g}"],
            timeout=timeout, ro_mounts={d["tests_dir"]: "/tests"}, as_root=True,
        )
        line = next((ln for ln in res.output.splitlines()[::-1] if ln.startswith("CAPEVAL_RESULT ")), None)
        if line is None:
            if res.timed_out:  # the whole grading run exceeded its cap: the solution is too slow
                return {"pass": 0.0, "passed_public": False, "fail": {"reason": "grading_timeout"}}
            raise GraderError(f"LCB grader produced no result (rc={res.returncode}): {res.output[-1500:]}")
        r = json.loads(line[len("CAPEVAL_RESULT "):])
        return {"pass": float(r["fail"] is None and r["passed"] == r["total"]),
                "passed_public": r["passed"] >= d["n_public"], "tests_passed": r["passed"],
                "tests_total": r["total"], "fail": r["fail"]}

    async def run(self, ctx: RunCtx, policy: Policy, item: Item, sample: int) -> dict:
        task, files = self._task(item)
        ep = await self._episode(ctx, policy, item, sample, task=task, files=files,
                                 graded_files=["solution.py"],
                                 graded_note="Only `/work/solution.py` is graded.")
        g = await self.grade(ctx, item, ep["files"]["solution.py"])
        return {"scores": {"pass": g["pass"]},
                "diag": {**agent_diag(ep), "passed_public": g["passed_public"],
                         "public_pass_hidden_fail": bool(g["passed_public"] and not g["pass"])},
                "transcript": {"turns": ep["turns"], "grade": g,
                               "solution": clip_middle(ep["files"]["solution.py"] or "", 6000, 2000)}}


# ---------------------------------------------------------------------------------------------
# Exercism (Python track)
# ---------------------------------------------------------------------------------------------
EXERCISM_GIT = "https://github.com/exercism/python.git"
EXERCISM_COMMIT = "6105f6d9d3ef66ce3d5c8f529d2882af3d76b706"  # 2026-09-30


def _ensure_exercism(cache_dir: str) -> str:
    root = os.path.join(cache_dir, f"exercism-python-{EXERCISM_COMMIT[:12]}")
    if os.path.isdir(os.path.join(root, "exercises")):
        return root
    tmp = root + ".tmp"
    subprocess.run(["rm", "-rf", tmp], check=True)
    os.makedirs(tmp)
    for cmd in (["git", "init", "-q"], ["git", "remote", "add", "origin", EXERCISM_GIT],
                ["git", "fetch", "-q", "--depth", "1", "origin", EXERCISM_COMMIT],
                ["git", "checkout", "-q", "FETCH_HEAD"]):
        subprocess.run(cmd, cwd=tmp, check=True, timeout=600)
    os.replace(tmp, root)
    return root


def _pytest_counts(out: str) -> dict:
    counts = {}
    for n, what in re.findall(r"(\d+) (passed|failed|error|errors|skipped)", out):
        counts[what.rstrip("s") if what == "errors" else what] = int(n)
    return counts


class Exercism(_Agentic):
    name = "exercism"

    def settings(self) -> dict:
        return {**self._scaffold_settings(), "commit": EXERCISM_COMMIT, "grader_version": 3}

    def load(self, cache_dir: str) -> list[Item]:
        root = os.path.join(_ensure_exercism(cache_dir), "exercises", "practice")
        items = []
        for slug in sorted(os.listdir(root)):
            ex = os.path.join(root, slug)
            try:
                cfg = json.load(open(os.path.join(ex, ".meta", "config.json")))
            except (OSError, json.JSONDecodeError):
                self.dropped[slug] = "no .meta/config.json"
                continue
            files = cfg.get("files", {})
            sol, tests, example = files.get("solution", []), files.get("test", []), files.get("example", [])
            if len(sol) != 1 or len(example) != 1 or not tests:
                self.dropped[slug] = f"unsupported layout (solution={sol}, example={example}, test={tests})"
                continue
            docs = []
            for name in ("introduction.md", "instructions.md", "instructions.append.md"):
                p = os.path.join(ex, ".docs", name)
                if os.path.exists(p):
                    docs.append(open(p).read().strip())
            read = lambda rel: open(os.path.join(ex, rel)).read()  # noqa: E731
            editor = files.get("editor", [])
            # Other files next to the tests (e.g. error-handling's test_utils.py) support the tests:
            # given to the grader only.
            support = [f for f in sorted(os.listdir(ex)) if os.path.isfile(os.path.join(ex, f))
                       and not f.startswith(".") and f not in {*sol, *tests, *editor}]
            items.append(Item(id=slug, cluster=slug, group="all", data={
                "slug": slug, "stub_name": sol[0], "stub": read(sol[0]), "example": read(example[0]),
                "tests": {t: read(t) for t in tests},
                "support": {f: read(f) for f in support},
                "editor": {e: read(e) for e in editor},
                "instructions": "\n\n".join(docs)}))
        return items

    async def validate(self, ctx: RunCtx, items: list[Item]) -> list[Item]:
        async def check(it: Item):
            ref = await self.grade(ctx, it, it.data["example"])
            stub = await self.grade(ctx, it, it.data["stub"])
            return it, ref, stub
        keep = []
        for it, ref, stub in await asyncio.gather(*[check(it) for it in items]):
            if not ref["pass"]:
                self.dropped[it.id] = f"reference solution fails the grader: {ref['output_tail'][-300:]}"
            elif stub["pass"]:
                self.dropped[it.id] = "the unmodified stub already passes"
            else:
                keep.append(it)
        return keep

    async def grade(self, ctx: RunCtx, item: Item, solution: str | None) -> dict:
        if solution is None:
            return {"pass": 0.0, "counts": {}, "output_tail": "no solution file"}
        d = item.data
        files = {**d["editor"], **d["support"], **d["tests"], d["stub_name"]: solution}
        res = await ctx.sandbox.run_once(
            files, ["python3", "-m", "pytest", "-q", "-p", "no:cacheprovider", *sorted(d["tests"])],
            timeout=180)
        counts = _pytest_counts(res.output)
        ok = res.returncode == 0 and counts.get("passed", 0) > 0
        return {"pass": float(ok), "counts": counts, "timed_out": res.timed_out,
                "output_tail": res.output[-1200:]}

    async def run(self, ctx: RunCtx, policy: Policy, item: Item, sample: int) -> dict:
        d = item.data
        module = d["stub_name"][:-3] if d["stub_name"].endswith(".py") else d["stub_name"]
        task = (f"Solve this Exercism exercise in Python.\n\n{d['instructions']}\n\n"
                f"Put your solution in `/work/{d['stub_name']}`, which currently contains this stub:\n"
                f"```python\n{d['stub'].rstrip()}\n```\n"
                f"Hidden unit tests (pytest) will import your module (`from {module} import ...`) and check "
                "the behaviour described above, including exception types and messages where the "
                "instructions specify them. No tests are provided: write and run your own checks. Keep the "
                "names from the stub.")
        files = {d["stub_name"]: d["stub"], **d["editor"]}
        ep = await self._episode(ctx, policy, item, sample, task=task, files=files,
                                 graded_files=[d["stub_name"]],
                                 graded_note=f"Only `/work/{d['stub_name']}` is graded.")
        g = await self.grade(ctx, item, ep["files"][d["stub_name"]])
        c = g["counts"]
        frac = c.get("passed", 0) / max(1, c.get("passed", 0) + c.get("failed", 0) + c.get("error", 0))
        return {"scores": {"pass": g["pass"]},
                "diag": {**agent_diag(ep), "test_frac": frac},
                "transcript": {"turns": ep["turns"], "grade": g,
                               "solution": clip_middle(ep["files"][d["stub_name"]] or "", 6000, 2000)}}


# ---------------------------------------------------------------------------------------------
# CanItEdit
# ---------------------------------------------------------------------------------------------
_TIMING_TEST_RE = re.compile(r"timeit|perf_counter|time\.time\(|process_time|monotonic\(")


class CanItEdit(_Agentic):
    name = "canitedit"

    def settings(self) -> dict:
        return {**self._scaffold_settings(), "kinds": ["descriptive", "lazy"], "grader_version": 3}

    def load(self, cache_dir: str) -> list[Item]:
        from huggingface_hub import hf_hub_download

        df = pd.read_parquet(hf_hub_download("nuprl/CanItEdit", "data/test-00000-of-00001.parquet",
                                             repo_type="dataset"))
        items = []
        for r in df.itertuples():
            if _TIMING_TEST_RE.search(r.tests):
                # e.g. 60_unique_number asserts the new code beats the old by 10% under timeit: the
                # verdict depends on machine load (seen flipping between runs), not only on the code.
                for kind in ("descriptive", "lazy"):
                    self.dropped[f"{r.full_name}/{kind}"] = "timing-based test (verdict depends on machine load)"
                continue
            for kind in ("descriptive", "lazy"):
                items.append(Item(id=f"{r.full_name}/{kind}", cluster=str(r.full_name), group=kind, data={
                    "before": r.before, "after": r.after, "tests": r.tests,
                    "instruction": getattr(r, f"instruction_{kind}")}))
        return items

    async def validate(self, ctx: RunCtx, items: list[Item]) -> list[Item]:
        # Same code for both instruction kinds: grade each problem once.
        probs = {}
        for it in items:
            probs.setdefault(it.cluster, it)

        async def check(it: Item):
            return it.cluster, await self.grade(ctx, it, it.data["after"]), await self.grade(ctx, it, it.data["before"])
        verdict = {}
        for cl, ref, before in await asyncio.gather(*[check(it) for it in probs.values()]):
            if not ref["pass"]:
                verdict[cl] = f"reference solution fails the grader: {ref['output_tail'][-300:]}"
            elif before["pass"]:
                verdict[cl] = "the unedited code already passes"
        keep = []
        for it in items:
            if it.cluster in verdict:
                self.dropped[it.id] = verdict[it.cluster]
            else:
                keep.append(it)
        return keep

    async def grade(self, ctx: RunCtx, item: Item, code: str | None) -> dict:
        if code is None:
            return {"pass": 0.0, "output_tail": "no main.py"}
        res = await ctx.sandbox.run_once(
            {"main.py": code, "_grade.py": code + "\n\n" + item.data["tests"]},
            ["python3", "/grade/_grade.py"], timeout=180)
        return {"pass": float(res.returncode == 0), "timed_out": res.timed_out,
                "output_tail": res.output[-1200:]}

    async def run(self, ctx: RunCtx, policy: Policy, item: Item, sample: int) -> dict:
        d = item.data
        task = (f"Edit the Python code in `/work/main.py` according to this instruction:\n\n"
                f"{d['instruction']}\n\nCurrent contents of `/work/main.py`:\n```python\n{d['before'].rstrip()}\n```\n"
                "Hidden tests will check the edited code; keep the rest of the existing interface working.")
        ep = await self._episode(ctx, policy, item, sample, task=task, files={"main.py": d["before"]},
                                 graded_files=["main.py"], graded_note="Only `/work/main.py` is graded.")
        g = await self.grade(ctx, item, ep["files"]["main.py"])
        return {"scores": {"pass": g["pass"]}, "diag": agent_diag(ep),
                "transcript": {"turns": ep["turns"], "grade": g,
                               "solution": clip_middle(ep["files"]["main.py"] or "", 6000, 2000)}}
