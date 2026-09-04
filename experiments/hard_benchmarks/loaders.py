"""Benchmark loaders -> list[Problem] (LiveCodeBench format) + hardest-N selection by difficulty label.

Difficulty ranks: larger = harder. Ties (coarse labels like easy/medium/hard) are broken by a FIXED
seeded shuffle, never by model performance, so "hardest 512" of a benchmark with 700 'hard' problems
is a random 512 of those hard problems.
"""
from __future__ import annotations

import base64
import json
import os
import pickle
import random
import zlib

from grader import Problem

os.environ.setdefault("HF_HOME", "/root/hf_cache")


def select_hardest(problems: list[Problem], n: int, seed: int = 12345) -> list[Problem]:
    ranked = [p for p in problems if p.difficulty_rank is not None]
    if not ranked:
        raise ValueError("benchmark has no difficulty labels; use --subset all")
    rng = random.Random(seed)
    order = list(range(len(ranked)))
    rng.shuffle(order)  # tie-breaker
    order.sort(key=lambda i: -ranked[i].difficulty_rank)
    return [ranked[i] for i in order[:n]]


# ---------------------------------------------------------------------------------------------
# LiveCodeBench (livecodebench/code_generation_lite)
# ---------------------------------------------------------------------------------------------
_LCB_RANK = {"easy": 0.0, "medium": 1.0, "hard": 2.0}


def _lcb_tests(row) -> list[dict]:
    pub = json.loads(row["public_test_cases"])
    try:
        priv = json.loads(row["private_test_cases"])
    except Exception:
        priv = json.loads(pickle.loads(zlib.decompress(base64.b64decode(row["private_test_cases"].encode()))))
    return pub + priv


_LCB_FILES = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]


def _lcb_rows():
    """All rows of livecodebench/code_generation_lite (the loading script is unsupported by datasets 5,
    so read the release jsonl shards directly; release_vN = shards 1..N)."""
    from huggingface_hub import hf_hub_download
    for f in _LCB_FILES:
        path = hf_hub_download("livecodebench/code_generation_lite", f, repo_type="dataset")
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def load_lcb(date_from: str | None = None, date_to: str | None = None) -> list[Problem]:
    out = []
    for row in _lcb_rows():
        d = str(row["contest_date"])[:10]
        if date_from and d < date_from:
            continue
        if date_to and d >= date_to:
            continue
        tests = _lcb_tests(row)
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        fn = meta.get("func_name")
        kinds = {t.get("testtype") for t in tests}
        call_based = "functional" in kinds
        out.append(Problem(
            benchmark="lcb", task_id=row["question_id"], question=row["question_content"],
            starter_code=row["starter_code"] or "", fn_name=fn if call_based else None,
            inputs=[t["input"] for t in tests], outputs=[t["output"] for t in tests],
            difficulty=row["difficulty"], difficulty_rank=_LCB_RANK[row["difficulty"]],
            meta={"platform": str(row["platform"]), "contest_date": d},
        ))
    return out


LOADERS = {
    "lcb_all": lambda: load_lcb(),
}


# ---------------------------------------------------------------------------------------------
# open-r1/codeforces (verifiable config, train + test): rating = difficulty
# ---------------------------------------------------------------------------------------------
MAX_TESTS_PER_PROBLEM = 60  # examples + official first, then generated tests, capped


def _cf_question(row) -> str:
    q = row["description"] or ""
    if row.get("input_format"):
        q += "\n\nInput\n\n" + row["input_format"]
    if row.get("output_format"):
        q += "\n\nOutput\n\n" + row["output_format"]
    exs = row.get("examples") or []
    if exs:
        q += "\n\nExamples\n"
        for e in exs:
            q += "\n\nInput\n\n" + e["input"].rstrip("\n") + "\n\nOutput\n\n" + e["output"].rstrip("\n")
    if row.get("note"):
        q += "\n\nNote\n\n" + row["note"]
    return q


def _norm(s: str) -> str:
    return s.replace("\r\n", "\n")


def load_codeforces(splits=("train", "test")) -> list[Problem]:
    """Problem list WITHOUT generated tests (those are fetched lazily per contest by
    ``attach_codeforces_generated_tests`` for the sampled problems only — the full set is 115 GB)."""
    from datasets import load_dataset
    out = []
    for split in splits:
        ds = load_dataset("open-r1/codeforces", "verifiable", split=split)
        for row in ds:
            if row["rating"] is None or row["interaction_format"] or row["input_mode"] != "stdio":
                continue
            tests = [(_norm(e["input"]), _norm(e["output"])) for e in (row["examples"] or [])]
            tests += [(_norm(t["input"]), _norm(t["output"])) for t in (row["official_tests"] or [])]
            # keep only problems whose hidden tests are actually available (complete official set,
            # or a generated-tests file) — grading on the statement's examples alone is too lenient
            if not (row["official_tests_complete"] or (row["generated_tests"] or 0) > 0):
                continue
            checker = ({"kind": "open_r1_python", "src": row["generated_checker"]}
                       if row["generated_checker"] else None)
            tl = float(row["time_limit"] or 1.0)
            out.append(Problem(
                benchmark="codeforces", task_id=row["id"], question=_cf_question(row), starter_code="",
                fn_name=None, inputs=[t[0] for t in tests], outputs=[t[1] for t in tests],
                difficulty=str(row["rating"]), difficulty_rank=float(row["rating"]),
                checker=checker, timeout=max(6.0, 3.0 * tl),
                meta={"split": split, "contest_id": row["contest_id"], "year": row["contest_start_year"],
                      "n_generated": int(row["generated_tests"] or 0), "time_limit": tl},
            ))
    return out


def attach_codeforces_generated_tests(problems: list[Problem]) -> None:
    """Download each sampled problem's contest parquet and append its generated tests (capped)."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    by_contest: dict[str, list[Problem]] = {}
    for p in problems:
        if p.meta.get("n_generated", 0) > 0:
            by_contest.setdefault(p.meta["contest_id"], []).append(p)
    for cid, ps in by_contest.items():
        try:
            path = hf_hub_download("open-r1/codeforces", f"generated_tests/test_cases_{cid}.parquet",
                                   repo_type="dataset")
        except Exception as e:  # noqa: BLE001
            print(f"[codeforces] no generated tests file for contest {cid}: {e}", flush=True)
            continue
        t = pq.read_table(path, columns=["problem_id", "input", "output", "test_i"]).to_pylist()
        for p in ps:
            rows = sorted((r for r in t if r["problem_id"] == p.task_id), key=lambda r: int(r["test_i"]))
            seen = set(zip(p.inputs, p.outputs))
            for r in rows:
                if len(p.inputs) >= MAX_TESTS_PER_PROBLEM:
                    break
                pair = (_norm(r["input"]), _norm(r["output"]))
                if pair in seen:
                    continue
                seen.add(pair)
                p.inputs.append(pair[0])
                p.outputs.append(pair[1])
            p.meta["n_tests_used"] = len(p.inputs)


LOADERS["codeforces"] = load_codeforces
POST_SELECT_HOOKS = {"codeforces": attach_codeforces_generated_tests}


# ---------------------------------------------------------------------------------------------
# AetherCode v1_2024 (m-a-p/AetherCode): Easy < Medium < Hard < Extremely Hard; testlib checkers
# ---------------------------------------------------------------------------------------------
_AETHER_RANK = {"Easy": 0.0, "Medium": 1.0, "Hard": 2.0, "Extremely Hard": 3.0}
AETHER_MAX_TESTS = 100


def _aether_shards():
    import glob
    fs = sorted(glob.glob("/root/hf_cache/hub/datasets--m-a-p--AetherCode/snapshots/*/v1_2024/*.parquet"))
    if not fs:
        from huggingface_hub import snapshot_download
        snapshot_download("m-a-p/AetherCode", repo_type="dataset", allow_patterns=["v1_2024/*"])
        fs = sorted(glob.glob("/root/hf_cache/hub/datasets--m-a-p--AetherCode/snapshots/*/v1_2024/*.parquet"))
    return fs


def load_aethercode() -> list[Problem]:
    """Small columns only; ``test_cases`` (12 GB across the set) are attached per sampled problem."""
    import pyarrow.parquet as pq
    out = []
    for f in _aether_shards():
        t = pq.read_table(f, columns=["id", "description", "time_limit", "memory_limit", "checker", "year",
                                      "date", "difficulty", "contest_category", "contest_name"])
        for i, row in enumerate(t.to_pylist()):
            tl = float(row["time_limit"] or 1000) / 1000.0
            checker = {"kind": "testlib", "src": row["checker"]} if row["checker"] else None
            out.append(Problem(
                benchmark="aethercode", task_id=str(row["id"]), question=row["description"], starter_code="",
                fn_name=None, inputs=[], outputs=[], difficulty=row["difficulty"],
                difficulty_rank=_AETHER_RANK[row["difficulty"]], checker=checker, timeout=max(6.0, 3.0 * tl),
                meta={"shard": f, "row": i, "date": row["date"], "contest": row["contest_name"],
                      "category": row["contest_category"], "time_limit": tl},
            ))
    return out


def attach_aethercode_tests(problems: list[Problem]) -> None:
    """Read ``test_cases`` row by row (pyarrow cannot convert this nested column as one chunked array)."""
    import pyarrow.parquet as pq
    by_shard: dict[str, dict[int, Problem]] = {}
    for p in problems:
        by_shard.setdefault(p.meta["shard"], {})[p.meta["row"]] = p
    for f, rows in by_shard.items():
        pf = pq.ParquetFile(f)
        i = 0
        for batch in pf.iter_batches(batch_size=1, columns=["test_cases"]):
            if i in rows:
                tests = batch.column(0)[0].as_py()
                p = rows[i]
                p.inputs = [t["input"] for t in tests[:AETHER_MAX_TESTS]]
                p.outputs = [t["output"] for t in tests[:AETHER_MAX_TESTS]]
                p.meta["n_tests_total"] = len(tests)
            i += 1


LOADERS["aethercode"] = load_aethercode
POST_SELECT_HOOKS["aethercode"] = attach_aethercode_tests


# ---------------------------------------------------------------------------------------------
# TACO test split (BAAI/TACO): EASY < MEDIUM < MEDIUM_HARD < HARD < VERY_HARD (200 each)
# ---------------------------------------------------------------------------------------------
_TACO_RANK = {"EASY": 0.0, "MEDIUM": 1.0, "MEDIUM_HARD": 2.0, "HARD": 3.0, "VERY_HARD": 4.0}
TACO_MAX_TESTS = 60


def load_taco() -> list[Problem]:
    """stdin/stdout problems only (the 55 call-based ``fn_name`` rows, all codewars-style, are dropped)."""
    import glob
    import pyarrow.parquet as pq
    f = glob.glob("/root/hf_cache/hub/datasets--BAAI--TACO/snapshots/*/ALL/test-*.parquet")[0]
    t = pq.read_table(f, columns=["question", "starter_code", "input_output", "difficulty", "source", "url",
                                  "time_limit", "date"])
    out = []
    for i, row in enumerate(t.to_pylist()):
        io = json.loads(row["input_output"])
        if io.get("fn_name"):
            continue
        ins, outs = io["inputs"][:TACO_MAX_TESTS], io["outputs"][:TACO_MAX_TESTS]
        if not ins or not all(isinstance(x, str) for x in ins + outs):
            continue
        out.append(Problem(
            benchmark="taco", task_id=f"taco_test_{i}", question=row["question"], starter_code="",
            fn_name=None, inputs=ins, outputs=outs, difficulty=row["difficulty"],
            difficulty_rank=_TACO_RANK[row["difficulty"]],
            meta={"source": row["source"], "url": row["url"], "date": row["date"], "n_tests_total": len(io["inputs"])},
        ))
    return out


LOADERS["taco"] = load_taco


# ---------------------------------------------------------------------------------------------
# USACO (codegenning/usacobench_formatted): bronze < silver < gold < platinum; official tests
# ---------------------------------------------------------------------------------------------
_USACO_RANK = {"bronze": 0.0, "silver": 1.0, "gold": 2.0, "platinum": 3.0}
USACO_MAX_TESTS = 60


def load_usaco() -> list[Problem]:
    import glob
    import pyarrow.parquet as pq
    out = []
    for f in sorted(glob.glob("/root/hf_cache/hub/datasets--codegenning--usacobench_formatted/snapshots/*/data/*.parquet")):
        for row in pq.read_table(f).to_pylist():
            io = json.loads(row["input_output"])
            ins, outs = io["inputs"][:USACO_MAX_TESTS], io["outputs"][:USACO_MAX_TESTS]
            if not ins:
                continue
            out.append(Problem(
                benchmark="usaco", task_id=row["id"], question=row["question"].strip(), starter_code="",
                fn_name=None, inputs=ins, outputs=outs, difficulty=row["difficulty"],
                difficulty_rank=_USACO_RANK[row["difficulty"]], timeout=8.0,
                meta={"n_tests_total": len(io["inputs"])},
            ))
    return out


LOADERS["usaco"] = load_usaco


# ---------------------------------------------------------------------------------------------
# OJBench (He-Ren/OJBench_testdata): NOI + ICPC, easy < medium < hard; DMOJ-style test archives
# ---------------------------------------------------------------------------------------------
_OJ_RANK = {"easy": 0.0, "medium": 1.0, "hard": 2.0}
OJ_MAX_TESTS = 60


def _oj_root() -> str:
    import glob
    fs = glob.glob("/root/hf_cache/hub/datasets--He-Ren--OJBench_testdata/snapshots/*")
    if not fs:
        from huggingface_hub import snapshot_download
        snapshot_download("He-Ren/OJBench_testdata", repo_type="dataset")
        fs = glob.glob("/root/hf_cache/hub/datasets--He-Ren--OJBench_testdata/snapshots/*")
    return fs[0]


def _oj_question(prompt: str) -> str:
    """The benchmark's prompt already carries an LCB-style '### Format' tail; keep only the statement."""
    cut = prompt.find("\n### Format")
    body = prompt[:cut] if cut > 0 else prompt
    if body.startswith("### Problem Description\n"):
        body = body[len("### Problem Description\n"):]
    return body.strip()


def load_ojbench() -> list[Problem]:
    """Problems whose tests can be judged here: exact / float-tolerance / testlib checkers. Interactive
    problems and problems with Kattis-style output validators (no reference answer to compare against)
    are dropped, loudly."""
    import io
    import zipfile

    import yaml
    root = _oj_root()
    rows = [json.loads(l) for l in open(os.path.join(root, "prompts", "full.jsonl")) if l.strip()]
    rows = [r for r in rows if r["language"] == "python"]
    out, dropped = [], []
    for r in rows:
        d = os.path.join(root, "NOI", f"loj-{r['id']}") if r["dataset"] == "NOI" else os.path.join(root, "ICPC", str(r["id"]))
        if not os.path.isdir(d):
            dropped.append((r["id"], "no test dir"))
            continue
        cfg = yaml.safe_load(open(os.path.join(d, "init.yml")))
        if cfg.get("interactive"):
            dropped.append((r["id"], "interactive"))
            continue
        checker = None
        ck = cfg.get("checker")
        if ck is not None:
            name = ck.get("name") if isinstance(ck, dict) else ck
            if name == "floats":
                checker = {"kind": "floats", "precision": int(ck.get("args", {}).get("precision", 6))}
            elif name == "bridged":
                files = ck["args"]["files"]
                srcs = {f: open(os.path.join(d, f), encoding="utf-8", errors="replace").read() for f in files}
                if any(f.endswith("testlib.h") for f in files):
                    main = next(f for f in files if f.endswith((".cpp", ".cc")) and "testlib" not in f)
                    checker = {"kind": "testlib", "src": srcs[main]}
                else:
                    dropped.append((r["id"], "kattis-style output validator"))
                    continue
            else:
                dropped.append((r["id"], f"unknown checker {name}"))
                continue
        z = zipfile.ZipFile(os.path.join(d, cfg["archive"]))
        names = set(z.namelist())
        ins, outs = [], []
        for tc in cfg["test_cases"][:OJ_MAX_TESTS]:
            if tc["in"] not in names or tc["out"] not in names:
                continue
            ins.append(z.read(tc["in"]).decode("utf-8", errors="replace"))
            outs.append(z.read(tc["out"]).decode("utf-8", errors="replace"))
        if not ins:
            dropped.append((r["id"], "no tests"))
            continue
        out.append(Problem(
            benchmark="ojbench", task_id=f"{r['dataset']}_{r['id']}", question=_oj_question(r["prompt"]),
            starter_code="", fn_name=None, inputs=ins, outputs=outs, difficulty=r["difficulty"],
            difficulty_rank=_OJ_RANK[r["difficulty"]], checker=checker, timeout=10.0,
            meta={"dataset": r["dataset"], "n_tests_total": len(cfg["test_cases"])},
        ))
    if dropped:
        print(f"[ojbench] dropped {len(dropped)}/{len(rows)}: {dropped}", flush=True)
    return out


LOADERS["ojbench"] = load_ojbench


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def _snap(repo: str) -> str:
    import glob
    hits = glob.glob(f"/root/hf_cache/hub/datasets--{repo.replace('/', '--')}/snapshots/*")
    if not hits:
        raise FileNotFoundError(f"{repo} is not in the HF cache; download it first")
    return hits[0]


def _parquet_rows(pattern: str, columns=None):
    import glob
    import pyarrow.parquet as pq
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=4, columns=columns):  # small batches: nested test columns
            yield from batch.to_pylist()


def _nl(s: str) -> str:
    return (s or "").replace("\r\n", "\n")


# Checkers that accepted an EMPTY output in the self-test (scratch selfcheck): unsafe as graders.
LENIENT_CHECKER_TASKS = {("aethercode", "60179"), ("icpc_eval", "55")}


def _checker_bin(src: str, kind: str) -> dict | None:
    from compile_checkers import checker_path
    path = checker_path(src)
    if not os.path.exists(path):
        return {"kind": "MISSING"}  # checker failed to compile: the caller drops the problem
    return {"kind": kind, "bin": path}


# ---------------------------------------------------------------------------------------------
# AetherCode v1_2024 (m-a-p/AetherCode): ICPC/OI problems, stdin, testlib checkers
# ---------------------------------------------------------------------------------------------
_AETHER_RANK = {"Easy": 0.0, "Medium": 1.0, "Hard": 2.0, "Extremely Hard": 3.0}


def load_aethercode() -> list[Problem]:
    out = []
    for r in _parquet_rows(_snap("m-a-p/AetherCode") + "/v1_2024/*.parquet"):
        tests = r["test_cases"] or []
        if not tests:
            continue
        chk = _checker_bin(r["checker"], "testlib") if (r["checker"] or "").strip() else None
        if ("aethercode", str(r["id"])) in LENIENT_CHECKER_TASKS:
            continue
        if chk and chk["kind"] == "MISSING":
            print(f"[aethercode] dropping {r['id']}: its checker did not compile", flush=True)
            continue
        out.append(Problem(
            benchmark="aethercode", task_id=str(r["id"]), question=_nl(r["description"]), starter_code="",
            fn_name=None, inputs=[_nl(t["input"]) for t in tests], outputs=[_nl(t["output"]) for t in tests],
            difficulty=r["difficulty"], difficulty_rank=_AETHER_RANK[r["difficulty"]],
            time_limit=(r["time_limit"] or 1000) / 1000.0, checker=chk,
            meta={"date": r["date"], "contest": r["contest_name"], "category": r["contest_category"]},
        ))
    return out


# ---------------------------------------------------------------------------------------------
# ICPC-Eval (RUC-AIBOX/ICPC-Eval): 2024 regionals + 2023 WF, stdin, 12 spj; no difficulty labels
# ---------------------------------------------------------------------------------------------
def _compose_statement(desc, inp, outp, examples, note) -> str:
    s = _nl(desc).strip()
    if inp:
        s += "\n\nInput\n\n" + _nl(inp).strip()
    if outp:
        s += "\n\nOutput\n\n" + _nl(outp).strip()
    for i, (ei, eo) in enumerate(examples or []):
        s += f"\n\nExample {i + 1}\n\nInput\n{_nl(ei).rstrip()}\n\nOutput\n{_nl(eo).rstrip()}"
    if note:
        s += "\n\nNote\n\n" + _nl(note).strip()
    return s


def load_icpc_eval() -> list[Problem]:
    out = []
    for r in _parquet_rows(_snap("RUC-AIBOX/ICPC-Eval") + "/data/*.parquet"):
        tests = r["test_cases"] or []
        if not tests:
            continue
        chk = _checker_bin(r["spj_code"], "spj2") if (r["spj_code"] or "").strip() else None
        if ("icpc_eval", str(r["id"])) in LENIENT_CHECKER_TASKS:
            continue
        if chk and chk["kind"] == "MISSING":
            print(f"[icpc_eval] dropping {r['id']}: its spj did not compile", flush=True)
            continue
        q = _compose_statement(r["description"], r["input"], r["output"], r["examples"], r["note"])
        out.append(Problem(
            benchmark="icpc_eval", task_id=str(r["id"]), question=q, starter_code="", fn_name=None,
            inputs=[_nl(t[0]) for t in tests], outputs=[_nl(t[1]) for t in tests],
            difficulty=None, difficulty_rank=None, time_limit=(r["time_limit_ms"] or 1000) / 1000.0,
            checker=chk, meta={"source": r["source"], "year": r["year"], "type": r["type"]},
        ))
    return out


# ---------------------------------------------------------------------------------------------
# USACO (codegenning/usacobench_formatted): bronze/silver/gold/platinum, stdin
# ---------------------------------------------------------------------------------------------
_USACO_RANK = {"bronze": 0.0, "silver": 1.0, "gold": 2.0, "platinum": 3.0}


def load_usaco() -> list[Problem]:
    out = []
    for r in _parquet_rows(_snap("codegenning/usacobench_formatted") + "/data/*.parquet"):
        io = json.loads(r["input_output"])
        if not io.get("inputs"):
            continue
        out.append(Problem(
            benchmark="usaco", task_id=r["id"], question=_nl(r["question"]), starter_code="", fn_name=None,
            inputs=[_nl(x) for x in io["inputs"]], outputs=[_nl(x) for x in io["outputs"]],
            difficulty=r["difficulty"], difficulty_rank=_USACO_RANK[r["difficulty"]],
            time_limit=4.0,  # USACO gives Python 4 s (2x the C++ limit)
        ))
    return out


# ---------------------------------------------------------------------------------------------
# LeetCodeDataset (newfacade/LeetCodeDataset): functional, assert-style check(candidate)
# ---------------------------------------------------------------------------------------------
_LEET_RANK = {"Easy": 0.0, "Medium": 1.0, "Hard": 2.0}


def load_leetcode() -> list[Problem]:
    import ast
    base = _snap("newfacade/LeetCodeDataset")
    out = []
    for f in ("LeetCodeDataset-train.jsonl", "LeetCodeDataset-test.jsonl"):
        with open(os.path.join(base, f)) as fh:
            for line in fh:
                r = json.loads(line)
                tree = ast.parse(r["test"])
                fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check")
                stmts = [ast.unparse(n) for n in fn.body]
                if not stmts:
                    continue
                stmts[0] = f"candidate = {r['entry_point']}\n" + stmts[0]
                out.append(Problem(
                    benchmark="leetcode", task_id=r["task_id"], question=_nl(r["problem_description"]),
                    starter_code=r["starter_code"], fn_name=None, inputs=[], outputs=[],
                    assert_tests=stmts, prelude=r["prompt"],
                    difficulty=r["difficulty"], difficulty_rank=_LEET_RANK[r["difficulty"]],
                    meta={"split": f.split("-")[1].split(".")[0], "date": r.get("estimated_date")},
                ))
    return out


# ---------------------------------------------------------------------------------------------
# TACO-verified (likaixin/TACO-verified): stdin + call-based, EASY..VERY_HARD
# ---------------------------------------------------------------------------------------------
_TACO_RANK = {"EASY": 0.0, "MEDIUM": 1.0, "MEDIUM_HARD": 2.0, "HARD": 3.0, "VERY_HARD": 4.0}


def _parse_seconds(s) -> float | None:
    import re
    if not s:
        return None
    m = re.search(r"([\d.]+)", str(s))
    return float(m.group(1)) if m else None


def load_taco_verified() -> list[Problem]:
    with open(os.path.join(_snap("likaixin/TACO-verified"), "taco_verified.json")) as fh:
        data = json.load(fh)
    out = []
    for r in data:
        io = r["input_output"]
        io = json.loads(io) if isinstance(io, str) else io
        if not io or not io.get("inputs"):
            continue
        fn = io.get("fn_name")
        if fn:
            inputs = ["\n".join(json.dumps(a) for a in args) for args in io["inputs"]]
            outputs = [json.dumps(o) for o in io["outputs"]]
        else:
            inputs = [_nl(x if isinstance(x, str) else "\n".join(map(str, x))) for x in io["inputs"]]
            outputs = [_nl(x if isinstance(x, str) else "\n".join(map(str, x))) for x in io["outputs"]]
        rank = _TACO_RANK.get(r["difficulty"])
        out.append(Problem(
            benchmark="taco_verified", task_id=str(r["id"]), question=_nl(r["question"]),
            starter_code=r["starter_code"] or "", fn_name=fn, inputs=inputs, outputs=outputs,
            difficulty=r["difficulty"], difficulty_rank=rank, time_limit=_parse_seconds(r.get("time_limit")),
            meta={"source": r["source"], "date": r.get("date"), "url": r.get("url")},
        ))
    return out


# ---------------------------------------------------------------------------------------------
# Codeforces (open-r1/codeforces, `verifiable` config): rating labels, stdin, python checkers.
# Tests (~64 MB per contest) are fetched lazily by ``prepare_codeforces`` for the chosen problems.
# ---------------------------------------------------------------------------------------------
_CF_COLS = ["id", "contest_id", "contest_start_year", "time_limit", "title", "description", "input_format",
            "output_format", "interaction_format", "note", "examples", "rating", "official_tests",
            "official_tests_complete", "generated_checker", "generated_tests", "input_mode", "executable"]


def load_codeforces() -> list[Problem]:
    base = _snap("open-r1/codeforces")
    out = []
    for r in _parquet_rows(base + "/verifiable/*.parquet", columns=_CF_COLS):
        if r["interaction_format"] or r["input_mode"] != "stdio" or not r["executable"]:
            continue
        if r["rating"] is None:
            continue
        exs = [(e["input"], e["output"]) for e in (r["examples"] or [])]
        q = _compose_statement(r["description"], r["input_format"], r["output_format"], exs, r["note"])
        chk = {"kind": "python", "src": r["generated_checker"]} if r["generated_checker"] else None
        out.append(Problem(
            benchmark="codeforces", task_id=r["id"], question=q, starter_code="", fn_name=None,
            inputs=[_nl(t["input"]) for t in (r["official_tests"] or [])],
            outputs=[_nl(t["output"]) for t in (r["official_tests"] or [])],
            difficulty=str(r["rating"]), difficulty_rank=float(r["rating"]),
            time_limit=float(r["time_limit"] or 1.0), checker=chk,
            meta={"contest_id": r["contest_id"], "year": r["contest_start_year"], "title": r["title"],
                  "n_generated": int(r["generated_tests"] or 0), "official_complete": bool(r["official_tests_complete"])},
        ))
    return out


def prepare_codeforces(problems: list[Problem]) -> None:
    """Attach the generated tests (downloaded per contest) to the chosen problems, in place."""
    from concurrent.futures import ThreadPoolExecutor
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    need = {p.meta["contest_id"] for p in problems if p.meta["n_generated"] > 0}

    def fetch(cid):
        try:
            return cid, hf_hub_download("open-r1/codeforces", f"generated_tests/test_cases_{cid}.parquet",
                                        repo_type="dataset")
        except Exception as e:  # noqa: BLE001
            return cid, e

    with ThreadPoolExecutor(16) as ex:
        paths = dict(ex.map(fetch, sorted(need)))
    for p in problems:
        if p.meta["n_generated"] <= 0:
            continue
        path = paths[p.meta["contest_id"]]
        if isinstance(path, Exception):
            p.meta["generated_tests_error"] = str(path)[:200]
            continue
        t = pq.read_table(path).to_pylist()
        rows = sorted((r for r in t if r["problem_id"] == p.task_id), key=lambda r: r.get("test_i", 0))
        p.inputs = p.inputs + [_nl(r["input"]) for r in rows]
        p.outputs = p.outputs + [_nl(r["output"]) for r in rows]
        p.meta["n_tests_generated_attached"] = len(rows)


# ---------------------------------------------------------------------------------------------
# OJBench (He-Ren/OJBench_testdata): NOI + ICPC, easy/medium/hard, DMOJ test archives
# ---------------------------------------------------------------------------------------------
_OJ_RANK = {"easy": 0.0, "medium": 1.0, "hard": 2.0}


def load_ojbench() -> list[Problem]:
    import yaml
    base = _snap("He-Ren/OJBench_testdata")
    prompts = {}
    with open(os.path.join(base, "prompts", "full.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            if r["language"] == "python":
                prompts[str(r["id"])] = r
    out = []
    for sub in ("NOI", "ICPC"):
        for d in sorted(os.listdir(os.path.join(base, sub))):
            pdir = os.path.join(base, sub, d)
            pid = d.split("-")[-1] if sub == "NOI" else d
            r = prompts.get(pid) or prompts.get(d)
            if r is None:
                continue
            cfg = yaml.safe_load(open(os.path.join(pdir, "init.yml")))
            chk = cfg.get("checker")
            checker = None
            if chk:
                name = chk.get("name") if isinstance(chk, dict) else chk
                if name == "floats":
                    checker = {"kind": "floats", "precision": int((chk.get("args") or {}).get("precision", 6))}
                else:  # bridged/testlib validators, interactors: not supported
                    continue
            if os.path.isdir(os.path.join(pdir, "output_validators")) or "interactive" in cfg:
                continue
            out.append(Problem(
                benchmark="ojbench", task_id=f"{sub}/{d}", question=_nl(r["prompt"]), starter_code="",
                fn_name=None, inputs=[], outputs=[], difficulty=r["difficulty"],
                difficulty_rank=_OJ_RANK[r["difficulty"]], checker=checker,
                time_limit=float(cfg.get("time_limit", 1.0) or 1.0),
                meta={"dir": pdir, "archive": cfg.get("archive"), "cases": cfg.get("test_cases") or []},
            ))
    return out


def prepare_ojbench(problems: list[Problem]) -> None:
    import zipfile
    for p in problems:
        zp = os.path.join(p.meta["dir"], p.meta["archive"])
        with zipfile.ZipFile(zp) as z:
            names = set(z.namelist())
            for c in p.meta["cases"]:
                if c["in"] in names and c["out"] in names:
                    p.inputs.append(_nl(z.read(c["in"]).decode("utf-8", "replace")))
                    p.outputs.append(_nl(z.read(c["out"]).decode("utf-8", "replace")))
        del p.meta["cases"]


LOADERS.update({
    "aethercode": load_aethercode,
    "icpc_eval": load_icpc_eval,
    "usaco": load_usaco,
    "leetcode": load_leetcode,
    "taco_verified": load_taco_verified,
    "codeforces": load_codeforces,
    "ojbench": load_ojbench,
})
PREPARE = {"codeforces": prepare_codeforces, "ojbench": prepare_ojbench}
# The loaders above embed AetherCode's tests at load time, so the earlier lazy hook must not run.
POST_SELECT_HOOKS.pop("aethercode", None)
