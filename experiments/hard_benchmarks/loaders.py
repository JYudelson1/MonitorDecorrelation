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
