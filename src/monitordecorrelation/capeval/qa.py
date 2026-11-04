"""Knowledge & reasoning benchmarks (single-turn): MMLU-Pro, GPQA-Diamond, MATH-500.

Each sample is ONE generation with the thinking budget of ``policy.Policy.generate`` (default 36864
thinking + 2048 answer tokens = 38912, Qwen's recommended output length for hard reasoning benchmarks). The answer is extracted
from the VISIBLE reply only (never from the thinking); a reply with no extractable answer scores 0 and
is counted in ``extract_fail``, which the report shows next to the forced / truncated rates.
"""

from __future__ import annotations

import random
import re

import pandas as pd

from monitordecorrelation.capeval.bench import Benchmark, Item, RunCtx, budget_dict, clip_middle, gen_diag
from monitordecorrelation.capeval.policy import Budget, Policy

LETTERS = "ABCDEFGHIJ"

# "The answer is (B)", "Answer: B", "**Answer:** (B)" — the LAST one in the reply wins.
_ANSWER_RE = re.compile(r"(?i:answer)\s*(?:(?i:is)|:)?\s*:?\s*\(?\s*([A-J])\s*\)?(?![A-Za-z0-9])")
_BOXED_LETTER_RE = re.compile(r"\\boxed\{\s*\(?\s*([A-J])\s*\)?\s*\}")


def extract_letter(text: str, valid: str) -> str | None:
    """The final multiple-choice letter in ``text`` (restricted to ``valid``), or None."""
    clean = re.sub(r"\\text(?:bf)?\{([^{}]*)\}", r"\1", text or "")
    clean = clean.replace("*", "").replace("$", "")
    for rx in (_ANSWER_RE, _BOXED_LETTER_RE):
        hits = [m.group(1) for m in rx.finditer(clean) if m.group(1) in valid]
        if hits:
            return hits[-1]
    return None


def _hf_file(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset")


class _SingleTurn(Benchmark):
    family = "knowledge"
    metrics = ("accuracy",)
    priority = 1

    def __init__(self, *, n_samples: int, budget: Budget) -> None:
        self.n_samples = n_samples
        self.budget = budget

    def prompt(self, item: Item) -> str:
        raise NotImplementedError

    def grade(self, item: Item, answer: str) -> tuple[float, bool, str | None]:
        """-> (score, extract_fail, prediction)."""
        raise NotImplementedError

    async def run(self, ctx: RunCtx, policy: Policy, item: Item, sample: int) -> dict:
        g = await policy.generate([{"role": "user", "content": self.prompt(item)}], budget=self.budget,
                                  seed_parts=(self.name, item.id, sample), priority=self.priority)
        score, extract_fail, pred = self.grade(item, g.answer)
        return {
            "scores": {"accuracy": score},
            "diag": {**gen_diag([g]), "extract_fail": extract_fail},
            "transcript": {"prediction": pred, "gold": item.data["gold"], "answer": g.answer,
                           "cot": clip_middle(g.cot, 1500, 1500), "cot_chars": len(g.cot)},
        }


class MMLUPro(_SingleTurn):
    """MMLU-Pro (TIGER-Lab), a class-stratified random subset (``per_category`` per category, seed 0),
    zero-shot CoT with the official "The answer is (X)" format."""

    name = "mmlu_pro"

    def __init__(self, *, per_category: int, **kw) -> None:
        super().__init__(**kw)
        self.per_category = per_category

    def settings(self) -> dict:
        return {"per_category": self.per_category, "n_samples": self.n_samples, **budget_dict(self.budget),
                "prompt_version": 1}

    def load(self, cache_dir: str) -> list[Item]:
        df = pd.read_parquet(_hf_file("TIGER-Lab/MMLU-Pro", "data/test-00000-of-00001.parquet"))
        rng = random.Random(0)
        items: list[Item] = []
        for cat in sorted(df.category.unique()):
            rows = df[df.category == cat].sort_values("question_id")
            idx = sorted(rng.sample(range(len(rows)), min(self.per_category, len(rows))))
            for _, r in rows.iloc[idx].iterrows():
                opts = [str(o) for o in r.options]
                items.append(Item(id=str(r.question_id), cluster=str(r.question_id), group=str(cat),
                                  data={"question": r.question, "options": opts, "gold": str(r.answer),
                                        "category": str(cat)}))
        return items

    def prompt(self, item: Item) -> str:
        d = item.data
        opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(d["options"]))
        return (f"The following is a multiple choice question about {d['category']}. Think step by step "
                'and then finish your answer with "The answer is (X)" where X is the correct letter '
                f"choice.\n\nQuestion: {d['question']}\nOptions:\n{opts}")

    def grade(self, item: Item, answer: str):
        pred = extract_letter(answer, LETTERS[: len(item.data["options"])])
        return float(pred == item.data["gold"]), pred is None, pred


class GPQADiamond(_SingleTurn):
    """GPQA-Diamond (198 questions, options A–D in a fixed order) with simple-evals' zero-shot prompt.
    Data: the ungated ``fingertap/GPQA-Diamond`` mirror (the canonical ``Idavidrein/gpqa`` needs an HF
    token). Several samples per question tighten the per-question accuracy."""

    name = "gpqa_diamond"

    def settings(self) -> dict:
        return {"n_samples": self.n_samples, **budget_dict(self.budget), "prompt_version": 1}

    def load(self, cache_dir: str) -> list[Item]:
        df = pd.read_parquet(_hf_file("fingertap/GPQA-Diamond", "test/gpqa_diamond.parquet"))
        return [Item(id=f"gpqa{i}", cluster=f"gpqa{i}", group="all",
                     data={"question": r.question, "gold": str(r.answer).strip()})
                for i, r in enumerate(df.itertuples())]

    def prompt(self, item: Item) -> str:
        return ("Answer the following multiple choice question. The last line of your response should be "
                "of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
                f"Think step by step before answering.\n\n{item.data['question']}")

    def grade(self, item: Item, answer: str):
        pred = extract_letter(answer, "ABCD")
        return float(pred == item.data["gold"]), pred is None, pred


class Math500(_SingleTurn):
    """MATH-500 (HuggingFaceH4), boxed final answer, graded with ``math_verify`` (symbolic equivalence)."""

    name = "math500"

    def settings(self) -> dict:
        return {"n_samples": self.n_samples, **budget_dict(self.budget), "prompt_version": 1}

    def load(self, cache_dir: str) -> list[Item]:
        df = pd.read_json(_hf_file("HuggingFaceH4/MATH-500", "test.jsonl"), lines=True)
        return [Item(id=str(r.unique_id), cluster=str(r.unique_id), group=f"level{r.level}",
                     data={"problem": r.problem, "gold": str(r.answer), "subject": r.subject})
                for r in df.itertuples()]

    def prompt(self, item: Item) -> str:
        return f"{item.data['problem']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

    def grade(self, item: Item, answer: str):
        # math_verify's timeouts are signal-based: this runs on the event-loop (main) thread.
        from math_verify import parse, verify

        if "\\boxed" not in (answer or ""):
            return 0.0, True, None
        pred = parse(answer)
        if not pred:
            return 0.0, True, None
        gold = parse(f"${item.data['gold']}$")
        try:
            ok = bool(verify(gold, pred))
        except Exception:  # noqa: BLE001 — an unverifiable expression is a wrong answer
            ok = False
        return float(ok), False, str(pred[-1])[:200]
