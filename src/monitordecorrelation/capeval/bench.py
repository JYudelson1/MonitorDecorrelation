"""The benchmark interface shared by every capeval benchmark, plus small helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from monitordecorrelation.capeval.policy import Budget, Generation, Policy


@dataclass(frozen=True)
class Item:
    id: str
    cluster: str        # bootstrap resampling unit (items sharing a source problem share a cluster)
    group: str          # breakdown category (subject, difficulty, …)
    data: dict = field(hash=False, compare=False)


@dataclass
class RunCtx:
    """Shared services a benchmark may need (only the ones it uses must be set)."""

    sandbox: Any = None   # sandbox.SandboxManager (agentic benchmarks)
    judge: Any = None     # coherence.Judge (judged benchmarks)
    cache_dir: str = ""


class Benchmark:
    """One benchmark. ``run`` returns a record ``{"scores": {metric: float}, "diag": {...},
    "transcript": {...}}`` for one (model, item, sample); the harness adds the bookkeeping fields."""

    name: str = ""
    family: str = ""                       # "knowledge" | "agentic" | "judge"
    metrics: tuple[str, ...] = ("accuracy",)  # the first one is the headline metric
    lower_is_better: frozenset[str] = frozenset()
    n_samples: int = 1
    priority: int = 1                      # PrioritySemaphore priority of its sampling calls

    def load(self, cache_dir: str) -> list[Item]:
        raise NotImplementedError

    def settings(self) -> dict:
        """Everything that changes what a sample means (for the resume fingerprint + the output)."""
        raise NotImplementedError

    def fingerprint(self) -> str:
        blob = json.dumps({"name": self.name, **self.settings()}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    async def run(self, ctx: RunCtx, policy: Policy, item: Item, sample: int) -> dict:
        raise NotImplementedError


def clip_middle(text: str, head: int, tail: int) -> str:
    """``text`` with its middle elided when longer than ``head + tail`` chars."""
    if len(text) <= head + tail:
        return text
    return f"{text[:head]}\n[... {len(text) - head - tail} characters omitted ...]\n{text[-tail:]}"


def gen_diag(gens: list[Generation]) -> dict:
    """Aggregate generation diagnostics over the calls of one sample (one per turn)."""
    return {
        "n_gen": len(gens),
        "n_forced": sum(g.forced for g in gens),
        "n_truncated": sum(g.truncated for g in gens),
        "n_ctx_clipped": sum(g.ctx_clipped for g in gens),
        "context_exhausted": any(g.context_exhausted for g in gens),
        "gen_tokens": sum(g.gen_tokens for g in gens),
        "max_prompt_tokens": max((g.prompt_tokens for g in gens), default=0),
    }


def budget_dict(b: Budget) -> dict:
    return {"think_budget": b.think_budget, "answer_budget": b.answer_budget}
