"""Free-form coherence / consistency / "RL-fried" benchmark (LLM-judged).

The model holds MT-Bench's 80 two-turn conversations (writing, roleplay, reasoning, math, coding,
extraction, STEM, humanities — the second turn always builds on the first, which is what makes
cross-turn consistency checkable). A judge (an OpenRouter model, blind to which model it is grading)
reads the whole conversation — each turn's visible reply AND its (abridged) thinking — and returns three
1–10 ratings:

- ``coherence``   — fluent, well-formed, organized, responsive (higher = better);
- ``consistency`` — no contradictions within a reply, across the two turns, or between the reasoning and
  the reply (higher = better);
- ``rl_fried``    — signs of RL over-optimization: loops, degenerate/garbled text, tics, a fixed template
  regardless of the request, telegraphic grammar, language switching, ignoring the request, claiming
  success without doing the work (higher = MORE fried, i.e. worse).

Plus one objective metric computed here: ``repetition`` = fraction of repeated word 4-grams in the
replies (higher = worse). Judge calls go through ``Judge`` (bounded retries: a content-filtered
transcript is recorded as a judge failure, never retried forever) under the repo's cross-process
OpenRouter concurrency permit.
"""

from __future__ import annotations

import asyncio
import json
import re

import pandas as pd

from monitordecorrelation.capeval.bench import Benchmark, Item, RunCtx, budget_dict, clip_middle, gen_diag
from monitordecorrelation.capeval.policy import Budget, Policy, add_usage

JUDGE_SYSTEM = ("You are a meticulous evaluator of language-model behaviour. You grade transcripts "
                "strictly according to the rubric and answer with JSON only.")

JUDGE_PROMPT = """Below is a two-turn conversation between a user and an AI assistant. For each assistant turn you see its private reasoning (possibly abridged; it may be empty) and its visible reply.

Rate the ASSISTANT on three integer scales from 1 to 10:

1. "coherence": Are the replies fluent, well-formed, logically organized, and responsive to what the user actually asked? 10 = perfectly coherent; 1 = incoherent or garbled.
2. "consistency": Is the assistant consistent with itself — no contradictions within a reply, the second reply correctly builds on the first, and the visible replies agree with the assistant's own reasoning? 10 = fully consistent; 1 = contradicts itself repeatedly.
3. "rl_fried": How strongly does the assistant show signs of a model damaged by over-optimization with reinforcement learning ("RL-fried")? Signs include: repetitive loops or phrases; degenerate, garbled or cut-off text; odd verbal tics; a fixed template applied regardless of the request; telegraphic or broken grammar; spurious language switching or strange tokens; ignoring what the user asked; bizarre formatting; reward-hacking behaviour such as claiming success without doing the work. 1 = a completely normal, healthy assistant; 10 = severely fried.

Judge behaviour, not factual correctness (unless an error itself shows incoherence or inconsistency). Markers like "[... N characters omitted ...]" were inserted by us to abridge long text: do not penalize them.

<conversation>
{conversation}
</conversation>

Answer with ONLY a JSON object of this form:
{{"coherence": <1-10>, "consistency": <1-10>, "rl_fried": <1-10>, "fried_signs": ["<short phrase>", ...], "rationale": "<two or three sentences>"}}"""

_JSON_RE = re.compile(r"\{.*\}", re.S)
_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def repetition(text: str, n: int = 4) -> float:
    words = (text or "").split()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def parse_judgement(text: str) -> dict | None:
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for k in ("coherence", "consistency", "rl_fried"):
        v = d.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 1 <= v <= 10:
            return None
        out[k] = float(v)
    out["fried_signs"] = [str(s)[:200] for s in d.get("fried_signs", []) if isinstance(s, (str, int, float))][:10]
    out["rationale"] = str(d.get("rationale", ""))[:2000]
    return out


class Judge:
    """OpenRouter JSON-verdict judge with a BOUNDED retry policy.

    Not the repo's ``post_chat`` loop: that one retries every non-"stop" finish forever, which is right
    for a training run's monitors but hangs an eval on a transcript the provider deterministically
    filters (``finish_reason: content_filter`` — seen live on a few MT-Bench conversations). Here:
    fatal statuses (bad request / key / credits) raise; transport errors, 408/429/5xx, the in-flight
    402 and malformed bodies retry up to ``max_attempts`` times with backoff; a content-filtered reply
    is retried once and then given up on. A sample whose judging gives up is kept with
    ``diag.judge_fail`` (+ ``judge_fail_reason``), excluded from the judge metrics, counted in the
    report, and re-judged by ``--resume``. Each attempt holds the repo's cross-process OpenRouter permit.
    """

    def __init__(self, model: str, *, max_concurrency: int, max_tokens: int = 2000,
                 timeout: float = 300.0, parse_attempts: int = 3, max_attempts: int = 8) -> None:
        from monitordecorrelation.monitors.openrouter import resolve_api_key

        self.model = model
        self.sem = asyncio.Semaphore(max_concurrency)
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.parse_attempts = parse_attempts
        self.max_attempts = max_attempts
        self.api_key = resolve_api_key()

    def _call(self, body: dict) -> tuple[str | None, dict]:
        """One judgement request (blocking). -> (reply text, or None when it gave up; call info)."""
        import time

        import httpx

        from monitordecorrelation.globalsem import openrouter_slot
        from monitordecorrelation.monitors.openrouter import _FATAL_STATUS, _OPENROUTER_URL, _is_retryable_402

        last, filtered = "", 0
        for attempt in range(1, self.max_attempts + 1):
            try:
                with openrouter_slot():
                    resp = httpx.post(_OPENROUTER_URL, json=body, timeout=self.timeout,
                                      headers={"Authorization": f"Bearer {self.api_key}"})
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last = f"{type(e).__name__}: {e}"
            else:
                if resp.status_code in _FATAL_STATUS and not _is_retryable_402(resp):
                    raise RuntimeError(f"judge {self.model}: HTTP {resp.status_code}: {resp.text[:3000]}")
                if resp.status_code >= 400:
                    last = f"HTTP {resp.status_code}: {resp.text[:300]}"
                else:
                    try:
                        data = resp.json()
                    except ValueError:
                        data = None
                    # Every answered request is billed (a filtered or empty reply included): count it.
                    u = data.get("usage") if isinstance(data, dict) else None
                    if isinstance(u, dict):
                        add_usage(f"openrouter:{self.model}", input_tokens=u.get("prompt_tokens") or 0,
                                  cache_hit=(u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                                  output=u.get("completion_tokens") or 0, cost_usd=u.get("cost") or 0.0)
                    try:
                        choice = data["choices"][0]
                        finish = choice.get("finish_reason")
                        content = choice["message"].get("content")
                    except (KeyError, IndexError, TypeError) as e:
                        last = f"malformed response body ({type(e).__name__})"
                    else:
                        info = {"attempts": attempt, "finish_reason": finish, "usage": data.get("usage"),
                                "id": data.get("id")}
                        if finish == "content_filter":
                            filtered += 1
                            last = "content_filter"
                            if filtered >= 2:
                                return None, {**info, "fail_reason": "content_filter"}
                        elif isinstance(content, str) and content.strip():
                            return content, info
                        else:
                            last = f"empty reply (finish_reason={finish})"
            if attempt < self.max_attempts:
                time.sleep(min(30.0, 2.0 * 2 ** attempt))
        return None, {"attempts": self.max_attempts, "fail_reason": f"api_error: {last}"[:500]}

    async def judge(self, conversation: str) -> tuple[dict | None, dict]:
        """-> (parsed verdict, or None if the judge gave up / never returned parseable JSON; info)."""
        body = {"model": self.model, "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                             {"role": "user", "content": JUDGE_PROMPT.format(conversation=conversation)}]}
        info: dict = {"requests": 0, "raw": None}
        for _ in range(self.parse_attempts):
            info["requests"] += 1
            async with self.sem:
                text, call = await asyncio.to_thread(self._call, body)
            info.update(call)
            if text is None:
                return None, info
            info["raw"] = text[:4000]
            parsed = parse_judgement(text)
            if parsed is not None:
                info.pop("fail_reason", None)
                return parsed, info
            info["fail_reason"] = "unparseable reply"
        return None, info


def render_conversation(turns: list[dict]) -> str:
    parts = []
    for i, t in enumerate(turns, 1):
        parts.append(f"[User, turn {i}]\n{t['user']}\n")
        parts.append(f"[Assistant reasoning, turn {i}]\n{clip_middle(t['thinking'], 2500, 1500) or '(none)'}\n")
        parts.append(f"[Assistant reply, turn {i}]\n{clip_middle(t['reply'], 8000, 3000) or '(empty reply)'}\n")
    return "\n".join(parts)


class MTBenchCoherence(Benchmark):
    name = "coherence"
    family = "judge"
    metrics = ("coherence", "consistency", "rl_fried", "repetition")
    lower_is_better = frozenset({"rl_fried", "repetition"})
    priority = 1

    def __init__(self, *, n_samples: int, budget: Budget, judge_model: str) -> None:
        self.n_samples = n_samples
        self.budget = budget
        self.judge_model = judge_model

    def settings(self) -> dict:
        return {"n_samples": self.n_samples, **budget_dict(self.budget), "judge_model": self.judge_model,
                "judge_prompt_version": 1}

    def load(self, cache_dir: str) -> list[Item]:
        from huggingface_hub import hf_hub_download

        df = pd.read_json(hf_hub_download("HuggingFaceH4/mt_bench_prompts", "raw/question.jsonl",
                                          repo_type="dataset"), lines=True)
        items = []
        for r in df.itertuples():
            turns = list(r.prompt) if not isinstance(r.prompt, str) else eval_list(r.prompt)
            items.append(Item(id=f"mtb{r.question_id}", cluster=f"mtb{r.question_id}", group=str(r.category),
                              data={"turns": [str(t) for t in turns]}))
        return items

    async def run(self, ctx: RunCtx, policy: Policy, item: Item, sample: int) -> dict:
        messages: list[dict] = []
        gens = []
        convo = []
        for t, q in enumerate(item.data["turns"]):
            messages.append({"role": "user", "content": q})
            g = await policy.generate(messages, budget=self.budget, seed_parts=(self.name, item.id, sample, t),
                                      priority=0 if t else self.priority)
            gens.append(g)
            convo.append({"user": q, "thinking": g.cot, "reply": g.answer, "forced": g.forced,
                          "truncated": g.truncated, "context_exhausted": g.context_exhausted})
            if g.context_exhausted:
                break
            messages.append({"role": "assistant", "content": g.answer})
        replies = " ".join(c["reply"] for c in convo)
        rec = {
            "scores": {"repetition": repetition(replies)},
            "diag": {**gen_diag(gens), "reply_chars": sum(len(c["reply"]) for c in convo),
                     "empty_reply": any(not c["reply"].strip() for c in convo),
                     "cjk_frac": len(_CJK_RE.findall(replies)) / max(1, len(replies))},
            "transcript": {"conversation": [{**c, "thinking": clip_middle(c["thinking"], 3000, 1500),
                                             "thinking_chars": len(c["thinking"])} for c in convo]},
        }
        await self.judge_record(ctx, rec, convo)
        return rec

    async def judge_record(self, ctx: RunCtx, rec: dict, convo: list[dict] | None = None) -> None:
        """Fill ``rec``'s judge scores in place (also used on resume for records whose judging failed)."""
        convo = convo if convo is not None else rec["transcript"]["conversation"]
        verdict, info = await ctx.judge.judge(render_conversation(convo))
        rec["transcript"]["judge"] = {"verdict": verdict, **info}
        rec["diag"]["judge_fail"] = verdict is None
        rec["diag"]["judge_fail_reason"] = info.get("fail_reason") if verdict is None else None
        if verdict is not None:
            for k in ("coherence", "consistency", "rl_fried"):
                rec["scores"][k] = verdict[k]


def eval_list(s: str) -> list[str]:
    """MT-Bench's ``prompt`` column is a Python-repr list of strings: parse it without eval."""
    import ast

    v = ast.literal_eval(s)
    if not isinstance(v, list):
        raise ValueError(f"unexpected MT-Bench prompt field: {s[:100]}")
    return v
