"""The model under evaluation: rendering a conversation, sampling with a thinking budget, parsing.

One ``Policy`` per evaluated model (a base model or a checkpoint). Everything model-family specific
lives here so the benchmarks only ever see ``generate(messages) -> Generation``:

- **Rendering.** Qwen-style policies use the HF chat template (``enable_thinking=True``; Qwen3.5's
  generation prompt already opens ``<think>``), Inkling (``thinkingmachines/*``) uses TML via
  ``tml_renderers`` with a fixed reasoning effort. Conversation history is re-rendered every call from
  plain ``{"role", "content"}`` messages whose assistant turns hold the VISIBLE reply only (earlier
  thinking is dropped, as both chat formats do natively), so multi-turn context grows with what was
  said, not with every token of reasoning.
- **Thinking budget (budget forcing).** A call may spend ``think_budget`` tokens. If it is cut off while
  still thinking, the renderer's wrap-up cue ("Considering the limited time… </think>") is appended and
  the answer is sampled with ``answer_budget`` more tokens; if it is cut off while already answering, the
  answer is continued with ``answer_budget`` more. So a long thinker is never scored on an EMPTY answer
  merely because it hit a token cap; every forced / still-truncated call is recorded
  (``Generation.forced`` / ``.truncated``) and reported per model so differences in it are visible.
- **Context limit.** The prompt plus both budgets must fit in the model's context window (from tinker's
  server capabilities). If it does not, the thinking budget is shrunk (``ctx_clipped``); if fewer than
  ``MIN_ROOM`` tokens are left, nothing is sampled and ``context_exhausted`` is set.
- **Seeds.** Every sampling call is seeded from its *position* (benchmark, item, sample, turn, call) —
  never from the model — so a checkpoint and its base model see common random numbers.
"""

from __future__ import annotations

import asyncio
import contextvars
import heapq
import itertools
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

import tinker

from monitordecorrelation.rl.episodes import derive_sample_seed
from monitordecorrelation.rl.renderers import THINK_BUDGET_SUFFIX, TmlRenderer, is_tml_policy

MIN_ROOM = 512  # fewer free context tokens than this -> context_exhausted (nothing is sampled)

# ---------------------------------------------------------------------------------------------
# Token usage. The harness sets ``USAGE`` to a fresh dict around each sample (one asyncio task), and
# every API request made on its behalf — tinker sampling calls here, judge calls in coherence.py (their
# worker threads inherit the context) — adds what the provider reported to it, keyed by billing
# account: ``tinker:<base model>`` (a base model and its checkpoints share one) or
# ``openrouter:<judge model>``. Requests that failed without a response report nothing and are not counted.
# ---------------------------------------------------------------------------------------------
USAGE: contextvars.ContextVar[dict | None] = contextvars.ContextVar("capeval_usage", default=None)
_USAGE_LOCK = threading.Lock()
USAGE_FIELDS = ("requests", "input_miss", "input_hit", "output", "cost_usd")


def add_usage(account: str, *, input_tokens: int, cache_hit: int, output: int, cost_usd: float = 0.0) -> None:
    """Add one request's usage to the current sample's ``USAGE`` dict (no-op outside a sample)."""
    usage = USAGE.get()
    if usage is None:
        return
    cache_hit = max(0, min(int(cache_hit or 0), int(input_tokens or 0)))
    with _USAGE_LOCK:
        a = usage.setdefault(account, dict.fromkeys(USAGE_FIELDS, 0))
        a["requests"] += 1
        a["input_hit"] += cache_hit
        a["input_miss"] += int(input_tokens or 0) - cache_hit
        a["output"] += int(output or 0)
        a["cost_usd"] += float(cost_usd or 0.0)


class PrioritySemaphore:
    """An asyncio semaphore whose waiters are served lowest-``priority`` first (FIFO within a priority).

    Used for the global cap on in-flight tinker requests: multi-turn work (agent turns, second chat
    turns) gets priority 0 so long sequential chains keep moving while thousands of independent
    single-turn requests (priority 1) wait — otherwise every agent turn would queue behind the whole
    single-turn backlog and the agentic benchmarks would set the wall-clock.
    """

    def __init__(self, value: int) -> None:
        if value < 1:
            raise ValueError("PrioritySemaphore needs value >= 1")
        self._value = value
        self._waiters: list[tuple[int, int, asyncio.Future]] = []
        self._seq = itertools.count()

    async def acquire(self, priority: int) -> None:
        if self._value > 0 and not self._waiters:
            self._value -= 1
            return
        fut = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (priority, next(self._seq), fut))
        try:
            await fut
        except asyncio.CancelledError:
            # Granted (result set) but cancelled before resuming: hand the permit on.
            if fut.done() and not fut.cancelled():
                self.release()
            raise

    def release(self) -> None:
        while self._waiters:
            _, _, fut = heapq.heappop(self._waiters)
            if not fut.done():  # skip waiters that were cancelled while queued
                fut.set_result(None)
                return
        self._value += 1

    @asynccontextmanager
    async def slot(self, priority: int):
        await self.acquire(priority)
        try:
            yield
        finally:
            self.release()


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float
    top_p: float
    top_k: int


@dataclass(frozen=True)
class Budget:
    """Per-call token budget: ``think_budget`` for the first call (thinking + any answer), then
    ``answer_budget`` for a forced/continued answer."""

    think_budget: int
    answer_budget: int


@dataclass
class Generation:
    cot: str = ""
    answer: str = ""
    gen_tokens: int = 0            # tokens SAMPLED (forced wrap-up tokens excluded)
    prompt_tokens: int = 0
    forced: bool = False           # thinking hit think_budget and was force-closed
    continued: bool = False        # the answer hit think_budget and was continued
    truncated: bool = False        # the final call still hit its token cap (answer may be cut off)
    ctx_clipped: bool = False      # the context window forced a smaller thinking budget
    context_exhausted: bool = False  # not enough context left to sample at all
    n_calls: int = 0
    stop_reason: str = ""

    def diag(self) -> dict:
        d = asdict(self)
        d.pop("cot")
        d.pop("answer")
        return d


class Policy:
    """A sampling client + how to talk to its model family. See the module docstring."""

    def __init__(
        self,
        *,
        key: str,
        base_model: str,
        sampling_client: tinker.SamplingClient,
        sampling: SamplingConfig,
        semaphore: PrioritySemaphore,
        context_length: int,
        thinking_effort: float,
        hf_tokenizer: Any | None = None,
        max_attempts: int = 6,
    ) -> None:
        self.key = key
        self.base_model = base_model
        self.client = sampling_client
        self.sampling = sampling
        self.sem = semaphore
        self.context_length = context_length
        self.max_attempts = max_attempts
        self.family = "tml" if is_tml_policy(base_model) else "hf"
        if self.family == "tml":
            from tml_renderers import chat

            self._chat = chat
            self._tml = TmlRenderer(effort=thinking_effort)
            self.stop = list(self._tml.stop_tokens or [])
            self._force = list(self._tml.force_answer_tokens())
        else:
            if hf_tokenizer is None:
                raise ValueError(f"{base_model}: HF-templated policies need hf_tokenizer")
            self.tok = hf_tokenizer
            self.stop = []  # the sampler stops on the model's EOS (<|im_end|>)
            self._force = list(self.tok.encode(THINK_BUDGET_SUFFIX, add_special_tokens=False))
            # Literal special-token strings in user text (tool output, file contents) would be
            # tokenized AS the control tokens by the chat template and forge turn structure — break
            # them up (TML's renderer encodes text ordinarily, so it needs no such step).
            specials = set(getattr(self.tok, "all_special_tokens", []) or [])
            specials |= {t.content for t in getattr(self.tok, "added_tokens_decoder", {}).values()}
            self._specials = sorted((s for s in specials if s and s.startswith("<") and len(s) > 2),
                                    key=len, reverse=True)
            self._strip_tail = [s for s in ("<|im_end|>", "<|endoftext|>", self.tok.eos_token) if s]

    # ------------------------------------------------------------------ rendering
    def _sanitize(self, text: str) -> str:
        for s in self._specials:
            if s in text:
                text = text.replace(s, s[0] + " " + s[1:])
        return text

    def prompt_tokens(self, messages: list[dict]) -> list[int]:
        """Render ``[{"role": "user"|"assistant", "content": str}, ...]`` (ending on a user turn) to the
        prompt token ids, generation prompt included."""
        if not messages or messages[-1]["role"] != "user":
            raise ValueError("a prompt must end with a user message")
        if self.family == "tml":
            chat = self._chat
            kinds = {"user": chat.AuthorKind.User, "assistant": chat.AuthorKind.Model}
            msgs = [chat.Message(content=chat.Text(m["content"]), author=chat.Author(kinds[m["role"]]))
                    for m in messages]
            spans, _ = self._tml._renderer.render_for_completion_with_effort(msgs, self._tml.effort)
            mi = self._tml._tml_tinker.token_spans_to_tinker_model_input(spans)
            toks: list[int] = []
            for chunk in mi.chunks:
                chunk_tokens = getattr(chunk, "tokens", None)
                if chunk_tokens is None:
                    raise TypeError(f"non-text chunk {type(chunk).__name__} in a text-only prompt")
                toks.extend(int(t) for t in chunk_tokens)
            return toks
        msgs = [{"role": m["role"], "content": self._sanitize(m["content"])} for m in messages]
        out = self.tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True,
                                           tokenize=True, return_dict=False)
        return list(getattr(out, "input_ids", out))

    def _prompt_opens_think(self, prompt: list[int]) -> bool:
        return self.tok.decode(prompt[-8:]).rstrip().endswith("<think>")

    def _in_thinking(self, prompt: list[int], toks: list[int]) -> bool:
        if self.family == "tml":
            return self._tml.in_open_think(toks)
        text = self.tok.decode(toks)
        opened = self._prompt_opens_think(prompt) or "<think>" in text
        return opened and "</think>" not in text

    def parse(self, prompt: list[int], toks: list[int]) -> tuple[str, str]:
        """-> (thinking, visible answer)."""
        if self.family == "tml":
            cot, answer, _ = self._tml.parse(toks)
            return cot, answer
        text = self.tok.decode(toks)
        for s in self._strip_tail:
            text = text.replace(s, "")
        if "</think>" in text:
            cot, answer = text.split("</think>", 1)
            return cot.replace("<think>", "").strip(), answer.strip()
        if self._prompt_opens_think(prompt) or "<think>" in text:
            return text.replace("<think>", "").strip(), ""  # never left the thinking block
        return "", text.strip()

    # ------------------------------------------------------------------ sampling
    async def _sample(self, tokens: list[int], max_tokens: int, seed: int, priority: int):
        params = tinker.SamplingParams(
            max_tokens=max_tokens, temperature=self.sampling.temperature, top_p=self.sampling.top_p,
            top_k=self.sampling.top_k, seed=seed, **({"stop": self.stop} if self.stop else {}),
        )
        mi = tinker.ModelInput.from_ints(tokens)
        for attempt in range(self.max_attempts):
            try:
                async with self.sem.slot(priority):
                    res = await self.client.sample_async(mi, 1, params)
                seqs = list(res.sequences)
                if len(seqs) != 1:
                    raise RuntimeError(f"asked for 1 sample, got {len(seqs)}")
                add_usage(f"tinker:{self.base_model}", input_tokens=len(tokens),
                          cache_hit=getattr(res, "prompt_cache_hit_tokens", 0), output=len(seqs[0].tokens))
                return seqs[0]
            except Exception as e:  # noqa: BLE001 — classified below
                status = getattr(e, "status_code", None)
                if status is not None and 400 <= status < 500 and status not in (408, 429):
                    raise  # a malformed request never fixes itself
                if attempt == self.max_attempts - 1:
                    raise
                await asyncio.sleep(min(120.0, 5.0 * 2 ** attempt))
        raise AssertionError("unreachable")

    async def generate(self, messages: list[dict], *, budget: Budget, seed_parts: tuple,
                       priority: int = 1) -> Generation:
        prompt = self.prompt_tokens(messages)
        g = Generation(prompt_tokens=len(prompt))
        room = self.context_length - len(prompt) - 1
        first = budget.think_budget
        reserve = budget.answer_budget + len(self._force)
        if first + reserve > room:
            first = room - reserve
            g.ctx_clipped = True
        if first < MIN_ROOM:
            g.context_exhausted = True
            return g
        seq = await self._sample(prompt, first, derive_sample_seed(*seed_parts, 0), priority)
        g.n_calls = 1
        toks = list(seq.tokens)
        g.gen_tokens = len(toks)
        stop = str(seq.stop_reason)
        if stop != "stop":
            extra: list[int] = []
            if self._in_thinking(prompt, toks):
                extra = list(self._force)
                g.forced = True
            else:
                g.continued = True
            seq2 = await self._sample(prompt + toks + extra, budget.answer_budget,
                                      derive_sample_seed(*seed_parts, 1), priority)
            g.n_calls = 2
            toks = toks + extra + list(seq2.tokens)
            g.gen_tokens += len(seq2.tokens)
            stop = str(seq2.stop_reason)
        g.stop_reason = stop
        g.truncated = stop != "stop"
        g.cot, g.answer = self.parse(prompt, toks)
        return g


@dataclass
class ModelSpec:
    """One evaluated model. ``key`` is unique; base models are keyed ``base:<model>``."""

    key: str
    base_model: str
    is_base: bool
    tinker_path: str | None = None        # as given in the checkpoints file
    sampler_path: str | None = None       # the sampler-weights path actually sampled from
    weights_info: dict = field(default_factory=dict)
