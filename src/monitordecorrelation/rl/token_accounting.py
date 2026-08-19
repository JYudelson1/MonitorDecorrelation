"""Per-rollout token accounting for budgeted sampling: what a thinking budget *actually* costs on
tinker, next to what it *would* cost if tinker enforced budgets in-engine.

The thinking budget in ``rl/thinking_budget.py`` is a two-call protocol (sample → force-close →
resume), so a forced rollout re-sends its prompt **and** its truncated reasoning as the prompt of the
continuation request. That overhead is invisible in the plain ``tokens/input_total`` curve, and it is
the number that decides whether a budget is actually saving anything. So every budgeted batch carries
two parallel figures:

* **ideal** — one request per rollout, as an in-engine budget would do it: prefill = the prompt,
  decode = every token of the final completion (a forced ``</think>`` still costs a decode step when
  the engine forces it).
* **actual** — what we really spent: the prefill of *every* request this rollout needed, and the
  tokens the sampler really generated (which excludes the injected closer — we splice that in for
  free, so actual decode can come in a hair *below* ideal).

Both are split into prefix-cache hits and misses, and the hits are **measured, not modelled**:
``tinker.SampleResponse.prompt_cache_hit_tokens`` reports how much of a request's prompt was served
from the cache (block-granular — empirically 64-token blocks — and counted once for the shared prompt
of a ``num_samples > 1`` request). Cached prefill is billed at a discount (tinker's pricing page:
"input tokens that hit the prompt cache (80% off)"), so hits vs misses is the cost-relevant split,
not raw prefill.

Charging convention, chosen to line up with the pre-existing ``tokens/input_total`` (prompt tokens
counted once **per rollout**) and with how tinker bills: a request that draws ``n`` samples from one
prompt is charged ``n × prompt_len`` of prefill, of which ``(n-1) × prompt_len`` are cache hits — the
prompt is prefilled once and the other ``n-1`` copies are billed as hits — plus whatever
``prompt_cache_hit_tokens`` says about that one shared prefill.

Everything here is pure integer bookkeeping: no tinker import, no I/O.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

#: Key under which one rollout's account rides in ``Rollout.meta``. Absent ⇒ no budget was in play,
#: and the ``tokens/…_ideal|actual`` metrics are not emitted at all (so an unbudgeted run's metric
#: rows are byte-identical to what they were before budgets existed).
META_KEY = "token_account"


@dataclass
class RolloutTokenAccount:
    """Ideal-vs-actual prefill/decode for ONE rollout, accumulated request by request.

    ``prefill_*`` counts prompt tokens processed; ``decode_*`` counts tokens generated. ``*_hit`` is
    the prefix-cache-served part of the corresponding prefill (miss = total − hit).
    """

    prefill_ideal: int = 0
    prefill_ideal_hit: int = 0
    prefill_actual: int = 0
    prefill_actual_hit: int = 0
    decode_ideal: int = 0
    decode_actual: int = 0
    n_requests: int = 0
    n_injected: int = 0       # tokens we spliced in (the documented closing text)
    forced: bool = False      # did the budget actually bind for this rollout?
    thinking_tokens: int = 0  # tokens inside <think>…</think> in the final completion

    def add_request(
        self, *, prompt_len: int, cache_hit: int, generated: int, is_first: bool
    ) -> None:
        """Charge one sampling request to this rollout.

        ``cache_hit`` is this rollout's share of the request's measured cache hit (see the module
        docstring for how a ``num_samples > 1`` request is split). ``is_first`` marks the pass-1
        request — the only one an in-engine budget would have needed, hence the only one charged to
        the *ideal* prefill.
        """
        self.n_requests += 1
        self.prefill_actual += prompt_len
        self.prefill_actual_hit += cache_hit
        self.decode_actual += generated
        if is_first:
            self.prefill_ideal += prompt_len
            self.prefill_ideal_hit += cache_hit

    def finish(self, *, n_completion_tokens: int, n_injected: int, forced: bool,
               thinking_tokens: int) -> None:
        """Close the account once the rollout's final token sequence is known.

        An in-engine budget would have decoded the whole completion, forced closer included — so the
        ideal decode is the full completion length, while ours skipped ``n_injected`` decode steps.
        """
        self.decode_ideal = n_completion_tokens
        self.n_injected = n_injected
        self.forced = forced
        self.thinking_tokens = thinking_tokens

    def as_meta(self) -> dict:
        return asdict(self)


@dataclass
class BatchTokenAccount:
    """Sum of a batch's rollout accounts, ready to be flattened into ``tokens/…`` metric keys."""

    n: int = 0
    totals: dict[str, int] = field(default_factory=dict)

    def add(self, account: dict) -> None:
        self.n += 1
        for k, v in account.items():
            if isinstance(v, bool):
                v = int(v)
            self.totals[k] = self.totals.get(k, 0) + int(v)


def budget_token_metrics(accounts: list[dict]) -> dict[str, float]:
    """``tokens/…`` metrics for one batch of budgeted rollouts (``{}`` when the budget was off).

    Emits, for prefill and decode alike, the **ideal** figure (what an in-engine budget would have
    cost) beside the **actual** one, each prefill split into cache hits and misses, plus the two
    ratios that say whether the budget is paying for itself and the rates that say how often it
    actually bound.
    """
    if not accounts:
        return {}
    acc = BatchTokenAccount()
    for a in accounts:
        acc.add(a)
    t, n = acc.totals, float(acc.n)

    def ratio(num: float, den: float) -> float:
        return num / den if den else float("nan")

    out = {
        # --- prefill: what the sampler had to read -------------------------------------------
        "tokens/prefill_ideal_total": float(t["prefill_ideal"]),
        "tokens/prefill_actual_total": float(t["prefill_actual"]),
        "tokens/prefill_ideal_cache_hit_total": float(t["prefill_ideal_hit"]),
        "tokens/prefill_ideal_cache_miss_total": float(t["prefill_ideal"] - t["prefill_ideal_hit"]),
        "tokens/prefill_actual_cache_hit_total": float(t["prefill_actual_hit"]),
        "tokens/prefill_actual_cache_miss_total": float(t["prefill_actual"] - t["prefill_actual_hit"]),
        # --- decode: what the sampler had to write -------------------------------------------
        "tokens/decode_ideal_total": float(t["decode_ideal"]),
        "tokens/decode_actual_total": float(t["decode_actual"]),
        # --- per rollout ----------------------------------------------------------------------
        "tokens/prefill_ideal_per_rollout": t["prefill_ideal"] / n,
        "tokens/prefill_actual_per_rollout": t["prefill_actual"] / n,
        "tokens/decode_ideal_per_rollout": t["decode_ideal"] / n,
        "tokens/decode_actual_per_rollout": t["decode_actual"] / n,
        # --- the overhead the two-call protocol costs ------------------------------------------
        # >1 means the budget's re-prefill; the cache-miss ratio is the one that costs full price.
        "tokens/prefill_overhead_ratio": ratio(t["prefill_actual"], t["prefill_ideal"]),
        "tokens/prefill_miss_overhead_ratio": ratio(
            t["prefill_actual"] - t["prefill_actual_hit"], t["prefill_ideal"] - t["prefill_ideal_hit"]
        ),
        "tokens/prefill_actual_cache_hit_rate": ratio(t["prefill_actual_hit"], t["prefill_actual"]),
        # --- how often the budget bound ---------------------------------------------------------
        "tokens/budget_forced_rate": t["forced"] / n,          # reasoning hit the cap
        "tokens/sampling_requests_per_rollout": t["n_requests"] / n,  # 1.0 = no continuation needed
        "tokens/thinking_per_rollout": t["thinking_tokens"] / n,
    }
    return out
