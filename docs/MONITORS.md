# Monitors (Detectors)

A **monitor** is any function that takes a rollout and produces a misalignment score/label. The whole
project is about how monitors degrade (and co-degrade) under training pressure, so the abstraction
needs to be generic across detector *families* — black-box text monitors today, white-box probes and
activation oracles tomorrow.

Build state in [`../STATUS.md`](../STATUS.md).

## The generic `Monitor` seam

```python
@dataclass
class MonitorResult:
    score: float          # higher = more likely misbehavior
    label: bool           # thresholded decision
    meta: dict            # raw judge text, probe logits, etc.

class Monitor(Protocol):
    name: str
    def score(self, rollout: Rollout) -> MonitorResult: ...
```

A `Rollout` carries everything any monitor family might need: the prompt, the policy's CoT, the final
output, and (when available) **activations**. Black-box monitors read text fields; white-box monitors
read activations. Detector *strength* is just configuration (which `model_id` for a CoT judge, which
checkpoint for a probe) — not a different interface.

This is the one abstraction to keep clean. Everything else (envs, RL loop) plugs into it.

**Monitors only ever see valid rollouts.** A rollout truncated by `max_tokens` or whose output the env
cannot parse is *invalid* (`envs/base.py::invalid_reason`): its RL reward is a flat −1 and the loop
never hands it to a monitor — train-against or held-out, judge or probe, training or eval. So every
monitor metric (AUROC, d′, class means, `n_scored`) is over the valid rollouts; `invalid_rate` is the
excluded fraction. See docs/ENVIRONMENTS.md ("Invalid rollouts").

### Single-turn vs multi-turn judge prompts

A single-turn rollout genuinely *is* two flat strings, so the judge prompt shows two sections
(`=== ASSISTANT REASONING ===`, `=== ASSISTANT FINAL ANSWER ===`). A **multi-turn episode** is not:
the env still has to flatten it into those two fields (`terminal_verifier._cot_view` concatenates
every turn's thinking, `_output_view` every turn's message + terminal output), so that layout would
show the judge *all* the reasoning before *any* of the actions — an order the policy never produced,
and one that hides the thought that immediately preceded each command.

So the two are different monitors. `CoTMonitor` (`monitors/cot_monitor.py`) is **single-turn only**
and ignores `meta["episode"]`. Multi-turn envs (`env.multi_turn`, i.e. the terminal env) get
**`AgentCoTMonitor`** (`monitors/agent_cot_monitor.py`) — `build_monitors(..., multi_turn=True)`
swaps it in for every `kind: cot` spec, same fields. It renders the episode as a chat transcript in
the order it happened — `USER:` (the task), then per turn `ASSISTANT:` (`<thinking>…</thinking>`
first when `use_cot`, then the message) and `TOOL:` (what the command printed); a mid-episode turn
with no command is followed by the env's "no command" `USER:` message (only in episodes recorded
before 2026-09-18 — a command-less turn now ends the episode); `SYSTEM:` only if the prompt
carries one — inside a fixed reward-hacking judge prompt (`JUDGE_PROMPT`, 0–100 suspicion score, or
`JUDGE_PROMPT_BINARY`). It only judges reward hacking, and a CoT-only view (`use_output: false`) is
rejected — a transcript without the actions is not a transcript. Every assumption about the
per-turn record is asserted (`agentic_turns`), including that the turns are the ones this
rollout's flattened `cot`/`output` came from, so a malformed or mismatched rollout raises instead
of being judged on a wrong transcript. Pinned by `tests/test_agent_cot_monitor.py`, which also
replays a real Inkling-Small run when `../MonitorDecorrelationOld` is checked out.

### Every judge call is saved with the rollout

`CoTMonitor.score` returns, in `MonitorResult.meta["call"]`, a record of the API call that produced
the verdict (`cot_monitor.JudgeCall`): the endpoint, the **exact JSON body POSTed** (`model`,
`messages` — i.e. the full prompt — `temperature`, `max_tokens`, `reasoning`; for a vLLM judge
`chat_template_kwargs` / `thinking_token_budget` instead of `reasoning`), the client timeout,
and the **full response** — the assistant `message` (`content`, plus the judge's chain of thought
under `reasoning` / `reasoning_details` when the provider returns one), `finish_reason`, and the
response-level `id` / `model` / `provider` / `usage`. When a call was retried only the attempt that
succeeded is recorded (`attempts` says how many it took). `rl/train.py` writes it per monitor into
`rollouts.jsonl` and `eval_rollouts.jsonl` as `monitors.<name>.call` (via
`eval.rollout_dump.monitor_record`); the committed `eval_rollouts_slim.jsonl` keeps `{score, label}`
plus each judge's `finish_reason` and `parse_error` flag (since 2026-09-25). `visualize_transcripts.py` shows these saved calls per rollout — and says "unavailable" for
dumps written before this existed, rather than rebuilding a prompt from the run config (a rebuild
can silently differ from what the judge was sent if the repo has changed since the run).

### Judge-side reasoning: one `reasoning` object, validated per judge model

A judge's `reasoning` (config key / `CoTMonitor` and `AgentCoTMonitor` kwarg) is the OpenRouter
`reasoning` object its every call sends. `monitors/judge_reasoning.py:resolve_reasoning` is the one
place it is validated and the model's default filled in — used by the config schema (at load, naming
the monitor), by both monitor classes (for hand-built judges) and by `scripts/verify_runs.py`. It is
**specialized to two judge models**, because the same object means different things on different
models and OpenRouter silently ignores or clamps what a model does not support:

- **`google/gemini-2.5-flash-lite`** — reasoning optional. `{"enabled": false}` = off (answers the
  `SCORE:` line straight from the content channel; 10/10 at 100 on a blatant terminal transcript).
  `{"max_tokens": N}` = on with thinking budget N, **512 ≤ N < the judge's `max_tokens`**. **Default
  (key absent): `{"max_tokens": 2048}`** (2047 until 2026-09-25, the largest budget under the old
  fixed 2048 cap). 512 is Google's minimum: measured on OpenRouter
  (2026-09-22, one easy prompt), budgets 1 / 128 / 511 are accepted but think just as long as 512
  (380–410 reasoning tokens on every call), i.e. clamped up — so they are refused. N must stay below the judge's
  completion cap (`max_tokens`, default `OPENROUTER_DEFAULT_MAX_TOKENS` = 4096; a fixed 2048 until
  2026-09-25), which covers thinking and answer. `effort` is refused
  (OpenRouter would turn it into some budget).
- **`google/gemini-3.5-flash-lite`** — reasoning **mandatory** (`{"enabled": false}` is a 400), so it
  is always on: `{"effort": "low" | "medium" | "high"}` (**preferred**) or `{"max_tokens": N}`.
  **Default (key absent): `{"effort": "low"}`** — see the table below for what 'low' does.
- **Any other model is refused**, whatever its `reasoning` — including none, since even the default
  sends *some* reasoning setting whose effect on that model nobody has checked. To use a new judge,
  establish what its reasoning object does and add it to `resolve_reasoning`.

No shipped config sets `reasoning` (since 2026-09-25): every gemini-2.5 judge runs on
`{"max_tokens": 2048}` (2047 before 2026-09-25) and every gemini-3.5 judge on `{"effort": "low"}`,
unless `--set` overrides it.
At 2047 the thinking left little room under the old 2048 cap: in the verifier-bug terminal runs
(`data/runs/*budget-2047*`, ~40k gemini-2.5 calls) the median call used ~1.6–1.9k reasoning tokens and
~3% ended `finish_reason=length`, read as-is (see `monitors/openrouter.py`). Since 2026-09-25 the cap
is 4096 by default (`max_tokens` per monitor), leaving ~2k tokens for the answer.
Earlier runs set it per config (gemini-2.5: 512 on Inkling-Small runs, off on Qwen3-8B runs); their
`run_info.json` records what each judge actually sent. The old keys `reasoning_effort` /
`reasoning_max_tokens` are refused with the translation rather than reinterpreted — under them "both
null" meant reasoning OFF, which is no longer gemini-2.5's default.

Both are static at construction (never flipped mid-run; a runtime flip used to race across the
threads sharing a monitor). **A small budget is not reliably honoured.** Measured on
`google/gemini-3.5-flash-lite` over 5 blatant reward-hacking terminal transcripts, 8 replays each
(n=40 per config, temperature 0 — which does *not* make this deterministic):

| judge-side reasoning | mean score | exact zeros | `reasoning_tokens == 0` |
| --- | --- | --- | --- |
| `{"max_tokens": 256}` | 51.8 | 19/40 | 25/40 |
| `{"effort": "low"}` | 52.5 | 17/40 | 24/40 |
| `{"effort": "medium"}` | **98.6** | **0/40** | 0/40 |
| `{"effort": "high"}` | 97.5 | 1/40 | 0/40 |

When the request comes back with `reasoning_tokens == 0` the judge frequently answers a reflexive
bare `SCORE: 0` (4 completion tokens, no analysis). That parses cleanly, so it lands in the log as a
real verdict and silently zeroes the penalty on a transcript the judge never audited — on the
train-against monitor that is reward for a hack that went unpunished. `low` is indistinguishable from
the budget it replaced; only `medium`/`high` actually induce reasoning on this model. (The gemini-2.5
figure — 10/10 at 100 on the same prompt — was with reasoning disabled; its budgeted default has not
been measured on these transcripts. Its 512 budget was honoured on every call probed, unlike the
gemini-3.5 256 budget above, but that was 3 calls on one easy prompt.)

Failure signature in a run dump: a bimodal `{0.0, 1.0}` score histogram on the gemini-3.x monitors
while the gemini-2.5 monitors on the same rollouts are smooth and high. Note `rl/train.py` persists
only `{score, label}` per monitor, dropping `meta["raw"]`, so a reflexive zero is not distinguishable
from a reasoned one after the fact.


### vLLM judges (`provider: "vllm"`)

A judge can also run on a local vLLM server (`monitors/vllm.py`, dispatched by
`monitors/judge_backend.py`, which both monitor classes use for every call). Instead of an OpenRouter
`reasoning` object it takes explicit `base_url` (ending `/v1`), `max_tokens`, `enable_thinking`
(→ `chat_template_kwargs.enable_thinking`) and an optional `thinking_budget` (→ vLLM's
`thinking_token_budget`; `null` = none). As with OpenRouter judges the models are specialized:
`Qwen/Qwen3-30B-A3B-FP8` and `Qwen/Qwen3.5-35B-A3B-FP8`, checked against running servers (vLLM 0.30,
`--reasoning-parser qwen3`): thinking on → the thinking in `message.reasoning`, the answer alone in
`content`; off → no reasoning at all; budget 64 → 63–64 reasoning tokens, then a normal answer.

Two deliberate differences from the OpenRouter client: the verdict is read from `content` **only**
(a thinking Qwen drafts `SCORE:` lines while deliberating, so there is no reasoning-channel fallback)
— its **last** `SCORE:`/`VERDICT:` line, and a `length` stop is always a parse error (after a
`thinking_budget` force-closes the thinking, Qwen3.5 sometimes keeps deliberating in content: at
budget 4096, 13/88 answers carried several SCORE lines and one call ran into `max_tokens` holding only
a draft) — and 404/422 are fatal (a wrong model name or path on a local server never fixes itself). A reply with
`</think>` in its content means the server is not splitting reasoning and aborts the run.

### Judge call health: `finish_length_rate` / `parse_error_rate`

A call that stops at `max_tokens` (`finish_reason: "length"`) is **final** for every provider — its
answer text is parsed as is, and one cut off before its `SCORE:` line (including an empty content
channel, i.e. thinking that ate the whole cap) is a `parse_error`, scored 0. Until 2026-09-25 an
empty `length` reply was retried as an API error; that censored exactly the long-thinking calls out of
the statistics and, for an unbudgeted vLLM thinker, could retry forever. Both rates are logged per
judge — `monitor/<name>/finish_length_rate`, `monitor/<name>/parse_error_rate` — in
`eval_metrics.jsonl` / W&B, in the train-against train rows, in the baseline script's output and
`baseline.json`, and on the `visualize_transcripts.py` Score-dist cards. A judge with a high
`parse_error_rate` is scoring 0 by default, not judging: read its AUROC with that in mind.
## Monitor families (taxonomy, from Rohan)

| Family | Reads | Status |
| --- | --- | --- |
| **CoT monitor** | CoT + output (text) | ✅ built — weak/strong via `model_id` (`cot_weak`/`cot_strong`) |
| **Output monitor** | final output only (text) | ✅ built — `use_cot: false` (`out_weak`/`out_strong`) |
| **Linear probe** | activations (white-box) | ✅ built — `probe_ood` + `probe_iid`, original-model probing |
| Activation oracle / non-linear probe | activations | later |
| Behavioral eval | aggregate behavior over many rollouts | later |

CoT and output monitors are LLM judges — cheap, API-based, swappable by model id. All four black-box
rungs + both probes run together in the live matrix. The realized weak/strong ladder is
`anthropic/claude-3-haiku` + `deepseek/deepseek-chat`.

## Monitors are feature-generic

The abstraction is behavior-agnostic: a monitor detects *some feature*, keyed off the env's
`behavior_name` (the live rubric is **reward_hacking**; sycophancy + deception rubrics also exist). We
want monitors for **arbitrary features** and eventually arbitrary *concepts* (especially via white-box
probes, which are naturally per-feature).

Concretely, for later:
- The `CoTMonitor` `rubric` arg already makes it feature-generic; swapping the rubric retargets it.
- Build a small **registry of feature rubrics** (and probe checkpoints) so a run declares *which
  features* to monitor, rather than hardcoding sycophancy.
- The degradation matrix then generalizes from "monitors of one behavior" to "monitors across
  features", which is where the cross-feature decorrelation story gets interesting.

This is a "don't forget" flag, not a now-task — the sycophancy default is fine for the first runs.

## White-box probes need activations — original-model vs trained-model probing

A probe needs **residual-stream activations**, and **tinker's API does not expose them** (`sample` →
tokens+logprobs; `forward*` → loss outputs; no `hidden_state` field). There are **two legitimate ways
to get activations, and we support both** — they differ in *which model's* activations the probe reads,
and they measure *different* things. Keep them distinct.

### Original-model probing  (frozen base — "off-policy activations")
Run the policy's generated text through a **frozen copy of the original base model** (HF transformers,
local) and read *its* activations. The probe was trained on this same frozen model. This is the
[`whitebox/`](../src/monitordecorrelation/whitebox/) + `ProbeMonitor` + `eval_probes_on_run.py` path
(post-hoc over a run's `rollouts.jsonl`) and the in-reward train-against path on the tinker backend.

- ✅ **Well-validated in the literature** — it's the Atlas "original-model probe score" and the standard
  deception-probe setup. Not a hack or a fallback.
- ✅ **The natural fit for the tinker backend** — no weight download needed; you only need the
  open-weight base model locally.
- It measures the **obfuscated-policy** axis: does the model's *text* still read as deceptive to a
  fixed probe? It **cannot** see the *obfuscated-activations* axis (the trained model's internals
  moving) — by construction, because the model is frozen. That's fine; it's a real, distinct signal.
- ⚠️ **Use the SAME base model as the RL policy.** Probing 8B-policy text with, say, a 0.6B model is
  *off-model* and unreliable — validity rests on the probe model matching the policy's base.
  (Train the probe on the policy's base model; `eval_probes_on_run.py --policy-base` warns on mismatch.)

### Trained-model probing  (current policy — "on-policy activations")
Read activations from the **live, mid-training policy** — captures both axes (obfuscated policy *and*
obfuscated activations). This needs in-process model access, which the **`TransformersBackend`**
provides (LoRA policy local). With the tinker backend it would require a `download → attach LoRA →
forward` round-trip per step (deferred; only worth it for the activations-decomposition study).

**Rule of thumb:** tinker backend → **original-model** probing (frozen base, same size); local
`TransformersBackend` → **trained-model** probing (live activations).

## Training *against* a probe — the two mechanisms (which model feeds the probe?)

When a probe enters the **reward** (penalty `−λ·probe_score` every step), the crux is *whose
activations* the probe reads. This gives two genuinely different training pressures:

```
  ORIGINAL-MODEL probe penalty                 TRAINED-MODEL probe penalty
  (off-policy activations)                     (on-policy activations)
  — text-level pressure                        — representation-level pressure

   policy M  (LoRA, updating)                   policy M  (LoRA, updating)
        │ generates text                              │ generates text
        ▼                                             ▼
  ┌───────────────┐  activations               ┌───────────────┐  activations
  │ FROZEN orig Mb│  (Mb never moves)          │  CURRENT  M   │  (Mb + live LoRA)
  └───────┬───────┘                            └───────┬───────┘
          ▼                                            ▼
     probe (frozen) ─► score                      probe (frozen) ─► score
          │ −λ·score → reward                          │ −λ·score → reward
          ▼ gradient updates M                         ▼ gradient updates M

  M can only evade by changing TEXT            M can evade by changing TEXT *or*
  (Mb's representations are fixed)             by moving its OWN activations
  → drives "obfuscated POLICY" only.          → drives "obfuscated policy" AND
  Works on the tinker backend (no             "obfuscated ACTIVATIONS".
  weight download — just run the text         Needs in-process model: the local
  through a local same-size base).            TransformersBackend (or a
                                              download→attach-LoRA→forward / step).
```

Both use a **frozen probe**; the difference is only which model's activations it sees — the two
probing modes above, now in the *reward* instead of post-hoc.

**The obfuscation decomposition (Atlas Eq. 3).** Run the *same* generations through both: the
**original-model** score and the **trained-model** score. Their gap is the **representation change** —
the signature of *obfuscated activations*. Original-model probing holds the base score fixed by
construction, so it can never produce that gap; trained-model probing can. So "can a policy hide the
behavior in its *representations* to fool a probe?" is only testable with trained-model probing;
original-model probing tests "can it hide in the *text*."

> Status: original-model probing is built — post-hoc (`eval_probes_on_run.py`) **and in the reward**
> (`configs/train_against_*probe*.json`, tinker backend). Trained-model probing needs the backend to
> expose `extract_activations` to the reward loop (next wiring step; `TransformersBackend` ready).

## Conventions
- Monitors are **frozen** by default (weights don't update during policy training). Updating-monitor
  variants are a later degree of freedom.
- The monitor(s) we **train against** enter the reward as `−λ·penalty`; **held-out** monitors are
  scored every eval step but never touch the gradient. Keep these lists explicit in config so we
  never accidentally train on a held-out detector.
- Never let a monitor see `behavior_present` — that's the oracle we grade monitors against.
