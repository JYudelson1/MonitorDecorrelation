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
`messages` — i.e. the full prompt — `temperature`, `max_tokens`, `reasoning`), the client timeout,
and the **full response** — the assistant `message` (`content`, plus the judge's chain of thought
under `reasoning` / `reasoning_details` when the provider returns one), `finish_reason`, and the
response-level `id` / `model` / `provider` / `usage`. When a call was retried only the attempt that
succeeded is recorded (`attempts` says how many it took). `rl/train.py` writes it per monitor into
`rollouts.jsonl` and `eval_rollouts.jsonl` as `monitors.<name>.call` (via
`eval.rollout_dump.monitor_record`); the committed `eval_rollouts_slim.jsonl` keeps `{score, label}`
only. `visualize_transcripts.py` shows these saved calls per rollout — and says "unavailable" for
dumps written before this existed, rather than rebuilding a prompt from the run config (a rebuild
can silently differ from what the judge was sent if the repo has changed since the run).

### Judge-side reasoning: use `reasoning_effort`, not a token budget

Judges answer with one line (`SCORE: <n>`), so thinking is pure cost and reasoning is **off** by
default (`reasoning: {"enabled": false}`). Models that *mandate* reasoning — gemini-3.x — reject that
with a 400, so they must ask for reasoning explicitly. Two knobs, mutually exclusive:

- `reasoning_effort: "low" | "medium" | "high"` — **preferred**.
- `reasoning_max_tokens: <n>` — legacy budget, kept for configs that still set it.

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
the budget it replaced; only `medium`/`high` actually induce reasoning on this model. gemini-2.5
judges are unaffected (they run with reasoning disabled and scored 10/10 at 100 on the same prompt).

Failure signature in a run dump: a bimodal `{0.0, 1.0}` score histogram on the gemini-3.x monitors
while the gemini-2.5 monitors on the same rollouts are smooth and high. Note `rl/train.py` persists
only `{score, label}` per monitor, dropping `meta["raw"]`, so a reflexive zero is not distinguishable
from a reasoned one after the fact.

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
