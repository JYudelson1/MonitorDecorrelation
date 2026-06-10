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

## Monitor families (taxonomy, from Rohan)

| Family | Reads | Status target |
| --- | --- | --- |
| **CoT monitor** | the reasoning trace (text) | first — weak/strong via `model_id` |
| **Output monitor** | final output only (text) | early |
| **Linear probe** | activations (white-box) | needs activations path (below) |
| Activation oracle / non-linear probe | activations | later |
| Behavioral eval | aggregate behavior over many rollouts | later |

CoT and output monitors are LLM judges — cheap, API-based, swappable by model id. That's why Exp 2
(monitor *strength*) is first: it's all black-box, no activation plumbing.

## Monitors are feature-generic (flagged for later)

Right now the only monitor is a CoT judge with a **sycophancy** rubric, because sycophancy is the
first setting. But the abstraction is deliberately behavior-agnostic: a monitor detects *some
feature*, and sycophancy is just feature #1. We will want monitors for **arbitrary features** —
other misbehaviors (reward hacking, deception), and eventually arbitrary *concepts* (especially via
white-box probes, which are naturally per-feature).

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
