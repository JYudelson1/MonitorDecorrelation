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

## ⚠️ The activations problem (top derisking item)

White-box monitors (probes, activation oracles) need **residual-stream activations** from the policy.
**Tinker's API does not expose them.** Surveyed this session:

- `sample()` → tokens + (optional) logprobs.
- `forward()` / `forward_backward()` → loss outputs.
- No `hidden_state` / `activation` field anywhere in `tinker.types`.

So we cannot read internals straight from the tinker-hosted policy. Likely workaround:

1. Use `save_weights_for_sampler` / `save_state` to **pull the policy weights** (base + LoRA) at a
   checkpoint.
2. Run a **local forward pass** (e.g. HF transformers) on the same tokens to extract activations.
3. Train / apply the probe on those local activations.

**This must be verified before committing to any white-box experiment (Exp 1, combined matrix).**
Open sub-questions:
- Can we download base+LoRA weights and reconstruct the exact forward pass locally (tokenizer,
  dtype, layer indexing all matching)?
- Is `forward_backward_custom` an alternative path to get intermediate signals without a local copy?
- Cost/latency of a local forward pass per eval batch.

Until this is settled, all black-box monitor work (Exp 2) proceeds unblocked.

## Conventions
- Monitors are **frozen** by default (weights don't update during policy training). Updating-monitor
  variants are a later degree of freedom.
- The monitor(s) we **train against** enter the reward as `−λ·penalty`; **held-out** monitors are
  scored every eval step but never touch the gradient. Keep these lists explicit in config so we
  never accidentally train on a held-out detector.
- Never let a monitor see `behavior_present` — that's the oracle we grade monitors against.
