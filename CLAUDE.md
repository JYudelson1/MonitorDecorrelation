# CLAUDE.md — orientation for this repo

Read this first if you're (re-)starting work on Monitor Decorrelation. This is a **context-heavy**
project; the docs exist so context survives across sessions. Don't get ahead of the plan.

## What this project is (one paragraph)

We study **monitor decorrelation under training pressure**: when you RL a policy against a
misalignment detector X, does X just stop working, does the model actually get more aligned, and —
the crux — do *held-out* detectors Y, Z degrade in lockstep? Headline output is a
**detector-degradation correlation matrix**. Full framing in
[`docs/RESEARCH_CONTEXT.md`](docs/RESEARCH_CONTEXT.md).

## Where to look

| If you need… | Read |
| --- | --- |
| Why / prior art / links | [`docs/RESEARCH_CONTEXT.md`](docs/RESEARCH_CONTEXT.md) |
| Concrete experiments + metrics | [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md) |
| RL environments | [`docs/ENVIRONMENTS.md`](docs/ENVIRONMENTS.md) |
| The monitor abstraction + activations risk | [`docs/MONITORS.md`](docs/MONITORS.md) |
| RL backends / GRPO / logging | [`docs/INFRA.md`](docs/INFRA.md) |
| What's actually built/tested | [`STATUS.md`](STATUS.md) |
| How to run the scripts | [`USAGE.md`](USAGE.md) |
| Original source docs | `docs/Original Project Proposal.md`, `docs/Joey Research Ideas.md` |

## Conventions

- **Status taxonomy.** Every component in `STATUS.md` is one of:
  `⬜ unimplemented` / `🟡 implemented-untested` / `🔴 tested-buggy` / `🟢 tested-correct`.
  Update `STATUS.md` whenever you change a component's state. This is the source of truth for build
  progress — keep it honest (don't mark 🟢 without an actual test).
- **Minimal tinker.** RL runs on [tinker](https://tinker-docs.thinkingmachines.ai/) with a small,
  hand-rolled loop on the raw SDK (`ServiceClient` → `create_lora_training_client` → sample →
  `forward_backward` → `optim_step`). We keep our own loop, but **use tinker-cookbook *primitives*
  where it ships the canonical implementation** (e.g. `tinker_cookbook.rl.metrics.incorporate_kl_penalty`
  for the KL-to-base penalty) rather than reinventing them — don't adopt the cookbook's training loop.
  The online tinker docs are good — consult them rather than guessing the API.
- **The `Monitor` seam is sacred.** Same `.score(rollout)` interface across CoT judges, output
  judges, and white-box probes; detector strength is config, not a new interface. See `docs/MONITORS.md`.
- **Never train on the oracle.** `behavior_present` grades detectors; it must never enter a
  monitor or the reward. Held-out detectors never touch the gradient.
- **Ground truth is the crux.** Always log `behavior_rate` as a primary metric over
  training — the key signature is behavior dipping then climbing as the policy learns to *hide*
  rather than *stop*. A monitor's accuracy alone is ambiguous. See docs/EXPERIMENT_PLAN.md.
- **Package:** code lives under `src/monitordecorrelation/`. Use `uv` for everything
  (`uv run python ...`, `uv add ...`).
- **Secrets:** `TINKER_API_KEY` and monitor API keys go in `.env` (gitignored). Never commit keys.

## Top open risk

White-box probes need activations, and **tinker doesn't expose them** — likely need to pull weights
and run a local forward pass. Verify before building any white-box experiment. Details in
`docs/MONITORS.md`.

## Current focus

Derisking week. Order: (1) docs ✅, (2) package skeleton + `Monitor`/`Env` protocols, (3) sycophancy
RL smoke test on tinker (black-box CoT monitor), (4) Exp 2 monitor-strength read. See `STATUS.md`.
