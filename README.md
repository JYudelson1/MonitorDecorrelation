# Monitor Decorrelation

*Investigating correlation (and correlated degradation) of monitoring surfaces under training pressure.*

Part of the [Aether](docs/RESEARCH_CONTEXT.md) project proposals, June 2026 

## High-level goal

We are excited about doing alignment science on training methods that appear to be part of the
short-timelines alignment plan. Labs are deploying those methods faster than the field understands
them. We want to do the basic science needed to (a) understand why these methods work when they
work, (b) ensure they don't silently degrade the held-out evals we'll need to trust them, and (c)
improve the most promising ones.

## The question

The default alignment plan looks like it will rely on **defense-in-depth**: we train on proxies for
the thing we want — or *against* proxies of misalignment — while keeping some **held-out evals to
validate** our methods. For that to work, our monitors/evals need to be **decorrelated**, especially
in the sense that *training on proxy X doesn't degrade the ability of proxy Y to keep monitoring the
model.* It would be really bad if labs thought they had held-out evals, but their training silently
degraded those evals' usefulness.

So: take a misalignment proxy/detector **X** and train against it. Two things should happen —

1. **X becomes useless as a detector**, and
2. **the model becomes more aligned** on some distribution, maybe generalizably.

We want to empirically map what's in (1) vs. (2) across many proxies, and — crucially — measure
whether degradation of X **correlates** with degradation of held-out detectors Y, Z, … When we train
against X, do other detectors degrade in lockstep, or do they remain informative?

**Headline output:** a **detector-degradation correlation matrix** — how much training against each
proxy degrades each other proxy.

## Where things are now

- **Headline experiment (running):** the **cross-method** degradation matrix — train Qwen3-8B to
  reward-hack while evading each of 6 monitors in turn (`{weak,strong} × {CoT judge, output judge}` +
  an OOD deception probe + an iid reward-hack probe), plus a no-monitor control. Outputs: AUROC-drop +
  suspiciousness-gap-drop grids.
- **Setting:** **MBPP-Honeypot reward-hacking** (`behavior_present` = *strict* reward-hacking =
  hardcoding ∧ passes-visible-not-all). Sycophancy + MASK deception were the earlier derisking settings.
- **Infra:** all RL runs on [tinker](https://tinker-docs.thinkingmachines.ai/) with a minimal
  hand-rolled loop (cookbook primitives for the loss/KL layer); `scripts/queue_runs.sh` fans the matrix
  across seeds.

## Docs

This is a context-heavy project. The detailed context lives in `docs/` so it survives across work
sessions:

| Doc | What's in it |
| --- | --- |
| [`docs/RESEARCH_CONTEXT.md`](docs/RESEARCH_CONTEXT.md) | Research framing, related work, the two senses of "decorrelation", links. |
| [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md) | Concrete experiment designs, metrics, the degradation matrix. |
| [`docs/ENVIRONMENTS.md`](docs/ENVIRONMENTS.md) | The RL environment catalog (sycophancy, coding reward-hacking, ImpossibleBench, LiveCodeBench). |
| [`docs/MONITORS.md`](docs/MONITORS.md) | The generic monitor abstraction (black-box + white-box) and the tinker-activations question. |
| [`docs/INFRA.md`](docs/INFRA.md) | RL backend abstraction (tinker now, verl later), GRPO, and local-wandb logging. |
| [`USAGE.md`](USAGE.md) | The runnable scripts (entrypoints) and how to run them. |
| [`STATUS.md`](STATUS.md) | Living implementation checklist — what's built, tested, and correct. |
| [`CLAUDE.md`](CLAUDE.md) | Orientation + conventions for anyone (incl. future Claude sessions) picking this up. |

The original source documents are in [`docs/Original Project Proposal.md`](docs/Original%20Project%20Proposal.md)
and [`docs/Joey Research Ideas.md`](docs/Joey%20Research%20Ideas.md).

## Setup

```bash
uv sync                      # install deps (tinker SDK)
cp .env.example .env         # then add your TINKER_API_KEY and monitor API keys
```
