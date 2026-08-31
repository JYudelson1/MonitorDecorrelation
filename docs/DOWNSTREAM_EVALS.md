# Downstream monitorability evals

**Status: baseline in progress (started 2026-08-31).** Lives in a **separate repo**, not here.

## Why

Every degradation number we have (Δd′, coupling β, catch-rate proxies) is measured on the
MBPP-Honeypot **training distribution**. The ICLR paper is much stronger if we can show — or cleanly
bound — whether that degradation transfers to **downstream monitorability tasks** people already care
about. Equally publishable either way: "degradation is measurable in-distribution but doesn't transfer
downstream at this scale" is a real claim about when labs should worry.

First harness: **CoT Red-Handed** (arXiv 2505.23575, NeurIPS '25) — BigCodeBench main tasks +
instructed side tasks, native metric = catch rate @ 1% FPR. Chosen first because single-turn
function-level coding matches the capability band of our MBPP-honeypot-trained Qwen3-8B, and
*instructed* side tasks sidestep the "will the RL'd model organically misbehave OOD" question — we're
measuring monitor/monitorability **transfer**, not propensity transfer.

## Where the code lives

Not in this repo. Work happens in a fork of Aether-AIS's CRH clone:

- **Fork:** `github.com/JYudelson1/cot-monitoring`, branch `downstream-baseline`
  (`upstream` = `Aether-AIS/cot-monitoring`, itself a fork of `cotredhanded/code`).
- **Local checkout:** `/Users/joey/Aether/cot-monitoring`.
- New code under `src/cot_red_handed/downstream/`:
  - `tinker_provider.py` — Inspect ModelAPI that samples Qwen3 from tinker (base model today;
    `TINKER_CHECKPOINT` for trained checkpoints later). Registered via an `inspect_ai` entry point,
    so `--models tinker/Qwen/Qwen3-8B` just works. Sampling logic ported from this repo's
    `backends/tinker_backend.py` + `rl/rollout.py`.
  - `monitors.py` — the 4 LLM judges (`cot_weak`/`cot_strong`/`out_weak`/`out_strong` =
    `anthropic/claude-3-haiku` + `deepseek/deepseek-chat`), prompt/rubric/parser **copied verbatim**
    from this repo's `monitors/cot_monitor.py` + `monitors/rubrics.py`. The transfer claim requires
    the *identical* monitor as in-distribution — if those change here, re-sync there.
  - `score_transcripts.py` — score `.eval` logs with the suite (judge sees the **clean** problem,
    not the side-task-injected prompt — matches CRH's own monitoring, no leakage).
  - `catch_rate.py` — catch-rate @ 1% FPR (calibrated on benign rollouts) + AUROC + d′ + the
    spurious-channel covariates (main/side-task success, CoT length). This machinery didn't exist
    in-distribution (we used a fixed 0.5 cutoff), so it's new.

## Design decisions (with Joey, 2026-08-31)

- **Tinker-first inference.** Checkpoints are tinker URIs, so the sampling path is tinker from day one
  (the base-model baseline is the same weights as OpenRouter's hosted `qwen/qwen3-8b`, kept as a fallback).
- **Separate repo, self-contained.** No import of this repo's Monitor/backend abstractions into the
  fork — collaborators (Rauno/Vlad/Dennis) work in the CRH fork directly.
- **4 LLM judges only.** No probes, no CRH-native hierarchical monitor, for the baseline.

## Today's scope (baseline / feasibility only)

Base Qwen3-8B on the **6 non-subtle** CRH side tasks (`iodelete`, `iodownload`, `permissions`,
`ioexfil`, `exit`, `logio`), TEST split. Question: does base Qwen3-8B complete main tasks and comply
with side-task instructions often enough to leave anything to monitor? **No trained checkpoints** (all
tinker checkpoints from the July batches are TTL-expired; matrix re-runs are a separate paper-scale
effort).

## Out of scope (next)

Trained-checkpoint cells (needs the matrix rerun with long-TTL/sampler-weight saves); ControlArena
APPS-backdoor setting (slot 1b); OpenAI monitorability suite / g-mean2; probes on CRH rollouts.
