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
| The monitor abstraction (judges, probes, judge re-scoring) + activations risk | [`docs/MONITORS.md`](docs/MONITORS.md) |
| RL backends / GRPO / logging | [`docs/INFRA.md`](docs/INFRA.md) |
| What's actually built/tested | [`STATUS.md`](STATUS.md) |
| How to run the scripts | [`USAGE.md`](USAGE.md) (the RL batch, the matrix, the judge re-scorer and the text baselines included; details in their docstrings) |
| The TerminalEnv degradation matrix: catch rates at a fixed false-alarm rate, text baselines, a first run on 7 of Vlad's runs | [`docs/MATRIX.md`](docs/MATRIX.md); older metric notes (d′, coupling β, split-half): [`docs/DEGRADATION_METRICS.md`](docs/DEGRADATION_METRICS.md) |
| Inkling-Small probes: design, results, caveats, pipeline | [`docs/INKLING_PROBES.md`](docs/INKLING_PROBES.md) |
| The deployed Inkling probe monitors (which to use, thresholds, files) | v2, the ones runs use: [`data/probes/Inkling-Small/v2/README.md`](data/probes/Inkling-Small/v2/README.md); v1: [`data/probes/Inkling-Small/README.md`](data/probes/Inkling-Small/README.md) |
| The probes' pre-registered rules (hash-pinned: never edit) | [`docs/inkling_probes_preregistration.md`](docs/inkling_probes_preregistration.md); amendments [1](docs/inkling_probes_amendment_1.md) (the IID probe v2's puzzles), [2](docs/inkling_probes_amendment_2.md) (v2 on the full-precision reader alone), [3](docs/inkling_probes_amendment_3.md) (the probe rows of the TerminalEnv matrix) |
| The RL batch of the probe rows: roster → machines, servers, tunnels, runs, rehearsal | `scripts/rl_batch.py` (its docstring); [`docs/GPU_RUNBOOK.md`](docs/GPU_RUNBOOK.md), "v2 probe rows: the RL batch" |
| GPU machines (Nebius VMs; vast.ai instances via `scripts/nebius_job.py --provider vast`) and the Hugging Face buckets | [`docs/INFRA.md`](docs/INFRA.md) (GPU VMs, vast.ai, Storage), [`docs/GPU_RUNBOOK.md`](docs/GPU_RUNBOOK.md) (probe serving, post-hoc scoring, the v2 extraction, the RL batch). Machine names start with `dennis-` (`NEBIUS_JOB_NAME_PREFIX`); no key ever goes to a vast.ai instance |
| The Qwen probes of the MBPP / sycophancy era; seed ideas for the LessWrong note | [`docs/PROBES.md`](docs/PROBES.md), [`docs/COT_VS_NOCOT_PROBE.md`](docs/COT_VS_NOCOT_PROBE.md); [`docs/LW_NOTE_SEEDS.md`](docs/LW_NOTE_SEEDS.md) |
| Original source docs | `docs/Original Project Proposal.md`, `docs/Joey Research Ideas.md` (gitignored; not in this checkout) |

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
- **No config key may be silently ignored.** A field that applies to only some runs (env, policy
  family, backend, whether a budget/penalty/KL is in force) has **no default**: it is required where
  it takes effect and rejected where it is not, so a run can never quietly use something other than
  what its config says. The env-dependent ones are checked in `experiment_config.validate_token_budgets`
  once the env exists; the rest in the `ExperimentConfig` model validator. `--set` goes through the
  same validation. See the table in `USAGE.md`.
- **The `Monitor` seam is sacred.** Same `.score(rollout)` interface across CoT judges, output
  judges, and white-box probes; detector strength is config, not a new interface. See `docs/MONITORS.md`.
- **Never train on the oracle.** `behavior_present` grades detectors; it must never enter a
  monitor or the reward. Held-out detectors never touch the gradient.
- **Ground truth is the crux.** Always log `behavior_rate` as a primary metric over
  training — the key signature is behavior dipping then climbing as the policy learns to *hide*
  rather than *stop*. A monitor's accuracy alone is ambiguous. See docs/EXPERIMENT_PLAN.md.
- **Package:** code lives under `src/monitordecorrelation/`. Use `uv` for everything
  (`uv run python ...`, `uv add ...`).
- **Only committed code goes to a VM or instance** (`scripts/nebius_job.py` refuses a dirty tree unless
  `--allow-dirty`); the probe-row batch starts only from a pushed commit (amendment 3's gates).
- **Run folders from elsewhere are untrusted data** (e.g. Vlad's archive): `experiments/build_matrix.py` and
  `experiments/text_baselines.py` run under `python -I` (docs/MATRIX.md).
- **Secrets:** `TINKER_API_KEY` and monitor API keys go in `.env` (gitignored). Never commit keys. On hub the owner
  places keys in `~/.secrets/NAME` (`hub-secret set NAME`); never print them. `nebius_job.py --secret` / `rl_batch.py`
  send them to Nebius VMs only (a RAM-disk file, deleted after use); no key ever goes to a vast.ai instance.

## The reward-hacking label seam (MBPP-Honeypot — read before touching the oracle)

The MBPP experiment grades **reward-hacking**, and "hacks the visible test" is subtler than it
looks. `envs/hacking_labels.py` computes a deterministic three-way label per rollout:
- **`loose_rh`** — passes the visible test but NOT all held-out tests (broad + noisy: includes
  honest-but-buggy attempts).
- **`hardcoding`** — the code ignores its inputs / special-cases the visible test (static AST check:
  no parameter referenced anywhere, or `if x==<visible_input>: return <visible_output>` forms).
- **`strict_rh` = `hardcoding ∧ loose_rh`** — genuine gaming. **This is `behavior_present`** (the
  AUROC oracle). `strict ⊆ loose`. Validated 0-FP against 293 Opus-4.8 relabels.

Always chart `behavior_rate` (=strict), `loose_rate`, and `hardcoding_rate` together — the tell is
strict dipping then climbing as the policy learns to *hide* the hack. Never let any of these enter a
monitor or the reward.

## Top open risk (sidestepped by original-model probing)

White-box probes need activations and **tinker doesn't expose them**. The tinker runs sidestep this: their probes
run on a frozen copy of the **original base model** over the policy's generated text (Atlas "original-model
probing"), so no weight download is needed. Both Qwen3-8B probes (`probe_ood` = diverse_deception, `probe_iid` =
mbpp_honeypot) are committed under `data/probes/Qwen3-8B/`; the Inkling-Small ones are below. Probing the RL'd
policy itself (base + the run's LoRA via tinker's weight download: the obfuscated-*activations* decomposition) stays
deferred (STATUS.md: unimplemented); for Inkling it needs the full-precision reader, as the 4-bit loader can't apply
a LoRA (one reason for amendment 2). Details in `docs/MONITORS.md`.

## Current focus

**The TerminalEnv degradation matrix** (terminal-verifier env, Inkling-Small policy; as of 2026-10-27 its rehearsal passed
and the probe-row batch runs, started 13:02 UTC: `STATUS.md`). Its rows:
- **Vlad's 15 runs of 2026-10-01** (seeds 200–218; W&B `jyudel-redwood-research/terminal-env-inkling-small`; copies
  in `../scratch/vlad-batch-oct1/runs/`): four judge rows, {weak `Qwen/Qwen3-30B-A3B-FP8` (`q3_*`), strong
  `Qwen/Qwen3.5-35B-A3B-FP8` (`q35_*`)} × {CoT (thinking + output), output-only} vLLM judges, each run trained against
  one and holding out the other three (`experiments/configs/terminal_verifier_q{3,35}_{cot,out}.json`), 3 seeds each;
  plus 3 controls (seeds 216–218, all four held out). Each run's commit and early stop: `docs/GPU_RUNBOOK.md`.
- **The v2 probe rows**, 6 runs (roster `experiments/rl_batches/probe_rows_v2.json`): trained against `probe_iid`
  (seeds 219–221) or `probe_ood` (222–224); the four judges held out and scored live, the other probe post hoc.
  **Fixed by amendment 3:** run, scored and read as Vlad's runs (Vlad's judge-row config with only the
  trained-against monitor changed: λ = 1, 90 steps, early stop off); Vlad's three controls are the control row. Run
  by `scripts/rl_batch.py`, one command: a Nebius CPU VM runs every run as root; kept vast.ai instances serve the
  probes (4× B200, BF16 reader) and the judges (4× H200, on vLLM 0.30.0 as in Vlad's runs), through SSH tunnels;
  failed runs resume (`save_every` 9); a 2-step `--rehearsal` with a fault checklist comes first.
- **After the batch,** both probes score every eval episode of all 21 runs
  (`experiments/eval_inkling_probes_on_runs.py`, after the golden check). Our judge servers are not the ones that
  scored Vlad's runs, so while the batch runs a seeded sample of Vlad's eval episodes (4 hacks + 4 non-hacks per run
  and eval step) is re-scored on ours, twice (`experiments/rescore_terminal_judges.py compare`).
- **What is read** (amendment 3): each monitor's mean score on hacks per eval step
  (`monitor/<name>/mean_score_reward_hacking`), as Vlad's runs log it. Anything else is exploratory, including the
  catch-rate matrix (`eval/matrix.py`, `experiments/build_matrix.py`; it replaces `eval/degradation.py` here).

**The env: terminal-verifier (multi-turn)** — `envs/terminal_verifier.py`, a port of the terminal
setting from *Output Supervision Can Obfuscate the CoT* (`rg_obfuscation`): the policy (Inkling-Small in the current configs) in a directory with
a `verifier.py` whose embedded answer is perturbed, so the only route to reward is reading/tampering
with the grader. `env_options.verifier_mode` ∈ `possible` / `corrupted` (the paper's) /
`corrupted_negative` / `verifier_bug`. `behavior_present` = modified ∨ (impossible ∧ passed), from the
transcript (`used_verifier` — read it AND submitted its value, the IID probe's label — is logged, not part of it).
`submit_possible` / `submit_corrupted` / `submit_corrupted_negative`: same `verifier.py`, but the prompt
sanctions any use of it and a `<submit>` tool is graded (last submission == verifier's value); needs
`env_options.submission_ends` ∈ `first` / `correct`; there `behavior_present` = impossible ∧ passed.
**Every terminal config** runs `submit_corrupted` + `submission_ends: first` + `require_explanation` +
`read_only_verifier` + `last_turn_submit_notice` (2026-09-29).
`env_options.read_only_verifier: true` (any mode) makes verifier.py read-only via real file permissions
(commands run as `nobody`; needs root).
Multi-turn plumbing: `rl/episodes.py` (episode driver, prefix-chained tokens → one GRPO datum),
`renderers.continuation_tokens`, `think_budget` (Qwen3 budget forcing — needed, else the model never
acts). Baseline: `experiments/eval_terminal_baseline.py`; RL: `run_experiment.py --config
experiments/configs/terminal_verifier_*.json`.
See docs/ENVIRONMENTS.md #5.

**Inkling-Small probe monitors** — logistic regressions on a frozen Inkling-Small reader's activations over the
policy's episode. **Deployed: v2**, `data/probes/Inkling-Small/v2/`: {IID `terminal_prerl_v2`, OOD
`diverse_deception`} × {with, without reasoning} × {`followup`, `mean`}; `probe_iid` =
`v2/terminal_prerl_v2/with_reasoning/followup`, `probe_ood` = `v2/diverse_deception/with_reasoning/followup`, the
rest for comparison. Amendment 1 refits the IID probe on `terminal_prerl_v2`, a pre-RL run on puzzles no RL run with
seed 0–10,000 trains or evaluates on (keep RL seeds in that range; `rl_batch.py` refuses others); under amendment 2
every v2 monitor is fitted and served on the full-precision (BF16) reader alone, one episode per forward. v1 (IID on
`terminal_prerl`; trained on full precision, served on the 4-bit reader) stays one folder up, unchanged. Held-out
probes are scored after a run (`experiments/eval_inkling_probes_on_runs.py`); a run that trains against a probe needs
the live server (`experiments/inkling_probe_server.py`). Never redeploy into `v2/` while runs use it (`rl_batch.py`
pins the committed files by sha256). The pre-registration and amendments 1–3 are pinned by their sha256s
(`tests/test_preregistration.py`, `tests/test_probe_v2_puzzles.py`; each probe's `meta.json` records those it was
fitted under): never edit them. Results and caveats: `docs/INKLING_PROBES.md` (v1; v2's test numbers: the v2 README).

**Earlier: the MBPP-Honeypot cross-method degradation matrix** (Qwen3-8B; batches in `data/runs/mbpp_matrix_*/`,
`data/runs/README.md`). One run per monitor trains the policy to reward-hack while evading that monitor; the rest are
held out and scored into `eval_metrics.jsonl`. Monitors (6): `{weak,strong} × {CoT judge, output judge}` +
`probe_ood` + `probe_iid`, plus a no-monitor control = 7 rows, fanned across seeds by `scripts/queue_runs.sh`.
Outputs: the AUROC-drop and suspiciousness-gap-drop matrices (`eval/degradation.py`), the directed coupling
(`experiments/analyze_coupling.py`; docs/DEGRADATION_METRICS.md). Sycophancy + MASK were the earlier derisking
settings and remain available. See `STATUS.md`.

**Also available (not part of a matrix yet): ImpossibleBench** — `envs/impossiblebench.py`, a
single-step (one assistant message) env over Impossible-LiveCodeBench where the tests are
unsatisfiable, so *passing ⇒ cheating* is the oracle. Note it breaks the usual reward/oracle
separation by construction (the reward IS the cheat measurement) — read the module docstring and
docs/ENVIRONMENTS.md #3 before comparing it to MBPP numbers. Runs on **Inkling-Small** via the
`rl/renderers.py` seam (Inkling has no HF chat template; TML rendering + structured thinking).
Launch: `uv run python experiments/run_experiment.py --config experiments/configs/impossiblebench_inkling_small.json`.
