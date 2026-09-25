# Usage — executable entrypoints

The runnable scripts (everything else under `src/` is library code). All run via `uv`; secrets load
from `.env` (`TINKER_API_KEY`, `OPENROUTER_API_KEY`, …).

> **Real runs are GPU (rented CUDA box); local MPS is for plumbing only.** The current experiment is
> **Qwen3-8B** — RL on tinker (remote) + white-box probe forward passes on the base model. Probe scoring
> at 2048-token sequences is GPU-preferred; on a CUDA machine `uv sync --extra gpu` adds the optional
> kernels (flash-attn etc., Linux-gated so a plain `uv sync` on a Mac skips them). Full step-by-step in
> [`docs/GPU_RUNBOOK.md`](docs/GPU_RUNBOOK.md).
>
> *(Legacy caveat for the Qwen3.5-4B env: its hybrid linear-attention OOMs MPS at 4096 tokens —
> single-sequence O(T²), ~27 GiB; `batch_size=1` doesn't help, only shorter sequences do. Not on the
> Qwen3-8B path.)*

| Script | What it does | Outputs |
| --- | --- | --- |
| `experiments/run_experiment.py` | **The RL experiment runner.** GRPO with any mix of monitors (CoT judges + white-box probes), each train-against or held-out, from a validated JSON/YAML config. | `data/runs/<run>/` + wandb |
| `scripts/queue_runs.sh` | **Batch launcher.** Fan a config (or dir) × seeds `0..N-1` into ≤J parallel runs; per-job logs, resume (`-s`), detach (`-D`), shared probe server (`-P`). | one `data/runs/<run>/` per (config×seed) |
| `experiments/probe_server.py` | Shared frozen-model activation server so parallel runs don't each load the base model (`queue_runs.sh -P` starts it). | localhost server |
| `experiments/make_mbpp_matrix_configs.py` | Generate the matrix configs (`--penalty-coef` xor `--penalty-schedule`, `--tag`, `--n-steps`) → `configs/mbpp_matrix[_lowpen]/row_*.json`. | matrix config files |
| `experiments/analyze_coupling.py` | **The headline analysis.** Directed detector-coupling β(A→B) over a set of runs → pooled (7b) + conditioned-on-target (10) heatmaps + CIs. | `data/runs/_coupling/*.png` |
| `scripts/verify_runs.py` | **Post-launch check.** Does each run actually train against the monitor its name claims? Also checks judge reasoning settings, a shared monitor battery + hyperparameters across the batch, and backend queue pauses. Exits 1 on any problem. | stdout |
| `experiments/train_probe.py` | Train a white-box linear probe on contrastive deception/hacking data (base model, Atlas §4.3). `--preset {simple_deception,diverse_deception,mbpp}`. | `data/probes/<model>/<datasets>/` |
| `experiments/build_syco_pairs.py` | Build real-CoT sycophancy contrastive pairs (resample policy, label by ground-truth/judge/both) for the on-domain probe. | a `*.jsonl` of pairs |
| `experiments/test_many_monitors.py` | Monitor bakeoff: policy behavior frequency + each candidate monitor's detection AUROC (class-balanced). | `data/runs/<name>/monitor_bakeoff.json` |
| `experiments/eval_probes_on_run.py` | Post-hoc: score a probe over a run's `rollouts.jsonl`, per step (original-model probing). | `data/runs/<run>/probe_eval_<name>.jsonl` |
| `experiments/eval_terminal_baseline.py` | **No-training baseline** on the terminal-verifier env: sample real multi-turn episodes, print peek/tamper/pass rates + transcripts, optionally judge AUROC (`--judge`). | `data/runs/tv_baseline_<model>/episodes.jsonl` |
| `scripts/plot_run.py` | Render a run's metrics to PNGs locally (no W&B server). | `data/runs/<run>/*.png` |
| `visualize_transcripts.py` | **Local run viewer.** Serves a web UI over `data/runs/`: every run's config + monitors, all `metrics.jsonl` / `eval_metrics.jsonl` series (charted + tabulated), per-monitor **score histograms split by `behavior_present`** at one selectable RL train step or eval step (or the single step of an `eval_terminal_monitors_baseline.py` dir), and the **full untruncated rollouts** (prompt, CoT, answer, per-turn transcript, env grading record, every monitor's score, and — for every LLM judge whose call the run saved — the exact prompt, request parameters and response incl. the judge's chain of thought; older dumps show "unavailable"). Re-scans on refresh, so new runs and new steps appear while a run is training; a dump rewritten in place (e.g. by `eval_terminal_monitors_baseline.py`) is detected and re-indexed, never mistaken for an append. Stdlib-only. | `http://127.0.0.1:8000` |
| `tests/test_*.py` | Offline unit tests (env, GRPO alignment, metrics, probe logic). | stdout (run via `uv run pytest`) |

## The experiment runner (config-driven)

A run is fully described by a **schema-validated** JSON or YAML config (see `experiments/configs/`):
policy, env, RL hyperparams, and a `monitors` list of mixed kinds (`cot` / `probe`), each
`role: train_against` or `held_out`. A malformed config is rejected at load (pydantic, extra keys
forbidden). So "train against CoT", "train against a probe", "train against strong CoT while holding
out weak CoT + two probes" are all just different configs — no per-experiment scripts.

```bash
uv run python experiments/run_experiment.py --config experiments/configs/mbpp_matrix_sep18/row_control.json
uv run python experiments/run_experiment.py --config experiments/configs/terminal_verifier_control.json
# quick one-off overrides of top-level fields:
uv run python experiments/run_experiment.py --config <cfg> --set run_name=quick n_steps=2 batch_size=2
# per-monitor overrides: monitors.<name|model:substr|*>.<field> (a {…}/[…] value is parsed as JSON)
uv run python experiments/run_experiment.py --config <cfg> \
    --set 'monitors.model:gemini-3.5.reasoning={"effort":"medium"}' run_name=<...>_eff-medium
```

Config fields (top level): `run_name experiment description policy backend{tinker,transformers}
env{sycophancy,mbpp_honeypot,impossiblebench,terminal_verifier} subset env_options n_steps batch_size
group_size eval_every eval_size eval_samples_per_prompt max_tokens think_budget answer_tokens
thinking_effort penalty_coef penalty_schedule kl_coef kl_discount_factor lora_rank lr seed
n_prompts_pool probe_server_url monitors`.
`monitors: []` is legal and means a pure-task-reward run (no monitor in the gradient, none measured).
`env_options` holds env-specific constructor kwargs (impossiblebench + terminal_verifier).

**No key is ever silently ignored.** A field that only applies to *some* runs has no default: it is
**required** where it takes effect and **rejected** where it does not, so a config can never describe
something the run then quietly ignores. The conditional fields:

| field | required when | rejected when |
| --- | --- | --- |
| `subset` | env is sycophancy / impossiblebench / terminal_verifier | env is mbpp_honeypot (no slices) |
| `thinking_effort` | policy is `thinkingmachines/*` (TML-rendered) | any other policy — no such knob |
| `max_tokens` | a turn is one sampling call: every single-turn env, or a multi-turn env with `think_budget: null` | a thinking budget is in force (the budget + `answer_tokens` size the two calls) |
| `answer_tokens` | a thinking budget is in force | no budget — nothing ever forces an answer |
| `think_budget` | — (defaults to `"auto"` = the env's own) | single-turn envs, which never budget a turn |
| `penalty_coef` / `penalty_schedule` | exactly one, on a run with a `train_against` monitor | a control run (no monitor penalty exists to weight) — repo controls spell it `"penalty_coef": null`; `run_grpo` re-checks this for a hand-built `RunConfig` |
| `kl_discount_factor` | `kl_coef` > 0 | `kl_coef` == 0 (nothing to discount) |
| `probe_server_url` | — | no monitor is a probe |

The env-dependent three (`max_tokens` / `answer_tokens` / `think_budget`) are checked once the env is
built, so the error names the env; the rest fail at config load. `--set` goes through the same
validation, so `--set max_tokens=4096` on a budgeted run is an error, not a no-op. An archived run's
`config.json` written before these rules may no longer load for a *run*; re-scoring scripts read only
its `monitors` (`load_monitor_specs`) and are unaffected.

Each monitor: `{kind:"cot", name, role, model_id,
use_cot?, use_output?, behavior?, threshold?, reasoning?, binary_judge?}` (output monitor = `use_cot:false`) or
`{kind:"probe", name, role, probe_path, probe_model?, threshold?, batch_size?}`.

**Judge-side reasoning (`reasoning`).** The OpenRouter `reasoning` object every call of that judge
sends, validated **per judge model** at config load (`monitors/judge_reasoning.py`), naming the monitor:

| judge `model_id` | accepted `reasoning` | absent (default) |
| --- | --- | --- |
| `google/gemini-2.5-flash-lite` | `{"enabled": false}` (off) · `{"max_tokens": N}`, 512 ≤ N < 2048 (on, thinking budget N) | `{"max_tokens": 2047}` — on, largest budget |
| `google/gemini-3.5-flash-lite` | `{"effort": "low"\|"medium"\|"high"}` (preferred) · `{"max_tokens": N}` — it mandates reasoning (`enabled:false` is a 400) | `{"effort": "low"}` |
| **any other model** | **error** | **error** |

512 is Google's minimum budget: a smaller one is accepted but clamped up (measured: budgets 1/128/511
think as long as 512), so it is refused rather than let mean something else. Any other judge model is
refused outright — even with no `reasoning` key — because what a reasoning setting does on it has not
been established; implement its support in `resolve_reasoning` first. No shipped config sets it, so
every judge runs on its model's default unless overridden with `--set`. The old keys
`reasoning_effort` / `reasoning_max_tokens` are refused with the translation (both null used to mean
reasoning OFF). Pick it per run without forking the config — the value replaces the whole object:

```bash
--set 'monitors.g35_out.reasoning={"effort":"medium"}'            # one judge, by name
--set 'monitors.model:gemini-3.5.reasoning={"effort":"medium"}'   # every gemini-3.5 judge (keeps the matrix
                                                                  # rows comparable — see docs/MONITORS.md)
--set 'monitors.model:gemini-2.5.reasoning={"enabled":false}'     # gemini-2.5 judges: reasoning off
--set 'monitors.model:gemini-2.5.reasoning={"max_tokens":1024}'   # … or on, with a bigger budget
--set monitors.g25_out.reasoning=null                             # back to the model's default
```

So the older configs whose judges are claude / deepseek models — `matrix8b/`, `mbpp_matrix/`,
`mbpp_matrix_lowpen/`, `mbpp_matrix_sched01/`, `full_matrix.json`, `train_against_*.json` — and what
`make_matrix_configs.py` / `make_mbpp_matrix_configs.py` generate **no longer load** until those models
get reasoning support (the `queue_runs.sh` examples below still name some of those dirs).

Through `scripts/queue_runs.sh -- …` write the JSON **without spaces** (the extra overrides are
word-split). `run_info.json` records the resolved object each judge actually sent.

⚠️ On `gemini-3.5-flash-lite`, `low` is **not** a working setting: ~60% of calls come back with
`reasoning_tokens == 0` and the unthinking judge answers a reflexive bare `SCORE: 0`, which parses as
a real verdict (17/40 on blatant hacks, vs 0/40 at `medium`). Table + failure signature in
[`docs/MONITORS.md`](docs/MONITORS.md).

For running the full matrix across seeds, use `scripts/queue_runs.sh` (next section) rather than a
hand-rolled loop.

### Queueing many runs (seeds × configs, in parallel)

`scripts/queue_runs.sh` fans a config (or a whole directory of configs) across seeds `0..N-1` and runs
up to `J` at a time (`xargs -P`; BSD/GNU compatible). Each `(config × seed)` becomes its own run:
`run_name`'s `_s<n>` token is rewritten to `_s<seed>`, so every job lands in its own
`data/runs/<run>/` with its rollouts/metrics/plots. Full per-job output → `data/runs/<run>/run.log`;
the terminal shows one ▶/✓/✗ line per job and tails the log of any job that fails.

```bash
# the whole 7-row matrix, seeds 0-2, 4 in parallel:
scripts/queue_runs.sh -c experiments/configs/mbpp_matrix -n 3 -j 4
scripts/queue_runs.sh -c experiments/configs/mbpp_matrix -n 3 -j 4 -d   # -d = dry-run (print plan only)
scripts/queue_runs.sh -c experiments/configs/mbpp_matrix -n 3 -j 4 -s   # -s = resume: skip finished runs
scripts/queue_runs.sh -c experiments/configs/mbpp_matrix -n 3 -j 4 -D   # -D = detach: nohup the batch,
                                                                       #   survives SSH disconnect
# ⭐ recommended for a real batch: -s -P -D (resume-safe · shared probe server · detached), high -j:
scripts/queue_runs.sh -c experiments/configs/mbpp_matrix_lowpen -n 5 -j 12 -s -P -D
scripts/queue_runs.sh -c <cfg> -U http://127.0.0.1:8177 ...   # -U = reuse an already-running probe server
# extra --set overrides after `--` apply to every job (e.g. a fast smoke of the whole matrix):
scripts/queue_runs.sh -c experiments/configs/mbpp_matrix -n 1 -j 2 -- n_steps=3 batch_size=4
```

#### Shared resource caps across parallel runs

Two caps are **cross-process**, so they hold whether you fan out with `queue_runs.sh` or just start
runs by hand in separate terminals / tmux windows — you do *not* need to divide the per-run worker
counts by the number of runs:

| Resource | Cap | Set in |
| --- | --- | --- |
| Concurrent model-code subprocesses | half this box's cores | `CODE_EXEC_MAX_CONCURRENT` in `src/monitordecorrelation/globalsem.py` |
| Concurrent in-flight OpenRouter calls | 1024 | `OPENROUTER_MAX_CONCURRENT` in the same file |

Change either number at that one assignment; nothing else reads a copy. The same two semaphores
cover training and evaluation (every judge call goes through `openrouter.chat`; every code execution
through `code_exec.run_tests`, `TerminalSession.run` or the ImpossibleBench executor).

A permit is an exclusive `flock` on a slot file under `$MD_GLOBAL_SEM_DIR` (default
`/tmp/monitordecorrelation-sem` — keep it on a local filesystem). That means the kernel takes the
permit back however a run ends: `Ctrl-C`, a crash, `kill -9`, the OOM killer, or the machine losing
power. There is no stale state to clean up, and nothing to reset after a bad batch. If you *do* want
two groups of runs to have independent budgets, give them different `MD_GLOBAL_SEM_DIR` values.

A run is marked done by a `data/runs/<run>/QUEUE_DONE` sentinel (written on success) — that's what `-s`
checks. Follow a single live job with `tail -f data/runs/<run>/run.log`. With `-D`, the plan still
prints to your terminal, then the batch re-execs under `nohup` into `data/runs/_batch_<ts>.log` and
returns a PID immediately (follow it with `tail -f data/runs/_batch_<ts>.log`).

**Shared probe server (`-P`).** Probes read activations by running the policy's text through a frozen
copy of the base model. Without `-P`, *each* run process loads its own ~16 GB copy, so `-j` is capped by
GPU memory (~4 on an 80 GB card). `-P` starts one `experiments/probe_server.py` (model read from the
first config's `policy`, port `$PROBE_PORT`, default 8177), points every run at it via
`PROBE_SERVER_URL`, and tears it down when the batch ends — so `-j` is then bounded by tinker/API limits
instead of memory. `-U http://host:port` reuses an already-running server. Per-config equivalent:
`{"probe_server_url": "..."}`. Start one standalone with
`uv run python experiments/probe_server.py --model Qwen/Qwen3-8B`.

### Analyzing a run set — directed coupling (the headline)

```bash
uv run python experiments/analyze_coupling.py 'data/runs/mbpp_Qwen3-8B_*lowpen' --bootstrap 3000
uv run python experiments/analyze_coupling.py 'data/runs/mbpp_*' --metric dprime_margin --out data/runs/_coup
```
β(A→B) = per unit detector A's reliability (d′) degrades within a run, how much does B's — the DIRECTED,
asymmetric coupling. Writes `coupling_pooled.png` (all runs, driver→responder) + `coupling_by_target.png`
(conditioned on the training target: *does training against i degrade held-out j?*) + `coupling.json`.
`--metric auroc` (→ d′, resolution-aware clip) or `dprime_margin` (native, non-saturating); `--bootstrap N`
adds per-cell 90% CIs. Reads `eval_metrics.jsonl` (+ dir name for the target). See `docs/DEGRADATION_METRICS.md`.

**W&B syncing is automatic iff you're logged in.** `run_experiment.py` picks `wandb_mode` per run:
`online` if a credential is configured locally (`wandb login` → `~/.netrc`, or `WANDB_API_KEY` set),
else `offline`. The check is purely local (no network, never prompts), so it's disconnect-safe. Set
`WANDB_MODE` explicitly (`online`/`offline`/`disabled`) to override. The chosen mode is printed in each
run's banner. Offline runs lose nothing — upload later with `wandb sync wandb/offline-run-*`.

## ImpossibleBench × Inkling-Small

```bash
IB=experiments/configs/impossiblebench_inkling_small.json

# full run (40 steps, 16x8 rollouts/step)
uv run python experiments/run_experiment.py --config $IB

# smoke test first (2 steps, tiny batch, short completions) — same code path, minutes not hours
uv run python experiments/run_experiment.py --config $IB --set run_name=ib_smoke n_steps=2 \
    batch_size=2 group_size=4 eval_size=2 eval_every=1 max_tokens=2048 thinking_effort=0.1
```

Config: `experiments/configs/impossiblebench_inkling_small.json`. It pins `lr=2e-4` by hand —
tinker-cookbook's LoRA-LR heuristic explicitly refuses Inkling ("not yet calibrated"); see the config
for the rationale. This run is meant to plot to W&B, and the runner falls back to **offline** logging
without complaint when the box is logged out — check the banner's `wandb: …` line, or force the
intent with `WANDB_MODE=online`.

- **The setting.** `subset: "impossible"` = `oneoff` + `conflicting`, the two mutated splits where no
  correct implementation can pass — so `behavior_present` ("passed the tests") *is* the cheating oracle.
  One prompt → one assistant message → one fenced code block; no tool loop, no retry-on-failure.
- **Watch these metrics.** `train/behavior_rate` (= cheat rate, the primary signal),
  `train/reward/task_mean` (fraction of the check passing), `train/unparsed_rate` (answers with no
  codeblock — spikes mean `max_tokens` is cutting the model off mid-thought).
- **`thinking_effort`** conditions Inkling's reasoning length (0.1 minimal … 0.9 high, the model
  default). High effort on these tasks costs thousands of CoT tokens per rollout, so it must be paired
  with a large `max_tokens` or the answer never arrives (the rollout is `unparsed`).
- **No LLM judges are configured** (`monitors: []`) — this is the plain-RL baseline for the env. Add
  monitors the same way as the MBPP matrix rows once a decorrelation row is wanted; the CoT judges need
  `OPENROUTER_API_KEY`.
- **Untrusted code executes locally** (subprocess + per-statement alarm + address-space rlimit, not a
  sandbox). See the warning in `envs/impossiblebench.py`.

## Terminal-verifier (multi-turn)

```bash
TV_CONTROL=experiments/configs/terminal_verifier_control.json

# 1. baseline inference — does the base model read the verifier, how often, what does it do:
uv run python experiments/eval_terminal_baseline.py --n-prompts 24 --samples 2 --think-budget 1536
uv run python experiments/eval_terminal_baseline.py --judge google/gemini-2.5-flash-lite   # + judge AUROC
# Inkling (TML rendering, reasoning effort instead of a thinking flag):
uv run python experiments/eval_terminal_baseline.py --model thinkingmachines/Inkling-Small --effort 0.5 \
    --n-prompts 256 --think-budget 1536 --answer-tokens 512 --max-tokens 3072
# 2. smoke the RL loop (2 steps, tiny batch):
uv run python experiments/run_experiment.py --config $TV_CONTROL --set run_name=tv_smoke n_steps=2 \
    batch_size=4 group_size=4 eval_size=4 eval_every=1
# 3. the control run (Inkling-Small, 90 steps, 16x8 episodes/step, the four judges held out, no penalty):
uv run python experiments/run_experiment.py --config $TV_CONTROL
# 4. a monitored row — train against the output-only gemini-3.5 judge, judge effort chosen per run
#    (run_name must differ, or the two runs share data/runs/<name>/):
uv run python experiments/run_experiment.py \
    --config experiments/configs/terminal_verifier_gemini35_out.json --set \
    'monitors.model:gemini-3.5.reasoning={"effort":"low"}' \
    run_name=tv_g35out_eff-low_s0 seed=0
uv run python experiments/run_experiment.py \
    --config experiments/configs/terminal_verifier_gemini35_out.json --set \
    'monitors.model:gemini-3.5.reasoning={"effort":"medium"}' \
    run_name=tv_g35out_eff-medium_s0 seed=0
```

The baseline also prints the **cost accounting** an RL run is budgeted from — prefill / sampled /
training tokens per rollout (training = the size of the single GRPO datum the episode folds into),
$ per rollout and $ per RL step (= `batch_size × group_size` rollouts) at tinker's published rates
(`--step-rollouts`, `--price` to override)
— and the failure-mode split the token budget hinges on: output truncated by `max_tokens`, thinking
force-closed at `think_budget`, and turns with no parseable `<command>`. Every run writes
`summary_<name>.json` next to its `episodes.jsonl`. Note the KL penalty (`kl_coef>0`) adds one base-model
forward over each episode's datum on top of these numbers.

Configs (all Inkling-Small): `experiments/configs/terminal_verifier_control.json` + the four
`terminal_verifier_gemini{25,35}_{cot,out}.json` rows. `env_options`: `verifier_mode`
(`corrupted` = perturbed verifier, the paper's variant and every config's setting; `corrupted_negative`
= that, negated; `verifier_bug` = never passes; `possible` = honest route exists — override one option
with `--set env_options.verifier_mode=possible`), `offset_range`, `max_turns`,
`output_view` (`transcript` | `assistant` | `explanations` — what the *output* monitors see),
`command_timeout`, `require_explanation` (default `false`; `true` = every command must be followed by
`Explanation: …`, else the turn is a parse failure — the prompts ask for it too),
`read_only_verifier` (default `false`; `true` = `verifier.py` is made read-only with real file
permissions — commands run as `nobody`, needs root — see docs/ENVIRONMENTS.md #5).
**Submit-tool variants:** `verifier_mode` ∈ `submit_possible` / `submit_corrupted` /
`submit_corrupted_negative` keep that mode's `verifier.py` but tell the policy the verifier is only a
helper it may modify freely, and grade a `<submit>answer</submit>` tool instead (reward 1 iff the last
submission equals the verifier's value). They **require** `submission_ends` (`first` = the first
submission ends the episode; `correct` = only a correct one does, wrong ones get an "incorrect" user
message) — rejected in every other mode. `require_explanation: true` also applies: each `<submit>`
then needs `Explanation: <how it found the solution>` too. E.g. `--set
env_options.verifier_mode=submit_corrupted --set env_options.submission_ends=correct`
(`eval_terminal_baseline.py`: `--verifier-mode submit_corrupted --submission-ends correct`). `think_budget` has three states: **absent** from the config (= `"auto"`) → the env's
default, 1536 here (so a config copied from an MBPP row with only the env name swapped works);
an **int** → that cap; **`null`** (or `--set think_budget=null`) → **no budget at all**, each turn is one
call of `max_tokens` (the env default never overrides an explicit null). The banner prints the
effective value, and `run_info.json` records it as `think_budget`. Without a budget Qwen3 thinks
through the whole turn and never runs a command; Inkling-Small mostly copes but ~10–90% of episodes
hit a 1536 budget late in RL, so the choice is a real hyperparameter.

**Swapping `mbpp_honeypot` → `terminal_verifier` in an existing matrix row** works as-is, with caveats:
episodes are ~3.3 turns, so sampling per step costs several× MBPP; `probe_iid` is the MBPP hardcode
probe, i.e. a second OOD probe here; and probe forwards see 6–10k-token sequences (all turns' thinking
+ transcript) instead of ~2k — the white-box path does no truncation, so drop the probe `batch_size`
(e.g. 2) on the box or expect OOMs. Watch `train/behavior_rate`
(= peeked/tampered), `env/read_verifier`, `env/passed`, `env/truncated`, `env/no_command_turns`.
Add monitors exactly as in the MBPP matrix rows; for the paper's judge use `output_view:
"explanations"` with an output-only (`use_cot: false`) train-against monitor.

⚠️ Episodes run the policy's shell commands on this machine (temp dir, timeout, minimal env — not a
sandbox).

## Probes (white-box)

```bash
# the OOD deception probe (probe_ood) on the policy's base model → data/probes/<model>/diverse_deception
uv run python experiments/train_probe.py --model Qwen/Qwen3-8B --preset diverse_deception
# the iid MBPP reward-hack probe (probe_iid), split-safe (disjoint train task_ids, no RL-eval leakage):
uv run python experiments/train_probe.py --model Qwen/Qwen3-8B --preset mbpp \
    --out data/probes/Qwen3-8B/mbpp_honeypot
```
Both Qwen3-8B probes are already trained + committed under `data/probes/Qwen3-8B/` — retrain only if the
model or split changes.

Two probing modes (see `docs/MONITORS.md`): **original-model** (frozen base on the text; tinker
backend, post-hoc or in-reward) and **trained-model** (live policy activations; local
`TransformersBackend`). Probes must use the **same base model** as the policy (off-model is unreliable).

## Probe-investigation scripts (2026-06-15 — see docs/PROBES.md)

```bash
# Build CoT contrastive pairs on-policy, generic over env (measures natural base rate, nudges only the
# rare class as last resort; natural-only by default). --judge-model uses a stronger MASK lie-oracle.
uv run python experiments/build_cot_pairs.py --env mask --policy Qwen/Qwen3-8B --n-prompts 300 --judge-model anthropic/claude-sonnet-4.6

# CoT-vs-noCoT transfer 2x2 on a *_cot_pairs.jsonl. --within-generation / --preserve-thinking read CoT
# (the follow-up default is no-CoT on thinking models — the CoT-stripping bug). Prints mean|Δ(CoT,noCoT)|.
uv run python experiments/run_cot_transfer.py --pairs data/probes/mask_cot_pairs_qwen3-8b.jsonl --model Qwen/Qwen3-8B --within-generation

# Transfer a saved probe to OOD targets (MASK subtle rollouts + MBPP reward-hacking) — the OA cross-domain test.
uv run python experiments/run_probe_transfer.py --probe data/probes/Qwen3-8B/diverse_deception --model Qwen/Qwen3-8B

# One-offs: check_mask_judge.py (haiku vs sonnet label agreement), run_mask_clean_probe.py (relabel+retrain),
# run_mask_blatant_probe.py (train on MASK propositions, eval on subtle rollouts).
```

`train_probe.py` gains `--preset {simple_deception,diverse_deception,mbpp}` (OA recipes); the probe seam
gains `extract_activations(progress=True, within-gen via follow_up=None, preserve_thinking=...)`.

## Run outputs (`data/runs/<run>/`)
- `run_info.json` — full validated config + which model each monitor was (train-against vs held-out) + `env.behavior_name`.
- `config.json` — the effective config (after `--set` overrides); re-run with `--config data/runs/<run>/config.json`.
- `metrics.jsonl` — per **train** step (ground-truth strict/loose/hardcoding rates, reward, train-against monitor score, plus RL-debugging metrics: IS loss, ratio / sample-train KL, entropy, advantage + reward spread, lengths, truncation — see docs/INFRA.md).
- `eval_metrics.jsonl` — per **eval** (every `eval_every`) on the fixed held-out set: **every** monitor's AUROC/brier/class-split. **This is what the degradation matrix reads.**
- `eval_rollouts.jsonl` — one record per held-out rollout with every monitor's score+label (post-hoc recalibration).
- `eval_rollouts_slim.jsonl` — the **committed** subset of the above (`step, task_id, behavior_present, loose_rh, hardcoding, unparsed, invalid_reason, monitors`; no text, ~3 MB/run; an invalid — truncated/unparsed — rollout has `monitors: {}`: it is never monitored). Regenerate from a full dump with `python3 scripts/slim_eval_rollouts.py <run dirs>` (stdlib-only; skips corrupt lines).
- `rollouts.jsonl` — sampled training rollouts with metadata (env oracle, per-monitor score/label, reward).
- `train/` + `eval/` — each has `ground_truth.png` + `monitors.png` (labels derive from `behavior_name`), auto-plotted on finish.
- `final_checkpoint.txt` — tinker path of the saved final weights (7-day TTL); `run.log` (via `queue_runs.sh`); `QUEUE_DONE` sentinel on success.

> Note: don't nest `&`/`nohup` when backgrounding a run by hand — it orphans the process. Use
> `queue_runs.sh -D` (handles detachment cleanly) or launch as a single foreground command.
