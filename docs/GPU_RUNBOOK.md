# GPU run-book — the real train-against-probe runs

> ⚠️ **SUPERSEDED (2026-06-29).** This runbook is the pre-pivot **Qwen3.5-4B / sycophancy** setup
> (`train_against_iid_probe.json`, `max_tokens=4096`, `mean_score_syco`, 4 detectors). The live
> experiment is the **MBPP-Honeypot cross-method matrix on Qwen3-8B** — its copy-paste run-day runbook
> lives in [`experiments/configs/mbpp_matrix/README.md`](../experiments/configs/mbpp_matrix/README.md)
> (7 rows via `scripts/queue_runs.sh`). Kept below only for the still-valid **provisioning / GPU
> kernel-install** steps (`uv sync --extra gpu`, flash-attn) and the MPS-OOM background.
>
> **Current (2026-10-12):** the first section below, on serving the Inkling-Small probes and scoring finished runs
> with them, is up to date. **2026-10-23:** the second, on the IID probe v2 on the full-precision reader, is the plan
> for v2 (not yet run), on Nebius or on vast.ai. Everything after them is the superseded Qwen3.5-4B runbook.

## Inkling-Small probes: serving on one B300, and post-hoc scoring

The Inkling-Small TerminalEnv probes ([`INKLING_PROBES.md`](INKLING_PROBES.md)) need a GPU only to run the frozen
reader: the 4-bit Inkling-Small checkpoint (`thinkingmachines/Inkling-Small-NVFP4`, 171 GB), which our loader fits on
**one NVIDIA B300** (~167 GiB plus a 12 GiB buffer; a B200 is too small). There are two ways to use them:
- **Serving**, needed by train-against runs: the reward needs the score during training.
- **Post-hoc scoring**, enough for held-out probes: their scores are only measured, so finished runs can be scored
  later from their saved eval episodes, all in one GPU session.

**Pin the reader** to the revision the probes record (`reader_info.revision` in each `meta.json`,
`b6a99534467840620d411e4cd4ad5819b2610d9c`). The server reads the revision of its checkpoint from the metadata that
`hf download --local-dir` writes next to the weights, and the monitor refuses a server that runs another revision.
A checkpoint fetched any other way has no such metadata: the server and the monitor then only warn, and nothing but
the golden check guards against a different checkpoint.

**Commit first.** The launcher pushes the repo to the VM and refuses uncommitted changes, unless `--allow-dirty`
(which saves the patch in `--out`). Files under `data/runs/`, such as a run folder pushed for scoring, don't count
as code ([`INFRA.md`](INFRA.md#gpu-vms-on-nebius-scriptsnebius_jobpy)).

**Interrupting the launcher.** Ctrl-C, SIGTERM and SIGHUP all make it delete the VM (unless `--keep`) before it
stops with exit 130. One that arrives during the delete is held until the delete completes. Start a long `run` under `nohup` (or in `tmux`)
so that closing the terminal doesn't stop it.

### Serving

1. **A B300 VM with the reader.** With the Nebius launcher:

   ```bash
   python3 scripts/nebius_job.py up --hours 12 --max-gpu-price 4.5 --preset 1gpu-24vcpu-346gb --out runs/{stamp}-{name} \
       --hf thinkingmachines/Inkling-Small-NVFP4@b6a99534467840620d411e4cd4ad5819b2610d9c   # prints the ssh line
   ```

   `up` picks the cheapest region and creates a spot VM. It sets up a RAM disk, uv and the repo (at `~/md` on the VM),
   downloads the weights to `/mnt/ram/Inkling-Small-NVFP4` with `hf download --revision … --local-dir`, and keeps the
   VM. The VM shuts itself down after `--hours`, but delete it as soon as you are done:
   `python3 scripts/nebius_job.py down --name <vm>` (the name is in the output and in `--out`'s `job.json`). If you
   keep a spending ledger, pass it (`--budget-file`, or set `NEBIUS_BUDGET_FILE`): `down` then also settles the kept
   VM's reservation, which otherwise stays at its worst case until `reap` finds the VM gone.

   On any other box with a B300: clone the repo, `uv sync --locked`, and download the reader the same way, to a fast
   local disk:

   ```bash
   uv run hf download thinkingmachines/Inkling-Small-NVFP4 --revision b6a99534467840620d411e4cd4ad5819b2610d9c \
       --local-dir /mnt/ram/Inkling-Small-NVFP4
   ```

2. **Optional: check the serving path first** (exits 1 on failure). It loads its own copy of the reader, and one B300
   can't hold two (~179 GiB each), so run it before starting the server, or stop the server first. It runs every
   deployed probe's golden check at both batchings, the HTTP server → `InklingProbeMonitor` path for one IID and one
   OOD monitor (each dataset's with-reasoning follow-up probe; their scores must equal the scorer's), and a
   throughput test. In this form it hasn't run on a GPU yet: the B300 check of 2026-10-10 sent only the two OOD
   monitors through HTTP, so the IID monitor has never been served over HTTP on a GPU.

   ```bash
   cd ~/md && uv run python experiments/check_probe_server.py --model /mnt/ram/Inkling-Small-NVFP4 \
       --out results/check_probe_server.json
   ```

3. **Start the server on the VM**, in `~/md`, detached (e.g. in `tmux`):

   ```bash
   uv run python experiments/inkling_probe_server.py --model /mnt/ram/Inkling-Small-NVFP4 --max-batch-tokens 1 \
       --probes data/probes/Inkling-Small/terminal_prerl/with_reasoning/followup,data/probes/Inkling-Small/diverse_deception/with_reasoning/followup
   ```

   It loads the reader, runs every probe's golden check, prints what it serves, and listens on `127.0.0.1:8178`.
   List every probe any run will ask for. `--max-batch-tokens 1`, the default, scores one episode per forward pass
   (the batching rule fired). The golden check refuses to start if a rendering differs, if a logit moves by more than
   the probe's `golden_tolerance`, or if the logits moved by more than 0.3 on average (`--golden-mean-tolerance`).
   By its own calibration that mean bound also refuses a correct setup, with probability at most about 1.2% per
   follow-up probe and about 1% that any of the four follow-up probes fails it: if only the mean check fails, read the
   reported mean and spread before overriding it. The real-B300 check of 2026-10-10 predates this bound and the
   server's current error handling, so the server in its current form has not yet run on a GPU.

4. **Tunnel from the RL machine,** and point the runs' monitors at it (`"server_url": "http://127.0.0.1:8178"`):

   ```bash
   ssh -N -L 8178:127.0.0.1:8178 <user>@<vm>
   ```

**Capacity.** About 3.9k tokens/s one episode at a time on a B300. Every live monitor sends its own requests, so each
costs its own forward pass per episode, even when two probes read the same view. With the TerminalEnv configs' batch
sizes, a training step waits up to about a minute per live probe on the with-reasoning view, and one B300 can serve
two or three train-against runs with one live probe each. These estimates take every episode as valid and as long as
the pre-RL ones; only valid episodes are scored. Score held-out probes after the runs instead (below).

**Preemption.** A spot VM can be stopped at 60 s notice; ours were, about 1.4 times per VM-hour, mostly in their first
minutes. Nothing restarts a preempted server yet. Runs don't fail: their monitors retry until the server is back, so
train-against runs pause. To recover, start a new VM and server as above and re-open the tunnel on the same local
port. An on-demand VM costs several times more but isn't preempted.

**Failures.** A request the server can never serve (malformed, an unknown probe, an episode that can't be rendered)
aborts the run. Any other failure, such as the GPU running out of memory, is retried: the server drops the failed
job and the monitor asks again. A failure that repeats on every attempt, e.g. one episode too big for the GPU,
therefore stalls the runs, each warning every 3 minutes in its log, and nothing restarts the server or skips the
episode. Watch the runs' logs. The status codes are in
[`MONITORS.md`](MONITORS.md#inkling-probe-monitors-kind-inkling_probe).

### Post-hoc scoring of finished runs

`experiments/eval_inkling_probes_on_runs.py` loads the reader once and replays every valid eval episode of each run
exactly as in training, scoring all the probes in one pass per view. It writes per run and probe
`probe_eval_<name>.jsonl`, tracked by git: per eval step, AUROC against `behavior_present` and `used_verifier`,
recall and false-alarm rate at 0.5, class means and counts, and the same on the tasks no probe saw (`*_unseen`).
`eval/degradation.py` reads these files into the AUROC-drop and end-AUROC grids (not the suspiciousness-gap grid,
which reads `eval_metrics.jsonl` only).
- **Unseen tasks** are those whose grid is in neither the `train` nor the `val` list of `--grids` (by default the
  deployed probes' `grids.json`). This goes by grid, not by task id: task ids carry the seed, but grids repeat across
  seeds (seed s's eval task i is seed 0's task s + i), so with any seed below 660, the configs' 0 included, a run's
  eval grids are partly or wholly the IID probe's own. Of control-seed-218's 64 eval grids, 48 are in the probe's
  train or val split and 16 in test. Seeds from about 660 to 7,250 keep a run's train and eval grids clear of the
  probe's data ([`INKLING_PROBES.md`](INKLING_PROBES.md#9-known-problems-and-caveats), caveat 4). Episodes whose
  question shows no grid are left out of the `*_unseen` fields. The log gives each run's count of grids the probes
  saw, of test grids and of new ones. (`--task-splits FILE`, the old option, reads the `grids.json` next to FILE.)
- **Resuming:** per-episode scores are appended to `probe_scores_inkling.jsonl`, so an interrupted session goes on
  where it stopped. A stored score is reused only while its line in `eval_rollouts.jsonl` still holds the same
  episode, by the sha256 of the episode's replay item: `run_experiment.py --resume` cuts `eval_rollouts.jsonl` back
  and appends new episodes on the same lines. Score files written before this check are scored again, once, in
  full.

```bash
# on a 1× B300 with the reader in /mnt/ram and the runs' folders (eval_rollouts.jsonl + run_info.json)
uv run python experiments/eval_inkling_probes_on_runs.py --model /mnt/ram/Inkling-Small-NVFP4 \
    --probes data/probes/Inkling-Small/terminal_prerl/with_reasoning/followup,data/probes/Inkling-Small/diverse_deception/with_reasoning/followup \
    --names probe_iid,probe_ood  data/runs/tv_*_s*      # the TerminalEnv runs: tv_<row>_s<seed>
```

As one launcher job: `eval_rollouts.jsonl` is gitignored, so push it explicitly, and write the results to a folder
that is fetched back (`--out-dir results` writes `results/<run name>/`). The pushed run folder doesn't count as
uncommitted code.

```bash
python3 scripts/nebius_job.py run --hours 3 --out runs/{stamp}-{name} \
    --hf thinkingmachines/Inkling-Small-NVFP4@b6a99534467840620d411e4cd4ad5819b2610d9c \
    --push data/runs/<run>/eval_rollouts.jsonl:data/runs/<run>/eval_rollouts.jsonl \
    --fetch results -- 'uv run python experiments/eval_inkling_probes_on_runs.py --model /mnt/ram/Inkling-Small-NVFP4 \
        --probes data/probes/Inkling-Small/terminal_prerl/with_reasoning/followup,data/probes/Inkling-Small/diverse_deception/with_reasoning/followup \
        --names probe_iid,probe_ood --out-dir results data/runs/<run>'
```

The launcher brings them back to `<--out>/results/<run>/`; copy `probe_eval_*.jsonl` from there into the run's
folder and commit them. **Cost:** a 90-step TerminalEnv run saves about 7,900 eval episodes. The two with-reasoning
probes share one pass over them here: about one GPU-hour per run if all of them are valid and as long as the pre-RL
episodes; only valid episodes are scored. Adding the without-reasoning probes adds about two-thirds of that again:
that view averages 1,077 tokens per episode, against 1,607 with reasoning.

## Inkling-Small probes v2: the full-precision reader, from extraction to serving

The IID probe v2 ([amendment 1](inkling_probes_amendment_1.md)) is extracted, fitted and served on the
full-precision reader alone, one episode per forward pass ([amendment 2](inkling_probes_amendment_2.md)). The reader
is `thinkingmachines/Inkling-Small` @ `8cc5877b44d343f88b92086aa1fb72897950f06a` (BF16, ~496 GiB). One copy needs three
or four B300s (four B200s), so an 8-GPU VM holds two. The v1 monitors and the section above are unchanged.

### 1. The extraction session (8× B200 or 8× B300 on Nebius, or 4× B200 on vast.ai; GPU)

The plan is `experiments/session_plans/probe_session_v2_bf16.json`:
- two copies of the reader, on GPUs 0–3 and 4–7, each extracting half of `terminal_prerl_v2` one episode per
  forward;
- then copy 0 runs the full-precision diagnostics: 128 `terminal_prerl_v2` inputs and 100 `diverse_deception` items,
  batched at 65,536 tokens and one at a time;
- features go to `probemaximizer/inkling-probes-bulk/features/bf16/terminal_prerl_v2`, the diagnostics to
  `probemaxxing/inkling-probes/diagnostics-v2/bf16/`.

The two datasets' `items.jsonl` are gitignored, so push them.

On an 8× B200 spot VM (us-central1: $20.80/h on 2026-10-23); for 8× B300, `--platform gpu-b300-sxm --preset
8gpu-192vcpu-2768gb`:

```bash
python3 scripts/nebius_job.py run --platform gpu-b200-sxm --preset 8gpu-160vcpu-1792gb --hours 2.5 \
    --name md-v2-bf16 --out runs/{stamp}-{name} \
    --hf thinkingmachines/Inkling-Small@8cc5877b44d343f88b92086aa1fb72897950f06a \
    --push data/probe_datasets/terminal_prerl_v2/items.jsonl:data/probe_datasets/terminal_prerl_v2/items.jsonl \
    --push data/probe_datasets/diverse_deception/items.jsonl:data/probe_datasets/diverse_deception/items.jsonl \
    --upload results=hf://buckets/probemaxxing/inkling-probes/runs/{stamp}-{name} --fetch results \
    -- 'mkdir -p results && uv run python experiments/run_probe_session.py --plan experiments/session_plans/probe_session_v2_bf16.json 2>&1 | tee results/session.log; exit ${PIPESTATUS[0]}'
```

- **Spot or on demand.** The v1 full-precision VM, on spot, was preempted right after its features were done, before
  the diagnostics. Spot is worth it when it is much cheaper (8× B200: $20.80/h spot, $68/h on demand, 2026-10-23); the
  session resumes on relaunch.
- **Time.** The v1 session's own timings (2026-10-10) give about 12 minutes from create to the first shard: VM, setup,
  the 496 GiB download in 98 s, about 2 minutes to load each copy. Batched, a copy extracted about 10k tokens/s.
  - One episode at a time is unmeasured on full precision. On 4-bit it was 2.5–3.9k tokens/s, against 8.8k batched.
  - `terminal_prerl_v2` is about 32M tokens: 11,817 episodes × (1,607 + 1,077 tokens, the v1 averages with and
    without reasoning).
  - At 3–6k tokens/s per copy, two copies take 45–90 minutes. With the overhead that is 1–1.7 VM-hours.
- **Cost:** 1–1.7 h × the 8× price (B200s are somewhat slower). On 8× B300 at $75.92/h ($9.49 per GPU-hour,
  2026-10-23) that is $76–130; at the $26–29/h of 2026-10-10, $27–49. On 8× B200 spot at $20.80/h, $25–45.
  - The budget ledger reserves `--hours` × the price before it creates the VM.
  - If the budget can't hold that reservation, a shorter `--hours` works too: the session is resumable, and a relaunch
    extracts only what's missing.

**On vast.ai, one 4× B200** (when Nebius has no capacity, as on 2026-10-23;
[`INFRA.md`](INFRA.md#vastai-instances-run---provider-vast)).
`experiments/session_plans/probe_session_v2_bf16_vast.json` is the same session with one copy of the reader on GPUs
0–3 and every store a local directory under `results/` (features, diagnostics-v2, status): a vast.ai instance gets no
HF token, so everything comes back with `--fetch results`. From the main checkout, with `--out` in `scratch/` (not
backed up; the features are ~51 GB):

```bash
python3 scripts/nebius_job.py run --provider vast --gpu B200 --gpus 4 --disk-gib 800 --hours 5 --reserve-minutes 60 \
    --max-cost 160 --name dennis-probes-v2 \
    --purpose "Inkling-Small probes v2: full-precision features of terminal_prerl_v2" \
    --out /srv/agents/projects/monitor-correlation-probes/scratch/gpu_jobs/{stamp}-{name} \
    --hf thinkingmachines/Inkling-Small@8cc5877b44d343f88b92086aa1fb72897950f06a \
    --push data/probe_datasets/terminal_prerl_v2/items.jsonl:data/probe_datasets/terminal_prerl_v2/items.jsonl \
    --push data/probe_datasets/diverse_deception/items.jsonl:data/probe_datasets/diverse_deception/items.jsonl \
    --fetch results \
    -- 'mkdir -p results && uv run python experiments/run_probe_session.py --plan experiments/session_plans/probe_session_v2_bf16_vast.json 2>&1 | tee results/session.log; exit ${PIPESTATUS[0]}'
```

- **Tried on a small instance only:** the launcher's smoke test passed on a GTX 1660 SUPER
  ([`INFRA.md`](INFRA.md#vastai-instances-run---provider-vast)); no 4× B200 has run it yet. Check with `--dry-run`
  first. The instance pulls its image before anything runs: 7.5 minutes on the smoke test's host, more than 15 on
  another. At $30/h that is $4–8, and the launcher waits up to 40 minutes.
- **Cost:** $30.22/h in Oregon on 2026-10-23, the 859 GB of disk included. One copy extracts the ~32M tokens in about
  1.5–3 h at 3–6k tokens/s; with the setup and the copy-back, 2–4 h: $60–120. The worst case, $151 for 5 h, is
  checked against `hub-gpu limits` ($500 a day) and `--max-cost`; the ledger records it as a vast.ai entry, not counted
  against the Nebius limit.
- **Deadline:** the session's `NEBIUS_JOB_DEADLINE` is the instance's end less the hour kept for the copy-back
  (`--reserve-minutes 60`). It starts no shard that wouldn't be done by then, and exits 75 if that stopped it. The
  launcher then fetches what is done and ends the instance. To resume, run the same command with
  `--push <out>/results:results` added: the session skips the shards already in its stores.
- **Disk:** the container may not mount a tmpfs (permission denied on the smoke test's host; bootstrap.log says), so
  `/mnt/ram` is a plain directory. The weights (496 GiB), the session's copy of the features in `/mnt/ram/session`,
  the stores' copy in `results/` (~51 GB each) and the venv with uv's cache (~13 GB) then share the instance's disk:
  800 GiB leaves about 150 GiB spare. The image doesn't count against it.
- **Copy-back:** `--fetch results` rsyncs `results/` (the features, `diagnostics-v2/bf16/`, `status/`, `session.log`)
  into `<out>/results/`, resumable, every 10 minutes while the session runs and once more after it. Through vast.ai's
  SSH proxy one rsync ran at 4–12 MB/s and the launcher's eight at once at 37 MB/s (2026-10-23), so the ~51 GB would
  take about 25 minutes. They come back as they are extracted, at 5–9 MB/s, so the last fetch has only the last shards
  and the diagnostics left. If the host fails, only what came after the last fetch is lost. The features keep the
  store's layout
  (`<out>/results/features/bf16/terminal_prerl_v2/<view>/shard_*`), which step 2 reads as it is. To put them in the
  bulk bucket too, upload them from hub, where the token stays:

  ```bash
  uv run hf buckets sync <out>/results/features/bf16/terminal_prerl_v2 \
      hf://buckets/probemaximizer/inkling-probes-bulk/features/bf16/terminal_prerl_v2
  ```

### 2. Fit, check, deploy (CPU)

```bash
# the v2 features (~50 GB in full; the probes need these tensors)
uv run python scripts/fetch_features.py hf://buckets/probemaximizer/inkling-probes-bulk/features/bf16 FEATURES/bf16 \
    --datasets terminal_prerl_v2 --keep followup,mean_reasoning,mean_action,router_logits,counts,answer_logp
uv run python experiments/fit_probes.py --reader bf16 --features FEATURES/bf16 --iid-dataset terminal_prerl_v2 \
    --probes iid --readouts followup,mean --out PROBES/bf16
uv run python experiments/compare_readers.py --probes PROBES --features FEATURES --iid-dataset terminal_prerl_v2 \
    --diagnostics DIAGNOSTICS-v2 --out OUT/compare_v2        # the full-precision jitter and the batching rule's numbers
```

- After the vast.ai session the features and diagnostics are already here, in the layout this step expects, so link
  them in instead of fetching them: `ln -s <out>/results/features/bf16/terminal_prerl_v2 FEATURES/bf16/` and
  `ln -s <out>/results/diagnostics-v2/bf16 DIAGNOSTICS-v2/` (each folder made first).
- `fit_probes.py` refuses v2 features that aren't full precision at that revision with `max_batch_tokens` 1, and the
  v2 probes record both amendments' sha256s.
- The OOD probe keeps its v1 full-precision fit (`data/probe_results/fitted/bf16/diverse_deception/`). The deployment
  takes its thresholds from `terminal_prerl_v2`'s honest train + val episodes.
- The v2 monitors deploy to their own folder, `data/probes/Inkling-Small/v2/`, next to the v1 ones
  (`experiments/deploy_probes.py --iid-dataset terminal_prerl_v2 --serving-reader bf16`; its docstring has the
  command).

### 3. Serving on full precision

The server takes every GPU it can see (`device_map="auto"`). Two copies on one 8× B300 VM:

```bash
python3 scripts/nebius_job.py up --platform gpu-b300-sxm --preset 8gpu-192vcpu-2768gb --on-demand --hours 12 \
    --out runs/{stamp}-{name} --hf thinkingmachines/Inkling-Small@8cc5877b44d343f88b92086aa1fb72897950f06a
# on the VM, in ~/md, each copy detached (e.g. in tmux), with the v2 monitors the runs use
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python experiments/inkling_probe_server.py --model /mnt/ram/Inkling-Small \
    --port 8178 --probes data/probes/Inkling-Small/v2/terminal_prerl_v2/with_reasoning/followup,...
CUDA_VISIBLE_DEVICES=4,5,6,7 uv run python experiments/inkling_probe_server.py --model /mnt/ram/Inkling-Small \
    --port 8179 --probes data/probes/Inkling-Small/v2/terminal_prerl_v2/with_reasoning/followup,...
# on the RL machine: both ports; point each run at one of them
ssh -N -L 8178:127.0.0.1:8178 -L 8179:127.0.0.1:8179 <user>@<vm>
```

- **Reader checks.** A server refuses a monitor deployed for another reader, and each monitor checks the server's
  reader and revision. So a v1 monitor (served on 4-bit) can't be served here by mistake.
- **Capacity:** unmeasured. Before starting the servers, run `experiments/check_probe_server.py --model
  /mnt/ram/Inkling-Small --probes-root data/probes/Inkling-Small/v2 --out results/check_probe_server_v2.json` on the
  VM: it also gives tokens/s one episode at a time.
- **How long the servers run.** As long as the train-against runs.
  - Vlad's 90-step runs took 9–12 h when trained against a judge (5–6 minutes a step) and 2.4–2.8 h as controls
    (1.3 minutes a step; W&B, 2026-09-30).
  - A probe run's step adds the server's time for its 128 episodes, about 0.6–1.2 minutes at 3–6k tokens/s.
- **Cost:** 8× B300 costs $26–76/h (2026-10-10 vs 2026-10-23 prices) and 8× B200 $20.80/h spot or $68/h on demand
  (2026-10-23), against $1–9.5/h for the one-B300 4-bit server of v1.

## Legacy: the Qwen3.5-4B runbook (superseded)

White-box probe forwards are **GPU-only at realistic sequence lengths**. On macOS/MPS a single
4096-token forward through the 4B hybrid model peaks ~27 GiB (no flash kernels → full `[B,heads,T,T]`
attention materialized) and OOMs; locally we're stuck at `batch_size=1` + ~7.5s/forward. On a CUDA box
flash-attn removes the wall and `ProbeMonitor.batch_size` auto-bumps to 8. See `STATUS.md` → Known
issues. The local **smoke** (`train_against_iid_probe_smoke.json`, `max_tokens=1024`) only validates
plumbing; the **real** run (`train_against_iid_probe.json`, `max_tokens=4096`, `eval_size=32`) runs here.

## 0. Provision

- Linux + a CUDA GPU with ≥ ~40 GB VRAM (4B in bf16 + 4096-token activations + flash-attn workspace;
  an A100-40G/80G or L40S is comfortable). Single GPU is fine — the policy trains on **tinker**
  (remote); the local GPU only runs the frozen probe model's forward passes.
- Python via `uv` (see `CLAUDE.md`). Disk for the model download (~8 GB) + run logs.

## 1. Code + deps

```bash
git clone <repo> && cd MonitorDecorrelation
uv sync --extra gpu          # installs flash-attn + flash-linear-attention + causal-conv1d
                             # (Linux/CUDA-gated; a plain `uv sync` on Mac skips them)
```

`.env` (gitignored — create it on the box):

```
TINKER_API_KEY=...
OPENROUTER_API_KEY=...        # for the CoT held-out monitors (claude-3.5-haiku / sonnet-4.6)
```

## 2. Get the probes onto the box

`data/probes/` is gitignored, so probes don't travel with the repo. Two options:

- **Rsync** the local probes + the model-independent dataset:
  ```bash
  rsync -av data/probes/Qwen3.5-4B/ <box>:.../data/probes/Qwen3.5-4B/
  rsync -av data/probes/syco_cot_pairs.jsonl <box>:.../data/probes/
  ```
- **Or rebuild** on the box (also re-derisks the probe — see the open AUROC task before trusting it):
  ```bash
  uv run python experiments/train_probe.py --model Qwen/Qwen3.5-4B --datasets doluschat        # ood
  uv run python experiments/build_syco_pairs.py --policy Qwen/Qwen3.5-4B --label-by both \
      --out data/probes/syco_cot_pairs.jsonl                                                   # iid pairs
  uv run python experiments/train_probe.py --model Qwen/Qwen3.5-4B --datasets sycophancy_cot   # iid probe
  ```

> Durability: `LinearProbe.push_to_hub(repo_id)` exists — push the probes + pairs to HF once so future
> boxes pull instead of regenerate (the dataset is model-independent; build it once).

## 3. Sanity check (cheap)

```bash
uv run python -c "import torch; print('cuda', torch.cuda.is_available())"
uv run pytest -q                                  # 17 offline tests
# tiny live check: 1 step, no probe-at-4096 surprises
uv run python experiments/run_experiment.py --config experiments/configs/train_against_iid_probe.json \
    --set run_name=gpu_smoke n_steps=1 eval_size=8
```

Confirm the eval line shows all 4 monitors with finite AUROC + the per-phase timers — and that
`score_s` is now small (flash-attn working), not the ~250s it was on MPS.

## 4. The real run

```bash
uv run python experiments/run_experiment.py --config experiments/configs/train_against_iid_probe.json
```

Outputs land in `data/runs/tap_iid_4b_40/` (`metrics.jsonl`, `eval_metrics.jsonl`, `run_info.json`, the
effective `config.json`, `train/` + `eval/` plots). Watch `eval/` — AUROC/brier/mean_score_syco over
training **is** the degradation signal. (`brier`/`mean_score_syco` stay defined if behavior pins at 0/1
where AUROC goes NaN.)

## 5. The degradation matrix (the headline)

The matrix is **cross-run**: one run trains against one detector. Run the matched set — identical
config except `train_against` — over the four detectors `{cot_weak, cot_strong, probe_iid, probe_ood}`
(`full_matrix.json` lists all four; flip which is train-against per run, the rest held-out), then:

```bash
uv run python -c "from monitordecorrelation.eval.degradation import build_degradation_matrix, plot_degradation_matrix; \
  import glob; runs=sorted(glob.glob('data/runs/<the matched runs>')); \
  r=build_degradation_matrix(runs, out_dir='data/runs/degradation_matrix'); print(r['matrix'])"
```

`D[i][j]` = AUROC drop of detector *j* when training against *i*. Off-diagonal = cross-method
decorrelation — the result the whole project is built around.

## Gotchas

- **Same base model for probe and policy.** Off-model probing (different Qwen size) is unreliable — keep
  `probe_model == policy`. See `docs/MONITORS.md`.
- **Never train on the oracle.** `behavior_present` grades detectors only; held-out detectors never
  touch the gradient.
- **iid probe is weak (~0.7 AUROC)** — fix the probe before trusting its degradation row (open task).
- Don't nest `&`/`nohup` when backgrounding; launch as one command (`caffeinate -s` on a Mac, or just
  use the GPU box's `tmux`/`nohup` singly).
