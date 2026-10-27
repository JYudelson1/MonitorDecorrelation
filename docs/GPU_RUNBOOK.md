# GPU run-book — the real train-against-probe runs

> ⚠️ **SUPERSEDED (2026-06-29).** This runbook is the pre-pivot **Qwen3.5-4B / sycophancy** setup
> (`train_against_iid_probe.json`, `max_tokens=4096`, `mean_score_syco`, 4 detectors). The live
> experiment is the **MBPP-Honeypot cross-method matrix on Qwen3-8B** — its copy-paste run-day runbook
> lives in [`experiments/configs/mbpp_matrix/README.md`](../experiments/configs/mbpp_matrix/README.md)
> (7 rows via `scripts/queue_runs.sh`). Kept below only for the still-valid **provisioning / GPU
> kernel-install** steps (`uv sync --extra gpu`, flash-attn) and the MPS-OOM background.
>
> **Current (2026-10-27):** the first section below serves the v1 Inkling-Small probes (the 4-bit reader on one B300)
> and scores finished runs with them (as of 2026-10-12). The second, the IID probe v2 on the full-precision reader,
> ran on 2026-10-23: the extraction on a vast.ai 4× B200, the monitors deployed in `data/probes/Inkling-Small/v2/` (the
> ones runs use). The third, "v2 probe rows: the RL batch", runs the v2 probes' RL rows: its rehearsal passed on
> 2026-10-27 and the batch started that day. Everything after them is the superseded Qwen3.5-4B runbook.

## Inkling-Small probes: serving on one B300, and post-hoc scoring

This section is v1's monitors (`data/probes/Inkling-Small/`); v2's, which the runs use, need the full-precision reader
(the next two sections).

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
   throughput test. With the v1 monitors it hasn't run on a GPU in this form: the B300 check of 2026-10-10 sent only
   the two OOD monitors through HTTP, so v1's IID monitor has never been served over HTTP on a GPU. (With the v2
   monitors on a 4× B200 it passed on 2026-10-27: "v2 probe rows: the RL batch", below.)

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
   server's current error handling, so the server in its current form has not yet run on a B300 with the v1 monitors
   (it has with the v2 ones on 4× B200, 2026-10-27).

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
job and the monitor asks again. An episode too big for the GPU even with its attention in query chunks is answered
413, which aborts its run (2026-10-23; as a 503 before, it stalled the runs). Any other failure that repeats on every
attempt still stalls the runs, each warning every 3 minutes in its log, and nothing restarts the server. Watch the
runs' logs. The status codes are in
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
- **The v2 monitors** (amendment 3: both probes on every eval episode of all 21 runs, after the batch): the same script
  on the full-precision reader, `--model /mnt/ram/Inkling-Small` on every visible GPU (4× B200, or 3–4 B300s), with
  the v2 folders; `--grids` defaults to their own deployment's, `data/probes/Inkling-Small/v2/grids.json`. As on the
  server, every probe's golden check runs first (amendment 2).

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
  - One episode at a time on full precision: 3.5–4.6k tokens/s per 4× B200 copy (the v2 extraction, 2026-10-23;
    unmeasured when this plan was written). On 4-bit it was 2.5–3.9k tokens/s, against 8.8k batched.
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
    --max-cost 210 --name dennis-probes-v2 \
    --purpose "Inkling-Small probes v2: full-precision features of terminal_prerl_v2" \
    --out /srv/agents/projects/monitor-correlation-probes/scratch/gpu_jobs/{stamp}-{name} \
    --hf thinkingmachines/Inkling-Small@8cc5877b44d343f88b92086aa1fb72897950f06a \
    --push data/probe_datasets/terminal_prerl_v2/items.jsonl:data/probe_datasets/terminal_prerl_v2/items.jsonl \
    --push data/probe_datasets/diverse_deception/items.jsonl:data/probe_datasets/diverse_deception/items.jsonl \
    --fetch results \
    -- 'mkdir -p results && uv run python experiments/run_probe_session.py --plan experiments/session_plans/probe_session_v2_bf16_vast.json 2>&1 | tee results/session.log; exit ${PIPESTATUS[0]}'
```

- **Run on 2026-10-23** on a 4× B200: the features at 770d1f8, every shard verified; one episode per forward,
  ~0.35–0.48 s per with-reasoning episode (3.5–4.6k tokens/s). Before it only the launcher's smoke test had run on
  vast.ai, on a GTX 1660 SUPER ([`INFRA.md`](INFRA.md#vastai-instances-run---provider-vast)). Check with `--dry-run`
  first. The instance pulls its image before anything runs: 7.5 minutes on the smoke test's host, more than 15 on
  another, under a minute where it is cached. At $40/h, 15 minutes is $10; the launcher waits up to 40 minutes.
- **Cost:** on 2026-10-23 the cheapest 4× B200 with 859 GB of disk was $30.22/h at 07:20 UTC and $40.22/h at 09:40.
  One copy extracts the ~32M tokens in about 1.5–3 h at 3–6k tokens/s; with the setup and the copy-back, 2–4 h:
  $60–160. The worst case, price × 5 h ($151–201), is checked against `hub-gpu limits` ($500 a day) and `--max-cost`,
  which refuses an offer above it: set it from the dry run's price. The ledger records the instance as a vast.ai entry,
  not counted against the Nebius limit.
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

The server takes every GPU it can see (`device_map="auto"`). The RL batch serves one copy on a vast.ai 4× B200
(below). Two copies on one 8× B300 VM:

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
- **Capacity:** ~6,000 tokens/s one episode at a time on a 4× B200 copy (`check_probe_server.py`'s throughput on the
  golden episodes, the rehearsal, 2026-10-27); the v2 extraction ran at 3.5–4.6k tokens/s. Before starting the
  servers, run `experiments/check_probe_server.py --model /mnt/ram/Inkling-Small --probes-root
  data/probes/Inkling-Small/v2 --out results/check_probe_server_v2.json` on the VM (the rehearsal does so on its probe
  instance).
- **How long the servers run.** As long as the train-against runs.
  - Vlad's 90-step runs took 9–12 h when trained against a judge (5–6 minutes a step) and 2.4–2.8 h as controls
    (1.3 minutes a step; W&B, 2026-09-30).
  - A probe run's step adds the server's time for its 128 episodes, about 0.6–1.2 minutes at 3–6k tokens/s.
- **Cost:** 8× B300 costs $26–76/h (2026-10-10 vs 2026-10-23 prices) and 8× B200 $20.80/h spot or $68/h on demand
  (2026-10-23), against $1–9.5/h for the one-B300 4-bit server of v1.

## v2 probe rows: the RL batch

The probe rows of the TerminalEnv degradation matrix: runs that train against `probe_iid` (v2) or `probe_ood`, with
Vlad's four Qwen judges held out and scored live, as in his rows (seeds 200–218), and the other probe held out too but
scored after the runs (`experiments/eval_inkling_probes_on_runs.py`). `scripts/rl_batch.py` runs a whole batch from a
roster as one command; its docstring has every detail.

**The roster** (`experiments/rl_batches/`) states every run: its config, seed (0 to 10,000, the range the v2 exclusion
covers), run name and early stop, plus `n_steps` and the W&B entity and project for all.
- The batch: `probe_rows_v2.json`, as [amendment 3](inkling_probes_amendment_3.md) fixes it: `probe_iid` on seeds
  219–221, `probe_ood` on 222–224 (`probe-iid-seed-219`, …); no controls: Vlad's three (216–218) are the control row.
- The configs: `experiments/configs/terminal_verifier_probe_{iid,ood}.json`, Vlad's q3_cot row with the train-against
  judge replaced by the v2 probe at `http://127.0.0.1:8178`; the four judges held out spec for spec;
  `stop_after_zero_behavior_steps` null (as his judge rows of c30c466; the roster states it per run); `save_every` 9.
- `n_steps`: 90 as his rows, or `--n-steps 360`: in his batch only the 360-step strong-cot-seed-207 shows its
  trained-against judge falling (q35_cot on hacks down to 0.76).

**The machines** (names start with `dennis-`; tags `owner=dennis`, `experiment=probes-v2-rl`, `step=…`):

| role | what | where | $/h (2026-10-23) |
|---|---|---|---|
| rl | every run, one process each, as root (the read-only verifier) | Nebius cpu-d3 32vcpu-128gb, on demand | 1.06 |
| probe | `inkling_probe_server.py`, both v2 probes, one episode per forward, 127.0.0.1:8178 | vast.ai 4× B200 (one BF16 copy) | 42.04 |
| judges | Vlad's four vLLM engines: Qwen3-30B-A3B-FP8 on GPU 0 at :8001, three Qwen3.5-35B-A3B-FP8 at :8003–8005 behind `load_balancer.py` at :8002 | vast.ai 4× H200 (`--judge-gpu`: H200, else H200 NVL) | 16.08 (H200 NVL; no 4× H200 was offered) |

- **Tunnels:** the runs reach the servers at 127.0.0.1:8178 and :8001/:8002, as the configs say, through one
  `scripts/tunnel.sh` loop per server machine on the RL machine: `ssh -N -L` with ExitOnForwardFailure and keepalives
  (10 s × 3), reconnecting forever (2 s, doubling to 30 s). A dead link ends its ssh within ~30 s, which closes the
  forwarded connections; the probe client and the judge client both retry forever, so the runs wait, then go on.
- **Keys:** the tunnel key is made for the batch on the RL machine, in its RAM disk; only its public key reaches the
  vast.ai instances, appended to root's authorized_keys as `restrict,port-forwarding,permitopen="127.0.0.1:<port>",
  command="/bin/false"`: it opens the server's ports and nothing else. hub-gpu's key stays on this machine. The runs'
  keys (`tinker_api_key`, `wandb_api_key` in `~/.secrets`) go to the RL machine only, over SSH into its RAM disk, read
  by the supervisor's root shell as it starts and deleted at once; nothing ever goes to a vast.ai instance.
- **Pinned probes:** the v2 monitor folders must be committed. Their sha256 (probe.npz + meta.json, the server's own
  identity of a probe: a server without it answers 409, fatal to a run) is taken from the committed files, checked on
  the probe instance before the server starts (also before every restart), on the RL machine before the runs start,
  and in the server's `/v1/meta` at every health check.
- **vLLM:** 0.30.0, the version Vlad's judges ran on: he used "the latest version", and his batch ran from
  2026-09-30 to 10-02, between 0.30.0 (09-22) and 0.31.0 (10-05). Installed by `scripts/setup_vllm.sh` into
  `.venv-vllm/` (`VLLM_VERSION=0.30.0`; the env's packages in `logs/vllm/freeze.txt`), with every flag of his
  commands (`--reasoning-parser qwen3`, `--scheduling-policy priority`, `--kv-cache-dtype fp8`, eagle3 and mtp
  `--speculative-config`, `max_cudagraph_capture_size`) and `thinking_token_budget` in requests. His runs extended
  past step 90 (207, 215) may have resumed on a later release. Every engine on 127.0.0.1 (`VLLM_HOST`), and a dead one is started again alone,
  its log kept (`VLLM_RESTART`). The judges' Hugging Face revisions (main on 2026-10-23, unchanged since before his
  batch) are in `rl_batch.JUDGE_REVISIONS`.

**Before a batch:**
1. The v2 monitors deployed and committed at `data/probes/Inkling-Small/v2/{terminal_prerl_v2,diverse_deception}/
   with_reasoning/followup` (`plan` refuses otherwise), this branch merged, a clean tree, kept clean while `start`
   launches the machines (the rehearsal's second attempt, 2026-10-27: a doc edit made during the launch, and the RL
   VM's push refused the dirty tree).
2. The keys: `hub-secret set tinker_api_key`, `hub-secret set wandb_api_key` (the owner, in this project's terminal).
   W&B: the key's default entity is the owner's own, so the runs get `WANDB_ENTITY=jyudel-redwood-research` and
   `WANDB_MODE=online`; the supervisor refuses to start a run without the key or with another entity, and stops a run
   whose W&B URL names another entity or project.
3. hub-gpu: 8 GPUs free at once (4× B200 + 4× H200 is exactly the project's 8: the v2 extraction instance must have
   ended), and the daily limit raised for the batch (below); the driver shortens the leases to the room left, never
   under 2 h.
4. Nebius: bin/gpu-job's settings in the environment.

```bash
cd /srv/agents/projects/monitor-correlation-probes/MonitorDecorrelation
export NEBIUS_REGIONS_FILE=../bin/nebius-regions.json NEBIUS_BIN=~/.local/bin/nebius NEBIUS_SSH_KEY=~/.ssh/nebius_vm \
    NEBIUS_BUDGET_FILE=../output/gpu_jobs/budget.json NEBIUS_JOB_NAME_PREFIX=dennis- RSYNC_BIN=~/.local/bin/rsync
python3 scripts/rl_batch.py plan --roster experiments/rl_batches/probe_rows_v2.json     # pins, estimate, prices
# Vlad's runs, slim (eval_rollouts.jsonl + run_info.json, hard links; strong-cot-seed-207 cut at step 90, line 7,936)
VLAD=../scratch/posthoc_inputs/vlad15
# the rehearsal: the same batch, 2 steps with an eval at each, and the checklist (~1.6 h of machines, ~$100), logged
# to its own project in the team's space (<project>-rehearsal); its compare on two of Vlad's runs tests the compare's
# plumbing. Not --rehearsal-entity with the owner's own space: W&B refused this key there (2026-10-27, "the provided
# API key cannot access this resource", at wandb.init; every run of that rehearsal failed at its start)
nohup python3 scripts/rl_batch.py start --rehearsal --roster experiments/rl_batches/probe_rows_v2.json \
    --out ../output/gpu_jobs/{stamp}-{name} \
    --compare-runs $VLAD/weak-out-seed-200 $VLAD/strong-cot-seed-203 > rehearsal.out 2>&1 &
# the batch (90 steps, amendment 3)
nohup python3 scripts/rl_batch.py start --roster experiments/rl_batches/probe_rows_v2.json \
    --out ../output/gpu_jobs/{stamp}-{name} > batch.out 2>&1 &
python3 scripts/rl_batch.py status --out DIR          # machines, leases, runs (steps, s/step), the probe queue
# amendment 3's judge check, while the batch runs: Vlad's 15 runs, 4 + 4 episodes per eval step, scored twice
nohup python3 scripts/rl_batch.py judge-compare --out DIR --runs $VLAD/* --workers 128 > judge_compare.out 2>&1 &
```

**What `start` does:** the three `up`s at once (the probe instance downloads the reader, ~3 min, and starts the
server after the sha check; the judges install vLLM and start the engines, ~15 min); the tunnels; the supervisor on the
RL machine, which starts each run once every server answers through them; then the watch, every minute: syncs of the
run folders into `DIR/runs/` every 10 minutes (the boot disk goes with the VM), health on each instance and through each
tunnel every 5, the leases every 10. At the end, or on a fatal start, Ctrl-C or the RL machine gone for 30 min: a last
sync, then every machine ended and verified gone. Everything is in `DIR`: `batch.json`, `driver.log`, `launch/*.log`
(each launcher's own log), `runs/`, `vm/` (the supervisor's status and log), `rehearsal.json`.

**Costs and hours** (`plan` prints them for the roster; 2026-10-23 prices; the probe server one episode per forward,
~0.36 s per episode on 4× B200, measured on the v2 extraction):

| batch | probe server busy | machines up | cost |
|---|---|---|---|
| 6 probe rows, 90 steps | 11.3 h | ~12.3 h | $730–780 |
| 6 probe rows, 360 steps | 44.8 h | ~46 h | $2,700–2,900 |
| rehearsal (2 steps, the checklist) | 0.6 h | ~1.6 h | ~$100 |

- The probe server is the batch's pace: one copy scores every valid train and eval episode of six runs, one at a time,
  so a step takes ~8 minutes, not Vlad's controls' 1.3. A second copy would halve the hours (same probe cost), but
  needs 4 more GPUs than hub's 8.
- hub-gpu allows $500 in any 24 hours: a 90-step batch needs ~$750 within ~12 h. The owner raises `usd_per_day` for
  the batch; else leases stop being extended once the room runs out, and the servers end at their deadlines (the runs
  wait, then resume when servers return: `restart-server … --new-instance`).

**Judges: 4× H200 is enough.** Six probe rows put on the judges only their evals: 31 evals × 256 episodes × 4 judges
per run, ~8,200 calls an hour on the q3 GPU and as many on the three q35 engines over the batch's ~11 hours: 36% of what
Vlad's batch had them serve (12 rows at once, 2026-09-30: 22,500 calls/h on his one q3 GPU, 22,700 on his three q35
engines; W&B timings), with the same per-call work (his eval episodes' judge calls: q3 ~1.3k completion tokens, q35
~2.4–2.9k).
8× H200 would need the owner to raise the GPU limit from 8 to 12; if more GPUs come, a second probe copy helps more.

**Failure handling:**

| what fails | what happens |
|---|---|
| a tunnel's ssh dies (link, host) | its loop reconnects (≤ 30 s between tries); the clients retry; the watch repairs a dead loop and authorizes the key again |
| the probe server dies or hangs | the runs wait (the probe client retries forever); after 10 min down the watch restarts it on its instance (`serve`, the shas checked first), at most 3 times |
| one judge engine dies | setup_vllm.sh starts it again alone (its log kept); the balancer sends its calls to the others meanwhile; judge calls retry forever |
| an instance goes away | the watch starts a new one (once per server), moves its tunnel there and ends the old one; by hand: `restart-server ROLE --new-instance` |
| a run fails | the supervisor starts it again with `--resume` from its last save (every 9 steps), after a backoff, up to 8 failed attempts |
| a run aborts on a fatal monitor answer (a judge's 400, a probe 409/422) | the same: resumed from its last save; a repeat fails it after its attempts |
| the RL machine's supervisor dies | the runs go on (sessions of their own); the watch starts a new supervisor, which takes them over |
| this driver dies (pod restart, SIGTERM) | nothing extends the leases: the vast.ai instances end within `--lease-hours` (4), the RL machine powers off; `resume --out DIR` takes the batch back, `down --out DIR` ends it now |
| a run's judge calls fail with EMFILE ("Too many open files") at an eval | the run aborts (a monitor that cannot score stops it) and resumes from its last save. Seen 2026-10-27 at the default soft limit of 1024 (3 of 6 runs); fixed live with `prlimit --pid … --nofile=65536:1048576` on the runs and the supervisor, and since 8a1aa03 every run raises its own limit at start |
| an episode too long for the GPU | retried with its attention in query chunks; if even that doesn't fit, a 413: the run aborts and is resumed from its last save (the rehearsal, 2026-10-27: 8,012 tokens scored, 12,012 and 16,012 a 413; Vlad's valid episodes stay under ~7.5k) |

- By hand: `restart-server probe|judges --out DIR [--new-instance]`, `restart-run|stop-run RUN --out DIR`,
  `sync|extend --out DIR` (a running driver takes them as requests within a minute), `down --out DIR` (stops the
  runs, syncs, ends every machine and verifies), `watchdog --out DIR` (one check: extend and sync, or end a batch that
  is over; for cron, or a hub scheduled job the owner approves).

**The rehearsal's checklist** (`DIR/rehearsal.json`, each item ok or not): check_probe_server.py on the serving GPUs
before the server starts; synthetic 8k/12k/16k-token episodes sent to the server (`experiments/
check_probe_long_episodes.py`); W&B in the rehearsal's entity (`--rehearsal-entity`, default the roster's); `nobody`
unable to read the keys' files or the supervisor's environment on the RL machine; the probe server stopped and
restarted, a judge engine killed, each
tunnel's ssh killed, with the runs waiting, not failing; a run killed after its step-0 save and resumed; every run
done with no failed attempt; the per-step wall time and the probe queue's depth with every run at once; and the
judge compare (`experiments/rescore_terminal_judges.py compare`, 4 + 4 episodes per run and eval step) on the runs
`--compare-runs` names, against the rehearsal's judges: a test of its plumbing. Amendment 3's judge check is
`judge-compare` against the batch's own judges while it runs: Vlad's 15 runs, ~2,300 episodes, two passes into
`DIR/judge_compare/pass{1,2}/` (the second is the retest).

**The rehearsal (2026-10-27)**, the v2 probe rows at 2 steps, ran on vast.ai (4× B200 probe server, 4× H200
judges) and a Nebius CPU VM:
- **Attempt 1:** W&B refused the key in the owner's own space ("the provided API key cannot access this resource", at
  wandb.init), so every run failed at its start. The rehearsal now logs to its own project in the team's space
  (`<project>-rehearsal`, the default).
- **Attempt 2:** the RL VM's launch refused a dirty tree (a doc edit made during the launch).
- **Attempt 3 passed:** every run finished its 2 steps with no failed attempt; `nobody` could not read the keys; W&B in
  the roster's entity and project; the probe server killed and started again with the runs waiting; each tunnel's ssh
  killed and reconnected; the judge compare on weak-out-seed-200 and strong-cot-seed-203 (370 episodes per judge):
  mean difference ours − Vlad's live within ±0.011 for every judge, Pearson 0.87–0.91.
- **Marked NOT ok, but passed:** check_probe_server.py (its report on the instance: `ok` true, golden max |Δlogit|
  2e-14, ~6,000 tokens/s one episode per forward, end to end Δ 0.0) and the killed judge engine (setup_vllm.sh started
  it again within ~30 s, serving). vast.ai's SSH prints a login banner on stderr at every connection; `run_text`
  joined it to the command's output, so the JSON didn't parse, and the batch's health watch would have taken healthy
  servers for dead the same way. Fixed in 5fa2f0c (`SSH_NOISE`, `json_in`; a test in `tests/test_rl_batch.py`).
- **Not run:** the kill-and-resume drill; the 2-step runs had finished before it. So the supervisor's restart with
  `--resume` is tested offline only (`tests/test_rl_batch.py`); `--resume` itself is tested offline
  (`tests/test_resume.py`) and in small live runs (2026-09-30, `STATUS.md`), and was used live in Vlad's batch.
- **Long episodes:** 8,012 tokens scored (3 s); 12,012 and 16,012 were 413 (Known risks, below).

**The batch** (6 runs, 90 steps) started 2026-10-27 at 13:02 UTC from 5fa2f0c, logging to
`jyudel-redwood-research/terminal-env-inkling-small`.

**Known risks:**
- **Long episodes:** the reader's dense attention bias is ~340 B × T² per episode (`whitebox/features.py`): ~49 GB at
  12k tokens, ~87 GB at 16k, against an estimated ~55 GB free per B200 beside the weights; the rehearsal's server had
  ~17 GiB free per GPU (2026-10-27). Vlad's 72,145 valid eval episodes reached 7,541 tokens (none above 8k), and the
  training episodes of Vlad's runs stay under ~7.5k too; a policy pushed to evade a probe may write longer. Since
  2026-10-23:
  - an episode that runs out of memory alone is retried with its attention in query chunks (~6 GB of attention
    transients up to ~22k tokens); whatever fits runs exactly as before. On the rehearsal's 4× B200 (2026-10-27):
    8,012 tokens scored in 3 s; 12,012 and 16,012 ran out of memory even in query chunks and were answered 413. On
    that server the limit lies between 8,012 and 12,012 tokens;
  - one that doesn't fit even so is answered 413 and aborts its run, which the supervisor resumes; a repeat fails it
    after its attempts. Not turned into a score (e.g. 1.0, as a judge's parse error): that would be a penalty the
    probe never gave, and leaving the rollout out would pay for length (docs/MONITORS.md);
  - the server sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` itself unless the environment configures the
    allocator; the post-hoc scorer (`eval_inkling_probes_on_runs.py`) retries the same way and counts what still
    doesn't fit (`n_unscorable`).
- **Judge calls wait up to 7,200 s each** (`monitors/vllm.py`, `VLLM_TIMEOUT`, as Vlad's recorded calls). A 20-minute
  read timeout would re-sample every call slower than that, which his calls were not, so it is not changed: the
  tunnels' keepalives end a dead connection within ~30 s, and a hung engine is restarted by the watch.
- **tinker's side** can't be pinned: Inkling-Small's serving may have changed since 2026-09-30.

**Parity with Vlad's runs** (W&B `jyudel-redwood-research/terminal-env-inkling-small`):
- **Code.** His batch ran on c30c466 (the rows of seeds 200–205, 208–211) and 4400bf0 (the controls 216–218, the
  resumed 207); 214 on cebb41a, the resumed 206 and 215 on 1f3c1c5 and e8038b1. c30c466 → 4400bf0 changed configs only.
  4400bf0 → HEAD changed nothing in `rl/` (`renderers.py` included) or in the judges (`monitors/`: only the new
  `inkling_probe_monitor.py`); `envs/terminal_verifier.py` gained a docstring paragraph and `grid_key`/`prompt_grid`
  (used by the analysis only); `experiment_config.py`, `run_experiment.py` and `backends/tinker_backend.py` gained the
  `inkling_probe` monitor, `replay_env_options` (read only by it) and `init_from_tinker_ckpt` (off unless set). None of
  it changes a run without those. The judge requests are his byte for byte: `rescore_terminal_judges.py`'s dry run
  rebuilt every one of his seven local runs' recorded requests identically (temperature 1.0, max_tokens 16384,
  thinking on, thinking_token_budget 12288, eval priority 1, a 7,200 s client timeout).
- **Packages** (his runs' W&B requirements.txt vs uv.lock): tinker 0.29.1, tinker-cookbook 0.5.7 and tml-renderers 0.1.0
  are the same. transformers 5.5.4 → 5.19.0, torch 2.12.0 → 2.14.1, tokenizers 0.22.2 → 0.23.3, safetensors 0.7.0 →
  0.8.0, huggingface-hub 1.18.0 → 2.2.0 (and torchvision 0.29.1, new) changed, for the Inkling probes. None of them
  renders the policy's prompts (tml-renderers' own o200k tokenizer does) or samples (tinker does); the judges' prompts
  are rendered by vLLM's own environment. His resumed 206 and 215 already ran on transformers 5.18.0 / torch 2.14.1.
- **Run settings:** his; early stop null where his judge rows had it (his controls and 214 had 16, stated per run in
  the roster); `save_every` 9 instead of 999 (no effect on training: `tests/test_resume.py`).
- **Judges:** the same engines, flags and models (HF main, unchanged since before his batch); vLLM 0.30.0, his; `--disable-uvicorn-access-log` and `--host 127.0.0.1` (logging and binding only); H200 NVL if no
  H200 is offered (the same GH100 chip and 141 GB, PCIe, lower power); through SSH tunnels (latency only); a third of
  his load. The batch's `judge-compare` (amendment 3) measures what is left.
- **The RL machine:** root, Ubuntu 24.04 (glibc 2.39, dash as /bin/sh), CPython 3.11.16, as his; 32 vCPUs for six runs
  where his box had 112 for twelve plus the judges.
- **W&B:** `WANDB_ENTITY=jyudel-redwood-research` (the default of Vlad's key; the owner's key defaults to the owner's own).

**The plumbing test** (`scripts/rl_batch.py tunnel-test`, 2026-10-23; no model, no vLLM, no tinker, no key on the GPU):
a Nebius cpu-d3 2vcpu-8gb ($0.07/h) and a vast.ai 1× GTX 1660 S ($0.065/h, `up --provider vast` with a dummy HTTP
server on 127.0.0.1:8178), the batch's own tunnel code between them.
- The dummy reached from the Nebius VM through the tunnel 10 s after its loop started; the tunnel key ran no command
  (`/bin/false`) and opened no other port ("administratively prohibited"); a killed ssh was replaced and the server
  reachable again 8.6 s later; with the vast.ai host blackholed on the Nebius VM (`ip route add blackhole`), ssh
  noticed after 37 s and the tunnel was back 11.6 s after the route was removed. Both machines ended and verified
  gone. $0.04 for the three attempts of that day (the first found that vast.ai's authorized_keys ends without a
  newline, fixed; the second was refused by the launcher's push check, the tree being dirty).
- `experiment=probes-v2-rl` (23 characters) was accepted as a tag.

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
