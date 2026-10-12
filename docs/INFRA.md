# Infrastructure

How the RL substrate, algorithm, and logging fit together, plus our own GPU jobs and storage (the last two
sections). Build state in [`../STATUS.md`](../STATUS.md).

## Layers

```
experiment script
  └─ RL algorithm (GRPO)          # rewards -> group-relative advantages   (rl/, backend-agnostic)
       └─ RLBackend               # sample rollouts + apply gradient step  (backends/)
            ├─ TinkerBackend      # the only implemented backend
            └─ VerlBackend (stub) # later escape hatch for off-tinker policies
       └─ Monitors                # score rollouts -> penalty + held-out metrics
       └─ Env                     # task_reward + behavior_present
       └─ Logger                  # local wandb + rollout sampling
```

The **algorithm/backend split** is deliberate: GRPO just turns rewards into advantages and is
indifferent to who runs the forward/backward. Swapping `tinker` → `verl` should not touch the
algorithm.

## Backends

- **`tinker` (implemented path).** Raw SDK, minimal loop: `ServiceClient` →
  `create_lora_training_client(base_model, rank)` → sample via a sampling client → train step. We keep
  our own loop but **delegate the loss layer to tinker-cookbook primitives** (`compute_advantages`,
  `assemble_training_data`, `incorporate_kl_penalty`, `rl.train.train_step`, `hyperparam_utils.get_lr`)
  rather than reinventing them. Connectivity verified (`experiments/check_tinker.py`).
- **`verl` (stub, deferred).** The reason it exists: **tinker hosts no small DeepSeek-R1 distills**,
  and we may specifically want one as a natively-verbalizing policy. verl would let us RL an
  off-tinker policy. Mildly painful setup — only build it if we actually want DeepSeek.

### Policy model note
The policy is the RL'd model, so it must be **tinker-hosted** and should **natively verbalize CoT**
(small 2025-era reasoner that has to think through the task — otherwise there's no load-bearing CoT
to monitor or obfuscate). On tinker that means a small **Qwen3 with thinking on**. Caveat:
`Qwen3-4B-Instruct-2507` is the *non-thinking* variant, so the small verbalizing default is
**`Qwen/Qwen3-8B`** (hybrid thinking). For a true small R1-distill policy we'd need the verl backend.

## RL algorithm

- **GRPO** by default (group-relative advantages: sample `group_size` rollouts per prompt, normalize
  rewards within the group, no value network). Matches tinker's streamlined RLVR recipe.
- **DR-GRPO / DAPO** as easy swaps via `rl_algo` (token-level loss normalization, clip-higher,
  dynamic sampling). Start with vanilla GRPO; revisit if we see length/normalization pathologies.
- Reward per rollout: `task_reward − penalty_coef·monitor_penalty(train_against)`. Per-token **KL-to-base
  penalty** is active (`kl_coef=1e-4`, via cookbook `incorporate_kl_penalty` against a base-model sampler)
  — it anchors the policy and prevents the reward-hack collapse (echoes the Obfuscation Atlas finding
  that high KL + penalty can recover honesty).

## Logging (wandb)

- **wandb auto-syncs iff logged in** — `run_experiment.py` picks `online` when a credential is
  configured locally (netrc / `WANDB_API_KEY`), else `offline` (local, never prompts; `WANDB_MODE`
  overrides). Project `monitor-decorrelation`; all runs of a sweep share a **group** per (experiment,
  model) + tags (env, model, seed, train-against). This is where the metric curves live — especially
  the ground-truth trajectory (see EXPERIMENT_PLAN.md "Ground truth is the crux").
- **Rollout persistence sampling:** `log_fraction` (default `1.0` = keep everything) controls what
  fraction of full rollouts we store verbatim, since logging every transcript may get expensive at
  scale. Aggregate metrics are always logged regardless.

### Metrics to log every eval step
- `behavior_rate` — **primary.** Per env.
- For every monitor (train-against *and* held-out): accuracy / AUROC vs. ground truth.
- Monitor penalty, task reward, total reward, KL.
- Derived: pairwise degradation `D[i][j]` once we have multiple detectors.

### RL-debugging metrics (every train step, W&B `train/…`, also in `metrics.jsonl`)
- `loss/loss`, `loss/loss_per_token` — the importance-sampling objective, `-(exp(train_lp − sample_lp)·A).sum()`
  (tinker's `loss:sum`). Cookbook `train_step` drops the `forward_backward` result, so
  `rl/grpo.optim_metrics` recomputes it exactly from the trainer's returned logprobs. Group-centred
  advantages + ratio ≈ 1 make it ≈ `−Σ A·len`: it tracks the length/advantage correlation, not convergence.
- `loss/ratio_{mean,min,max,std}`, `loss/kl_sample_train_{v1,v2}` — sampler/trainer mismatch (should be
  ≈1 / ≈0 on-policy); `loss/entropy` — sampling entropy (−mean sampled logprob).
- `loss/adv/*`, `loss/frac_zero_adv_groups`, `reward/group_std_mean`, `reward/frac_constant_groups` — how
  much gradient signal the batch carried (a constant-reward group contributes nothing).
- `reward/total_{std,min,max}`, `rollout/output_tokens_{mean,min,max}`, `rollout/truncated_rate`,
  `loss/n_action_tokens`, `loss/learning_rate`.
- `invalid_rate` (train + eval; eval also `truncated_rate`; terminal env also `no_submission_rate`) —
  truncated or unparseable rollouts and, in the terminal env, episodes that never submitted: flat −1
  reward and **never monitored**, so every `monitor/*` metric and `reward/monitor_suspiciousness_mean`
  is over the remaining valid rollouts (`monitor/<m>/n_scored` = their count). `reward/penalty_mean` is
  still averaged over all rollouts (0 for invalid ones) — the reward actually deducted per rollout.
- Not available: grad norm (tinker's `optim_step` returns no metrics for LoRA runs) and post-update KL
  (would cost an extra logprob pass per step).

## GPU VMs on Nebius (`scripts/nebius_job.py`)

The policy trains on tinker, but some jobs need our own GPUs: extracting Inkling-Small probe features, serving the
Inkling probes, and scoring finished runs with them ([`INKLING_PROBES.md`](INKLING_PROBES.md),
[`GPU_RUNBOOK.md`](GPU_RUNBOOK.md)). `scripts/nebius_job.py` turns such a job into one command on a fresh Nebius VM
(NVIDIA B300 by default), and always cleans up. It is stdlib-only Python, so plain `python3` runs it.

```bash
python3 scripts/nebius_job.py prices                    # every region's price and capacity now (read-only)
python3 scripts/nebius_job.py run --hours 2 --out runs/{stamp}-{name} --hf ORG/REPO[@REV] --fetch results -- '<job>'
python3 scripts/nebius_job.py up --hours 4 --out runs/{stamp}-{name}    # keep the VM; prints the ssh line
python3 scripts/nebius_job.py exec --host user@ip --out runs/x -- '<job>'   # any SSH box, no Nebius
python3 scripts/nebius_job.py down --name <vm>          # delete a VM and its boot disk, then verify (--budget-file:
                                                        # also settle a kept VM's reservation in the ledger)
python3 scripts/nebius_job.py list                      # instances and disks in every region
python3 scripts/nebius_job.py budget                    # the spending ledger: limit, spent, reserved
python3 scripts/nebius_job.py reap                      # clean up after a launcher that died; settle kept VMs gone
```

`run` and `up` accept `--dry-run`: the price check, then the create call, cloud-init and remote scripts, with nothing
created.

**Setup.**
- The `nebius` CLI, authenticated with rights to create VMs in your projects (`NEBIUS_BIN` if it isn't on `PATH`).
- **Where VMs may go:** a regions file (`--regions-file` or `NEBIUS_REGIONS_FILE`) with one project per region, its
  subnet and the region's public-image project. The shape is in
  [`scripts/nebius_regions.example.json`](../scripts/nebius_regions.example.json); keep the real one outside the repo.
  A single project also works (`NEBIUS_PROJECT_ID`, `NEBIUS_SUBNET_ID`).
- **SSH key:** `NEBIUS_SSH_KEY` names a private key; its `.pub` goes into the VM's cloud-init.
- **Optional spending guard:** a ledger file (`--budget-file` or `NEBIUS_BUDGET_FILE`, created as
  `{"limit_usd": N, "entries": []}`) and a spot bid per GPU-hour (`--max-gpu-price` or `NEBIUS_MAX_GPU_PRICE`).
- **Optional HF token** with write access to the buckets (`--hf-token-file`), only for `--upload`.

**What `run` does.**
1. Quote every region of the regions file with Nebius's cost calculator, read the capacity advice, and create the VM
   in the cheapest region that has the platform, capacity and quota, under `--max-price` if given. If creation fails
   there, clean up and try the next.
2. Bootstrap over SSH: a tmpfs RAM disk at `/mnt/ram`, lingering for the user, uv.
3. Download the `--hf` weights into `/mnt/ram` (public repos, no token), detached.
4. Push the repo (tracked files plus untracked files that aren't ignored, so `.env`, `.venv` and `.git` never travel)
   and any `--push` inputs; `uv sync --locked`. The repo lands in `~/md` on the VM, with a `CODE_VERSION.json` (next
   paragraph).
5. Run the job, detached and polled. It gets `NEBIUS_JOB_DEADLINE` (the VM's scheduled shutdown, epoch seconds).
6. `--upload` paths go straight from the VM to an HF bucket; `--fetch` paths and all logs come back into `--out`.
7. Delete the VM and verify that neither it nor its boot disk is left. This runs in a `finally`, so errors,
   preemption and Ctrl-C delete too; only `--keep` (and `up`) opt out. A failed instance listing is retried within
   the delete's deadline, and if listing never works, the VM's known id is still deleted.

**Signals.** Ctrl-C, SIGTERM and SIGHUP (the launching machine shutting down, its terminal closing) are handled
alike: the launcher deletes the VM (unless `--keep`), then stops. One that arrives while the VM is being deleted is held: the delete
completes, then the launcher stops with exit 130 and makes no further attempt. SIGTERM or SIGHUP during the post-job
fetches skips the rest of them, so the delete starts sooner. A SIGHUP the launcher inherited as ignored stays ignored,
so under `nohup` it outlives its terminal.

**Only committed code is pushed.** `run`, `up` and `exec` refuse a working tree with uncommitted changes (tracked
changes, or untracked files that would travel) before anything is priced or created, and check again at each push.
`--allow-dirty` pushes them anyway and saves them in that attempt's `--out`: `code.diff`, the patch from the commit
to what was pushed, and `code_untracked.txt`. Files under `data/runs/` don't count: they are what experiments write,
not code, so a run folder pushed for post-hoc scoring doesn't block a launch. The VM's copy of the repo has no
`.git`, so each push writes `CODE_VERSION.json` into it (commit, dirty flag, patch hash), and code on the VM stamps
its outputs from that file (`src/monitordecorrelation/provenance.py`).

`job.json` in `--out` records the VM, the flags, the code version pushed, timestamps, the exit code, whether the VM
was preempted, every region's quote, the price every 15 minutes while the VM existed, and an estimated cost.

**Exit status:** 0 on success; the job's own code if it failed; 2 for an infrastructure failure (a delete that failed
included, whatever signal came); 3 if the VM stopped mid-run; 75 if the job stopped early and no attempt was left;
130 on Ctrl-C, SIGTERM or SIGHUP, also when it arrived while the VM was being deleted.

### Spot VMs: how they behave

Since 2026-10-08 Nebius prices preemptible VMs on a per-region spot market:
- **Prices move** as often as every 15 minutes, and a running VM's price moves too; billing is per second. On one day
  the same 1× B300 cost about 8 times more in one region than in another, so the launcher always quotes every region.
- **A pricing model is required** for a spot VM: follow the market price, or bid through a pricing policy
  (`--max-gpu-price`). With a bid, Nebius never charges more than it, and the VM doesn't run while the market is
  above it. You pay the market price, not the bid.
- **Preemption:** Nebius sends SIGTERM 60 s before stopping a spot VM. Disks survive a stop; RAM, including
  `/mnt/ram`, does not. A stopped VM still exists: only deleting it ends all billing.

**What we saw** (one night of probe-feature extraction, 2026-10-10, summarized by `scripts/spot_events.py` from the
launcher's `job.json` records): 12 VMs from 17 attempts, about 8 GPU-hours, about $30.
- **4 preemptions,** about 1.4 per VM-hour. Three came within 2–5 minutes of the VM starting. Three 8-GPU VMs were
  created that night: two were stopped before they finished booting, and the third was stopped after 32 minutes.
  Every other VM that survived its first ~5 minutes ran to its end.
- **10 creations failed for lack of capacity,** each after a ~6-minute schedule timeout, at no cost, even when the
  capacity advice said HIGH. Capacity advice is not a reservation.
- **2 creations hit the public-IPv4 quota** (3 per region for our tenant), so at most 3 VMs fit in one region.
- **Bid-triggered preemptions** happen when the market rises past the bid; prices rose through that night. Bid about
  $1/GPU-hour above the current price.

### Lessons learned (all built into the launcher)

1. **Weights go into RAM.** The VM's network SSD sustains only ~47 MB/s once the page cache fills; HF → RAM ran at
   hundreds of MB/s to GB/s. Not into `/dev/shm`: systemd-logind (`RemoveIPC=yes`, the default) deletes a user's
   `/dev/shm` files when their last session ends, which once silently deleted a finished 171 GB download. Use a
   separate tmpfs plus `loginctl enable-linger`.
2. **Detach long remote commands properly:** `setsid nohup bash -lc '…' > log 2>&1 < /dev/null &`, with any `cd`
   inside the quotes. `cd x && nohup … &` keeps the SSH channel open until the step ends.
3. **Never `pkill -f PATTERN`** when the pattern also appears in your own command line: it kills the invoking shell.
4. **`--recovery-policy FAIL`**, so the in-VM `shutdown -h +N` spending cap really stops the VM.
5. **The managed boot disk needs a name,** and `instance delete` removes it too. Verify with `instance list` and
   `disk list`.
6. **A fresh `known_hosts` per run:** public IPs are reused across VMs.
7. **No HF token for downloads** of public models: they run at full speed without one.
8. **A preempted VM's serial console** (`nebius compute instance logs`) shows when it really stopped, even after the
   VM is deleted. 8-GPU VMs print every PCIe device to a slow console and take ~4 minutes from RUNNING to SSH; 1-GPU
   VMs about 1 minute.
9. **Is the bid still above the price?** A pricing policy whose `status.scheduling_state` is
   `SCHEDULING_STATE_BLOCKED` bids below the current spot price: its VMs won't be scheduled, and running ones get
   preempted.
10. **The ledger reserves the worst case** (bid × GPUs × `--hours`) until a VM is settled. After preemptions it can
    refuse a relaunch whose likely cost would fit. That is the point of a hard cap. A kept VM (`up`, `run --keep`)
    stays reserved until `down` deletes it with the ledger (`--budget-file` or `NEBIUS_BUDGET_FILE`), or `reap` finds
    it gone; either settles it from the prices its launcher sampled. A `run --keep` that never created a VM isn't
    left reserved.

### Preemption, deadlines and relaunching

- **Relaunch:** `run --attempts N` (default 3). After a spot preemption, or a job exit of 75 ("stopped early, saved
  its progress, run me again"), a fresh VM runs the job again in whichever region is cheapest then, within the
  budget. Attempt k logs to `--out/retry-k/`. While no region qualifies, it re-checks every 15 minutes for
  `--wait-hours` (default 2).
- **The job must resume from what it saved.** The probe extraction does (next section).
- **Soft deadline:** a job that reads `NEBIUS_JOB_DEADLINE` can stop cleanly before `--hours` cuts it off. The
  extraction starts a shard only if the shard and its upload fit before the deadline, and on SIGTERM (the 60 s notice)
  it starts nothing new and lets the uploads finish. Either way it exits 75, also when uploads were still running or
  had failed: what didn't reach the store is extracted again on resume.
- **A launcher that died** (e.g. its machine restarted): the VM powers itself off 15 minutes after its job ends,
  instead of idling until `--hours`. `reap` (which `run` and `up` also do first when a ledger is set) deletes such a
  VM once it has stopped, or once it is overdue: 30 minutes past its `--hours` cap, or an hour old with no job
  started. Then it settles the reservation from the prices that launcher sampled.
  - reap reads the VM's `job.json` from the `--out` the ledger recorded, an absolute path (older entries hold a
    relative one, tried from the current directory, the repo and the ledger's folder), and trusts it only if it names
    that VM. If it can't find the record, it can't tell whether the job still runs, so only the `--hours` cap (plus
    30 minutes) makes the VM overdue.
  - reap never deletes a kept VM (`up`, `run --keep`). It settles a kept VM's reservation once the VM is gone.
- **`scripts/spot_events.py runs/`** builds a table of every VM attempt from the `job.json` records: creation failures
  by kind and region, minutes to run, how each VM ended, cost, and preemptions per VM-hour.

## Storage: Hugging Face buckets, saved as you go

The Inkling probe data lives in two **private** Hugging Face storage buckets. Access must be granted by the owners
(membership in both organizations); the probes in the repo work without them.

| bucket | holds |
|---|---|
| `probemaxxing/inkling-probes` | `datasets/` (the probe datasets' items), `inputs/` (the raw pre-RL rollouts), `features/nvfp4/` (4-bit features, plus whole hidden states of a few episodes under `*/full/`), `diagnostics/` (reader diagnostics), `probes/` (fitted probes, comparisons, deployed copies), `runs/` and `sessions/` (GPU job reports and status) |
| `probemaximizer/inkling-probes-bulk` | `features/bf16/` (full-precision features) |

There is one bucket per reader's features.

**Why HF buckets.** They are free within the organizations' storage quota and reachable from every region, so a
relaunched job can move to the cheapest region and still find its work; the data ends up there anyway. From a 1× B300
VM uploads ran at ~170 MB/s with one uploader and ~530 MB/s with four, several times what extraction produces. Nebius
disks and shared filesystems are tied to one region; Nebius Object Storage would need access keys on the VM.

**The token.** With `--upload`, the launcher sends an HF token with write access over SSH into an owner-only file on
the VM before the job starts, so the job can save its own progress. It is deleted when the upload step ends, and with
the VM, and never appears on a command line, in a log or in `job.json`. No other secret goes to the VM.

**Saving as you go** (`whitebox/store.py`). A store is a bucket prefix (`hf://buckets/NS/BUCKET/PREFIX`) or a local
directory.
- Work is saved in **units**: a unit's data files first, then its marker file in a separate call, because bucket
  batches are not transactional. A marker in the store means its unit is complete.
- Uploads run on a background thread, so the GPU never waits for the network. `flush` waits for them, e.g. within
  the 60 s preemption notice.
- Permanent errors (quota, permission) are not retried: they mark the store broken, the extraction stops making
  work it can't save, and it exits 1, since a relaunch could save nothing either.

**Resuming.** `experiments/extract_probe_features.py --store` (and the session driver
`experiments/run_probe_session.py`) save each 256-item shard as it finishes, and on any VM skip every shard whose
marker and data are already in the store. A preemption therefore loses only the shards in progress.
- `index.json` pins the settings that decide what a shard holds (dataset, reader, views, shard size, feature format,
  library versions, batching budget). A run with other settings refuses to start rather than mix shards.
- The dataset is pinned too: extraction refuses an `items.jsonl` whose sha256 differs from its manifest's
  `items_sha256`, before any model loads.
- `--part i/n` splits a dataset's shards between processes, or VMs, that share one store.
- Exit codes of `extract_probe_features.py` and of the session driver `run_probe_session.py`:
  - **75**, resumable: stopped by the deadline or SIGTERM, also when uploads were still running or had failed then.
    A shard whose marker and data are not both in the store isn't done, so the next run extracts it again.
  - **1**, failed: the store refuses uploads (quota, permission), or uploads failed with nothing stopping the run.
    In the session, a required reader that fails ends it with 1 (a relaunch wouldn't help).
  - **76** (session only): `--hours` left no time for even one shard, so a relaunch would only repeat that.
- **It paid off:** the full-precision session's VM was preempted 32 minutes in, when every essential shard was
  already in the bucket. Nothing had to be redone.

**Getting features back.** `scripts/fetch_features.py STORE DEST` downloads complete shards only, writing each
marker last so an interrupted fetch resumes cleanly. `--keep followup,counts,answer_logp` keeps only some tensors per
shard, for machines with little disk. `scripts/bucket_speed_test.py` measures upload speed from a VM.
