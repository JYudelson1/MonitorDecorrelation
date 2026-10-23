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
(NVIDIA B300 by default, or a CPU-only VM: [below](#cpu-vms-keys-and-root-jobs)), or on a vast.ai instance
([below](#vastai-instances-run---provider-vast)), and always cleans up. It is stdlib-only Python, so plain `python3`
runs it.

```bash
python3 scripts/nebius_job.py prices                    # every region's price and capacity now (read-only)
python3 scripts/nebius_job.py run --hours 2 --out runs/{stamp}-{name} --hf ORG/REPO[@REV] --fetch results -- '<job>'
python3 scripts/nebius_job.py run --provider vast --gpu B200 --gpus 4 --hours 5 --name dennis-job --out runs/x -- '<job>'
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
- **Optional keys for the job** (`--secret`, below), one file each in `~/.secrets` (on hub: `hub-secret set NAME`).
- **vast.ai** instead needs none of these, only hub's `hub-gpu` and rsync on the launching machine
  ([below](#vastai-instances-run---provider-vast)).
- **Optional name prefix:** with `NEBIUS_JOB_NAME_PREFIX` set (e.g. `dennis-`), a `--name` that doesn't start with it is
  refused, for Nebius VMs and vast.ai instances alike, so they can be told apart in a shared console.

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

### CPU VMs, keys and root jobs

Some jobs need no GPU. The pre-RL rollouts of the terminal env sample from tinker, so they need the tinker key, and
they need root, because `read_only_verifier` runs the policy's commands as `nobody`.
- **CPU VMs:** `--platform cpu-d3 --preset 32vcpu-128gb --on-demand --image-family ubuntu24.04-driverless` ($1.06/h on
  demand; `16vcpu-64gb` is $0.53/h). CPU VMs run on-demand only. The capacity advice lists GPU presets only, so a CPU
  preset is priced in every region with its capacity and quota unknown. `NEBIUS_MAX_GPU_PRICE`, a spot bid, is ignored
  for `--on-demand` (`--max-gpu-price` with it is refused), and the budget reserves the on-demand price × `--hours`.
  The setup runs `uv sync --locked` and prints the Python version, with no torch/CUDA check (`exec --no-gpu` does the
  same on any box).
- **Keys:** `--secret NAME[=ENV]` (repeatable) gives the job the key in `~/.secrets/NAME` on the launching machine as
  `$ENV` (default: NAME upper-cased, e.g. `tinker_api_key` → `TINKER_API_KEY`). It travels like the HF token: over SSH
  stdin into an owner-only file on the VM's RAM disk, right before the job. The job step reads it as it starts and
  deletes it when it ends; the launcher deletes it again if it stops first. A missing or empty key is refused before
  anything is created. Only key names reach the logs, `job.json` and the dry run.
- **Root jobs:** `--root-job` runs the job as root, on a box like the RL runs' (Ubuntu 24.04 with glibc 2.39, dash as
  `/bin/sh`, CPython 3.11.16, the repo private). The setup installs uv's CPython `--sandbox-python` (default 3.11.16)
  into `/opt/uv-python`, where `nobody` can run it, syncs the venv on exactly that version and runs `chmod 750 $HOME`.
  It then fails unless, as root, the env's sandbox check passes, the venv runs that version and `nobody` cannot read
  the repo. The job runs through `sudo -n -E -H`, with its environment (`--env`, the keys, `NEBIUS_JOB_DEADLINE`) and
  root's HOME. Call `.venv/bin/python …` or `uv run --no-sync …` in it: a syncing `uv run` as root would rewrite the
  venv (the launcher sets `UV_NO_SYNC=1` as a guard). `job.json` records `root_job`, `sandbox_python` and the key
  names.

The pre-RL rollouts (512 prompts × 64 samples, about 1.5 h; pick the episode workers):

```bash
python3 scripts/nebius_job.py run --hours 3 --platform cpu-d3 --preset 32vcpu-128gb --on-demand \
  --image-family ubuntu24.04-driverless --secret tinker_api_key --root-job --out runs/{stamp}-{name} \
  --upload data/runs/pre-rl-probe-v2=hf://buckets/probemaxxing/inkling-probes/inputs/pre-rl-probe-v2/{stamp} \
  --fetch data/runs/pre-rl-probe-v2 -- .venv/bin/python experiments/eval_terminal_monitors_baseline.py \
  --config experiments/configs/terminal_verifier_control.json --n-prompts 512 --samples 64 --episode-workers N \
  --run-name pre-rl-probe-v2 --set 'monitors=[]' seed=1000000
```

### vast.ai instances (`run --provider vast`)

When Nebius has no capacity (2026-10-23: no B200 or B300 anywhere), `run --provider vast` runs the job on a vast.ai
instance through hub's `hub-gpu` CLI. It is billed to the account the owner bills this project to (aether), within the
limits the owner sets for the project (`hub-gpu limits`: $500 a day, 8 GPUs at once, timed runs of at most 72 h).
Nothing to set up: hub holds the vast.ai key, and `hub-gpu` makes the project's SSH key (`~/.ssh/hub-gpu`) at its first
start. The fetch needs rsync on the launching machine (on hub: `rsync` in `.hub/apt-packages`; or `--rsync`,
`RSYNC_BIN`).

```bash
python3 scripts/nebius_job.py run --provider vast --gpu B200 --gpus 4 --disk-gib 800 --hours 5 --max-cost 160 \
    --name dennis-job --purpose "what it is for" --out runs/{stamp}-{name} --hf ORG/REPO@REV --fetch results -- '<job>'
python3 scripts/nebius_job.py run --provider vast --dry-run …   # the offers, the limits, the start call, the scripts
hub-gpu list; hub-gpu end NAME                                    # by hand, if ever needed
```

- **The offer:** the cheapest of `hub-gpu offers` for `--gpu` × `--gpus` (default: one GPU of any model) with
  `--disk-gib` of disk, whose driver runs CUDA `--min-cuda` (default 13.0, for the locked cu130 torch), under
  `--max-price` (USD/h for the whole instance); or `--offer ID`, among the 50 cheapest of those. A GPU older than Turing
  is skipped for CUDA 13: its driver may run CUDA 13.0 (a GTX 1080 was the cheapest such offer), but CUDA 13 builds
  have no kernels for it. If the start is refused (the offer was taken meanwhile), the cheapest of a fresh listing is
  started instead, once.
- **Spending:** before the start, the worst case (price × `--hours`) must fit in what `hub-gpu limits` leaves of the
  project's 24 hours, and under `--max-cost` if given. The instance is timed: vast.ai destroys it at its end, whatever
  runs on it. The disk is in the price (`--disk-gib`, passed in GB); bandwidth, which some hosts charge, is not.
- **The ledger** (`--budget-file`) records the instance as a vast.ai entry, reserved at its worst case and settled at
  its price × the time from its start to its verified end. It does not count against `limit_usd`, which is Nebius's
  cap: vast.ai is billed to another account, under hub's limits, which the launcher checks instead. `budget` lists
  such entries apart, and `reap` leaves them alone.
- **The instance** is named `--name` plus a random suffix (lowercase letters, digits and `-`, at most 24 characters,
  so `--name` has at most 17; e.g. `dennis-probes-v2`), with a purpose line (`--purpose`, by default the job command)
  and tags (`--tag`, plus `launcher=nebius-job` and `commit=…`). The owner sees all three, in hub and in the account's
  vast.ai console.
- **Then as on Nebius,** over SSH as root: the same push (refused with uncommitted changes, `CODE_VERSION.json`,
  `--push`), `uv sync --locked` and the CUDA check, the unauthenticated `--hf` downloads to `/mnt/ram/<name>`, the job
  detached and polled, and `job.json` (`provider`, and a `vast` part: limits, offer, start answer, last list row).
  - **Image:** hub-gpu may list the instance as running at once, then as loading while it pulls its image
    (vastai/pytorch): 7.5 minutes on the smoke test's host, more than 15 on another. The launcher waits for SSH while
    it loads, up to 40 minutes and never past the job's deadline; that time is billed.
  - **RAM disk:** the bootstrap mounts a tmpfs at `/mnt/ram` if the container may (80% of its memory limit). A vast.ai
    container may not (permission denied, 2026-10-23), so `/mnt/ram` is then a plain directory on the instance's disk,
    and bootstrap.log says so. Size `--disk-gib` for the weights, what the job writes and the venv (~13 GB with uv's
    cache); the image itself doesn't count. The bootstrap also installs uv, rsync and a C compiler if they are missing,
    and makes uv use its own Python only.
  - **Deadline:** the job's `NEBIUS_JOB_DEADLINE` is the instance's end less `--reserve-minutes` (default 30, at most a
    quarter of `--hours`), kept for the fetch. The launcher stops waiting for the job halfway into that reserve,
    fetches what is there and ends the instance. There is no power-off after the job.
  - **Fetch:** rsync over SSH, uncompressed and resumable (tar over SSH if the instance has no rsync). vast.ai's SSH
    proxy, the only address the test hosts had, limits each connection: one rsync ran at 4–12 MB/s, eight at once at
    37 MB/s (2026-10-23). So a directory with more than 64 MB left to copy goes in up to 8 rsyncs at once, split by
    size. And while the job runs, the `--fetch` paths are rsynced every 10 minutes, leaving out files still being
    written (`*.part`, `*.tmp`): what the job saved is here before the instance's end, and the final fetch after the
    job has little left. Then `hub-gpu end`, verified in `hub-gpu list`. Errors, Ctrl-C, SIGTERM and SIGHUP end it too,
    held as on Nebius.
  - **An instance that goes away mid-job** (ended, gone from the list, exited) is reported in the log and `job.json`
    (exit 3). Nothing relaunches it: no `--attempts`, and no `--keep` or `up` either. To resume, run again with the
    fetched results pushed back (`--push <out>/results:results`).
- **No secrets on the instance**, someone else's machine: `--secret`, `--hf-token-file` and `--upload` (an HF bucket
  needs a token) are refused, as are an `--env` that looks like a key and a `--push` from a key folder or with a `.env`
  file. Results come back only through `--fetch`; to put them in a bucket, upload them from the launching machine.
- **Nebius's flags** (`--platform`, `--max-gpu-price`, `--regions`, `--keep`, `--root-job`, …) are refused with
  `--provider vast`, and the vast.ai ones without it. `NEBIUS_MAX_GPU_PRICE` (set by hub's `bin/gpu-job`) is ignored.
- **Smoke test (2026-10-23):** `run --provider vast --hours 0.5 --max-price 0.6` on the cheapest qualifying offer, a
  GTX 1660 SUPER in Vietnam at $0.074/h. Started at 08:30, it loaded its image for 7.5 minutes; then SSH (through
  vast.ai's proxy), the bootstrap (no tmpfs; uv installed), the push, the setup in 58 s (uv's CPython 3.11, the locked
  torch 2.14.1+cu130, CUDA seen on the compute-capability-7.5 GPU), the job in 4 s (nvidia-smi, a CUDA matmul,
  `NEBIUS_JOB_DEADLINE` = the end less 7.5 minutes), the rsync fetch, and `hub-gpu end` verified 40 s later: 10
  minutes, $0.01. A first instance on another host was still loading after the SSH wait's 15 minutes then; the
  launcher ended it, verified ($0.01), and now waits through loading. Four more runs that morning checked the fetch: a
  host with the image cached had SSH within a minute; the rounds while a 13-minute job ran brought back what it had
  saved; eight rsyncs at once fetched 32 files (1.6 GB, sha256-checked) at 37 MB/s. `hub-gpu limits` counted $0.08 for
  all six instances. (Starts before 08:07 UTC were refused by a hub bug.)

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
| `probemaxxing/inkling-probes` | `datasets/` (the probe datasets' items and manifests), `inputs/` (the raw pre-RL rollouts the IID dataset was built from), `features/nvfp4/` (4-bit features, plus the whole hidden states of 10 episodes under `terminal_prerl/*/full/`), `diagnostics/` (reader diagnostics), `probes/fitted/` (every fit's raw output, including the per-episode scores the repo leaves out), `runs/` (logs of the extraction jobs and the serving check) |
| `probemaximizer/inkling-probes-bulk` | `features/bf16/` (full-precision features) |

There is one bucket per reader's features. The buckets hold only what git can't: the deployed monitors and the
collected results live in the repo (`data/probes/Inkling-Small/`, `data/probe_results/`), so no second copy of them
can go stale. While a GPU session runs it also keeps a small resume file under `sessions/<plan name>/`; nothing reads
it once the session is done.

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
