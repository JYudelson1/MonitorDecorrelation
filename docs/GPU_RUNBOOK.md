# GPU run-book — the real train-against-probe runs

> ⚠️ **SUPERSEDED (2026-06-29).** This runbook is the pre-pivot **Qwen3.5-4B / sycophancy** setup
> (`train_against_iid_probe.json`, `max_tokens=4096`, `mean_score_syco`, 4 detectors). The live
> experiment is the **MBPP-Honeypot cross-method matrix on Qwen3-8B** — its copy-paste run-day runbook
> lives in [`experiments/configs/mbpp_matrix/README.md`](../experiments/configs/mbpp_matrix/README.md)
> (7 rows via `scripts/queue_runs.sh`). Kept below only for the still-valid **provisioning / GPU
> kernel-install** steps (`uv sync --extra gpu`, flash-attn) and the MPS-OOM background.
>
> **Current (2026-10-12):** the first section below, on serving the Inkling-Small probes and scoring finished runs
> with them, is up to date. Everything after it is the superseded Qwen3.5-4B runbook.

## Inkling-Small probes: serving on one B300, and post-hoc scoring

The Inkling-Small TerminalEnv probes ([`INKLING_PROBES.md`](INKLING_PROBES.md)) need a GPU only to run the frozen
reader: the 4-bit Inkling-Small checkpoint (`thinkingmachines/Inkling-Small-NVFP4`, 171 GB), which our loader fits on
**one NVIDIA B300** (~167 GiB plus a 12 GiB buffer; a B200 is too small). There are two ways to use them:
- **Serving**, needed by train-against runs: the reward needs the score during training.
- **Post-hoc scoring**, enough for held-out probes: their scores are only measured, so finished runs can be scored
  later from their saved eval episodes, all in one GPU session.

Pin the reader to the revision the probes record (`reader_info.revision` in each `meta.json`,
`b6a99534467840620d411e4cd4ad5819b2610d9c`): the monitor refuses a server that runs another one.

### Serving

1. **A B300 VM with the reader.** With the Nebius launcher ([`INFRA.md`](INFRA.md#gpu-vms-on-nebius-scriptsnebius_jobpy)):

   ```bash
   python3 scripts/nebius_job.py up --hours 12 --max-gpu-price 4.5 --preset 1gpu-24vcpu-346gb \
       --hf thinkingmachines/Inkling-Small-NVFP4@b6a99534467840620d411e4cd4ad5819b2610d9c   # prints the ssh line
   ```

   `up` picks the cheapest region, creates a spot VM, sets up a RAM disk, uv and the repo, downloads the weights to
   `/mnt/ram/Inkling-Small-NVFP4` and keeps the VM; it shuts itself down after `--hours`. Elsewhere: clone the repo,
   `uv sync --locked`, and download the checkpoint at that revision to a fast local disk.

2. **Start the server on the VM**, in the repo, detached (e.g. in `tmux`):

   ```bash
   uv run python experiments/inkling_probe_server.py --model /mnt/ram/Inkling-Small-NVFP4 --max-batch-tokens 1 \
       --probes data/probes/Inkling-Small/terminal_prerl/with_reasoning/followup,data/probes/Inkling-Small/diverse_deception/with_reasoning/followup
   ```

   It loads the reader, runs every probe's golden check (and refuses to start if one fails), prints what it serves,
   and listens on `127.0.0.1:8178`. List every probe any run will ask for: probes on the same view share one forward
   pass. `--max-batch-tokens 1`, the default, scores one episode per forward pass (the batching rule fired).

3. **Tunnel from the RL machine,** and point the runs' monitors at it (`"server_url": "http://127.0.0.1:8178"`):

   ```bash
   ssh -N -L 8178:127.0.0.1:8178 <user>@<vm>
   ```

4. **Optional end-to-end check** before the runs (exits 1 on failure): golden checks at both batchings, the HTTP
   server → `InklingProbeMonitor` path against the scorer, and throughput.

   ```bash
   uv run python experiments/check_probe_server.py --model /mnt/ram/Inkling-Small-NVFP4 --out results/check_probe_server.json
   ```

**Capacity.** About 3.9k tokens/s one episode at a time on a B300. With the TerminalEnv configs' batch sizes, a
training step waits about a minute for its scores, and one B300 can serve two or three concurrent train-against runs.

**Preemption.** A spot VM can be stopped at 60 s notice; ours were, about 1.4 times per VM-hour, mostly in their first
minutes. Nothing restarts a preempted server yet. Runs don't fail: their monitors retry until the server is back, so
train-against runs pause. To recover, start a new VM and server as above and re-open the tunnel on the same local
port. An on-demand VM costs several times more but isn't preempted.

### Post-hoc scoring of finished runs

`experiments/eval_inkling_probes_on_runs.py` loads the reader once, replays every valid eval episode of each run
exactly as in training, and writes per run and probe `probe_eval_<name>.jsonl` (per eval step: AUROC against
`behavior_present` and `used_verifier`, recall and false-alarm rate at 0.5, class means and counts), the file
`eval/degradation.py` reads for post-hoc probes. Per-episode scores are appended to `probe_scores_inkling.jsonl`, so
an interrupted session resumes where it stopped.

```bash
# on a 1× B300 with the reader in /mnt/ram and the runs' folders (eval_rollouts.jsonl + run_info.json)
uv run python experiments/eval_inkling_probes_on_runs.py --model /mnt/ram/Inkling-Small-NVFP4 \
    --probes data/probes/Inkling-Small/terminal_prerl/with_reasoning/followup,data/probes/Inkling-Small/diverse_deception/with_reasoning/followup \
    --names probe_iid,probe_ood  data/runs/terminal_*
```

As one launcher job: `eval_rollouts.jsonl` is gitignored, so push it explicitly, and write the results to a folder
that is fetched back (`--out-dir results` writes `results/<run name>/`):

```bash
python3 scripts/nebius_job.py run --hours 3 --out runs/{stamp}-{name} \
    --hf thinkingmachines/Inkling-Small-NVFP4@b6a99534467840620d411e4cd4ad5819b2610d9c \
    --push data/runs/<run>/eval_rollouts.jsonl:data/runs/<run>/eval_rollouts.jsonl \
    --fetch results -- 'uv run python experiments/eval_inkling_probes_on_runs.py --model /mnt/ram/Inkling-Small-NVFP4 \
        --probes data/probes/Inkling-Small/terminal_prerl/with_reasoning/followup,data/probes/Inkling-Small/diverse_deception/with_reasoning/followup \
        --names probe_iid,probe_ood --out-dir results data/runs/<run>'
```

The launcher brings them back to `<--out>/results/<run>/`; copy `probe_eval_*.jsonl` from there into the run's
folder. **Cost:** a 90-step TerminalEnv run saves about
7,900 eval episodes. The two with-reasoning probes share one pass over them: about one GPU-hour per run. Adding the
without-reasoning probes adds about half of that again.

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
