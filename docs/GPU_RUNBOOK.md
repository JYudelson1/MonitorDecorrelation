# GPU run-book — the real train-against-probe runs

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
</content>
</invoke>
