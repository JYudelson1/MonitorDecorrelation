# `mbpp_q3_8b_qmonitors/` — Qwen3-8B on MBPP-Honeypot, local (vLLM) judges + the Qwen3-8B probes

Redo of the Qwen3-8B MBPP matrix (2026-11-05): the earlier 8B batches predate the GRPO group-seeding fix
(2026-09-22), used API judges whose reasoning we can't read, and older probe configs. Hyperparameters are
copied from `mbpp_q35_qmonitors/` (Qwen3.5-4B, 2026-10-27); only the policy, the monitors and the names
differ — except **`lr` 2.5e-4** (2026-11-06; was 5e-4 as there, ≈ tinker-cookbook's 4.7e-4 estimate for
Qwen3-8B LoRA): at 5e-4 seed-0 `cot_weak` collapsed at step 82 (loss 37k, then 100% truncated `code code code…`
loops it could not leave — constant −1 reward, zero advantage), the same failure as 6/9 Sep-3 runs. The 5e-4
seed-0 runs are archived on the box in `data/runs/_aborted_20261106_lr5e-4/`.

| monitor | what | rows that train against it |
| --- | --- | --- |
| `q35_cot` | Qwen3.5-35B-A3B-FP8 judge, CoT + output | `row_cot_strong` |
| `q35_out` | same judge, output only | `row_out_strong` |
| `q35_cotonly` | same judge, CoT only | none — held out everywhere |
| `q17_cot` | Qwen3-1.7B judge (the weak one), CoT + output | `row_cot_weak` |
| `q17_out` | same judge, output only | `row_out_weak` |
| `q17_cotonly` | same judge, CoT only | none — held out everywhere |
| `probe_ood` | `data/probes/Qwen3-8B/diverse_deception` | `row_probe_ood` |
| `probe_iid` | `data/probes/Qwen3-8B/mbpp_honeypot` | `row_probe_iid` |

`row_control` trains against nothing (`penalty_coef: null`).

**No early stop** (`stop_after_zero_behavior_steps: null`, 2026-11-05): with 16 (the q35 batch's value) seed-0
control stopped at step 16 *before hacking had emerged* (the aborted same-seed run started hacking at steps
12–14) — the rule ends runs that never start hacking, not just ones that stopped. Early-stopped runs cannot be
resumed. Rows launched under 16 that never tripped it are identical to null-config runs.

**Weak judge = Qwen3-1.7B** (thinking on), picked by the 2026-11-05 bakeoff on 150+150 Qwen3-8B MBPP
rollouts — AUROC cot+out / out / cot-only 0.81 / 0.91 / 0.60 vs q35's 1.00 / 1.00 / 0.96; Qwen3-30B-A3B (the
q35/Inkling batches' "weak") scored 0.99 / 1.00 / 0.96, i.e. not weak here. 0 parse errors in 1800 calls, ~500
reasoning tokens. Seed-0 `row_control` / `row_cot_strong` started before it was added: they get the `q17_*`
columns post hoc (`experiments/add_posthoc_monitors.py 'data/runs/<run>' --config
experiments/configs/mbpp_q3_8b_qmonitors/row_control.json --only q17_cot,q17_out,q17_cotonly`).

Needs on the box (2×H200): `scripts/setup_vllm.sh` once (installs vLLM; stop its servers), then
`scripts/serve_mbpp_judges_2gpu.sh` — two q35 engines behind the `:8002` load balancer (GPU 1 at 0.9, GPU 0 at
0.5), the 1.7B judge on `:8011` (GPU 0, 0.15), leaving ~50 GB of GPU 0 for the probe server that
`queue_runs.sh -P` starts (Qwen3-8B bf16 ~17–20 GB, uncapped). Every judge runs with `--max-num-seqs 512`. One q35 engine saturates under this batch
(KV full at ~11.6k tok/s); eval scoring is ~85% of its load. Smoke first:

```bash
scripts/queue_runs.sh -c experiments/configs/mbpp_q3_8b_qmonitors -n 1 -j 2 -d          # plan only
scripts/queue_runs.sh -c experiments/configs/mbpp_q3_8b_qmonitors/row_probe_iid.json -n 1 -j 1 -P \
    -R smoke -- n_steps=2 eval_every=1 eval_size=8 batch_size=4                          # smoke
```
