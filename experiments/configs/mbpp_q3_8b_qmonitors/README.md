# `mbpp_q3_8b_qmonitors/` — Qwen3-8B on MBPP-Honeypot, local (vLLM) judges + the Qwen3-8B probes

Redo of the Qwen3-8B MBPP matrix (2026-11-05): the earlier 8B batches predate the GRPO group-seeding fix
(2026-09-22), used API judges whose reasoning we can't read, and older probe configs. Hyperparameters are
copied from `mbpp_q35_qmonitors/` (Qwen3.5-4B, 2026-10-27); only the policy, the monitors and the names
differ. `lr` 5e-4 as there (tinker-cookbook's estimate for Qwen3-8B LoRA is 4.7e-4).

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

**Weak judge = Qwen3-1.7B** (thinking on), picked by the 2026-11-05 bakeoff on 150+150 Qwen3-8B MBPP
rollouts — AUROC cot+out / out / cot-only 0.81 / 0.91 / 0.60 vs q35's 1.00 / 1.00 / 0.96; Qwen3-30B-A3B (the
q35/Inkling batches' "weak") scored 0.99 / 1.00 / 0.96, i.e. not weak here. 0 parse errors in 1800 calls, ~500
reasoning tokens. Seed-0 `row_control` / `row_cot_strong` started before it was added: they get the `q17_*`
columns post hoc (`experiments/add_posthoc_monitors.py 'data/runs/<run>' --config
experiments/configs/mbpp_q3_8b_qmonitors/row_control.json --only q17_cot,q17_out,q17_cotonly`).

Needs on the box: the q35 judge at `localhost:8002` (`scripts/setup_vllm.sh`), the 1.7B judge at `localhost:8011`
(GPU 0 next to the probe server: `vllm serve Qwen/Qwen3-1.7B --port 8011 --reasoning-parser qwen3
--scheduling-policy priority --max-num-seqs 256 --kv-cache-dtype fp8 --max-model-len 32768
--gpu-memory-utilization 0.3`, with setup_vllm.sh's vLLM binary/CUDA_HOME) and a GPU for the probes'
Qwen3-8B forward passes (`queue_runs.sh -P` shares one copy). The probe server and the vLLM judges have
not been run together on one box before — smoke first:

```bash
scripts/queue_runs.sh -c experiments/configs/mbpp_q3_8b_qmonitors -n 1 -j 2 -d          # plan only
scripts/queue_runs.sh -c experiments/configs/mbpp_q3_8b_qmonitors/row_probe_iid.json -n 1 -j 1 -P \
    -R smoke -- n_steps=2 eval_every=1 eval_size=8 batch_size=4                          # smoke
```
