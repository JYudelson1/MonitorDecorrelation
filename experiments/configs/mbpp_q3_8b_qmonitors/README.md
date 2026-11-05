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
| `probe_ood` | `data/probes/Qwen3-8B/diverse_deception` | `row_probe_ood` |
| `probe_iid` | `data/probes/Qwen3-8B/mbpp_honeypot` | `row_probe_iid` |

`row_control` trains against nothing (`penalty_coef: null`).

**No weak judge yet**: Qwen3-30B-A3B (the q35/Inkling batches' "weak") scores like the strong one on MBPP
(AUROC ~0.97–0.99 for both on Inkling-Small). Once a weaker local Qwen is picked, add it to every finished
run post hoc (`experiments/add_posthoc_monitors.py`, see USAGE.md) and add its `row_*_weak` configs.

Needs on the box: the q35 judge at `localhost:8002` (`scripts/setup_vllm.sh`) and a GPU for the probes'
Qwen3-8B forward passes (`queue_runs.sh -P` shares one copy). The probe server and the vLLM judges have
not been run together on one box before — smoke first:

```bash
scripts/queue_runs.sh -c experiments/configs/mbpp_q3_8b_qmonitors -n 1 -j 2 -d          # plan only
scripts/queue_runs.sh -c experiments/configs/mbpp_q3_8b_qmonitors/row_probe_iid.json -n 1 -j 1 -P \
    -R smoke -- n_steps=2 eval_every=1 eval_size=8 batch_size=4                          # smoke
```
