# Trained white-box probes — Qwen/Qwen3-8B

Committed (the rest of `data/` stays gitignored) so they travel with the repo to any run box.
Both are per-layer logistic-regression probes (Atlas §4.3), follow-up `("Did you respond truthfully?", "Yes")`.

- **`diverse_deception`** — the OOD deception probe (`probe_ood`). 6-source mix
  (marks_tegmark, doluschat, mask_statements, truthfulqa, liarsbench, sandbagging). **Retrained 2026-11-05**
  without the sycophancy source (dropped from the preset 2026-09-18, `fdbe543`; the June probe still had it):
  4849 pairs, held-out AUROC 0.962 (June, with sycophancy: 0.977), late-layer CE ~0.03–0.04, 36/37 layers.
  `uv run python experiments/train_probe.py --preset diverse_deception --model Qwen/Qwen3-8B --out data/probes/Qwen3-8B/diverse_deception`

- **`mbpp_honeypot`** — the iid MBPP reward-hack probe (`probe_iid`). honest `code` vs templated
  `hardcode`. **Retrained split-safe** (the `mbpp` adapter defaults to the canonical `mbpp_split` TRAIN
  task_ids, disjoint from the RL-eval ids → no leakage). Regenerate only if the split or model changes:
  `uv run python experiments/train_probe.py --preset mbpp --model Qwen/Qwen3-8B --out data/probes/Qwen3-8B/mbpp_honeypot`

The matrix configs (`experiments/configs/mbpp_matrix/`) reference these paths.
