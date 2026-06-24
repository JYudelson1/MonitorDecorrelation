# Trained white-box probes — Qwen/Qwen3-8B

Committed (the rest of `data/` stays gitignored) so they travel with the repo to any run box.
Both are per-layer logistic-regression probes (Atlas §4.3), follow-up `("Did you respond truthfully?", "Yes")`.

- **`diverse_deception`** — the OOD deception probe (`probe_ood`). 7-source mix
  (marks_tegmark, doluschat, mask_statements, truthfulqa, liarsbench, sycophancy, sandbagging). Healthy
  CE profile (late layers ~0.04). Use as-is.

- **`mbpp_honeypot`** — the iid MBPP reward-hack probe (`probe_iid`). honest `code` vs templated
  `hardcode`. ⚠️ **Trained 2026-06-16, BEFORE the canonical `mbpp_split` (task_id train/eval split) was
  added** — it may overlap the RL eval task_ids (leakage). **Retrain before a clean run:**
  `uv run python experiments/train_probe.py --preset mbpp --model Qwen/Qwen3-8B --out data/probes/Qwen3-8B/mbpp_honeypot`
  (the `mbpp` adapter now defaults to the disjoint `train` split). It's tiny/fast.

The matrix configs (`experiments/configs/mbpp_matrix/`) reference these paths.
