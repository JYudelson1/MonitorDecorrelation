# Do today's fit_probes.py and probe_fit.py reproduce the fitted probes?

2026-10-12T07:05:46Z; commit 269485102a5e; sha256 fit_probes.py 320f1df5a531, probe_fit.py f95e167c3f74; 1672 s (out-of-fold fits 1612 s, refits 56 s). Cells: recomputed out-of-fold AUROC (mean and sd) equal to cv_grid.csv's to 5 decimals. Δ: the largest absolute difference from what was saved (0 = bit for bit).

| probe | cells matched / checked (of grid) | layers recomputed | selection | Δ weights | Δ t₁/t₅ | Δ metrics | Δ test logits | verdict |
|---|---|---|---|---|---|---|---|---|
| nvfp4/terminal_prerl/with_reasoning/followup | 102 / 102 (252) | 27 | reproduced (25, C 0.0001) | 0 | 0 | 0 (12) | 0 | pass |
| nvfp4/terminal_prerl/without_reasoning/followup | 24 / 24 (252) | 4 | reproduced (22, C 0.001, flat) | 0 | 0 | 0 (12) | 0 | pass |
| nvfp4/diverse_deception/without_reasoning/followup | 72 / 72 (252) | 16 | reproduced (26, C 0.01) | 0 | 0 | 0 (33) | 0 | pass |
