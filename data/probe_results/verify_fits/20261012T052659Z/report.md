# Do today's fit_probes.py and probe_fit.py reproduce the fitted probes?

2026-10-12T06:34:05Z; code sha256 fit_probes.py 83cfb844029d, probe_fit.py 5ce6ede25b68; 4025 s (out-of-fold fits 3555 s, refits 461 s). Cells: recomputed out-of-fold AUROC (mean and sd) equal to cv_grid.csv's to 5 decimals. Δ: the largest absolute difference from what was saved (0 = bit for bit).

| probe | cells matched / checked (of grid) | layers recomputed | selection | Δ weights | Δ t₁/t₅ | Δ metrics | Δ test logits | verdict |
|---|---|---|---|---|---|---|---|---|
| bf16/terminal_prerl/with_reasoning/followup | 100 / 100 (252) | 30 | reproduced (25, C 0.0001) | 0 | 0 | 0 (12) | 0 | pass |
| bf16/terminal_prerl/without_reasoning/followup | 24 / 24 (252) | 4 | reproduced (22, C 0.001, flat) | 0 | 0 | 0 (12) | 0 | pass |
| bf16/diverse_deception/without_reasoning/followup | 64 / 64 (252) | 14 | reproduced (26, C 0.01) | 0 | 0 | 0 (33) | 0 | pass |
| bf16/terminal_prerl/with_reasoning/mean | 113 / 113 (252) | 22 | reproduced (30, C 0.1) | 0 | 0 | 0 (12) | 0 | pass |
| bf16/terminal_prerl/without_reasoning/mean | 24 / 24 (252) | 4 | reproduced (22, C 0.001, flat) | 0 | 0 | 0 (12) | 0 | pass |
| bf16/diverse_deception/without_reasoning/mean | 32 / 32 (252) | 6 | reproduced (22, C 0.01) | 0 | 0 | 0 (33) | 0 | pass |
| bf16/terminal_prerl/with_reasoning/router | 240 / 240 (240) | 40 | reproduced (19, C 0.01) | 0 | 0 | 0 (12) | 0 | pass |
| bf16/terminal_prerl/without_reasoning/router | 240 / 240 (240) | 40 | reproduced (18, C 0.01) | 0 | 0 | 0 (12) | 0 | pass |
| bf16/diverse_deception/without_reasoning/router | 240 / 240 (240) | 40 | reproduced (23, C 0.1) | 0 | 0 | 0 (33) | 0 | pass |
| bf16/terminal_prerl/with_reasoning/router_all | 6 / 6 (6) | 1 | reproduced (0, C 0.001, flat) | 0 | 0 | 0 (12) | 0 | pass |
| bf16/terminal_prerl/without_reasoning/router_all | 6 / 6 (6) | 1 | reproduced (0, C 0.001, flat) | 0 | 0 | 0 (12) | 0 | pass |
| bf16/diverse_deception/without_reasoning/router_all | 6 / 6 (6) | 1 | reproduced (0, C 0.01) | 0 | 0 | 0 (33) | 0 | pass |
| nvfp4/terminal_prerl/with_reasoning/mean | 120 / 120 (252) | 22 | reproduced (30, C 0.1) | 0 | 0 | 0 (12) | 0 | pass |
| nvfp4/terminal_prerl/without_reasoning/mean | 24 / 24 (252) | 4 | reproduced (22, C 0.001, flat) | 0 | 0 | 0 (12) | 0 | pass |
| nvfp4/diverse_deception/without_reasoning/mean | 32 / 32 (252) | 6 | reproduced (22, C 0.01) | 0 | 0 | 0 (33) | 0 | pass |
| nvfp4/terminal_prerl/with_reasoning/router | 240 / 240 (240) | 40 | reproduced (22, C 0.01) | 0 | 0 | 0 (12) | 0 | pass |
| nvfp4/terminal_prerl/without_reasoning/router | 240 / 240 (240) | 40 | reproduced (18, C 0.01) | 0 | 0 | 0 (12) | 0 | pass |
| nvfp4/diverse_deception/without_reasoning/router | 240 / 240 (240) | 40 | reproduced (28, C 0.1) | 0 | 0 | 0 (33) | 0 | pass |
| nvfp4/terminal_prerl/with_reasoning/router_all | 6 / 6 (6) | 1 | reproduced (0, C 0.001, flat) | 0 | 0 | 0 (12) | 0 | pass |
| nvfp4/terminal_prerl/without_reasoning/router_all | 6 / 6 (6) | 1 | reproduced (0, C 0.001, flat) | 0 | 0 | 0 (12) | 0 | pass |
| nvfp4/diverse_deception/without_reasoning/router_all | 6 / 6 (6) | 1 | reproduced (0, C 0.01) | 0 | 0 | 0 (33) | 0 | pass |
