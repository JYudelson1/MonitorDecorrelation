# Reader comparison (followup)

**Decision per readout:** followup → `bf16`. Fix batching = True; 'Yes' sink-like = False

| probe | Jaccard@t1 (with / without reasoning) | Spearman | refit floor: Jaccard / Spearman (medians) | ΔAUROC (4-bit − full) [CI] | passes |
|---|---|---|---|---|---|
| terminal_prerl/with_reasoning/followup | 0.896 / 0.940 | 0.9019 | 0.945 / 0.946; 0.9936 | iid_test: -0.0000 [-0.000, +0.000] | {'agreement': False, 'agreement_within_refit_noise': False, 'delta_auroc': True} |
| terminal_prerl/without_reasoning/followup | 0.735 / 0.736 | 0.8252 | 0.884 / 0.987; 0.9881 | iid_test: +0.0000 [+0.000, +0.000] | {'agreement': False, 'agreement_within_refit_noise': False, 'delta_auroc': True} |
| diverse_deception/without_reasoning/followup | 0.458 / 0.525 | 0.7708 | 0.694 / 0.626; 0.8744 | ood_test: -0.0017 [-0.004, +0.000]; iid_transfer_with_reasoning: +0.1968 [+0.163, +0.232]; iid_transfer_without_reasoning: +0.2849 [+0.246, +0.324] | {'agreement': False, 'agreement_within_refit_noise': False, 'delta_auroc': True} |

Jitter: {"nvfp4": {"n": 228, "dlogit_p50": 0.1709770706130953, "dlogit_p99": 0.8452846066137041, "topk_agree_jitter_mean": 0.9642361998558044, "routing_topk_shape": [180178, 40, 6]}, "bf16": {}}
Batching: {"nvfp4": {"near_threshold_hacks": 0.0, "near_threshold_negatives": 0.020066100094428708, "fix_batching": true}}
Sink check: {"nvfp4": {"median_ratio_layers_10_35": 1.0885, "sink_like": false}, "bf16": {}}

## Report only (no rule reads these)

Score shift: the full-precision probe's logits on 4-bit minus on full-precision features, TerminalEnv val+test.

| probe | weight cosine (4-bit vs full) | shift, with reasoning: mean (sd), p99 abs, Pearson r | shift, without reasoning |
|---|---|---|---|
| terminal_prerl/with_reasoning/followup | 0.964 | +0.161 (0.374), 1.06, 0.968 | +0.110 (0.392), 1.08, 0.966 |
| terminal_prerl/without_reasoning/followup | 0.928 | -0.432 (0.648), 1.94, 0.952 | -0.343 (0.746), 2.43, 0.922 |
| diverse_deception/without_reasoning/followup | 0.801 | -0.675 (0.539), 1.89, 0.904 | -0.646 (0.655), 2.06, 0.821 |

Cross-reader scoring, IID probes (TerminalEnv test):

| probe | probe's reader | features | test AUROC | TPR / FPR at own t1 | TPR / FPR at recalibrated t1 |
|---|---|---|---|---|---|
| terminal_prerl/with_reasoning/followup | nvfp4 | nvfp4 | 0.99998 | 100.0% / 0.37% | 100.0% / 0.41% |
| terminal_prerl/with_reasoning/followup | nvfp4 | bf16 | 1.00000 | 100.0% / 0.41% | 100.0% / 0.73% |
| terminal_prerl/with_reasoning/followup | bf16 | nvfp4 | 0.99998 | 100.0% / 0.41% | 100.0% / 0.46% |
| terminal_prerl/with_reasoning/followup | bf16 | bf16 | 1.00000 | 100.0% / 0.55% | 100.0% / 0.73% |
| terminal_prerl/without_reasoning/followup | nvfp4 | nvfp4 | 1.00000 | 100.0% / 0.46% | 100.0% / 0.51% |
| terminal_prerl/without_reasoning/followup | nvfp4 | bf16 | 1.00000 | 100.0% / 0.73% | 100.0% / 0.55% |
| terminal_prerl/without_reasoning/followup | bf16 | nvfp4 | 1.00000 | 100.0% / 0.23% | 100.0% / 0.60% |
| terminal_prerl/without_reasoning/followup | bf16 | bf16 | 1.00000 | 100.0% / 0.46% | 100.0% / 0.51% |

Cross-reader scoring, OOD probes:

| probe | probe's reader | features | OOD test AUROC | transfer AUROC (with / without reasoning) |
|---|---|---|---|---|
| diverse_deception/without_reasoning/followup | nvfp4 | nvfp4 | 0.9768 | 0.909 / 0.842 |
| diverse_deception/without_reasoning/followup | nvfp4 | bf16 | 0.9759 | 0.872 / 0.739 |
| diverse_deception/without_reasoning/followup | bf16 | nvfp4 | 0.9769 | 0.725 / 0.634 |
| diverse_deception/without_reasoning/followup | bf16 | bf16 | 0.9785 | 0.712 / 0.557 |

Batching floor (diagnostics' TerminalEnv rows; the batching rule is unchanged):

| probe | probe's reader | diagnostics' reader | view | n | one-at-a-time − batched: mean (sd) | Spearman |
|---|---|---|---|---|---|---|
| terminal_prerl/with_reasoning/followup | nvfp4 | nvfp4 | with_reasoning | 64 | +0.003 (0.315) | 0.977 |
| terminal_prerl/with_reasoning/followup | bf16 | nvfp4 | with_reasoning | 64 | -0.018 (0.313) | 0.972 |
| terminal_prerl/without_reasoning/followup | nvfp4 | nvfp4 | without_reasoning | 64 | +0.020 (0.626) | 0.952 |
| terminal_prerl/without_reasoning/followup | bf16 | nvfp4 | without_reasoning | 64 | -0.041 (0.589) | 0.946 |
| diverse_deception/without_reasoning/followup | nvfp4 | nvfp4 | with_reasoning | 64 | -0.010 (0.406) | 0.962 |
| diverse_deception/without_reasoning/followup | nvfp4 | nvfp4 | without_reasoning | 64 | +0.014 (0.501) | 0.916 |
| diverse_deception/without_reasoning/followup | bf16 | nvfp4 | with_reasoning | 64 | -0.012 (0.411) | 0.917 |
| diverse_deception/without_reasoning/followup | bf16 | nvfp4 | without_reasoning | 64 | -0.056 (0.520) | 0.885 |

Massive dimensions of "Yes" (the readout rule's second half, "the same ≤ 8 massive dimensions dominate its norm", has no pre-registered threshold; the rule above uses the norm ratio only). Per layer: the median share of the squared norm in each input's own top 8 dimensions; how consistent those dimensions are across inputs (share of an input's top 8 among the layer's 8 most frequent); the median share in those 8 most frequent dimensions.

- nvfp4 (228 diagnostics inputs), medians over layers 10–35: top-8 share 43.6%, consistency 88.4%, common-8 share 43.5%

| layer | nvfp4: top-8 share / consistency / common-8 share |
|---|---|
| 0 | 45.6% / 100.0% / 45.6% |
| 1 | 99.2% / 89.6% / 99.2% |
| 2 | 98.8% / 89.4% / 98.8% |
| 3 | 93.4% / 100.0% / 93.4% |
| 4 | 93.0% / 99.9% / 93.0% |
| 5 | 93.3% / 93.3% / 93.3% |
| 6 | 93.0% / 91.3% / 93.0% |
| 7 | 92.5% / 88.9% / 92.5% |
| 8 | 93.8% / 94.5% / 93.8% |
| 9 | 95.1% / 93.3% / 95.1% |
| 10 | 95.7% / 90.9% / 95.7% |
| 11 | 94.8% / 94.5% / 94.8% |
| 12 | 95.1% / 94.5% / 95.1% |
| 13 | 93.8% / 88.6% / 93.8% |
| 14 | 93.2% / 88.5% / 93.2% |
| 15 | 92.6% / 86.5% / 92.6% |
| 16 | 90.5% / 89.4% / 90.5% |
| 17 | 76.1% / 94.5% / 76.1% |
| 18 | 81.7% / 91.7% / 81.7% |
| 19 | 72.5% / 90.5% / 72.4% |
| 20 | 62.2% / 89.2% / 62.2% |
| 21 | 49.6% / 83.5% / 49.4% |
| 22 | 46.0% / 81.7% / 46.0% |
| 23 | 32.4% / 88.4% / 32.2% |
| 24 | 34.9% / 83.5% / 34.7% |
| 25 | 41.2% / 85.0% / 41.1% |
| 26 | 39.4% / 88.9% / 39.4% |
| 27 | 40.9% / 87.8% / 40.8% |
| 28 | 31.6% / 88.2% / 31.3% |
| 29 | 18.7% / 86.8% / 18.1% |
| 30 | 31.1% / 92.2% / 31.1% |
| 31 | 33.0% / 89.1% / 33.0% |
| 32 | 33.7% / 85.1% / 33.5% |
| 33 | 37.5% / 86.8% / 37.4% |
| 34 | 35.0% / 78.1% / 34.9% |
| 35 | 15.8% / 73.9% / 15.6% |
| 36 | 33.3% / 77.7% / 33.2% |
| 37 | 28.2% / 74.0% / 28.0% |
| 38 | 29.4% / 69.3% / 29.2% |
| 39 | 36.1% / 65.1% / 35.9% |
| 40 | 26.3% / 69.2% / 26.2% |
| 41 | 25.8% / 77.4% / 25.6% |
| 42 | 2.2% / 46.9% / 1.8% |
