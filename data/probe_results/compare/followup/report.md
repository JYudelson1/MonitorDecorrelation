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
