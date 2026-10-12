# Reader comparison (mean, router, router_all)

**Decision per readout:** mean → `nvfp4`; router → `bf16`; router_all → `bf16`. Fix batching = False; 'Yes' sink-like = False

| probe | Jaccard@t1 (with / without reasoning) | Spearman | refit floor: Jaccard / Spearman (medians) | ΔAUROC (4-bit − full) [CI] | passes |
|---|---|---|---|---|---|
| terminal_prerl/with_reasoning/mean | 0.973 / 0.849 | 0.9857 | 0.676 / 0.739; 0.7592 | iid_test: +0.0000 [+0.000, +0.000] | {'agreement': True, 'agreement_within_refit_noise': True, 'delta_auroc': True} |
| terminal_prerl/without_reasoning/mean | 0.896 / 0.928 | 0.9970 | 0.895 / 0.877; 0.9576 | iid_test: +0.0000 [+0.000, +0.000] | {'agreement': True, 'agreement_within_refit_noise': True, 'delta_auroc': True} |
| diverse_deception/without_reasoning/mean | 0.667 / 0.725 | 0.9949 | 0.500 / 0.548; 0.9408 | ood_test: -0.0011 [-0.003, +0.000]; iid_transfer_with_reasoning: +0.0191 [+0.011, +0.029]; iid_transfer_without_reasoning: +0.0981 [+0.077, +0.120] | {'agreement': False, 'agreement_within_refit_noise': True, 'delta_auroc': True} |
| terminal_prerl/with_reasoning/router | 0.775 / 0.512 | 0.7744 | 0.892 / 0.905; 0.9844 | iid_test: +0.0000 [+0.000, +0.000] | {'agreement': False, 'agreement_within_refit_noise': False, 'delta_auroc': True} |
| terminal_prerl/without_reasoning/router | 0.700 / 0.730 | 0.6397 | 0.855 / 0.943; 0.9834 | iid_test: +0.0000 [+0.000, +0.000] | {'agreement': False, 'agreement_within_refit_noise': False, 'delta_auroc': True} |
| diverse_deception/without_reasoning/router | 0.366 / 0.391 | 0.7776 | 0.560 / 0.684; 0.9368 | ood_test: -0.0031 [-0.008, +0.002]; iid_transfer_with_reasoning: +0.3051 [+0.223, +0.388]; iid_transfer_without_reasoning: +0.0412 [-0.027, +0.103] | {'agreement': False, 'agreement_within_refit_noise': False, 'delta_auroc': False} |
| terminal_prerl/with_reasoning/router_all | 0.771 / 0.462 | 0.8340 | 0.900 / 0.911; 0.9790 | iid_test: +0.0000 [+0.000, +0.000] | {'agreement': False, 'agreement_within_refit_noise': False, 'delta_auroc': True} |
| terminal_prerl/without_reasoning/router_all | 0.857 / 0.807 | 0.7867 | 0.870 / 0.961; 0.9794 | iid_test: +0.0000 [+0.000, +0.000] | {'agreement': False, 'agreement_within_refit_noise': False, 'delta_auroc': True} |
| diverse_deception/without_reasoning/router_all | 0.447 / 0.382 | 0.8654 | 0.470 / 0.442; 0.8966 | ood_test: -0.0001 [-0.003, +0.002]; iid_transfer_with_reasoning: +0.0169 [-0.021, +0.053]; iid_transfer_without_reasoning: +0.1456 [+0.105, +0.187] | {'agreement': False, 'agreement_within_refit_noise': False, 'delta_auroc': False} |
Routing agreement at "Yes" between the readers, terminal_prerl/with_reasoning (10688 items): mean 0.9522, lowest 0.8623 at MoE layer 29
Routing agreement at "Yes" between the readers, terminal_prerl/without_reasoning (10688 items): mean 0.9511, lowest 0.8400 at MoE layer 37
Routing agreement at "Yes" between the readers, diverse_deception/without_reasoning (9698 items): mean 0.9313, lowest 0.8277 at MoE layer 37

Jitter: {}
Batching: {}
Sink check: {}
