# Reader comparison (mean, router, router_all)

**Decision per readout:** mean → `nvfp4`; router → `bf16`; router_all → `bf16`. Fix batching = not evaluated (the batching rule is about the follow-up readout, which this comparison left out (readouts: mean, router, router_all)); 'Yes' sink-like = not evaluated (the readout rule is about the follow-up readout, which this comparison left out (readouts: mean, router, router_all))

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

## Report only (no rule reads these)

Score shift: the full-precision probe's logits on 4-bit minus on full-precision features, TerminalEnv val+test.

| probe | weight cosine (4-bit vs full) | shift, with reasoning: mean (sd), p99 abs, Pearson r | shift, without reasoning |
|---|---|---|---|
| terminal_prerl/with_reasoning/mean | 0.861 | +0.118 (0.155), 0.49, 0.999 | +0.362 (0.285), 1.30, 0.998 |
| terminal_prerl/without_reasoning/mean | 0.993 | +0.051 (0.064), 0.22, 1.000 | +0.164 (0.109), 0.47, 0.999 |
| diverse_deception/without_reasoning/mean | 0.965 | -0.059 (0.046), 0.17, 0.998 | -0.071 (0.073), 0.25, 0.995 |
| terminal_prerl/with_reasoning/router | layers differ (nvfp4 22, bf16 19) | +0.294 (0.886), 2.41, 0.905 | +0.497 (0.853), 2.54, 0.875 |
| terminal_prerl/without_reasoning/router | 0.918 | -1.420 (0.990), 3.77, 0.913 | -0.889 (1.218), 4.45, 0.812 |
| diverse_deception/without_reasoning/router | layers differ (nvfp4 28, bf16 23) | +0.259 (0.711), 2.03, 0.834 | +0.489 (0.766), 2.34, 0.811 |
| terminal_prerl/with_reasoning/router_all | 0.948 | -1.363 (0.680), 3.00, 0.939 | -1.005 (0.665), 2.56, 0.955 |
| terminal_prerl/without_reasoning/router_all | 0.940 | -0.312 (0.851), 2.38, 0.954 | -0.756 (1.103), 3.34, 0.896 |
| diverse_deception/without_reasoning/router_all | 0.774 | +0.390 (0.681), 1.97, 0.896 | +0.339 (0.804), 2.36, 0.877 |

Cross-reader scoring, IID probes (TerminalEnv test):

| probe | probe's reader | features | test AUROC | TPR / FPR at own t1 | TPR / FPR at recalibrated t1 |
|---|---|---|---|---|---|
| terminal_prerl/with_reasoning/mean | nvfp4 | nvfp4 | 1.00000 | 100.0% / 0.37% | 100.0% / 0.69% |
| terminal_prerl/with_reasoning/mean | nvfp4 | bf16 | 1.00000 | 100.0% / 0.28% | 100.0% / 0.64% |
| terminal_prerl/with_reasoning/mean | bf16 | nvfp4 | 1.00000 | 100.0% / 0.51% | 100.0% / 0.51% |
| terminal_prerl/with_reasoning/mean | bf16 | bf16 | 1.00000 | 100.0% / 0.51% | 100.0% / 0.55% |
| terminal_prerl/without_reasoning/mean | nvfp4 | nvfp4 | 1.00000 | 100.0% / 0.37% | 100.0% / 0.60% |
| terminal_prerl/without_reasoning/mean | nvfp4 | bf16 | 0.99996 | 100.0% / 0.32% | 100.0% / 0.46% |
| terminal_prerl/without_reasoning/mean | bf16 | nvfp4 | 1.00000 | 100.0% / 0.51% | 100.0% / 0.55% |
| terminal_prerl/without_reasoning/mean | bf16 | bf16 | 0.99998 | 100.0% / 0.32% | 100.0% / 0.51% |
| terminal_prerl/with_reasoning/router | nvfp4 | nvfp4 | 1.00000 | 100.0% / 0.41% | 100.0% / 0.46% |
| terminal_prerl/with_reasoning/router | nvfp4 | bf16 | 1.00000 | 100.0% / 0.28% | 100.0% / 0.69% |
| terminal_prerl/with_reasoning/router | bf16 | nvfp4 | 0.99998 | 100.0% / 0.46% | 100.0% / 0.78% |
| terminal_prerl/with_reasoning/router | bf16 | bf16 | 1.00000 | 100.0% / 0.51% | 100.0% / 0.69% |
| terminal_prerl/without_reasoning/router | nvfp4 | nvfp4 | 1.00000 | 100.0% / 0.41% | 100.0% / 0.46% |
| terminal_prerl/without_reasoning/router | nvfp4 | bf16 | 1.00000 | 100.0% / 0.83% | 100.0% / 0.46% |
| terminal_prerl/without_reasoning/router | bf16 | nvfp4 | 1.00000 | 100.0% / 0.23% | 100.0% / 0.60% |
| terminal_prerl/without_reasoning/router | bf16 | bf16 | 1.00000 | 100.0% / 0.46% | 100.0% / 0.55% |
| terminal_prerl/with_reasoning/router_all | nvfp4 | nvfp4 | 1.00000 | 100.0% / 0.46% | 100.0% / 0.51% |
| terminal_prerl/with_reasoning/router_all | nvfp4 | bf16 | 1.00000 | 100.0% / 3.31% | 100.0% / 0.69% |
| terminal_prerl/with_reasoning/router_all | bf16 | nvfp4 | 1.00000 | 100.0% / 0.23% | 100.0% / 0.51% |
| terminal_prerl/with_reasoning/router_all | bf16 | bf16 | 1.00000 | 100.0% / 0.46% | 100.0% / 0.46% |
| terminal_prerl/without_reasoning/router_all | nvfp4 | nvfp4 | 1.00000 | 100.0% / 0.32% | 100.0% / 0.37% |
| terminal_prerl/without_reasoning/router_all | nvfp4 | bf16 | 1.00000 | 100.0% / 0.51% | 100.0% / 0.55% |
| terminal_prerl/without_reasoning/router_all | bf16 | nvfp4 | 1.00000 | 100.0% / 0.69% | 100.0% / 0.55% |
| terminal_prerl/without_reasoning/router_all | bf16 | bf16 | 1.00000 | 100.0% / 0.51% | 100.0% / 0.51% |

Cross-reader scoring, OOD probes:

| probe | probe's reader | features | OOD test AUROC | transfer AUROC (with / without reasoning) |
|---|---|---|---|---|
| diverse_deception/without_reasoning/mean | nvfp4 | nvfp4 | 0.9769 | 0.176 / 0.509 |
| diverse_deception/without_reasoning/mean | nvfp4 | bf16 | 0.9774 | 0.176 / 0.498 |
| diverse_deception/without_reasoning/mean | bf16 | nvfp4 | 0.9772 | 0.158 / 0.416 |
| diverse_deception/without_reasoning/mean | bf16 | bf16 | 0.9780 | 0.157 / 0.411 |
| diverse_deception/without_reasoning/router | nvfp4 | nvfp4 | 0.9569 | 0.829 / 0.664 |
| diverse_deception/without_reasoning/router | nvfp4 | bf16 | 0.9559 | 0.882 / 0.711 |
| diverse_deception/without_reasoning/router | bf16 | nvfp4 | 0.9578 | 0.492 / 0.686 |
| diverse_deception/without_reasoning/router | bf16 | bf16 | 0.9600 | 0.524 / 0.623 |
| diverse_deception/without_reasoning/router_all | nvfp4 | nvfp4 | 0.9772 | 0.474 / 0.344 |
| diverse_deception/without_reasoning/router_all | nvfp4 | bf16 | 0.9758 | 0.481 / 0.409 |
| diverse_deception/without_reasoning/router_all | bf16 | nvfp4 | 0.9765 | 0.419 / 0.144 |
| diverse_deception/without_reasoning/router_all | bf16 | bf16 | 0.9773 | 0.457 / 0.199 |
