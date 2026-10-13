# Readout fallback: how the comparison is decided (written 2026-10-12T06:53:06Z, before any followup_prev fit)

Pre-registration (docs/inkling_probes_preregistration.md, readout rule): "compare `followup` with `followup_prev` by dev
cross-validation and deploy the higher; on a tie, `followup`." The rule's trigger (massive dimensions) is ambiguous; the
owner chose on 2026-10-12 to run the comparison rather than settle the trigger.

- Reader: bf16, the reader every deployed probe is trained on. Probes: IID with/without reasoning, OOD.
- Fit: experiments/fit_probes.py --readouts followup_prev, same folds, repeats, seed, C grid and selection as the
  followup fits (code at 82043dc).
- Number compared: each fit's selection.cv_auroc (the mean out-of-fold AUROC of the configuration the pre-registered
  selection picks).
- Decision: followup_prev only if its cv_auroc is strictly higher than followup's; equal means followup.
- The bf16 IID followup fits have cv_auroc 1.0, so for IID followup cannot lose; only the OOD probe can change
  (followup cv_auroc 0.98270).
- If followup_prev is higher anywhere, nothing is redeployed until the owner decides.
