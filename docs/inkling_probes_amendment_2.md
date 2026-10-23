# Amendment 2 to the Inkling-Small probe pre-registration: the v2 probes on the full-precision reader alone

Written 2026-10-23, before any feature of `terminal_prerl_v2` exists; the commit that adds this file is the record of
that. It applies to the probes of [amendment 1](inkling_probes_amendment_1.md): the IID probe v2, and the OOD probe with
its thresholds from `terminal_prerl_v2`. It changes which reader extracts and serves them and how their features are
batched, and keeps everything else. Its sha256 is pinned by `tests/test_probe_v2_puzzles.py` and recorded in every
probe fitted under it. The v1 monitors stay as deployed: fitted on full precision, served on 4-bit.

## Why

- **A run trains against the score the probe was fitted on.** The v1 monitors are served on the 4-bit reader, with t₁%
  and t₅% recalibrated on its features. The reader comparison found that the 4-bit reader changes what the "Yes"
  probes flag beyond noise. Across the two readers they rank episodes with Spearman 0.77–0.92, below the
  pre-registered 0.95 and below their own refits (`data/probe_results/compare/followup/decision.json`). On the reader
  they were fitted on, the probes keep their own thresholds.
- **The RL'd policy can then be probed with the same reader.** Probing a run's own policy, the base model plus the
  run's LoRA, is the deferred route to the obfuscated-activations decomposition. It needs the full-precision reader:
  the 4-bit loader (`whitebox/inkling_nvfp4.py`) can't apply a LoRA. With the v2 probes on full precision, probes of
  the base model and of the RL'd one share a reader.
- **The cost:** a full-precision copy takes three or four B300s instead of one. It needs them for the extraction, and
  for as long as runs train against a v2 probe. An 8-GPU machine holds two copies.

## The change

- **Reader:** `thinkingmachines/Inkling-Small` @ `8cc5877b44d343f88b92086aa1fb72897950f06a`, BF16, loaded by
  transformers. The deployed v1 probes were all fitted on this reader.
  - It extracts `terminal_prerl_v2` and scores every v2 monitor, live and post hoc.
  - The 4-bit reader extracts nothing for v2.
  - The reader comparison is not repeated: its outcome, fit on full precision, stands.
- **Batching:** the extraction and the scoring run one episode per forward pass (`max_batch_tokens` 1). This is the
  batching rule's pre-registered fix ("one episode per forward, for extraction and serving").
  - The rule fired on 2026-10-10, on the 4-bit reader's diagnostics. Those are the only diagnostics that ran: the
    full-precision reader's VM was preempted before its own.
  - The v1 features were extracted batched (65,536 tokens per forward), while the v1 server scores one episode at a
    time. v2 does both one at a time.
- **Thresholds:** those of the fitting reader, never recalibrated onto another.
  - IID probe v2: t₁% and t₅% from its out-of-fold honest dev logits, as the pre-registration fits them.
  - OOD probe: the 99th and 95th percentiles of its logits on `terminal_prerl_v2`'s honest train + val episodes, per
    view (amendment 1), on the v2 features.
- **The OOD probe** keeps its weights: the full-precision fit on `diverse_deception`
  (`data/probe_results/fitted/bf16/diverse_deception/without_reasoning/`), whose features were extracted batched. Only
  its thresholds come from the v2 features.
- **Golden check:** each v2 monitor's golden episodes and logits come from the v2 features. The server re-scores them
  one episode at a time.

## Measured in the same session (reported; decides nothing)

- **The full-precision reader's batching jitter.** The session's diagnostics run on the pre-registered sample: 32 hacks
  and 32 honest episodes of `terminal_prerl_v2`'s train split in both views (128 inputs), and 50 `diverse_deception`
  pairs (100 items). Each is read batched at 65,536 tokens, as the v1 extraction was, and one at a time.
- **The batching rule's numbers on that jitter:** the shares of hacks and of honest episodes whose logit lies within
  its 99th percentile of t₁%. They are computed as for the 4-bit reader (`experiments/compare_readers.py`, the IID
  with-reasoning follow-up probe) and for each deployed follow-up monitor (`experiments/deploy_probes.py`).

## Unchanged

- **Amendment 1**, except where it extracts on both readers and serves on the 4-bit one. That includes the v2
  puzzles and their exclusion, the episodes, the machine, the dataset, the fitting, its gates and what it reports.
- **The external test set** of amendment 1 (all of `terminal_prerl`) uses that dataset's existing full-precision
  features, extracted batched on 2026-10-10. Its false-alarm rates carry that batching jitter, and they are reported
  with the jitter measured above.
- **The pre-registration's rules**, with the outcomes already decided: reader (full precision), batching (one episode
  per forward), readout ("Yes").

## Gates, decided now

These add to amendment 1's. If a gate fails, the work stops there and is reported.

- **Extraction:** the v2 features' `index.json` names the reader above (loader `transformers`, revision
  `8cc5877b…`) and `max_batch_tokens` 1. `experiments/fit_probes.py` refuses any other v2 features.
- **Serving:** a v2 monitor scores a run only on a server whose golden check passed, with the reader above, on the
  machine that serves it.
