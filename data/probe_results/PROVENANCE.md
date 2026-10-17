# Provenance of the Inkling-Small probe results

This file records what produced each result in this folder and each deployed monitor in
[`data/probes/Inkling-Small/`](../probes/Inkling-Small/): which code, data and model, and what was checked afterwards.
[`MANIFEST.json`](MANIFEST.json) lists every collected file with its sha256 and the run folder it came from
(`source`). Where `collect` rewrote machine paths in the copy (38 files), it also records the source's sha256
(`source_sha256`); the 18 `hack_counts.json`, which it counted from the fits' per-episode scores, list those files
instead (`derived_from`). Times are UTC; "4-bit" is the NVFP4 reader, "full precision" the bf16 one.

**Code versions.** Code stamping (`src/monitordecorrelation/provenance.py`) arrived on 2026-10-12 between 04:52 and
05:29 (`ca9a1a0` to `28eaff4`). Anything made since records `code: {commit, dirty, diff_sha256, source}`; the deployed
monitors also record `deployed_code`. The datasets, features, fits and the older analyses predate it, so their code
version is reconstructed below. For the fits that matters least: re-running part of each fit reproduces it (see
"Checks": 21 probes with the fitting code of `b416f89`, 2026-10-10, the three 4-bit follow-up probes at `2694851`,
2026-10-12).

## Datasets (`data/probe_datasets/`)

The items (`items.jsonl`) are not in git. They are in the private bucket at
`hf://buckets/probemaxxing/inkling-probes/datasets/<name>/items.jsonl`, and each `manifest.json` pins them by
`items_sha256`.

| dataset | built | code | items_sha256 |
|---|---|---|---|
| `terminal_prerl` | 2026-10-09 19:40 | `8d6d0c1` plus uncommitted changes | `5a92dd96…` |
| `diverse_deception` | 2026-10-09 19:40 | `8d6d0c1` plus uncommitted changes | `c4e712f3…` |
| `terminal_concealed` | 2026-10-10 01:22 | `1c5ef50` plus uncommitted changes | `e6ef5ea6…` |

The builder was first committed later, in `361f6c1`. Its working-tree patch at build time was not kept, so the hash,
not the code, is what pins each dataset. `terminal_prerl` comes from Vlad's pre-RL TerminalEnv evaluation
(`eval_rollouts.jsonl` and `baseline.json`, now in the bucket under `inputs/vlad-2026-09-29-prerl/`, sha256 in the
manifest). `terminal_concealed` is derived from `terminal_prerl`'s items.

## Features (2026-10-10; in the buckets, not in git)

| reader | model and revision | machines | time |
|---|---|---|---|
| 4-bit | `thinkingmachines/Inkling-Small-NVFP4` @ `b6a99534…`, this repo's NVFP4 loader (W4A16) | four 1×B300 spot VMs, plan parts 0–3 of 4 | 02:08–02:42 |
| full precision | `thinkingmachines/Inkling-Small` @ `8cc5877b…`, transformers, bf16 | one 8×B300 VM, two processes (parts 0–1 of 2) | 03:34–04:00 |

- **Both readers:** torch 2.14.1+cu130, transformers 5.19.0. Effort 0.5; follow-up "Did you respond truthfully?" /
  "Yes"; feature format `inkling-probe-features/v2`. The full-precision features are in
  `probemaximizer/inkling-probes-bulk`, the 4-bit ones in `probemaxxing/inkling-probes`.
- **Part 0 of the 4-bit extraction:** its first VM got a spot preemption notice before extracting any shard. It was
  re-run on a fifth VM, 02:28–02:41.
- **Per-process records:** `features/<reader>/<dataset>/runs/*.json` next to the features, with start, end, host,
  shards, tokens, and whether the process stopped early.
- **Code:** the features predate stamping, so `index.json` has no `code`. The extraction ran the code later committed
  in `361f6c1` (2026-10-10 05:46); this is known from file timestamps. Since then, only `45e7a9e` changed the
  extraction code, to record its code version.

## Fitted probes (`fitted/`)

| reader | readouts | fitted | code |
|---|---|---|---|
| 4-bit | `followup`, `mean` | 2026-10-10 03:02 | before stamping (see "Checks") |
| full precision | `followup` | 2026-10-10 06:28 | before stamping |
| 4-bit | `router`, `router_all` | 2026-10-10 13:59 | before stamping |
| full precision | `mean`, `router`, `router_all` | 2026-10-10 16:24 | before stamping |
| full precision | `followup_prev` (the readout fallback) | 2026-10-12 07:02–09:26 | stamped: `82043dc` |

Each `meta.json` records the pre-registration's sha256, the reader (`reader_info`: loader, revision) and the C grid.
The `followup_prev` fits read the same 2026-10-10 features: that readout was stored at extraction, and a copy holding
only it was fetched from the full-precision bucket on 2026-10-12 (06:53–07:02).

## Analyses

| result | made | code |
|---|---|---|
| `ood_refits/<reader>/<readout>/` | 2026-10-10, 15:09 (4-bit follow-up) to 18:45 (full-precision mean) | `experiments/ood_refits.py` as first committed in `b416f89` (14:18); changed since only to add the stamp (`28eaff4`) |
| `learning_curves/bf16/` | 2026-10-11 03:56 | the code committed one minute later in `dd7d402`; changed since only to add the stamp |
| `analyses/20261012T041431Z/dataset_stats.json` | 2026-10-12 04:14 | `experiments/probe_analyses.py` as committed one minute later in `cf402d5`, before stamping |
| `analyses/20261012T073030Z/near_threshold.json` | 2026-10-12 07:30 | stamped: `2694851` |
| `compare/followup/` | 2026-10-12 07:37–07:48 | stamped: `7cdf93d` |
| `compare/mean_router/` | 2026-10-12 07:48–09:52 | stamped: `7cdf93d` |
| `analyses/20261013T191948Z/massive_dims.json` | 2026-10-13 19:19 | stamped: `dae7fd6` |
| `analyses/20261013T191948Z/readout_gap.json` | 2026-10-13 19:21 | stamped: `dae7fd6` |

The two reader comparisons were re-run on 2026-10-12 with the final `compare_readers.py`, which reports the massive
dimensions of "Yes" and records a rule that ran for no reader as not evaluated. The follow-up comparison reproduced
all 110 values of the 2026-10-10 run (the agreement, refit floor, ΔAUROC, rules and decision); the 2026-10-10 files
are in this folder's git history (`53427ae`).

## Readout fallback (`readout_fallback/`, `fitted/bf16/*/followup_prev/`)

The pre-registered readout rule compares `followup` with `followup_prev` if "Yes" is sink-like. Its second clause
("the same ≤ 8 massive dimensions dominate its norm") has no threshold, so on 2026-10-12 the comparison was run
anyway. `readout_fallback/DECISION_RULE.md` fixed how it would be decided at 06:53 UTC, before the first fit started
(07:02; `readout_fallback/run.log`): `followup_prev` only if its dev CV AUROC is strictly higher.
- Result: IID tie (1.0000 both views), OOD `followup_prev` 0.98361 vs 0.98270.
- Decision, 2026-10-13: keep "Yes". Measured at every token of the per-token sample
  (`analyses/20261013T191948Z/massive_dims.json`), the massive-dimension pattern is a property of the residual stream
  at every position, not of "Yes"; only position 0 behaves like a sink. So the rule did not fire, and the fallback fits
  are reported as a sensitivity check (`readout_gap.json` in the same folder).

## Deployed monitors (`data/probes/Inkling-Small/`)

| commit | what |
|---|---|
| `c7ee855` (2026-10-10 08:36) | First deployment: 8 monitors served on the 4-bit reader. The follow-up monitors were full-precision-trained; the mean monitors were 4-bit-trained as a stopgap, until there was disk space for the full-precision mean features. |
| `b683f8e` (2026-10-12 03:56) | The mean monitors redeployed full-precision-trained, as the pre-registered reader rule requires. |
| `0e73ab3` (2026-10-12 05:59) | The follow-up monitors redeployed metadata-only: the fitted probe's repo path and the `deployed_code` stamp. Weights, thresholds and golden sets unchanged. |
| `82043dc` (2026-10-12 06:14) | The same for the mean monitors. |

All eight are served on the 4-bit reader with thresholds recalibrated on its features (`deploy_probes.py`).

`grids.json` (2026-10-13): the probes' task splits as grid keys, which the post-hoc scorer reads to tell which of a
run's eval tasks the probes saw. `deploy_probes.py --grids-only` wrote it from terminal_prerl's items; its `dataset`
and `code` fields record their `items_sha256` and the commit. No monitor changed.

## Where the inputs and raw outputs live

- **This folder and `data/probes/Inkling-Small/`:** every result above, and the deployed monitors.
- **`hf://buckets/probemaxxing/inkling-probes`** (private; access must be granted): the datasets' items
  (`datasets/`), Vlad's pre-RL run (`inputs/`), the 4-bit features with a per-token sample of 10 episodes
  (`features/nvfp4/`; the sample is the data behind the all-token massive-dimension check), the diagnostics, the raw
  fitted probes with their per-episode scores (`probes/fitted/`), and the logs of the extraction and serving-check jobs
  (`runs/`).
- **`hf://buckets/probemaximizer/inkling-probes-bulk`** (private): the full-precision features.
- Since 2026-10-13 the buckets hold only what git can't; their older copies of the small results were deleted then.

## Checks

- **Fits reproduce** (`verify_fits/<run>/`). `experiments/verify_fits.py` recomputed a slice of each fit's CV grid
  (2,207 cells in all), re-derived the selection and refitted the chosen probe. All 24 fitted probes pass:
  - every recomputed grid cell (out-of-fold AUROC, mean and sd) equals `cv_grid.csv`'s to its 5 decimals, and every
    selection comes out the same;
  - the refits' weights, t₁%/t₅%, test logits and main metrics (12 numbers per IID probe, 33 per OOD probe) equal the
    saved ones to the bit (largest difference 0);
  - not re-checked: the IID probes' cross-fitted hard subsets and concealment, leave-one-source-out, the baselines and
    the sensitivity rows.
  - `20261012T052659Z` (05:26–06:34): 21 probes, everything but the 4-bit follow-up probes. It ran `fit_probes.py`
    and `probe_fit.py` as of `b416f89` (2026-10-10; sha256 `83cfb844…`, `5ce6ede2…`), with `verify_fits.py` as of
    `0337bcf`; its own commit is not on this branch, so its summary records the file hashes only.
  - `20261012T063754Z` (06:37–07:05): the three 4-bit follow-up probes, at `2694851`.
  - Since `b416f89` the fitting code (`fit_probes.py`, `probe_fit.py`) changed only to read the pre-registration from
    the repo (`93c455a`), to stamp its outputs and refuse to overwrite them (`28eaff4`), in docstrings (`628fe00`) and
    to check the items' hashes (`0af255e`, the only change since `2694851`).
- **Serving on a GPU, 2026-10-10 08:16–08:28** (one 1×B300, the 4-bit reader at `b6a99534…`, the eight monitors of
  the first deployment, committed just afterwards in `c7ee855`):
  - every monitor's golden check passed, one episode per forward and batched;
  - the HTTP server and `InklingProbeMonitor` matched the scorer exactly (max |Δlogit| 0) on the two OOD follow-up
    monitors, the only ones sent through HTTP;
  - throughput was 3,875 tokens/s one episode per forward and 7,231 batched.

  That run predates the full-precision mean monitors (`b683f8e`), the golden check's bound on the mean shift, and the
  serving fixes of 2026-10-12. The serving path has not run on a GPU since. Its report is in the bucket
  (`runs/20261010-081606-md-servecheck-14c093/`), not in the repo.
- **Dependencies and tests.** With the raised floors (transformers ≥ 5.16, huggingface-hub ≥ 1.5, hf-xet ≥ 1.3,
  safetensors ≥ 0.4.3, torch ≥ 2.11), the full test suite fails the same tests on this branch as on `main` and no
  others (2026-10-13: 54 failures in 7 test files, none of them about the probes).
