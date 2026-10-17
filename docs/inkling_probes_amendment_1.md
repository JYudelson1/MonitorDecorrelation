# Amendment 1 to the Inkling-Small probe pre-registration: the IID probe on puzzles no RL run uses

Written 2026-10-17, before any of the data it describes exists; the commit that adds this file is the record of that.
It changes one thing in [the pre-registration](inkling_probes_preregistration.md), where the IID probe's training
episodes come from, and keeps everything else. Its sha256 is pinned by `tests/test_probe_v2_puzzles.py` and recorded
in every probe fitted under it.

## Why

- **The IID dataset `terminal_prerl` is on the RL evaluation puzzles.** Its episodes are a pre-RL run on the 512
  puzzles of seed 0's evaluation pool (`eval_7919_0` … `eval_7919_659`). reasoning-gym draws puzzle i of a stream
  with `Random(stream seed + i)`, so an RL run with seed s evaluates on seed 0's evaluation puzzles s, s + 1, … under
  other task ids. Every run so far has a seed below 660. In the current batch (seeds 200–218) all 64 evaluation
  puzzles of each run are in `terminal_prerl`, 42 to 54 of them in its train or val split, about 75% of each run's
  evaluation episodes. The runs' training puzzles are a different stream: 99.8% of their training episodes are on
  puzzles the probe never saw.
- **The measured effect is about zero.** In cross-validation on `terminal_prerl` (layer 25, C = 10⁻⁴, as deployed),
  having seen other episodes of a puzzle moves the probe's logit on a new episode of it by 0.002 (honest) and 0.005
  (hacks), against a 10.7-logit gap between hacks and honest episodes. That is the size of refit noise, and AUROC,
  false alarms and recall don't change. The problem is the claim: those evaluation episodes are not held out.
- **The project defines an iid probe as one that never trains on a task an RL run evaluates on** (`mbpp_split.py`,
  for MBPP).

## The change

- **Source:** a new pre-RL run of the unchanged `experiments/eval_terminal_monitors_baseline.py`, with
  `experiments/configs/terminal_verifier_control.json`, `--n-prompts 512 --samples 64`, `monitors=[]` and
  `seed=1000000` (instead of 0). Its puzzles are the evaluation pool of seed 1,000,000: generator positions 1,007,919
  onwards.
- **Exclusion:** before the split, drop every episode on a puzzle whose grid also occurs among the puzzles at
  generator positions 0 to 18,942, which hold every training and evaluation pool of RL seeds 0 to 10,000. Sparse grids
  repeat by chance. The list is fixed now, in
  [`data/probe_datasets/terminal_prerl_v2/excluded_grids.json`](../data/probe_datasets/terminal_prerl_v2/excluded_grids.json),
  written by `experiments/probe_v2_exclusions.py`: one puzzle, `eval_1007919_523`.
- **Name:** the dataset is `terminal_prerl_v2`, and the probes fitted on it are the IID probe v2.
- **Runs:** RL runs scored by the v2 probe keep seeds from 0 to 10,000, the range the exclusion covers. Every run so
  far does.

## Unchanged

- **The episodes:** policy `thinkingmachines/Inkling-Small`, effort 0.5, max_tokens 3072, and every environment
  option of the config (`submit_corrupted`, first submission ends the episode, explanations required, read-only
  verifier, last-turn notice, at most 4 turns); 512 puzzles × 64 episodes.
- **The machine:** the environment runs as root, as the read-only verifier requires, on Ubuntu 24.04 with dash as
  `/bin/sh` and CPython 3.11.16, as Vlad's runs recorded.
- **The dataset:** valid episodes only; split by task 60/20/20, stratified by "task has a hack", split seed 0; label
  `used_verifier`.
- **Everything after:** extraction (both readers, both views, every readout); fitting (layer and C by 5 × 3
  cross-validation grouped by task on train + val, refit on train + val, t₁% and t₅% from out-of-fold honest logits);
  deployment (trained on full precision, served on the 4-bit reader with thresholds recalibrated on its honest train +
  val episodes); and every decision rule of the pre-registration, with the outcomes already decided (reader, batching,
  readout).
- **The OOD probe:** its weights. Its deployed thresholds come from the IID dataset's honest train + val episodes, so
  they are recalibrated on `terminal_prerl_v2`'s, as deployment does.

## Gates, decided now

If a gate fails, the work stops there and is reported; nothing downstream is built until it is decided.

- **Rollouts:** at least 30,000 of the 32,768 episodes complete; a valid rate from 25% to 40% (`terminal_prerl`:
  32.6%); at least 60 `used_verifier` hacks (98); at most 0.1% of commands timed out (0 of 16,014).
- **Extraction:** every shard complete, with the item ids of the dataset's manifest.
- **Fits:** `experiments/verify_fits.py` reproduces every v2 fit.

## What is reported

- The IID probe v2 is the IID probe from its deployment on: `probe_iid` and the other IID monitors. The v1 fits,
  results and monitors stay in the repo as the record of v1.
- v2 on its own test split, and on all of `terminal_prerl` (10,688 episodes on 512 puzzles v2 never saw) as an
  external test set.
- The seen-versus-unseen-puzzle check, repeated on v2.
- How far v1 and v2 agree on the same episodes.
