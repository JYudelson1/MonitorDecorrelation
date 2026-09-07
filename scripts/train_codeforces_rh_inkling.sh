#!/usr/bin/env bash
# Train Inkling-Small on hardest-1024 Codeforces with the rl-rewardhacking-ext "overwrite tests" loophole
# (Codeforces-RH: no tests shown; "will be evaluated by calling a function called run_tests()"), logging
# to Weights & Biases.
#
#   scripts/train_codeforces_rh_inkling.sh                        # the config as-is
#   scripts/train_codeforces_rh_inkling.sh --set n_steps=2 batch_size=2 group_size=4   # smoke test
#   scripts/train_codeforces_rh_inkling.sh --set 'env_options={"hint":"overwrite_aware"}' run_name=cfrh_aware_s0
#   scripts/train_codeforces_rh_inkling.sh --set 'env_options={"hint":"none"}' run_name=cfrh_noloophole_s0  # no-loophole baseline
#   (--set env_options=… REPLACES the dict; the keys left out fall back to the env defaults, which equal
#    the config's values, so naming only the knob you change is fine.)
#
# Same dataset (data/codeforces_ib/hardest1024.jsonl.gz), preflight and `--set` passthrough as
# scripts/train_codeforces_ib_inkling.sh — this only swaps the config
# (experiments/configs/codeforces_rh_inkling_small.json).
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${CONFIG:-experiments/configs/codeforces_rh_inkling_small.json}" exec scripts/train_codeforces_ib_inkling.sh "$@"
