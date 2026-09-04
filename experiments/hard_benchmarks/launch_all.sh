#!/usr/bin/env bash
# Fan out every (benchmark, subset) measurement as its own process; each samples 64 problems x k.
cd "$(dirname "$0")/../.."
export HF_HOME=/root/hf_cache
K=${K:-4}; N=${N:-64}
run() { nohup uv run python experiments/hard_benchmarks/run_eval.py --benchmark "$1" --subset "$2" \
          --n-problems "$N" --k "$K" --effort 0.5 --max-tokens 32768 --max-prompt-tokens 32768 \
          --concurrency 64 > "data/hard_benchmarks/launch_${1}__${2}.out" 2>&1 & }
run lcb_all all;        run lcb_all hard512;    run lcb_all hard350
run aethercode all;     run aethercode hard130
run icpc_eval all
run usaco all;          run usaco hard228;      run usaco hard86
run leetcode all;       run leetcode hard512;   run leetcode hard1024
run taco_verified all;  run taco_verified hard512; run taco_verified hard1024
run codeforces all;     run codeforces hard512; run codeforces hard1024
run ojbench all;        run ojbench hard111
wait
