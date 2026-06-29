#!/usr/bin/env bash
#
# queue_runs.sh — fan a set of experiment configs across seeds and run them in parallel.
#
# For every (config × seed) pair it invokes the standard runner
#   uv run python experiments/run_experiment.py --config <cfg> --set seed=<s> run_name=<derived> ...
# deriving run_name from the config's own run_name by swapping its `_s<n>` token to `_s<seed>`
# (so each run lands in its own data/runs/<run>/ with its rollouts, metrics, plots, etc.).
#
# Parallelism is capped with xargs -P (BSD- and GNU-xargs compatible — works on macOS and Linux).
# Each job's full stdout/stderr is captured to data/runs/<run>/run.log; the terminal shows one
# high-level start/done/FAILED line per job, and the tail of any job that fails.
#
# Usage:
#   scripts/queue_runs.sh -c <config|dir> [-n NUM_SEEDS] [-j MAX_JOBS] [-d] [-s] [-- EXTRA --set k=v ...]
#
#   -c PATH   a config file OR a directory of configs (*.json/*.yaml/*.yml). REQUIRED.
#   -n N      run seeds 0..N-1 (default: 1).
#   -j J      max parallel jobs (default: 4).
#   -d        dry-run: print the plan and exit, launch nothing.
#   -s        skip-existing: skip any run that already finished (data/runs/<run>/QUEUE_DONE present).
#   -h        this help.
#   Anything after a literal `--` is appended verbatim to every run's `--set` overrides,
#             e.g.  -- n_steps=20 kl_coef=0   (handy for a quick short-run smoke of the whole matrix).
#
# Examples:
#   # the full 7-row matrix, 3 seeds each (0,1,2), 4 at a time:
#   scripts/queue_runs.sh -c experiments/configs/mbpp_matrix -n 3 -j 4
#   # one config, 5 seeds, just show what would run:
#   scripts/queue_runs.sh -c experiments/configs/mbpp_matrix/row_control.json -n 5 -d
#   # resume a half-finished batch (skip the runs that already completed):
#   scripts/queue_runs.sh -c experiments/configs/mbpp_matrix -n 3 -j 4 -s
#
# To follow a single live job:  tail -f data/runs/<run_name>/run.log
#
set -euo pipefail

# Absolute path to THIS script and the repo root (so xargs can re-exec the worker, and the runner's
# relative data/runs paths resolve regardless of the caller's cwd).
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
REPO_ROOT="$(cd "$(dirname "$SELF")/.." && pwd)"

# ---- worker mode: `bash queue_runs.sh __worker <seed>\t<cfg>\t<runname>\t<logpath>` -------------------
# Re-entrant entry point invoked once per job by xargs. Kept above arg-parsing so it never falls through
# to the launcher logic. Extra --set overrides arrive via the exported QR_EXTRA_SETS env var.
if [ "${1:-}" = "__worker" ]; then
    cd "$REPO_ROOT"
    IFS=$'\t' read -r seed cfg runname logpath <<<"$2"
    mkdir -p "$(dirname "$logpath")"
    # shellcheck disable=SC2086  # QR_EXTRA_SETS is intentionally word-split into separate k=v tokens
    printf '▶ %s  %-44s seed=%s  → %s\n' "$(date +%H:%M:%S)" "$runname" "$seed" "$logpath"
    if uv run python experiments/run_experiment.py \
            --config "$cfg" --set seed="$seed" run_name="$runname" ${QR_EXTRA_SETS:-} \
            >"$logpath" 2>&1; then
        : >"$(dirname "$logpath")/QUEUE_DONE"   # completion sentinel for -s/--skip-existing
        printf '✓ %s  %-44s done\n' "$(date +%H:%M:%S)" "$runname"
    else
        rc=$?
        printf '✗ %s  %-44s FAILED (exit %s) — last 15 log lines:\n' "$(date +%H:%M:%S)" "$runname" "$rc"
        tail -n 15 "$logpath" | sed 's/^/      │ /'
        exit "$rc"
    fi
    exit 0
fi

# ---- launcher mode -----------------------------------------------------------------------------------
CONFIG=""; NUM_SEEDS=1; MAX_JOBS=4; DRY_RUN=0; SKIP_EXISTING=0
while getopts ":c:n:j:dsh" opt; do
    case "$opt" in
        c) CONFIG="$OPTARG" ;;
        n) NUM_SEEDS="$OPTARG" ;;
        j) MAX_JOBS="$OPTARG" ;;
        d) DRY_RUN=1 ;;
        s) SKIP_EXISTING=1 ;;
        h) sed -n '2,40p' "$SELF"; exit 0 ;;
        \?) echo "unknown option -$OPTARG (try -h)" >&2; exit 2 ;;
        :) echo "option -$OPTARG needs an argument" >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))
EXTRA_SETS="$*"   # everything after `--` (or any trailing args) → passthrough --set overrides

[ -n "$CONFIG" ] || { echo "error: -c <config|dir> is required (try -h)" >&2; exit 2; }
cd "$REPO_ROOT"
[ -e "$CONFIG" ] || { echo "error: config path not found: $CONFIG" >&2; exit 2; }

# Collect the config files (a single file, or every *.json/*.yaml/*.yml in a directory).
configs=()
if [ -d "$CONFIG" ]; then
    while IFS= read -r f; do configs+=("$f"); done < <(find "$CONFIG" -maxdepth 1 -type f \
        \( -name '*.json' -o -name '*.yaml' -o -name '*.yml' \) | sort)
    [ "${#configs[@]}" -gt 0 ] || { echo "error: no *.json/*.yaml/*.yml configs in $CONFIG" >&2; exit 2; }
else
    configs=("$CONFIG")
fi

# Read a config's base run_name (json or yaml) — one uv call per config, not per job.
read_run_name() {
    uv run python - "$1" <<'PY'
import sys, json, pathlib
p = pathlib.Path(sys.argv[1]); t = p.read_text()
try:
    d = json.loads(t)
except json.JSONDecodeError:
    import yaml; d = yaml.safe_load(t)
print(d["run_name"])
PY
}

# Build the (seed, cfg, runname, logpath) job records, tab-delimited, NUL-separated for xargs -0.
records=(); planned=0; skipped=0
echo "Planning: ${#configs[@]} config(s) × $NUM_SEEDS seed(s), up to $MAX_JOBS parallel"
[ -n "$EXTRA_SETS" ] && echo "  extra overrides: $EXTRA_SETS"
echo
for cfg in "${configs[@]}"; do
    base_name="$(read_run_name "$cfg")"
    for ((s = 0; s < NUM_SEEDS; s++)); do
        # swap the _s<digits> token for this seed; if absent, append _s<seed>
        if [[ "$base_name" =~ _s[0-9]+ ]]; then
            runname="$(printf '%s' "$base_name" | sed -E "s/_s[0-9]+/_s${s}/")"
        else
            runname="${base_name}_s${s}"
        fi
        logpath="data/runs/${runname}/run.log"
        if [ "$SKIP_EXISTING" -eq 1 ] && [ -e "data/runs/${runname}/QUEUE_DONE" ]; then
            printf '  skip  %-44s (already done)\n' "$runname"
            skipped=$((skipped + 1)); continue
        fi
        printf '  queue %-44s seed=%s  (%s)\n' "$runname" "$s" "$cfg"
        records+=("${s}"$'\t'"${cfg}"$'\t'"${runname}"$'\t'"${logpath}")
        planned=$((planned + 1))
    done
done

echo
echo "→ $planned job(s) to run, $skipped skipped."
[ "$planned" -gt 0 ] || { echo "nothing to do."; exit 0; }
if [ "$DRY_RUN" -eq 1 ]; then echo "(dry-run: launching nothing)"; exit 0; fi

echo "Launching at $(date +%H:%M:%S) — full logs in data/runs/<run>/run.log"
echo "----------------------------------------------------------------------"
export QR_EXTRA_SETS="$EXTRA_SETS"
# xargs -0 -n1 -P: one NUL-delimited record per worker, MAX_JOBS at a time. `bash "$SELF" __worker`
# gets the record appended as its final arg. `|| true` so a single failed job doesn't abort the batch
# (xargs exits 123 if any child failed); we report failures inline above and summarise below.
fail=0
printf '%s\0' "${records[@]}" | xargs -0 -n1 -P "$MAX_JOBS" bash "$SELF" __worker || fail=$?
echo "----------------------------------------------------------------------"
if [ "$fail" -ne 0 ]; then
    echo "⚠️  batch finished at $(date +%H:%M:%S) with at least one FAILED job (see ✗ lines above)."
    exit 1
fi
echo "✓ batch finished at $(date +%H:%M:%S) — all $planned job(s) completed."
