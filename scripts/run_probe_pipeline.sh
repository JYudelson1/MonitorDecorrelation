#!/usr/bin/env bash
# The Inkling probe pipeline on hub, after the GPU session, as one resumable entry point:
#   fetch -> fit -> verify -> compare -> analyses -> deploy -> report
#
#   scripts/run_probe_pipeline.sh [--dry-run] [--stages fetch,fit,...] [--run DIR] [options]  2>&1 | tee -a pipeline.log
#
# Every command is printed ("+ ...") before it runs. Outputs go to a new <scratch>/pipeline/<UTC timestamp>/ (--run
# picks the folder). Each finished step leaves a marker in <run>/.done/, so rerunning with the same --run resumes
# after the last finished step; a failed step stops the pipeline. The report steps always run (they are quick).
# --dry-run prints the commands and runs nothing. It works from the repo root, wherever it is called from.
#
# Before it, the GPU session extracts the features (not run here). From the repo root, on a fresh spot VM, through
# the launcher (USAGE.md, "Inkling probes"); e.g. the full-precision reader on an 8x B300:
#   python3 scripts/nebius_job.py run --hours 1.0 --max-gpu-price 3.5 --preset 8gpu-192vcpu-2768gb --attempts 2 \
#     --hf thinkingmachines/Inkling-Small@8cc5877 \
#     --push data/probe_datasets/terminal_prerl/items.jsonl:data/probe_datasets/terminal_prerl/items.jsonl \
#     --push data/probe_datasets/diverse_deception/items.jsonl:data/probe_datasets/diverse_deception/items.jsonl \
#     --push data/probe_datasets/terminal_concealed/items.jsonl:data/probe_datasets/terminal_concealed/items.jsonl \
#     --upload 'results=hf://buckets/probemaxxing/inkling-probes/runs/{stamp}-{name}' --fetch results -- \
#     'mkdir -p results && uv run python experiments/run_probe_session.py \
#        --plan experiments/session_plans/probe_session_bf16.json 2>&1 | tee results/session.log; exit ${PIPESTATUS[0]}'
# The plan names each reader's feature store and the diagnostics store. The 4-bit reader ran on four 1-GPU VMs
# (--preset 1gpu-24vcpu-346gb --hf thinkingmachines/Inkling-Small-NVFP4@b6a9953, plans
# experiments/session_plans/probe_session_nvfp4_part{0..3}of4.json).
#
# Stages (--stages picks some, in this order):
#   fetch     scripts/fetch_features.py: each reader's features per readout set, keeping only that set's tensors,
#             into <features>/<set>/<reader>; and each reader's diagnostics, if any, into <diagnostics>/<reader>
#   fit       experiments/fit_probes.py per reader and readout set, into <run>/probes/<reader>
#   verify    experiments/verify_fits.py, into <run>/verify_fits/<UTC timestamp>
#   compare   experiments/compare_readers.py per readout set, into <run>/compare/<set> (needs two readers)
#   analyses  experiments/ood_refits.py per reader and --refit-readouts, into <run>/ood_refits/<reader>/<readout>;
#             experiments/learning_curves.py per --curves-readers, into <run>/learning_curves/<reader>;
#             experiments/probe_analyses.py dataset-stats and near-threshold (the --deploy-reader's follow-up
#             probes on its own features), into <run>/analyses
#   deploy    experiments/deploy_probes.py: the --deploy-reader's probes served on the --serve-reader, into
#             --deployed (data/probes/Inkling-Small; review and commit it)
#   report    experiments/probe_report.py collect (into data/probe_results) and render --docs
#
# Readout sets (--set NAME=READOUTS, repeatable; default followup=followup and mean_router=mean,router,router_all):
# one fetch per set and reader, so hub's disk holds only the tensors a set needs (the 2026-10-10 run had followup in
# ../scratch/features and the rest in ../scratch/features_mr; reuse such folders with --features-of SET=DIR).
# Fitting baselines needs the follow-up and mean readouts in one folder: one set with both, and --fit-args '--jobs 6'.
#
# Options (defaults in brackets):
#   --run DIR             the run folder [<scratch>/pipeline/<UTC timestamp>]
#   --scratch DIR         [../scratch]
#   --features DIR        features root, <features>/<set>/<reader> [<scratch>/features]
#   --features-of SET=DIR a set's features folder, holding <reader>/ (repeatable)
#   --diagnostics DIR     [<scratch>/diagnostics]
#   --readers LIST        [nvfp4,bf16]
#   --store READER=URL    a reader's feature store (repeatable) [the 2026-10-10 buckets]
#   --diagnostics-store URL  [hf://buckets/probemaxxing/inkling-probes/diagnostics]
#   --set NAME=READOUTS   a readout set (repeatable)
#   --fit-args ARGS       [--skip-baselines --jobs 6]
#   --refit-readouts LIST [followup,mean]      --refits N [50]
#   --curves-readers LIST [the readers]
#   --deploy-reader R [bf16]   --serve-reader R [nvfp4]   --deploy-readouts LIST [followup,mean]
#   --deployed DIR        [data/probes/Inkling-Small]
#   --results DIR         probe_report.py collect's --out [data/probe_results]
#   --docs FILE           rendered by probe_report.py if it exists [docs/INKLING_PROBES.md]
#   --python CMD          [.venv/bin/python if present, else "uv run python"]
#   --stages LIST         [fetch,fit,verify,compare,analyses,deploy,report]
#   --dry-run
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SELF=$HERE/$(basename "${BASH_SOURCE[0]}")
cd "$HERE/.."
STAGES_ALL=(fetch fit verify compare analyses deploy report)
DRY=0 RUN="" SCRATCH=../scratch FEATURES="" DIAGNOSTICS="" READERS=nvfp4,bf16 STAGES=$(IFS=,; echo "${STAGES_ALL[*]}")
DIAG_STORE=hf://buckets/probemaxxing/inkling-probes/diagnostics
FIT_ARGS="--skip-baselines --jobs 6" REFIT_READOUTS=followup,mean REFITS=50 CURVES_READERS=""
DEPLOY_READER=bf16 SERVE_READER=nvfp4 DEPLOY_READOUTS=followup,mean DEPLOYED=data/probes/Inkling-Small
RESULTS=data/probe_results DOCS=docs/INKLING_PROBES.md PYTHON=""
declare -A STORE=([nvfp4]=hf://buckets/probemaxxing/inkling-probes/features/nvfp4
                  [bf16]=hf://buckets/probemaximizer/inkling-probes-bulk/features/bf16)
declare -A FEAT_OF=()
SET_NAMES=() SET_READOUTS=()

usage() { sed -n '2,/^set -euo/p' "$SELF" | sed '$d; s/^# \{0,1\}//'; }
die() { echo "run_probe_pipeline.sh: $*" >&2; exit 2; }
pair() { [[ "$2" == *=* && -n "${2%%=*}" && -n "${2#*=}" ]] || die "$1 wants NAME=VALUE, got '$2'"; }

while (($#)); do
  case "$1" in
    --dry-run) DRY=1; shift; continue ;;
    -h|--help) usage; exit 0 ;;
  esac
  (($# >= 2)) || die "$1 needs a value"
  case "$1" in
    --run) RUN=$2 ;;
    --scratch) SCRATCH=$2 ;;
    --features) FEATURES=$2 ;;
    --features-of) pair "$1" "$2"; FEAT_OF[${2%%=*}]=${2#*=} ;;
    --diagnostics) DIAGNOSTICS=$2 ;;
    --readers) READERS=$2 ;;
    --store) pair "$1" "$2"; STORE[${2%%=*}]=${2#*=} ;;
    --diagnostics-store) DIAG_STORE=$2 ;;
    --set) pair "$1" "$2"; SET_NAMES+=("${2%%=*}"); SET_READOUTS+=("${2#*=}") ;;
    --fit-args) FIT_ARGS=$2 ;;
    --refit-readouts) REFIT_READOUTS=$2 ;;
    --refits) REFITS=$2 ;;
    --curves-readers) CURVES_READERS=$2 ;;
    --deploy-reader) DEPLOY_READER=$2 ;;
    --serve-reader) SERVE_READER=$2 ;;
    --deploy-readouts) DEPLOY_READOUTS=$2 ;;
    --deployed) DEPLOYED=$2 ;;
    --results) RESULTS=$2 ;;
    --docs) DOCS=$2 ;;
    --python) PYTHON=$2 ;;
    --stages) STAGES=$2 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
  shift 2
done

((${#SET_NAMES[@]})) || { SET_NAMES=(followup mean_router); SET_READOUTS=(followup mean,router,router_all); }
RUN=${RUN:-$SCRATCH/pipeline/$(date -u +%Y%m%dT%H%M%SZ)}
FEATURES=${FEATURES:-$SCRATCH/features} DIAGNOSTICS=${DIAGNOSTICS:-$SCRATCH/diagnostics}
CURVES_READERS=${CURVES_READERS:-$READERS}
IFS=, read -r -a READER_LIST <<<"$READERS"
IFS=, read -r -a STAGE_LIST <<<"$STAGES"
for s in "${STAGE_LIST[@]}"; do [[ " ${STAGES_ALL[*]} " == *" $s "* ]] || die "unknown stage '$s'"; done
if [[ -n "$PYTHON" ]]; then read -r -a PY <<<"$PYTHON"
elif [[ -x .venv/bin/python ]]; then PY=(.venv/bin/python)
else PY=(uv run python); fi

wants() { [[ ",$STAGES," == *",$1,"* ]]; }
quote() {                                                         # the command as one pastable line
  local out=() a q="'"
  for a in "$@"; do
    if [[ "$a" =~ ^[A-Za-z0-9_./:=,@%+-]+$ ]]; then out+=("$a"); else out+=("$q${a//$q/$q\\$q$q}$q"); fi
  done
  echo "${out[*]}"
}
feat() { echo "${FEAT_OF[$1]:-$FEATURES/$1}/$2"; }             # feat SET READER
set_of() {                                                        # the set holding a readout
  local i
  for i in "${!SET_NAMES[@]}"; do
    [[ ",${SET_READOUTS[$i]}," == *",$1,"* ]] && { echo "${SET_NAMES[$i]}"; return 0; }
  done
  die "no --set holds readout '$1'"
}
keep_of() {                                                       # the tensors a readout set needs
  local r t tensors out="" readouts
  IFS=, read -r -a readouts <<<"$1"
  for r in "${readouts[@]}"; do
    case "$r" in
      mean) tensors="mean_action mean_reasoning" ;;
      router|router_all) tensors="router_logits router_topk router_margin" ;;
      *) tensors=$r ;;
    esac
    for t in $tensors counts answer_logp; do
      if [[ ",$out," != *",$t,"* ]]; then out+=${out:+,}$t; fi
    done
  done
  echo "$out"
}
common() {                                                        # the readouts in both comma-lists
  local r out="" list
  IFS=, read -r -a list <<<"$1"
  for r in "${list[@]}"; do
    if [[ ",$2," == *",$r,"* ]]; then out+=${out:+,}$r; fi
  done
  echo "$out"
}
step() {                                                          # step ID CMD...: run CMD unless ID is done
  local id=$1; shift
  if [[ -e "$RUN/.done/$id" ]]; then echo "# done already: $id"; return 0; fi
  echo "+ $(quote "$@")"
  ((DRY)) && return 0
  "$@"
  mkdir -p "$RUN/.done" && date -u +%Y-%m-%dT%H:%M:%SZ >"$RUN/.done/$id"
}
always() {                                                        # always CMD...: run CMD (no marker)
  echo "+ $(quote "$@")"
  ((DRY)) && return 0
  "$@"
}

FETCH_DIAGNOSTICS='import sys
from monitordecorrelation.whitebox.store import open_store
store, dest = open_store(sys.argv[1]), sys.argv[2]
files = store.files()
for rel in files:
    store.get(rel, f"{dest}/{rel}")
print(f"{len(files)} files from {sys.argv[1]} to {dest}" if files else f"no diagnostics in {sys.argv[1]}")'

SETS=""
for i in "${!SET_NAMES[@]}"; do SETS+=" ${SET_NAMES[$i]}=${SET_READOUTS[$i]}"; done
echo "# run $RUN; readers $READERS; sets$SETS; stages $STAGES; python ${PY[*]}$( ((DRY)) && echo '; dry run' || true)"
if wants analyses; then                                           # fail now, not after hours of fitting
  for ro in followup ${REFIT_READOUTS//,/ }; do set_of "$ro" >/dev/null; done
fi
((DRY)) || mkdir -p "$RUN"

if wants fetch; then
  for i in "${!SET_NAMES[@]}"; do
    for r in "${READER_LIST[@]}"; do
      [[ -n "${STORE[$r]:-}" ]] || die "no --store for reader $r"
      step "fetch-${SET_NAMES[$i]}-$r" "${PY[@]}" scripts/fetch_features.py "${STORE[$r]}" \
        "$(feat "${SET_NAMES[$i]}" "$r")" --keep "$(keep_of "${SET_READOUTS[$i]}")" --workers 4
    done
  done
  for r in "${READER_LIST[@]}"; do
    step "fetch-diagnostics-$r" "${PY[@]}" -c "$FETCH_DIAGNOSTICS" "$DIAG_STORE/$r" "$DIAGNOSTICS/$r"
  done
fi

if wants fit; then
  read -r -a FIT_EXTRA <<<"$FIT_ARGS"
  for i in "${!SET_NAMES[@]}"; do
    for r in "${READER_LIST[@]}"; do
      step "fit-${SET_NAMES[$i]}-$r" "${PY[@]}" experiments/fit_probes.py --reader "$r" \
        --features "$(feat "${SET_NAMES[$i]}" "$r")" --out "$RUN/probes/$r" --readouts "${SET_READOUTS[$i]}" \
        "${FIT_EXTRA[@]}"
    done
  done
fi

if wants verify; then
  CHECKS=()
  for i in "${!SET_NAMES[@]}"; do
    for r in "${READER_LIST[@]}"; do CHECKS+=(--check "$r/${SET_READOUTS[$i]}=$(feat "${SET_NAMES[$i]}" "$r")"); done
  done
  step verify "${PY[@]}" experiments/verify_fits.py --probes "$RUN/probes" --out "$RUN/verify_fits" "${CHECKS[@]}"
fi

if wants compare; then
  if ((${#READER_LIST[@]} < 2)); then
    echo "# compare needs two readers; skipped"
  else
    for i in "${!SET_NAMES[@]}"; do
      step "compare-${SET_NAMES[$i]}" "${PY[@]}" experiments/compare_readers.py --probes "$RUN/probes" \
        --features "${FEAT_OF[${SET_NAMES[$i]}]:-$FEATURES/${SET_NAMES[$i]}}" --diagnostics "$DIAGNOSTICS" \
        --out "$RUN/compare/${SET_NAMES[$i]}" --readouts "${SET_READOUTS[$i]}"
    done
  fi
fi

if wants analyses; then
  IFS=, read -r -a REFIT_LIST <<<"$REFIT_READOUTS"
  for r in "${READER_LIST[@]}"; do
    for ro in "${REFIT_LIST[@]}"; do
      S=$(set_of "$ro")
      step "refits-$r-$ro" "${PY[@]}" experiments/ood_refits.py \
        --probe "$RUN/probes/$r/diverse_deception/without_reasoning/$ro" --features "$(feat "$S" "$r")" \
        --out "$RUN/ood_refits/$r/$ro" --refits "$REFITS"
    done
  done
  S=$(set_of followup)
  IFS=, read -r -a CURVE_LIST <<<"$CURVES_READERS"
  for r in "${CURVE_LIST[@]}"; do
    step "curves-$r" "${PY[@]}" experiments/learning_curves.py --probes "$RUN/probes/$r" \
      --features "$(feat "$S" "$r")" --out "$RUN/learning_curves/$r"
  done
  step analyses-dataset-stats "${PY[@]}" experiments/probe_analyses.py dataset-stats --out "$RUN/analyses"
  P=$RUN/probes/$DEPLOY_READER
  NEAR=$P/terminal_prerl/with_reasoning/followup,$P/terminal_prerl/without_reasoning/followup
  NEAR+=,$P/diverse_deception/without_reasoning/followup
  step analyses-near-threshold "${PY[@]}" experiments/probe_analyses.py near-threshold --probes "$NEAR" \
    --features "$(feat "$S" "$DEPLOY_READER")" --out "$RUN/analyses"
fi

if wants deploy; then
  for i in "${!SET_NAMES[@]}"; do
    ROS=$(common "${SET_READOUTS[$i]}" "$DEPLOY_READOUTS")
    [[ -n "$ROS" ]] || continue
    step "deploy-${SET_NAMES[$i]}" "${PY[@]}" experiments/deploy_probes.py --probes "$RUN/probes/$DEPLOY_READER" \
      --features "$(feat "${SET_NAMES[$i]}" "$SERVE_READER")" --diagnostics "$DIAGNOSTICS/$SERVE_READER" \
      --serving-reader "$SERVE_READER" --readouts "$ROS" --out "$DEPLOYED"
  done
fi

if wants report; then
  COMPARES=()
  for name in "${SET_NAMES[@]}"; do COMPARES+=(--compare "$name=$RUN/compare/$name"); done
  always "${PY[@]}" experiments/probe_report.py collect --scratch "$RUN" "${COMPARES[@]}" \
    --verify-fits "$RUN/verify_fits" --analyses "$RUN/analyses" --out "$RESULTS"
  if [[ -f "$DOCS" ]]; then
    always "${PY[@]}" experiments/probe_report.py render "$DOCS" --results "$RESULTS" --deployed "$DEPLOYED"
  else
    echo "# no $DOCS: nothing to render"
  fi
fi
echo "# finished: $RUN"
