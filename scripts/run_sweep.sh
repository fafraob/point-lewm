#!/usr/bin/env bash
# ===========================================================================
#  Run an evaluation sweep on the local machine.
#
#      pixi run bash scripts/run_sweep.sh config/eval_sweep/lidar_tworoom.conf
#      pixi run bash scripts/run_sweep.sh config/eval_sweep/lidar_tworoom.conf --dry-run
#      pixi run bash scripts/run_sweep.sh config/eval_sweep/lidar_tworoom.conf --seeds "1 42"
#      pixi run bash scripts/run_sweep.sh config/eval_sweep/lidar_tworoom.conf --gpus "0,1"
#
#  A sweep config (config/eval_sweep/*.conf, plain bash) names the eval config,
#  the checkpoints to evaluate (EVALS, one "label | hydra overrides" line each),
#  the evaluation seeds (SEEDS) and the episodes per seed (NUM_EVAL). Every
#  (checkpoint, seed) pair is one evaluation; they run one per GPU, in parallel
#  over the GPUs listed in --gpus (default: every GPU nvidia-smi reports, or
#  CUDA_VISIBLE_DEVICES when that is set).
#
#  Results land in ${RESULTS_ROOT}/${SWEEP_NAME}/<label>/seed<seed>/ with
#      results.txt   the evaluation's resolved config and success rate (appended)
#      eval.log      its stdout/stderr
#      exit_code     0 on success
#  and a frozen copy of the config (sweep.conf). Aggregate with
#      python scripts/summarize_evals.py ${RESULTS_ROOT}/${SWEEP_NAME}
#
#  The evaluation seed selects WHICH 50 start-goal windows are drawn from the
#  dataset (or, with eval.starts_file, which pinned list is read), so seeds are
#  the unit of repetition: report mean +- sd over seeds, never a single seed.
#
#  EVAL_TASK in the config selects the evaluation script:
#      eval-lidar      eval_lidar.py     point-cloud world models, recorded goal cloud
#      eval-3dtarget   eval_3dtarget.py  point-cloud world models, goal from a 3-D target
#      eval            eval.py           image world models (LeWM / DINO-WM checkpoints)
#  A failed evaluation does not abort the sweep; it is listed in the summary and
#  makes the script exit non-zero.
# ===========================================================================

set -uo pipefail

usage() {
  echo "usage: $0 <sweep.conf> [--dry-run] [--seeds \"S1 S2 ...\"] [--gpus \"0,1\"]" >&2
  exit 1
}

[[ $# -ge 1 ]] || usage
conf="$1"; shift
[[ -f "$conf" ]] || { echo "ERROR: no such sweep config: $conf" >&2; exit 1; }

dry_run=false
seeds_override=""
gpus_override=""
while (( $# > 0 )); do
  case "$1" in
    --dry-run) dry_run=true ;;
    --seeds)   [[ $# -ge 2 ]] || usage; seeds_override="$2"; shift ;;
    --seeds=*) seeds_override="${1#--seeds=}" ;;
    --gpus)    [[ $# -ge 2 ]] || usage; gpus_override="$2"; shift ;;
    --gpus=*)  gpus_override="${1#--gpus=}" ;;
    *) usage ;;
  esac
  shift
done

# shellcheck source=/dev/null
source "$conf"

: "${RESULTS_ROOT:=${PLWM_RESULTS_ROOT:-eval_results}}"
: "${EVAL_TASK:=eval-lidar}"
: "${SAVE_VIDEOS:=false}"
: "${PYTHON:=python}"

if [[ -n "$seeds_override" ]]; then
  read -ra SEEDS <<< "$seeds_override"
  for s in "${SEEDS[@]}"; do
    [[ "$s" =~ ^-?[0-9]+$ ]] || { echo "ERROR: --seeds takes space-separated integers, got '$s'" >&2; exit 1; }
  done
fi

for var in SWEEP_NAME EVAL_CONFIG NUM_EVAL; do
  [[ -n "${!var:-}" ]] || { echo "ERROR: $var is not set in $conf" >&2; exit 1; }
done
(( ${#EVALS[@]} > 0 )) || { echo "ERROR: EVALS is empty in $conf" >&2; exit 1; }
(( ${#SEEDS[@]} > 0 )) || { echo "ERROR: SEEDS is empty in $conf" >&2; exit 1; }
[[ -f "config/eval/${EVAL_CONFIG}.yaml" ]] || {
  echo "ERROR: config/eval/${EVAL_CONFIG}.yaml does not exist (run from the repository root)" >&2; exit 1; }

case "$EVAL_TASK" in
  eval-lidar)    script=eval_lidar.py ;;
  eval-3dtarget) script=eval_3dtarget.py ;;
  eval)          script=eval.py ;;
  *) echo "ERROR: unknown EVAL_TASK '$EVAL_TASK' (eval-lidar | eval-3dtarget | eval)" >&2; exit 1 ;;
esac

# --- GPUs ------------------------------------------------------------------
if [[ -n "$gpus_override" ]]; then
  IFS=',' read -ra GPUS <<< "$gpus_override"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
fi
(( ${#GPUS[@]} > 0 )) || GPUS=(0)
parallel=${#GPUS[@]}

# --- work items -------------------------------------------------------------
declare -a labels=()
for spec in "${EVALS[@]}"; do
  [[ "$spec" == *"|"* ]] || { echo "ERROR: EVALS entry has no '|' separator: $spec" >&2; exit 1; }
  label="${spec%%|*}"; label="${label// /}"
  [[ -n "$label" ]] || { echo "ERROR: EVALS entry has an empty label: $spec" >&2; exit 1; }
  for prev in ${labels[@]+"${labels[@]}"}; do
    [[ "$prev" != "$label" ]] || { echo "ERROR: duplicate label '$label'" >&2; exit 1; }
  done
  labels+=("$label")
done

n_evals=${#EVALS[@]}
n_seeds=${#SEEDS[@]}
n_items=$(( n_evals * n_seeds ))
sweep_dir="${RESULTS_ROOT}/${SWEEP_NAME}"

echo "=========================================================================="
echo "sweep      : ${SWEEP_NAME}"
echo "script     : ${script}  (config/eval/${EVAL_CONFIG}.yaml)"
echo "protocol   : ${NUM_EVAL} episodes x ${n_seeds} seeds (${SEEDS[*]}) per checkpoint"
echo "evaluations: ${n_evals} checkpoints x ${n_seeds} seeds = ${n_items}, ${parallel} at a time on GPU(s) ${GPUS[*]}"
echo "results    : ${sweep_dir}/<label>/seed<seed>/results.txt"
echo "--------------------------------------------------------------------------"
i=0
for spec in "${EVALS[@]}"; do
  label="${spec%%|*}"; label="${label// /}"
  for seed in "${SEEDS[@]}"; do
    printf '  item %3d  %-24s seed=%-4s %s\n' "$i" "$label" "$seed" "${spec#*|}"
    i=$(( i + 1 ))
  done
done
echo "=========================================================================="

if [[ -d "$sweep_dir" ]] && compgen -G "${sweep_dir}/*/*/results.txt" > /dev/null; then
  echo "NOTE: ${sweep_dir} already holds results. The evaluation scripts APPEND to results.txt;" >&2
  echo "      summarize_evals.py reports the last block per seed. Move the directory aside for a clean sweep." >&2
fi

$dry_run && { echo "[dry-run] nothing run."; exit 0; }

mkdir -p "$sweep_dir"
cp "$conf" "${sweep_dir}/sweep.conf"
if [[ -n "$seeds_override" ]]; then
  sed -i "s/^SEEDS=(.*/SEEDS=(${SEEDS[*]})  # overridden by --seeds/" "${sweep_dir}/sweep.conf"
fi

# One evaluation steps NUM_EVAL environments and encodes their clouds in a
# single process; with several evaluations sharing the machine, per-library
# thread pools sized to every core would only fight each other.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-1}
export LANCE_CPU_THREADS=${LANCE_CPU_THREADS:-1}
export LANCE_IO_THREADS=${LANCE_IO_THREADS:-8}
export MUJOCO_GL=${MUJOCO_GL:-egl}

run_item() {  # run_item <index> <gpu>
  local idx=$1 gpu=$2
  local spec="${EVALS[$(( idx / n_seeds ))]}"
  local label="${spec%%|*}"; label="${label// /}"
  local overrides="${spec#*|}"
  local seed="${SEEDS[$(( idx % n_seeds ))]}"
  local outdir="${sweep_dir}/${label}/seed${seed}"
  mkdir -p "$outdir"
  local start=$SECONDS
  echo "[gpu $gpu] item $idx: ${label} seed=${seed} -> ${outdir}"
  # $overrides is deliberately unquoted: a whitespace-separated list of Hydra
  # key=value tokens that must word-split. Sweep overrides come last so they
  # win. hydra.run.dir is pinned per item so concurrent runs never share it.
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$script" \
    --config-name="${EVAL_CONFIG}" \
    seed="${seed}" \
    eval.num_eval="${NUM_EVAL}" \
    output.dir="${outdir}" \
    output.filename=results.txt \
    "++output.video=${SAVE_VIDEOS}" \
    hydra.run.dir="${outdir}/hydra" \
    ${COMMON_OVERRIDES[@]+"${COMMON_OVERRIDES[@]}"} \
    $overrides > "${outdir}/eval.log" 2>&1
  local rc=$?
  echo "$rc" > "${outdir}/exit_code"
  if (( rc == 0 )); then
    echo "[gpu $gpu] item $idx OK in $(( SECONDS - start ))s"
  else
    echo "[gpu $gpu] item $idx FAILED (exit $rc) after $(( SECONDS - start ))s -- tail of ${outdir}/eval.log:"
    tail -15 "${outdir}/eval.log" | sed "s/^/[gpu $gpu]   /"
  fi
}

# Round-robin queues, one per GPU, each running its items sequentially.
pids=()
for (( q = 0; q < parallel; q++ )); do
  (
    for (( idx = q; idx < n_items; idx += parallel )); do
      run_item "$idx" "${GPUS[$q]}"
    done
  ) &
  pids+=($!)
done
wait "${pids[@]}"

# --- summary ----------------------------------------------------------------
echo
echo "=========================================================================="
echo "[sweep] ${SWEEP_NAME} finished $(date '+%F %T')"
failed=0
for spec in "${EVALS[@]}"; do
  label="${spec%%|*}"; label="${label// /}"
  for seed in "${SEEDS[@]}"; do
    f="${sweep_dir}/${label}/seed${seed}/exit_code"
    rc=$( [[ -f "$f" ]] && cat "$f" || echo "no-result" )
    if [[ "$rc" == "0" ]]; then
      printf '  OK    %-24s seed=%s\n' "$label" "$seed"
    else
      printf '  FAIL  %-24s seed=%s  (exit %s, see %s)\n' "$label" "$seed" "$rc" "${sweep_dir}/${label}/seed${seed}/eval.log"
      failed=$(( failed + 1 ))
    fi
  done
done
echo
echo "[sweep] aggregate with:  python scripts/summarize_evals.py ${sweep_dir}"
echo "=========================================================================="
exit $(( failed > 0 ))
