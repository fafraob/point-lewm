#!/bin/bash
# ===========================================================================
#  Attention maps for every arm x environment: compute, then render top-down.
#
#      scripts/attention_all.sh                 # all of them (CLS attention)
#      METHOD=occlusion scripts/attention_all.sh # or: scripts/occlusion_all.sh
#      scripts/attention_all.sh --dry-run       # check paths, print the calls
#      scripts/attention_all.sh reacher         # only entries matching a word
#      scripts/attention_all.sh --skip-done     # resume: leave finished entries alone
#      FRAMES=5 scripts/attention_all.sh        # more frames each
#
#  Output: attention/<env>_<arm>/ with, per frame, the attention values
#  (.ply, `property float attention`), the per-point weights (.npy) and a
#  token-centre render; the figure panels themselves land in the sensor/
#  subdirectory, which is what scripts/figure_attention.py composes.
#
#  The camera and the ramp are applied at RENDER time, so a different look costs
#  a re-render, not a re-run of the model:
#      scripts/render_attention.sh            # every entry, current presets
#      scripts/render_attention.sh reacher    # just one environment
#
#  Each entry is one model load (~2-3 min), so the full sweep is ~20 min.
# ===========================================================================

set -uo pipefail
cd "$(dirname "$0")/.."

# Tables follow the repository convention in paths.py: the LiDAR tables live
# under $PLWM_DATA_ROOT (default data/). LOGS is the root the ENTRIES below are
# resolved against: `.` for the released checkpoints (checkpoints/<env>/<model>/),
# $PLWM_LOGS_ROOT for training runs of your own.
DATA=${DATA:-${PLWM_DATA_ROOT:-data}}
LOGS=${LOGS:-.}
METHOD=${METHOD:-attention}     # attention | occlusion | grad
OUT=${OUT:-$METHOD}             # attention/<env>_<arm>, occlusion/<env>_<arm>, ...
case "$METHOD" in
  attention) SUFFIX=rollout ;;  # viz_attention names files by what it computed
  *)         SUFFIX=$METHOD ;;
esac
FRAMES=${FRAMES:-4}
SEED=${SEED:-0}
# Extra viz_attention flags for every entry, e.g. occlusion resolution:
#   VIZ_ARGS="--cell-size 0.08 --occlusion-batch 4" scripts/occlusion_all.sh
VIZ_ARGS=${VIZ_ARGS:-}

# Rendering is scripts/render_attention.sh, which holds a per-environment camera
# preset. It is a separate script because the two stages have very different
# costs: computing a cloud is a model load (minutes), rendering it is seconds,
# and the look is what gets iterated on. SUBDIR is where its panels land.
SUBDIR=${SUBDIR:-sensor}

# arm | env | run | policy | table under $DATA | extra viz_attention args
# The checkpoint is $LOGS/<run>/checkpoints/<policy>. With the defaults (LOGS=.,
# run=.) that is the released checkpoint layout, checkpoints/<env>/<model>/weights.pt
# (scripts/download_checkpoints.py). For training runs of your own set
# LOGS=experiment_logs and use <run folder> | <config name>/weights_final.pt.
ENTRIES=(
  # OGB-Cube is regrouped at load time. Both OGB-Cube checkpoints were trained
  # and are evaluated with group_radius 0.03 m. Figure 6 renders the OGB-Cube
  # maps with a load-time ball-query radius of 0.074 m (--group-radius 0.074) so
  # that the groups cover the whole cloud for display. This affects only the
  # visualization. The encoder weights are unchanged.
  "point_lewm      | cube    | . | cube/point-lewm/weights.pt          | cube.lance     | --group-radius 0.074"
  "point_delta_jepa | cube    | . | cube/point-delta-jepa/weights.pt    | cube.lance     | --group-radius 0.074"
  "point_lewm      | tworoom | . | tworoom/point-lewm/weights.pt       | two_room.lance"
  "point_delta_jepa | tworoom | . | tworoom/point-delta-jepa/weights.pt | two_room.lance"
  "point_lewm      | reacher | . | reacher/point-lewm/weights.pt       | reacher.lance"
  "point_delta_jepa | reacher | . | reacher/point-delta-jepa/weights.pt | reacher.lance"
  "point_lewm      | pusht   | . | pusht/point-lewm/weights.pt         | pusht.lance"
  "point_delta_jepa | pusht   | . | pusht/point-delta-jepa/weights.pt   | pusht.lance"
  # Utonia: a frozen PTv3 with no CLS token, so ATTENTION DOES NOT APPLY -- it is
  # occlusion-only, and that is also the one measure comparable with the arms
  # above. Needs the frozen backbone at checkpoints/utonia/utonia.pth.
  "utonia    | cube    | . | cube/utonia-wm/weights.pt           | cube.lance     | only=occlusion"
)

dry_run=false
skip_done=false
filter=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=true ;;
    --skip-done) skip_done=true ;;
    *) filter="$arg" ;;
  esac
done

trim() { echo "$1" | sed 's/^ *//; s/ *$//'; }

skipped=0
done_count=0
for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r arm env run policy table extra <<< "$entry"
  arm=$(trim "$arm"); env=$(trim "$env"); run=$(trim "$run")
  policy=$(trim "$policy"); table=$(trim "$table"); extra=$(trim "${extra:-}")
  [[ -n "$filter" && "$entry" != *"$filter"* ]] && continue
  # `only=<method>` restricts an entry to one method (Utonia has no attention).
  if [[ "$extra" == *"only="* ]]; then
    want=${extra#*only=}; want=${want%% *}
    [[ "$METHOD" == "$want" ]] || { echo "[$arm / $env] skip: $METHOD not supported here"; continue; }
    extra=${extra//only=$want/}
  fi
  # Everything else in that column is a VISUALISATION override (e.g. the cube
  # regrouping). Occlusion measures the model itself, so it must run exactly as
  # trained -- drop the overrides there.
  [[ "$METHOD" == "occlusion" ]] && extra=""

  ckpt="$LOGS/$run/checkpoints/$policy"
  out="$OUT/${env}_${arm}"
  echo "=========================================================================="
  echo "[$arm / $env] $run -> $policy   (method: $METHOD)"

  missing=false
  [[ -f "$ckpt" ]]          || { echo "  SKIP: no checkpoint at $ckpt"; missing=true; }
  [[ -d "$DATA/$table" ]]   || { echo "  SKIP: no table at $DATA/$table"; missing=true; }
  if $missing; then skipped=$((skipped + 1)); continue; fi

  # --skip-done: resume a sweep. An entry counts as done only when it has BOTH
  # $FRAMES clouds and $FRAMES renders -- a run killed between the two stages
  # leaves PLYs with no images, and treating that as finished would quietly
  # leave a hole in the results.
  if $skip_done; then
    have_ply=$(ls "$out"/*_${SUFFIX}.ply 2>/dev/null | wc -l)
    have_png=$(ls "$out/$SUBDIR"/*_${SUFFIX}.png 2>/dev/null | wc -l)
    if (( have_ply >= FRAMES && have_png >= FRAMES )); then
      echo "  SKIP: already has $have_ply clouds and $have_png renders in $out"
      done_count=$((done_count + 1))
      continue
    fi
    if (( have_ply > 0 || have_png > 0 )); then
      echo "  incomplete ($have_ply clouds, $have_png renders of $FRAMES) -- redoing"
    fi
  fi

  if $dry_run; then
    echo "  would write $out ($FRAMES frames)${extra:+  [$extra]}"
    done_count=$((done_count + 1))
    continue
  fi

  rm -rf "$out"
  pixi run python scripts/viz_attention.py \
    --run "$run" --policy "$policy" --logs-root "$LOGS" \
    --table "$DATA/$table" -n "$FRAMES" --seed "$SEED" --method "$METHOD" --out "$out" \
    $extra $VIZ_ARGS || { skipped=$((skipped+1)); continue; }
  SUFFIX="$SUFFIX" ROOT="$OUT" SUBDIR="$SUBDIR" scripts/render_attention.sh "$out" \
    || { skipped=$((skipped+1)); continue; }
  done_count=$((done_count + 1))
done

echo "=========================================================================="
echo "done: $done_count   skipped: $skipped   output under $OUT/"
