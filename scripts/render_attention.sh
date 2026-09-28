#!/bin/bash
# ===========================================================================
#  Render the attention clouds into the panels figure_attention.py composes.
#
#      scripts/render_attention.sh                # every attention/<env>_<arm>
#      scripts/render_attention.sh reacher        # only dirs matching a word
#      scripts/render_attention.sh attention/pusht_point_lewm    # or an exact dir
#      scripts/render_attention.sh --dry-run      # print the calls, run nothing
#
#  WHY THIS EXISTS. The look of these panels is not one setting: each
#  environment needs its own camera and point size, and those were found by
#  eye, one render at a time. Leaving them in shell history means the figure
#  cannot be reproduced by anyone else -- including us, next month. The table
#  below IS the figure's definition; change a number here and re-run, and every
#  panel of that environment moves together.
#
#  The ramp is applied at render time (the .ply stores raw attention values), so
#  a different look costs a re-render, not a re-run of the model. Recomputing
#  the clouds themselves is scripts/attention_all.sh.
#
#  Output: attention/<env>_<arm>/sensor/*.png, which is the subdirectory
#  figure_attention.py reads by default (--subdir sensor).
# ===========================================================================

set -uo pipefail
cd "$(dirname "$0")/.."

ROOT=${ROOT:-attention}         # attention/ or occlusion/
SUBDIR=${SUBDIR:-sensor}        # where the panels land inside each entry
SUFFIX=${SUFFIX:-rollout}       # which clouds to render (rollout | occlusion | grad)

# Shared by every panel. --log because attention spans two orders of magnitude
# on some checkpoints, and on a linear ramp that leaves a handful of red specks
# on a uniform field; the default --clip-percentile 99 then spends the colours
# on the range the bulk of the points actually occupy.
COMMON=(--size 1000x1000 --cmap coolwarm --log)

# env | camera and point size for that environment
#
#  * The three 2D-derived arenas are read straight down (--elevation 90): they
#    have almost no height structure, so any oblique view trades legibility for
#    a perspective that shows nothing.
#  * Cube is the exception. It is a real 3D workspace where the arm occludes the
#    table, so it is shot from the SENSOR's own bearing, nudged off it far
#    enough to open the occlusion shadows, and its points are sized in metres
#    (--radius-mode absolute) rather than as a fraction of the scene.
#  * --distance is a multiplier on the auto-fitted distance: below 1 crops into
#    the cloud's empty edges, which is what makes the arena fill the panel.
PRESETS=(
  "tworoom | --elevation 90 --distance 0.5  --radius 0.004"
  "reacher | --elevation 90 --distance 0.5  --radius 0.003"
  "pusht   | --elevation 90 --distance 0.45 --radius 0.003"
  "cube    | --from-sensor --azimuth 10 --elevation -10 --distance 0.7 --radius 0.008 --radius-mode absolute"
)

dry_run=false
targets=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=true ;;
    *) targets+=("$arg") ;;
  esac
done

# No target: every entry under $ROOT. A target that names a directory is used
# as-is; anything else is a substring filter, so `reacher` catches both arms.
dirs=()
if (( ${#targets[@]} == 0 )); then
  for d in "$ROOT"/*/; do [[ -d "$d" ]] && dirs+=("${d%/}"); done
else
  for t in "${targets[@]}"; do
    if [[ -d "$t" ]]; then
      dirs+=("${t%/}")
    else
      for d in "$ROOT"/*/; do [[ "$d" == *"$t"* ]] && dirs+=("${d%/}"); done
    fi
  done
fi

trim() { echo "$1" | sed 's/^ *//; s/ *$//'; }

rendered=0
skipped=0
for dir in "${dirs[@]}"; do
  env=$(basename "$dir"); env=${env%%_*}          # attention/<env>_<arm>
  preset=""
  for row in "${PRESETS[@]}"; do
    IFS='|' read -r name flags <<< "$row"
    [[ "$(trim "$name")" == "$env" ]] && preset=$(trim "$flags")
  done
  if [[ -z "$preset" ]]; then
    echo "[$dir] SKIP: no preset for environment '$env'"
    skipped=$((skipped + 1)); continue
  fi

  clouds=("$dir"/*_${SUFFIX}.ply)
  if [[ ! -e "${clouds[0]}" ]]; then
    echo "[$dir] SKIP: no *_${SUFFIX}.ply (run scripts/attention_all.sh first)"
    skipped=$((skipped + 1)); continue
  fi

  echo "=========================================================================="
  echo "[$dir] $env preset -> $dir/$SUBDIR   (${#clouds[@]} clouds)"
  if $dry_run; then
    echo "  pixi run python scripts/render_ply.py $dir/*_${SUFFIX}.ply \\"
    echo "    --out $dir/$SUBDIR ${COMMON[*]} $preset"
    rendered=$((rendered + 1)); continue
  fi

  # shellcheck disable=SC2086  # $preset is a deliberate word-split flag list
  pixi run python scripts/render_ply.py "${clouds[@]}" \
    --out "$dir/$SUBDIR" "${COMMON[@]}" $preset \
    || { skipped=$((skipped + 1)); continue; }
  rendered=$((rendered + 1))
done

echo "=========================================================================="
echo "rendered: $rendered   skipped: $skipped"
echo "compose them with: pixi run python scripts/figure_attention.py"
