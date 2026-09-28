#!/usr/bin/env bash
# Download the re-sensed LiDAR tables from the Hugging Face Hub and extract them.
#
#     pixi run download-data                 # all four environments
#     pixi run download-data cube pusht      # a subset
#
# Tables land in $PLWM_DATA_ROOT (default: <repo>/data), one Lance table per
# environment, exactly as config/eval/*.yaml and config/train/data/*.yaml expect:
#
#     tworoom  fafraob/point-cloud-tworoom   two_room.lance.tar.zst  (0.8 GB)  -> two_room.lance
#     reacher  fafraob/point-cloud-reacher   reacher.lance.tar.zst   (7.1 GB)  -> reacher.lance
#     pusht    fafraob/point-cloud-pusht     pusht.lance.tar.zst     (4.6 GB)  -> pusht.lance
#     cube     fafraob/point-cloud-ogb-cube  cube.lance.tar.zst      (43 GB)   -> cube.lance
#
# The archives are deleted after extraction (pass --keep-archives to keep them).
# The download resumes; an already extracted table is skipped.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${PLWM_DATA_ROOT:-$REPO_ROOT/data}"

declare -A HF_REPO=([tworoom]=fafraob/point-cloud-tworoom [reacher]=fafraob/point-cloud-reacher
                    [pusht]=fafraob/point-cloud-pusht [cube]=fafraob/point-cloud-ogb-cube)
declare -A TABLE=([tworoom]=two_room.lance [reacher]=reacher.lance [pusht]=pusht.lance [cube]=cube.lance)

keep=false; envs=()
for arg in "$@"; do
  case "$arg" in
    --keep-archives) keep=true ;;
    tworoom|reacher|pusht|cube) envs+=("$arg") ;;
    *) echo "usage: $0 [--keep-archives] [tworoom|reacher|pusht|cube ...]" >&2; exit 1 ;;
  esac
done
(( ${#envs[@]} > 0 )) || envs=(tworoom reacher pusht cube)

mkdir -p "$DATA_ROOT"
for env in "${envs[@]}"; do
  table="${TABLE[$env]}"; archive="${table}.tar.zst"
  if [[ -d "$DATA_ROOT/$table" ]]; then
    echo "[$env] $DATA_ROOT/$table exists, skipping"; continue
  fi
  echo "[$env] downloading ${HF_REPO[$env]}/$archive -> $DATA_ROOT"
  hf download "${HF_REPO[$env]}" "$archive" --repo-type dataset --local-dir "$DATA_ROOT"
  echo "[$env] extracting $archive"
  tar --zstd -xf "$DATA_ROOT/$archive" -C "$DATA_ROOT"
  [[ -d "$DATA_ROOT/$table" ]] || { echo "ERROR: $archive did not produce $DATA_ROOT/$table" >&2; exit 1; }
  $keep || rm -f "$DATA_ROOT/$archive"
  rm -rf "$DATA_ROOT/.cache"
  echo "[$env] ready: $DATA_ROOT/$table"
done
