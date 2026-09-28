#!/bin/bash
# Occlusion saliency for every arm x environment -- delete a cell of points,
# measure how far the embedding moves.
#
#     scripts/occlusion_all.sh              # all arms, including Utonia
#     scripts/occlusion_all.sh --dry-run
#     scripts/occlusion_all.sh cube
#     scripts/occlusion_all.sh --skip-done   # resume without redoing finished arms
#
# Thin wrapper over attention_all.sh so the run/checkpoint table lives in ONE
# place: two copies of it would drift, and a stale entry pairs the wrong policy
# with the wrong run without erroring.
#
# Why occlusion rather than attention: it is the only measure that means the
# same thing on PointViT (CLS attention over group tokens) and on Utonia (a
# frozen PTv3 with no CLS token, running under no_grad, whose voxelised
# coordinates also rule out gradient saliency). Numbers from this script are
# comparable across arms; attention maps are not.
exec env METHOD=occlusion "$(dirname "$0")/attention_all.sh" "$@"
