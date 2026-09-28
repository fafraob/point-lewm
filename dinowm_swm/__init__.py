"""DINO-WM (Zhou et al., 2024, arXiv:2411.04983) on the stable-worldmodel image
datasets, trained with the UNMODIFIED upstream code vendored under
``third_party/dino_wm`` (commit in third_party/dino_wm/UPSTREAM_COMMIT.txt).

This package only ADDS what upstream lacks for our data and training setup:

* :mod:`dinowm_swm.h5_dset`  -- a ``TrajDataset`` over the ``.h5`` datasets
  (pusht_expert_train / reacher / tworoom / cube_single_expert) and a window
  slicer that reads only the frames a sample needs;
* ``conf/``                   -- the Hydra primary config ``train_swm`` and one
  ``env/swm_<env>.yaml`` per dataset (upstream's conf/ groups are pulled in
  through hydra.searchpath, so encoder / predictor / decoder configs are the
  paper's);
* :mod:`dinowm_swm.train`     -- the launcher (offline wandb, .h5 location,
  mixed precision, resumable TOTAL-epoch budget);
* :mod:`dinowm_swm.planning`  -- the stable-worldmodel ``Costable`` wrapper so
  ``eval.py`` plans with a DINO-WM run through the same CEM / sequence-matched
  protocol as the LeWM image checkpoints.

Checkpoints are upstream's own format (``checkpoints/model_<epoch>.pth`` +
``hydra.yaml`` in the run folder), loadable by upstream ``plan.py``.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DINO_WM_ROOT = REPO_ROOT / "third_party" / "dino_wm"


def add_upstream_to_path():
    """Make the vendored dino_wm importable as top-level ``models``, ``datasets``,
    ``train`` ... (that is how upstream's own Hydra targets and pickled
    checkpoints refer to them). Idempotent."""
    p = str(DINO_WM_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)
    return DINO_WM_ROOT
