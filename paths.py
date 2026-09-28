"""Where this repository reads data from and writes results to.

Every path the code needs is derived from the repository root and can be moved
with an environment variable, so nothing in the code or the configs refers to a
particular machine:

======================  ====================  ==========================================
environment variable    default               holds
======================  ====================  ==========================================
``PLWM_DATA_ROOT``      ``<repo>/data``       the four LiDAR tables (``two_room.lance``,
                                              ``pusht.lance``, ``reacher.lance``,
                                              ``cube.lance``) and the source ``.h5`` files
``PLWM_LOGS_ROOT``      ``<repo>/experiment_logs``  training runs (one folder per run)
``PLWM_RESULTS_ROOT``   ``<repo>/eval_results``     evaluation sweeps
``PLWM_T2L_ROOT``       ``<repo>/experiment_logs/target2latent``  target-to-latent caches and heads
``PLWM_PROBE_ROOT``     ``<repo>/experiment_logs/probing``        latent-probing caches and results
======================  ====================  ==========================================

The Hydra configs read the same variables through ``${oc.env:PLWM_DATA_ROOT,data}``
style interpolations, so setting a variable once moves both the configs and the
scripts. The downloaded checkpoints and the fixed evaluation start lists live inside
the repository (``checkpoints/<env>/<model>/``, ``eval_starts/``) and are not configurable.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _root(var: str, default: str) -> str:
    value = os.environ.get(var)
    if value:
        return str(Path(value).expanduser())
    return str(REPO_ROOT / default)


DATA_ROOT = _root("PLWM_DATA_ROOT", "data")
LOGS_ROOT = _root("PLWM_LOGS_ROOT", "experiment_logs")
RESULTS_ROOT = _root("PLWM_RESULTS_ROOT", "eval_results")
T2L_ROOT = _root("PLWM_T2L_ROOT", "experiment_logs/target2latent")
PROBE_ROOT = _root("PLWM_PROBE_ROOT", "experiment_logs/probing")

#: file name of each environment's LiDAR table under ``DATA_ROOT``
DATASETS = {
    "cube": "cube.lance",
    "tworoom": "two_room.lance",
    "pusht": "pusht.lance",
    "reacher": "reacher.lance",
}


def dataset_path(env: str) -> str:
    """Absolute path of an environment's LiDAR table."""
    return str(Path(DATA_ROOT) / DATASETS[env])
