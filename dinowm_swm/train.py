"""Launch the vendored DINO-WM trainer (third_party/dino_wm/train.py) on a
stable-worldmodel .h5 dataset.

    torchrun --nproc_per_node=2 -m dinowm_swm.train env=swm_pusht run_dir=$PWD/experiment_logs/dinowm_pusht
    python -m dinowm_swm.train env=swm_pusht run_dir=$PWD/experiment_logs/dinowm_pusht
    python -m dinowm_swm.train env=swm_tworoom run_dir=... training.epochs=30 mixed_precision=no
    python -m dinowm_swm.train env=swm_pusht run_dir=... training.batch_size=32 training.predictor_lr=5e-4 training.action_encoder_lr=5e-4

(run from the repository root; the first line is how the paper's runs were
launched, the last restores the upstream single-GPU recipe, see
dinowm_swm/conf/train_swm.yaml).

``run_dir`` must be an absolute path (Hydra changes into it); with the default
layout that is ``<repo>/experiment_logs/dinowm_<env>`` (or
``$PLWM_LOGS_ROOT/dinowm_<env>``), i.e. the run named after the environment
under ``paths.LOGS_ROOT``.

Multi-GPU: upstream trains through `accelerate`, which reads torchrun's
RANK / WORLD_SIZE / LOCAL_RANK, wraps the trainable modules in DDP, leaves
the frozen encoder unwrapped (no trainable parameters) and shards every
batch: ``training.batch_size`` is the EFFECTIVE batch (upstream divides it by
the process count). The predictor, action encoder and proprio encoder all
receive gradients every step, so plain DDP has no unused-parameter problem.

Upstream code runs UNMODIFIED; this file only

* puts third_party/dino_wm on sys.path (its Hydra targets and pickled
  checkpoints refer to top-level ``models`` / ``datasets`` / ``train``),
* adds upstream's conf/ to the Hydra search path (encoder / predictor /
  decoder groups) while the primary config is dinowm_swm/conf/train_swm.yaml,
* sets the environment defaults an unattended run needs: WANDB_MODE=offline,
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD, SWM_H5_DIR (the directory holding the
  source ``.h5`` files, default ``paths.DATA_ROOT``), ACCELERATE_MIXED_PRECISION.
  TORCH_HOME is left to the user: on a machine without network access set it
  to a directory that already holds the DINOv2 torch.hub cache
  (``hub/facebookresearch_dinov2_main`` + ``hub/checkpoints/dinov2_vits14_pretrain.pth``);
  otherwise torch.hub downloads facebookresearch/dinov2 on first use,
* makes ``training.epochs`` a TOTAL, resume-aware budget: upstream resumes
  from ``<run_dir>/checkpoints/model_latest.pth`` and then trains
  ``training.epochs`` MORE epochs; here the trainer runs
  ``max(0, epochs - epochs_done)`` so a chain of time-limited runs on the same
  run_dir converges to exactly ``training.epochs`` and then exits 0 at once.
  ``<run_dir>/TRAINING_DONE`` is written when the budget is reached,
* loads checkpoints onto the LOCAL GPU: upstream's ``load_ckpt`` is a bare
  ``torch.load()`` of pickled nn.Modules, which unpickles every tensor onto
  the device it was saved from -- rank 0's cuda:0. Parameters and buffers are
  moved by ``accelerator.prepare``, but the causal attention mask in
  models/vit.py is a plain tensor attribute (``self.bias = ....to('cuda')``)
  and stays on cuda:0, so a 2-GPU resume fails on rank 1 with "expected self
  and mask to be on the same device".
  ``torch.load`` therefore gets ``map_location=cuda:LOCAL_RANK`` whenever the
  caller passes none (see :func:`_patch_torch_load_to_local_device`).
"""
import logging
import os
import sys
from pathlib import Path

from . import DINO_WM_ROOT, REPO_ROOT, add_upstream_to_path

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from paths import DATA_ROOT  # noqa: E402  (repository path convention, see paths.py)

log = logging.getLogger("dinowm_swm.train")
HERE = Path(__file__).resolve().parent
DONE_MARKER = "TRAINING_DONE"


def _maybe_test_backend():
    """Test hook: DINOWM_DIST_BACKEND=gloo pre-initialises the process group so
    two ranks can share ONE GPU (LOCAL_RANK=0 for both) on a single-GPU machine; accelerate
    adopts an existing group. Never set for real multi-GPU training (NCCL is
    the default)."""
    backend = os.environ.get("DINOWM_DIST_BACKEND")
    if backend and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
            log.warning("test hook: process group pre-initialised with backend %s", backend)


def _offline_defaults():
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    # upstream resumes with a bare torch.load() of pickled nn.Modules; torch
    # >= 2.6 defaults to weights_only=True and refuses. This env switch (torch
    # >= 2.6) restores the old default without touching upstream code.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    # the env/swm_*.yaml configs locate the source .h5 files through
    # ${oc.env:SWM_H5_DIR}; by default they live next to the Lance tables
    # (PLWM_DATA_ROOT, see paths.py). TORCH_HOME is deliberately NOT set here:
    # torch.hub downloads facebookresearch/dinov2 on first use, or point
    # TORCH_HOME at a pre-filled hub cache on an offline machine.
    os.environ.setdefault("SWM_H5_DIR", DATA_ROOT)


def _patch_torch_load_to_local_device(torch_module=None):
    """Make ``torch.load`` default to ``map_location=cuda:<LOCAL_RANK>``.

    Upstream resumes with ``torch.load(filename)`` on pickled modules saved by
    rank 0, so on any other rank the tensors come back on cuda:0. Injecting
    the local device at unpickle time fixes every tensor at once, including
    plain attributes that ``.to(device)`` / ``accelerator.prepare`` never see
    (the causal mask in models/vit.py). An explicit ``map_location`` from the
    caller is left alone; without CUDA nothing changes. Idempotent; returns
    the original ``torch.load`` so tests can restore it.
    """
    if torch_module is None:
        import torch as torch_module
    orig = torch_module.load
    if getattr(orig, "_dinowm_local_device", False):
        return orig
    def load(*args, **kwargs):
        if "map_location" not in kwargs and torch_module.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            kwargs["map_location"] = torch_module.device("cuda", local_rank)
        return orig(*args, **kwargs)
    load._dinowm_local_device = True
    load.__wrapped__ = orig
    torch_module.load = load
    return orig


def main():
    add_upstream_to_path()
    _offline_defaults()
    # upstream's config groups (encoder/, predictor/, decoder/, ...) resolve
    # through the search path; our primary config and env/ group come first.
    sys.argv.append(f"hydra.searchpath=[file://{DINO_WM_ROOT / 'conf'}]")

    import hydra
    import torch
    from omegaconf import OmegaConf

    @hydra.main(config_path=str(HERE / "conf"), config_name="train_swm", version_base=None)
    def run(cfg):
        assert cfg.run_dir and cfg.run_dir != "???", "pass run_dir=/absolute/run/folder"
        os.environ["ACCELERATE_MIXED_PRECISION"] = str(cfg.mixed_precision)
        os.environ["WANDB_MODE"] = str(cfg.wandb_mode)
        # TF32 for whatever stays fp32 under the autocast (harmless with bf16, a
        # free 10% for mixed_precision=no on Ampere)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        _maybe_test_backend()
        log.info("cwd (run dir): %s  (rank %s of %s)", os.getcwd(),
                 os.environ.get("RANK", "0"), os.environ.get("WORLD_SIZE", "1"))
        log.info("TORCH_HOME=%s SWM_H5_DIR=%s WANDB_MODE=%s ACCELERATE_MIXED_PRECISION=%s",
                 os.environ.get("TORCH_HOME"), os.environ.get("SWM_H5_DIR"),
                 os.environ["WANDB_MODE"], os.environ["ACCELERATE_MIXED_PRECISION"])

        # Upstream's top-level package is called `datasets`, which shadows
        # HuggingFace `datasets`; accelerate's MULTI-process dataloader path
        # probes for the HF package by name and would then import
        # IterableDataset from upstream's package. Tell accelerate the HF
        # package is absent (it is only used for HF IterableDataset sharding).
        import accelerate.data_loader as _adl
        import accelerate.utils.imports as _aui
        _adl.is_datasets_available = _aui.is_datasets_available = lambda: False

        from train import Trainer  # upstream, vendored

        _patch_torch_load_to_local_device(torch)  # resume: checkpoint tensors on THIS rank's GPU
        target = int(cfg.training.epochs)
        trainer = Trainer(cfg)          # resumes from checkpoints/model_latest.pth if present
        if os.environ.get("DINOWM_DIST_BACKEND"):
            # test hook (see _maybe_test_backend): accelerate did not create the
            # group, so tell it the real backend -- its metric gathers pick the
            # gloo-compatible collective from this attribute
            from accelerate import PartialState
            PartialState().backend = os.environ["DINOWM_DIST_BACKEND"]
        done = int(trainer.epoch)
        remaining = max(0, target - done)
        log.info("epoch budget: total %d, done %d, remaining %d", target, done, remaining)
        if remaining == 0:
            Path(DONE_MARKER).write_text(f"epochs={done}\n")
            log.info("budget already reached; nothing to do (%s written)", DONE_MARKER)
            return
        trainer.total_epochs = remaining
        trainer.run()
        Path(DONE_MARKER).write_text(f"epochs={trainer.epoch}\n")
        log.info("training complete at epoch %d (%s written)", trainer.epoch, DONE_MARKER)

    run()


if __name__ == "__main__":
    main()
