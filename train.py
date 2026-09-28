"""Train the point-cloud JEPA world models (Point-LeWM, Point-Delta-JEPA,
Utonia-WM, Vox-WM). An image data config (no `obs` block) selects the original
pixel pipeline instead, which is what the image Delta-JEPA baseline of Table 2
uses (`config/train/image_delta_jepa_<env>.yaml`).

All output for a run lives in ONE self-contained folder named by `run_name`
(default: timestamp + output_model_name) under experiment_logs/ or
$PLWM_LOGS_ROOT. Nothing is written loose in the project or to any global cache:

    experiment_logs/<run_name>/
    ├── .hydra/ , train.log            # Hydra's config snapshot + run log
    ├── events.out.tfevents.*          # TensorBoard scalars
    ├── hparams.yaml , config.yaml     # hyperparameters + fully-resolved config
    └── checkpoints/
        ├── last.ckpt , epoch=*.ckpt   # Lightning state — resume from these
        └── <output_model_name>/       # save_pretrained export — eval loads this
            ├── weights_epoch_*.pt
            ├── weights_final.pt       # copy of the newest export; the eval sweeps load this
            └── config.json

    View logs:    pixi run tensorboard          (== tensorboard --logdir experiment_logs)
    Resume:       pixi run train +ckpt_path=experiment_logs/<run>/checkpoints/last.ckpt
    Name a run:   pixi run train --config-name=<cfg> run_name=<cfg>   (the eval sweeps address runs by this name)
"""

import os
from functools import partial
from pathlib import Path

# Folder that holds every run: <repo>/experiment_logs, or $PLWM_LOGS_ROOT
# (see paths.py). config/train/launcher/local.yaml (hydra.run.dir) and the
# eval configs (logs_root) read the same variable.
from paths import LOGS_ROOT as EXPERIMENT_LOGS  # noqa: E402

# stable_worldmodel's get_cache_dir() (its dataset cache) defaults to the GLOBAL
# ~/.stable_worldmodel. Pin it next to the runs so nothing leaks out; doing it
# here, before importing stable_worldmodel, makes a bare `python train.py`
# match, and `setdefault` lets an explicit override win.
os.environ.setdefault("STABLEWM_HOME", EXPERIMENT_LOGS)

# Route fp16 Linear+bias through plain GEMM + bias-add instead of cuBLASLt's fused
# epilogue: on some GPU architectures (e.g. sm80/A100) cuBLASLt has no algorithm for
# the huge fp16 batches the point encoders produce (millions of points in one
# F.linear) and raises CUBLAS_STATUS_NOT_SUPPORTED. Torch reads this once at the
# first CUDA addmm, so set it before any forward pass. pixi.toml sets it too;
# doing it here makes a bare `python train.py` match.
os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")

import hydra
from hydra.core.hydra_config import HydraConfig
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import OmegaConf, open_dict

# There is no public setter to disable stable_pretraining's global run cache
# (spt.set treats cache_dir=None as "leave unchanged"), so keep_all_output_in_run_dir()
# reaches the config singleton directly. Guarded so a future rename fails loudly
# rather than silently reintroducing the global writes.
try:
    from stable_pretraining._config import get_config as _spt_get_config
except Exception:  # pragma: no cover - private API guard
    _spt_get_config = None

from module import SIGReg
from pc_encoders import collate_point_cloud
from utils import (
    attach_embedding_cache,
    get_column_normalizer,
    get_img_preprocessor,
    get_pc_preprocessor,
    SaveCkptCallback,
)


def keep_all_output_in_run_dir():
    """Stop stable_pretraining from writing anywhere but this run's directory.

    Two of its defaults would otherwise leak files:

    * The **Manager's global run cache** (``~/.cache/stable-pretraining/runs/…``)
      captures every run and even redirects our ``ModelCheckpoint`` into it.
      Disabling ``cache_dir`` drops the Manager into "legacy" mode, which honors
      the standard Lightning/Hydra layout our callbacks set up.
    * Auto **loggers/callbacks** — ``RegistryLogger`` (metrics.csv, sidecar.json,
      heartbeat), ``EnvironmentDumpCallback`` (environment_*.json,
      requirements_frozen_*.txt) and ``HuggingFaceCheckpointCallback``
      (hf_exports/) — fall back to ``default_root_dir`` and scatter files in the
      project root. Turn them off: TensorBoard already has the metrics and
      pixi.lock pins the environment.
    """
    spt.set(
        default_loggers={"registry": False},
        default_callbacks={"env_dump": False, "hf_checkpoint": False},
    )
    if _spt_get_config is not None:
        _spt_get_config().cache_dir = None


def resolve_world_size(cfg):
    """How many ranks Lightning will run, from the trainer config.

    Used to report the EFFECTIVE batch and to decide whether a whole-batch loss
    needs its cross-rank synchronisation (see :func:`multi_gpu_loss_trainer_kwargs`).
    """
    devices = cfg.trainer.get("devices", 1)
    accel = str(cfg.trainer.get("accelerator", "auto"))
    # an explicit device LIST (also OmegaConf's ListConfig, which is not a list)
    if not isinstance(devices, (str, bytes)) and hasattr(devices, "__len__"):
        n = len(devices)
    elif str(devices) in ("auto", "-1"):
        n = torch.cuda.device_count() if accel in ("gpu", "cuda", "auto") else 1
        n = max(n, 1)
    else:
        n = int(devices)
    return n * int(cfg.trainer.get("num_nodes", 1))


def multi_gpu_loss_trainer_kwargs(cfg, world_size):
    """Extra Trainer kwargs that keep whole-batch losses exact under DDP.

    SIGReg's Epps-Pulley statistic is computed over the batch. Under DDP,
    module.SIGReg switches itself to autograd-aware cross-rank collectives, so
    the statistic AND its parameter gradients match a single-GPU run at the
    same effective batch (2 x 64 == 1 x 128). The other batch statistic in its
    input path is the projector MLP's BatchNorm1d (jepa.encode routes the
    encoder output through the projector before it reaches the loss), so those
    are converted to SyncBatchNorm too. Both changes apply ONLY to a
    multi-rank SIGReg run: single-GPU runs and the per-sample arms (point_delta_jepa,
    utoniawm) are untouched.
    """
    if world_size > 1 and cfg.loss.get("sigreg") is not None:
        print(
            "[train] SIGReg under DDP: cross-rank synced statistic "
            "(module.SIGReg) + sync_batchnorm=True -- objective matches a "
            "single-GPU run at the same effective batch."
        )
        return {"sync_batchnorm": True}
    return {}


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses.

    The loss is assembled from the pieces present under ``cfg.loss``:

    * ``pred_loss`` -- always: 1-step teacher-forced latent prediction over the
      leading ``history_size`` frames.
    * ``sigreg_loss`` (``cfg.loss.sigreg``) -- LeWM's Gaussian regularizer over
      the full batch (its distribution test needs whole-batch statistics).
    * ``action_loss`` (``cfg.loss.action``) -- Delta-JEPA's LDAD action
      reconstruction from latent displacements (Delta-JEPA paper, Eq. 5/6). The
      faithful Delta-JEPA configs carry ``action`` and drop ``sigreg`` (LDAD is
      the anti-collapse mechanism there).
    """

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    # the explicit end index keeps the target length == ctx_len even if a
    # window carried more frames than ctx_len + n_preds; the configs pin
    # ctx_len = num_steps - n_preds (full-window teacher forcing), so every
    # frame appears in the prediction loss.
    tgt_emb = emb[:, n_preds : ctx_len + n_preds]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    output["pred_loss"] = F.mse_loss(pred_emb, tgt_emb)
    output["loss"] = output["pred_loss"]

    sigreg_cfg = cfg.loss.get("sigreg")
    if sigreg_cfg is not None:
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
        output["loss"] = output["loss"] + sigreg_cfg.weight * output["sigreg_loss"]

    action_cfg = cfg.loss.get("action")
    if action_cfg is not None:
        output["action_loss"] = self.model.action_decoder.loss(emb, batch["action"])
        output["loss"] = output["loss"] + action_cfg.weight * output["action_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)

    # Gradient accumulation (optimizer `frequency` below): backward runs every
    # micro-batch, so gradients SUM over the window. Average the loss here so
    # the accumulated gradient equals the full-batch mean -- otherwise
    # trainer.gradient_clip_val would clip at 1/frequency of its nominal
    # threshold. Scaled after log_dict: the logged losses stay comparable
    # across accumulation settings.
    accumulate = cfg.get("accumulate_grad_batches", 1)
    if stage == "fit" and accumulate > 1:
        output["loss"] = output["loss"] / accumulate
    return output

def check_model_config(cfg):
    """Fail on unfilled ``???`` model values BEFORE the dataset is opened.

    ``hydra.utils.instantiate`` would catch these anyway, but only after the
    dataset section below has scanned the whole Lance table -- in a batch job
    that turns a one-line config mistake into a long wait. The two
    keys train.py fills from the data itself are exempt.
    """
    filled = {"action_encoder.input_dim", "action_decoder.action_dim"}

    def walk(node, path=""):
        for key in node:
            sub = f"{path}.{key}" if path else str(key)
            if OmegaConf.is_missing(node, key):
                if sub not in filled:
                    yield sub
            elif OmegaConf.is_dict(node[key]):
                yield from walk(node[key], sub)

    missing = list(walk(cfg.model))
    if missing:
        raise ValueError(
            "model config has unfilled mandatory values: "
            + ", ".join(f"model.{m}" for m in missing)
            + ". Environment-geometry keys (norm_center / norm_scale / "
            "group_radius) are properties of the dataset; see docs/training.md, "
            "section 'Adding an environment'."
        )


@hydra.main(version_base=None, config_path="./config/train", config_name="point_lewm_cube")
def run(cfg):
    # DataLoader workers via 'spawn', NOT the 'forkserver' stable_worldmodel's
    # lance reader would otherwise force (its _force_forkserver only overrides
    # fork/None, so setting spawn here first is respected; both are equally
    # fork-safe for lancedb). forkserver has a fatal failure mode: its workers'
    # ppid is the forkserver -- not the trainer -- so torch's parent-death
    # watchdog never fires when the trainer is SIGKILLed (OOM kill, scancel),
    # and the forkserver itself stays alive on sockets its children inherited.
    # Every killed run then leaks num_workers x ~5 GB of immortal orphaned
    # pt_data_workers until someone kills them by hand. With spawn the workers
    # are direct children and reap themselves seconds after the trainer dies.
    # Cost: one interpreter re-import per worker per run (persistent_workers).
    import multiprocessing as mp

    if mp.get_start_method(allow_none=True) in (None, "fork"):
        mp.set_start_method("spawn", force=True)

    check_model_config(cfg)

    # State the effective batch (per-rank x ranks x accumulation -- a 2-GPU run
    # silently doubles it unless loader.batch_size is halved), and collect the
    # Trainer kwargs that keep a multi-rank SIGReg run exact (no-op otherwise).
    world_size = resolve_world_size(cfg)
    ddp_loss_kwargs = multi_gpu_loss_trainer_kwargs(cfg, world_size)
    accum = int(cfg.get("accumulate_grad_batches", 1))
    print(
        f"[train] {world_size} rank(s) x batch {cfg.loader.batch_size} x "
        f"{accum} accumulation = effective batch "
        f"{world_size * cfg.loader.batch_size * accum}"
    )


    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )

    # -- observation modality: image (pixels) by default, or point cloud (lidar).
    #    A point-cloud data config carries an `obs` block (modality: pointcloud);
    #    image configs omit it and fall through to the ViT image preprocessor.
    obs_cfg = cfg.data.get("obs", None)
    is_point_cloud = obs_cfg is not None and obs_cfg.get("modality") == "pointcloud"

    collate_fn = None
    if is_point_cloud:
        # Frozen-encoder (Utonia-WM) arms: training consumes PRECOMPUTED
        # embeddings instead of raw clouds. The cache (precompute_utonia.py)
        # is registered as a virtual dataset column named obs.source_key;
        # each frame then flows through the normal point-cloud pipeline as a
        # "1-point cloud" whose coordinates ARE the embedding, and the
        # encoder passes it straight through (see UtoniaEncoder). Everything
        # below (reshape transform, packed collate, JEPA) is unchanged.
        emb_cache = obs_cfg.get("emb_cache")
        if emb_cache:
            attach_embedding_cache(
                dataset, obs_cfg.source_key, emb_cache, cfg.model.encoder,
                repo_root=Path(__file__).resolve().parent, obs_cfg=obs_cfg,
            )
        transforms = [
            get_pc_preprocessor(
                source=obs_cfg.source_key,
                target=obs_cfg.target_key,
                in_channels=obs_cfg.in_channels,
            )
        ]
        obs_keys = {obs_cfg.source_key, obs_cfg.target_key}
        # RGB lidar (`lidar_contains_rgb: true`): the dataset carries a
        # `<source>_rgb` column of per-point colors, row-aligned with the cloud.
        # Reshape it like the coords and hand it to the collate_fn as feat_key,
        # so the packed batch's `feat` holds the colors (the encoder turns them
        # into per-voxel color features via point_features: [..., rgb]).
        feat_key = None
        if obs_cfg.get("lidar_contains_rgb"):
            rgb_source = f"{obs_cfg.source_key}_rgb"
            feat_key = f"{obs_cfg.target_key}_rgb"
            transforms.append(
                get_pc_preprocessor(source=rgb_source, target=feat_key, in_channels=3)
            )
            obs_keys |= {rgb_source, feat_key}
        # Full clouds are packed; any per-cloud point budget is applied by the
        # encoder itself with GPU FPS -- the workers only reshape and concat.
        collate_fn = partial(
            collate_point_cloud,
            point_key=obs_cfg.target_key,
            feat_key=feat_key,
            drop_value=obs_cfg.get("invalid_value"),
        )
    else:
        transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]
        obs_keys = {"pixels"}

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col in obs_keys or col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
        if cfg.model.get("action_decoder") is not None:
            # LDAD reconstructs the same frameskip-bundled macro-actions the
            # action encoder consumes
            cfg.model.action_decoder.action_dim = cfg.model.action_encoder.input_dim

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen, collate_fn=collate_fn)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False, collate_fn=collate_fn)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    # Token-latent arms (Utonia-WM): the predictor's token count must equal the
    # encoder's grid size, or every batch would reshape into nonsense. Both are
    # derived from the same config keys, so this only fires on a hand override.
    enc_tokens = getattr(world_model.encoder, "num_tokens", None)
    pred_tokens = getattr(world_model.predictor, "num_tokens", None)
    if enc_tokens is not None and pred_tokens is not None:
        assert enc_tokens == pred_tokens, (
            f"encoder grid has {enc_tokens} tokens but the predictor expects "
            f"{pred_tokens} -- check grid_x/grid_y/grid_z vs num_tokens"
        )

    # Warm-start: load a full-JEPA weight export (checkpoints/<name>/weights_epoch_*.pt)
    # into the freshly-built model, then train with a *fresh* optimizer + LR schedule.
    # Unlike a Lightning `+ckpt_path` resume, this inherits only the weights -- not the
    # dead cosine (which had annealed to lr=0) or the epoch counter -- so the new run
    # gets a clean warmup->cosine over its own trainer.max_epochs.
    warm_start = cfg.get("warm_start")
    if warm_start:
        sd = torch.load(warm_start, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = world_model.load_state_dict(sd, strict=False)
        print(
            f"[train] warm-start from {warm_start}: loaded "
            f"{len(sd) - len(unexpected)}/{len(sd)} tensors "
            f"({len(missing)} missing, {len(unexpected)} unexpected)"
        )

    # Gradient accumulation: spt.Module steps (and zero_grads) each optimizer
    # only every `frequency` batches; manual_backward still runs per batch, so
    # micro-batch gradients accumulate in between (lejepa_forward averages the
    # loss over the window to match full-batch gradient scale). The scheduler
    # is stepped once per OPTIMIZER step, but its smart default sizes the
    # anneal in BATCHES (trainer.estimated_stepping_batches) -- with
    # accumulation that leaves the cosine unfinished at end of training, so
    # size it explicitly in optimizer steps (0.01 warmup fraction = the
    # default's ratio).
    accumulate = int(cfg.get("accumulate_grad_batches", 1))
    # LR schedule, cfg.scheduler: "warmup_cosine" (the default when unset) is
    # the 1%-linear-warmup -> cosine-to-zero anneal; "constant" holds
    # cfg.optimizer.lr for the whole run, no warmup and no anneal (DINO-WM's
    # own setup: fixed per-component LRs, no scheduler anywhere). spt.Module
    # always builds and steps a scheduler, so "constant" is expressed as
    # torch's ConstantLR held at factor 1.0.
    schedule = cfg.get("scheduler", "warmup_cosine")
    if schedule == "warmup_cosine":
        scheduler = {"type": "LinearWarmupCosineAnnealingLR"}
        if accumulate > 1:
            opt_steps = (len(train) // accumulate) * cfg.trainer.max_epochs
            scheduler.update(warmup_steps=max(1, int(0.01 * opt_steps)), max_steps=opt_steps)
    elif schedule == "constant":
        scheduler = {"type": "ConstantLR", "factor": 1.0, "total_iters": 0}
    else:
        raise ValueError(
            f"unknown scheduler {schedule!r}; use 'warmup_cosine' or 'constant'"
        )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": accumulate,
        },
    }

    # SIGReg is only attached when the config asks for it: the Delta-JEPA
    # configs drop it entirely (LDAD replaces it as the anti-collapse term).
    module_kwargs = {}
    if cfg.loss.get("sigreg") is not None:
        module_kwargs["sigreg"] = SIGReg(**cfg.loss.sigreg.kwargs)

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
        **module_kwargs,
    )

    ##########################
    ##       training       ##
    ##########################

    keep_all_output_in_run_dir()

    # Hydra creates and names this run's folder (hydra.run.dir in
    # config/train/launcher/local.yaml); we point every writer below at it, so the
    # run is fully self-contained (see this module's docstring for the layout).
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()

    loggers = [TensorBoardLogger(save_dir=str(run_dir), name="", version="", default_hp_metric=False)]
    if cfg.wandb.enabled:
        loggers.append(WandbLogger(**cfg.wandb.config))
    for logger in loggers:
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    # Snapshot the fully-resolved config next to Hydra's own .hydra/ dump.
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    callbacks = [
        # Lightning state -> checkpoints/{last,epoch=*}.ckpt (resume)
        ModelCheckpoint(dirpath=str(run_dir / "checkpoints"), save_last=True),
        # save_pretrained export -> checkpoints/<name>/weights_epoch_*.pt + config.json (eval)
        SaveCkptCallback(
            run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
            cache_dir=str(run_dir),
        ),
    ]

    # Optional wall-clock budget IN HOURS that stops at an EPOCH boundary
    # (Lightning's trainer.max_time stops mid-epoch, and the export callback
    # above would then label a partial epoch as a full one). Useful for
    # unattended runs on a time-limited machine: e.g. ++max_train_hours=68 under
    # a 72-hour budget leaves the last epoch and the export room to finish, so
    # a dependent eval can load weights_final.pt (the callback above mirrors
    # the latest export there) even when 100 epochs do not fit the budget.
    # Hours, not a "DD:HH:MM:SS" string: YAML reads 2:20:00:00 as the base-60
    # integer 504000.
    if cfg.get("max_train_hours"):
        from datetime import timedelta
        from lightning.pytorch.callbacks import Timer
        callbacks.append(Timer(duration=timedelta(hours=float(cfg.max_train_hours)), interval="epoch"))

    trainer = pl.Trainer(
        **cfg.trainer,
        **ddp_loss_kwargs,
        default_root_dir=str(run_dir),
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=loggers,
        enable_checkpointing=True,
    )

    # Resume only when an explicit checkpoint is given, e.g.
    #   pixi run train +ckpt_path=experiment_logs/<run>/checkpoints/last.ckpt
    ckpt_path = cfg.get("ckpt_path") or None
    manager = spt.Manager(
        trainer=trainer, module=world_model, data=data_module, ckpt_path=ckpt_path,
    )
    manager()


if __name__ == "__main__":
    run()
